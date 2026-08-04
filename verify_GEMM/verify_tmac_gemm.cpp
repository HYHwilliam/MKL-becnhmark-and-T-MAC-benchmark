// verify_tmac_gemm.cpp
//
// T-MAC W2A16 / W3A16 / W4A16 correctness verification.
// Compares the tiled AVX2 implementation against a scalar golden reference.

#include <immintrin.h>
#include <stdint.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

static inline float _mm256_addv_ps(const __m256 value) {
    __m128 result = _mm256_extractf128_ps(value, 1);
    result = _mm_add_ps(result, _mm256_castps256_ps128(value));
    result = _mm_add_ps(result, _mm_movehl_ps(result, result));
    result = _mm_add_ss(result, _mm_movehdup_ps(result));
    return _mm_cvtss_f32(result);
}

#define extract_low_epi8_epi16(v) _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v))
#define extract_high_epi8_epi16(v) _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1))
#define extract_low_epi16_epi32(v) _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v))
#define extract_high_epi16_epi32(v) _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1))

template <int N>
struct SignedWideningAdder {
    __m256i lhs_low;
    __m256i lhs_high;

    inline void push(__m256i value, int k) {
        if (k == 0) {
            lhs_low = extract_low_epi8_epi16(value);
            lhs_high = extract_high_epi8_epi16(value);
        } else {
            lhs_low = _mm256_add_epi16(lhs_low, extract_low_epi8_epi16(value));
            lhs_high = _mm256_add_epi16(lhs_high, extract_high_epi8_epi16(value));
        }
    }

    inline __m256i get_low() { return lhs_low; }
    inline __m256i get_high() { return lhs_high; }
};

inline void partial_max_g4_int8_k8(float* lut_scales, const float* activations) {
    const __m256i gather_index = _mm256_set_epi32(112, 96, 80, 64, 48, 32, 16, 0);
    __m256 v0 = _mm256_i32gather_ps(activations + 0, gather_index, 1);
    __m256 v1 = _mm256_i32gather_ps(activations + 1, gather_index, 1);
    __m256 v2 = _mm256_i32gather_ps(activations + 2, gather_index, 1);
    __m256 v3 = _mm256_i32gather_ps(activations + 3, gather_index, 1);

    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 a0 = _mm256_andnot_ps(sign_mask, v0);
    __m256 a1 = _mm256_andnot_ps(sign_mask, v1);
    __m256 a2 = _mm256_andnot_ps(sign_mask, v2);
    __m256 a3 = _mm256_andnot_ps(sign_mask, v3);

    __m256 sum = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 maximum = _mm_max_ps(_mm256_extractf128_ps(sum, 1), _mm256_castps256_ps128(sum));
    maximum = _mm_max_ps(maximum, _mm_movehl_ps(maximum, maximum));
    maximum = _mm_max_ss(maximum, _mm_movehdup_ps(maximum));

    const float scale = _mm_cvtss_f32(maximum) / 127.0f;
    *lut_scales = std::max(*lut_scales, scale);
}

inline void lut_ctor_g4_int8_avx2(int32_t act_k, int8_t* qlut, const float* activations, float* lut_scales, float* lut_biases) {
    __m256 lut[16];
    float biases = 0.0f;
    const __m256i gather_index = _mm256_set_epi32(112, 96, 80, 64, 48, 32, 16, 0);
    const float scale = *lut_scales;
    const float inverse_scale = scale != 0.0f ? 1.0f / scale : 0.0f;

    for (int k = 0; k < act_k / 32; ++k) {
        __m256 v0 = _mm256_i32gather_ps(activations + k * 32 + 0, gather_index, 1);
        __m256 v1 = _mm256_i32gather_ps(activations + k * 32 + 1, gather_index, 1);
        __m256 v2 = _mm256_i32gather_ps(activations + k * 32 + 2, gather_index, 1);
        __m256 v3 = _mm256_i32gather_ps(activations + k * 32 + 3, gather_index, 1);

        for (int g = 1; g < 16; g += 2) {
            lut[g] = v0;
            lut[g] = (g & 0b0010) ? _mm256_add_ps(lut[g], v1) : _mm256_sub_ps(lut[g], v1);
            lut[g] = (g & 0b0100) ? _mm256_add_ps(lut[g], v2) : _mm256_sub_ps(lut[g], v2);
            lut[g] = (g & 0b1000) ? _mm256_add_ps(lut[g], v3) : _mm256_sub_ps(lut[g], v3);
        }

        for (int g = 0; g < 16; g += 2) {
            lut[g] = _mm256_sub_ps(_mm256_setzero_ps(), lut[15 - g]);
        }

        biases += _mm256_addv_ps(lut[0]);

        for (int g = 0; g < 16; ++g) {
            lut[g] = _mm256_mul_ps(lut[g], _mm256_set1_ps(inverse_scale));
        }

        __m256i quantized_lut[4];
        const __m256i shuffle = _mm256_setr_epi8(
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
        );

        for (int g = 0; g < 4; ++g) {
            __m256i q0 = _mm256_cvtps_epi32(_mm256_round_ps(lut[g * 4 + 0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i q1 = _mm256_cvtps_epi32(_mm256_round_ps(lut[g * 4 + 1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i q2 = _mm256_cvtps_epi32(_mm256_round_ps(lut[g * 4 + 2], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i q3 = _mm256_cvtps_epi32(_mm256_round_ps(lut[g * 4 + 3], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

            q0 = _mm256_packs_epi32(q0, q1);
            q2 = _mm256_packs_epi32(q2, q3);
            q0 = _mm256_packs_epi16(q0, q2);

            quantized_lut[g] = _mm256_shuffle_epi8(q0, shuffle);
        }

        int32_t* output = reinterpret_cast<int32_t*>(qlut);
        for (int g = 0; g < 4; ++g) {
            output[k * 32 + 0 * 4 + g] = _mm256_extract_epi32(quantized_lut[g], 0);
            output[k * 32 + 1 * 4 + g] = _mm256_extract_epi32(quantized_lut[g], 1);
            output[k * 32 + 2 * 4 + g] = _mm256_extract_epi32(quantized_lut[g], 2);
            output[k * 32 + 3 * 4 + g] = _mm256_extract_epi32(quantized_lut[g], 3);
            output[k * 32 + 4 * 4 + g] = _mm256_extract_epi32(quantized_lut[g], 4);
            output[k * 32 + 5 * 4 + g] = _mm256_extract_epi32(quantized_lut[g], 5);
            output[k * 32 + 6 * 4 + g] = _mm256_extract_epi32(quantized_lut[g], 6);
            output[k * 32 + 7 * 4 + g] = _mm256_extract_epi32(quantized_lut[g], 7);
        }
    }
    *lut_scales = scale;
    *lut_biases = biases;
}

template <int Bits, int ActK = 32>
inline void tbl_update_avx2(int32_t m, int32_t k_groups, int32_t bit_plane_base, float* output, const int8_t* lut, const uint8_t* weights, const float* scales, const float* lut_scales, const float* lut_biases) {
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    __m128i* vector_lut = static_cast<__m128i*>(_mm_malloc(static_cast<size_t>(k_groups) * sizeof(__m128i), 32));

    for (int k = 0; k < k_groups; ++k) {
        vector_lut[k] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut + k * 16));
    }

    SignedWideningAdder<ActK> adder;

    for (int i = 0; i < m / 2; i += 16) {
        __m256 c0 = _mm256_setzero_ps();
        __m256 c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps();
        __m256 c3 = _mm256_setzero_ps();

        const int ib0 = bit_plane_base + i / 4;
        const int ib1 = ib0 + 1;
        const int ib2 = ib0 + 2;
        const int ib3 = ib0 + 3;

        for (int kk = 0; kk < k_groups; kk += ActK) {
            for (int k = 0; k < ActK; ++k) {
                __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + i * k_groups + (kk + k) * 16));
                __m128i bottom = _mm_and_si128(packed, nibble_mask);
                __m128i top = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
                __m256i lut_pair = _mm256_set_m128i(vector_lut[kk + k], vector_lut[kk + k]);
                __m256i lookup_indices = _mm256_set_m128i(top, bottom);
                adder.push(_mm256_shuffle_epi8(lut_pair, lookup_indices), k);
            }

            __m256 v00 = _mm256_cvtepi32_ps(extract_low_epi16_epi32(adder.get_low()));
            __m256 v01 = _mm256_cvtepi32_ps(extract_high_epi16_epi32(adder.get_low()));
            __m256 v10 = _mm256_cvtepi32_ps(extract_low_epi16_epi32(adder.get_high()));
            __m256 v11 = _mm256_cvtepi32_ps(extract_high_epi16_epi32(adder.get_high()));

            const int group_index = kk / ActK;
            const float lut_scale = lut_scales[group_index];
            const float lut_bias = lut_biases[group_index];

            #define APPLY_LUT(value, bit_index) \
                (((bit_index) % Bits) != 0 ? _mm256_mul_ps((value), _mm256_set1_ps(lut_scale)) : _mm256_fmadd_ps((value), _mm256_set1_ps(lut_scale), _mm256_set1_ps(lut_bias)))

            if (kk == 0) {
                c0 = APPLY_LUT(v00, ib0); c1 = APPLY_LUT(v01, ib1);
                c2 = APPLY_LUT(v10, ib2); c3 = APPLY_LUT(v11, ib3);
            } else {
                c0 = _mm256_add_ps(c0, APPLY_LUT(v00, ib0)); c1 = _mm256_add_ps(c1, APPLY_LUT(v01, ib1));
                c2 = _mm256_add_ps(c2, APPLY_LUT(v10, ib2)); c3 = _mm256_add_ps(c3, APPLY_LUT(v11, ib3));
            }
            #undef APPLY_LUT
        }

        __m256 s0 = _mm256_loadu_ps(scales + static_cast<size_t>(ib0 / Bits) * 8);
        __m256 s1 = _mm256_loadu_ps(scales + static_cast<size_t>(ib1 / Bits) * 8);
        __m256 s2 = _mm256_loadu_ps(scales + static_cast<size_t>(ib2 / Bits) * 8);
        __m256 s3 = _mm256_loadu_ps(scales + static_cast<size_t>(ib3 / Bits) * 8);

        _mm256_storeu_ps(output + i * 2, _mm256_fmadd_ps(c0, s0, _mm256_loadu_ps(output + i * 2)));
        _mm256_storeu_ps(output + i * 2 + 8, _mm256_fmadd_ps(c1, s1, _mm256_loadu_ps(output + i * 2 + 8)));
        _mm256_storeu_ps(output + i * 2 + 16, _mm256_fmadd_ps(c2, s2, _mm256_loadu_ps(output + i * 2 + 16)));
        _mm256_storeu_ps(output + i * 2 + 24, _mm256_fmadd_ps(c3, s3, _mm256_loadu_ps(output + i * 2 + 24)));
    }
    _mm_free(vector_lut);
}

struct GemmBatchedBuffers {
    int8_t* qlut_all;
    float* lut_scales_all;
    float* lut_biases_all;
    int num_groups;
    int N;
};

GemmBatchedBuffers alloc_gemm_batched_buffers(int k_groups, int ActK, int N) {
    GemmBatchedBuffers buffer;
    buffer.num_groups = k_groups / ActK;
    buffer.N = N;
    const size_t lut_size_per_n = static_cast<size_t>(k_groups) * ActK * 16;
    buffer.qlut_all = static_cast<int8_t*>(_mm_malloc(lut_size_per_n * N, 32));
    buffer.lut_scales_all = static_cast<float*>(_mm_malloc(static_cast<size_t>(buffer.num_groups) * N * sizeof(float), 32));
    buffer.lut_biases_all = static_cast<float*>(_mm_malloc(static_cast<size_t>(buffer.num_groups) * N * sizeof(float), 32));
    return buffer;
}

void free_gemm_batched_buffers(GemmBatchedBuffers& buffer) {
    _mm_free(buffer.qlut_all);
    _mm_free(buffer.lut_scales_all);
    _mm_free(buffer.lut_biases_all);
}

template <int Bits, int ActK = 32>
void build_all_luts(int K, int N, const float* activations_batch, GemmBatchedBuffers& buffer) {
    const int k_groups = K / 4;
    const int group_span = ActK * 4;
    const size_t qlut_stride = static_cast<size_t>(k_groups) * ActK * 16;

    for (int n = 0; n < N; ++n) {
        const float* activations = activations_batch + static_cast<size_t>(n) * K;
        int8_t* qlut = buffer.qlut_all + static_cast<size_t>(n) * qlut_stride;
        float* lut_scales = buffer.lut_scales_all + static_cast<size_t>(n) * buffer.num_groups;
        float* lut_biases = buffer.lut_biases_all + static_cast<size_t>(n) * buffer.num_groups;

        for (int group = 0; group < buffer.num_groups; ++group) {
            lut_scales[group] = 0.0f;
            for (int sub = 0; sub < group_span / 32; ++sub) {
                partial_max_g4_int8_k8(&lut_scales[group], activations + group * group_span + sub * 32);
            }
            lut_ctor_g4_int8_avx2(group_span, qlut + group * (ActK * 16), activations + group * group_span, &lut_scales[group], &lut_biases[group]);
        }
    }
}

template <int Bits, int ActK = 32>
void tmac_gemm_batched_target(int M, int K, int N, const uint8_t* weights, const float* activations_batch, const float* scales, float* output, GemmBatchedBuffers& buffer) {
    const int k_groups = K / 4;
    build_all_luts<Bits, ActK>(K, N, activations_batch, buffer);
    std::memset(output, 0, static_cast<size_t>(N) * M * sizeof(float));

    constexpr int M_BLOCK = 512;

    for (int m_start = 0; m_start < M; m_start += M_BLOCK) {
        const int m_chunk = std::min(M_BLOCK, M - m_start);
        const size_t weight_offset = static_cast<size_t>(m_start / 2) * k_groups;
        const int bit_plane_base = m_start / 8;

        for (int n = 0; n < N; ++n) {
            float* output_tile = output + static_cast<size_t>(n) * M + m_start;
            const size_t qlut_stride = static_cast<size_t>(k_groups) * ActK * 16;
            const int8_t* qlut = buffer.qlut_all + static_cast<size_t>(n) * qlut_stride;
            const float* lut_scales = buffer.lut_scales_all + static_cast<size_t>(n) * buffer.num_groups;
            const float* lut_biases = buffer.lut_biases_all + static_cast<size_t>(n) * buffer.num_groups;

            tbl_update_avx2<Bits, ActK>(m_chunk, k_groups, bit_plane_base, output_tile, qlut, weights + weight_offset, scales, lut_scales, lut_biases);
        }
    }
}

template <int Bits, int ActK = 32>
void golden_gemm_scalar_reference(
    int M, int K, int N, const uint8_t* weights, const float* scales, float* output_reference, const GemmBatchedBuffers& buffer)
{
    const int k_groups = K / 4;
    const size_t qlut_stride = static_cast<size_t>(k_groups) * ActK * 16;
    std::memset(output_reference, 0, static_cast<size_t>(N) * M * sizeof(float));

    for (int n = 0; n < N; ++n) {
        const int8_t* qlut = buffer.qlut_all + static_cast<size_t>(n) * qlut_stride;
        const float* lut_scales = buffer.lut_scales_all + static_cast<size_t>(n) * buffer.num_groups;
        const float* lut_biases = buffer.lut_biases_all + static_cast<size_t>(n) * buffer.num_groups;
        float* output = output_reference + static_cast<size_t>(n) * M;

        for (int i = 0; i < M / 2; i += 16) {
            float c0[8] = {0.0f}, c1[8] = {0.0f}, c2[8] = {0.0f}, c3[8] = {0.0f};
            const int ib0 = i / 4;
            const int ib1 = ib0 + 1;
            const int ib2 = ib0 + 2;
            const int ib3 = ib0 + 3;

            for (int kk = 0; kk < k_groups; kk += ActK) {
                float v00[8] = {0.0f}, v01[8] = {0.0f}, v10[8] = {0.0f}, v11[8] = {0.0f};

                for (int k = 0; k < ActK; ++k) {
                    for (int byte_index = 0; byte_index < 16; ++byte_index) {
                        const uint8_t packed = weights[i * k_groups + (kk + k) * 16 + byte_index];
                        const uint8_t bottom = packed & 0x0f;
                        const uint8_t top = (packed >> 4) & 0x0f;
                        const float bottom_value = static_cast<float>(qlut[(kk + k) * 16 + bottom]);
                        const float top_value = static_cast<float>(qlut[(kk + k) * 16 + top]);

                        if (byte_index < 8) {
                            v00[byte_index] += bottom_value;
                            v10[byte_index] += top_value;
                        } else {
                            v01[byte_index - 8] += bottom_value;
                            v11[byte_index - 8] += top_value;
                        }
                    }
                }

                const float lut_scale = lut_scales[kk / ActK];
                const float lut_bias = lut_biases[kk / ActK];

                // [修正點 1]：嚴格模擬 AVX2 的 FMA 捨入與覆蓋行為
                auto accumulate = [&](float* destination, const float* source, int bit_index, bool is_first) {
                    const bool add_bias = (bit_index % Bits) == 0;
                    for (int lane = 0; lane < 8; ++lane) {
                        float value;
                        if (add_bias) {
                            // 模擬 _mm256_fmadd_ps
                            value = std::fma(source[lane], lut_scale, lut_bias); 
                        } else {
                            // 模擬 _mm256_mul_ps
                            value = source[lane] * lut_scale; 
                        }

                        if (is_first) {
                            destination[lane] = value;
                        } else {
                            destination[lane] += value;
                        }
                    }
                };

                accumulate(c0, v00, ib0, kk == 0);
                accumulate(c1, v01, ib1, kk == 0);
                accumulate(c2, v10, ib2, kk == 0);
                accumulate(c3, v11, ib3, kk == 0);
            }

            const int scale_index_0 = (ib0 / Bits) * 8;
            const int scale_index_1 = (ib1 / Bits) * 8;
            const int scale_index_2 = (ib2 / Bits) * 8;
            const int scale_index_3 = (ib3 / Bits) * 8;

            // [修正點 2]：最終輸出必須模擬 AVX2 Kernel 儲存時的 FMA 行為
            for (int lane = 0; lane < 8; ++lane) {
                output[i * 2 + lane] = std::fma(c0[lane], scales[scale_index_0 + lane], output[i * 2 + lane]);
                output[i * 2 + 8 + lane] = std::fma(c1[lane], scales[scale_index_1 + lane], output[i * 2 + 8 + lane]);
                output[i * 2 + 16 + lane] = std::fma(c2[lane], scales[scale_index_2 + lane], output[i * 2 + 16 + lane]);
                output[i * 2 + 24 + lane] = std::fma(c3[lane], scales[scale_index_3 + lane], output[i * 2 + 24 + lane]);
            }
        }
    }
}

template <int Bits>
bool run_verification(int M, int K, int N, std::mt19937& random_engine) {
    constexpr int ActK = 32;
    const int k_groups = K / 4;

    std::cout << "[Verify W" << Bits << "] M=" << M << ", K=" << K << ", N=" << N << " ... ";

    if (M % 32 != 0 || K % 4 != 0 || k_groups % ActK != 0) {
        std::cout << "SKIP (alignment requirement not met)\n";
        return false;
    }

    const size_t weight_count = static_cast<size_t>(M / 2) * k_groups * 16;
    const size_t activation_count = static_cast<size_t>(N) * K;
    const size_t output_count = static_cast<size_t>(N) * M;
    const size_t scale_group_count = static_cast<size_t>((M / 8 + Bits - 1) / Bits);
    const size_t scale_count = std::max<size_t>(8, (scale_group_count + 1) * 8);

    uint8_t* weights = static_cast<uint8_t*>(_mm_malloc(weight_count * sizeof(uint8_t), 32));
    float* activations = static_cast<float*>(_mm_malloc(activation_count * sizeof(float), 32));
    float* scales = static_cast<float*>(_mm_malloc(scale_count * sizeof(float), 32));
    float* output_tmac = static_cast<float*>(_mm_malloc(output_count * sizeof(float), 32));
    float* output_golden = static_cast<float*>(_mm_malloc(output_count * sizeof(float), 32));

    if (weights == nullptr || activations == nullptr || scales == nullptr || 
        output_tmac == nullptr || output_golden == nullptr) {
        std::cout << "FAIL (memory allocation error)\n";
        _mm_free(weights);
        _mm_free(activations);
        _mm_free(scales);
        _mm_free(output_tmac);
        _mm_free(output_golden);
        return false;
    }

    std::uniform_int_distribution<int> weight_distribution(0, 255);
    std::uniform_real_distribution<float> activation_distribution(0.1f, 5.0f);

    for (size_t index = 0; index < weight_count; ++index) {
        weights[index] = static_cast<uint8_t>(weight_distribution(random_engine));
    }

    for (size_t index = 0; index < activation_count; ++index) {
        activations[index] = activation_distribution(random_engine);
    }

    for (size_t index = 0; index < scale_count; ++index) {
        scales[index] = 1.0f + 0.1f * static_cast<float>(index % 7);
    }

    std::memset(output_tmac, 0, output_count * sizeof(float));
    std::memset(output_golden, 0, output_count * sizeof(float));

    GemmBatchedBuffers buffer = alloc_gemm_batched_buffers(k_groups, ActK, N);

    if (buffer.qlut_all == nullptr || buffer.lut_scales_all == nullptr || buffer.lut_biases_all == nullptr) {
        std::cout << "FAIL (LUT buffer allocation error)\n";
        free_gemm_batched_buffers(buffer);
        _mm_free(weights);
        _mm_free(activations);
        _mm_free(scales);
        _mm_free(output_tmac);
        _mm_free(output_golden);
        return false;
    }

    tmac_gemm_batched_target<Bits, ActK>(M, K, N, weights, activations, scales, output_tmac, buffer);
    golden_gemm_scalar_reference<Bits, ActK>(M, K, N, weights, scales, output_golden, buffer);

    size_t bitwise_mismatch_count = 0;
    size_t tolerance_mismatch_count = 0;
    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;
    size_t max_error_index = 0;

    for (size_t index = 0; index < output_count; ++index) {
        const float actual_float = output_tmac[index];
        const float golden_float = output_golden[index];
        uint32_t actual_bits = 0;
        uint32_t golden_bits = 0;

        std::memcpy(&actual_bits, &actual_float, sizeof(uint32_t));
        std::memcpy(&golden_bits, &golden_float, sizeof(uint32_t));

        if (actual_bits != golden_bits) {
            ++bitwise_mismatch_count;
        }

        const double actual = static_cast<double>(actual_float);
        const double golden = static_cast<double>(golden_float);

        if (!std::isfinite(actual) || !std::isfinite(golden)) {
            if (actual_bits != golden_bits) {
                ++tolerance_mismatch_count;
                if (tolerance_mismatch_count == 1) {
                    std::cout << "\n  First non-finite mismatch: index=" << index
                              << " batch=" << index / M << " row=" << index % M
                              << " golden=" << golden_float << " tmac=" << actual_float << "\n";
                }
            }
            continue;
        }

        const double absolute_error = std::fabs(actual - golden);
        const double relative_error = absolute_error / std::max(1.0, std::fabs(golden));

        if (absolute_error > max_absolute_error) {
            max_absolute_error = absolute_error;
            max_error_index = index;
        }

        max_relative_error = std::max(max_relative_error, relative_error);

        /*
         * Bitwise equality is the primary goal.
         *
         * This fallback tolerance distinguishes a minor floating-point
         * mismatch from a true indexing or packing error.
         */
        const double tolerance = 1e-6 * std::max(1.0, std::fabs(golden)) + 1e-7;

        if (absolute_error > tolerance) {
            if (tolerance_mismatch_count == 0) {
                std::cout << "\n  First numerical mismatch: index=" << index
                          << " batch=" << index / M << " row=" << index % M
                          << " golden=" << std::setprecision(10) << golden_float
                          << " tmac=" << actual_float << " error=" << absolute_error
                          << " tolerance=" << tolerance << " golden_bits=0x"
                          << std::hex << golden_bits << " tmac_bits=0x" << actual_bits << std::dec << "\n";
            }
            ++tolerance_mismatch_count;
        }
    }

    const bool bitwise_pass = bitwise_mismatch_count == 0;
    const bool numerical_pass = tolerance_mismatch_count == 0;

    if (bitwise_pass) {
        std::cout << "\033[32mBITWISE PASS\033[0m outputs=" << output_count 
                  << " max_abs_err=" << max_absolute_error << "\n";
    } else if (numerical_pass) {
        std::cout << "\033[33mNUMERICAL PASS\033[0m bitwise_mismatches=" << bitwise_mismatch_count 
                  << " numerical_mismatches=0 max_abs_err=" << max_absolute_error 
                  << " max_rel_err=" << max_relative_error << "\n";
    } else {
        std::cout << "\033[31mFAIL\033[0m bitwise_mismatches=" << bitwise_mismatch_count 
                  << " numerical_mismatches=" << tolerance_mismatch_count 
                  << " max_abs_err=" << max_absolute_error 
                  << " max_rel_err=" << max_relative_error 
                  << " max_error_index=" << max_error_index 
                  << " batch=" << max_error_index / M 
                  << " row=" << max_error_index % M << "\n";
    }

    free_gemm_batched_buffers(buffer);
    _mm_free(weights);
    _mm_free(activations);
    _mm_free(scales);
    _mm_free(output_tmac);
    _mm_free(output_golden);

    /*
     * 回傳 numerical_pass，而不是 bitwise_pass。
     *
     * 原因：
     * 1. bitwise_pass 可用來確認完全一致。
     * 2. 若不同編譯器或環境仍有極少 ULP 差異，
     * numerical_pass 可避免把正常浮點差異誤判為演算法錯誤。
     */
    return numerical_pass;
}

template <int Bits>
bool run_bit_suite() {
    std::mt19937 random_engine(42);
    bool all_passed = true;

    std::cout << "\n==================================================\n"
              << " W" << Bits << "A16 verification\n"
              << "==================================================\n";

    all_passed &= run_verification<Bits>(256, 256, 2, random_engine);
    all_passed &= run_verification<Bits>(512, 512, 2, random_engine);
    all_passed &= run_verification<Bits>(1024, 1024, 4, random_engine);
    all_passed &= run_verification<Bits>(1024, 2048, 4, random_engine);
    all_passed &= run_verification<Bits>(2048, 1024, 8, random_engine);
    all_passed &= run_verification<Bits>(4096, 4096, 2, random_engine);
    all_passed &= run_verification<Bits>(4096, 11008, 2, random_engine);
    all_passed &= run_verification<Bits>(11008, 4096, 2, random_engine);

    return all_passed;
}

int main() {
    std::cout << "==================================================================\n"
              << " T-MAC W2/W3/W4 GEMM Verification Suite\n"
              << " AVX2 tiled implementation vs scalar golden reference\n"
              << " Build: " << __DATE__ << " " << __TIME__ << "\n"
              << "==================================================================\n";

    const bool w2_passed = run_bit_suite<2>();
    const bool w3_passed = run_bit_suite<3>();
    const bool w4_passed = run_bit_suite<4>();

    std::cout << "\n==================================================================\n"
              << " Final result\n"
              << " W2: " << (w2_passed ? "PASS" : "FAIL") << "\n"
              << " W3: " << (w3_passed ? "PASS" : "FAIL") << "\n"
              << " W4: " << (w4_passed ? "PASS" : "FAIL") << "\n"
              << "==================================================================\n";

    return (w2_passed && w3_passed && w4_passed) ? 0 : 1;
}