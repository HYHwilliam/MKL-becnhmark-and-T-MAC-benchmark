#include <immintrin.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <string>

static inline float _mm256_addv_ps(const __m256 v) {
    __m128 res = _mm256_extractf128_ps(v, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(v));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

#define extract_low_epi8_epi16(v) _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v))
#define extract_high_epi8_epi16(v) _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1))
#define extract_low_epi16_epi32(v) _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v))
#define extract_high_epi16_epi32(v) _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1))

template <int N>
struct SignedWideningAdder {
    __m256i lhs_low;
    __m256i lhs_high;

    inline void push(__m256i v, int k) {
        if (k == 0) {
            lhs_low = extract_low_epi8_epi16(v);
            lhs_high = extract_high_epi8_epi16(v);
        } else {
            lhs_low = _mm256_add_epi16(lhs_low, extract_low_epi8_epi16(v));
            lhs_high = _mm256_add_epi16(lhs_high, extract_high_epi8_epi16(v));
        }
    }

    inline __m256i get_low() { return lhs_low; }
    inline __m256i get_high() { return lhs_high; }
};

inline void partial_max_g4_int8_k8(float* lut_scales, const float* b) {
    const __m256i vec_bi = _mm256_set_epi32(112, 96, 80, 64, 48, 32, 16, 0);

    __m256 vb0 = _mm256_i32gather_ps(b + 0, vec_bi, 1);
    __m256 vb1 = _mm256_i32gather_ps(b + 1, vec_bi, 1);
    __m256 vb2 = _mm256_i32gather_ps(b + 2, vec_bi, 1);
    __m256 vb3 = _mm256_i32gather_ps(b + 3, vec_bi, 1);

    const __m256 sign = _mm256_set1_ps(-0.0f);

    __m256 a0 = _mm256_andnot_ps(sign, vb0);
    __m256 a1 = _mm256_andnot_ps(sign, vb1);
    __m256 a2 = _mm256_andnot_ps(sign, vb2);
    __m256 a3 = _mm256_andnot_ps(sign, vb3);

    __m256 sum = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));

    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(sum, 1), _mm256_castps256_ps128(sum));

    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));

    float scale = _mm_cvtss_f32(max4) / 127.0f;
    *lut_scales = std::max(*lut_scales, scale);
}

inline void lut_ctor_g4_int8_avx2(int32_t act_k, int8_t* qlut, const float* b, float* lut_scales, float* lut_biases) {
    __m256 vec_lut[16];
    float biases = 0.0f;

    const __m256i vec_bi = _mm256_set_epi32(112, 96, 80, 64, 48, 32, 16, 0);

    float scales = *lut_scales;
    float inverse_scale = scales != 0.0f ? 1.0f / scales : 0.0f;

    for (int k = 0; k < act_k / 32; ++k) {
        __m256 vb0 = _mm256_i32gather_ps(b + k * 32 + 0, vec_bi, 1);
        __m256 vb1 = _mm256_i32gather_ps(b + k * 32 + 1, vec_bi, 1);
        __m256 vb2 = _mm256_i32gather_ps(b + k * 32 + 2, vec_bi, 1);
        __m256 vb3 = _mm256_i32gather_ps(b + k * 32 + 3, vec_bi, 1);

        for (int g = 1; g < 16; g += 2) {
            vec_lut[g] = vb0;
            if (g & 0b0010) vec_lut[g] = _mm256_add_ps(vec_lut[g], vb1);
            else vec_lut[g] = _mm256_sub_ps(vec_lut[g], vb1);

            if (g & 0b0100) vec_lut[g] = _mm256_add_ps(vec_lut[g], vb2);
            else vec_lut[g] = _mm256_sub_ps(vec_lut[g], vb2);

            if (g & 0b1000) vec_lut[g] = _mm256_add_ps(vec_lut[g], vb3);
            else vec_lut[g] = _mm256_sub_ps(vec_lut[g], vb3);
        }

        for (int g = 0; g < 16; g += 2) {
            vec_lut[g] = _mm256_sub_ps(_mm256_setzero_ps(), vec_lut[15 - g]);
        }

        biases += _mm256_addv_ps(vec_lut[0]);

        for (int g = 0; g < 16; ++g) {
            vec_lut[g] = _mm256_mul_ps(vec_lut[g], _mm256_set1_ps(inverse_scale));
        }

        __m256i vec_qlut[4];

        const __m256i shuffle = _mm256_setr_epi8(
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
        );

        for (int g = 0; g < 4; ++g) {
            __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 2], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 3], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

            i0 = _mm256_packs_epi32(i0, i1);
            i2 = _mm256_packs_epi32(i2, i3);
            i0 = _mm256_packs_epi16(i0, i2);

            vec_qlut[g] = _mm256_shuffle_epi8(i0, shuffle);
        }

        int32_t* qlut_i32 = reinterpret_cast<int32_t*>(qlut);

        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 0 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 0);
            qlut_i32[k * 32 + 1 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 1);
            qlut_i32[k * 32 + 2 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 2);
            qlut_i32[k * 32 + 3 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 3);
            qlut_i32[k * 32 + 4 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 4);
            qlut_i32[k * 32 + 5 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 5);
            qlut_i32[k * 32 + 6 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 6);
            qlut_i32[k * 32 + 7 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 7);
        }
    }

    *lut_scales = scales;
    *lut_biases = biases;
}

template <int Bits, int ActK = 32>
inline void tbl_update_avx2(int32_t m, int32_t k_groups, int32_t bit_plane_base, float* c, const int8_t* lut, const uint8_t* a, const float* scales, const float* lut_scales, const float* lut_biases) {
    const __m128i vec_mask = _mm_set1_epi8(0x0f);
    __m128i* vec_lut = static_cast<__m128i*>(_mm_malloc(static_cast<size_t>(k_groups) * sizeof(__m128i), 32));

    for (int k = 0; k < k_groups; ++k) {
        vec_lut[k] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut + k * 16));
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
                __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i * k_groups + (kk + k) * 16));
                __m128i bottom = _mm_and_si128(packed, vec_mask);
                __m128i top = _mm_and_si128(_mm_srli_epi16(packed, 4), vec_mask);
                __m256i lut_pair = _mm256_set_m128i(vec_lut[kk + k], vec_lut[kk + k]);
                __m256i indices = _mm256_set_m128i(top, bottom);
                adder.push(_mm256_shuffle_epi8(lut_pair, indices), k);
            }

            __m256 v00 = _mm256_cvtepi32_ps(extract_low_epi16_epi32(adder.get_low()));
            __m256 v01 = _mm256_cvtepi32_ps(extract_high_epi16_epi32(adder.get_low()));
            __m256 v10 = _mm256_cvtepi32_ps(extract_low_epi16_epi32(adder.get_high()));
            __m256 v11 = _mm256_cvtepi32_ps(extract_high_epi16_epi32(adder.get_high()));

            const int group_index = kk / ActK;
            const float lut_scale = lut_scales[group_index];
            const float lut_bias = lut_biases[group_index];

            #define APPLY_LUT(value, bit_index) \
                (((bit_index) % Bits) != 0 \
                    ? _mm256_mul_ps((value), _mm256_set1_ps(lut_scale)) \
                    : _mm256_fmadd_ps((value), _mm256_set1_ps(lut_scale), _mm256_set1_ps(lut_bias)))

            if (kk == 0) {
                c0 = APPLY_LUT(v00, ib0);
                c1 = APPLY_LUT(v01, ib1);
                c2 = APPLY_LUT(v10, ib2);
                c3 = APPLY_LUT(v11, ib3);
            } else {
                c0 = _mm256_add_ps(c0, APPLY_LUT(v00, ib0));
                c1 = _mm256_add_ps(c1, APPLY_LUT(v01, ib1));
                c2 = _mm256_add_ps(c2, APPLY_LUT(v10, ib2));
                c3 = _mm256_add_ps(c3, APPLY_LUT(v11, ib3));
            }

            #undef APPLY_LUT
        }

        __m256 s0 = _mm256_loadu_ps(scales + static_cast<size_t>(ib0 / Bits) * 8);
        __m256 s1 = _mm256_loadu_ps(scales + static_cast<size_t>(ib1 / Bits) * 8);
        __m256 s2 = _mm256_loadu_ps(scales + static_cast<size_t>(ib2 / Bits) * 8);
        __m256 s3 = _mm256_loadu_ps(scales + static_cast<size_t>(ib3 / Bits) * 8);

        _mm256_storeu_ps(c + i * 2, _mm256_fmadd_ps(c0, s0, _mm256_loadu_ps(c + i * 2)));
        _mm256_storeu_ps(c + i * 2 + 8, _mm256_fmadd_ps(c1, s1, _mm256_loadu_ps(c + i * 2 + 8)));
        _mm256_storeu_ps(c + i * 2 + 16, _mm256_fmadd_ps(c2, s2, _mm256_loadu_ps(c + i * 2 + 16)));
        _mm256_storeu_ps(c + i * 2 + 24, _mm256_fmadd_ps(c3, s3, _mm256_loadu_ps(c + i * 2 + 24)));
    }

    _mm_free(vec_lut);
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
void tmac_gemm_batched(int M, int K, int N, const uint8_t* weights, const float* activations_batch, const float* scales, float* output, GemmBatchedBuffers& buffer) {
    const int k_groups = K / 4;
    const int group_span = ActK * 4;
    const size_t qlut_stride = static_cast<size_t>(k_groups) * ActK * 16;

    for (int n = 0; n < N; ++n) {
        const float* activation = activations_batch + static_cast<size_t>(n) * K;
        int8_t* qlut = buffer.qlut_all + static_cast<size_t>(n) * qlut_stride;
        float* lut_scales = buffer.lut_scales_all + static_cast<size_t>(n) * buffer.num_groups;
        float* lut_biases = buffer.lut_biases_all + static_cast<size_t>(n) * buffer.num_groups;

        for (int group = 0; group < buffer.num_groups; ++group) {
            lut_scales[group] = 0.0f;

            for (int sub = 0; sub < group_span / 32; ++sub) {
                partial_max_g4_int8_k8(&lut_scales[group], activation + group * group_span + sub * 32);
            }

            lut_ctor_g4_int8_avx2(group_span, qlut + group * (ActK * 16), activation + group * group_span, &lut_scales[group], &lut_biases[group]);
        }
    }

    constexpr int M_BLOCK = 512;

    std::memset(output, 0, static_cast<size_t>(N) * M * sizeof(float));

    for (int m_start = 0; m_start < M; m_start += M_BLOCK) {
        const int m_chunk = std::min(M_BLOCK, M - m_start);
        const size_t weight_offset = static_cast<size_t>(m_start / 2) * k_groups;
        const int bit_plane_base = m_start / 8;

        for (int n = 0; n < N; ++n) {
            float* output_tile = output + static_cast<size_t>(n) * M + m_start;
            const int8_t* qlut = buffer.qlut_all + static_cast<size_t>(n) * qlut_stride;
            const float* lut_scales = buffer.lut_scales_all + static_cast<size_t>(n) * buffer.num_groups;
            const float* lut_biases = buffer.lut_biases_all + static_cast<size_t>(n) * buffer.num_groups;

            tbl_update_avx2<Bits, ActK>(m_chunk, k_groups, bit_plane_base, output_tile, qlut, weights + weight_offset, scales, lut_scales, lut_biases);
        }
    }
}

void pack_tmac_weights(int M, int K, const std::vector<uint8_t>& logical_weights, uint8_t* packed_weights) {
    const int k_groups = K / 4;
    const size_t packed_count = static_cast<size_t>(M / 32) * k_groups * 16;

    std::memset(packed_weights, 0, packed_count);

    for (int m = 0; m < M; ++m) {
        const int block_m = m / 32;
        const int m_in_block = m % 32;
        const int group = m_in_block / 8;
        const int lane = m_in_block % 8;

        const int byte_in_block = (group == 0 || group == 2) ? lane : 8 + lane;
        const bool top_nibble = group == 2 || group == 3;

        for (int k_group = 0; k_group < k_groups; ++k_group) {
            const size_t logical_base = static_cast<size_t>(m) * K + k_group * 4;
            const uint8_t b0 = logical_weights[logical_base + 0];
            const uint8_t b1 = logical_weights[logical_base + 1];
            const uint8_t b2 = logical_weights[logical_base + 2];
            const uint8_t b3 = logical_weights[logical_base + 3];

            const uint8_t index = static_cast<uint8_t>((b3 << 3) | (b2 << 2) | (b1 << 1) | b0);
            const size_t packed_offset = static_cast<size_t>(block_m) * k_groups * 16 + static_cast<size_t>(k_group) * 16 + byte_in_block;

            if (top_nibble) {
                packed_weights[packed_offset] |= static_cast<uint8_t>(index << 4);
            } else {
                packed_weights[packed_offset] |= static_cast<uint8_t>(index & 0x0f);
            }
        }
    }
}

template <int Bits, int ActK = 32>
void naive_batched_gemm(int M, int K, int N, const std::vector<uint8_t>& logical_weights, const std::vector<float>& activations_batch, const float* scales, float* output) {
    const int k_groups = K / 4;
    const int num_groups = k_groups / ActK;
    const int group_span = ActK * 4;

    std::fill(output, output + static_cast<size_t>(N) * M, 0.0f);

    for (int n = 0; n < N; ++n) {
        const float* activations = activations_batch.data() + static_cast<size_t>(n) * K;
        std::vector<double> group_biases(num_groups, 0.0);

        for (int group = 0; group < num_groups; ++group) {
            double activation_sum = 0.0;
            const int k_begin = group * group_span;
            const int k_end = k_begin + group_span;

            for (int k = k_begin; k < k_end; ++k) {
                activation_sum += static_cast<double>(activations[k]);
            }
            group_biases[group] = -activation_sum;
        }

        for (int m = 0; m < M; ++m) {
            const int bit_plane = (m / 8) % Bits;
            const bool gets_bias = bit_plane == 0;
            double final_accumulator = 0.0;

            for (int group = 0; group < num_groups; ++group) {
                const int k_begin = group * group_span;
                const int k_end = k_begin + group_span;
                double group_accumulator = 0.0;

                for (int k = k_begin; k < k_end; ++k) {
                    const double weight = logical_weights[static_cast<size_t>(m) * K + k] ? 1.0 : -1.0;
                    group_accumulator += weight * static_cast<double>(activations[k]);
                }
                final_accumulator += group_accumulator;

                if (gets_bias) {
                    final_accumulator += group_biases[group];
                }
            }

            const int scale_group = (m / 8) / Bits;
            const int scale_lane = m % 8;
            const float weight_scale = scales[static_cast<size_t>(scale_group) * 8 + scale_lane];

            output[static_cast<size_t>(n) * M + m] = static_cast<float>(final_accumulator * static_cast<double>(weight_scale));
        }
    }
}

struct VerificationConfig {
    int M;
    int K;
    int N;
};

template <int Bits>
bool verify_config(const VerificationConfig& config, std::mt19937& random_engine) {
    constexpr int ActK = 32;

    const int M = config.M;
    const int K = config.K;
    const int N = config.N;

    const int k_groups = K / 4;

    std::cout << "[Independent W" << Bits << "] " << M << "x" << K << "x" << N << " ... ";

    if (M % 32 != 0 || K % 128 != 0) {
        std::cout << "SKIP (M must be divisible by 32, K must be divisible by 128)\n";
        return false;
    }

    const size_t logical_weight_count = static_cast<size_t>(M) * K;
    const size_t packed_weight_count = static_cast<size_t>(M / 32) * k_groups * 16;
    const size_t activation_count = static_cast<size_t>(N) * K;
    const size_t output_count = static_cast<size_t>(N) * M;
    const size_t scale_count = std::max<size_t>(8, (static_cast<size_t>((M / 8 + Bits - 1) / Bits) + 1) * 8);

    std::vector<uint8_t> logical_weights(logical_weight_count);
    std::vector<float> activations_batch(activation_count);
    std::vector<float> output_naive(output_count, 0.0f);

    std::uniform_int_distribution<int> bit_distribution(0, 1);
    std::uniform_real_distribution<float> activation_distribution(-0.1f, 0.1f);

    for (uint8_t& value : logical_weights) {
        value = static_cast<uint8_t>(bit_distribution(random_engine));
    }

    for (float& value : activations_batch) {
        value = activation_distribution(random_engine);
    }

    uint8_t* packed_weights = static_cast<uint8_t*>(_mm_malloc(packed_weight_count, 32));
    float* scales = static_cast<float*>(_mm_malloc(scale_count * sizeof(float), 32));
    float* output_tmac = static_cast<float*>(_mm_malloc(output_count * sizeof(float), 32));

    if (packed_weights == nullptr || scales == nullptr || output_tmac == nullptr) {
        std::cout << "FAIL (allocation error)\n";
        _mm_free(packed_weights);
        _mm_free(scales);
        _mm_free(output_tmac);
        return false;
    }

    pack_tmac_weights(M, K, logical_weights, packed_weights);

    for (size_t index = 0; index < scale_count; ++index) {
        scales[index] = 1.0f;
    }

    GemmBatchedBuffers buffer = alloc_gemm_batched_buffers(k_groups, ActK, N);

    if (buffer.qlut_all == nullptr || buffer.lut_scales_all == nullptr || buffer.lut_biases_all == nullptr) {
        std::cout << "FAIL (LUT allocation error)\n";
        free_gemm_batched_buffers(buffer);
        _mm_free(packed_weights);
        _mm_free(scales);
        _mm_free(output_tmac);
        return false;
    }

    const auto tmac_start = std::chrono::high_resolution_clock::now();
    tmac_gemm_batched<Bits, ActK>(M, K, N, packed_weights, activations_batch.data(), scales, output_tmac, buffer);
    const auto tmac_end = std::chrono::high_resolution_clock::now();

    const auto naive_start = std::chrono::high_resolution_clock::now();
    naive_batched_gemm<Bits, ActK>(M, K, N, logical_weights, activations_batch, scales, output_naive.data());
    const auto naive_end = std::chrono::high_resolution_clock::now();

    const double tmac_ms = std::chrono::duration<double, std::milli>(tmac_end - tmac_start).count();
    const double naive_ms = std::chrono::duration<double, std::milli>(naive_end - naive_start).count();

    size_t mismatch_count = 0;
    size_t first_mismatch_index = 0;
    size_t max_error_index = 0;

    double max_absolute_error = 0.0;
    double mean_absolute_error = 0.0;
    double squared_error_sum = 0.0;

    constexpr double absolute_tolerance = 0.08;
    constexpr double relative_tolerance = 0.02;

    for (size_t index = 0; index < output_count; ++index) {
        const double actual = static_cast<double>(output_tmac[index]);
        const double reference = static_cast<double>(output_naive[index]);
        const double absolute_error = std::fabs(actual - reference);
        const double tolerance = absolute_tolerance + relative_tolerance * std::fabs(reference);

        mean_absolute_error += absolute_error;
        squared_error_sum += absolute_error * absolute_error;

        if (absolute_error > max_absolute_error) {
            max_absolute_error = absolute_error;
            max_error_index = index;
        }

        if (absolute_error > tolerance) {
            if (mismatch_count == 0) {
                first_mismatch_index = index;
            }
            ++mismatch_count;
        }
    }

    mean_absolute_error /= static_cast<double>(output_count);
    const double rmse = std::sqrt(squared_error_sum / static_cast<double>(output_count));
    const bool passed = mismatch_count == 0;

    std::cout << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m")
              << " mismatches=" << mismatch_count << "/" << output_count
              << " max_abs_err=" << max_absolute_error << " mean_abs_err=" << mean_absolute_error
              << " rmse=" << rmse << " tmac=" << tmac_ms << "ms naive=" << naive_ms << "ms\n";

    if (!passed) {
        const size_t index = first_mismatch_index;
        const double actual = static_cast<double>(output_tmac[index]);
        const double reference = static_cast<double>(output_naive[index]);
        const double error = std::fabs(actual - reference);
        const double tolerance = absolute_tolerance + relative_tolerance * std::fabs(reference);

        std::cout << "  First mismatch: index=" << index << " batch=" << index / M
                  << " row=" << index % M << " tmac=" << actual << " naive=" << reference
                  << " error=" << error << " tolerance=" << tolerance << "\n";
    }

    std::cout << "  Max-error element: index=" << max_error_index << " batch=" << max_error_index / M
              << " row=" << max_error_index % M << " tmac=" << output_tmac[max_error_index]
              << " naive=" << output_naive[max_error_index] << "\n";

    const size_t preview_count = std::min<size_t>(4, output_count);

    std::cout << "  T-MAC [0.." << preview_count - 1 << "] =";
    for (size_t i = 0; i < preview_count; ++i) {
        std::cout << " " << output_tmac[i];
    }

    std::cout << "\n  Naive [0.." << preview_count - 1 << "] =";
    for (size_t i = 0; i < preview_count; ++i) {
        std::cout << " " << output_naive[i];
    }
    std::cout << "\n";

    free_gemm_batched_buffers(buffer);
    _mm_free(packed_weights);
    _mm_free(scales);
    _mm_free(output_tmac);

    return passed;
}

template <int Bits>
bool run_independent_suite() {
    std::mt19937 random_engine(42);
    const std::vector<VerificationConfig> configurations = {
        {256, 256, 1},
        {256, 256, 3},
        {512, 512, 2},
        {1024, 1024, 4},
        {1024, 2048, 3},
        {2048, 1024, 3}
    };

    bool all_passed = true;

    std::cout << "\n============================================================\n"
              << " Independent W" << Bits << "A16 GEMM verification\n"
              << "============================================================\n";

    for (const VerificationConfig& configuration : configurations) {
        all_passed &= verify_config<Bits>(configuration, random_engine);
    }

    return all_passed;
}

int main() {
    std::cout << "==================================================================\n"
              << " T-MAC W2/W3/W4 Independent GEMM Verification\n"
              << " T-MAC packed/LUT kernel vs logical W_bit naive N-M-K loops\n"
              << " Activation range: [-0.1, 0.1]\n"
              << " Tolerance: abs=0.08, rel=0.02\n"
              << " Build: " << __DATE__ << " " << __TIME__ << "\n"
              << "==================================================================\n";

    const bool w2_passed = run_independent_suite<2>();
    const bool w3_passed = run_independent_suite<3>();
    const bool w4_passed = run_independent_suite<4>();

    std::cout << "\n==================================================================\n"
              << " Final result\n"
              << " W2: " << (w2_passed ? "PASS" : "FAIL") << "\n"
              << " W3: " << (w3_passed ? "PASS" : "FAIL") << "\n"
              << " W4: " << (w4_passed ? "PASS" : "FAIL") << "\n"
              << "==================================================================\n";

    return (w2_passed && w3_passed && w4_passed) ? 0 : 1;
}