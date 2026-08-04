// t_mac_gemm_w234.cpp
//
// Unified T-MAC W2A16 / W3A16 / W4A16 GEMM benchmark.
// W3 requires global bit-plane indexing across M tiles.
// The same global indexing is also used for W2 and W4.

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
    __m256i lhs_low, lhs_high;

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
            if (g & 0b0010) vec_lut[g] = _mm256_add_ps(vec_lut[g], vb1); else vec_lut[g] = _mm256_sub_ps(vec_lut[g], vb1);
            if (g & 0b0100) vec_lut[g] = _mm256_add_ps(vec_lut[g], vb2); else vec_lut[g] = _mm256_sub_ps(vec_lut[g], vb2);
            if (g & 0b1000) vec_lut[g] = _mm256_add_ps(vec_lut[g], vb3); else vec_lut[g] = _mm256_sub_ps(vec_lut[g], vb3);
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

struct Config {
    int M, K, N;
};

template <int Bits>
void run_config(const Config& config, std::mt19937& rng) {
    constexpr int ActK = 32;
    const int M = config.M, K = config.K, N = config.N;
    const int k_groups = K / 4;

    if (M % 32 != 0 || K % 4 != 0 || k_groups % ActK != 0) {
        std::cout << "SKIP " << M << "x" << K << "x" << N << " (alignment requirement not met)\n";
        return;
    }

    std::uniform_real_distribution<float> activation_distribution(0.1f, 5.0f);
    std::uniform_int_distribution<int> byte_distribution(0, 255);

    const size_t weight_allocation_count = static_cast<size_t>(M) * k_groups * 16;
    const size_t weight_initialized_count = static_cast<size_t>(M / 2) * k_groups * 16;
    const size_t activation_count = static_cast<size_t>(N) * K;
    const size_t output_count = static_cast<size_t>(N) * M;
    const size_t scale_count = std::max<size_t>(8, (static_cast<size_t>((M / 8 + Bits - 1) / Bits) + 1) * 8);

    uint8_t* weights = static_cast<uint8_t*>(_mm_malloc(weight_allocation_count, 32));
    float* activations = static_cast<float*>(_mm_malloc(activation_count * sizeof(float), 32));
    float* output = static_cast<float*>(_mm_malloc(output_count * sizeof(float), 32));
    float* scales = static_cast<float*>(_mm_malloc(scale_count * sizeof(float), 32));

    if (!weights || !activations || !output || !scales) {
        std::cerr << "Memory allocation failed for " << M << "x" << K << "x" << N << "\n";
        _mm_free(weights); _mm_free(activations); _mm_free(output); _mm_free(scales);
        return;
    }

    for (size_t i = 0; i < weight_initialized_count; ++i) weights[i] = static_cast<uint8_t>(byte_distribution(rng));
    for (size_t i = 0; i < activation_count; ++i) activations[i] = activation_distribution(rng);
    for (size_t i = 0; i < scale_count; ++i) scales[i] = 1.0f;

    GemmBatchedBuffers buffer = alloc_gemm_batched_buffers(k_groups, ActK, N);

    if (!buffer.qlut_all || !buffer.lut_scales_all || !buffer.lut_biases_all) {
        std::cerr << "Buffer allocation failed for " << M << "x" << K << "x" << N << "\n";
        free_gemm_batched_buffers(buffer);
        _mm_free(weights); _mm_free(activations); _mm_free(output); _mm_free(scales);
        return;
    }

    tmac_gemm_batched<Bits, ActK>(M, K, N, weights, activations, scales, output, buffer);

    const double total_flops_per_call = 2.0 * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N);
    const int iterations = static_cast<int>(std::max(2.0, std::min(100.0, 5e10 / total_flops_per_call)));

    const auto start = std::chrono::high_resolution_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        tmac_gemm_batched<Bits, ActK>(M, K, N, weights, activations, scales, output, buffer);
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const double average_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    const double gflops = total_flops_per_call / (average_ms / 1000.0) / 1e9;

    std::cout << std::left << std::setw(20) << (std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N))
              << std::setw(16) << average_ms
              << std::setw(16) << gflops << " GFLOPS"
              << " iterations=" << iterations << "\n";

    free_gemm_batched_buffers(buffer);
    _mm_free(weights); _mm_free(activations); _mm_free(output); _mm_free(scales);
}

template <int Bits>
void run_suite() {
    std::mt19937 rng(42);
    const std::vector<int> batch_sizes = {1, 8, 32, 128, 512};
    const std::vector<Config> square_shapes = {{1024, 1024, 0}, {2048, 2048, 0}, {4096, 4096, 0}, {8192, 8192, 0}};
    const std::vector<Config> k_expansion_shapes = {{1024, 4096, 0}, {2048, 8192, 0}, {4096, 11008, 0}, {4096, 14336, 0}};
    const std::vector<Config> m_expansion_shapes = {{4096, 1024, 0}, {8192, 2048, 0}, {11008, 4096, 0}, {14336, 4096, 0}};
    const std::vector<Config> asymmetric_shapes = {{1024, 2048, 0}, {2048, 1024, 0}, {2048, 4096, 0}, {4096, 2048, 0}};

    std::cout << "\n==================================================================\n"
              << " T-MAC W" << Bits << "A16 Batched GEMM Benchmark\n"
              << " Bits = " << Bits << ", ActK = 32, M_BLOCK = 512\n"
              << "==================================================================\n";

    std::cout << "\n=== Square GEMM ===\n";
    for (int N : batch_sizes) {
        //std::cout << "\n--- N = " << N << " ---\n";
        for (const Config& shape : square_shapes) run_config<Bits>({shape.M, shape.K, N}, rng);
    }

    std::cout << "\n=== Rectangular GEMM: K Expansion ===\n";
    for (int N : batch_sizes) {
        //std::cout << "\n--- N = " << N << " ---\n";
        for (const Config& shape : k_expansion_shapes) run_config<Bits>({shape.M, shape.K, N}, rng);
    }

    std::cout << "\n=== Rectangular GEMM: M Expansion ===\n";
    for (int N : batch_sizes) {
        //std::cout << "\n--- N = " << N << " ---\n";
        for (const Config& shape : m_expansion_shapes) run_config<Bits>({shape.M, shape.K, N}, rng);
    }

    std::cout << "\n=== Medium Asymmetric GEMM ===\n";
    for (int N : batch_sizes) {
        //std::cout << "\n--- N = " << N << " ---\n";
        for (const Config& shape : asymmetric_shapes) run_config<Bits>({shape.M, shape.K, N}, rng);
    }
}

int main(int argc, char** argv) {
    if (argc == 1) {
        run_suite<2>(); run_suite<3>(); run_suite<4>();
        return 0;
    }

    const int bits = std::stoi(argv[1]);
    if (bits == 2) run_suite<2>();
    else if (bits == 3) run_suite<3>();
    else if (bits == 4) run_suite<4>();
    else {
        std::cerr << "Usage: " << argv[0] << " [2|3|4]\n";
        return 1;
    }

    return 0;
}