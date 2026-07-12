#include <immintrin.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <cmath>

typedef float float_type;

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

inline int32_t partial_max_g4_int8_k8(float* lut_scales, float* b) {
    const __m256i vec_bi = _mm256_set_epi32(112, 96, 80, 64, 48, 32, 16, 0);

    __m256 vec_b0 = _mm256_i32gather_ps(b + 0, vec_bi, 1);
    __m256 vec_b1 = _mm256_i32gather_ps(b + 1, vec_bi, 1);
    __m256 vec_b2 = _mm256_i32gather_ps(b + 2, vec_bi, 1);
    __m256 vec_b3 = _mm256_i32gather_ps(b + 3, vec_bi, 1);

    const __m256 vec_sign = _mm256_set1_ps(-0.0f);
    __m256 vec_babs0 = _mm256_andnot_ps(vec_sign, vec_b0);
    __m256 vec_babs1 = _mm256_andnot_ps(vec_sign, vec_b1);
    __m256 vec_babs2 = _mm256_andnot_ps(vec_sign, vec_b2);
    __m256 vec_babs3 = _mm256_andnot_ps(vec_sign, vec_b3);

    __m256 abssum = _mm256_add_ps(_mm256_add_ps(vec_babs0, vec_babs1),
                                _mm256_add_ps(vec_babs2, vec_babs3));

    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(abssum, 1), _mm256_castps256_ps128(abssum));
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));

    float scales = _mm_cvtss_f32(max4) / 127.0f;
    *lut_scales = std::max(*lut_scales, scales);

    return 0;
}

inline int32_t lut_ctor_g4_int8_avx2(int32_t act_k, int8_t* qlut, float_type* b, float_type* lut_scales, float_type* lut_biases) {
    __m256 vec_lut[16];
    float biases = 0.0;
    const __m256i vec_bi = _mm256_set_epi32(112, 96, 80, 64, 48, 32, 16, 0);
    float scales = *lut_scales;
    float t_scales = scales ? 1.0f / scales : 0.0f;

    for (int k = 0; k < act_k / 32; ++k) {
        __m256 vec_b0 = _mm256_i32gather_ps(b + k * 32 + 0, vec_bi, 1);
        __m256 vec_b1 = _mm256_i32gather_ps(b + k * 32 + 1, vec_bi, 1);
        __m256 vec_b2 = _mm256_i32gather_ps(b + k * 32 + 2, vec_bi, 1);
        __m256 vec_b3 = _mm256_i32gather_ps(b + k * 32 + 3, vec_bi, 1);

        for (int g = 1; g < 16; g += 2) {
            vec_lut[g] = vec_b0;
            if (g & 0b0010) vec_lut[g] = _mm256_add_ps(vec_lut[g], vec_b1); else vec_lut[g] = _mm256_sub_ps(vec_lut[g], vec_b1);
            if (g & 0b0100) vec_lut[g] = _mm256_add_ps(vec_lut[g], vec_b2); else vec_lut[g] = _mm256_sub_ps(vec_lut[g], vec_b2);
            if (g & 0b1000) vec_lut[g] = _mm256_add_ps(vec_lut[g], vec_b3); else vec_lut[g] = _mm256_sub_ps(vec_lut[g], vec_b3);
        }
        for (int g = 0; g < 16; g += 2) vec_lut[g] = _mm256_sub_ps(_mm256_setzero_ps(), vec_lut[15 - g]);
        biases += _mm256_addv_ps(vec_lut[0]);

        for (int g = 0; g < 16; ++g) vec_lut[g] = _mm256_mul_ps(vec_lut[g], _mm256_set1_ps(t_scales));

        __m256i vec_qlut[4];
        const __m256i shuf = _mm256_setr_epi8(0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15,0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15);
        for (int g = 0; g < 4; g += 1) {
            __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 2], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 3], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            i0 = _mm256_packs_epi32(i0, i1); i2 = _mm256_packs_epi32(i2, i3); i0 = _mm256_packs_epi16(i0, i2);
            vec_qlut[g] = _mm256_shuffle_epi8(i0, shuf);
        }
        int32_t* qlut_i32 = reinterpret_cast<int32_t*>(qlut);
        for (int g = 0; g < 4; ++g) { qlut_i32[k * 32 + 0 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 0); }
        for (int g = 0; g < 4; ++g) { qlut_i32[k * 32 + 1 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 1); }
        for (int g = 0; g < 4; ++g) { qlut_i32[k * 32 + 2 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 2); }
        for (int g = 0; g < 4; ++g) { qlut_i32[k * 32 + 3 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 3); }
        for (int g = 0; g < 4; ++g) { qlut_i32[k * 32 + 4 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 4); }
        for (int g = 0; g < 4; ++g) { qlut_i32[k * 32 + 5 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 5); }
        for (int g = 0; g < 4; ++g) { qlut_i32[k * 32 + 6 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 6); }
        for (int g = 0; g < 4; ++g) { qlut_i32[k * 32 + 7 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 7); }
    }
    *lut_scales = scales; *lut_biases = biases;
    return 0;
}

template <int Bits, int ActK = 32>
inline int32_t tbl_update_avx2(int32_t m, int32_t k_groups, float_type* c, int8_t* lut, uint8_t* a, float_type* scales, float_type* lut_scales, float_type* lut_biases) {
    const __m128i vec_mask = _mm_set1_epi8(0x0f);

    __m128i* vec_lut = (__m128i*)_mm_malloc(k_groups * sizeof(__m128i), 32);
    for (int k = 0; k < k_groups; k++) {
        vec_lut[k] = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 16));
    }

    SignedWideningAdder<ActK> adder;
    for (int i = 0; i < m / 2; i += 16) {
        __m256 vec_c0 = _mm256_setzero_ps();
        __m256 vec_c1 = _mm256_setzero_ps();
        __m256 vec_c2 = _mm256_setzero_ps();
        __m256 vec_c3 = _mm256_setzero_ps();

        for (int kk = 0; kk < k_groups; kk += ActK) {
            for (int k = 0; k < ActK; k++) {
                __m128i vec_as = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i * k_groups + (kk + k) * 16));
                __m128i vec_a_bot = _mm_and_si128(vec_as, vec_mask);
                __m128i vec_a_top = _mm_and_si128(_mm_srli_epi16(vec_as, 4), vec_mask);
                __m256i vec_lut_ = _mm256_set_m128i(vec_lut[kk + k], vec_lut[kk + k]);
                __m256i vec_a = _mm256_set_m128i(vec_a_top, vec_a_bot);
                __m256i vec_v = _mm256_shuffle_epi8(vec_lut_, vec_a);
                adder.push(vec_v, k);
            }
            __m256 vec_v_low_low = _mm256_cvtepi32_ps(extract_low_epi16_epi32(adder.get_low()));
            __m256 vec_v_low_high = _mm256_cvtepi32_ps(extract_high_epi16_epi32(adder.get_low()));
            __m256 vec_v_high_low = _mm256_cvtepi32_ps(extract_low_epi16_epi32(adder.get_high()));
            __m256 vec_v_high_high = _mm256_cvtepi32_ps(extract_high_epi16_epi32(adder.get_high()));

            int group_idx = kk / ActK;
            float lut_s = lut_scales[group_idx];
            float lut_b = lut_biases[group_idx];

            #define lut_fma(vs, ib) ((ib) % Bits) ? (_mm256_mul_ps((vs), _mm256_set1_ps(lut_s))) : (_mm256_fmadd_ps((vs), _mm256_set1_ps(lut_s), _mm256_set1_ps(lut_b)))
            if (kk == 0) {
                vec_c0 = lut_fma(vec_v_low_low, (i / 4 )); vec_c1 = lut_fma(vec_v_low_high, (i / 4 + 1));
                vec_c2 = lut_fma(vec_v_high_low, (i / 4 + 2)); vec_c3 = lut_fma(vec_v_high_high, (i / 4 + 3));
            } else {
                vec_c0 = _mm256_add_ps(vec_c0, lut_fma(vec_v_low_low, (i / 4 ))); vec_c1 = _mm256_add_ps(vec_c1, lut_fma(vec_v_low_high, (i / 4 + 1)));
                vec_c2 = _mm256_add_ps(vec_c2, lut_fma(vec_v_high_low, (i / 4 + 2))); vec_c3 = _mm256_add_ps(vec_c3, lut_fma(vec_v_high_high, (i / 4 + 3)));
            }
            #undef lut_fma
        }

        __m256 vec_s0 = _mm256_loadu_ps(scales + ((i / 4 ) / Bits) * 8);
        __m256 vec_s1 = _mm256_loadu_ps(scales + ((i / 4 + 1) / Bits) * 8);
        __m256 vec_s2 = _mm256_loadu_ps(scales + ((i / 4 + 2) / Bits) * 8);
        __m256 vec_s3 = _mm256_loadu_ps(scales + ((i / 4 + 3) / Bits) * 8);

        _mm256_storeu_ps(c + i * 2,      _mm256_fmadd_ps(vec_c0, vec_s0, _mm256_loadu_ps(c + i * 2)));
        _mm256_storeu_ps(c + i * 2 + 8,  _mm256_fmadd_ps(vec_c1, vec_s1, _mm256_loadu_ps(c + i * 2 + 8)));
        _mm256_storeu_ps(c + i * 2 + 16, _mm256_fmadd_ps(vec_c2, vec_s2, _mm256_loadu_ps(c + i * 2 + 16)));
        _mm256_storeu_ps(c + i * 2 + 24, _mm256_fmadd_ps(vec_c3, vec_s3, _mm256_loadu_ps(c + i * 2 + 24)));
    }
    _mm_free(vec_lut);
    return 0;
}


int main() {
    std::cout << "Initializing T-MAC AVX2 Benchmark (N=1 GEMV)..." << std::endl;
    std::vector<int> sizes = {256, 1024, 2048, 4096, 8192};
    const int Bits = 4;
    const int ActK = 32;

    std::cout << "------------------------------------------------------\n";
    std::cout << std::left << std::setw(15) << "Matrix Size"
              << std::setw(15) << "Latency (ms)"
              << "Performance (GFLOPS)" << std::endl;
    std::cout << "------------------------------------------------------\n";

    for (int size : sizes) {
        int m = size;
        int k = size;

        int k_groups = k / 4;
        int num_groups = k_groups / ActK;

        float* activations = (float*)_mm_malloc(k * sizeof(float), 32);
        int8_t* qlut = (int8_t*)_mm_malloc(k_groups * ActK * 16 * sizeof(int8_t), 32);
        uint8_t* weights = (uint8_t*)_mm_malloc(m * k_groups * 16 * sizeof(uint8_t), 32);
        float* out_c = (float*)_mm_malloc(m * sizeof(float), 32);
        float* scales = (float*)_mm_malloc((m / Bits) * 16 * sizeof(float), 32);

        float* lut_scales = (float*)_mm_malloc(num_groups * sizeof(float), 32);
        float* lut_biases = (float*)_mm_malloc(num_groups * sizeof(float), 32);

        std::memset(out_c, 0, m * sizeof(float));
        std::memset(weights, 0x11, m * k_groups * 16 * sizeof(uint8_t));
        for(int i = 0; i < k; ++i) activations[i] = 1.0f;
        for(int i = 0; i < (m / Bits) * 16; ++i) scales[i] = 1.0f;

        int iterations = 100;
        auto start = std::chrono::high_resolution_clock::now();

        const int group_span = ActK * 4;  

        for (int i = 0; i < iterations; ++i) {
            for (int g = 0; g < num_groups; ++g) {
                lut_scales[g] = 0.0f;

                for (int sub = 0; sub < group_span / 32; ++sub) {
                    partial_max_g4_int8_k8(&lut_scales[g], activations + g * group_span + sub * 32);
                }

                lut_ctor_g4_int8_avx2(
                    group_span,                       
                    qlut + g * (ActK * 16),            
                    activations + g * group_span,
                    &lut_scales[g],
                    &lut_biases[g]
                );
            }

            tbl_update_avx2<Bits, ActK>(m, k_groups, out_c, qlut, weights, scales, lut_scales, lut_biases);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        double avg_time_ms = duration.count() / iterations;

        double ops = 2.0 * m * k;
        double gflops = (ops / (avg_time_ms / 1000.0)) / 1e9;

        std::cout << std::left << size << "x" << size << "x1    "
                  << std::setw(15) << avg_time_ms
                  << gflops << " GFLOPS" << std::endl;

        bool has_nan_or_inf = false;
        double checksum = 0.0;
        for (int i = 0; i < m; ++i) {
            if (std::isnan(out_c[i]) || std::isinf(out_c[i])) {
                has_nan_or_inf = true;
                break;
            }
            checksum += out_c[i];
        }
        std::cout << "  checksum sum(out_c) = " << checksum;
        if (has_nan_or_inf) {
            std::cout << "  [NaN/Inf detected - RESULTS INVALID]";
        }
        std::cout << std::endl;

        _mm_free(activations); _mm_free(qlut); _mm_free(weights);
        _mm_free(out_c); _mm_free(scales);
        _mm_free(lut_scales); _mm_free(lut_biases);
    }
    return 0;
}