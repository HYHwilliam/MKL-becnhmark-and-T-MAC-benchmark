#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <omp.h>

// ==== LUT 建表：與單執行緒版完全相同 ====
void build_lut(int K, float* input, float* lut) {
    int num_chunks = K / 8;
    for (int kt = 0; kt < num_chunks; ++kt) {
        float* chunk_input = input + kt * 8;
        float* chunk_lut = lut + kt * 256;

        float base = 0;
        for (int i = 0; i < 8; ++i) base += (-1.0f) * chunk_input[i];
        chunk_lut[0] = base;

        for (int s = 0; s < 8; ++s) {
            float iValue = 2.0f * chunk_input[s];
            int step = 1 << s;
            for (int i = 0; i < step; ++i) {
                chunk_lut[step + i] = chunk_lut[i] + iValue;
            }
        }
    }
}

// ==== 單執行緒版：正確性對照基準（完全不變）====
void lut_gemm_cpu_avx2(int M, int K, int NUM_BITS, uint32_t* W, float* alpha,
                       float* q_bias, float* input, float* output, float* lut) {

    build_lut(K, input, lut);

    int num_chunks = K / 8;
    int k32 = K / 32;

    float bias_acc = 0.0f;
    for (int kt = 0; kt < num_chunks; ++kt) bias_acc += lut[kt * 256 + 255];
    __m256 vec_bias_acc = _mm256_set1_ps(bias_acc);

    const __m256i mask255 = _mm256_set1_epi32(255);

    for (int m = 0; m < M; m += 8) {
        __m256 reg_o = _mm256_setzero_ps();

        if (q_bias != nullptr) {
            __m256 vec_qbias = _mm256_loadu_ps(q_bias + m);
            reg_o = _mm256_fmadd_ps(vec_qbias, vec_bias_acc, reg_o);
        }

        for (int b = 0; b < NUM_BITS; ++b) {
            __m256 reg_t_o = _mm256_setzero_ps();

            for (int kt32 = 0; kt32 < k32; ++kt32) {
                __m256i reg_w = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&W[kt32 * NUM_BITS * M + b * M + m]));

                for (int j = 0; j < 4; ++j) {
                    __m256i idx = _mm256_and_si256(
                        _mm256_srli_epi32(reg_w, 8 * j), mask255);
                    __m256 vals = _mm256_i32gather_ps(
                        lut + (kt32 * 4 + j) * 256, idx, 4);
                    reg_t_o = _mm256_add_ps(reg_t_o, vals);
                }
            }

            __m256 vec_alpha = _mm256_loadu_ps(alpha + b * M + m);
            reg_o = _mm256_fmadd_ps(vec_alpha, reg_t_o, reg_o);
        }

        _mm256_storeu_ps(output + m, reg_o);
    }
}

// ==== OpenMP 多執行緒版：m 維度平行化，W/alpha/q_bias/lut 唯讀共享 ====
// 用 mi 當迴圈變數（m = mi*8）以符合 omp for 需要的規範迴圈形式
void lut_gemm_avx2_omp(int M, int K, int NUM_BITS, uint32_t* W, float* alpha,
                       float* q_bias, float* output, float* lut, float bias_acc) {

    int k32 = K / 32;
    int m_groups = M / 8;
    __m256 vec_bias_acc = _mm256_set1_ps(bias_acc);
    const __m256i mask255 = _mm256_set1_epi32(255);

    #pragma omp parallel for schedule(static)
    for (int mi = 0; mi < m_groups; ++mi) {
        int m = mi * 8;
        __m256 reg_o = _mm256_setzero_ps();

        if (q_bias != nullptr) {
            __m256 vec_qbias = _mm256_loadu_ps(q_bias + m);
            reg_o = _mm256_fmadd_ps(vec_qbias, vec_bias_acc, reg_o);
        }

        for (int b = 0; b < NUM_BITS; ++b) {
            __m256 reg_t_o = _mm256_setzero_ps();

            for (int kt32 = 0; kt32 < k32; ++kt32) {
                __m256i reg_w = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&W[kt32 * NUM_BITS * M + b * M + m]));

                for (int j = 0; j < 4; ++j) {
                    __m256i idx = _mm256_and_si256(
                        _mm256_srli_epi32(reg_w, 8 * j), mask255);
                    __m256 vals = _mm256_i32gather_ps(
                        lut + (kt32 * 4 + j) * 256, idx, 4);
                    reg_t_o = _mm256_add_ps(reg_t_o, vals);
                }
            }

            __m256 vec_alpha = _mm256_loadu_ps(alpha + b * M + m);
            reg_o = _mm256_fmadd_ps(vec_alpha, reg_t_o, reg_o);
        }

        _mm256_storeu_ps(output + m, reg_o);
    }
}

int main() {
    std::cout << "LUT-GEMM AVX2 OpenMP 多執行緒 Benchmark\n";
    std::cout << "------------------------------------------------------\n";

    std::vector<int> sizes = {1024, 2048, 4096, 8192};
    std::vector<int> thread_counts = {1, 2, 4, 8};
    const int NUM_BITS = 4;

    for (int size : sizes) {
        int M = size, K = size;

        uint32_t* W = new uint32_t[(K / 32) * NUM_BITS * M];
        float* alpha = new float[NUM_BITS * M];
        float* q_bias = new float[M];
        float* input = new float[K];
        float* output = new float[M];
        float* output_single = new float[M];
        float* lut = new float[(K / 8) * 256];

        std::fill_n(W, (K / 32) * NUM_BITS * M, 0x11111111u);
        std::fill_n(alpha, NUM_BITS * M, 1.0f);
        std::fill_n(q_bias, M, 0.0f);
        std::fill_n(input, K, 1.0f);
        std::fill_n(output, M, 0.0f);
        std::fill_n(output_single, M, 0.0f);

        // 單執行緒基準（同時建好 lut，供多執行緒版本共用讀取）
        lut_gemm_cpu_avx2(M, K, NUM_BITS, W, alpha, q_bias, input, output_single, lut);

        int num_chunks = K / 8;
        float bias_acc = 0.0f;
        for (int kt = 0; kt < num_chunks; ++kt) bias_acc += lut[kt * 256 + 255];

        std::cout << "\n=== Matrix Size " << M << "x" << K << " ===\n";

        for (int nt : thread_counts) {
            omp_set_num_threads(nt);

            for (int warmup = 0; warmup < 5; ++warmup) {
                std::fill_n(output, M, 0.0f);
                lut_gemm_avx2_omp(M, K, NUM_BITS, W, alpha, q_bias, output, lut, bias_acc);
            }

            int iterations = 20;
            auto start_time = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iterations; ++it) {
                std::fill_n(output, M, 0.0f);
                lut_gemm_avx2_omp(M, K, NUM_BITS, W, alpha, q_bias, output, lut, bias_acc);
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            double avg_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count() / iterations;
            double gflops = (2.0 * M * K) / (avg_ms / 1000.0) / 1e9;

            int mismatches = 0;
            double max_err = 0.0;
            bool has_nan_or_inf = false;
            for (int i = 0; i < M; ++i) {
                if (std::isnan(output[i]) || std::isinf(output[i])) { has_nan_or_inf = true; break; }
                double err = std::fabs(output[i] - output_single[i]);
                max_err = std::max(max_err, err);
                if (err > 1e-4 * std::fabs(output_single[i]) + 1e-4) ++mismatches;
            }

            float expected = -2.0f * K;
            bool analytical_ok = !has_nan_or_inf && std::fabs(output[0] - expected) < std::fabs(expected) * 0.01f;

            std::cout << "  threads=" << nt << "  latency=" << avg_ms << "ms  "
                       << gflops << " GFLOPS  "
                       << (mismatches == 0 && !has_nan_or_inf ? "OK" : "MISMATCH x" + std::to_string(mismatches))
                       << "  max_err=" << max_err
                       << "  analytical=" << (analytical_ok ? "OK" : "MISMATCH")
                       << (has_nan_or_inf ? "  [NaN/Inf detected]" : "") << "\n";
        }

        delete[] W; delete[] alpha; delete[] q_bias; delete[] input;
        delete[] output; delete[] output_single; delete[] lut;
    }
    return 0;
}
