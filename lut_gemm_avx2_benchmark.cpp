#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstdint>
#include <cmath>

// ============================================================================
// LUT 建表（維持純量，成本相對查表迴圈可忽略，邏輯與官方一致）
// ============================================================================
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

// ============================================================================
// AVX2 查表與累加：一次處理 8 個 m（對應 T-MAC 的並行精神，但用 gather 取代 shuffle）
// ============================================================================
void lut_gemm_cpu_avx2(int M, int K, int NUM_BITS, uint32_t* W, float* alpha,
                       float* q_bias, float* input, float* output, float* lut) {

    build_lut(K, input, lut);

    int num_chunks = K / 8;
    int k32 = K / 32;

    // bias 項的累加值與 m 無關，先算一次（對應官方 lut[..][255] 加總）
    float bias_acc = 0.0f;
    for (int kt = 0; kt < num_chunks; ++kt) bias_acc += lut[kt * 256 + 255];
    __m256 vec_bias_acc = _mm256_set1_ps(bias_acc);

    const __m256i mask255 = _mm256_set1_epi32(255);

    for (int m = 0; m < M; m += 8) {
        __m256 reg_o = _mm256_setzero_ps();

        // --- bias 補償：q_bias[m..m+8) * bias_acc ---
        if (q_bias != nullptr) {
            __m256 vec_qbias = _mm256_loadu_ps(q_bias + m);
            reg_o = _mm256_fmadd_ps(vec_qbias, vec_bias_acc, reg_o);
        }

        // --- bit-serial 迴圈 ---
        for (int b = 0; b < NUM_BITS; ++b) {
            __m256 reg_t_o = _mm256_setzero_ps();

            for (int kt32 = 0; kt32 < k32; ++kt32) {
                // W 佈局 [kt32][NUM_BITS][M]，m 是最快變化維度 -> 連續讀取 8 個 uint32
                __m256i reg_w = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&W[kt32 * NUM_BITS * M + b * M + m]));

                // 拆解 4 個 byte，每個 byte 各自 gather 查表（對應官方 reg_w0~reg_w3）
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

// ============================================================================
// Benchmark 主程式（與 MKL / T-MAC 版本格式一致）
// ============================================================================
int main() {
    std::cout << "初始化 LUT-GEMM AVX2 Benchmark (N=1 GEMV)..." << std::endl;
    std::vector<int> sizes = {256, 1024, 2048, 4096, 8192};
    const int NUM_BITS = 4;

    std::cout << "------------------------------------------------------\n";
    std::cout << std::left << std::setw(15) << "Matrix Size"
              << std::setw(15) << "Latency (ms)"
              << "Performance (GFLOPS)" << std::endl;
    std::cout << "------------------------------------------------------\n";

    for (int size : sizes) {
        int M = size;
        int K = size;

        uint32_t* W = new uint32_t[(K / 32) * NUM_BITS * M];
        float* alpha = new float[NUM_BITS * M];
        float* q_bias = new float[M];
        float* input = new float[K];
        float* output = new float[M];
        float* lut = new float[(K / 8) * 256];

        std::fill_n(W, (K / 32) * NUM_BITS * M, 0x11111111u);
        std::fill_n(alpha, NUM_BITS * M, 1.0f);
        std::fill_n(q_bias, M, 0.0f);
        std::fill_n(input, K, 1.0f);
        std::fill_n(output, M, 0.0f);

        // Warm up
        lut_gemm_cpu_avx2(M, K, NUM_BITS, W, alpha, q_bias, input, output, lut);

        int iterations = 100; // AVX2 版速度接近 T-MAC，沿用相同疊代次數
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            lut_gemm_cpu_avx2(M, K, NUM_BITS, W, alpha, q_bias, input, output, lut);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        double avg_time_ms = duration.count() / iterations;
        double ops = 2.0 * M * K;
        double gflops = (ops / (avg_time_ms / 1000.0)) / 1e9;

        std::cout << std::left << size << "x" << size << "x1    "
                  << std::setw(15) << avg_time_ms
                  << gflops << " GFLOPS" << std::endl;

        // 正確性驗證：解析解為 output[m] = -2 * K（與純量版相同推導）
        bool has_nan_or_inf = false;
        double checksum = 0.0;
        for (int i = 0; i < M; ++i) {
            if (std::isnan(output[i]) || std::isinf(output[i])) {
                has_nan_or_inf = true;
                break;
            }
            checksum += output[i];
        }
        float expected = -2.0f * K;
        bool ok = !has_nan_or_inf
                  && std::fabs(output[0] - expected) < std::fabs(expected) * 0.01f;

        std::cout << "  checksum sum(output) = " << checksum
                   << "  |  output[0] = " << output[0]
                   << " (expected " << expected << ") "
                   << (ok ? "OK" : "MISMATCH - RESULTS SUSPECT");
        if (has_nan_or_inf) std::cout << "  [NaN/Inf detected]";
        std::cout << std::endl;

        delete[] W; delete[] alpha; delete[] q_bias; delete[] input; delete[] output; delete[] lut;
    }
    return 0;
}