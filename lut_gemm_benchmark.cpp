#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstdint>
#include <cmath>

// ============================================================================
// LUT-GEMM 核心邏輯 (忠實移植自 mv_fp16_bias.hpp)
// ============================================================================
void lut_gemm_cpu(int M, int K, int NUM_BITS, uint32_t* W, float* alpha, float* q_bias, float* input, float* output, float* lut) {

    int num_chunks = K / 8; // 每 8 個 Activation 為一組 (mu = 8)

    // ------------------------------------------------------------------------
    // Step 1: LUT Construction (動態規劃快速建表，對應 GPU 的 shared memory LUT)
    // ------------------------------------------------------------------------
    for (int kt = 0; kt < num_chunks; ++kt) {
        float* chunk_input = input + kt * 8;
        float* chunk_lut = lut + kt * 256;

        // 算出 index 00000000 的基準值 (全部乘上 -1)
        float base = 0;
        for (int i = 0; i < 8; ++i) {
            base += (-1.0f) * chunk_input[i];
        }
        chunk_lut[0] = base;   // [修正] 原本誤寫成 chunk_lut = base（型別錯誤，無法編譯）

        // 使用動態規劃依序生成 256 種組合 (與官方 GPU 邏輯完全一致)
        for (int s = 0; s < 8; ++s) {
            float iValue = 2.0f * chunk_input[s];
            int step = 1 << s;
            for (int i = 0; i < step; ++i) {
                chunk_lut[step + i] = chunk_lut[i] + iValue;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Step 2: Table Lookup & Accumulate (對應 nqmv_bias 的運算迴圈)
    // ------------------------------------------------------------------------
    for (int m = 0; m < M; ++m) {
        float reg_o = 0.0f;

        // Bias 補償計算
        if (q_bias != nullptr) {
            float reg_a = q_bias[m];
            float reg_t_o = 0.0f;
            for (int kt = 0; kt < num_chunks; ++kt) {
                reg_t_o += lut[kt * 256 + 255]; // Index 255 代表全為 +1 的組合
            }
            reg_o += reg_a * reg_t_o;
        }

        // Bit-Serial 迴圈：依序處理 NUM_BITS 個位元面
        for (int b = 0; b < NUM_BITS; ++b) {
            float reg_t_o = 0.0f;
            float scale = alpha[b * M + m]; // 取得該位元面的縮放因子

            // 遍歷 K 維度 (每次處理 32 個權重 = 4 個 8-bit chunk)
            for (int kt32 = 0; kt32 < K / 32; ++kt32) {
                // 讀取打包的 32-bit 權重 (bW 陣列)
                uint32_t reg_w = W[kt32 * NUM_BITS * M + b * M + m];

                // 將 32-bit 拆解成 4 個 8-bit 的索引進行查表
                int reg_w0 = (reg_w >>  0) & 255;
                reg_t_o += lut[(kt32 * 4 + 0) * 256 + reg_w0];

                int reg_w1 = (reg_w >>  8) & 255;
                reg_t_o += lut[(kt32 * 4 + 1) * 256 + reg_w1];

                int reg_w2 = (reg_w >> 16) & 255;
                reg_t_o += lut[(kt32 * 4 + 2) * 256 + reg_w2];

                int reg_w3 = (reg_w >> 24) & 255;
                reg_t_o += lut[(kt32 * 4 + 3) * 256 + reg_w3];
            }
            // 乘上 Scale 並累加到最終結果
            reg_o += scale * reg_t_o;
        }
        output[m] = reg_o;
    }
}

// ============================================================================
// Benchmark 主程式
// ============================================================================
int main() {
    std::cout << "初始化 LUT-GEMM (CPU 移植版) Benchmark (N=1 GEMV)..." << std::endl;
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

        // 配置記憶體 (匹配 LUT-GEMM 的資料結構)
        uint32_t* W = new uint32_t[(K / 32) * NUM_BITS * M];
        float* alpha = new float[NUM_BITS * M];
        float* q_bias = new float[M];
        float* input = new float[K];
        float* output = new float[M];

        // 查表用記憶體: (K / 8) 個 chunk，每個 chunk 256 格
        float* lut = new float[(K / 8) * 256];

        // 假資料初始化
        std::fill_n(W, (K / 32) * NUM_BITS * M, 0x11111111u);
        std::fill_n(alpha, NUM_BITS * M, 1.0f);
        std::fill_n(q_bias, M, 0.0f);
        std::fill_n(input, K, 1.0f);
        std::fill_n(output, M, 0.0f);

        // Warm up
        lut_gemm_cpu(M, K, NUM_BITS, W, alpha, q_bias, input, output, lut);

        // CPU 純標量(Scalar)查表極慢，測試疊代次數降低至 10 次
        int iterations = 10;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            lut_gemm_cpu(M, K, NUM_BITS, W, alpha, q_bias, input, output, lut);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        double avg_time_ms = duration.count() / iterations;
        double ops = 2.0 * M * K; // N=1 GEMV 的理論浮點運算量
        double gflops = (ops / (avg_time_ms / 1000.0)) / 1e9;

        std::cout << std::left << size << "x" << size << "x1    "
                  << std::setw(15) << avg_time_ms
                  << gflops << " GFLOPS" << std::endl;

        // ------------------------------------------------------------------
        // 正確性驗證：解析解為 output[m] = -2 * K（推導見上方說明）
        // ------------------------------------------------------------------
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
                  && std::fabs(output[0] - expected) < std::fabs(expected) * 0.01f; // 1% 容忍度

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