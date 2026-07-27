#include <iostream>
#include <vector>
#include <chrono>
#include <mkl.h>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include <random>
#include <cmath>
#include "common/benchmark.h"

// 純量三層迴圈 ground truth：用真正的 signed A 值計算，作為驗證基準
void naive_int8_gemv(int M, int K, const int8_t* A_signed, const uint8_t* B, int32_t* C_gt) {
    for (int m = 0; m < M; ++m) {
        int32_t acc = 0;
        for (int k = 0; k < K; ++k) {
            acc += static_cast<int32_t>(A_signed[m * K + k]) * static_cast<int32_t>(B[k]);
        }
        C_gt[m] = acc;
    }
}

int main() {
    int num_threads = 1;
    mkl_set_num_threads(num_threads);

    std::cout << "Initializing Intel MKL INT8 Benchmark (Random Input, Offset-Corrected)..." << std::endl;
    std::cout << "Number of threads used: " << mkl_get_max_threads() << std::endl;

    std::vector<int> sizes = {256, 1024, 2048, 4096, 8192};
    const int n = 1;
    int iterations = 1000;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> a_dist(-16, 16);
    std::uniform_int_distribution<int> b_dist(0, 16);

    print_benchmark_header();

    for (int size : sizes) {
        int m = size, k = size;

        float alpha = 1.0f;
        float beta  = 0.0f;
        int8_t  bo = 0;
        int32_t co = 0;

        // A_signed：人類看得懂的 signed 資料，做為 naive 對照與最終驗證依據
        std::vector<int8_t> A_signed(m * k);
        for (auto& v : A_signed) v = static_cast<int8_t>(a_dist(rng));

        // A_shifted：實際餵給 MKL 的資料，每個元素 = A_signed + 128
        // 已驗證：MKL 的 cblas_gemm_s8u8s32 會把第一個矩陣參數當成 unsigned 讀取，
        // 因此必須先位移 +128 存成 uint8，再用 ao=-128 補償回正確的 signed 語意
        AlignedBuffer<uint8_t> A_shifted(static_cast<uint8_t*>(mkl_malloc(m * k * sizeof(uint8_t), 64)), mkl_free);
        AlignedBuffer<uint8_t> B(static_cast<uint8_t*>(mkl_malloc(k * n * sizeof(uint8_t), 64)), mkl_free);
        AlignedBuffer<int32_t> C(static_cast<int32_t*>(mkl_malloc(m * n * sizeof(int32_t), 64)), mkl_free);

        if (A_shifted == NULL || B == NULL || C == NULL) {
            std::cerr << "Memory allocation failed!" << std::endl;
            return 1;
        }

        for (int i = 0; i < m * k; ++i)
            A_shifted[i] = static_cast<uint8_t>(static_cast<int>(A_signed[i]) + 128);
        for (int i = 0; i < k * n; ++i) B[i] = static_cast<uint8_t>(b_dist(rng));
        //std::fill_n(static_cast<int32_t*>(C), m * n, 0);

        int8_t ao = -128; // 補償 A_shifted 的 +128 位移，還原成正確的 signed 內積

        // Warm up
        cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasNoTrans, CblasFixOffset,
                           m, n, k, alpha,
                           reinterpret_cast<const int8_t*>(static_cast<uint8_t*>(A_shifted)), k, ao,
                           B, n, bo,
                           beta, C, n, &co);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            std::fill_n(static_cast<int32_t*>(C), m * n, 0);
            cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasNoTrans, CblasFixOffset,
                               m, n, k, alpha,
                               reinterpret_cast<const int8_t*>(static_cast<uint8_t*>(A_shifted)), k, ao,
                               B, n, bo,
                               beta, C, n, &co);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double avg_time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        double gflops = compute_gflops(m, k, avg_time_ms);
        print_benchmark_row(size, avg_time_ms, gflops);

        // ---- 印出 input / output（用原始 signed 值顯示，符合人類直覺）----
        std::cout << "  input(A)[0..7] = ";
        for (int i = 0; i < 8 && i < k; ++i) std::cout << (int)A_signed[i] << " ";
        std::cout << std::endl;
        std::cout << "  input(B)[0..7] = ";
        for (int i = 0; i < 8 && i < k; ++i) std::cout << (int)static_cast<uint8_t>(B[i]) << " ";
        std::cout << std::endl;
        std::cout << "  output[0..7]   = ";
        for (int i = 0; i < 8 && i < m; ++i) std::cout << C[i] << " ";
        std::cout << std::endl;

        // ---- 驗證：與純量三層迴圈比對 ----
        std::vector<int32_t> C_gt(m);
        std::vector<uint8_t> B_copy(k);
        for (int i = 0; i < k; ++i) B_copy[i] = B[i];
        naive_int8_gemv(m, k, A_signed.data(), B_copy.data(), C_gt.data());

        int mismatches = 0;
        int64_t max_abs_err = 0;
        for (int i = 0; i < m; ++i) {
            int64_t err = std::llabs((int64_t)C[i] - (int64_t)C_gt[i]);
            max_abs_err = std::max(max_abs_err, err);
            if (err != 0) ++mismatches;
        }

        std::cout << "  [Verify] mismatches=" << mismatches << "/" << m
                   << "  max_abs_err=" << max_abs_err
                   << "  (" << (mismatches == 0 ? "OK - exact match" : "MISMATCH") << ")" << std::endl;
    }

    return 0;
}