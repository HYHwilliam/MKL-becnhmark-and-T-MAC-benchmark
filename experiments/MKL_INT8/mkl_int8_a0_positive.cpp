#include <iostream>
#include <vector>
#include <chrono>
#include <mkl.h>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include "benchmark.h"

int main() {
    int num_threads = 1;
    mkl_set_num_threads(num_threads);

    std::cout << "Initializing Intel MKL INT8 Benchmark..." << std::endl;
    std::cout << "Number of threads used: " << mkl_get_max_threads() << std::endl;

    std::vector<int> sizes = {256, 1024, 2048, 4096, 8192};

    const int n = 1;
    int iterations = 1000;

    print_benchmark_header();

    for (int size : sizes) {
        int m = size, k = size;

        float alpha = 1.0f;
        float beta  = 0.0f;

        int8_t  ao = 0;
        int8_t  bo = 0;
        int32_t co = 0;

        AlignedBuffer<int8_t>  A(static_cast<int8_t*>(mkl_malloc(m * k * sizeof(int8_t), 64)), mkl_free);
        AlignedBuffer<uint8_t> B(static_cast<uint8_t*>(mkl_malloc(k * n * sizeof(uint8_t), 64)), mkl_free);
        AlignedBuffer<int32_t> C(static_cast<int32_t*>(mkl_malloc(m * n * sizeof(int32_t), 64)), mkl_free);

        if (A == NULL || B == NULL || C == NULL) {
            std::cerr << "Memory allocation failed!" << std::endl;
            return 1;
        }

        std::fill_n(static_cast<int8_t*>(A), m * k, static_cast<int8_t>(1));
        std::fill_n(static_cast<uint8_t*>(B), k * n, static_cast<uint8_t>(2));
        std::fill_n(static_cast<int32_t*>(C), m * n, 0);

        // Warm up
        cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasNoTrans, CblasFixOffset,
                           m, n, k, alpha,
                           A, k, ao,
                           B, n, bo,
                           beta,
                           C, n, &co);

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasNoTrans, CblasFixOffset,
                               m, n, k, alpha,
                               A, k, ao,
                               B, n, bo,
                               beta,
                               C, n, &co);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        double avg_time_ms = duration.count() / iterations;

        double gflops = compute_gflops(m, k, avg_time_ms);
        print_benchmark_row(size, avg_time_ms, gflops);

        // // ---- input / output ----
        // std::cout << "  input(A)[0..7] = ";
        // for (int i = 0; i < 8 && i < k; ++i) std::cout << (int)A[i] << " ";
        // std::cout << std::endl;

        // std::cout << "  input(B)[0..7] = ";
        // for (int i = 0; i < 8 && i < k; ++i) std::cout << (int)B[i] << " ";
        // std::cout << std::endl;

        // std::cout << "  output[0..7]   = ";
        // for (int i = 0; i < 8 && i < m; ++i) std::cout << C[i] << " ";
        // std::cout << std::endl;

        int32_t sample = C[0];
        int32_t expected = 2 * k;
        bool ok = (sample == expected);
        std::cout << "  checksum C[0] = " << sample
                   << " (expected " << expected << ") "
                   << (ok ? "OK" : "MISMATCH - RESULTS SUSPECT") << std::endl;
    }

    return 0;
}