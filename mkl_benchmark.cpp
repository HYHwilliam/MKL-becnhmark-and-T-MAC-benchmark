#include <iostream>
#include <vector>
#include <chrono>
#include <mkl.h>
#include <immintrin.h>   // [FIX 1] needed for F16C conversion intrinsics
#include <iomanip>
#include <cmath>          // [FIX 4] for std::isnan/isinf sanity check

// [FIX 1] MKL_F16 is a raw 16-bit storage type, NOT an arithmetic type.
// Assigning an int literal (A[i] = 1) does not produce the half-precision
// value 1.0f -- it stores the raw bit pattern 0x0001, which is a tiny
// subnormal (~5.96e-8), not 1.0. We need a real float -> half conversion.
static inline MKL_F16 float_to_half(float f) {
    __m128 v = _mm_set_ss(f);
    __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
    return (MKL_F16)_mm_extract_epi16(h, 0);
}

// [FIX 4] Convert a computed half-precision result back to float so we can
// sanity-check it (NaN/Inf check + printed checksum), instead of trusting
// the run blindly.
static inline float half_to_float(MKL_F16 h) {
    __m128i hv = _mm_cvtsi32_si128((uint16_t)h);
    __m128 fv = _mm_cvtph_ps(hv);
    return _mm_cvtss_f32(fv);
}

int main() {
    int num_threads = 1;
    mkl_set_num_threads(num_threads);

    std::cout << "初始化 Intel MKL Benchmark..." << std::endl;
    std::cout << "使用執行緒數量: " << mkl_get_max_threads() << std::endl;

    std::vector<int> sizes = {256, 1024, 2048, 4096, 8192};

    // [FIX 2] Match T-MAC's benchmark shape. T-MAC's kernel is a GEMV
    // (N=1) -- that's the actual operation being measured (decode-time,
    // one token at a time). Running MKL as a full size x size x size GEMM
    // measures a completely different amount of work (up to `size` times
    // more FLOPs for the same nominal "8192x8192" row) and makes the two
    // benchmarks' throughput numbers incomparable. Force N=1 here so both
    // benchmarks measure the same M x K x 1 problem.
    const int n = 1;

    // [FIX 3] Match iteration count to the T-MAC benchmark (100) so both
    // benchmarks average over the same number of samples with similar
    // noise characteristics. (Was 10.)
    int iterations = 100;

    std::cout << "------------------------------------------------------\n";
    std::cout << std::left << std::setw(15) << "Matrix Size"
              << std::setw(15) << "Latency (ms)"
              << "Performance (GFLOPS)" << std::endl;   // [FIX 2] GFLOPS not TFLOPS,
                                                          // since GEMV at these sizes
                                                          // won't reach TFLOPS scale --
                                                          // keeps units comparable to
                                                          // the T-MAC report too.
    std::cout << "------------------------------------------------------\n";

    for (int size : sizes) {
        int m = size, k = size;
        // n is fixed at 1, see [FIX 2] above.

        MKL_F16 alpha = float_to_half(1.0f);   // 原本是 float alpha = 1.0f;
        MKL_F16 beta  = float_to_half(0.0f);   // 原本是 float beta  = 0.0f;

        MKL_F16* A = (MKL_F16*)mkl_malloc(m * k * sizeof(MKL_F16), 64);
        MKL_F16* B = (MKL_F16*)mkl_malloc(k * n * sizeof(MKL_F16), 64);
        MKL_F16* C = (MKL_F16*)mkl_malloc(m * n * sizeof(MKL_F16), 64);

        if (A == NULL || B == NULL || C == NULL) {
            std::cerr << "記憶體配置失敗！" << std::endl;
            return 1;
        }

        // [FIX 1] Use real float->half conversion instead of raw int assignment.
        for (int i = 0; i < m * k; ++i) A[i] = float_to_half(1.0f);
        for (int i = 0; i < k * n; ++i) B[i] = float_to_half(2.0f);
        for (int i = 0; i < m * n; ++i) C[i] = float_to_half(0.0f);

        // Warm up
        cblas_hgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, k, B, n, beta, C, n);

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            cblas_hgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, alpha, A, k, B, n, beta, C, n);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        double avg_time_ms = duration.count() / iterations;

        // [FIX 2] FLOPs for M x K x 1 GEMV = 2 * m * k (was 2*m*n*k with n=size).
        double ops = 2.0 * m * k;
        double gflops = (ops / (avg_time_ms / 1000.0)) / 1e9;

        std::cout << std::left << size << "x" << size << "x1    "
                  << std::setw(15) << avg_time_ms
                  << gflops << " GFLOPS" << std::endl;

        // [FIX 4] Basic correctness sanity check: with A=1.0, B=2.0, every
        // output element should equal 2.0 * k (dot product of k ones-times-twos).
        // Print it so a silently wrong/garbage result is visible immediately,
        // instead of only ever seeing a timing number.
        float sample = half_to_float(C[0]);
        float expected = 2.0f * k;
        bool ok = !std::isnan(sample) && !std::isinf(sample)
                  && std::fabs(sample - expected) < expected * 0.05f; // 5% tol for fp16 rounding
        std::cout << "  checksum C[0] = " << sample
                   << " (expected ~" << expected << ") "
                   << (ok ? "OK" : "MISMATCH - RESULTS SUSPECT") << std::endl;

        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
    }

    return 0;
}