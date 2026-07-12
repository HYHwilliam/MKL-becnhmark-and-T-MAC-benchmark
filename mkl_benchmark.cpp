#include <iostream>
#include <vector>
#include <chrono>
#include <mkl.h>
#include <immintrin.h>   
#include <iomanip>
#include <cmath>          

static inline MKL_F16 float_to_half(float f) {
    __m128 v = _mm_set_ss(f);
    __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
    return (MKL_F16)_mm_extract_epi16(h, 0);
}

static inline float half_to_float(MKL_F16 h) {
    __m128i hv = _mm_cvtsi32_si128((uint16_t)h);
    __m128 fv = _mm_cvtph_ps(hv);
    return _mm_cvtss_f32(fv);
}

int main() {
    int num_threads = 1;
    mkl_set_num_threads(num_threads);

    std::cout << "Initializing Intel MKL Benchmark..." << std::endl;
    std::cout << "Number of threads used: " << mkl_get_max_threads() << std::endl;

    std::vector<int> sizes = {256, 1024, 2048, 4096, 8192};

    const int n = 1;
    int iterations = 100;

    std::cout << "------------------------------------------------------\n";
    std::cout << std::left << std::setw(15) << "Matrix Size"
              << std::setw(15) << "Latency (ms)"
              << "Performance (GFLOPS)" << std::endl;   
    std::cout << "------------------------------------------------------\n";

    for (int size : sizes) {
        int m = size, k = size;

        MKL_F16 alpha = float_to_half(1.0f);   
        MKL_F16 beta  = float_to_half(0.0f);   

        MKL_F16* A = (MKL_F16*)mkl_malloc(m * k * sizeof(MKL_F16), 64);
        MKL_F16* B = (MKL_F16*)mkl_malloc(k * n * sizeof(MKL_F16), 64);
        MKL_F16* C = (MKL_F16*)mkl_malloc(m * n * sizeof(MKL_F16), 64);

        if (A == NULL || B == NULL || C == NULL) {
            std::cerr << "Memory allocation failed!" << std::endl;
            return 1;
        }

        for (int i = 0; i < m * k; ++i) A[i] = float_to_half(1.0f);
        for (int i = 0; i < k * n; ++i) B[i] = float_to_half(2.0f);
        for (int i = 0; i < m * n; ++i) C[i] = float_to_half(0.0f);

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

        double ops = 2.0 * m * k;
        double gflops = (ops / (avg_time_ms / 1000.0)) / 1e9;

        std::cout << std::left << size << "x" << size << "x1    "
                  << std::setw(15) << avg_time_ms
                  << gflops << " GFLOPS" << std::endl;

        float sample = half_to_float(C[0]);
        float expected = 2.0f * k;
        bool ok = !std::isnan(sample) && !std::isinf(sample)
                  && std::fabs(sample - expected) < expected * 0.05f; 
        std::cout << "  checksum C[0] = " << sample
                   << " (expected ~" << expected << ") "
                   << (ok ? "OK" : "MISMATCH - RESULTS SUSPECT") << std::endl;

        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
    }

    return 0;
}