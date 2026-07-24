#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

int main() {
    std::cout << "初始化 Naive Triple-Loop GEMV Benchmark (N=1)..." << std::endl;
    std::vector<int> sizes = {256, 1024, 2048, 4096, 8192};

    std::cout << "------------------------------------------------------\n";
    std::cout << std::left << std::setw(15) << "Matrix Size"
              << std::setw(15) << "Latency (ms)"
              << "Performance (GFLOPS)" << std::endl;
    std::cout << "------------------------------------------------------\n";

    const int N = 1; // GEMV

    for (int size : sizes) {
        int M = size, K = size;

        // W[M][K]，全部設為 1.0f；x[K] 全部設為 1.0f
        std::vector<float> W(M * K, 1.0f);
        std::vector<float> x(K, 1.0f);
        std::vector<float> y(M * N, 0.0f);

        int iterations = (size <= 2048) ? 20 : 5; // 大矩陣太慢，減少次數

        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter) {
            for (int i = 0; i < M; ++i) {          
                for (int j = 0; j < N; ++j) {       
                    float acc = 0.0f;
                    for (int k = 0; k < K; ++k) {   
                        acc += W[i * K + k] * x[k];
                    }
                    y[i * N + j] = acc;
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        double avg_time_ms = duration.count() / iterations;

        double ops = 2.0 * M * K * N;
        double gflops = (ops / (avg_time_ms / 1000.0)) / 1e9;

        std::cout << std::left << size << "x" << size << "x1    "
                  << std::setw(15) << avg_time_ms
                  << gflops << " GFLOPS" << std::endl;

        float expected = (float)K;
        bool ok = std::fabs(y[0] - expected) < 1e-3f * expected;
        std::cout << "  input[0..7]  = ";
        for (int i = 0; i < 8; ++i) std::cout << x[i] << " ";
        std::cout << "\n  output[0..7] = ";
        for (int i = 0; i < 8; ++i) std::cout << y[i] << " ";
        std::cout << "\n  y[0] = " << y[0] << " (expected " << expected << ") "
                   << (ok ? "OK" : "MISMATCH") << std::endl;
    }
    return 0;
}