#pragma once

#include <iostream>
#include <iomanip>

inline void print_benchmark_header() {
    std::cout << "------------------------------------------------------\n";
    std::cout << std::left << std::setw(15) << "Matrix Size"
              << std::setw(15) << "Latency (ms)"
              << "Performance (GFLOPS)" << std::endl;
    std::cout << "------------------------------------------------------\n";
}

inline double compute_gflops(long long m, long long k, double avg_time_ms) {
    double ops = 2.0 * m * k;
    return (ops / (avg_time_ms / 1000.0)) / 1e9;
}

inline void print_benchmark_row(int size, double avg_time_ms, double gflops) {
    std::cout << std::left << size << "x" << size << "x1    "
              << std::setw(15) << avg_time_ms
              << gflops << " GFLOPS" << std::endl;
}
