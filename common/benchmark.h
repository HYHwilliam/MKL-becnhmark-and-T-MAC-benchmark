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

template <typename T>
class AlignedBuffer {
public:
    AlignedBuffer(T* ptr, void (*free_fn)(void*)) : ptr_(ptr), free_fn_(free_fn) {}
    ~AlignedBuffer() { if (ptr_) free_fn_(ptr_); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    operator T*() const { return ptr_; }

private:
    T* ptr_;
    void (*free_fn_)(void*);
};
