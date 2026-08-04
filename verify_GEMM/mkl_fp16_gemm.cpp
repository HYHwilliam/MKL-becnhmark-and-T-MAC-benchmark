// mkl_gemm.cpp
//
// Intel oneMKL FP16 GEMM benchmark for comparison with t_mac_gemm.cpp.
//
// Matrix convention:
//   weights     : M x K, row-major
//   activations : N x K, row-major
//   output      : N x M, row-major
//
// Computes:
//   output = activations * transpose(weights)
//
// Benchmark alignment with t_mac_gemm.cpp:
//   - Same M, K, N test shapes
//   - Same activation distribution: Uniform[0.1, 5.0]
//   - Same dynamic iteration formula
//   - One untimed warm-up call
//   - Default one thread, because the current T-MAC kernel is single-threaded
//   - Timing includes only the operator execution, not allocation, initialization,
//     conversion, verification, or printing

#include <mkl.h>
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

struct Config {
    int M;
    int K;
    int N;
};

static inline MKL_F16 fp32_to_fp16(float value) {
    const __m128 input = _mm_set_ss(value);
    const __m128i packed = _mm_cvtps_ph(input, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const uint16_t bits = static_cast<uint16_t>(_mm_extract_epi16(packed, 0));
    return static_cast<MKL_F16>(bits);
}

static inline float fp16_to_fp32(MKL_F16 value) {
    const uint16_t bits = static_cast<uint16_t>(value);
    const __m128i packed = _mm_cvtsi32_si128(static_cast<int>(bits));
    const __m128 result = _mm_cvtph_ps(packed);
    return _mm_cvtss_f32(result);
}

template <typename T>
class MklBuffer {
public:
    MklBuffer() : ptr_(nullptr) {}

    explicit MklBuffer(std::size_t count) {
        ptr_ = static_cast<T*>(mkl_malloc(count * sizeof(T), 64));
    }

    ~MklBuffer() {
        mkl_free(ptr_);
    }

    MklBuffer(const MklBuffer&) = delete;
    MklBuffer& operator=(const MklBuffer&) = delete;

    MklBuffer(MklBuffer&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    MklBuffer& operator=(MklBuffer&& other) noexcept {
        if (this != &other) {
            mkl_free(ptr_);
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    T& operator[](std::size_t index) { return ptr_[index]; }
    const T& operator[](std::size_t index) const { return ptr_[index]; }
    explicit operator bool() const { return ptr_ != nullptr; }

private:
    T* ptr_;
};

static void mkl_fp16_gemm(int M, int K, int N, const MKL_F16* weights, const MKL_F16* activations, MKL_F16* output) {
    const MKL_F16 alpha = fp32_to_fp16(1.0f);
    const MKL_F16 beta = fp32_to_fp16(0.0f);

    cblas_hgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, 
        N, M, K, 
        alpha, activations, K, 
        weights, K, 
        beta, output, M
    );
}

static float scalar_reference_element(int M, int K, int N, int n, int m, const MKL_F16* weights, const MKL_F16* activations) {
    (void)M;
    (void)N;

    float sum = 0.0f;
    const std::size_t activation_base = static_cast<std::size_t>(n) * K;
    const std::size_t weight_base = static_cast<std::size_t>(m) * K;

    for (int k = 0; k < K; ++k) {
        const float activation = fp16_to_fp32(activations[activation_base + k]);
        const float weight = fp16_to_fp32(weights[weight_base + k]);
        sum += activation * weight;
    }

    return sum;
}

static bool verify_sampled_outputs(int M, int K, int N, const MKL_F16* weights, const MKL_F16* activations, const MKL_F16* output, double& max_abs_error, double& max_rel_error, int& mismatch_count) {
    constexpr int max_samples = 64;

    max_abs_error = 0.0;
    max_rel_error = 0.0;
    mismatch_count = 0;

    const std::size_t total_outputs = static_cast<std::size_t>(N) * M;
    const int sample_count = static_cast<int>(std::min<std::size_t>(max_samples, total_outputs));

    for (int sample = 0; sample < sample_count; ++sample) {
        std::size_t linear_index = 0;

        if (sample_count == 1) {
            linear_index = 0;
        } else {
            linear_index = static_cast<std::size_t>(sample) * (total_outputs - 1) / static_cast<std::size_t>(sample_count - 1);
        }

        const int n = static_cast<int>(linear_index / M);
        const int m = static_cast<int>(linear_index % M);

        const float reference_fp32 = scalar_reference_element(M, K, N, n, m, weights, activations);
        const float reference_fp16 = fp16_to_fp32(fp32_to_fp16(reference_fp32));
        const float actual = fp16_to_fp32(output[linear_index]);

        if (!std::isfinite(actual) || !std::isfinite(reference_fp16)) {
            ++mismatch_count;
            continue;
        }

        const double abs_error = std::fabs(static_cast<double>(actual) - static_cast<double>(reference_fp16));
        const double rel_error = abs_error / std::max(1.0, std::fabs(static_cast<double>(reference_fp16)));

        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);

        const double tolerance = 0.01 * std::fabs(static_cast<double>(reference_fp16)) + 0.05;

        if (abs_error > tolerance) {
            if (mismatch_count == 0) {
                std::cout << "\n  First mismatch:"
                          << " n=" << n << " m=" << m
                          << " reference=" << reference_fp16
                          << " actual=" << actual
                          << " abs_error=" << abs_error << "\n";
            }
            ++mismatch_count;
        }
    }

    return mismatch_count == 0;
}

static void run_config(const Config& config, std::mt19937& rng) {
    const int M = config.M;
    const int K = config.K;
    const int N = config.N;

    const std::size_t weight_count = static_cast<std::size_t>(M) * K;
    const std::size_t activation_count = static_cast<std::size_t>(N) * K;
    const std::size_t output_count = static_cast<std::size_t>(N) * M;

    MklBuffer<MKL_F16> weights(weight_count);
    MklBuffer<MKL_F16> activations(activation_count);
    MklBuffer<MKL_F16> output(output_count);

    if (!weights || !activations || !output) {
        std::cerr << "Memory allocation failed for " << M << "x" << K << "x" << N << "\n";
        return;
    }

    // Match t_mac_gemm.cpp activation distribution exactly.
    std::uniform_real_distribution<float> activation_distribution(0.1f, 5.0f);

    // Dense FP16 baseline weights must contain both signs; otherwise,
    // large positive-only dot products can overflow FP16 output.
    std::uniform_real_distribution<float> weight_distribution(-1.0f, 1.0f);

    for (std::size_t i = 0; i < weight_count; ++i) {
        weights[i] = fp32_to_fp16(weight_distribution(rng));
    }

    for (std::size_t i = 0; i < activation_count; ++i) {
        activations[i] = fp32_to_fp16(activation_distribution(rng));
    }

    // Exactly one untimed warm-up, matching t_mac_gemm.cpp.
    mkl_fp16_gemm(M, K, N, weights.data(), activations.data(), output.data());

    // Verify outside the timed region.
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    int mismatch_count = 0;

    const bool verification_passed = verify_sampled_outputs(
        M, K, N, weights.data(), activations.data(), output.data(),
        max_abs_error, max_rel_error, mismatch_count
    );

    // Match t_mac_gemm.cpp iteration formula exactly.
    const double total_flops_per_call = 2.0 * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N);
    const int iterations = static_cast<int>(std::max(2.0, std::min(100.0, 5e10 / total_flops_per_call)));

    const auto start = std::chrono::high_resolution_clock::now();

    for (int iteration = 0; iteration < iterations; ++iteration) {
        mkl_fp16_gemm(M, K, N, weights.data(), activations.data(), output.data());
    }

    const auto end = std::chrono::high_resolution_clock::now();

    const double average_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    const double gflops = total_flops_per_call / (average_ms / 1000.0) / 1e9;

    std::cout << std::left << std::setw(20) << (std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N))
              << std::setw(16) << average_ms
              << std::setw(16) << gflops << " GFLOPS"
              << (verification_passed ? "\033[32m [PASS]\033[0m" : "\033[31m [FAIL]\033[0m")
              << " samples=" << std::min<std::size_t>(64, output_count)
              << " mismatches=" << mismatch_count
              << " max_abs_err=" << max_abs_error
              << " max_rel_err=" << max_rel_error
              << " iterations=" << iterations << "\n";
}

int main(int argc, char** argv) {
    int thread_count = 1;
    if (argc > 1) {
        thread_count = std::max(1, std::atoi(argv[1]));
    }

    mkl_set_dynamic(0);
    mkl_set_num_threads(thread_count);

    std::mt19937 rng(42);

    const std::vector<int> batch_sizes = {1, 8, 32, 128, 512};

    const std::vector<Config> square_shapes = {
        {1024, 1024, 0}, {2048, 2048, 0}, {4096, 4096, 0}, {8192, 8192, 0}
    };

    const std::vector<Config> k_expansion_shapes = {
        {1024, 4096, 0}, {2048, 8192, 0}, {4096, 11008, 0}, {4096, 14336, 0}
    };

    const std::vector<Config> m_expansion_shapes = {
        {4096, 1024, 0}, {8192, 2048, 0}, {11008, 4096, 0}, {14336, 4096, 0}
    };

    const std::vector<Config> asymmetric_shapes = {
        {1024, 2048, 0}, {2048, 1024, 0}, {2048, 4096, 0}, {4096, 2048, 0}
    };

    std::cout << "==================================================================\n"
              << " Intel oneMKL FP16 GEMM Benchmark\n"
              << " output = activations[N x K] * weights[M x K]^T\n"
              << " MKL threads: " << mkl_get_max_threads() << "\n"
              << " Build: " << __DATE__ << " " << __TIME__ << "\n"
              << "==================================================================\n";

    std::cout << "\n=== Square GEMM ===\n";
    for (int N : batch_sizes) {
        std::cout << "\n--- N = " << N << " ---\n";
        for (const Config& shape : square_shapes) {
            run_config({shape.M, shape.K, N}, rng);
        }
    }

    std::cout << "\n=== Rectangular GEMM: K Expansion ===\n";
    for (int N : batch_sizes) {
        std::cout << "\n--- N = " << N << " ---\n";
        for (const Config& shape : k_expansion_shapes) {
            run_config({shape.M, shape.K, N}, rng);
        }
    }

    std::cout << "\n=== Rectangular GEMM: M Expansion ===\n";
    for (int N : batch_sizes) {
        std::cout << "\n--- N = " << N << " ---\n";
        for (const Config& shape : m_expansion_shapes) {
            run_config({shape.M, shape.K, N}, rng);
        }
    }

    std::cout << "\n=== Medium Asymmetric GEMM ===\n";
    for (int N : batch_sizes) {
        std::cout << "\n--- N = " << N << " ---\n";
        for (const Config& shape : asymmetric_shapes) {
            run_config({shape.M, shape.K, N}, rng);
        }
    }

    return 0;
}