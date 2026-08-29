#include <mkl.h>
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct Shape
{
    int M;
    int K;
    int N;
    const char* category;
};

std::vector<int> parse_threads(const std::string& text)
{
    std::vector<int> result;
    std::stringstream stream(text);
    std::string token;

    while (std::getline(stream, token, ','))
    {
        result.push_back(std::max(1, std::stoi(token)));
    }

    if (result.empty())
    {
        result = {1, 2, 4, 8, 16};
    }

    return result;
}

double median(std::vector<double> values)
{
    if (values.empty())
    {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    const size_t middle = values.size() / 2;

    if (values.size() % 2 == 0)
    {
        return 0.5 * (values[middle - 1] + values[middle]);
    }

    return values[middle];
}

double percentile(std::vector<double> values, double ratio)
{
    if (values.empty())
    {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    const size_t index = static_cast<size_t>(
        std::min<double>(
            values.size() - 1,
            std::ceil(ratio * values.size()) - 1));

    return values[index];
}

int sample_count(double logical_flops)
{
    if (logical_flops < 1.0e8)
    {
        return 21;
    }

    if (logical_flops < 1.0e9)
    {
        return 11;
    }

    if (logical_flops < 1.0e10)
    {
        return 7;
    }

    return 5;
}

MKL_F16 float_to_fp16(float value)
{
    const uint16_t bits = _cvtss_sh(
        value,
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    MKL_F16 result;

    static_assert(sizeof(MKL_F16) == sizeof(uint16_t));

    std::memcpy(&result, &bits, sizeof(bits));

    return result;
}

float fp16_to_float(MKL_F16 value)
{
    uint16_t bits;

    std::memcpy(&bits, &value, sizeof(bits));

    return _cvtsh_ss(bits);
}

void generate_fp16_data(
    const Shape& shape,
    std::vector<MKL_F16>& weights,
    std::vector<MKL_F16>& activations)
{
    weights.resize(
        static_cast<size_t>(shape.M) *
        static_cast<size_t>(shape.K));

    activations.resize(
        static_cast<size_t>(shape.N) *
        static_cast<size_t>(shape.K));

    std::mt19937 generator(
        static_cast<uint32_t>(
            42 + shape.M + shape.K + shape.N));

    std::uniform_real_distribution<float> distribution(
        -1.0f,
        1.0f);

    for (MKL_F16& value : weights)
    {
        value = float_to_fp16(distribution(generator));
    }

    for (MKL_F16& value : activations)
    {
        value = float_to_fp16(distribution(generator));
    }
}

void run_shape(
    const Shape& shape,
    const std::vector<int>& thread_counts)
{
    std::vector<MKL_F16> weights;
    std::vector<MKL_F16> activations;

    generate_fp16_data(
        shape,
        weights,
        activations);

    std::vector<MKL_F16> output(
        static_cast<size_t>(shape.N) *
        static_cast<size_t>(shape.M));

    const MKL_F16 alpha = float_to_fp16(1.0f);
    const MKL_F16 beta = float_to_fp16(0.0f);

    const double logical_flops =
        2.0 *
        static_cast<double>(shape.M) *
        static_cast<double>(shape.K) *
        static_cast<double>(shape.N);

    for (int threads : thread_counts)
    {
        mkl_set_dynamic(0);
        mkl_set_num_threads_local(threads);

        for (int warmup = 0; warmup < 10; ++warmup)
        {
            cblas_hgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                shape.N,
                shape.M,
                shape.K,
                alpha,
                activations.data(),
                shape.K,
                weights.data(),
                shape.K,
                beta,
                output.data(),
                shape.M);
        }

        const int samples = sample_count(logical_flops);

        std::vector<double> timings;
        timings.reserve(samples);

        for (int sample = 0; sample < samples; ++sample)
        {
            const auto start =
                std::chrono::high_resolution_clock::now();

            cblas_hgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                shape.N,
                shape.M,
                shape.K,
                alpha,
                activations.data(),
                shape.K,
                weights.data(),
                shape.K,
                beta,
                output.data(),
                shape.M);

            const auto end =
                std::chrono::high_resolution_clock::now();

            timings.push_back(
                std::chrono::duration<double, std::milli>(
                    end - start)
                    .count());
        }

        const double total_ms =
            median(timings);

        const double p90_ms =
            percentile(timings, 0.90);

        const double gflops =
            logical_flops /
            (total_ms / 1000.0) /
            1.0e9;

        double checksum = 0.0;

        for (const MKL_F16& value : output)
        {
            checksum += fp16_to_float(value);
        }

        std::cout
            << "RESULT "
            << "backend=MKL "
            << "dtype=fp16 "
            << "category=\"" << shape.category << "\" "
            << "shape="
            << shape.M << "x"
            << shape.N << "x"
            << shape.K << " "
            << "threads=" << threads << " "
            << "total_ms="
            << std::fixed
            << std::setprecision(6)
            << total_ms << " "
            << "p90_ms="
            << p90_ms << " "
            << "gflops="
            << gflops << " "
            << "samples="
            << samples << " "
            << "checksum="
            << checksum
            << "\n";
    }
}

int main(int argc, char** argv)
{
    std::vector<int> threads = {
        1,
        2,
        4,
        8,
        16
    };

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--threads" && i + 1 < argc)
        {
            threads = parse_threads(argv[++i]);
        }
        else
        {
            std::cerr
                << "Usage: "
                << argv[0]
                << " [--threads 1,2,4,8,16]\n";

            return 1;
        }
    }

    const std::vector<Shape> shapes = {
        {
            256,
            256,
            256,
            "Small Square"
        },
        {
            1024,
            1024,
            1024,
            "Medium Square"
        },
        {
            4096,
            4096,
            4096,
            "Large Square"
        },
        {
            4096,
            2048,
            1024,
            "Rectangular"
        },
        {
            1024,
            512,
            1024,
            "Medium Rectangular"
        }
    };

    std::cout
        << "============================================================\n";

    std::cout
        << "oneMKL FP16 CPU-only HackMD Benchmark\n";

    std::cout
        << "Dense FP16 x FP16 using cblas_hgemm\n";

    std::cout
        << "============================================================\n";

    for (const Shape& shape : shapes)
    {
        run_shape(
            shape,
            threads);
    }

    return 0;
}