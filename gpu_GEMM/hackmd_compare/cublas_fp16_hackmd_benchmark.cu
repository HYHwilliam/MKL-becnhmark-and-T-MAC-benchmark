#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Shape
{
    int M;
    int K;
    int N;
    const char* category;
};

void check_cuda(cudaError_t status, const char* message)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(message) + ": " +
            cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const char* message)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error(
            std::string(message) +
            ": cuBLAS status=" +
            std::to_string(static_cast<int>(status)));
    }
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
        return 0.5 *
            (values[middle - 1] + values[middle]);
    }

    return values[middle];
}

double percentile(
    std::vector<double> values,
    double ratio)
{
    if (values.empty())
    {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    const size_t index =
        static_cast<size_t>(
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

void generate_fp16_data(
    const Shape& shape,
    std::vector<__half>& weights,
    std::vector<__half>& activations)
{
    weights.resize(
        static_cast<size_t>(shape.M) *
        static_cast<size_t>(shape.K));

    activations.resize(
        static_cast<size_t>(shape.N) *
        static_cast<size_t>(shape.K));

    std::mt19937 generator(
        static_cast<uint32_t>(
            42 +
            shape.M +
            shape.K +
            shape.N));

    std::uniform_real_distribution<float> distribution(
        -1.0f,
        1.0f);

    for (__half& value : weights)
    {
        value = __float2half(
            distribution(generator));
    }

    for (__half& value : activations)
    {
        value = __float2half(
            distribution(generator));
    }
}

void run_gemm(
    cublasHandle_t handle,
    const Shape& shape,
    const __half* d_weights,
    const __half* d_activations,
    __half* d_output)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    check_cublas(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            shape.M,
            shape.N,
            shape.K,
            &alpha,
            d_weights,
            CUDA_R_16F,
            shape.K,
            d_activations,
            CUDA_R_16F,
            shape.K,
            &beta,
            d_output,
            CUDA_R_16F,
            shape.M,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx");
}

double checksum_fp16(
    const std::vector<__half>& values)
{
    double checksum = 0.0;

    for (const __half& value : values)
    {
        checksum +=
            static_cast<double>(
                __half2float(value));
    }

    return checksum;
}

double verify_output(
    const Shape& shape,
    const std::vector<__half>& weights,
    const std::vector<__half>& activations,
    const std::vector<__half>& output)
{
    constexpr int verification_points = 16;

    double max_abs_error = 0.0;

    const size_t output_size =
        static_cast<size_t>(shape.N) *
        static_cast<size_t>(shape.M);

    for (int point = 0;
         point < verification_points;
         ++point)
    {
        const size_t index =
            output_size == 1
                ? 0
                : static_cast<size_t>(
                      (static_cast<double>(point) /
                       (verification_points - 1)) *
                      (output_size - 1));

        const int n =
            static_cast<int>(
                index /
                static_cast<size_t>(shape.M));

        const int m =
            static_cast<int>(
                index %
                static_cast<size_t>(shape.M));

        float reference = 0.0f;

        for (int k = 0; k < shape.K; ++k)
        {
            const float a =
                __half2float(
                    activations[
                        static_cast<size_t>(n) *
                            shape.K +
                        k]);

            const float w =
                __half2float(
                    weights[
                        static_cast<size_t>(m) *
                            shape.K +
                        k]);

            reference += a * w;
        }

        const float reference_fp16 =
            __half2float(
                __float2half(reference));

        const float gpu_value =
            __half2float(output[index]);

        const double abs_error =
            std::abs(
                static_cast<double>(
                    gpu_value -
                    reference_fp16));

        max_abs_error =
            std::max(
                max_abs_error,
                abs_error);
    }

    return max_abs_error;
}

void run_shape(
    cublasHandle_t handle,
    const Shape& shape)
{
    std::vector<__half> weights;
    std::vector<__half> activations;

    generate_fp16_data(
        shape,
        weights,
        activations);

    std::vector<__half> output(
        static_cast<size_t>(shape.N) *
        static_cast<size_t>(shape.M));

    const size_t weights_bytes =
        weights.size() * sizeof(__half);

    const size_t activations_bytes =
        activations.size() * sizeof(__half);

    const size_t output_bytes =
        output.size() * sizeof(__half);

    __half* d_weights = nullptr;
    __half* d_activations = nullptr;
    __half* d_output = nullptr;

    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    try
    {
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_weights),
                weights_bytes),
            "cudaMalloc weights");

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_activations),
                activations_bytes),
            "cudaMalloc activations");

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_output),
                output_bytes),
            "cudaMalloc output");

        check_cuda(
            cudaMemcpy(
                d_weights,
                weights.data(),
                weights_bytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy weights");

        check_cuda(
            cudaMemcpy(
                d_activations,
                activations.data(),
                activations_bytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy activations");

        check_cuda(
            cudaEventCreate(&start_event),
            "cudaEventCreate start");

        check_cuda(
            cudaEventCreate(&stop_event),
            "cudaEventCreate stop");

        for (int warmup = 0;
             warmup < 10;
             ++warmup)
        {
            run_gemm(
                handle,
                shape,
                d_weights,
                d_activations,
                d_output);
        }

        check_cuda(
            cudaDeviceSynchronize(),
            "warmup synchronize");

        const double logical_flops =
            2.0 *
            static_cast<double>(shape.M) *
            static_cast<double>(shape.K) *
            static_cast<double>(shape.N);

        const int samples =
            sample_count(logical_flops);

        std::vector<double> timings;
        timings.reserve(samples);

        for (int sample = 0;
             sample < samples;
             ++sample)
        {
            check_cuda(
                cudaEventRecord(start_event),
                "cudaEventRecord start");

            run_gemm(
                handle,
                shape,
                d_weights,
                d_activations,
                d_output);

            check_cuda(
                cudaEventRecord(stop_event),
                "cudaEventRecord stop");

            check_cuda(
                cudaEventSynchronize(stop_event),
                "cudaEventSynchronize");

            float elapsed_ms = 0.0f;

            check_cuda(
                cudaEventElapsedTime(
                    &elapsed_ms,
                    start_event,
                    stop_event),
                "cudaEventElapsedTime");

            timings.push_back(
                static_cast<double>(
                    elapsed_ms));
        }

        check_cuda(
            cudaMemcpy(
                output.data(),
                d_output,
                output_bytes,
                cudaMemcpyDeviceToHost),
            "cudaMemcpy output");

        const double total_ms =
            median(timings);

        const double p90_ms =
            percentile(
                timings,
                0.90);

        const double gflops =
            logical_flops /
            (total_ms / 1000.0) /
            1.0e9;

        const double checksum =
            checksum_fp16(output);

        const double verify_max_abs_error =
            verify_output(
                shape,
                weights,
                activations,
                output);

        std::cout
            << "RESULT "
            << "backend=cuBLAS "
            << "dtype=fp16 "
            << "compute=fp32 "
            << "category=\""
            << shape.category
            << "\" "
            << "shape="
            << shape.M << "x"
            << shape.N << "x"
            << shape.K << " "
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
            << "verify_max_abs_error="
            << verify_max_abs_error << " "
            << "checksum="
            << checksum
            << "\n";

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);

        cudaFree(d_weights);
        cudaFree(d_activations);
        cudaFree(d_output);
    }
    catch (...)
    {
        if (d_weights != nullptr)
        {
            cudaFree(d_weights);
        }

        if (d_activations != nullptr)
        {
            cudaFree(d_activations);
        }

        if (d_output != nullptr)
        {
            cudaFree(d_output);
        }

        throw;
    }
}

int main()
{
    try
    {
        int device_count = 0;

        check_cuda(
            cudaGetDeviceCount(&device_count),
            "cudaGetDeviceCount");

        if (device_count <= 0)
        {
            throw std::runtime_error(
                "No CUDA GPU detected");
        }

        check_cuda(
            cudaSetDevice(0),
            "cudaSetDevice");

        cudaDeviceProp properties{};

        check_cuda(
            cudaGetDeviceProperties(
                &properties,
                0),
            "cudaGetDeviceProperties");

        cublasHandle_t handle;

        check_cublas(
            cublasCreate(&handle),
            "cublasCreate");

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
            << "cuBLAS FP16 GPU HackMD Benchmark\n";

        std::cout
            << "GPU: "
            << properties.name
            << "\n";

        std::cout
            << "Compute Capability: "
            << properties.major
            << "."
            << properties.minor
            << "\n";

        std::cout
            << "FP16 input/output, FP32 accumulation\n";

        std::cout
            << "CUDA Event kernel timing, H2D/D2H excluded\n";

        std::cout
            << "============================================================\n";

        for (const Shape& shape : shapes)
        {
            run_shape(
                handle,
                shape);
        }

        cublasDestroy(handle);

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "ERROR: "
            << error.what()
            << "\n";

        return 1;
    }
}