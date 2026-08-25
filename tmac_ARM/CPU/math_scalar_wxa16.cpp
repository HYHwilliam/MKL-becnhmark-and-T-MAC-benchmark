#include "benchmark_common.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#if defined(__FLT16_MANT_DIG__)
using fp16 = _Float16;
#else
#error "This compiler must support _Float16 for the A16 benchmark."
#endif

namespace
{

template <int Bits>
__attribute__((noinline)) void dense_scalar_kernel(int m, int k, int n, const uint8_t* qweights, const fp16* activations,
                                                   const fp16* scales, fp16* output)
{
    const int zero_point = 1 << (Bits - 1);
    const int weight_groups = k / bench::kWeightScaleGroupSize;
    for (int row = 0; row < n; ++row)
    {
        const size_t xbase = static_cast<size_t>(row) * k;
        for (int out_col = 0; out_col < m; ++out_col)
        {
            const size_t wbase = static_cast<size_t>(out_col) * k;
            const size_t sbase = static_cast<size_t>(out_col) * weight_groups;
            float sum = 0.0f;
            for (int kk = 0; kk < k; ++kk)
            {
                const int weight = static_cast<int>(qweights[wbase + kk]) - zero_point;
                const float scale = static_cast<float>(scales[sbase + kk / bench::kWeightScaleGroupSize]);
                sum += static_cast<float>(weight) * scale * static_cast<float>(activations[xbase + kk]);
            }
            output[static_cast<size_t>(row) * m + out_col] = static_cast<fp16>(sum);
        }
    }
}

template <int Bits>
bool verify_case(int m, int k, int n)
{
    const bench::RawInput raw = bench::make_raw_input<Bits>(m, k, n, static_cast<uint32_t>(1000 + Bits * 100 + m + k + n));
    const std::vector<fp16> activations = bench::cast_to_half<fp16>(raw.activations);
    const std::vector<fp16> scales = bench::cast_to_half<fp16>(raw.scales);
    std::vector<fp16> actual(static_cast<size_t>(n) * m);
    std::vector<fp16> reference;

    dense_scalar_kernel<Bits>(m, k, n, raw.qweights.data(), activations.data(), scales.data(), actual.data());
    bench::dense_reference<Bits>(m, k, n, raw.qweights, activations, scales, reference);
    const bench::Accuracy accuracy = bench::compare_outputs(actual, reference);
    const bool pass = accuracy.non_finite == 0 && accuracy.nmse <= 1e-6;
    bench::print_verification_row(m, k, n, 0, accuracy, pass);
    return pass;
}

template <int Bits>
bool verify_suite()
{
    bench::print_verification_header("C_SCALAR", Bits, false, false);
    bool pass = true;
    for (int n : {1, 8, 32, 128}) pass &= verify_case<Bits>(256, 256, n);
    pass &= verify_case<Bits>(1024, 256, 3);
    pass &= verify_case<Bits>(1024, 512, 8);
    pass &= verify_case<Bits>(512, 1024, 16);
    bench::print_verification_footer(pass);
    return pass;
}

template <int Bits>
void run_shape(const bench::Shape& shape)
{
    const bench::RawInput raw = bench::make_raw_input<Bits>(shape);
    const std::vector<fp16> activations = bench::cast_to_half<fp16>(raw.activations);
    const std::vector<fp16> scales = bench::cast_to_half<fp16>(raw.scales);
    std::vector<fp16> output(static_cast<size_t>(shape.n) * shape.m);

    const bench::BenchmarkStats stats = bench::measure(shape, [&]
    {
        dense_scalar_kernel<Bits>(shape.m, shape.k, shape.n, raw.qweights.data(), activations.data(), scales.data(), output.data());
    });
    bench::print_benchmark_result(shape, stats, bench::checksum(output));
}

template <int Bits>
int run(const bench::Options& options)
{
    if (!verify_suite<Bits>()) return 1;
    if (options.verify_only) return 0;
    bench::print_benchmark_header("C_SCALAR", Bits);
    for (const bench::Shape& shape : bench::benchmark_shapes()) if (shape.m <= options.max_size) run_shape<Bits>(shape);
    bench::print_benchmark_footer();
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const bench::Options options = bench::parse_options(argc, argv);
        if (options.bits == 2) return run<2>(options);
        if (options.bits == 3) return run<3>(options);
        return run<4>(options);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
