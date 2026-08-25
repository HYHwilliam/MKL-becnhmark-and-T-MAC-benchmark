#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench
{

constexpr int kWeightScaleGroupSize = 128;
constexpr double kTmacNmseLimit = 5e-4;

struct Shape
{
    int m;
    int k;
    int n;
    int warmup;
    int repeat;
    int samples;

    const char* kind() const { return n == 1 ? "GEMV" : "GEMM"; }
};

inline Shape make_shape(int m, int k, int n)
{
    const double ops = 2.0 * static_cast<double>(m) * k * n;
    int warmup = 1;
    int repeat = 1;
    int samples = 3;
    if (ops < 1.0e7) { warmup = 3; repeat = 10; samples = 5; }
    else if (ops < 1.0e8) { warmup = 3; repeat = 5; samples = 5; }
    else if (ops < 1.0e9) { warmup = 2; repeat = 2; samples = 5; }
    else if (ops < 5.0e9) { warmup = 1; repeat = 1; samples = 5; }
    return {m, k, n, warmup, repeat, samples};
}

inline std::vector<Shape> benchmark_shapes()
{
    constexpr std::array<int, 5> sizes = {256, 1024, 2048, 4096, 8192};
    constexpr std::array<int, 4> batches = {1, 8, 32, 128};
    std::vector<Shape> shapes;
    shapes.reserve(sizes.size() * batches.size());
    for (int n : batches) for (int size : sizes) shapes.push_back(make_shape(size, size, n));
    return shapes;
}

struct Options
{
    int bits = 2;
    int max_size = 8192;
    bool verify_only = false;
};

inline Options parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--bits" && i + 1 < argc) options.bits = std::stoi(argv[++i]);
        else if (arg == "--max-size" && i + 1 < argc) options.max_size = std::stoi(argv[++i]);
        else if (arg == "--verify-only") options.verify_only = true;
        else throw std::invalid_argument("Usage: program [--bits 2|3|4] [--max-size 256|1024|2048|4096|8192] [--verify-only]");
    }
    if (options.bits < 2 || options.bits > 4) throw std::invalid_argument("--bits must be 2, 3, or 4");
    return options;
}

inline uint32_t mix32(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

inline float deterministic_float(uint32_t index, uint32_t seed, float low, float high)
{
    const float unit = static_cast<float>(mix32(index ^ seed) & 0x00ffffffU) / static_cast<float>(0x01000000U);
    return low + (high - low) * unit;
}

struct RawInput
{
    std::vector<uint8_t> qweights;
    std::vector<float> activations;
    std::vector<float> scales;
};

template <int Bits>
RawInput make_raw_input(int m, int k, int n, uint32_t seed)
{
    static_assert(Bits >= 2 && Bits <= 4);
    if (m <= 0 || k <= 0 || n <= 0) throw std::invalid_argument("M, K, and N must be positive");
    if (k % kWeightScaleGroupSize != 0) throw std::invalid_argument("K must be divisible by 128");
    const int weight_groups = k / kWeightScaleGroupSize;
    RawInput input;
    input.qweights.resize(static_cast<size_t>(m) * k);
    input.activations.resize(static_cast<size_t>(n) * k);
    input.scales.resize(static_cast<size_t>(m) * weight_groups);
    const uint32_t mask = (1U << Bits) - 1U;
    for (size_t i = 0; i < input.qweights.size(); ++i) input.qweights[i] = static_cast<uint8_t>(mix32(static_cast<uint32_t>(i) ^ seed) & mask);
    for (size_t i = 0; i < input.activations.size(); ++i) input.activations[i] = deterministic_float(static_cast<uint32_t>(i), seed + 101U, -1.0f, 1.0f);
    for (size_t i = 0; i < input.scales.size(); ++i) input.scales[i] = deterministic_float(static_cast<uint32_t>(i), seed + 211U, 0.02f, 0.15f);
    return input;
}

template <int Bits>
RawInput make_raw_input(const Shape& shape)
{
    return make_raw_input<Bits>(shape.m, shape.k, shape.n, static_cast<uint32_t>(42 + shape.m + shape.k + shape.n));
}

template <typename Half>
std::vector<Half> cast_to_half(const std::vector<float>& source)
{
    std::vector<Half> result(source.size());
    for (size_t i = 0; i < source.size(); ++i) result[i] = static_cast<Half>(source[i]);
    return result;
}

template <int Bits, typename Half>
void dense_reference(int m, int k, int n, const std::vector<uint8_t>& qweights, const std::vector<Half>& activations,
                     const std::vector<Half>& scales, std::vector<Half>& output)
{
    static_assert(Bits >= 2 && Bits <= 4);
    const int zero_point = 1 << (Bits - 1);
    const int weight_groups = k / kWeightScaleGroupSize;
    output.resize(static_cast<size_t>(n) * m);
    for (int row = 0; row < n; ++row)
    {
        const size_t xbase = static_cast<size_t>(row) * k;
        for (int out_col = 0; out_col < m; ++out_col)
        {
            const size_t wbase = static_cast<size_t>(out_col) * k;
            const size_t sbase = static_cast<size_t>(out_col) * weight_groups;
            double sum = 0.0;
            for (int kk = 0; kk < k; ++kk)
            {
                const int signed_weight = static_cast<int>(qweights[wbase + kk]) - zero_point;
                const double scale = static_cast<double>(static_cast<float>(scales[sbase + kk / kWeightScaleGroupSize]));
                const double activation = static_cast<double>(static_cast<float>(activations[xbase + kk]));
                sum += static_cast<double>(signed_weight) * scale * activation;
            }
            output[static_cast<size_t>(row) * m + out_col] = static_cast<Half>(static_cast<float>(sum));
        }
    }
}


constexpr int kReferenceLutGroupSize = 4;
constexpr int kReferenceLutEntries = 16;
constexpr int kReferenceActivationGroupSize = 64;
constexpr int kReferenceKFactor = 16;

template <typename Half, typename Value>
inline Half reference_half(Value value)
{
    return static_cast<Half>(value);
}

template <typename Half>
inline Half reference_half_add(Half a, Half b)
{
    return reference_half<Half>(static_cast<float>(a) + static_cast<float>(b));
}

template <typename Half>
inline Half reference_half_sub(Half a, Half b)
{
    return reference_half<Half>(static_cast<float>(a) - static_cast<float>(b));
}

template <typename Half>
inline Half reference_half_mul(Half a, Half b)
{
    return reference_half<Half>(static_cast<float>(a) * static_cast<float>(b));
}

template <typename Half>
inline Half reference_half_fma(Half acc, Half a, Half b)
{
    return reference_half<Half>(std::fma(static_cast<float>(a), static_cast<float>(b), static_cast<float>(acc)));
}

inline int round_to_nearest_even_int(float value)
{
    const float lower_f = std::floor(value);
    int lower = static_cast<int>(lower_f);
    const float fraction = value - lower_f;
    if (fraction > 0.5f || (fraction == 0.5f && (lower & 1))) ++lower;
    return lower;
}

template <typename Half>
struct OfficialLutReference
{
    std::vector<int8_t> quantized_luts;
    std::vector<Half> lut_scales;
    std::vector<Half> lut_biases;
};

template <typename Half>
OfficialLutReference<Half> build_official_lut_reference(int n, int k, const std::vector<Half>& activations)
{
    if (k % kReferenceActivationGroupSize != 0) throw std::invalid_argument("K must be divisible by the activation group size");
    if (activations.size() != static_cast<size_t>(n) * k) throw std::invalid_argument("Activation size mismatch");

    const int k_groups = k / kReferenceLutGroupSize;
    const int activation_groups = k / kReferenceActivationGroupSize;
    OfficialLutReference<Half> reference;
    reference.quantized_luts.resize(static_cast<size_t>(n) * k_groups * kReferenceLutEntries);
    reference.lut_scales.resize(static_cast<size_t>(n) * activation_groups);
    reference.lut_biases.resize(static_cast<size_t>(n) * activation_groups);

    for (int row = 0; row < n; ++row)
    {
        for (int group = 0; group < activation_groups; ++group)
        {
            const Half* base_group = activations.data() + static_cast<size_t>(row) * k + group * kReferenceActivationGroupSize;
            Half maximum = reference_half<Half>(0.0f);

            for (int block32 = 0; block32 < 2; ++block32)
            {
                const Half* base = base_group + block32 * 32;
                for (int lane = 0; lane < 8; ++lane)
                {
                    Half sum = reference_half_add(reference_half<Half>(std::abs(static_cast<float>(base[lane * 4]))),
                                                  reference_half<Half>(std::abs(static_cast<float>(base[lane * 4 + 1]))));
                    sum = reference_half_add(sum, reference_half<Half>(std::abs(static_cast<float>(base[lane * 4 + 2]))));
                    sum = reference_half_add(sum, reference_half<Half>(std::abs(static_cast<float>(base[lane * 4 + 3]))));
                    if (static_cast<float>(sum) > static_cast<float>(maximum)) maximum = sum;
                }
            }
            const Half scale = reference_half<Half>(static_cast<float>(maximum) / 127.0f);

            Half bias = reference_half<Half>(0.0f);
            const Half inverse_scale = static_cast<float>(scale) == 0.0f
                ? reference_half<Half>(0.0f)
                : reference_half<Half>(1.0 / static_cast<double>(scale));

            for (int block32 = 0; block32 < 2; ++block32)
            {
                const Half* base = base_group + block32 * 32;
                Half tables[kReferenceLutEntries][8];
                for (int index = 1; index < kReferenceLutEntries; index += 2)
                {
                    for (int lane = 0; lane < 8; ++lane)
                    {
                        Half value = base[lane * 4];
                        value = (index & 2) ? reference_half_add(value, base[lane * 4 + 1]) : reference_half_sub(value, base[lane * 4 + 1]);
                        value = (index & 4) ? reference_half_add(value, base[lane * 4 + 2]) : reference_half_sub(value, base[lane * 4 + 2]);
                        value = (index & 8) ? reference_half_add(value, base[lane * 4 + 3]) : reference_half_sub(value, base[lane * 4 + 3]);
                        tables[index][lane] = value;
                    }
                }
                for (int index = 0; index < kReferenceLutEntries; index += 2)
                    for (int lane = 0; lane < 8; ++lane)
                        tables[index][lane] = reference_half<Half>(-static_cast<float>(tables[kReferenceLutEntries - 1 - index][lane]));

                float block_bias = 0.0f;
                for (int lane = 0; lane < 8; ++lane) block_bias += static_cast<float>(tables[0][lane]);
                bias = reference_half<Half>(static_cast<float>(bias) + block_bias);

                for (int lane = 0; lane < 8; ++lane)
                {
                    const int kg = group * (kReferenceActivationGroupSize / kReferenceLutGroupSize) + block32 * 8 + lane;
                    for (int index = 0; index < kReferenceLutEntries; ++index)
                    {
                        const Half scaled = reference_half_mul(tables[index][lane], inverse_scale);
                        const int rounded = round_to_nearest_even_int(static_cast<float>(scaled));
                        const int clamped = std::max(-128, std::min(127, rounded));
                        reference.quantized_luts[(static_cast<size_t>(row) * k_groups + kg) * kReferenceLutEntries + index] = static_cast<int8_t>(clamped);
                    }
                }
            }

            reference.lut_scales[static_cast<size_t>(row) * activation_groups + group] = scale;
            reference.lut_biases[static_cast<size_t>(row) * activation_groups + group] = bias;
        }
    }
    return reference;
}

template <typename Half>
bool bitwise_equal(const std::vector<Half>& lhs, const std::vector<Half>& rhs)
{
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i)
        if (std::memcmp(&lhs[i], &rhs[i], sizeof(Half)) != 0) return false;
    return true;
}

template <typename Half>
bool lut_reference_matches(const OfficialLutReference<Half>& reference, const std::vector<int8_t>& quantized_luts,
                           const std::vector<Half>& lut_scales, const std::vector<Half>& lut_biases)
{
    return reference.quantized_luts == quantized_luts
        && bitwise_equal(reference.lut_scales, lut_scales)
        && bitwise_equal(reference.lut_biases, lut_biases);
}

template <int Bits, typename Half>
void official_tmac_reference(int m, int k, int n, const std::vector<uint8_t>& qweights, const std::vector<Half>& activations,
                             const std::vector<Half>& scales, const OfficialLutReference<Half>& lut_reference,
                             std::vector<Half>& output)
{
    static_assert(Bits >= 2 && Bits <= 4);
    if (k % kWeightScaleGroupSize != 0 || k % kReferenceActivationGroupSize != 0) throw std::invalid_argument("Unsupported K for T-MAC reference");
    const int k_groups = k / kReferenceLutGroupSize;
    const int k_tiles = k_groups / kReferenceKFactor;
    const int activation_groups = k / kReferenceActivationGroupSize;
    const int weight_groups = k / kWeightScaleGroupSize;
    if (qweights.size() != static_cast<size_t>(m) * k || activations.size() != static_cast<size_t>(n) * k
        || scales.size() != static_cast<size_t>(m) * weight_groups)
        throw std::invalid_argument("T-MAC reference input size mismatch");

    output.resize(static_cast<size_t>(n) * m);
    for (int row = 0; row < n; ++row)
    {
        for (int logical_row = 0; logical_row < m; ++logical_row)
        {
            Half plane_sums[Bits];
            for (int bit = 0; bit < Bits; ++bit) plane_sums[bit] = reference_half<Half>(0.0f);

            for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
            {
                const int kg_begin = k_tile * kReferenceKFactor;
                const int activation_group = (kg_begin * kReferenceLutGroupSize) / kReferenceActivationGroupSize;
                const int weight_group = (kg_begin * kReferenceLutGroupSize) / kWeightScaleGroupSize;
                const Half lut_scale = lut_reference.lut_scales[static_cast<size_t>(row) * activation_groups + activation_group];
                const Half lut_bias = lut_reference.lut_biases[static_cast<size_t>(row) * activation_groups + activation_group];
                const Half weight_scale = scales[static_cast<size_t>(logical_row) * weight_groups + weight_group];

                for (int bit = 0; bit < Bits; ++bit)
                {
                    int lookup_sum = 0;
                    for (int k_inner = 0; k_inner < kReferenceKFactor; ++k_inner)
                    {
                        const int kg = kg_begin + k_inner;
                        uint8_t index = 0;
                        const size_t qbase = static_cast<size_t>(logical_row) * k + kg * kReferenceLutGroupSize;
                        for (int g = 0; g < kReferenceLutGroupSize; ++g)
                            index |= static_cast<uint8_t>(((qweights[qbase + g] >> bit) & 1U) << g);
                        lookup_sum += lut_reference.quantized_luts[(static_cast<size_t>(row) * k_groups + kg) * kReferenceLutEntries + index];
                    }
                    const Half lookup = reference_half<Half>(static_cast<float>(lookup_sum));
                    const Half reconstructed = bit == 0
                        ? reference_half_fma(lut_bias, lookup, lut_scale)
                        : reference_half_mul(lookup, lut_scale);
                    plane_sums[bit] = reference_half_fma(plane_sums[bit], reconstructed, weight_scale);
                }
            }

            float result = 0.5f * static_cast<float>(plane_sums[0]) + static_cast<float>(plane_sums[1]);
            if constexpr (Bits >= 3) result += 2.0f * static_cast<float>(plane_sums[2]);
            if constexpr (Bits >= 4) result += 4.0f * static_cast<float>(plane_sums[3]);
            output[static_cast<size_t>(row) * m + logical_row] = reference_half<Half>(result);
        }
    }
}

struct Accuracy
{
    double nmse = 0.0;
    double max_abs_error = 0.0;
    size_t non_finite = 0;
};

template <typename ActualHalf, typename ReferenceHalf>
Accuracy compare_outputs(const std::vector<ActualHalf>& actual, const std::vector<ReferenceHalf>& reference)
{
    if (actual.size() != reference.size()) throw std::invalid_argument("Output size mismatch");
    long double squared_error = 0.0L;
    long double squared_actual = 0.0L;
    double max_abs_error = 0.0;
    size_t non_finite = 0;
    for (size_t i = 0; i < actual.size(); ++i)
    {
        const double a = static_cast<double>(static_cast<float>(actual[i]));
        const double r = static_cast<double>(static_cast<float>(reference[i]));
        if (!std::isfinite(a) || !std::isfinite(r))
        {
            ++non_finite;
            continue;
        }
        const double error = a - r;
        squared_error += static_cast<long double>(error) * error;
        squared_actual += static_cast<long double>(a) * a;
        max_abs_error = std::max(max_abs_error, std::abs(error));
    }
    Accuracy result;
    result.non_finite = non_finite;
    result.max_abs_error = max_abs_error;
    result.nmse = squared_actual == 0.0L ? (squared_error == 0.0L ? 0.0 : std::numeric_limits<double>::infinity())
                                        : static_cast<double>(squared_error / squared_actual);
    return result;
}

template <typename Half>
double checksum(const std::vector<Half>& output)
{
    long double sum = 0.0L;
    for (Half value : output) sum += static_cast<long double>(static_cast<float>(value));
    return static_cast<double>(sum);
}

inline double median(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 0 ? 0.5 * (values[middle - 1] + values[middle]) : values[middle];
}

inline double percentile(std::vector<double> values, double ratio)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(std::min<double>(values.size() - 1, std::ceil(ratio * values.size()) - 1));
    return values[index];
}

struct BenchmarkStats
{
    double median_ms = 0.0;
    double p90_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

template <typename Function>
BenchmarkStats measure(const Shape& shape, Function&& function)
{
    for (int i = 0; i < shape.warmup; ++i) function();
    std::vector<double> latencies;
    latencies.reserve(shape.samples);
    for (int sample = 0; sample < shape.samples; ++sample)
    {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < shape.repeat; ++i) function();
        const auto end = std::chrono::steady_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(end - start).count() / shape.repeat);
    }
    return {median(latencies), percentile(latencies, 0.90), *std::min_element(latencies.begin(), latencies.end()), *std::max_element(latencies.begin(), latencies.end())};
}

inline double dense_equivalent_gops(const Shape& shape, double latency_ms)
{
    const double operations = 2.0 * static_cast<double>(shape.m) * shape.k * shape.n;
    return operations / (latency_ms * 1.0e6);
}

inline const char* verification_case_name(int mode)
{
    switch (mode)
    {
        case 0: return "Random";
        case 1: return "ZeroAct";
        case 2: return "AltAct";
        case 3: return "MinWeight";
        case 4: return "MaxWeight";
        case 5: return "UnitScale";
        case 6: return "AltWeight";
        case 7: return "PosAct";
        case 8: return "NegAct";
        case 9: return "TinyScale";
        case 10: return "NegScale";
        case 11: return "AltScale";
        default: return "Unknown";
    }
}

inline void print_separator(char fill = '=')
{
    std::cout << std::string(132, fill) << '\n';
}

inline void print_verification_header(const char* implementation, int bits, bool has_schedule, bool has_kernel_match)
{
    print_separator();
    std::cout << implementation << " W" << bits << "A16 Correctness Verification\n";
    print_separator('-');
    std::cout << std::left << std::setw(7) << "M" << std::setw(7) << "K" << std::setw(7) << "N" << std::setw(13) << "Case";
    if (has_schedule) std::cout << std::right << std::setw(6) << "BM" << std::setw(6) << "BN" << std::setw(6) << "KF" << std::setw(10) << "Packing" << std::setw(8) << "LUT";
    if (has_kernel_match) std::cout << std::setw(14) << "KernelDiff";
    std::cout << std::right << std::setw(16) << "NMSE" << std::setw(16) << "MaxAbsErr" << std::setw(12) << "NonFinite" << std::setw(10) << "Result" << '\n';
    print_separator('-');
}

inline void print_verification_row(int m, int k, int n, const char* case_name, const Accuracy& accuracy, bool pass, int bm = 0, int bn = 0,
                                   int kfactor = 0, bool packing_ok = true, bool lut_ok = true, long long kernel_mismatches = -1)
{
    std::cout << std::left << std::setw(7) << m << std::setw(7) << k << std::setw(7) << n << std::setw(13) << case_name;
    if (bm > 0)
    {
        std::cout << std::right << std::setw(6) << bm << std::setw(6) << bn << std::setw(6) << kfactor << std::setw(10) << (packing_ok ? "PASS" : "FAIL") << std::setw(8) << (lut_ok ? "PASS" : "FAIL");
    }
    if (kernel_mismatches >= 0) std::cout << std::setw(14) << kernel_mismatches;
    std::cout << std::right << std::scientific << std::setprecision(3) << std::setw(16) << accuracy.nmse << std::setw(16) << accuracy.max_abs_error
              << std::fixed << std::setw(12) << accuracy.non_finite << std::setw(10) << (pass ? "PASS" : "FAIL") << '\n';
}

inline void print_verification_row(int m, int k, int n, int mode, const Accuracy& accuracy, bool pass, int bm = 0, int bn = 0,
                                   int kfactor = 0, bool packing_ok = true, bool lut_ok = true, long long kernel_mismatches = -1)
{
    print_verification_row(m, k, n, verification_case_name(mode), accuracy, pass, bm, bn, kfactor, packing_ok, lut_ok, kernel_mismatches);
}

inline void print_verification_footer(bool pass)
{
    print_separator('-');
    std::cout << "Verification result: " << (pass ? "PASS" : "FAIL") << '\n';
    print_separator();
}

inline void print_benchmark_header(const char* implementation, int bits)
{
    std::cout << '\n';
    print_separator();
    std::cout << implementation << " W" << bits << "A16 Benchmark\n";
    std::cout << "Matrix: W[MxK], X[NxK], Y[NxM] = X * W^T | N = 1, 8, 32, 128\n";
    std::cout << "Median and P90 are computed across independent samples. Each sample reports average latency across its repeat count.\n";
    if (std::string(implementation).find("TMAC") == 0) std::cout << "Timing includes activation-LUT construction and compute; offline weight/scale packing is excluded.\n";
    else std::cout << "Timing includes only the direct dense WnA16 scalar kernel; input generation and verification are excluded.\n";
    print_separator('-');
    std::cout << std::left << std::setw(7) << "Kind" << std::right << std::setw(7) << "M" << std::setw(7) << "K" << std::setw(7) << "N"
              << std::setw(7) << "BM" << std::setw(7) << "BN" << std::setw(6) << "KF" << std::setw(7) << "Warm" << std::setw(7) << "Rep"
              << std::setw(8) << "Sample" << std::setw(14) << "Median(ms)" << std::setw(12) << "P90(ms)" << std::setw(12) << "Eq.GOPS"
              << std::setw(15) << "Checksum" << '\n';
    print_separator('-');
}

inline void print_benchmark_result(const Shape& shape, const BenchmarkStats& stats, double checksum_value, int bm = 0, int bn = 0, int kfactor = 0)
{
    std::cout << std::left << std::setw(7) << shape.kind() << std::right << std::setw(7) << shape.m << std::setw(7) << shape.k << std::setw(7) << shape.n;
    if (bm > 0) std::cout << std::setw(7) << bm << std::setw(7) << bn << std::setw(6) << kfactor;
    else std::cout << std::setw(7) << "-" << std::setw(7) << "-" << std::setw(6) << "-";
    std::cout << std::setw(7) << shape.warmup << std::setw(7) << shape.repeat << std::setw(8) << shape.samples << std::fixed << std::setprecision(4)
              << std::setw(14) << stats.median_ms << std::setw(12) << stats.p90_ms << std::setw(12) << dense_equivalent_gops(shape, stats.median_ms)
              << std::scientific << std::setprecision(4) << std::setw(15) << checksum_value << std::fixed << '\n';
}

inline void print_benchmark_footer()
{
    print_separator();
}

} // namespace bench
