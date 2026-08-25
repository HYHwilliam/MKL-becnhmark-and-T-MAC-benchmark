// Audit revision: official temporal-first reduction and independent layout verification.
#include "benchmark_common.h"
#include "tmac_layout.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#if defined(__FLT16_MANT_DIG__)
using fp16 = _Float16;
#else
#error "This compiler must support _Float16 for the A16 benchmark."
#endif

namespace
{
using namespace tmac_layout;

inline float to_float(fp16 value) { return static_cast<float>(value); }
inline fp16 to_half(float value) { return static_cast<fp16>(value); }
inline fp16 half_add(fp16 a, fp16 b) { return to_half(to_float(a) + to_float(b)); }
inline fp16 half_sub(fp16 a, fp16 b) { return to_half(to_float(a) - to_float(b)); }
inline fp16 half_mul(fp16 a, fp16 b) { return to_half(to_float(a) * to_float(b)); }
inline fp16 half_madd(fp16 acc, fp16 a, fp16 b) { return to_half(std::fma(to_float(a), to_float(b), to_float(acc))); }

inline int8_t quantize_int8_rne(fp16 value)
{
    const int rounded = static_cast<int>(std::nearbyint(to_float(value)));
    return static_cast<int8_t>(std::max(-128, std::min(127, rounded)));
}

struct Workspace
{
    std::vector<int8_t> quantized_luts;
    std::vector<fp16> lut_scales;
    std::vector<fp16> lut_biases;
    std::vector<fp16> expanded;
};

inline void update_lut_scale_from_32(fp16* scale, const fp16* activations)
{
    fp16 maximum = to_half(0.0f);
    for (int lane = 0; lane < 8; ++lane)
    {
        fp16 sum = half_add(to_half(std::abs(to_float(activations[lane * 4]))), to_half(std::abs(to_float(activations[lane * 4 + 1]))));
        sum = half_add(sum, to_half(std::abs(to_float(activations[lane * 4 + 2]))));
        sum = half_add(sum, to_half(std::abs(to_float(activations[lane * 4 + 3]))));
        if (lane == 0 || to_float(sum) > to_float(maximum)) maximum = sum;
    }
    const fp16 candidate = to_half(to_float(maximum) / 127.0f);
    if (to_float(candidate) > to_float(*scale)) *scale = candidate;
}

inline void construct_quantized_luts_64(const fp16* activations, fp16 lut_scale, int8_t* destination, fp16* lut_bias)
{
    fp16 tables[kLutEntryCount][8];
    fp16 bias = to_half(0.0f);
    const fp16 inverse_scale = to_float(lut_scale) == 0.0f ? to_half(0.0f) : static_cast<fp16>(1.0 / static_cast<double>(lut_scale));

    for (int block32 = 0; block32 < 2; ++block32)
    {
        const fp16* base = activations + block32 * 32;
        for (int index = 1; index < kLutEntryCount; index += 2)
        {
            for (int lane = 0; lane < 8; ++lane)
            {
                fp16 value = base[lane * 4];
                value = (index & 2) ? half_add(value, base[lane * 4 + 1]) : half_sub(value, base[lane * 4 + 1]);
                value = (index & 4) ? half_add(value, base[lane * 4 + 2]) : half_sub(value, base[lane * 4 + 2]);
                value = (index & 8) ? half_add(value, base[lane * 4 + 3]) : half_sub(value, base[lane * 4 + 3]);
                tables[index][lane] = value;
            }
        }
        for (int index = 0; index < kLutEntryCount; index += 2)
            for (int lane = 0; lane < 8; ++lane) tables[index][lane] = to_half(-to_float(tables[kLutEntryCount - 1 - index][lane]));

        float horizontal_sum = 0.0f;
        for (int lane = 0; lane < 8; ++lane) horizontal_sum += to_float(tables[0][lane]);
        bias = to_half(to_float(bias) + horizontal_sum);

        for (int lane = 0; lane < 8; ++lane)
            for (int index = 0; index < kLutEntryCount; ++index)
                destination[(block32 * 8 + lane) * kLutEntryCount + index] = quantize_int8_rne(half_mul(tables[index][lane], inverse_scale));
    }
    *lut_bias = bias;
}

inline void build_activation_luts(int n, int k, const fp16* activations, Workspace& workspace)
{
    const int k_groups = k / kLutGroupSize;
    const int activation_groups = k / kActivationGroupSize;
    workspace.quantized_luts.resize(static_cast<size_t>(n) * k_groups * kLutEntryCount);
    workspace.lut_scales.resize(static_cast<size_t>(n) * activation_groups);
    workspace.lut_biases.resize(static_cast<size_t>(n) * activation_groups);

    for (int row = 0; row < n; ++row)
    {
        for (int group = 0; group < activation_groups; ++group)
        {
            const fp16* group_activations = activations + static_cast<size_t>(row) * k + group * kActivationGroupSize;
            fp16 scale = to_half(0.0f);
            update_lut_scale_from_32(&scale, group_activations);
            update_lut_scale_from_32(&scale, group_activations + 32);
            fp16 bias = to_half(0.0f);
            int8_t* qlut = workspace.quantized_luts.data() + (static_cast<size_t>(row) * k_groups + group * (kActivationGroupSize / kLutGroupSize)) * kLutEntryCount;
            construct_quantized_luts_64(group_activations, scale, qlut, &bias);
            workspace.lut_scales[static_cast<size_t>(row) * activation_groups + group] = scale;
            workspace.lut_biases[static_cast<size_t>(row) * activation_groups + group] = bias;
        }
    }
}

template <int Bits>
__attribute__((noinline)) void tmac_scalar(const PackedWeights& weights, const PackedScales<fp16>& scales, int n,
                                           const fp16* activations, fp16* output, Workspace& workspace)
{
    const int logical_m = weights.logical_m;
    const int k = weights.k;
    const int bm = weights.schedule.bm;
    const int bn = weights.schedule.bn;
    const int kfactor = weights.schedule.kfactor;
    const int m_tiles = logical_m * Bits / bm;
    const int logical_rows_per_tile = bm / Bits;
    const int blocks32 = bm / kExpandedRowsPerVector;
    const int k_groups = k / kLutGroupSize;
    const int k_tiles = k_groups / kfactor;
    const int activation_groups = k / kActivationGroupSize;

    build_activation_luts(n, k, activations, workspace);
    workspace.expanded.resize(static_cast<size_t>(bn) * bm);

    for (int no = 0; no < n; no += bn)
    {
        const int n_size = std::min(bn, n - no);
        for (int m_tile = 0; m_tile < m_tiles; ++m_tile)
        {
            fp16* tile_accumulator = workspace.expanded.data();
            std::fill(tile_accumulator, tile_accumulator + static_cast<size_t>(n_size) * bm, to_half(0.0f));

            for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
            {
                const int kg_begin = k_tile * kfactor;
                const int activation_group = (kg_begin * kLutGroupSize) / kActivationGroupSize;
                const int weight_group = (kg_begin * kLutGroupSize) / kWeightScaleGroupSize;
                const bool add_bias = ((kg_begin * kLutGroupSize) % kActivationGroupSize) == 0;

                for (int ni = 0; ni < n_size; ++ni)
                {
                    const int row = no + ni;
                    fp16* row_accumulator = tile_accumulator + static_cast<size_t>(ni) * bm;
                    const int8_t* qlut_base = workspace.quantized_luts.data() + (static_cast<size_t>(row) * k_groups + kg_begin) * kLutEntryCount;

                    for (int block32 = 0; block32 < blocks32; ++block32)
                    {
                        const uint8_t* packed_base = weights.bytes.data() + packed_weight_offset(weights, m_tile, k_tile, block32, 0);
                        int16_t bottom[16] = {};
                        int16_t top[16] = {};
                        for (int k_inner = 0; k_inner < kfactor; ++k_inner)
                        {
                            const int8_t* table = qlut_base + static_cast<size_t>(k_inner) * kLutEntryCount;
                            const uint8_t* packed_indices = packed_base + static_cast<size_t>(k_inner) * kPackedByteLanes;
                            for (int byte_lane = 0; byte_lane < kPackedByteLanes; ++byte_lane)
                            {
                                const uint8_t packed = packed_indices[byte_lane];
                                bottom[byte_lane] = static_cast<int16_t>(bottom[byte_lane] + table[packed & 0x0f]);
                                top[byte_lane] = static_cast<int16_t>(top[byte_lane] + table[(packed >> 4) & 0x0f]);
                            }
                        }

                        const fp16 lut_scale = workspace.lut_scales[static_cast<size_t>(row) * activation_groups + activation_group];
                        const fp16 lut_bias = workspace.lut_biases[static_cast<size_t>(row) * activation_groups + activation_group];
                        for (int group8 = 0; group8 < 4; ++group8)
                        {
                            const int expanded_group8 = block32 * 4 + group8;
                            const int bit_plane = expanded_group8 % Bits;
                            const int logical_block8 = expanded_group8 / Bits;
                            const fp16* weight_scales = packed_scale_ptr(scales, m_tile, weight_group, logical_block8 * kFp16OutputLanes);
                            fp16* destination = row_accumulator + expanded_group8 * kFp16OutputLanes;
                            for (int lane = 0; lane < kFp16OutputLanes; ++lane)
                            {
                                const int16_t lookup_sum = group8 == 0 ? bottom[lane] : group8 == 1 ? bottom[8 + lane] : group8 == 2 ? top[lane] : top[8 + lane];
                                const fp16 lookup = to_half(static_cast<float>(lookup_sum));
                                const fp16 reconstructed = add_bias && bit_plane == 0 ? half_madd(lut_bias, lookup, lut_scale) : half_mul(lookup, lut_scale);
                                destination[lane] = half_madd(destination[lane], reconstructed, weight_scales[lane]);
                            }
                        }
                    }
                }
            }

            for (int ni = 0; ni < n_size; ++ni)
            {
                const int row = no + ni;
                const fp16* row_accumulator = tile_accumulator + static_cast<size_t>(ni) * bm;
                fp16* output_tile = output + static_cast<size_t>(row) * logical_m + m_tile * logical_rows_per_tile;
                for (int block8 = 0; block8 < logical_rows_per_tile / kFp16OutputLanes; ++block8)
                {
                    const fp16* planes = row_accumulator + block8 * Bits * kFp16OutputLanes;
                    for (int lane = 0; lane < kFp16OutputLanes; ++lane)
                    {
                        float result = 0.5f * to_float(planes[lane]) + to_float(planes[kFp16OutputLanes + lane]);
                        if constexpr (Bits >= 3) result += 2.0f * to_float(planes[2 * kFp16OutputLanes + lane]);
                        if constexpr (Bits >= 4) result += 4.0f * to_float(planes[3 * kFp16OutputLanes + lane]);
                        output_tile[block8 * kFp16OutputLanes + lane] = to_half(result);
                    }
                }
            }
        }
    }
}

inline uint16_t half_bits(fp16 value)
{
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template <int Bits>
bool verify_case(int m, int k, int n, int mode, const Schedule* schedule_override = nullptr, const char* label = nullptr)
{
    bench::RawInput raw = bench::make_raw_input<Bits>(m, k, n, static_cast<uint32_t>(1000 + Bits * 100 + m + k + n + mode));
    std::vector<fp16> activations = bench::cast_to_half<fp16>(raw.activations);
    std::vector<fp16> scales = bench::cast_to_half<fp16>(raw.scales);
    if (mode == 1) std::fill(activations.begin(), activations.end(), to_half(0.0f));
    if (mode == 2) for (size_t i = 0; i < activations.size(); ++i) activations[i] = to_half((i & 1U) ? 1.0f : -1.0f);
    if (mode == 3) std::fill(raw.qweights.begin(), raw.qweights.end(), 0);
    if (mode == 4) std::fill(raw.qweights.begin(), raw.qweights.end(), static_cast<uint8_t>((1U << Bits) - 1U));
    if (mode == 5) std::fill(scales.begin(), scales.end(), to_half(1.0f));
    if (mode == 6) for (size_t i = 0; i < raw.qweights.size(); ++i) raw.qweights[i] = (i & 1U) ? static_cast<uint8_t>((1U << Bits) - 1U) : 0;
    if (mode == 7) std::fill(activations.begin(), activations.end(), to_half(0.5f));
    if (mode == 8) std::fill(activations.begin(), activations.end(), to_half(-0.5f));
    if (mode == 9) std::fill(scales.begin(), scales.end(), to_half(0.001f));
    if (mode == 10) std::fill(scales.begin(), scales.end(), to_half(-0.25f));
    if (mode == 11) for (size_t i = 0; i < scales.size(); ++i) scales[i] = to_half((i & 1U) ? -0.25f : 0.25f);

    const Schedule schedule = schedule_override ? *schedule_override : choose_single_thread_schedule<Bits>(m, n, k);
    const PackedWeights packed_weights = pack_weights_tmac<Bits>(raw.qweights, m, k, schedule);
    const PackedScales<fp16> packed_scales = pack_scales_tmac<Bits>(scales, m, k, schedule);
    validate_packed_operands<Bits>(packed_weights, packed_scales, n);
    const bool packing_ok = verify_weight_packing<Bits>(raw.qweights, packed_weights)
        && verify_weight_layout_against_official_transform<Bits>(raw.qweights, packed_weights)
        && verify_scale_layout_against_official_transform<Bits>(scales, packed_scales);

    Workspace workspace;
    std::vector<fp16> actual(static_cast<size_t>(n) * m);
    std::vector<fp16> independent_output;
    std::vector<fp16> dense;
    tmac_scalar<Bits>(packed_weights, packed_scales, n, activations.data(), actual.data(), workspace);

    const bench::OfficialLutReference<fp16> independent_lut = bench::build_official_lut_reference(n, k, activations);
    const bool lut_ok = bench::lut_reference_matches(independent_lut, workspace.quantized_luts, workspace.lut_scales, workspace.lut_biases);
    bench::official_tmac_reference<Bits>(m, k, n, raw.qweights, activations, scales, independent_lut, independent_output);
    bench::dense_reference<Bits>(m, k, n, raw.qweights, activations, scales, dense);

    size_t kernel_mismatches = 0;
    for (size_t i = 0; i < actual.size(); ++i) if (half_bits(actual[i]) != half_bits(independent_output[i])) ++kernel_mismatches;
    const bench::Accuracy accuracy = bench::compare_outputs(actual, dense);
    const bool pass = packing_ok && lut_ok && kernel_mismatches == 0 && accuracy.non_finite == 0 && accuracy.nmse <= bench::kTmacNmseLimit;
    bench::print_verification_row(m, k, n, label ? label : bench::verification_case_name(mode), accuracy, pass,
                                  schedule.bm, schedule.bn, schedule.kfactor, packing_ok, lut_ok, static_cast<long long>(kernel_mismatches));
    return pass;
}

template <int Bits>
bool verify_schedule_sweep()
{
    const int sweep_m = Bits == 3 ? 768 : Bits == 2 ? 2560 : 1280;
    constexpr int k = 256;
    bool pass = true;
    for (const Schedule& schedule : official_schedule_candidates<Bits>(sweep_m, 1, k)) pass &= verify_case<Bits>(sweep_m, k, 1, 0, &schedule, "BMSweep");

    constexpr int bn_m = 256;
    constexpr int bn_n = 128;
    const Schedule preferred = choose_single_thread_schedule<Bits>(bn_m, bn_n, k);
    for (const Schedule& schedule : official_schedule_candidates<Bits>(bn_m, bn_n, k))
        if (schedule.bm == preferred.bm && schedule.kfactor == preferred.kfactor) pass &= verify_case<Bits>(bn_m, k, bn_n, 0, &schedule, "BNSweep");
    return pass;
}

template <int Bits>
bool verify_suite()
{
    bench::print_verification_header("TMAC_SCALAR", Bits, true, true);
    bool pass = true;
    for (int n : {1, 8, 32, 128}) pass &= verify_case<Bits>(256, 256, n, 0);
    pass &= verify_case<Bits>(1024, 256, 3, 0);
    pass &= verify_case<Bits>(1024, 512, 8, 0);
    pass &= verify_case<Bits>(512, 1024, 16, 0);
    for (int mode = 1; mode <= 11; ++mode) pass &= verify_case<Bits>(256, 256, 8, mode);
    pass &= verify_schedule_sweep<Bits>();
    bench::print_verification_footer(pass);
    return pass;
}

template <int Bits>
void run_shape(const bench::Shape& shape)
{
    const bench::RawInput raw = bench::make_raw_input<Bits>(shape);
    const std::vector<fp16> activations = bench::cast_to_half<fp16>(raw.activations);
    const std::vector<fp16> scales = bench::cast_to_half<fp16>(raw.scales);
    const Schedule schedule = choose_single_thread_schedule<Bits>(shape.m, shape.n, shape.k);
    const PackedWeights packed_weights = pack_weights_tmac<Bits>(raw.qweights, shape.m, shape.k, schedule);
    const PackedScales<fp16> packed_scales = pack_scales_tmac<Bits>(scales, shape.m, shape.k, schedule);
    validate_packed_operands<Bits>(packed_weights, packed_scales, shape.n);
    std::vector<fp16> output(static_cast<size_t>(shape.n) * shape.m);
    Workspace workspace;

    const bench::BenchmarkStats stats = bench::measure(shape, [&]
    {
        tmac_scalar<Bits>(packed_weights, packed_scales, shape.n, activations.data(), output.data(), workspace);
    });
    bench::print_benchmark_result(shape, stats, bench::checksum(output), schedule.bm, schedule.bn, schedule.kfactor);
}

template <int Bits>
int run(const bench::Options& options)
{
    if (!verify_suite<Bits>()) return 1;
    if (options.verify_only) return 0;
    bench::print_benchmark_header("TMAC_SCALAR", Bits);
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
