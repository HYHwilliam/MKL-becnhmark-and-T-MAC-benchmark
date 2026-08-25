// Single-thread freeze candidate: official temporal-first NEON flow, pairwise widening-add, and strengthened independent verification.
#include "benchmark_common.h"
#include "tmac_layout.h"

#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "This file requires AArch64 NEON."
#endif
#if !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error "This file requires ARMv8.2-A FP16 vector arithmetic."
#endif

namespace
{
using namespace tmac_layout;
using fp16 = float16_t;

inline float to_float(fp16 value) { return static_cast<float>(value); }
inline fp16 to_half(float value) { return static_cast<fp16>(value); }

inline uint16_t half_bits(fp16 value)
{
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
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
    const float16x8x4_t values = vld4q_f16(activations);
    float16x8_t sum = vaddq_f16(vabsq_f16(values.val[0]), vabsq_f16(values.val[1]));
    sum = vaddq_f16(sum, vabsq_f16(values.val[2]));
    sum = vaddq_f16(sum, vabsq_f16(values.val[3]));
    const fp16 candidate = to_half(to_float(vmaxvq_f16(sum)) / 127.0f);
    if (to_float(candidate) > to_float(*scale)) *scale = candidate;
}

inline float official_horizontal_sum(float16x8_t value)
{
    float sum = 0.0f;
    for (int lane = 0; lane < 8; ++lane) sum += to_float(value[lane]);
    return sum;
}

inline void construct_quantized_luts_64(const fp16* activations, fp16 lut_scale, int8_t* destination, fp16* lut_bias)
{
    float16x8_t tables[kLutEntryCount];
    fp16 bias = to_half(0.0f);
    const fp16 inverse_scale = to_float(lut_scale) == 0.0f ? to_half(0.0f) : static_cast<fp16>(1.0 / static_cast<double>(lut_scale));

    for (int block32 = 0; block32 < 2; ++block32)
    {
        const float16x8x4_t values = vld4q_f16(activations + block32 * 32);
        for (int index = 1; index < kLutEntryCount; index += 2)
        {
            tables[index] = values.val[0];
            tables[index] = (index & 2) ? vaddq_f16(tables[index], values.val[1]) : vsubq_f16(tables[index], values.val[1]);
            tables[index] = (index & 4) ? vaddq_f16(tables[index], values.val[2]) : vsubq_f16(tables[index], values.val[2]);
            tables[index] = (index & 8) ? vaddq_f16(tables[index], values.val[3]) : vsubq_f16(tables[index], values.val[3]);
        }
        for (int index = 0; index < kLutEntryCount; index += 2) tables[index] = vnegq_f16(tables[kLutEntryCount - 1 - index]);
        bias = to_half(to_float(bias) + official_horizontal_sum(tables[0]));

        alignas(16) int8_t quantized[kLutEntryCount][8];
        for (int index = 0; index < kLutEntryCount; ++index)
        {
            const float16x8_t scaled = vmulq_n_f16(tables[index], inverse_scale);
            vst1_s8(quantized[index], vqmovn_s16(vcvtnq_s16_f16(scaled)));
        }
        for (int lane = 0; lane < 8; ++lane)
            for (int index = 0; index < kLutEntryCount; ++index)
                destination[(block32 * 8 + lane) * kLutEntryCount + index] = quantized[index][lane];
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

inline float16x8_t reconstruct_lookup(float16x8_t lookup_sum, fp16 lut_scale, fp16 lut_bias, bool add_bias)
{
    return add_bias ? vfmaq_n_f16(vdupq_n_f16(lut_bias), lookup_sum, lut_scale) : vmulq_n_f16(lookup_sum, lut_scale);
}

template <int Bits>
__attribute__((noinline)) void tmac_neon(const PackedWeights& weights, const PackedScales<fp16>& scales, int n,
                                         const fp16* activations, fp16* output, Workspace& workspace)
{
    const int logical_m = weights.logical_m;
    const int k = weights.k;
    const int bm = weights.schedule.bm;
    const int bn = weights.schedule.bn;
    constexpr int kfactor = kDefaultKFactor;
    if (weights.schedule.kfactor != kfactor)
        throw std::invalid_argument("NEON kernel requires official kfactor=16 for g=4, act_group=64, group_size=128");
    const int m_tiles = logical_m * Bits / bm;
    const int logical_rows_per_tile = bm / Bits;
    const int blocks32 = bm / kExpandedRowsPerVector;
    const int k_groups = k / kLutGroupSize;
    const int k_tiles = k_groups / kfactor;
    const int activation_groups = k / kActivationGroupSize;
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0f);

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
                    int8x16_t lut_registers[kDefaultKFactor];
                    for (int k_inner = 0; k_inner < kfactor; ++k_inner)
                        lut_registers[k_inner] = vld1q_s8(qlut_base + static_cast<size_t>(k_inner) * kLutEntryCount);

                    for (int block32 = 0; block32 < blocks32; ++block32)
                    {
                        const uint8_t* packed_base = weights.bytes.data() + packed_weight_offset(weights, m_tile, k_tile, block32, 0);
                        const uint8x16_t packed_indices0 = vld1q_u8(packed_base);
                        const uint8x16_t packed_indices1 = vld1q_u8(packed_base + kPackedByteLanes);
                        const int8x16_t bottom0 = vqtbl1q_s8(lut_registers[0], vandq_u8(packed_indices0, nibble_mask));
                        const int8x16_t bottom1 = vqtbl1q_s8(lut_registers[1], vandq_u8(packed_indices1, nibble_mask));
                        const int8x16_t top0 = vqtbl1q_s8(lut_registers[0], vshrq_n_u8(packed_indices0, 4));
                        const int8x16_t top1 = vqtbl1q_s8(lut_registers[1], vshrq_n_u8(packed_indices1, 4));

                        int16x8_t bottom_low = vaddl_s8(vget_low_s8(bottom0), vget_low_s8(bottom1));
                        int16x8_t bottom_high = vaddl_high_s8(bottom0, bottom1);
                        int16x8_t top_low = vaddl_s8(vget_low_s8(top0), vget_low_s8(top1));
                        int16x8_t top_high = vaddl_high_s8(top0, top1);

                        for (int k_inner = 2; k_inner < kfactor; k_inner += 2)
                        {
                            const uint8x16_t packed_indices_even = vld1q_u8(packed_base + static_cast<size_t>(k_inner) * kPackedByteLanes);
                            const uint8x16_t packed_indices_odd = vld1q_u8(packed_base + static_cast<size_t>(k_inner + 1) * kPackedByteLanes);

                            const int8x16_t bottom_even = vqtbl1q_s8(lut_registers[k_inner], vandq_u8(packed_indices_even, nibble_mask));
                            const int8x16_t bottom_odd = vqtbl1q_s8(lut_registers[k_inner + 1], vandq_u8(packed_indices_odd, nibble_mask));
                            const int8x16_t top_even = vqtbl1q_s8(lut_registers[k_inner], vshrq_n_u8(packed_indices_even, 4));
                            const int8x16_t top_odd = vqtbl1q_s8(lut_registers[k_inner + 1], vshrq_n_u8(packed_indices_odd, 4));

                            bottom_low = vaddq_s16(bottom_low, vaddl_s8(vget_low_s8(bottom_even), vget_low_s8(bottom_odd)));
                            bottom_high = vaddq_s16(bottom_high, vaddl_high_s8(bottom_even, bottom_odd));
                            top_low = vaddq_s16(top_low, vaddl_s8(vget_low_s8(top_even), vget_low_s8(top_odd)));
                            top_high = vaddq_s16(top_high, vaddl_high_s8(top_even, top_odd));
                        }

                        const float16x8_t lookup_sums[4] = {vcvtq_f16_s16(bottom_low), vcvtq_f16_s16(bottom_high),
                                                           vcvtq_f16_s16(top_low), vcvtq_f16_s16(top_high)};
                        const fp16 lut_scale = workspace.lut_scales[static_cast<size_t>(row) * activation_groups + activation_group];
                        const fp16 lut_bias = workspace.lut_biases[static_cast<size_t>(row) * activation_groups + activation_group];
                        for (int group8 = 0; group8 < 4; ++group8)
                        {
                            const int expanded_group8 = block32 * 4 + group8;
                            const int bit_plane = expanded_group8 % Bits;
                            const int logical_block8 = expanded_group8 / Bits;
                            const float16x8_t weight_scale = vld1q_f16(packed_scale_ptr(scales, m_tile, weight_group, logical_block8 * kFp16OutputLanes));
                            const float16x8_t reconstructed = reconstruct_lookup(lookup_sums[group8], lut_scale, lut_bias, add_bias && bit_plane == 0);
                            fp16* destination = row_accumulator + expanded_group8 * kFp16OutputLanes;
                            const float16x8_t previous = vld1q_f16(destination);
                            vst1q_f16(destination, vfmaq_f16(previous, reconstructed, weight_scale));
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
                    const float16x8_t p0 = vld1q_f16(planes);
                    float32x4_t low = vmulq_n_f32(vcvt_f32_f16(vget_low_f16(p0)), 0.5f);
                    float32x4_t high = vmulq_n_f32(vcvt_f32_f16(vget_high_f16(p0)), 0.5f);
                    const float16x8_t p1 = vld1q_f16(planes + kFp16OutputLanes);
                    low = vaddq_f32(low, vcvt_f32_f16(vget_low_f16(p1)));
                    high = vaddq_f32(high, vcvt_f32_f16(vget_high_f16(p1)));
                    if constexpr (Bits >= 3)
                    {
                        const float16x8_t p2 = vld1q_f16(planes + 2 * kFp16OutputLanes);
                        low = vfmaq_n_f32(low, vcvt_f32_f16(vget_low_f16(p2)), 2.0f);
                        high = vfmaq_n_f32(high, vcvt_f32_f16(vget_high_f16(p2)), 2.0f);
                    }
                    if constexpr (Bits >= 4)
                    {
                        const float16x8_t p3 = vld1q_f16(planes + 3 * kFp16OutputLanes);
                        low = vfmaq_n_f32(low, vcvt_f32_f16(vget_low_f16(p3)), 4.0f);
                        high = vfmaq_n_f32(high, vcvt_f32_f16(vget_high_f16(p3)), 4.0f);
                    }
                    vst1q_f16(output_tile + block8 * kFp16OutputLanes, vcombine_f16(vcvt_f16_f32(low), vcvt_f16_f32(high)));
                }
            }
        }
    }
}

inline fp16 reference_neon_half_mul(fp16 value, fp16 scale)
{
    const float16x8_t result = vmulq_n_f16(vdupq_n_f16(value), scale);
    return vgetq_lane_f16(result, 0);
}

inline fp16 reference_neon_lookup_fma(fp16 bias, fp16 lookup, fp16 scale)
{
    const float16x8_t result = vfmaq_n_f16(vdupq_n_f16(bias), vdupq_n_f16(lookup), scale);
    return vgetq_lane_f16(result, 0);
}

inline fp16 reference_neon_accumulate_fma(fp16 acc, fp16 value, fp16 scale)
{
    const float16x8_t result = vfmaq_f16(vdupq_n_f16(acc), vdupq_n_f16(value), vdupq_n_f16(scale));
    return vgetq_lane_f16(result, 0);
}

template <int Bits>
inline fp16 reference_neon_reduce(const fp16 (&planes)[Bits])
{
    float32x4_t result = vmulq_n_f32(vdupq_n_f32(to_float(planes[0])), 0.5f);
    result = vaddq_f32(result, vdupq_n_f32(to_float(planes[1])));
    if constexpr (Bits >= 3) result = vfmaq_n_f32(result, vdupq_n_f32(to_float(planes[2])), 2.0f);
    if constexpr (Bits >= 4) result = vfmaq_n_f32(result, vdupq_n_f32(to_float(planes[3])), 4.0f);
    return vget_lane_f16(vcvt_f16_f32(result), 0);
}

template <int Bits>
void independent_neon_reference(int m, int k, int n, const std::vector<uint8_t>& qweights,
                                const std::vector<fp16>& activations, const std::vector<fp16>& scales,
                                const bench::OfficialLutReference<fp16>& lut_reference,
                                std::vector<fp16>& output)
{
    static_assert(Bits >= 2 && Bits <= 4);
    if (k % kWeightScaleGroupSize != 0 || k % kActivationGroupSize != 0)
        throw std::invalid_argument("Unsupported K for NEON reference");

    const int k_groups = k / kLutGroupSize;
    const int k_tiles = k_groups / kDefaultKFactor;
    const int activation_groups = k / kActivationGroupSize;
    const int weight_groups = k / kWeightScaleGroupSize;
    if (qweights.size() != static_cast<size_t>(m) * k || activations.size() != static_cast<size_t>(n) * k
        || scales.size() != static_cast<size_t>(m) * weight_groups)
        throw std::invalid_argument("NEON reference input size mismatch");

    output.resize(static_cast<size_t>(n) * m);
    for (int row = 0; row < n; ++row)
    {
        for (int logical_row = 0; logical_row < m; ++logical_row)
        {
            fp16 plane_sums[Bits];
            for (int bit = 0; bit < Bits; ++bit) plane_sums[bit] = to_half(0.0f);

            for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
            {
                const int kg_begin = k_tile * kDefaultKFactor;
                const int activation_group = (kg_begin * kLutGroupSize) / kActivationGroupSize;
                const int weight_group = (kg_begin * kLutGroupSize) / kWeightScaleGroupSize;
                const fp16 lut_scale = lut_reference.lut_scales[static_cast<size_t>(row) * activation_groups + activation_group];
                const fp16 lut_bias = lut_reference.lut_biases[static_cast<size_t>(row) * activation_groups + activation_group];
                const fp16 weight_scale = scales[static_cast<size_t>(logical_row) * weight_groups + weight_group];

                for (int bit = 0; bit < Bits; ++bit)
                {
                    int lookup_sum = 0;
                    for (int k_inner = 0; k_inner < kDefaultKFactor; ++k_inner)
                    {
                        const int kg = kg_begin + k_inner;
                        uint8_t index = 0;
                        const size_t qbase = static_cast<size_t>(logical_row) * k + kg * kLutGroupSize;
                        for (int g = 0; g < kLutGroupSize; ++g)
                            index |= static_cast<uint8_t>(((qweights[qbase + g] >> bit) & 1U) << g);
                        lookup_sum += lut_reference.quantized_luts[(static_cast<size_t>(row) * k_groups + kg) * kLutEntryCount + index];
                    }

                    const fp16 lookup = to_half(static_cast<float>(lookup_sum));
                    const fp16 reconstructed = bit == 0
                        ? reference_neon_lookup_fma(lut_bias, lookup, lut_scale)
                        : reference_neon_half_mul(lookup, lut_scale);
                    plane_sums[bit] = reference_neon_accumulate_fma(plane_sums[bit], reconstructed, weight_scale);
                }
            }

            output[static_cast<size_t>(row) * m + logical_row] = reference_neon_reduce(plane_sums);
        }
    }
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
    tmac_neon<Bits>(packed_weights, packed_scales, n, activations.data(), actual.data(), workspace);

    const bench::OfficialLutReference<fp16> independent_lut = bench::build_official_lut_reference(n, k, activations);
    const bool lut_ok = bench::lut_reference_matches(independent_lut, workspace.quantized_luts, workspace.lut_scales, workspace.lut_biases);
    independent_neon_reference<Bits>(m, k, n, raw.qweights, activations, scales, independent_lut, independent_output);
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
    bench::print_verification_header("TMAC_NEON", Bits, true, true);
    bool pass = true;
    for (int n : {1, 8, 32, 128}) pass &= verify_case<Bits>(256, 256, n, 0);
    pass &= verify_case<Bits>(256, 128, 3, 0, nullptr, "K128Edge");
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
        tmac_neon<Bits>(packed_weights, packed_scales, shape.n, activations.data(), output.data(), workspace);
    });
    bench::print_benchmark_result(shape, stats, bench::checksum(output), schedule.bm, schedule.bn, schedule.kfactor);
}

template <int Bits>
int run(const bench::Options& options)
{
    if (!verify_suite<Bits>()) return 1;
    if (options.verify_only) return 0;
    bench::print_benchmark_header("TMAC_NEON", Bits);
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
