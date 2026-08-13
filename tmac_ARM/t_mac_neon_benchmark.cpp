// Verification revision: ARM-FP16-explicit-FMA-v3
#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "This file requires AArch64 NEON."
#endif
#if !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error "This file requires ARMv8.2-A FP16 vector arithmetic (-march=armv8.2-a+fp16)."
#endif

namespace tmac_neon
{
using fp16 = float16_t;

// Microsoft/T-MAC calls this g. g=4 gives a 16-entry INT8 table that exactly
// fits one 128-bit NEON table register. This is not LUT-GEMM's mu=8 table.
constexpr int kLutGroupSize = 4;
constexpr int kLutEntryCount = 1 << kLutGroupSize;
constexpr int kPackedByteLanes = 16;
constexpr int kFp16OutputLanes = 8;
constexpr int kExpandedRowsPerVector = 32;
constexpr int kActivationGroupSize = 64;
constexpr int kWeightScaleGroupSize = 128;
constexpr int kLookupGroupsPerKernelTile = 16;

static_assert(kLutEntryCount == 16);
static_assert(kLookupGroupsPerKernelTile * kLutGroupSize == kActivationGroupSize);
static_assert(kWeightScaleGroupSize % kActivationGroupSize == 0);

inline float to_float(fp16 value) { return static_cast<float>(value); }
inline fp16 to_half(float value) { return static_cast<fp16>(value); }
inline fp16 half_add(fp16 a, fp16 b) { return to_half(to_float(a) + to_float(b)); }
inline fp16 half_mul(fp16 a, fp16 b) { return to_half(to_float(a) * to_float(b)); }

inline fp16 neon_half_add(fp16 a, fp16 b)
{
    return vgetq_lane_f16(vaddq_f16(vdupq_n_f16(a), vdupq_n_f16(b)), 0);
}

inline fp16 neon_half_mul(fp16 a, fp16 b)
{
    return vgetq_lane_f16(vmulq_f16(vdupq_n_f16(a), vdupq_n_f16(b)), 0);
}

inline fp16 neon_half_fma(fp16 accumulator, fp16 a, fp16 b)
{
    return vgetq_lane_f16(vfmaq_f16(vdupq_n_f16(accumulator), vdupq_n_f16(a), vdupq_n_f16(b)), 0);
}

inline uint16_t half_bits(fp16 value)
{
    uint16_t bits;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline uint32_t ordered_half_bits(fp16 value)
{
    const uint16_t bits = half_bits(value);
    return (bits & 0x8000u) ? 0x8000u - (bits & 0x7fffu) : 0x8000u + bits;
}

inline uint32_t half_ulp_distance(fp16 a, fp16 b)
{
    const uint32_t ordered_a = ordered_half_bits(a);
    const uint32_t ordered_b = ordered_half_bits(b);
    return ordered_a > ordered_b ? ordered_a - ordered_b : ordered_b - ordered_a;
}

struct Schedule
{
    int expanded_m_tile = 0;
    int lookup_groups_per_tile = kLookupGroupsPerKernelTile;
};

template <int Bits>
Schedule official_compatible_schedule(int logical_m, int k)
{
    static_assert(Bits >= 2 && Bits <= 4, "Bits must be W2, W3, or W4");
    if (logical_m <= 0 || logical_m % kFp16OutputLanes != 0)
        throw std::invalid_argument("M must be positive and divisible by 8");
    if (k <= 0 || k % kWeightScaleGroupSize != 0)
        throw std::invalid_argument("K must be positive and divisible by 128");
    const int bm = Bits == 3 ? 192 : 256; // smallest official qgemm.py candidates
    if ((logical_m * Bits) % bm != 0)
        throw std::invalid_argument("M is incompatible with the official expanded-M tile");
    return {bm, kLookupGroupsPerKernelTile};
}

struct PackedWeights
{
    int logical_m = 0;
    int k = 0;
    int bits = 0;
    Schedule schedule;
    std::vector<uint8_t> bytes;
};

struct PackedScales
{
    int logical_m = 0;
    int k = 0;
    int bits = 0;
    Schedule schedule;
    std::vector<fp16> values;
};

struct Workspace
{
    std::vector<int8_t> quantized_luts;
    std::vector<fp16> lut_scales;
    std::vector<fp16> lut_biases;
    std::vector<fp16> expanded_bitplane_sums;
};

inline size_t packed_weight_offset(const PackedWeights& packed, int m_tile, int k_tile,
                                   int block32, int k_inner)
{
    const int k_tiles = (packed.k / kLutGroupSize) / packed.schedule.lookup_groups_per_tile;
    const int blocks32 = packed.schedule.expanded_m_tile / kExpandedRowsPerVector;
    return (((static_cast<size_t>(m_tile) * k_tiles + k_tile) * blocks32 + block32) *
            packed.schedule.lookup_groups_per_tile + k_inner) * kPackedByteLanes;
}

template <int Bits>
PackedWeights pack_weights_tmac(const std::vector<uint8_t>& qweights, int logical_m, int k)
{
    const Schedule schedule = official_compatible_schedule<Bits>(logical_m, k);
    if (qweights.size() != static_cast<size_t>(logical_m) * k)
        throw std::invalid_argument("qweight size mismatch");

    const int expanded_m = logical_m * Bits;
    const int k_groups = k / kLutGroupSize;
    const int m_tiles = expanded_m / schedule.expanded_m_tile;
    const int k_tiles = k_groups / schedule.lookup_groups_per_tile;
    const int blocks32 = schedule.expanded_m_tile / kExpandedRowsPerVector;
    const size_t packed_size = static_cast<size_t>(m_tiles) * k_tiles * blocks32 *
                               schedule.lookup_groups_per_tile * kPackedByteLanes;
    if (packed_size != static_cast<size_t>(logical_m) * k * Bits / 8)
        throw std::logic_error("packed size is not M*K*Bits/8");

    PackedWeights packed{logical_m, k, Bits, schedule, std::vector<uint8_t>(packed_size, 0)};
    for (int logical_row = 0; logical_row < logical_m; ++logical_row)
    {
        const int logical_block8 = logical_row / kFp16OutputLanes;
        const int lane = logical_row % kFp16OutputLanes;
        for (int bit_plane = 0; bit_plane < Bits; ++bit_plane)
        {
            // Official expanded-M order: [logical block8][bit plane][lane].
            const int expanded_row = (logical_block8 * Bits + bit_plane) * kFp16OutputLanes + lane;
            const int m_tile = expanded_row / schedule.expanded_m_tile;
            const int local = expanded_row % schedule.expanded_m_tile;
            const int block32 = local / kExpandedRowsPerVector;
            const int group8 = (local % kExpandedRowsPerVector) / kFp16OutputLanes;
            const int byte_lane = (group8 == 0 || group8 == 2) ? lane : kFp16OutputLanes + lane;
            const bool high_nibble = group8 >= 2;

            for (int kg = 0; kg < k_groups; ++kg)
            {
                uint8_t lut_index = 0;
                const size_t qbase = static_cast<size_t>(logical_row) * k + kg * kLutGroupSize;
                for (int g = 0; g < kLutGroupSize; ++g)
                    lut_index |= static_cast<uint8_t>(((qweights[qbase + g] >> bit_plane) & 1u) << g);
                const int k_tile = kg / schedule.lookup_groups_per_tile;
                const int k_inner = kg % schedule.lookup_groups_per_tile;
                uint8_t& destination = packed.bytes[packed_weight_offset(
                    packed, m_tile, k_tile, block32, k_inner) + byte_lane];
                destination |= high_nibble ? static_cast<uint8_t>(lut_index << 4) : lut_index;
            }
        }
    }
    return packed;
}

template <int Bits>
PackedScales pack_scales_tmac(const std::vector<fp16>& row_major_scales,
                              int logical_m, int k, const Schedule& schedule)
{
    const int m_tiles = logical_m * Bits / schedule.expanded_m_tile;
    const int logical_rows_per_tile = schedule.expanded_m_tile / Bits;
    const int weight_groups = k / kWeightScaleGroupSize;
    if (row_major_scales.size() != static_cast<size_t>(logical_m) * weight_groups)
        throw std::invalid_argument("scale size mismatch");
    PackedScales packed{logical_m, k, Bits, schedule,
                        std::vector<fp16>(static_cast<size_t>(m_tiles) * weight_groups * logical_rows_per_tile)};
    for (int logical_row = 0; logical_row < logical_m; ++logical_row)
    {
        const int m_tile = logical_row / logical_rows_per_tile;
        const int local_row = logical_row % logical_rows_per_tile;
        for (int group = 0; group < weight_groups; ++group)
            packed.values[(static_cast<size_t>(m_tile) * weight_groups + group) *
                          logical_rows_per_tile + local_row] =
                row_major_scales[static_cast<size_t>(logical_row) * weight_groups + group];
    }
    return packed;
}

inline const fp16* packed_scale_ptr(const PackedScales& scales, int m_tile,
                                    int weight_group, int local_logical_row)
{
    const int logical_rows_per_tile = scales.schedule.expanded_m_tile / scales.bits;
    const int weight_groups = scales.k / kWeightScaleGroupSize;
    return scales.values.data() +
           (static_cast<size_t>(m_tile) * weight_groups + weight_group) * logical_rows_per_tile +
           local_logical_row;
}

template <int Bits>
bool verify_weight_packing(const std::vector<uint8_t>& qweights, const PackedWeights& packed)
{
    const int k_groups = packed.k / kLutGroupSize;
    for (int logical_row = 0; logical_row < packed.logical_m; ++logical_row)
    {
        const int logical_block8 = logical_row / kFp16OutputLanes;
        const int lane = logical_row % kFp16OutputLanes;
        for (int bit_plane = 0; bit_plane < Bits; ++bit_plane)
        {
            const int expanded_row = (logical_block8 * Bits + bit_plane) * kFp16OutputLanes + lane;
            const int m_tile = expanded_row / packed.schedule.expanded_m_tile;
            const int local = expanded_row % packed.schedule.expanded_m_tile;
            const int block32 = local / kExpandedRowsPerVector;
            const int group8 = (local % kExpandedRowsPerVector) / kFp16OutputLanes;
            const int byte_lane = (group8 == 0 || group8 == 2) ? lane : kFp16OutputLanes + lane;
            const bool high_nibble = group8 >= 2;
            for (int kg = 0; kg < k_groups; ++kg)
            {
                const int k_tile = kg / packed.schedule.lookup_groups_per_tile;
                const int k_inner = kg % packed.schedule.lookup_groups_per_tile;
                const uint8_t byte = packed.bytes[packed_weight_offset(
                    packed, m_tile, k_tile, block32, k_inner) + byte_lane];
                const uint8_t index = high_nibble ? static_cast<uint8_t>((byte >> 4) & 0x0f)
                                                  : static_cast<uint8_t>(byte & 0x0f);
                for (int g = 0; g < kLutGroupSize; ++g)
                {
                    const uint8_t expected = static_cast<uint8_t>((qweights[
                        static_cast<size_t>(logical_row) * packed.k + kg * kLutGroupSize + g] >> bit_plane) & 1u);
                    if (((index >> g) & 1u) != expected) return false;
                }
            }
        }
    }
    return true;
}

// Same ARM arithmetic sequence as Microsoft/T-MAC commit 7042f8f, file
// python/t_mac/intrins/lut_ctor.cc: LD4 deinterleaves eight groups of four FP16 values.
inline void update_lut_scale_from_32_activations(fp16* scale, const fp16* activations)
{
    const float16x8x4_t values = vld4q_f16(activations);
    float16x8_t absolute_sum = vaddq_f16(vabsq_f16(values.val[0]), vabsq_f16(values.val[1]));
    absolute_sum = vaddq_f16(absolute_sum, vabsq_f16(values.val[2]));
    absolute_sum = vaddq_f16(absolute_sum, vabsq_f16(values.val[3]));
    const fp16 candidate = to_half(to_float(vmaxvq_f16(absolute_sum)) / 127.0f);
    if (to_float(candidate) > to_float(*scale)) *scale = candidate;
}

inline float horizontal_sum_for_official_bias(float16x8_t value)
{
    // GCC's ARM float16_t is a storage type; scalar lane arithmetic is promoted
    // to float in the official vaddvq_f16 macro, then rounded once on assignment.
    float sum = 0.0f;
    for (int lane = 0; lane < 8; ++lane) sum += to_float(value[lane]);
    return sum;
}

inline void construct_quantized_luts_64(const fp16* activations, fp16 lut_scale,
                                        int8_t* destination, fp16* lut_bias)
{
    float16x8_t tables[kLutEntryCount];
    fp16 bias = to_half(0.0f);
    const fp16 inverse_scale = to_float(lut_scale) == 0.0f
                                   ? to_half(0.0f)
                                   : to_half(static_cast<float>(1.0 / static_cast<double>(to_float(lut_scale))));
    for (int block32 = 0; block32 < 2; ++block32)
    {
        const float16x8x4_t values = vld4q_f16(activations + block32 * 32);
        for (int index = 1; index < kLutEntryCount; index += 2)
        {
            tables[index] = values.val[0];
            tables[index] = (index & 2) ? vaddq_f16(tables[index], values.val[1])
                                        : vsubq_f16(tables[index], values.val[1]);
            tables[index] = (index & 4) ? vaddq_f16(tables[index], values.val[2])
                                        : vsubq_f16(tables[index], values.val[2]);
            tables[index] = (index & 8) ? vaddq_f16(tables[index], values.val[3])
                                        : vsubq_f16(tables[index], values.val[3]);
        }
        for (int index = 0; index < kLutEntryCount; index += 2)
            tables[index] = vnegq_f16(tables[kLutEntryCount - 1 - index]);
        bias = to_half(to_float(bias) + horizontal_sum_for_official_bias(tables[0]));

        alignas(16) int8_t quantized_by_index[kLutEntryCount][8];
        for (int index = 0; index < kLutEntryCount; ++index)
        {
            const float16x8_t scaled = vmulq_n_f16(tables[index], inverse_scale);
            vst1_s8(quantized_by_index[index], vqmovn_s16(vcvtnq_s16_f16(scaled)));
        }
        for (int lane = 0; lane < 8; ++lane)
            for (int index = 0; index < kLutEntryCount; ++index)
                destination[(block32 * 8 + lane) * kLutEntryCount + index] =
                    quantized_by_index[index][lane];
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
        for (int group = 0; group < activation_groups; ++group)
        {
            const fp16* group_activations = activations +
                static_cast<size_t>(row) * k + group * kActivationGroupSize;
            fp16 scale = to_half(0.0f);
            update_lut_scale_from_32_activations(&scale, group_activations);
            update_lut_scale_from_32_activations(&scale, group_activations + 32);
            fp16 bias = to_half(0.0f);
            int8_t* qlut = workspace.quantized_luts.data() +
                (static_cast<size_t>(row) * k_groups + group * kLookupGroupsPerKernelTile) * kLutEntryCount;
            construct_quantized_luts_64(group_activations, scale, qlut, &bias);
            workspace.lut_scales[static_cast<size_t>(row) * activation_groups + group] = scale;
            workspace.lut_biases[static_cast<size_t>(row) * activation_groups + group] = bias;
        }
}

struct SignedWideningAdder16
{
    int16x8_t low = vdupq_n_s16(0);
    int16x8_t high = vdupq_n_s16(0);
    void push(int8x16_t values, int index)
    {
        const int16x8_t widened_low = vmovl_s8(vget_low_s8(values));
        const int16x8_t widened_high = vmovl_high_s8(values);
        if (index == 0) { low = widened_low; high = widened_high; }
        else { low = vaddq_s16(low, widened_low); high = vaddq_s16(high, widened_high); }
    }
};

inline float16x8_t reconstruct_lookup(float16x8_t lookup_sum, fp16 lut_scale, fp16 lut_bias, bool add_bias)
{
    return add_bias ? vfmaq_n_f16(vdupq_n_f16(lut_bias), lookup_sum, lut_scale) : vmulq_n_f16(lookup_sum, lut_scale);
}

template <int Bits>
void tmac_gemm_a16(const PackedWeights& weights, const PackedScales& scales, int n,
                   const fp16* activations, fp16* output, Workspace& workspace)
{
    static_assert(Bits >= 2 && Bits <= 4);
    if (n <= 0 || weights.bits != Bits || scales.bits != Bits || weights.logical_m != scales.logical_m || weights.k != scales.k || weights.schedule.expanded_m_tile != scales.schedule.expanded_m_tile || weights.schedule.lookup_groups_per_tile != scales.schedule.lookup_groups_per_tile)
        throw std::invalid_argument("incompatible T-MAC packed operands");

    const int logical_m = weights.logical_m;
    const int k = weights.k;
    const int bm = weights.schedule.expanded_m_tile;
    const int m_tiles = logical_m * Bits / bm;
    const int logical_rows_per_tile = bm / Bits;
    const int blocks32 = bm / kExpandedRowsPerVector;
    const int k_groups = k / kLutGroupSize;
    const int k_tiles = k_groups / kLookupGroupsPerKernelTile;
    const int activation_groups = k / kActivationGroupSize;
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0f);
    const size_t expected_weight_bytes = static_cast<size_t>(logical_m) * k * Bits / 8;
    const size_t expected_scale_values = static_cast<size_t>(m_tiles) * (k / kWeightScaleGroupSize) * logical_rows_per_tile;
    if (bm <= 0 || bm % kExpandedRowsPerVector != 0 || bm % Bits != 0 || logical_m * Bits % bm != 0 || weights.schedule.lookup_groups_per_tile != kLookupGroupsPerKernelTile || k_groups % kLookupGroupsPerKernelTile != 0 || weights.bytes.size() != expected_weight_bytes || scales.values.size() != expected_scale_values)
        throw std::invalid_argument("invalid T-MAC layout metadata or packed buffer size");

    build_activation_luts(n, k, activations, workspace);
    workspace.expanded_bitplane_sums.resize(static_cast<size_t>(bm));
    for (int row = 0; row < n; ++row)
        for (int m_tile = 0; m_tile < m_tiles; ++m_tile)
        {
            fp16* expanded = workspace.expanded_bitplane_sums.data();
            std::fill(expanded, expanded + bm, to_half(0.0f));
            for (int block32 = 0; block32 < blocks32; ++block32)
            {
                float16x8_t accumulators[4] = {
                    vdupq_n_f16(to_half(0.0f)), vdupq_n_f16(to_half(0.0f)),
                    vdupq_n_f16(to_half(0.0f)), vdupq_n_f16(to_half(0.0f))};
                for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
                {
                    const int kg_begin = k_tile * kLookupGroupsPerKernelTile;
                    const int activation_group = k_tile;
                    const int weight_group = (kg_begin * kLutGroupSize) / kWeightScaleGroupSize;
                    const int8_t* qlut_base = workspace.quantized_luts.data() +
                        (static_cast<size_t>(row) * k_groups + kg_begin) * kLutEntryCount;
                    const uint8_t* packed_base = weights.bytes.data() +
                        packed_weight_offset(weights, m_tile, k_tile, block32, 0);
                    SignedWideningAdder16 bottom_adder;
                    SignedWideningAdder16 top_adder;
                    for (int k_inner = 0; k_inner < kLookupGroupsPerKernelTile; ++k_inner)
                    {
                        const uint8x16_t packed_indices = vld1q_u8(
                            packed_base + static_cast<size_t>(k_inner) * kPackedByteLanes);
                        const int8x16_t table = vld1q_s8(
                            qlut_base + static_cast<size_t>(k_inner) * kLutEntryCount);
                        bottom_adder.push(vqtbl1q_s8(table, vandq_u8(packed_indices, nibble_mask)), k_inner);
                        top_adder.push(vqtbl1q_s8(table, vshrq_n_u8(packed_indices, 4)), k_inner);
                    }
                    const float16x8_t lookup_sums[4] = {
                        vcvtq_f16_s16(bottom_adder.low), vcvtq_f16_s16(bottom_adder.high),
                        vcvtq_f16_s16(top_adder.low), vcvtq_f16_s16(top_adder.high)};
                    const fp16 lut_scale = workspace.lut_scales[
                        static_cast<size_t>(row) * activation_groups + activation_group];
                    const fp16 lut_bias = workspace.lut_biases[
                        static_cast<size_t>(row) * activation_groups + activation_group];
                    for (int group8 = 0; group8 < 4; ++group8)
                    {
                        const int expanded_group8 = block32 * 4 + group8;
                        const int bit_plane = expanded_group8 % Bits;
                        const int local_logical_block8 = expanded_group8 / Bits;
                        const float16x8_t weight_scale = vld1q_f16(packed_scale_ptr(
                            scales, m_tile, weight_group, local_logical_block8 * kFp16OutputLanes));
                        const float16x8_t reconstructed_lookup = reconstruct_lookup(lookup_sums[group8], lut_scale, lut_bias, bit_plane == 0);
                        accumulators[group8] = vfmaq_f16(accumulators[group8], reconstructed_lookup, weight_scale);
                    }
                }
                for (int group8 = 0; group8 < 4; ++group8)
                    vst1q_f16(expanded + (block32 * 4 + group8) * kFp16OutputLanes,
                              accumulators[group8]);
            }

            fp16* output_tile = output + static_cast<size_t>(row) * logical_m +
                                m_tile * logical_rows_per_tile;
            for (int block8 = 0; block8 < logical_rows_per_tile / kFp16OutputLanes; ++block8)
            {
                const fp16* planes = expanded + block8 * Bits * kFp16OutputLanes;
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
                vst1q_f16(output_tile + block8 * kFp16OutputLanes,
                          vcombine_f16(vcvt_f16_f32(low), vcvt_f16_f32(high)));
            }
        }
}

template <int Bits>
void scalar_packed_reference(const PackedWeights& weights, const PackedScales& scales,
                             int n, const Workspace& workspace, std::vector<fp16>& output)
{
    const int logical_m = weights.logical_m;
    const int k_groups = weights.k / kLutGroupSize;
    const int k_tiles = k_groups / kLookupGroupsPerKernelTile;
    const int logical_rows_per_tile = weights.schedule.expanded_m_tile / Bits;
    const int activation_groups = weights.k / kActivationGroupSize;
    output.assign(static_cast<size_t>(n) * logical_m, to_half(0.0f));
    for (int row = 0; row < n; ++row)
        for (int logical_row = 0; logical_row < logical_m; ++logical_row)
        {
            const int m_tile = logical_row / logical_rows_per_tile;
            const int local_row = logical_row % logical_rows_per_tile;
            const int local_block8 = local_row / kFp16OutputLanes;
            const int lane = local_row % kFp16OutputLanes;
            std::array<fp16, Bits> plane_sums{};
            for (int bit_plane = 0; bit_plane < Bits; ++bit_plane)
            {
                plane_sums[bit_plane] = to_half(0.0f);
                const int expanded_group8 = local_block8 * Bits + bit_plane;
                const int block32 = expanded_group8 / 4;
                const int group8 = expanded_group8 % 4;
                const int byte_lane = (group8 == 0 || group8 == 2) ? lane : kFp16OutputLanes + lane;
                const bool high_nibble = group8 >= 2;
                for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
                {
                    int lookup_sum = 0;
                    for (int k_inner = 0; k_inner < kLookupGroupsPerKernelTile; ++k_inner)
                    {
                        const int kg = k_tile * kLookupGroupsPerKernelTile + k_inner;
                        const uint8_t byte = weights.bytes[packed_weight_offset(
                            weights, m_tile, k_tile, block32, k_inner) + byte_lane];
                        const uint8_t index = high_nibble ? static_cast<uint8_t>((byte >> 4) & 0x0f)
                                                          : static_cast<uint8_t>(byte & 0x0f);
                        lookup_sum += workspace.quantized_luts[
                            (static_cast<size_t>(row) * k_groups + kg) * kLutEntryCount + index];
                    }
                    const int weight_group = (k_tile * kActivationGroupSize) / kWeightScaleGroupSize;
                    const fp16 lookup_sum_fp16 = to_half(static_cast<float>(lookup_sum));
                    const fp16 lut_scale = workspace.lut_scales[static_cast<size_t>(row) * activation_groups + k_tile];
                    const fp16 reconstructed_lookup = bit_plane == 0 ? neon_half_fma(workspace.lut_biases[static_cast<size_t>(row) * activation_groups + k_tile], lookup_sum_fp16, lut_scale) : neon_half_mul(lookup_sum_fp16, lut_scale);
                    plane_sums[bit_plane] = neon_half_fma(plane_sums[bit_plane], reconstructed_lookup, *packed_scale_ptr(scales, m_tile, weight_group, local_row));
                }
            }
            float result = 0.5f * to_float(plane_sums[0]) + to_float(plane_sums[1]);
            if constexpr (Bits >= 3) result += 2.0f * to_float(plane_sums[2]);
            if constexpr (Bits >= 4) result += 4.0f * to_float(plane_sums[3]);
            output[static_cast<size_t>(row) * logical_m + logical_row] = to_half(result);
        }
}

template <int Bits>
void independent_dense_reference(int logical_m, int k, int n,
                                 const std::vector<uint8_t>& qweights,
                                 const std::vector<fp16>& activations,
                                 const std::vector<fp16>& scales,
                                 std::vector<fp16>& output)
{
    const int zero_point = 1 << (Bits - 1);
    const int weight_groups = k / kWeightScaleGroupSize;
    output.resize(static_cast<size_t>(n) * logical_m);
    for (int row = 0; row < n; ++row)
        for (int m = 0; m < logical_m; ++m)
        {
            double sum = 0.0;
            for (int kk = 0; kk < k; ++kk)
            {
                const int signed_weight = static_cast<int>(qweights[
                    static_cast<size_t>(m) * k + kk]) - zero_point;
                sum += static_cast<double>(signed_weight) * to_float(scales[
                    static_cast<size_t>(m) * weight_groups + kk / kWeightScaleGroupSize]) *
                    to_float(activations[static_cast<size_t>(row) * k + kk]);
            }
            output[static_cast<size_t>(row) * logical_m + m] = to_half(static_cast<float>(sum));
        }
}

template <int Bits>
bool verify_case(int logical_m, int k, int n, uint32_t seed, int mode)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> weight_distribution(0, (1 << Bits) - 1);
    std::uniform_real_distribution<float> activation_distribution(-1.0f, 1.0f);
    std::uniform_real_distribution<float> scale_distribution(0.02f, 0.15f);
    const int weight_groups = k / kWeightScaleGroupSize;
    std::vector<uint8_t> qweights(static_cast<size_t>(logical_m) * k);
    std::vector<fp16> activations(static_cast<size_t>(n) * k);
    std::vector<fp16> scales(static_cast<size_t>(logical_m) * weight_groups);
    for (uint8_t& value : qweights) value = static_cast<uint8_t>(weight_distribution(rng));
    for (fp16& value : activations) value = to_half(activation_distribution(rng));
    for (fp16& value : scales) value = to_half(scale_distribution(rng));
    if (mode == 1) std::fill(activations.begin(), activations.end(), to_half(0.0f));
    if (mode == 2)
        for (size_t i = 0; i < activations.size(); ++i)
            activations[i] = to_half((i & 1) ? 1.0f : -1.0f);
    if (mode == 3) std::fill(qweights.begin(), qweights.end(), 0);
    if (mode == 4) std::fill(qweights.begin(), qweights.end(), static_cast<uint8_t>((1 << Bits) - 1));
    if (mode == 5)
        for (size_t i = 0; i < qweights.size(); ++i)
            qweights[i] = (i & 1) ? static_cast<uint8_t>((1 << Bits) - 1) : 0;
    if (mode == 6) std::fill(scales.begin(), scales.end(), to_half(1.0f));

    const PackedWeights packed_weights = pack_weights_tmac<Bits>(qweights, logical_m, k);
    const PackedScales packed_scales = pack_scales_tmac<Bits>(scales, logical_m, k, packed_weights.schedule);
    const bool packing_ok = verify_weight_packing<Bits>(qweights, packed_weights);
    Workspace workspace;
    std::vector<fp16> actual(static_cast<size_t>(n) * logical_m);
    tmac_gemm_a16<Bits>(packed_weights, packed_scales, n, activations.data(), actual.data(), workspace);
    std::vector<fp16> packed_reference;
    scalar_packed_reference<Bits>(packed_weights, packed_scales, n, workspace, packed_reference);
    std::vector<fp16> dense_reference;
    independent_dense_reference<Bits>(logical_m, k, n, qweights, activations, scales, dense_reference);

    size_t kernel_mismatches = 0;
    size_t signed_zero_mismatches = 0;
    size_t non_finite = 0;
    uint32_t max_packed_ulp = 0;
    double max_packed_abs_error = 0.0;
    double max_dense_error = 0.0;
    double mean_dense_error = 0.0;
    double squared_error_sum = 0.0;
    double squared_reference_sum = 0.0;
    for (size_t i = 0; i < actual.size(); ++i)
    {
        if (half_bits(actual[i]) != half_bits(packed_reference[i])) ++kernel_mismatches;
        if (to_float(actual[i]) == 0.0f && to_float(packed_reference[i]) == 0.0f && half_bits(actual[i]) != half_bits(packed_reference[i])) ++signed_zero_mismatches;
        max_packed_ulp = std::max(max_packed_ulp, half_ulp_distance(actual[i], packed_reference[i]));
        max_packed_abs_error = std::max(max_packed_abs_error, std::abs(static_cast<double>(to_float(actual[i])) - to_float(packed_reference[i])));
        if (!std::isfinite(to_float(actual[i]))) ++non_finite;
        const double error = std::abs(static_cast<double>(to_float(actual[i])) - to_float(dense_reference[i]));
        max_dense_error = std::max(max_dense_error, error);
        mean_dense_error += error;
        squared_error_sum += error * error;
        const double reference_value = to_float(dense_reference[i]);
        squared_reference_sum += reference_value * reference_value;
    }
    mean_dense_error /= static_cast<double>(actual.size());
    const double nmse = squared_reference_sum == 0.0 ? (squared_error_sum == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : squared_error_sum / squared_reference_sum;
    // Microsoft/T-MAC commit 7042f8f, python/t_mac/ops/qgemm.py::_verify uses NMSE <= 5e-4.
    const bool dense_accuracy_ok = nmse <= 5e-4;
    const bool passed = packing_ok && kernel_mismatches == 0 && non_finite == 0 && dense_accuracy_ok;
    std::cout << "VERIFY W" << Bits << "A16 M=" << logical_m << " K=" << k << " N=" << n << " mode=" << mode << " packing=" << (packing_ok ? "PASS" : "FAIL") << " kernel_bit_mismatches=" << kernel_mismatches << " signed_zero_mismatches=" << signed_zero_mismatches << " max_packed_ulp=" << max_packed_ulp << " max_packed_abs_error=" << max_packed_abs_error << " non_finite=" << non_finite << " max_dense_abs_error=" << max_dense_error << " mean_dense_abs_error=" << mean_dense_error << " nmse=" << nmse << " result=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

template <int Bits>
bool verify_suite()
{
    bool passed = true;
    constexpr int minimum_logical_m = Bits == 2 ? 128 : 64;
    passed &= verify_case<Bits>(256, 128, 1, 1000 + Bits, 0); // GEMV
    passed &= verify_case<Bits>(256, 256, 3, 2000 + Bits, 0); // GEMM
    passed &= verify_case<Bits>(minimum_logical_m * 3, 384, 5, 2500 + Bits, 0); // three M tiles, six activation groups, three scale groups
    for (int mode = 1; mode <= 6; ++mode)
        passed &= verify_case<Bits>(256, 128, 3, 3000 + Bits + mode, mode);
    return passed;
}

template <int Bits>
void benchmark_gemv(int logical_m, int k)
{
    std::mt19937 rng(42 + logical_m + Bits);
    std::uniform_int_distribution<int> weight_distribution(0, (1 << Bits) - 1);
    std::uniform_real_distribution<float> activation_distribution(-1.0f, 1.0f);
    const int weight_groups = k / kWeightScaleGroupSize;
    std::vector<uint8_t> qweights(static_cast<size_t>(logical_m) * k);
    std::vector<fp16> activations(k);
    std::vector<fp16> scales(static_cast<size_t>(logical_m) * weight_groups, to_half(0.05f));
    for (uint8_t& value : qweights) value = static_cast<uint8_t>(weight_distribution(rng));
    for (fp16& value : activations) value = to_half(activation_distribution(rng));
    const PackedWeights packed_weights = pack_weights_tmac<Bits>(qweights, logical_m, k);
    const PackedScales packed_scales = pack_scales_tmac<Bits>(scales, logical_m, k, packed_weights.schedule);
    Workspace workspace;
    std::vector<fp16> output(logical_m);
    tmac_gemm_a16<Bits>(packed_weights, packed_scales, 1, activations.data(), output.data(), workspace);
    const int iterations = 20;
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) tmac_gemm_a16<Bits>(packed_weights, packed_scales, 1, activations.data(), output.data(), workspace);
    const auto end = std::chrono::steady_clock::now();
    const double latency_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    const double effective_gops = 2.0 * logical_m * k / (latency_ms / 1000.0) / 1e9;
    double checksum = 0.0;
    for (fp16 value : output) checksum += to_float(value);
    std::cout << std::left << std::setw(16) << (std::to_string(logical_m) + "x" + std::to_string(k) + "x1") << std::setw(16) << latency_ms << std::setw(18) << effective_gops << checksum << '\n';
}
} // namespace tmac_neon

int main(int argc, char** argv)
{
    const bool verify_only = argc == 2 && std::string(argv[1]) == "--verify-only";
    if (argc > 2 || (argc == 2 && !verify_only))
    {
        std::cerr << "Usage: " << argv[0] << " [--verify-only]\n";
        return 2;
    }
    const bool verified = tmac_neon::verify_suite<2>() & tmac_neon::verify_suite<3>() & tmac_neon::verify_suite<4>();
    if (!verified) return 1;
    if (verify_only) return 0;
    std::cout << "T-MAC AArch64 NEON W4A16 GEMV benchmark (g=4, 16-entry INT8 LUT)\n";
    std::cout << std::left << std::setw(16) << "Shape" << std::setw(16) << "Latency(ms)" << std::setw(18) << "Effective GOPS" << "Checksum\n";
    for (int size : {256, 1024, 2048, 4096, 8192}) tmac_neon::benchmark_gemv<4>(size, size);
    return 0;
}
