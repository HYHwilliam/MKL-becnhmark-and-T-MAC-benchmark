#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tmac_layout
{

constexpr int kLutGroupSize = 4;
constexpr int kLutEntryCount = 1 << kLutGroupSize;
constexpr int kPackedByteLanes = 16;
constexpr int kFp16OutputLanes = 8;
constexpr int kExpandedRowsPerVector = 32;
constexpr int kActivationGroupSize = 64;
constexpr int kWeightScaleGroupSize = 128;
constexpr int kDefaultKFactor = 16;

static_assert(kLutEntryCount == 16);
static_assert(kDefaultKFactor * kLutGroupSize == kActivationGroupSize);

struct Schedule
{
    int bm = 0;
    int bn = 0;
    int kfactor = kDefaultKFactor;
};

template <int Bits>
std::vector<Schedule> official_schedule_candidates(int logical_m, int n, int k)
{
    static_assert(Bits >= 2 && Bits <= 4);
    if (logical_m <= 0 || n <= 0 || k <= 0) throw std::invalid_argument("M, N, and K must be positive");
    if (logical_m % kFp16OutputLanes != 0) throw std::invalid_argument("M must be divisible by 8");
    if (k % kActivationGroupSize != 0 || k % kWeightScaleGroupSize != 0) throw std::invalid_argument("K must be divisible by 64 and 128");

    const int expanded_m = logical_m * Bits;
    const std::vector<int> bm_values = Bits == 3 ? std::vector<int>{192, 384, 576, 768}
                                                  : std::vector<int>{256, 128, 512, 1024, 320, 640};
    const std::vector<int> bn_values = n <= 8 ? std::vector<int>{8} : std::vector<int>{8, 16, 32, 64};
    const std::vector<int> kfactor_values = {8, 16};
    std::vector<Schedule> result;

    for (int bm : bm_values)
    {
        if (expanded_m % bm != 0 || bm % Bits != 0 || (bm / Bits) % kFp16OutputLanes != 0 || bm % kExpandedRowsPerVector != 0) continue;
        for (int bn : bn_values)
        {
            if (n > 8 && n % bn != 0) continue;
            for (int kfactor : kfactor_values)
            {
                const int k_tile = kfactor * kLutGroupSize;
                if (k_tile % kActivationGroupSize != 0 || kWeightScaleGroupSize % k_tile != 0) continue;
                result.push_back({bm, bn, kfactor});
            }
        }
    }

    if (result.empty()) throw std::invalid_argument("No official T-MAC schedule candidate exists for this shape");
    return result;
}

// Benchmark-only policy. Microsoft T-MAC selects among these legal candidates through tuning.
template <int Bits>
Schedule choose_single_thread_schedule(int logical_m, int n, int k)
{
    const std::vector<Schedule> candidates = official_schedule_candidates<Bits>(logical_m, n, k);
    const int preferred_bm = Bits == 3 ? 384 : 512;
    Schedule best = candidates.front();
    long long best_score = std::numeric_limits<long long>::min();

    for (const Schedule& candidate : candidates)
    {
        long long score = -static_cast<long long>(std::abs(candidate.bm - preferred_bm)) * 100LL;
        score += static_cast<long long>(candidate.bn) * 10LL;
        score += candidate.kfactor;
        if (score > best_score) { best_score = score; best = candidate; }
    }
    return best;
}

struct PackedWeights
{
    int logical_m = 0;
    int k = 0;
    int bits = 0;
    Schedule schedule;
    std::vector<uint8_t> bytes;
};

template <typename Half>
struct PackedScales
{
    int logical_m = 0;
    int k = 0;
    int bits = 0;
    Schedule schedule;
    std::vector<Half> values;
};

inline size_t packed_weight_offset(const PackedWeights& packed, int m_tile, int k_tile, int block32, int k_inner)
{
    const int k_tiles = (packed.k / kLutGroupSize) / packed.schedule.kfactor;
    const int blocks32 = packed.schedule.bm / kExpandedRowsPerVector;
    return (((static_cast<size_t>(m_tile) * k_tiles + k_tile) * blocks32 + block32) * packed.schedule.kfactor + k_inner) * kPackedByteLanes;
}

template <int Bits>
PackedWeights pack_weights_tmac(const std::vector<uint8_t>& qweights, int logical_m, int k, const Schedule& schedule)
{
    const int expanded_m = logical_m * Bits;
    const int k_groups = k / kLutGroupSize;
    const int m_tiles = expanded_m / schedule.bm;
    const int k_tiles = k_groups / schedule.kfactor;
    const int blocks32 = schedule.bm / kExpandedRowsPerVector;
    if (qweights.size() != static_cast<size_t>(logical_m) * k) throw std::invalid_argument("qweight size mismatch");

    const size_t packed_size = static_cast<size_t>(m_tiles) * k_tiles * blocks32 * schedule.kfactor * kPackedByteLanes;
    if (packed_size != static_cast<size_t>(logical_m) * k * Bits / 8) throw std::logic_error("packed weight size mismatch");
    PackedWeights packed{logical_m, k, Bits, schedule, std::vector<uint8_t>(packed_size, 0)};

    for (int logical_row = 0; logical_row < logical_m; ++logical_row)
    {
        const int logical_block8 = logical_row / kFp16OutputLanes;
        const int lane = logical_row % kFp16OutputLanes;
        for (int bit_plane = 0; bit_plane < Bits; ++bit_plane)
        {
            const int expanded_row = (logical_block8 * Bits + bit_plane) * kFp16OutputLanes + lane;
            const int m_tile = expanded_row / schedule.bm;
            const int local = expanded_row % schedule.bm;
            const int block32 = local / kExpandedRowsPerVector;
            const int group8 = (local % kExpandedRowsPerVector) / kFp16OutputLanes;
            const int byte_lane = (group8 == 0 || group8 == 2) ? lane : kFp16OutputLanes + lane;
            const bool high_nibble = group8 >= 2;

            for (int kg = 0; kg < k_groups; ++kg)
            {
                uint8_t lut_index = 0;
                const size_t qbase = static_cast<size_t>(logical_row) * k + kg * kLutGroupSize;
                for (int g = 0; g < kLutGroupSize; ++g) lut_index |= static_cast<uint8_t>(((qweights[qbase + g] >> bit_plane) & 1U) << g);
                const int k_tile = kg / schedule.kfactor;
                const int k_inner = kg % schedule.kfactor;
                uint8_t& dst = packed.bytes[packed_weight_offset(packed, m_tile, k_tile, block32, k_inner) + byte_lane];
                dst |= high_nibble ? static_cast<uint8_t>(lut_index << 4) : lut_index;
            }
        }
    }
    return packed;
}

template <int Bits, typename Half>
PackedScales<Half> pack_scales_tmac(const std::vector<Half>& row_major_scales, int logical_m, int k, const Schedule& schedule)
{
    const int expanded_m = logical_m * Bits;
    const int m_tiles = expanded_m / schedule.bm;
    const int logical_rows_per_tile = schedule.bm / Bits;
    const int weight_groups = k / kWeightScaleGroupSize;
    if (row_major_scales.size() != static_cast<size_t>(logical_m) * weight_groups) throw std::invalid_argument("scale size mismatch");

    PackedScales<Half> packed{logical_m, k, Bits, schedule,
                              std::vector<Half>(static_cast<size_t>(m_tiles) * weight_groups * logical_rows_per_tile)};
    for (int logical_row = 0; logical_row < logical_m; ++logical_row)
    {
        const int m_tile = logical_row / logical_rows_per_tile;
        const int local_row = logical_row % logical_rows_per_tile;
        for (int group = 0; group < weight_groups; ++group)
            packed.values[(static_cast<size_t>(m_tile) * weight_groups + group) * logical_rows_per_tile + local_row] =
                row_major_scales[static_cast<size_t>(logical_row) * weight_groups + group];
    }
    return packed;
}

template <typename Half>
inline const Half* packed_scale_ptr(const PackedScales<Half>& scales, int m_tile, int weight_group, int local_logical_row)
{
    const int logical_rows_per_tile = scales.schedule.bm / scales.bits;
    const int weight_groups = scales.k / kWeightScaleGroupSize;
    return scales.values.data() + (static_cast<size_t>(m_tile) * weight_groups + weight_group) * logical_rows_per_tile + local_logical_row;
}

template <int Bits, typename Half>
void validate_packed_operands(const PackedWeights& weights, const PackedScales<Half>& scales, int n)
{
    if (n <= 0 || weights.bits != Bits || scales.bits != Bits) throw std::invalid_argument("Packed bit metadata mismatch");
    if (weights.logical_m != scales.logical_m || weights.k != scales.k) throw std::invalid_argument("Packed shape metadata mismatch");
    if (weights.schedule.bm != scales.schedule.bm || weights.schedule.bn != scales.schedule.bn || weights.schedule.kfactor != scales.schedule.kfactor)
        throw std::invalid_argument("Packed schedules differ");

    const auto candidates = official_schedule_candidates<Bits>(weights.logical_m, n, weights.k);
    const bool schedule_ok = std::any_of(candidates.begin(), candidates.end(), [&](const Schedule& candidate)
    {
        return candidate.bm == weights.schedule.bm && candidate.bn == weights.schedule.bn && candidate.kfactor == weights.schedule.kfactor;
    });
    if (!schedule_ok) throw std::invalid_argument("Packed schedule is not an official T-MAC candidate");

    const size_t expected_weight_bytes = static_cast<size_t>(weights.logical_m) * weights.k * Bits / 8;
    const int logical_rows_per_tile = weights.schedule.bm / Bits;
    const int m_tiles = weights.logical_m * Bits / weights.schedule.bm;
    const int weight_groups = weights.k / kWeightScaleGroupSize;
    const size_t expected_scale_values = static_cast<size_t>(m_tiles) * weight_groups * logical_rows_per_tile;
    if (weights.bytes.size() != expected_weight_bytes) throw std::invalid_argument("Packed weight buffer size mismatch");
    if (scales.values.size() != expected_scale_values) throw std::invalid_argument("Packed scale buffer size mismatch");
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
            const int m_tile = expanded_row / packed.schedule.bm;
            const int local = expanded_row % packed.schedule.bm;
            const int block32 = local / kExpandedRowsPerVector;
            const int group8 = (local % kExpandedRowsPerVector) / kFp16OutputLanes;
            const int byte_lane = (group8 == 0 || group8 == 2) ? lane : kFp16OutputLanes + lane;
            const bool high_nibble = group8 >= 2;

            for (int kg = 0; kg < k_groups; ++kg)
            {
                const int k_tile = kg / packed.schedule.kfactor;
                const int k_inner = kg % packed.schedule.kfactor;
                const uint8_t byte = packed.bytes[packed_weight_offset(packed, m_tile, k_tile, block32, k_inner) + byte_lane];
                const uint8_t index = high_nibble ? static_cast<uint8_t>((byte >> 4) & 0x0f) : static_cast<uint8_t>(byte & 0x0f);
                const size_t qbase = static_cast<size_t>(logical_row) * packed.k + kg * kLutGroupSize;
                for (int g = 0; g < kLutGroupSize; ++g)
                {
                    const uint8_t expected = static_cast<uint8_t>((qweights[qbase + g] >> bit_plane) & 1U);
                    if (((index >> g) & 1U) != expected) return false;
                }
            }
        }
    }
    return true;
}


template <int Bits>
bool verify_weight_layout_against_official_transform(const std::vector<uint8_t>& qweights, const PackedWeights& packed)
{
    if (packed.bits != Bits || qweights.size() != static_cast<size_t>(packed.logical_m) * packed.k) return false;
    const int expanded_m = packed.logical_m * Bits;
    const int k_groups = packed.k / kLutGroupSize;
    const int m_tiles = expanded_m / packed.schedule.bm;
    const int k_tiles = k_groups / packed.schedule.kfactor;
    const int blocks32 = packed.schedule.bm / kExpandedRowsPerVector;
    std::vector<uint8_t> expected(packed.bytes.size(), 0);

    auto lut_index_from_expanded_row = [&](int expanded_row, int kg)
    {
        const int logical_block8 = expanded_row / (Bits * kFp16OutputLanes);
        const int within_block = expanded_row % (Bits * kFp16OutputLanes);
        const int bit_plane = within_block / kFp16OutputLanes;
        const int lane = within_block % kFp16OutputLanes;
        const int logical_row = logical_block8 * kFp16OutputLanes + lane;
        uint8_t index = 0;
        const size_t qbase = static_cast<size_t>(logical_row) * packed.k + kg * kLutGroupSize;
        for (int g = 0; g < kLutGroupSize; ++g) index |= static_cast<uint8_t>(((qweights[qbase + g] >> bit_plane) & 1U) << g);
        return index;
    };

    for (int m_tile = 0; m_tile < m_tiles; ++m_tile)
    {
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
        {
            for (int block32 = 0; block32 < blocks32; ++block32)
            {
                for (int k_inner = 0; k_inner < packed.schedule.kfactor; ++k_inner)
                {
                    const int kg = k_tile * packed.schedule.kfactor + k_inner;
                    for (int lane16 = 0; lane16 < kPackedByteLanes; ++lane16)
                    {
                        const int low_row = m_tile * packed.schedule.bm + block32 * kExpandedRowsPerVector + lane16;
                        const int high_row = low_row + kPackedByteLanes;
                        const uint8_t low = lut_index_from_expanded_row(low_row, kg);
                        const uint8_t high = lut_index_from_expanded_row(high_row, kg);
                        const size_t official_offset = (((static_cast<size_t>(m_tile) * k_tiles + k_tile) * blocks32 + block32)
                                                      * packed.schedule.kfactor + k_inner) * kPackedByteLanes + lane16;
                        expected[official_offset] = static_cast<uint8_t>(low | (high << 4));
                    }
                }
            }
        }
    }
    return expected == packed.bytes;
}

template <int Bits, typename Half>
bool verify_scale_layout_against_official_transform(const std::vector<Half>& row_major_scales, const PackedScales<Half>& packed)
{
    if (packed.bits != Bits) return false;
    const int logical_rows_per_tile = packed.schedule.bm / Bits;
    const int m_tiles = packed.logical_m * Bits / packed.schedule.bm;
    const int weight_groups = packed.k / kWeightScaleGroupSize;
    if (row_major_scales.size() != static_cast<size_t>(packed.logical_m) * weight_groups) return false;

    for (int m_tile = 0; m_tile < m_tiles; ++m_tile)
    {
        for (int weight_group = 0; weight_group < weight_groups; ++weight_group)
        {
            for (int local_row = 0; local_row < logical_rows_per_tile; ++local_row)
            {
                const int logical_row = m_tile * logical_rows_per_tile + local_row;
                const Half& expected = row_major_scales[static_cast<size_t>(logical_row) * weight_groups + weight_group];
                const Half& actual = packed.values[(static_cast<size_t>(m_tile) * weight_groups + weight_group) * logical_rows_per_tile + local_row];
                if (std::memcmp(&expected, &actual, sizeof(Half)) != 0) return false;
            }
        }
    }
    return true;
}

} // namespace tmac_layout
