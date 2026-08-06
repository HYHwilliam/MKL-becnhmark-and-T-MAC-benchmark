#ifndef TMAC_OFFICIAL_AVX2_HPP
#define TMAC_OFFICIAL_AVX2_HPP

#include <immintrin.h>
#include <stdint.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace tmac
{
using fp16_t = uint16_t;
constexpr int G = 4;
constexpr int SIMD_IN = 16;
constexpr int SIMD_OUT = 8;
constexpr int ACT_GROUP_SIZE = 64;
constexpr int WEIGHT_GROUP_SIZE = 128;
constexpr int DEFAULT_KFACTOR = 16;

inline fp16_t fp32_to_fp16(float value)
{
    __m128 v = _mm_set_ss(value);
    __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return static_cast<fp16_t>(_mm_extract_epi16(h, 0));
}

inline float fp16_to_fp32(fp16_t value)
{
    __m128i h = _mm_cvtsi32_si128(static_cast<int>(value));
    return _mm_cvtss_f32(_mm_cvtph_ps(h));
}

inline __m256 load_fp16x8(const fp16_t* src)
{
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src)));
}

inline void store_fp16x8(fp16_t* dst, __m256 value)
{
    __m128i h = _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), h);
}

inline float horizontal_sum(__m256 value)
{
    __m128 x = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}

#define TMAC_LOW_I8_TO_I16(v) _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v))
#define TMAC_HIGH_I8_TO_I16(v) _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1))
#define TMAC_LOW_I16_TO_I32(v) _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v))
#define TMAC_HIGH_I16_TO_I32(v) _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1))

template <int Count>
struct SignedWideningAdder
{
    __m256i low = _mm256_setzero_si256();
    __m256i high = _mm256_setzero_si256();

    inline void push(__m256i value, int index)
    {
        if (index == 0)
        {
            low = TMAC_LOW_I8_TO_I16(value);
            high = TMAC_HIGH_I8_TO_I16(value);
        }
        else
        {
            low = _mm256_add_epi16(low, TMAC_LOW_I8_TO_I16(value));
            high = _mm256_add_epi16(high, TMAC_HIGH_I8_TO_I16(value));
        }
    }
};

struct ScheduleConfig
{
    int bm = 0;
    int bn = 0;
    int kfactor = DEFAULT_KFACTOR;
};

template <int Bits>
std::vector<ScheduleConfig> official_schedule_candidates(int logical_m, int n, int k)
{
    static_assert(Bits >= 2 && Bits <= 4, "Bits must be 2, 3 or 4");
    if (logical_m <= 0 || n <= 0 || k <= 0) throw std::invalid_argument("M, N and K must be positive");
    if (logical_m % SIMD_OUT != 0) throw std::invalid_argument("Logical M must be divisible by SIMD_OUT");
    if (k % ACT_GROUP_SIZE != 0 || k % WEIGHT_GROUP_SIZE != 0) throw std::invalid_argument("K must be divisible by activation and weight group sizes");
    const int expanded_m = logical_m * Bits;
    const std::vector<int> bm_values = Bits == 3 ? std::vector<int>{192, 384, 576, 768} : std::vector<int>{256, 128, 512, 1024, 320, 640};
    const std::vector<int> bn_values = n <= 8 ? std::vector<int>{8} : std::vector<int>{8, 16, 32, 64};
    const std::vector<int> kfactor_values = {8, 16};
    std::vector<ScheduleConfig> result;

    for (int bm : bm_values)
    {
        if (expanded_m % bm != 0 || bm % Bits != 0 || (bm / Bits) % SIMD_OUT != 0 || bm % 32 != 0) continue;
        for (int bn : bn_values)
        {
            if (n >= bn && n % bn != 0) continue;
            for (int kfactor : kfactor_values)
            {
                const int k_tile = kfactor * G;
                if (k_tile % ACT_GROUP_SIZE != 0 || WEIGHT_GROUP_SIZE % k_tile != 0) continue;
                result.push_back({bm, bn, kfactor});
            }
        }
    }

    if (result.empty()) throw std::invalid_argument("No valid official schedule candidate for this shape");
    return result;
}

template <int Bits>
ScheduleConfig choose_schedule(int logical_m, int n, int k, int num_threads = 1)
{
    const std::vector<ScheduleConfig> candidates = official_schedule_candidates<Bits>(logical_m, n, k);
    const int expanded_m = logical_m * Bits;
    num_threads = std::max(1, num_threads);
    ScheduleConfig best = candidates.front();
    long long best_score = std::numeric_limits<long long>::min();

    for (const ScheduleConfig& candidate : candidates)
    {
        const int n_tiles = (n + candidate.bn - 1) / candidate.bn;
        const int m_tiles = expanded_m / candidate.bm;
        const bool parallel_n = n_tiles >= num_threads;
        const int parallel_tiles = parallel_n ? n_tiles : m_tiles;
        const int active_threads = std::min(num_threads, std::max(1, parallel_tiles));
        const bool full_utilization = active_threads == num_threads;
        const int preferred_bm = Bits == 3 ? 384 : 512;
        long long score = 0;
        score += full_utilization ? 1000000000LL : 0LL;
        score += static_cast<long long>(active_threads) * 1000000LL;
        score += parallel_n ? 100000LL : 0LL;
        score -= static_cast<long long>(std::abs(candidate.bm - preferred_bm)) * 100LL;
        score += static_cast<long long>(candidate.bn) * 10LL;
        score += candidate.kfactor;

        if (score > best_score)
        {
            best_score = score;
            best = candidate;
        }
    }

    return best;
}

inline void convert_fp16_block(const fp16_t* src, float* dst, int count)
{
    for (int i = 0; i < count; i += 8) _mm256_storeu_ps(dst + i, load_fp16x8(src + i));
}

inline void partial_max_g4_int8_k8(float* lut_scale, const float* activation)
{
    const __m256i offsets = _mm256_set_epi32(112, 96, 80, 64, 48, 32, 16, 0);
    __m256 a0 = _mm256_i32gather_ps(activation + 0, offsets, 1);
    __m256 a1 = _mm256_i32gather_ps(activation + 1, offsets, 1);
    __m256 a2 = _mm256_i32gather_ps(activation + 2, offsets, 1);
    __m256 a3 = _mm256_i32gather_ps(activation + 3, offsets, 1);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 sum = _mm256_add_ps(_mm256_add_ps(_mm256_andnot_ps(sign, a0), _mm256_andnot_ps(sign, a1)), _mm256_add_ps(_mm256_andnot_ps(sign, a2), _mm256_andnot_ps(sign, a3)));
    __m128 max4 = _mm_max_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
    *lut_scale = std::max(*lut_scale, _mm_cvtss_f32(max4) / 127.0f);
}

inline void lut_ctor_g4_int8_avx2(int act_k, int8_t* qlut, const float* activation, float lut_scale, float* lut_bias)
{
    __m256 lut[16];
    float bias = 0.0f;
    const __m256i offsets = _mm256_set_epi32(112, 96, 80, 64, 48, 32, 16, 0);
    const float inverse_scale = lut_scale == 0.0f ? 0.0f : 1.0f / lut_scale;
    const __m256 inverse = _mm256_set1_ps(inverse_scale);
    const __m256i shuffle = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    for (int block = 0; block < act_k / 32; ++block)
    {
        __m256 a0 = _mm256_i32gather_ps(activation + block * 32 + 0, offsets, 1);
        __m256 a1 = _mm256_i32gather_ps(activation + block * 32 + 1, offsets, 1);
        __m256 a2 = _mm256_i32gather_ps(activation + block * 32 + 2, offsets, 1);
        __m256 a3 = _mm256_i32gather_ps(activation + block * 32 + 3, offsets, 1);
        for (int index = 1; index < 16; index += 2)
        {
            lut[index] = a0;
            lut[index] = (index & 2) ? _mm256_add_ps(lut[index], a1) : _mm256_sub_ps(lut[index], a1);
            lut[index] = (index & 4) ? _mm256_add_ps(lut[index], a2) : _mm256_sub_ps(lut[index], a2);
            lut[index] = (index & 8) ? _mm256_add_ps(lut[index], a3) : _mm256_sub_ps(lut[index], a3);
        }
        for (int index = 0; index < 16; index += 2) lut[index] = _mm256_sub_ps(_mm256_setzero_ps(), lut[15 - index]);
        bias += horizontal_sum(lut[0]);
        for (int index = 0; index < 16; ++index) lut[index] = _mm256_mul_ps(lut[index], inverse);

        __m256i packed[4];
        for (int group = 0; group < 4; ++group)
        {
            __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(lut[group * 4 + 0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(lut[group * 4 + 1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(lut[group * 4 + 2], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(lut[group * 4 + 3], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            i0 = _mm256_packs_epi32(i0, i1);
            i2 = _mm256_packs_epi32(i2, i3);
            packed[group] = _mm256_shuffle_epi8(_mm256_packs_epi16(i0, i2), shuffle);
        }

        int32_t* dst = reinterpret_cast<int32_t*>(qlut + block * 8 * 16);
        alignas(32) int32_t lanes[8];
        for (int group = 0; group < 4; ++group)
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), packed[group]);
            for (int lane = 0; lane < 8; ++lane) dst[lane * 4 + group] = lanes[lane];
        }
    }
    *lut_bias = bias;
}

struct PackedWeights
{
    int logical_m = 0;
    int k = 0;
    int bits = 0;
    ScheduleConfig schedule;
    std::vector<uint8_t> data;
};

struct PackedScales
{
    int logical_m = 0;
    int k = 0;
    int bits = 0;
    ScheduleConfig schedule;
    std::vector<fp16_t> data;
};

template <int Bits>
PackedWeights pack_weights_official(const std::vector<uint8_t>& qweights, int logical_m, int k, const ScheduleConfig& schedule)
{
    const int expanded_m = logical_m * Bits;
    const int k_groups = k / G;
    const int m_tiles = expanded_m / schedule.bm;
    const int k_tiles = k_groups / schedule.kfactor;
    const int blocks32 = schedule.bm / 32;
    if (static_cast<int64_t>(qweights.size()) != static_cast<int64_t>(logical_m) * k) throw std::invalid_argument("qweights size mismatch");
    PackedWeights packed{logical_m, k, Bits, schedule, std::vector<uint8_t>(static_cast<size_t>(m_tiles) * k_tiles * blocks32 * schedule.kfactor * SIMD_IN, 0)};

    for (int logical_row = 0; logical_row < logical_m; ++logical_row)
    {
        const int logical_block = logical_row / SIMD_OUT;
        const int lane = logical_row % SIMD_OUT;
        for (int bit = 0; bit < Bits; ++bit)
        {
            const int expanded_row = (logical_block * Bits + bit) * SIMD_OUT + lane;
            const int mo = expanded_row / schedule.bm;
            const int local_m = expanded_row % schedule.bm;
            const int block32 = local_m / 32;
            const int group8 = (local_m % 32) / 8;
            const int packed_lane = local_m % 8;
            const int byte_in_vector = (group8 == 0 || group8 == 2) ? packed_lane : 8 + packed_lane;
            const bool high_nibble = group8 >= 2;

            for (int kg = 0; kg < k_groups; ++kg)
            {
                const int ko = kg / schedule.kfactor;
                const int ki = kg % schedule.kfactor;
                const size_t qbase = static_cast<size_t>(logical_row) * k + kg * G;
                const uint8_t index = static_cast<uint8_t>((((qweights[qbase + 3] >> bit) & 1) << 3) | (((qweights[qbase + 2] >> bit) & 1) << 2) | (((qweights[qbase + 1] >> bit) & 1) << 1) | ((qweights[qbase + 0] >> bit) & 1));
                const size_t offset = (((static_cast<size_t>(mo) * k_tiles + ko) * blocks32 + block32) * schedule.kfactor + ki) * SIMD_IN + byte_in_vector;
                if (high_nibble) packed.data[offset] |= static_cast<uint8_t>(index << 4);
                else packed.data[offset] |= index;
            }
        }
    }
    return packed;
}

template <int Bits>
PackedScales pack_scales_official(const std::vector<fp16_t>& scales, int logical_m, int k, const ScheduleConfig& schedule)
{
    const int expanded_m = logical_m * Bits;
    const int m_tiles = expanded_m / schedule.bm;
    const int logical_per_tile = schedule.bm / Bits;
    const int weight_groups = k / WEIGHT_GROUP_SIZE;
    if (static_cast<int64_t>(scales.size()) != static_cast<int64_t>(logical_m) * weight_groups) throw std::invalid_argument("scales size mismatch");
    PackedScales packed{logical_m, k, Bits, schedule, std::vector<fp16_t>(static_cast<size_t>(m_tiles) * weight_groups * logical_per_tile)};
    for (int m = 0; m < logical_m; ++m)
    {
        const int mo = m / logical_per_tile;
        const int local_m = m % logical_per_tile;
        for (int group = 0; group < weight_groups; ++group) packed.data[(static_cast<size_t>(mo) * weight_groups + group) * logical_per_tile + local_m] = scales[static_cast<size_t>(m) * weight_groups + group];
    }
    return packed;
}

inline size_t packed_weight_offset(const PackedWeights& weights, int mo, int ko, int block32, int ki)
{
    const int k_tiles = (weights.k / G) / weights.schedule.kfactor;
    const int blocks32 = weights.schedule.bm / 32;
    return (((static_cast<size_t>(mo) * k_tiles + ko) * blocks32 + block32) * weights.schedule.kfactor + ki) * SIMD_IN;
}

inline const fp16_t* packed_scale_ptr(const PackedScales& scales, int mo, int weight_group, int logical_offset)
{
    const int logical_per_tile = scales.schedule.bm / scales.bits;
    const int weight_groups = scales.k / WEIGHT_GROUP_SIZE;
    return scales.data.data() + (static_cast<size_t>(mo) * weight_groups + weight_group) * logical_per_tile + logical_offset;
}

struct TMacWorkspace
{
    std::vector<int8_t> qlut;
    std::vector<fp16_t> lut_scales;
    std::vector<fp16_t> lut_biases;
    std::vector<float> thread_scratch;
    int n = 0;
    int k = 0;
    int max_threads = 0;
    int bn = 0;
    int bm = 0;

    void resize(int n_value, int k_value, int threads, int bn_value, int bm_value)
    {
        n = n_value;
        k = k_value;
        max_threads = std::max(1, threads);
        bn = bn_value;
        bm = bm_value;
        qlut.resize(static_cast<size_t>(n) * (k / G) * (1 << G));
        lut_scales.resize(static_cast<size_t>(n) * (k / ACT_GROUP_SIZE));
        lut_biases.resize(static_cast<size_t>(n) * (k / ACT_GROUP_SIZE));
        thread_scratch.resize(static_cast<size_t>(max_threads) * bn * bm);
    }
};

struct Timing
{
    double preprocess_ms = 0.0;
    double kernel_ms = 0.0;
    double total_ms = 0.0;
    int requested_threads = 1;
    int active_threads = 1;
};

inline void build_luts_fp16(int n, int k, const fp16_t* activations, TMacWorkspace& workspace)
{
    const int act_groups = k / ACT_GROUP_SIZE;
    const int k_groups = k / G;
    alignas(32) float activation_float[ACT_GROUP_SIZE];
    for (int row = 0; row < n; ++row)
    {
        for (int group = 0; group < act_groups; ++group)
        {
            const fp16_t* src = activations + static_cast<size_t>(row) * k + group * ACT_GROUP_SIZE;
            convert_fp16_block(src, activation_float, ACT_GROUP_SIZE);
            float scale = 0.0f;
            for (int block = 0; block < ACT_GROUP_SIZE / 32; ++block) partial_max_g4_int8_k8(&scale, activation_float + block * 32);
            const fp16_t scale_half = fp32_to_fp16(scale);
            scale = fp16_to_fp32(scale_half);
            float bias = 0.0f;
            int8_t* qlut_dst = workspace.qlut.data() + (static_cast<size_t>(row) * k_groups + group * (ACT_GROUP_SIZE / G)) * (1 << G);
            lut_ctor_g4_int8_avx2(ACT_GROUP_SIZE, qlut_dst, activation_float, scale, &bias);
            workspace.lut_scales[static_cast<size_t>(row) * act_groups + group] = scale_half;
            workspace.lut_biases[static_cast<size_t>(row) * act_groups + group] = fp32_to_fp16(bias);
        }
    }
}

template <int Bits>
inline void compute_output_tile_reference(int no, int mo, int n_size, const PackedWeights& weights, const PackedScales& scales, const TMacWorkspace& workspace, fp16_t* output, float* tile_accumulator)
{
    const int bm = weights.schedule.bm;
    const int kfactor = weights.schedule.kfactor;
    const int logical_per_tile = bm / Bits;
    const int k_groups = weights.k / G;
    const int k_tiles = k_groups / kfactor;
    const int blocks32 = bm / 32;
    const int act_groups = weights.k / ACT_GROUP_SIZE;
    std::memset(tile_accumulator, 0, static_cast<size_t>(n_size) * bm * sizeof(float));
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);

    for (int ko = 0; ko < k_tiles; ++ko)
    {
        const int kg_begin = ko * kfactor;
        const int act_group = (kg_begin * G) / ACT_GROUP_SIZE;
        const int weight_group = (kg_begin * G) / WEIGHT_GROUP_SIZE;
        const bool add_bias = ((kg_begin * G) % ACT_GROUP_SIZE) == 0;

        for (int ni = 0; ni < n_size; ++ni)
        {
            const int row = no + ni;
            const float lut_scale = fp16_to_fp32(workspace.lut_scales[static_cast<size_t>(row) * act_groups + act_group]);
            const float lut_bias = add_bias ? fp16_to_fp32(workspace.lut_biases[static_cast<size_t>(row) * act_groups + act_group]) : 0.0f;
            const __m256 lut_scale_vec = _mm256_set1_ps(lut_scale);
            const __m256 lut_bias_vec = _mm256_set1_ps(lut_bias);
            float* row_accumulator = tile_accumulator + static_cast<size_t>(ni) * bm;
            const int8_t* qlut_base = workspace.qlut.data() + (static_cast<size_t>(row) * k_groups + kg_begin) * (1 << G);

            for (int block32 = 0; block32 < blocks32; ++block32)
            {
                SignedWideningAdder<DEFAULT_KFACTOR> adder;
                const uint8_t* packed_base = weights.data.data() + packed_weight_offset(weights, mo, ko, block32, 0);
                for (int ki = 0; ki < kfactor; ++ki)
                {
                    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed_base + static_cast<size_t>(ki) * SIMD_IN));
                    const __m128i low = _mm_and_si128(packed, nibble_mask);
                    const __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
                    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qlut_base + static_cast<size_t>(ki) * (1 << G)));
                    const __m256i lut_pair = _mm256_set_m128i(lut, lut);
                    adder.push(_mm256_shuffle_epi8(lut_pair, _mm256_set_m128i(high, low)), ki);
                }

                __m256 values[4] = {
                    _mm256_cvtepi32_ps(TMAC_LOW_I16_TO_I32(adder.low)),
                    _mm256_cvtepi32_ps(TMAC_HIGH_I16_TO_I32(adder.low)),
                    _mm256_cvtepi32_ps(TMAC_LOW_I16_TO_I32(adder.high)),
                    _mm256_cvtepi32_ps(TMAC_HIGH_I16_TO_I32(adder.high))
                };

                for (int group8 = 0; group8 < 4; ++group8)
                {
                    const int expanded_group = block32 * 4 + group8;
                    const int bit_plane = expanded_group % Bits;
                    const int logical_block = expanded_group / Bits;
                    const fp16_t* scale_ptr = packed_scale_ptr(scales, mo, weight_group, logical_block * SIMD_OUT);
                    __m256 value = _mm256_mul_ps(values[group8], lut_scale_vec);
                    if (bit_plane == 0 && add_bias) value = _mm256_add_ps(value, lut_bias_vec);
                    value = _mm256_mul_ps(value, load_fp16x8(scale_ptr));
                    const __m256 previous = _mm256_loadu_ps(row_accumulator + expanded_group * SIMD_OUT);
                    _mm256_storeu_ps(row_accumulator + expanded_group * SIMD_OUT, _mm256_add_ps(previous, value));
                }
            }
        }
    }

    for (int ni = 0; ni < n_size; ++ni)
    {
        const float* row_accumulator = tile_accumulator + static_cast<size_t>(ni) * bm;
        fp16_t* output_row = output + static_cast<size_t>(no + ni) * weights.logical_m + mo * logical_per_tile;
        for (int block = 0; block < logical_per_tile / SIMD_OUT; ++block)
        {
            const float* plane = row_accumulator + static_cast<size_t>(block) * Bits * SIMD_OUT;
            __m256 result = _mm256_mul_ps(_mm256_loadu_ps(plane), _mm256_set1_ps(0.5f));
            if constexpr (Bits >= 2) result = _mm256_add_ps(result, _mm256_loadu_ps(plane + SIMD_OUT));
            if constexpr (Bits >= 3) result = _mm256_fmadd_ps(_mm256_loadu_ps(plane + 2 * SIMD_OUT), _mm256_set1_ps(2.0f), result);
            if constexpr (Bits >= 4) result = _mm256_fmadd_ps(_mm256_loadu_ps(plane + 3 * SIMD_OUT), _mm256_set1_ps(4.0f), result);
            store_fp16x8(output_row + block * SIMD_OUT, result);
        }
    }
}

template <int Bits>
inline void compute_output_tile(int no, int mo, int n_size, const PackedWeights& weights, const PackedScales& scales, const TMacWorkspace& workspace, fp16_t* output, float* tile_accumulator)
{
    const int bm = weights.schedule.bm;
    const int kfactor = weights.schedule.kfactor;
    const int logical_per_tile = bm / Bits;
    const int k_groups = weights.k / G;
    const int k_tiles = k_groups / kfactor;
    const int blocks32 = bm / 32;
    const int act_groups = weights.k / ACT_GROUP_SIZE;
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);

    for (int ni = 0; ni < n_size; ++ni)
    {
        const int row = no + ni;
        float* row_accumulator = tile_accumulator + static_cast<size_t>(ni) * bm;

        for (int block32 = 0; block32 < blocks32; ++block32)
        {
            __m256 accumulator0 = _mm256_setzero_ps();
            __m256 accumulator1 = _mm256_setzero_ps();
            __m256 accumulator2 = _mm256_setzero_ps();
            __m256 accumulator3 = _mm256_setzero_ps();

            for (int ko = 0; ko < k_tiles; ++ko)
            {
                const int kg_begin = ko * kfactor;
                const int act_group = (kg_begin * G) / ACT_GROUP_SIZE;
                const int weight_group = (kg_begin * G) / WEIGHT_GROUP_SIZE;
                const bool add_bias = ((kg_begin * G) % ACT_GROUP_SIZE) == 0;
                const int8_t* qlut_base = workspace.qlut.data() + (static_cast<size_t>(row) * k_groups + kg_begin) * (1 << G);
                const uint8_t* packed_base = weights.data.data() + packed_weight_offset(weights, mo, ko, block32, 0);
                SignedWideningAdder<DEFAULT_KFACTOR> adder;

                for (int ki = 0; ki < kfactor; ++ki)
                {
                    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed_base + static_cast<size_t>(ki) * SIMD_IN));
                    const __m128i low = _mm_and_si128(packed, nibble_mask);
                    const __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
                    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qlut_base + static_cast<size_t>(ki) * (1 << G)));
                    const __m256i lut_pair = _mm256_set_m128i(lut, lut);
                    adder.push(_mm256_shuffle_epi8(lut_pair, _mm256_set_m128i(high, low)), ki);
                }

                const __m256 values0 = _mm256_cvtepi32_ps(TMAC_LOW_I16_TO_I32(adder.low));
                const __m256 values1 = _mm256_cvtepi32_ps(TMAC_HIGH_I16_TO_I32(adder.low));
                const __m256 values2 = _mm256_cvtepi32_ps(TMAC_LOW_I16_TO_I32(adder.high));
                const __m256 values3 = _mm256_cvtepi32_ps(TMAC_HIGH_I16_TO_I32(adder.high));
                const __m256 lut_scale_vec = _mm256_set1_ps(fp16_to_fp32(workspace.lut_scales[static_cast<size_t>(row) * act_groups + act_group]));
                const __m256 lut_bias_vec = _mm256_set1_ps(add_bias ? fp16_to_fp32(workspace.lut_biases[static_cast<size_t>(row) * act_groups + act_group]) : 0.0f);
                const int expanded0 = block32 * 4 + 0;
                const int expanded1 = block32 * 4 + 1;
                const int expanded2 = block32 * 4 + 2;
                const int expanded3 = block32 * 4 + 3;
                __m256 value0 = _mm256_mul_ps(values0, lut_scale_vec);
                __m256 value1 = _mm256_mul_ps(values1, lut_scale_vec);
                __m256 value2 = _mm256_mul_ps(values2, lut_scale_vec);
                __m256 value3 = _mm256_mul_ps(values3, lut_scale_vec);
                if (add_bias && expanded0 % Bits == 0) value0 = _mm256_add_ps(value0, lut_bias_vec);
                if (add_bias && expanded1 % Bits == 0) value1 = _mm256_add_ps(value1, lut_bias_vec);
                if (add_bias && expanded2 % Bits == 0) value2 = _mm256_add_ps(value2, lut_bias_vec);
                if (add_bias && expanded3 % Bits == 0) value3 = _mm256_add_ps(value3, lut_bias_vec);
                value0 = _mm256_mul_ps(value0, load_fp16x8(packed_scale_ptr(scales, mo, weight_group, (expanded0 / Bits) * SIMD_OUT)));
                accumulator0 = _mm256_add_ps(accumulator0, value0);
                value1 = _mm256_mul_ps(value1, load_fp16x8(packed_scale_ptr(scales, mo, weight_group, (expanded1 / Bits) * SIMD_OUT)));
                accumulator1 = _mm256_add_ps(accumulator1, value1);
                value2 = _mm256_mul_ps(value2, load_fp16x8(packed_scale_ptr(scales, mo, weight_group, (expanded2 / Bits) * SIMD_OUT)));
                accumulator2 = _mm256_add_ps(accumulator2, value2);
                value3 = _mm256_mul_ps(value3, load_fp16x8(packed_scale_ptr(scales, mo, weight_group, (expanded3 / Bits) * SIMD_OUT)));
                accumulator3 = _mm256_add_ps(accumulator3, value3);
            }

            _mm256_storeu_ps(row_accumulator + (block32 * 4 + 0) * SIMD_OUT, accumulator0);
            _mm256_storeu_ps(row_accumulator + (block32 * 4 + 1) * SIMD_OUT, accumulator1);
            _mm256_storeu_ps(row_accumulator + (block32 * 4 + 2) * SIMD_OUT, accumulator2);
            _mm256_storeu_ps(row_accumulator + (block32 * 4 + 3) * SIMD_OUT, accumulator3);
        }

        fp16_t* output_row = output + static_cast<size_t>(row) * weights.logical_m + mo * logical_per_tile;
        for (int block = 0; block < logical_per_tile / SIMD_OUT; ++block)
        {
            const float* plane = row_accumulator + static_cast<size_t>(block) * Bits * SIMD_OUT;
            __m256 result = _mm256_mul_ps(_mm256_loadu_ps(plane), _mm256_set1_ps(0.5f));
            if constexpr (Bits >= 2) result = _mm256_add_ps(result, _mm256_loadu_ps(plane + SIMD_OUT));
            if constexpr (Bits >= 3) result = _mm256_fmadd_ps(_mm256_loadu_ps(plane + 2 * SIMD_OUT), _mm256_set1_ps(2.0f), result);
            if constexpr (Bits >= 4) result = _mm256_fmadd_ps(_mm256_loadu_ps(plane + 3 * SIMD_OUT), _mm256_set1_ps(4.0f), result);
            store_fp16x8(output_row + block * SIMD_OUT, result);
        }
    }
}

template <int Bits>
Timing tmac_gemm_fp16(const PackedWeights& weights, const PackedScales& scales, int n, const fp16_t* activations, fp16_t* output, int num_threads, TMacWorkspace& workspace, bool reference_kernel = false)
{
    if (weights.bits != Bits || scales.bits != Bits || weights.logical_m != scales.logical_m || weights.k != scales.k) throw std::invalid_argument("Packed metadata mismatch");
    if (weights.schedule.bm != scales.schedule.bm || weights.schedule.bn != scales.schedule.bn || weights.schedule.kfactor != scales.schedule.kfactor) throw std::invalid_argument("Weight and scale schedules differ");
    const int bm = weights.schedule.bm;
    const int bn = weights.schedule.bn;
    const int expanded_m = weights.logical_m * Bits;
    const int m_tiles = expanded_m / bm;
    const int n_tiles = (n + bn - 1) / bn;
    const int requested_threads = std::max(1, num_threads);
    const bool parallel_n = n_tiles >= requested_threads;
    const int work_items = parallel_n ? n_tiles : m_tiles;
    const int active_threads = std::min(requested_threads, std::max(1, work_items));
    workspace.resize(n, weights.k, active_threads, bn, bm);
    omp_set_dynamic(0);
    omp_set_max_active_levels(1);

    const auto total_start = std::chrono::high_resolution_clock::now();
    build_luts_fp16(n, weights.k, activations, workspace);
    const auto preprocess_end = std::chrono::high_resolution_clock::now();

    if (active_threads == 1)
    {
        float* scratch = workspace.thread_scratch.data();
        for (int no_index = 0; no_index < n_tiles; ++no_index)
        {
            const int no = no_index * bn;
            const int n_size = std::min(bn, n - no);
            for (int mo = 0; mo < m_tiles; ++mo)
            {
                if (reference_kernel) compute_output_tile_reference<Bits>(no, mo, n_size, weights, scales, workspace, output, scratch);
                else compute_output_tile<Bits>(no, mo, n_size, weights, scales, workspace, output, scratch);
            }
        }
    }
    else if (parallel_n)
    {
        #pragma omp parallel for schedule(static) num_threads(active_threads)
        for (int no_index = 0; no_index < n_tiles; ++no_index)
        {
            const int tid = omp_get_thread_num();
            float* scratch = workspace.thread_scratch.data() + static_cast<size_t>(tid) * bn * bm;
            const int no = no_index * bn;
            const int n_size = std::min(bn, n - no);
            for (int mo = 0; mo < m_tiles; ++mo)
            {
                if (reference_kernel) compute_output_tile_reference<Bits>(no, mo, n_size, weights, scales, workspace, output, scratch);
                else compute_output_tile<Bits>(no, mo, n_size, weights, scales, workspace, output, scratch);
            }
        }
    }
    else
    {
        #pragma omp parallel for schedule(static) num_threads(active_threads)
        for (int mo = 0; mo < m_tiles; ++mo)
        {
            const int tid = omp_get_thread_num();
            float* scratch = workspace.thread_scratch.data() + static_cast<size_t>(tid) * bn * bm;
            for (int no_index = 0; no_index < n_tiles; ++no_index)
            {
                const int no = no_index * bn;
                const int n_size = std::min(bn, n - no);
                if (reference_kernel) compute_output_tile_reference<Bits>(no, mo, n_size, weights, scales, workspace, output, scratch);
                else compute_output_tile<Bits>(no, mo, n_size, weights, scales, workspace, output, scratch);
            }
        }
    }

    const auto kernel_end = std::chrono::high_resolution_clock::now();
    Timing timing;
    timing.preprocess_ms = std::chrono::duration<double, std::milli>(preprocess_end - total_start).count();
    timing.kernel_ms = std::chrono::duration<double, std::milli>(kernel_end - preprocess_end).count();
    timing.total_ms = std::chrono::duration<double, std::milli>(kernel_end - total_start).count();
    timing.requested_threads = requested_threads;
    timing.active_threads = active_threads;
    return timing;
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

template <int Bits>
void generate_inputs(int logical_m, int k, int n, std::vector<uint8_t>& qweights, std::vector<fp16_t>& activations, std::vector<fp16_t>& scales, uint32_t seed = 42)
{
    const int weight_groups = k / WEIGHT_GROUP_SIZE;
    qweights.resize(static_cast<size_t>(logical_m) * k);
    activations.resize(static_cast<size_t>(n) * k);
    scales.resize(static_cast<size_t>(logical_m) * weight_groups);
    const uint32_t mask = (1U << Bits) - 1U;
    for (size_t i = 0; i < qweights.size(); ++i) qweights[i] = static_cast<uint8_t>(mix32(static_cast<uint32_t>(i) ^ seed) & mask);
    for (size_t i = 0; i < activations.size(); ++i) activations[i] = fp32_to_fp16(deterministic_float(static_cast<uint32_t>(i), seed + 101, -1.0f, 1.0f));
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = fp32_to_fp16(deterministic_float(static_cast<uint32_t>(i), seed + 211, 0.02f, 0.15f));
}

template <int Bits>
void naive_gemm_fp16(int logical_m, int k, int n, const std::vector<uint8_t>& qweights, const std::vector<fp16_t>& activations, const std::vector<fp16_t>& scales, std::vector<fp16_t>& output)
{
    const int zero_point = 1 << (Bits - 1);
    const int weight_groups = k / WEIGHT_GROUP_SIZE;
    output.resize(static_cast<size_t>(n) * logical_m);
    for (int row = 0; row < n; ++row)
    {
        for (int m = 0; m < logical_m; ++m)
        {
            double sum = 0.0;
            for (int kk = 0; kk < k; ++kk)
            {
                const int q = static_cast<int>(qweights[static_cast<size_t>(m) * k + kk]) - zero_point;
                const float scale = fp16_to_fp32(scales[static_cast<size_t>(m) * weight_groups + kk / WEIGHT_GROUP_SIZE]);
                sum += static_cast<double>(q) * scale * fp16_to_fp32(activations[static_cast<size_t>(row) * k + kk]);
            }
            output[static_cast<size_t>(row) * logical_m + m] = fp32_to_fp16(static_cast<float>(sum));
        }
    }
}

template <int Bits>
bool verify_packed_weights(const std::vector<uint8_t>& qweights, const PackedWeights& packed)
{
    const int k_groups = packed.k / G;
    const int k_tiles = k_groups / packed.schedule.kfactor;
    const int blocks32 = packed.schedule.bm / 32;
    (void)k_tiles;
    (void)blocks32;
    for (int logical_row = 0; logical_row < packed.logical_m; ++logical_row)
    {
        const int logical_block = logical_row / SIMD_OUT;
        const int lane = logical_row % SIMD_OUT;
        for (int bit = 0; bit < Bits; ++bit)
        {
            const int expanded_row = (logical_block * Bits + bit) * SIMD_OUT + lane;
            const int mo = expanded_row / packed.schedule.bm;
            const int local_m = expanded_row % packed.schedule.bm;
            const int block32 = local_m / 32;
            const int group8 = (local_m % 32) / 8;
            const int packed_lane = local_m % 8;
            const int byte_in_vector = (group8 == 0 || group8 == 2) ? packed_lane : 8 + packed_lane;
            const bool high_nibble = group8 >= 2;
            for (int kg = 0; kg < k_groups; ++kg)
            {
                const int ko = kg / packed.schedule.kfactor;
                const int ki = kg % packed.schedule.kfactor;
                const size_t offset = packed_weight_offset(packed, mo, ko, block32, ki) + byte_in_vector;
                const uint8_t index = high_nibble ? static_cast<uint8_t>((packed.data[offset] >> 4) & 0x0f) : static_cast<uint8_t>(packed.data[offset] & 0x0f);
                for (int g = 0; g < G; ++g)
                {
                    const uint8_t expected = static_cast<uint8_t>((qweights[static_cast<size_t>(logical_row) * packed.k + kg * G + g] >> bit) & 1);
                    if (((index >> g) & 1) != expected) return false;
                }
            }
        }
    }
    return true;
}

inline double checksum_fp16(const fp16_t* values, size_t count)
{
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) sum += fp16_to_fp32(values[i]);
    return sum;
}

} // namespace tmac

#endif