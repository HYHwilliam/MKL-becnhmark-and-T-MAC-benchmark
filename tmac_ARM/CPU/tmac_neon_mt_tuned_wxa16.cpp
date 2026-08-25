// Tuned multi-thread T-MAC NEON benchmark.
// Baseline source: v6 hardened MT implementation.
// The production packing/LUT/NEON compute path remains frozen.
// Only target-side schedule selection is added: full official BM/BN/KF GridSearch.
// Official rule: parallelize N tiles when N / BN >= num_threads, otherwise parallelize M tiles.
#include "benchmark_common.h"
#include "tmac_layout.h"

#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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

inline fp16 half_from_bits(uint16_t bits)
{
    fp16 value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

struct GuardedOutput
{
    static constexpr size_t kGuardElements = 64;
    static constexpr uint16_t kFrontGuardBits = 0x3555;
    static constexpr uint16_t kBackGuardBits = 0x3aaa;
    static constexpr uint16_t kUnwrittenBits = 0x7e55;

    explicit GuardedOutput(size_t elements)
        : elements_(elements), storage_(elements + 2 * kGuardElements)
    {
        std::fill(storage_.begin(), storage_.begin() + kGuardElements, half_from_bits(kFrontGuardBits));
        std::fill(storage_.begin() + kGuardElements,
                  storage_.begin() + kGuardElements + elements_, half_from_bits(kUnwrittenBits));
        std::fill(storage_.begin() + kGuardElements + elements_,
                  storage_.end(), half_from_bits(kBackGuardBits));
    }

    fp16* data() { return storage_.data() + kGuardElements; }
    const fp16* data() const { return storage_.data() + kGuardElements; }

    bool guards_ok() const
    {
        for (size_t i = 0; i < kGuardElements; ++i)
            if (half_bits(storage_[i]) != kFrontGuardBits) return false;
        for (size_t i = 0; i < kGuardElements; ++i)
            if (half_bits(storage_[kGuardElements + elements_ + i]) != kBackGuardBits) return false;
        return true;
    }

    size_t unwritten_count() const
    {
        size_t count = 0;
        for (size_t i = 0; i < elements_; ++i)
            if (half_bits(data()[i]) == kUnwrittenBits) ++count;
        return count;
    }

    std::vector<fp16> copy_output() const
    {
        return std::vector<fp16>(data(), data() + elements_);
    }

private:
    size_t elements_ = 0;
    std::vector<fp16> storage_;
};

struct SharedWorkspace
{
    std::vector<int8_t> quantized_luts;
    std::vector<fp16> lut_scales;
    std::vector<fp16> lut_biases;
};

struct ThreadWorkspace
{
    std::vector<fp16> expanded;
};

struct FrozenSingleThreadWorkspace
{
    std::vector<int8_t> quantized_luts;
    std::vector<fp16> lut_scales;
    std::vector<fp16> lut_biases;
    std::vector<fp16> expanded;
};

class StaticThreadPool
{
public:
    explicit StaticThreadPool(int num_threads) : num_threads_(num_threads)
    {
        if (num_threads_ <= 0) throw std::invalid_argument("num_threads must be positive");
        if (num_threads_ == 1) return;

        workers_.reserve(static_cast<size_t>(num_threads_ - 1));
        for (int task_id = 1; task_id < num_threads_; ++task_id)
            workers_.emplace_back([this, task_id] { worker_loop(task_id); });
    }

    StaticThreadPool(const StaticThreadPool&) = delete;
    StaticThreadPool& operator=(const StaticThreadPool&) = delete;

    ~StaticThreadPool()
    {
        if (workers_.empty()) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (std::thread& worker : workers_) worker.join();
    }

    int num_threads() const { return num_threads_; }

    template <typename Function>
    void parallel_for(int extent, Function&& function)
    {
        if (extent <= 0) return;
        if (num_threads_ == 1)
        {
            for (int index = 0; index < extent; ++index) function(index, 0);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            extent_ = extent;
            function_ = std::forward<Function>(function);
            exception_ = nullptr;
            finished_workers_ = 0;
            ++generation_;
        }
        start_cv_.notify_all();

        try
        {
            run_task_range(0, extent, function_);
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!exception_) exception_ = std::current_exception();
        }

        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this] { return finished_workers_ == num_threads_ - 1; });
        function_ = nullptr;
        if (exception_) std::rethrow_exception(exception_);
    }

private:
    void run_task_range(int task_id, int extent, const std::function<void(int, int)>& function)
    {
        const int step = (extent + num_threads_ - 1) / num_threads_;
        const int begin = std::min(task_id * step, extent);
        const int end = std::min((task_id + 1) * step, extent);
        for (int index = begin; index < end; ++index) function(index, task_id);
    }

    void worker_loop(int task_id)
    {
        size_t observed_generation = 0;
        while (true)
        {
            std::function<void(int, int)> function;
            int extent = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_cv_.wait(lock, [&] { return stop_ || generation_ != observed_generation; });
                if (stop_) return;
                observed_generation = generation_;
                function = function_;
                extent = extent_;
            }

            try
            {
                run_task_range(task_id, extent, function);
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!exception_) exception_ = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++finished_workers_;
                if (finished_workers_ == num_threads_ - 1) done_cv_.notify_one();
            }
        }
    }

    int num_threads_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::function<void(int, int)> function_;
    std::exception_ptr exception_;
    int extent_ = 0;
    int finished_workers_ = 0;
    size_t generation_ = 0;
    bool stop_ = false;
};

struct Runtime
{
    explicit Runtime(int num_threads) : pool(num_threads), thread_workspaces(static_cast<size_t>(num_threads)) {}

    void prepare(int n, int k, int bn, int bm)
    {
        const int k_groups = k / kLutGroupSize;
        const int activation_groups = k / kActivationGroupSize;
        shared.quantized_luts.resize(static_cast<size_t>(n) * k_groups * kLutEntryCount);
        shared.lut_scales.resize(static_cast<size_t>(n) * activation_groups);
        shared.lut_biases.resize(static_cast<size_t>(n) * activation_groups);
        for (ThreadWorkspace& workspace : thread_workspaces)
            workspace.expanded.resize(static_cast<size_t>(bn) * bm);
    }

    SharedWorkspace shared;
    StaticThreadPool pool;
    std::vector<ThreadWorkspace> thread_workspaces;
};

enum class ParallelAxis
{
    Serial,
    N,
    M,
};

inline ParallelAxis official_parallel_axis(int n, int bn, int num_threads)
{
    if (num_threads <= 1) return ParallelAxis::Serial;
    return (n / bn >= num_threads) ? ParallelAxis::N : ParallelAxis::M;
}

inline const char* parallel_axis_name(ParallelAxis axis)
{
    if (axis == ParallelAxis::N) return "N";
    if (axis == ParallelAxis::M) return "M";
    return "Serial";
}

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

inline void build_activation_luts(int n, int k, const fp16* activations, SharedWorkspace& workspace)
{
    const int k_groups = k / kLutGroupSize;
    const int activation_groups = k / kActivationGroupSize;
    const size_t expected_luts = static_cast<size_t>(n) * k_groups * kLutEntryCount;
    const size_t expected_groups = static_cast<size_t>(n) * activation_groups;
    if (workspace.quantized_luts.size() != expected_luts || workspace.lut_scales.size() != expected_groups
        || workspace.lut_biases.size() != expected_groups)
        throw std::logic_error("Activation workspace must be prepared before T-MAC execution");

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

// Verification-only oracle copied from the frozen single-thread NEON baseline.
// Keep this path independent from compute_tile/tmac_neon_mt so scheduler/runtime
// regressions cannot validate themselves through the same dispatch path.
inline void build_frozen_activation_luts(int n, int k, const fp16* activations, FrozenSingleThreadWorkspace& workspace)
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

inline float16x8_t frozen_reconstruct_lookup(float16x8_t lookup_sum, fp16 lut_scale, fp16 lut_bias, bool add_bias)
{
    return add_bias ? vfmaq_n_f16(vdupq_n_f16(lut_bias), lookup_sum, lut_scale) : vmulq_n_f16(lookup_sum, lut_scale);
}

template <int Bits>
__attribute__((noinline)) void frozen_single_thread_neon(const PackedWeights& weights, const PackedScales<fp16>& scales, int n,
                                         const fp16* activations, fp16* output, FrozenSingleThreadWorkspace& workspace)
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

    build_frozen_activation_luts(n, k, activations, workspace);
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
                            const float16x8_t reconstructed = frozen_reconstruct_lookup(lookup_sums[group8], lut_scale, lut_bias, add_bias && bit_plane == 0);
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


template <int Bits>
inline void compute_tile(const PackedWeights& weights, const PackedScales<fp16>& scales, int n, int no, int m_tile,
                         const SharedWorkspace& shared, ThreadWorkspace& thread_workspace, fp16* output)
{
    const int logical_m = weights.logical_m;
    const int k = weights.k;
    const int bm = weights.schedule.bm;
    const int bn = weights.schedule.bn;
    constexpr int kfactor = kDefaultKFactor;
    const int logical_rows_per_tile = bm / Bits;
    const int blocks32 = bm / kExpandedRowsPerVector;
    const int k_groups = k / kLutGroupSize;
    const int k_tiles = k_groups / kfactor;
    const int activation_groups = k / kActivationGroupSize;
    const int n_size = std::min(bn, n - no);
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0f);

    fp16* tile_accumulator = thread_workspace.expanded.data();
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
            const int8_t* qlut_base = shared.quantized_luts.data() +
                (static_cast<size_t>(row) * k_groups + kg_begin) * kLutEntryCount;

            int8x16_t lut_registers[kDefaultKFactor];
            for (int k_inner = 0; k_inner < kfactor; ++k_inner)
                lut_registers[k_inner] = vld1q_s8(qlut_base + static_cast<size_t>(k_inner) * kLutEntryCount);

            for (int block32 = 0; block32 < blocks32; ++block32)
            {
                const uint8_t* packed_base = weights.bytes.data() +
                    packed_weight_offset(weights, m_tile, k_tile, block32, 0);

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
                    const uint8x16_t packed_indices_even =
                        vld1q_u8(packed_base + static_cast<size_t>(k_inner) * kPackedByteLanes);
                    const uint8x16_t packed_indices_odd =
                        vld1q_u8(packed_base + static_cast<size_t>(k_inner + 1) * kPackedByteLanes);

                    const int8x16_t bottom_even =
                        vqtbl1q_s8(lut_registers[k_inner], vandq_u8(packed_indices_even, nibble_mask));
                    const int8x16_t bottom_odd =
                        vqtbl1q_s8(lut_registers[k_inner + 1], vandq_u8(packed_indices_odd, nibble_mask));
                    const int8x16_t top_even =
                        vqtbl1q_s8(lut_registers[k_inner], vshrq_n_u8(packed_indices_even, 4));
                    const int8x16_t top_odd =
                        vqtbl1q_s8(lut_registers[k_inner + 1], vshrq_n_u8(packed_indices_odd, 4));

                    bottom_low = vaddq_s16(bottom_low,
                        vaddl_s8(vget_low_s8(bottom_even), vget_low_s8(bottom_odd)));
                    bottom_high = vaddq_s16(bottom_high, vaddl_high_s8(bottom_even, bottom_odd));
                    top_low = vaddq_s16(top_low,
                        vaddl_s8(vget_low_s8(top_even), vget_low_s8(top_odd)));
                    top_high = vaddq_s16(top_high, vaddl_high_s8(top_even, top_odd));
                }

                const float16x8_t lookup_sums[4] = {
                    vcvtq_f16_s16(bottom_low), vcvtq_f16_s16(bottom_high),
                    vcvtq_f16_s16(top_low), vcvtq_f16_s16(top_high)
                };
                const fp16 lut_scale =
                    shared.lut_scales[static_cast<size_t>(row) * activation_groups + activation_group];
                const fp16 lut_bias =
                    shared.lut_biases[static_cast<size_t>(row) * activation_groups + activation_group];

                for (int group8 = 0; group8 < 4; ++group8)
                {
                    const int expanded_group8 = block32 * 4 + group8;
                    const int bit_plane = expanded_group8 % Bits;
                    const int logical_block8 = expanded_group8 / Bits;
                    const float16x8_t weight_scale = vld1q_f16(
                        packed_scale_ptr(scales, m_tile, weight_group, logical_block8 * kFp16OutputLanes));
                    const float16x8_t reconstructed = reconstruct_lookup(
                        lookup_sums[group8], lut_scale, lut_bias, add_bias && bit_plane == 0);
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

            vst1q_f16(output_tile + block8 * kFp16OutputLanes,
                       vcombine_f16(vcvt_f16_f32(low), vcvt_f16_f32(high)));
        }
    }
}

template <int Bits>
__attribute__((noinline)) void tmac_neon_mt(const PackedWeights& weights, const PackedScales<fp16>& scales, int n,
                                            const fp16* activations, fp16* output, Runtime& runtime)
{
    const int logical_m = weights.logical_m;
    const int k = weights.k;
    const int bm = weights.schedule.bm;
    const int bn = weights.schedule.bn;
    constexpr int kfactor = kDefaultKFactor;
    if (weights.schedule.kfactor != kfactor)
        throw std::invalid_argument("NEON kernel requires official kfactor=16 for g=4, act_group=64, group_size=128");

    const int m_tiles = logical_m * Bits / bm;
    const int n_tiles = (n + bn - 1) / bn;

    // Official T-MAC wrapper keeps the activation preprocessor single-threaded.
    build_activation_luts(n, k, activations, runtime.shared);

    const ParallelAxis axis = official_parallel_axis(n, bn, runtime.pool.num_threads());
    if (axis == ParallelAxis::Serial)
    {
        ThreadWorkspace& thread_workspace = runtime.thread_workspaces[0];
        for (int no_tile = 0; no_tile < n_tiles; ++no_tile)
        {
            const int no = no_tile * bn;
            for (int m_tile = 0; m_tile < m_tiles; ++m_tile)
                compute_tile<Bits>(weights, scales, n, no, m_tile, runtime.shared, thread_workspace, output);
        }
        return;
    }

    if (axis == ParallelAxis::N)
    {
        runtime.pool.parallel_for(n_tiles, [&](int no_tile, int worker_id)
        {
            const int no = no_tile * bn;
            ThreadWorkspace& thread_workspace = runtime.thread_workspaces[static_cast<size_t>(worker_id)];
            for (int m_tile = 0; m_tile < m_tiles; ++m_tile)
                compute_tile<Bits>(weights, scales, n, no, m_tile, runtime.shared, thread_workspace, output);
        });
        return;
    }

    for (int no_tile = 0; no_tile < n_tiles; ++no_tile)
    {
        const int no = no_tile * bn;
        runtime.pool.parallel_for(m_tiles, [&](int m_tile, int worker_id)
        {
            ThreadWorkspace& thread_workspace = runtime.thread_workspaces[static_cast<size_t>(worker_id)];
            compute_tile<Bits>(weights, scales, n, no, m_tile, runtime.shared, thread_workspace, output);
        });
    }
}


template <int Bits>
__attribute__((noinline)) void tmac_neon_mt_compute_only_for_tuning(
    const PackedWeights& weights,
    const PackedScales<fp16>& scales,
    int n,
    fp16* output,
    Runtime& runtime)
{
    const int logical_m = weights.logical_m;
    const int bm = weights.schedule.bm;
    const int bn = weights.schedule.bn;
    constexpr int kfactor = kDefaultKFactor;
    if (weights.schedule.kfactor != kfactor)
        throw std::invalid_argument("Tuned NEON kernel requires official kfactor=16 for g=4, act_group=64, group_size=128");

    const int m_tiles = logical_m * Bits / bm;
    const int n_tiles = (n + bn - 1) / bn;

    const ParallelAxis axis = official_parallel_axis(n, bn, runtime.pool.num_threads());
    if (axis == ParallelAxis::Serial)
    {
        ThreadWorkspace& thread_workspace = runtime.thread_workspaces[0];
        for (int no_tile = 0; no_tile < n_tiles; ++no_tile)
        {
            const int no = no_tile * bn;
            for (int m_tile = 0; m_tile < m_tiles; ++m_tile)
                compute_tile<Bits>(weights, scales, n, no, m_tile, runtime.shared, thread_workspace, output);
        }
        return;
    }

    if (axis == ParallelAxis::N)
    {
        runtime.pool.parallel_for(n_tiles, [&](int no_tile, int worker_id)
        {
            const int no = no_tile * bn;
            ThreadWorkspace& thread_workspace = runtime.thread_workspaces[static_cast<size_t>(worker_id)];
            for (int m_tile = 0; m_tile < m_tiles; ++m_tile)
                compute_tile<Bits>(weights, scales, n, no, m_tile, runtime.shared, thread_workspace, output);
        });
        return;
    }

    for (int no_tile = 0; no_tile < n_tiles; ++no_tile)
    {
        const int no = no_tile * bn;
        runtime.pool.parallel_for(m_tiles, [&](int m_tile, int worker_id)
        {
            ThreadWorkspace& thread_workspace = runtime.thread_workspaces[static_cast<size_t>(worker_id)];
            compute_tile<Bits>(weights, scales, n, no, m_tile, runtime.shared, thread_workspace, output);
        });
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

struct MtOptions
{
    int bits = 2;
    int max_size = 8192;
    int threads = 4;
    int tune_number = 10;
    int tune_repeat = 10;
    int tune_cooldown_ms = 100;
    bool verify_only = false;
};

inline MtOptions parse_mt_options(int argc, char** argv)
{
    MtOptions options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--bits" && i + 1 < argc) options.bits = std::stoi(argv[++i]);
        else if (arg == "--max-size" && i + 1 < argc) options.max_size = std::stoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) options.threads = std::stoi(argv[++i]);
        else if (arg == "--tune-number" && i + 1 < argc) options.tune_number = std::stoi(argv[++i]);
        else if (arg == "--tune-repeat" && i + 1 < argc) options.tune_repeat = std::stoi(argv[++i]);
        else if (arg == "--tune-cooldown-ms" && i + 1 < argc) options.tune_cooldown_ms = std::stoi(argv[++i]);
        else if (arg == "--verify-only") options.verify_only = true;
        else throw std::invalid_argument(
            "Usage: program [--bits 2|3|4] [--threads N] "
            "[--max-size 256|1024|2048|4096|8192] "
            "[--tune-number N] [--tune-repeat N] [--tune-cooldown-ms MS] [--verify-only]");
    }

    if (options.bits < 2 || options.bits > 4)
        throw std::invalid_argument("--bits must be 2, 3, or 4");
    if (options.threads <= 0 || options.threads > 16)
        throw std::invalid_argument("--threads must be between 1 and 16");
    if (options.tune_number <= 0)
        throw std::invalid_argument("--tune-number must be positive");
    if (options.tune_repeat <= 0)
        throw std::invalid_argument("--tune-repeat must be positive");
    if (options.tune_cooldown_ms < 0)
        throw std::invalid_argument("--tune-cooldown-ms must be non-negative");
    return options;
}

template <int Bits>
bool verify_case(Runtime& runtime, int m, int k, int n, int mode, const Schedule* schedule_override = nullptr,
                 const char* label = nullptr, ParallelAxis expected_axis = ParallelAxis::Serial, bool check_axis = false,
                 bool hardening = false)
{
    bench::RawInput raw = bench::make_raw_input<Bits>(m, k, n,
        static_cast<uint32_t>(1000 + Bits * 100 + m + k + n + mode));
    std::vector<fp16> activations = bench::cast_to_half<fp16>(raw.activations);
    std::vector<fp16> scales = bench::cast_to_half<fp16>(raw.scales);
    if (mode == 1) std::fill(activations.begin(), activations.end(), to_half(0.0f));
    if (mode == 2) for (size_t i = 0; i < activations.size(); ++i) activations[i] = to_half((i & 1U) ? 1.0f : -1.0f);
    if (mode == 3) std::fill(raw.qweights.begin(), raw.qweights.end(), 0);
    if (mode == 4) std::fill(raw.qweights.begin(), raw.qweights.end(), static_cast<uint8_t>((1U << Bits) - 1U));
    if (mode == 5) std::fill(scales.begin(), scales.end(), to_half(1.0f));
    if (mode == 6) for (size_t i = 0; i < raw.qweights.size(); ++i)
        raw.qweights[i] = (i & 1U) ? static_cast<uint8_t>((1U << Bits) - 1U) : 0;
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

    runtime.prepare(n, k, schedule.bn, schedule.bm);
    GuardedOutput guarded_actual(static_cast<size_t>(n) * m);
    std::vector<fp16> independent_output;
    std::vector<fp16> dense;
    tmac_neon_mt<Bits>(packed_weights, packed_scales, n, activations.data(), guarded_actual.data(), runtime);
    const bool guard_ok = guarded_actual.guards_ok();
    const size_t unwritten = guarded_actual.unwritten_count();
    const std::vector<fp16> actual = guarded_actual.copy_output();

    const bench::OfficialLutReference<fp16> independent_lut = bench::build_official_lut_reference(n, k, activations);
    const bool lut_ok = bench::lut_reference_matches(
        independent_lut, runtime.shared.quantized_luts, runtime.shared.lut_scales, runtime.shared.lut_biases);
    independent_neon_reference<Bits>(m, k, n, raw.qweights, activations, scales, independent_lut, independent_output);
    bench::dense_reference<Bits>(m, k, n, raw.qweights, activations, scales, dense);

    size_t kernel_mismatches = 0;
    for (size_t i = 0; i < actual.size(); ++i)
        if (half_bits(actual[i]) != half_bits(independent_output[i])) ++kernel_mismatches;

    size_t st_mt_mismatches = 0;
    if (hardening)
    {
        FrozenSingleThreadWorkspace frozen_workspace;
        std::vector<fp16> frozen_output(static_cast<size_t>(n) * m);
        frozen_single_thread_neon<Bits>(packed_weights, packed_scales, n, activations.data(),
                                        frozen_output.data(), frozen_workspace);
        for (size_t i = 0; i < actual.size(); ++i)
            if (half_bits(actual[i]) != half_bits(frozen_output[i])) ++st_mt_mismatches;
    }

    const ParallelAxis actual_axis = official_parallel_axis(n, schedule.bn, runtime.pool.num_threads());
    const bool axis_ok = !check_axis || actual_axis == expected_axis;
    const bench::Accuracy accuracy = bench::compare_outputs(actual, dense);
    const bool pass = packing_ok && lut_ok && kernel_mismatches == 0 && st_mt_mismatches == 0
        && guard_ok && unwritten == 0 && accuracy.non_finite == 0
        && accuracy.nmse <= bench::kTmacNmseLimit && axis_ok;

    bench::print_verification_row(m, k, n, label ? label : bench::verification_case_name(mode), accuracy, pass,
                                  schedule.bm, schedule.bn, schedule.kfactor, packing_ok, lut_ok,
                                  static_cast<long long>(kernel_mismatches));
    if (hardening)
    {
        std::cout << "  HARDEN STvsMT=" << (st_mt_mismatches == 0 ? "PASS" : "FAIL")
                  << " mismatches=" << st_mt_mismatches
                  << " OutputCoverage=" << (unwritten == 0 ? "PASS" : "FAIL")
                  << " unwritten=" << unwritten
                  << " Guard=" << (guard_ok ? "PASS" : "FAIL") << '\n';
    }
    if (!guard_ok || unwritten != 0)
        std::cerr << "Output-write hardening failure: Guard=" << (guard_ok ? "PASS" : "FAIL")
                  << ", unwritten=" << unwritten << '\n';
    if (!axis_ok)
        std::cerr << "Parallel-axis mismatch: expected " << parallel_axis_name(expected_axis)
                  << ", got " << parallel_axis_name(actual_axis) << '\n';
    return pass;
}

template <int Bits>
Schedule schedule_with_bn(int m, int n, int k, int target_bn)
{
    const Schedule preferred = choose_single_thread_schedule<Bits>(m, n, k);
    for (const Schedule& schedule : official_schedule_candidates<Bits>(m, n, k))
        if (schedule.bm == preferred.bm && schedule.kfactor == preferred.kfactor && schedule.bn == target_bn)
            return schedule;
    for (const Schedule& schedule : official_schedule_candidates<Bits>(m, n, k))
        if (schedule.bn == target_bn && schedule.kfactor == kDefaultKFactor)
            return schedule;
    throw std::invalid_argument("No requested official BN schedule exists");
}

template <int Bits>
bool verify_parallel_branches(Runtime& runtime)
{
    bool pass = true;
    const int threads = runtime.pool.num_threads();
    if (threads <= 1) return pass;

    constexpr int k = 256;
    constexpr int n_parallel_n = 128;
    int target_bn = 8;
    if (threads <= 2) target_bn = 64;
    else if (threads <= 4) target_bn = 32;
    else if (threads <= 8) target_bn = 16;
    else target_bn = 8;

    if (n_parallel_n / target_bn >= threads)
    {
        const Schedule n_schedule = schedule_with_bn<Bits>(256, n_parallel_n, k, target_bn);
        pass &= verify_case<Bits>(runtime, 256, k, n_parallel_n, 0, &n_schedule, "MT_NSplit", ParallelAxis::N, true);
    }

    int m_parallel_m = 1024;
    if constexpr (Bits == 2)
    {
        if (threads > 8) m_parallel_m = 4096;
        else if (threads > 4) m_parallel_m = 2048;
    }
    else
    {
        if (threads > 8) m_parallel_m = 2048;
    }
    const Schedule m_schedule = choose_single_thread_schedule<Bits>(m_parallel_m, 1, k);
    pass &= verify_case<Bits>(runtime, m_parallel_m, k, 1, 0, &m_schedule, "MT_MSplit", ParallelAxis::M, true);
    return pass;
}

template <int Bits>
bool verify_schedule_sweep(Runtime& runtime)
{
    const int sweep_m = Bits == 3 ? 768 : Bits == 2 ? 2560 : 1280;
    constexpr int k = 256;
    bool pass = true;
    for (const Schedule& schedule : official_schedule_candidates<Bits>(sweep_m, 1, k))
        pass &= verify_case<Bits>(runtime, sweep_m, k, 1, 0, &schedule, "BMSweep");

    constexpr int bn_m = 256;
    constexpr int bn_n = 128;
    const Schedule preferred = choose_single_thread_schedule<Bits>(bn_m, bn_n, k);
    for (const Schedule& schedule : official_schedule_candidates<Bits>(bn_m, bn_n, k))
        if (schedule.bm == preferred.bm && schedule.kfactor == preferred.kfactor)
            pass &= verify_case<Bits>(runtime, bn_m, k, bn_n, 0, &schedule, "BNSweep");
    return pass;
}


template <int Bits>
Schedule hardening_schedule(int m, int n, int k, int target_bm, int target_bn)
{
    for (const Schedule& schedule : official_schedule_candidates<Bits>(m, n, k))
        if (schedule.bm == target_bm && schedule.bn == target_bn && schedule.kfactor == kDefaultKFactor)
            return schedule;
    throw std::invalid_argument("No requested hardening schedule exists in official candidate set");
}

template <int Bits>
bool verify_hardening_suite(Runtime& runtime)
{
    bool pass = true;
    const int threads = runtime.pool.num_threads();
    constexpr int target_bm = Bits == 3 ? 384 : 512;
    constexpr int rows_per_tile = target_bm / Bits;

    for (int k : {384, 640})
    {
        const Schedule schedule = hardening_schedule<Bits>(256, 8, k, target_bm, 8);
        pass &= verify_case<Bits>(runtime, 256, k, 8, 0, &schedule,
                                  k == 384 ? "HardK384" : "HardK640",
                                  official_parallel_axis(8, 8, threads), true, true);
    }

    if (threads > 1)
    {
        for (int tile_count : {threads - 1, threads, threads + 1})
        {
            const int m = tile_count * rows_per_tile;
            const int k = tile_count == threads ? 640 : 384;
            const Schedule schedule = hardening_schedule<Bits>(m, 1, k, target_bm, 8);
            const char* label = tile_count < threads ? "HardMTilesT-1"
                                : tile_count == threads ? "HardMTilesT"
                                                       : "HardMTilesT+1";
            pass &= verify_case<Bits>(runtime, m, k, 1, 0, &schedule, label,
                                      ParallelAxis::M, true, true);
        }

        const int n_before = 8 * (threads - 1);
        const int n_at = 8 * threads;
        const Schedule before_schedule = hardening_schedule<Bits>(256, n_before, 384, target_bm, 8);
        const Schedule at_schedule = hardening_schedule<Bits>(256, n_at, 640, target_bm, 8);
        pass &= verify_case<Bits>(runtime, 256, 384, n_before, 0, &before_schedule,
                                  "HardNBoundary-1", ParallelAxis::M, true, true);
        pass &= verify_case<Bits>(runtime, 256, 640, n_at, 0, &at_schedule,
                                  "HardNBoundary", ParallelAxis::N, true, true);
    }

    return pass;
}

template <int Bits>
bool verify_suite(Runtime& runtime)
{
    bench::print_verification_header("TMAC_NEON_MT", Bits, true, true);
    std::cout << "Threads=" << runtime.pool.num_threads()
              << " | Official rule: N/BN >= threads => N tiles, otherwise M tiles\n";
    bool pass = true;
    for (int n : {1, 8, 32, 128}) pass &= verify_case<Bits>(runtime, 256, 256, n, 0);
    pass &= verify_case<Bits>(runtime, 256, 128, 3, 0, nullptr, "K128Edge");
    pass &= verify_case<Bits>(runtime, 1024, 256, 3, 0);
    pass &= verify_case<Bits>(runtime, 1024, 512, 8, 0);
    pass &= verify_case<Bits>(runtime, 512, 1024, 16, 0);
    for (int mode = 1; mode <= 11; ++mode) pass &= verify_case<Bits>(runtime, 256, 256, 8, mode);
    pass &= verify_parallel_branches<Bits>(runtime);
    pass &= verify_schedule_sweep<Bits>(runtime);
    pass &= verify_hardening_suite<Bits>(runtime);
    bench::print_verification_footer(pass);
    return pass;
}

inline void print_mt_tuned_benchmark_header(int bits, int threads, const MtOptions& options)
{
    std::cout << '\n';
    bench::print_separator();
    std::cout << "TMAC_NEON_MT_TUNED W" << bits << "A16 Benchmark | Threads=" << threads << '\n';
    std::cout << "Matrix: W[MxK], X[NxK], Y[NxM] = X * W^T\n";
    std::cout << "Official parallel rule: N/BN >= threads => N tiles; otherwise M tiles.\n";
    std::cout << "Schedule policy: full official BM/BN/KF GridSearch; one discarded warmup; "
              << "score=mean(repeat costs), number=" << options.tune_number
              << ", repeat=" << options.tune_repeat
              << ", cooldown=" << options.tune_cooldown_ms << "ms.\n";
    std::cout << "Tuning measures qgemm compute only with activation LUT prebuilt. "
              << "Final benchmark timing includes one single-thread activation-LUT construction plus parallel compute.\n";
    std::cout << "Offline weight/scale packing, tuning time, and thread-pool construction are excluded from final latency.\n";
    bench::print_separator('-');
    std::cout << std::left << std::setw(7) << "Kind" << std::right << std::setw(7) << "M" << std::setw(7) << "K"
              << std::setw(7) << "N" << std::setw(7) << "BM" << std::setw(7) << "BN" << std::setw(6) << "KF"
              << std::setw(8) << "Axis" << std::setw(8) << "Tiles" << std::setw(7) << "Warm" << std::setw(7) << "Rep"
              << std::setw(8) << "Sample" << std::setw(14) << "Median(ms)" << std::setw(12) << "P90(ms)"
              << std::setw(12) << "Eq.GOPS" << std::setw(15) << "Checksum" << '\n';
    bench::print_separator('-');
}

struct TuneResult
{
    Schedule schedule;
    double mean_ms = std::numeric_limits<double>::infinity();
    double min_repeat_ms = std::numeric_limits<double>::infinity();
    int candidates = 0;
};

template <int Bits>
TuneResult tune_schedule_full_grid(const bench::Shape& shape,
                                   const std::vector<uint8_t>& qweights,
                                   const std::vector<fp16>& activations,
                                   const std::vector<fp16>& scales,
                                   Runtime& runtime,
                                   const MtOptions& options)
{
    const std::vector<Schedule> candidates =
        official_schedule_candidates<Bits>(shape.m, shape.n, shape.k);

    int max_bm = 0;
    int max_bn = 0;
    for (const Schedule& schedule : candidates)
    {
        max_bm = std::max(max_bm, schedule.bm);
        max_bn = std::max(max_bn, schedule.bn);
        if (schedule.kfactor != kDefaultKFactor)
            throw std::logic_error("Current frozen A16 kernel only supports kfactor=16");
    }

    runtime.prepare(shape.n, shape.k, max_bn, max_bm);
    build_activation_luts(shape.n, shape.k, activations.data(), runtime.shared);

    TuneResult best;
    best.candidates = static_cast<int>(candidates.size());

    for (const Schedule& schedule : candidates)
    {
        const PackedWeights packed_weights =
            pack_weights_tmac<Bits>(qweights, shape.m, shape.k, schedule);
        const PackedScales<fp16> packed_scales =
            pack_scales_tmac<Bits>(scales, shape.m, shape.k, schedule);
        validate_packed_operands<Bits>(packed_weights, packed_scales, shape.n);

        std::vector<fp16> output(static_cast<size_t>(shape.n) * shape.m);

        // One discarded warmup, matching the v10/AutoTVM-style tuning policy.
        tmac_neon_mt_compute_only_for_tuning<Bits>(
            packed_weights, packed_scales, shape.n, output.data(), runtime);

        double repeat_sum_ms = 0.0;
        double min_repeat_ms = std::numeric_limits<double>::infinity();

        for (int repeat_index = 0; repeat_index < options.tune_repeat; ++repeat_index)
        {
            const auto begin = std::chrono::steady_clock::now();
            for (int number_index = 0; number_index < options.tune_number; ++number_index)
            {
                tmac_neon_mt_compute_only_for_tuning<Bits>(
                    packed_weights, packed_scales, shape.n, output.data(), runtime);
            }
            const auto end = std::chrono::steady_clock::now();

            const double repeat_ms =
                std::chrono::duration<double, std::milli>(end - begin).count()
                / static_cast<double>(options.tune_number);

            repeat_sum_ms += repeat_ms;
            min_repeat_ms = std::min(min_repeat_ms, repeat_ms);
        }

        const double mean_ms =
            repeat_sum_ms / static_cast<double>(options.tune_repeat);

        if (mean_ms < best.mean_ms)
        {
            best.schedule = schedule;
            best.mean_ms = mean_ms;
            best.min_repeat_ms = min_repeat_ms;
        }

        if (options.tune_cooldown_ms > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(options.tune_cooldown_ms));
    }

    return best;
}

template <int Bits>
void run_tuned_shape(const bench::Shape& shape, Runtime& runtime, const MtOptions& options)
{
    const bench::RawInput raw = bench::make_raw_input<Bits>(shape);
    const std::vector<fp16> activations = bench::cast_to_half<fp16>(raw.activations);
    const std::vector<fp16> scales = bench::cast_to_half<fp16>(raw.scales);

    const TuneResult tuned =
        tune_schedule_full_grid<Bits>(shape, raw.qweights, activations, scales, runtime, options);

    const Schedule schedule = tuned.schedule;
    const PackedWeights packed_weights =
        pack_weights_tmac<Bits>(raw.qweights, shape.m, shape.k, schedule);
    const PackedScales<fp16> packed_scales =
        pack_scales_tmac<Bits>(scales, shape.m, shape.k, schedule);
    validate_packed_operands<Bits>(packed_weights, packed_scales, shape.n);

    std::vector<fp16> output(static_cast<size_t>(shape.n) * shape.m);
    runtime.prepare(shape.n, shape.k, schedule.bn, schedule.bm);

    const ParallelAxis axis =
        official_parallel_axis(shape.n, schedule.bn, runtime.pool.num_threads());

    std::cout << "TUNE W" << Bits << "A16"
              << " M=" << shape.m
              << " K=" << shape.k
              << " N=" << shape.n
              << " T=" << runtime.pool.num_threads()
              << " candidates=" << tuned.candidates
              << " -> BM=" << schedule.bm
              << " BN=" << schedule.bn
              << " KF=" << schedule.kfactor
              << " mean=" << std::fixed << std::setprecision(4) << tuned.mean_ms << " ms"
              << " min_repeat=" << tuned.min_repeat_ms << " ms"
              << " number=" << options.tune_number
              << " repeat=" << options.tune_repeat
              << '\n';

    const bench::BenchmarkStats stats = bench::measure(shape, [&]
    {
        tmac_neon_mt<Bits>(
            packed_weights, packed_scales, shape.n,
            activations.data(), output.data(), runtime);
    });

    int parallel_tiles = 1;
    if (axis == ParallelAxis::N)
        parallel_tiles = (shape.n + schedule.bn - 1) / schedule.bn;
    else if (axis == ParallelAxis::M)
        parallel_tiles = shape.m * Bits / schedule.bm;

    std::cout << std::left << std::setw(7) << shape.kind() << std::right
              << std::setw(7) << shape.m
              << std::setw(7) << shape.k
              << std::setw(7) << shape.n
              << std::setw(7) << schedule.bm
              << std::setw(7) << schedule.bn
              << std::setw(6) << schedule.kfactor
              << std::setw(8) << parallel_axis_name(axis)
              << std::setw(8) << parallel_tiles
              << std::setw(7) << shape.warmup
              << std::setw(7) << shape.repeat
              << std::setw(8) << shape.samples
              << std::fixed << std::setprecision(4)
              << std::setw(14) << stats.median_ms
              << std::setw(12) << stats.p90_ms
              << std::setw(12) << bench::dense_equivalent_gops(shape, stats.median_ms)
              << std::scientific << std::setprecision(4)
              << std::setw(15) << bench::checksum(output)
              << std::fixed << '\n';
}

template <int Bits>
int run(const MtOptions& options)
{
    Runtime runtime(options.threads);
    if (!verify_suite<Bits>(runtime)) return 1;
    if (options.verify_only) return 0;

    print_mt_tuned_benchmark_header(Bits, options.threads, options);
    for (const bench::Shape& shape : bench::benchmark_shapes())
        if (shape.m <= options.max_size)
            run_tuned_shape<Bits>(shape, runtime, options);
    bench::print_benchmark_footer();
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const MtOptions options = parse_mt_options(argc, argv);
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
