#include "tmac_official_avx2.hpp"
#include <iomanip>
#include <sstream>

using namespace tmac;

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
    while (std::getline(stream, token, ',')) result.push_back(std::max(1, std::stoi(token)));
    if (result.empty()) result = {1, 2, 4, 8};
    return result;
}

double median(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 0 ? 0.5 * (values[middle - 1] + values[middle]) : values[middle];
}

double percentile(std::vector<double> values, double ratio)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(std::min<double>(values.size() - 1, std::ceil(ratio * values.size()) - 1));
    return values[index];
}

int sample_count(double logical_flops)
{
    if (logical_flops < 1.0e8) return 21;
    if (logical_flops < 1.0e9) return 11;
    if (logical_flops < 1.0e10) return 7;
    return 5;
}

template <int Bits>
struct PreparedKernel
{
    PackedWeights weights;
    PackedScales scales;
    double tuning_ms = 0.0;
    bool autotuned = false;
};

template <int Bits>
PreparedKernel<Bits> prepare_kernel(const Shape& shape, int requested_threads, const std::vector<uint8_t>& qweights, const std::vector<fp16_t>& activations, const std::vector<fp16_t>& scales, std::vector<fp16_t>& output, bool autotune)
{
    const auto tune_start = std::chrono::high_resolution_clock::now();
    if (!autotune)
    {
        const ScheduleConfig schedule = choose_schedule<Bits>(shape.M, shape.N, shape.K, requested_threads);
        return {pack_weights_official<Bits>(qweights, shape.M, shape.K, schedule), pack_scales_official<Bits>(scales, shape.M, shape.K, schedule), 0.0, false};
    }

    double best_ms = std::numeric_limits<double>::infinity();
    PreparedKernel<Bits> best;
    for (const ScheduleConfig& schedule : official_schedule_candidates<Bits>(shape.M, shape.N, shape.K))
    {
        PackedWeights candidate_weights = pack_weights_official<Bits>(qweights, shape.M, shape.K, schedule);
        PackedScales candidate_scales = pack_scales_official<Bits>(scales, shape.M, shape.K, schedule);
        TMacWorkspace workspace;
        tmac_gemm_fp16<Bits>(candidate_weights, candidate_scales, shape.N, activations.data(), output.data(), requested_threads, workspace);
        std::vector<double> samples;
        samples.reserve(3);
        for (int sample = 0; sample < 3; ++sample) samples.push_back(tmac_gemm_fp16<Bits>(candidate_weights, candidate_scales, shape.N, activations.data(), output.data(), requested_threads, workspace).total_ms);
        const double candidate_ms = median(samples);
        if (candidate_ms < best_ms)
        {
            best_ms = candidate_ms;
            best.weights = std::move(candidate_weights);
            best.scales = std::move(candidate_scales);
        }
    }
    const auto tune_end = std::chrono::high_resolution_clock::now();
    best.tuning_ms = std::chrono::duration<double, std::milli>(tune_end - tune_start).count();
    best.autotuned = true;
    return best;
}

template <int Bits>
void run_shape(const Shape& shape, const std::vector<int>& thread_counts, bool autotune)
{
    std::vector<uint8_t> qweights;
    std::vector<fp16_t> activations;
    std::vector<fp16_t> scales;
    generate_inputs<Bits>(shape.M, shape.K, shape.N, qweights, activations, scales, static_cast<uint32_t>(42 + shape.M + shape.K + shape.N));
    std::vector<fp16_t> output(static_cast<size_t>(shape.N) * shape.M);
    const double logical_flops = 2.0 * static_cast<double>(shape.M) * shape.K * shape.N;

    for (int requested_threads : thread_counts)
    {
        PreparedKernel<Bits> prepared = prepare_kernel<Bits>(shape, requested_threads, qweights, activations, scales, output, autotune);
        const PackedWeights& packed_weights = prepared.weights;
        const PackedScales& packed_scales = prepared.scales;
        const ScheduleConfig schedule = packed_weights.schedule;
        TMacWorkspace workspace;

        for (int warmup = 0; warmup < 3; ++warmup) tmac_gemm_fp16<Bits>(packed_weights, packed_scales, shape.N, activations.data(), output.data(), requested_threads, workspace);

        const int samples = sample_count(logical_flops);
        std::vector<double> totals;
        std::vector<double> preprocesses;
        std::vector<double> kernels;
        totals.reserve(samples);
        preprocesses.reserve(samples);
        kernels.reserve(samples);
        int active_threads = 1;

        for (int sample = 0; sample < samples; ++sample)
        {
            const Timing timing = tmac_gemm_fp16<Bits>(packed_weights, packed_scales, shape.N, activations.data(), output.data(), requested_threads, workspace);
            totals.push_back(timing.total_ms);
            preprocesses.push_back(timing.preprocess_ms);
            kernels.push_back(timing.kernel_ms);
            active_threads = timing.active_threads;
        }

        const double total_ms = median(totals);
        const double preprocess_ms = median(preprocesses);
        const double kernel_ms = median(kernels);
        const double p90_ms = percentile(totals, 0.90);
        const double gflops = logical_flops / (total_ms / 1000.0) / 1e9;
        const double checksum = checksum_fp16(output.data(), output.size());
        std::cout << "RESULT category=\"" << shape.category << "\" bit=W" << Bits << " shape=" << shape.M << "x" << shape.K << "x" << shape.N << " threads=" << requested_threads << " active_threads=" << active_threads << " bm=" << schedule.bm << " bn=" << schedule.bn << " kfactor=" << schedule.kfactor << " total_ms=" << std::fixed << std::setprecision(6) << total_ms << " preprocess_ms=" << preprocess_ms << " kernel_ms=" << kernel_ms << " p90_ms=" << p90_ms << " gflops=" << gflops << " samples=" << samples << " autotuned=" << (prepared.autotuned ? 1 : 0) << " tuning_ms=" << prepared.tuning_ms << " checksum=" << checksum << "\n";
    }
}

template <int Bits>
void run_suite(const std::vector<int>& thread_counts, bool quick, bool autotune)
{
    const std::vector<Shape> quick_shapes = {{1024, 1024, 1, "Square"}, {1024, 1024, 8, "Square"}, {1024, 2048, 32, "Medium Asymmetric"}, {2048, 4096, 128, "Medium Asymmetric"}};
    const std::vector<int> batch_sizes = {1, 8, 32, 128, 512};
    const std::vector<std::pair<std::string, std::vector<std::pair<int, int>>>> groups = {
        {"Square", {{1024, 1024}, {2048, 2048}, {4096, 4096}, {8192, 8192}}},
        {"K Expansion", {{1024, 4096}, {2048, 8192}, {4096, 11008}, {4096, 14336}}},
        {"M Expansion", {{4096, 1024}, {8192, 2048}, {11008, 4096}, {14336, 4096}}},
        {"Medium Asymmetric", {{1024, 2048}, {2048, 1024}, {2048, 4096}, {4096, 2048}}}
    };
    std::cout << "==================================================================\n";
    std::cout << "T-MAC W" << Bits << "A16 FP16 Tiled GEMM Benchmark\n";
    std::cout << "Official schedule candidates, AVX2/F16C/OpenMP, median timing\n";
    std::cout << "==================================================================\n";

    if (quick)
    {
        for (const Shape& shape : quick_shapes) run_shape<Bits>(shape, thread_counts, autotune);
        return;
    }

    for (const auto& group : groups) for (int n : batch_sizes) for (const auto& mk : group.second) run_shape<Bits>({mk.first, mk.second, n, group.first.c_str()}, thread_counts, autotune);
}

int main(int argc, char** argv)
{
    int bits = 0;
    bool quick = false;
    bool autotune = false;
    std::vector<int> threads = {1, 2, 4, 8};

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--quick") quick = true;
        else if (arg == "--autotune") autotune = true;
        else if (arg == "--bits" && i + 1 < argc) bits = std::stoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) threads = parse_threads(argv[++i]);
        else
        {
            std::cerr << "Usage: " << argv[0] << " [--bits 2|3|4] [--threads 1,2,4,8] [--quick] [--autotune]\n";
            return 1;
        }
    }

    try
    {
        if (bits == 0 || bits == 2) run_suite<2>(threads, quick, autotune);
        if (bits == 0 || bits == 3) run_suite<3>(threads, quick, autotune);
        if (bits == 0 || bits == 4) run_suite<4>(threads, quick, autotune);
        if (bits != 0 && bits != 2 && bits != 3 && bits != 4) throw std::invalid_argument("Bits must be 2, 3 or 4");
    }
    catch (const std::exception& error)
    {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }

    return 0;
}