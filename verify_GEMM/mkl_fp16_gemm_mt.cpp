#include "tmac_official_avx2.hpp"
#include <mkl.h>
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
void run_shape(const Shape& shape, const std::vector<int>& thread_counts)
{
    std::vector<uint8_t> qweights;
    std::vector<fp16_t> activations;
    std::vector<fp16_t> scales;
    generate_inputs<Bits>(shape.M, shape.K, shape.N, qweights, activations, scales, static_cast<uint32_t>(42 + shape.M + shape.K + shape.N));
    const int zero_point = 1 << (Bits - 1);
    const int weight_groups = shape.K / WEIGHT_GROUP_SIZE;
    std::vector<MKL_F16> weights_fp16(static_cast<size_t>(shape.M) * shape.K);
    std::vector<MKL_F16> activations_fp16(activations.size());
    std::vector<MKL_F16> output(static_cast<size_t>(shape.N) * shape.M);
    static_assert(sizeof(MKL_F16) == sizeof(fp16_t), "MKL_F16 must be 16-bit");
    std::memcpy(activations_fp16.data(), activations.data(), activations.size() * sizeof(fp16_t));

    for (int m = 0; m < shape.M; ++m)
    {
        for (int k = 0; k < shape.K; ++k)
        {
            const int q = static_cast<int>(qweights[static_cast<size_t>(m) * shape.K + k]) - zero_point;
            const float scale = fp16_to_fp32(scales[static_cast<size_t>(m) * weight_groups + k / WEIGHT_GROUP_SIZE]);
            const fp16_t half = fp32_to_fp16(static_cast<float>(q) * scale);
            std::memcpy(&weights_fp16[static_cast<size_t>(m) * shape.K + k], &half, sizeof(half));
        }
    }

    MKL_F16 alpha;
    MKL_F16 beta;
    const fp16_t alpha_bits = fp32_to_fp16(1.0f);
    const fp16_t beta_bits = fp32_to_fp16(0.0f);
    std::memcpy(&alpha, &alpha_bits, sizeof(alpha));
    std::memcpy(&beta, &beta_bits, sizeof(beta));
    const double logical_flops = 2.0 * static_cast<double>(shape.M) * shape.K * shape.N;

    for (int threads : thread_counts)
    {
        mkl_set_dynamic(0);
        mkl_set_num_threads_local(threads);
        for (int warmup = 0; warmup < 3; ++warmup) cblas_hgemm(CblasRowMajor, CblasNoTrans, CblasTrans, shape.N, shape.M, shape.K, alpha, activations_fp16.data(), shape.K, weights_fp16.data(), shape.K, beta, output.data(), shape.M);

        const int samples = sample_count(logical_flops);
        std::vector<double> timings;
        timings.reserve(samples);

        for (int sample = 0; sample < samples; ++sample)
        {
            const auto start = std::chrono::high_resolution_clock::now();
            cblas_hgemm(CblasRowMajor, CblasNoTrans, CblasTrans, shape.N, shape.M, shape.K, alpha, activations_fp16.data(), shape.K, weights_fp16.data(), shape.K, beta, output.data(), shape.M);
            const auto end = std::chrono::high_resolution_clock::now();
            timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        const double total_ms = median(timings);
        const double p90_ms = percentile(timings, 0.90);
        const double gflops = logical_flops / (total_ms / 1000.0) / 1e9;
        double checksum = 0.0;

        for (size_t i = 0; i < output.size(); ++i)
        {
            fp16_t half;
            std::memcpy(&half, &output[i], sizeof(half));
            checksum += fp16_to_fp32(half);
        }

        std::cout << "RESULT category=\"" << shape.category << "\" bit=W" << Bits << " shape=" << shape.M << "x" << shape.K << "x" << shape.N << " threads=" << threads << " active_threads=" << threads << " total_ms=" << std::fixed << std::setprecision(6) << total_ms << " p90_ms=" << p90_ms << " gflops=" << gflops << " samples=" << samples << " checksum=" << checksum << "\n";
    }
}

template <int Bits>
void run_suite(const std::vector<int>& thread_counts, bool quick)
{
    const std::vector<Shape> quick_shapes = {{1024, 1024, 1, "Square"}, {1024, 1024, 8, "Square"}, {1024, 2048, 32, "Medium Asymmetric"}, {2048, 4096, 128, "Medium Asymmetric"}};
    const std::vector<int> batch_sizes = {1, 8, 32, 128, 512};
    const std::vector<std::pair<std::string, std::vector<std::pair<int, int>>>> groups = {
        {"Square", {{1024, 1024}, {2048, 2048}, {4096, 4096}, {8192, 8192}}},
        {"K Expansion", {{1024, 4096}, {2048, 8192}, {4096, 11008}, {4096, 14336}}},
        {"M Expansion", {{4096, 1024}, {8192, 2048}, {11008, 4096}, {14336, 4096}}},
        {"Medium Asymmetric", {{1024, 2048}, {2048, 1024}, {2048, 4096}, {4096, 2048}}}
    };

    if (quick)
    {
        for (const Shape& shape : quick_shapes) run_shape<Bits>(shape, thread_counts);
        return;
    }

    for (const auto& group : groups) for (int n : batch_sizes) for (const auto& mk : group.second) run_shape<Bits>({mk.first, mk.second, n, group.first.c_str()}, thread_counts);
}

int main(int argc, char** argv)
{
    int bits = 0;
    bool quick = false;
    std::vector<int> threads = {1, 2, 4, 8};

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--quick") quick = true;
        else if (arg == "--bits" && i + 1 < argc) bits = std::stoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) threads = parse_threads(argv[++i]);
        else
        {
            std::cerr << "Usage: " << argv[0] << " [--bits 2|3|4] [--threads 1,2,4,8] [--quick]\n";
            return 1;
        }
    }

    if (bits == 0 || bits == 2) run_suite<2>(threads, quick);
    if (bits == 0 || bits == 3) run_suite<3>(threads, quick);
    if (bits == 0 || bits == 4) run_suite<4>(threads, quick);
    if (bits != 0 && bits != 2 && bits != 3 && bits != 4) return 1;
    return 0;
}