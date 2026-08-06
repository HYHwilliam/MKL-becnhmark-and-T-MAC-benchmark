#include "tmac_official_avx2.hpp"
#include <iomanip>

using namespace tmac;

struct VerificationShape
{
    int M;
    int K;
    int N;
};

template <int Bits>
double error_bound_for_element(int row, int m, int k, const std::vector<fp16_t>& activations, const std::vector<fp16_t>& scales, const TMacWorkspace& workspace)
{
    const int act_groups = k / ACT_GROUP_SIZE;
    const int weight_groups = k / WEIGHT_GROUP_SIZE;
    const int k_groups = k / G;
    const double alpha_sum = 0.5 * ((1 << Bits) - 1);
    double bound = 0.0;

    for (int group = 0; group < act_groups; ++group)
    {
        const double lut_scale = fp16_to_fp32(workspace.lut_scales[static_cast<size_t>(row) * act_groups + group]);
        const double weight_scale = std::fabs(fp16_to_fp32(scales[static_cast<size_t>(m) * weight_groups + (group * ACT_GROUP_SIZE) / WEIGHT_GROUP_SIZE]));
        double lookup_residual_sum = 0.0;

        for (int local_kg = 0; local_kg < ACT_GROUP_SIZE / G; ++local_kg)
        {
            const int kg = group * (ACT_GROUP_SIZE / G) + local_kg;
            double max_residual = 0.0;

            for (int index = 0; index < (1 << G); ++index)
            {
                double exact = 0.0;
                for (int g = 0; g < G; ++g)
                {
                    const double activation = fp16_to_fp32(activations[static_cast<size_t>(row) * k + kg * G + g]);
                    exact += ((index >> g) & 1) ? activation : -activation;
                }

                const int8_t quantized = workspace.qlut[(static_cast<size_t>(row) * k_groups + kg) * (1 << G) + index];
                const double reconstructed = static_cast<double>(quantized) * lut_scale;
                max_residual = std::max(max_residual, std::fabs(reconstructed - exact));
            }

            lookup_residual_sum += max_residual;
        }

        double exact_bias = 0.0;
        for (int kk = group * ACT_GROUP_SIZE; kk < (group + 1) * ACT_GROUP_SIZE; ++kk) exact_bias -= fp16_to_fp32(activations[static_cast<size_t>(row) * k + kk]);
        const double stored_bias = fp16_to_fp32(workspace.lut_biases[static_cast<size_t>(row) * act_groups + group]);
        const double bias_rounding = 0.5 * std::fabs(stored_bias - exact_bias) * weight_scale;
        bound += lookup_residual_sum * alpha_sum * weight_scale + bias_rounding;
    }

    return bound;
}

void apply_activation_mode(std::vector<fp16_t>& activations, int mode)
{
    if (mode == 1) std::fill(activations.begin(), activations.end(), fp32_to_fp16(0.0f));
    if (mode == 2) std::fill(activations.begin(), activations.end(), fp32_to_fp16(0.5f));
    if (mode == 3) std::fill(activations.begin(), activations.end(), fp32_to_fp16(-0.5f));
    if (mode == 4) for (size_t i = 0; i < activations.size(); ++i) activations[i] = fp32_to_fp16((i & 1) ? 1.0f : -1.0f);
}

template <int Bits>
void apply_weight_mode(std::vector<uint8_t>& qweights, int mode)
{
    const uint8_t max_weight = static_cast<uint8_t>((1 << Bits) - 1);
    if (mode == 1) std::fill(qweights.begin(), qweights.end(), 0);
    if (mode == 2) std::fill(qweights.begin(), qweights.end(), max_weight);
    if (mode == 3) for (size_t i = 0; i < qweights.size(); ++i) qweights[i] = (i & 1) ? max_weight : 0;
}

void apply_scale_mode(std::vector<fp16_t>& scales, int mode)
{
    if (mode == 1) std::fill(scales.begin(), scales.end(), fp32_to_fp16(0.001f));
    if (mode == 2) std::fill(scales.begin(), scales.end(), fp32_to_fp16(1.0f));
}

template <int Bits>
bool run_case(const VerificationShape& shape, int seed, int activation_mode, int weight_mode, int scale_mode)
{
    const ScheduleConfig schedule = choose_schedule<Bits>(shape.M, shape.N, shape.K, 8);
    std::vector<uint8_t> qweights;
    std::vector<fp16_t> activations;
    std::vector<fp16_t> scales;
    generate_inputs<Bits>(shape.M, shape.K, shape.N, qweights, activations, scales, static_cast<uint32_t>(seed));
    apply_activation_mode(activations, activation_mode);
    apply_weight_mode<Bits>(qweights, weight_mode);
    apply_scale_mode(scales, scale_mode);

    const PackedWeights packed_weights = pack_weights_official<Bits>(qweights, shape.M, shape.K, schedule);
    const PackedScales packed_scales = pack_scales_official<Bits>(scales, shape.M, shape.K, schedule);
    const bool packing_ok = verify_packed_weights<Bits>(qweights, packed_weights);
    std::vector<fp16_t> reference;
    naive_gemm_fp16<Bits>(shape.M, shape.K, shape.N, qweights, activations, scales, reference);
    std::vector<fp16_t> output1(reference.size());
    std::vector<fp16_t> output2(reference.size());
    std::vector<fp16_t> output4(reference.size());
    std::vector<fp16_t> output8(reference.size());
    std::vector<fp16_t> output_reference(reference.size());
    TMacWorkspace workspace1;
    TMacWorkspace workspace2;
    TMacWorkspace workspace4;
    TMacWorkspace workspace8;
    TMacWorkspace workspace_reference;
    tmac_gemm_fp16<Bits>(packed_weights, packed_scales, shape.N, activations.data(), output1.data(), 1, workspace1);
    tmac_gemm_fp16<Bits>(packed_weights, packed_scales, shape.N, activations.data(), output2.data(), 2, workspace2);
    tmac_gemm_fp16<Bits>(packed_weights, packed_scales, shape.N, activations.data(), output4.data(), 4, workspace4);
    tmac_gemm_fp16<Bits>(packed_weights, packed_scales, shape.N, activations.data(), output8.data(), 8, workspace8);
    tmac_gemm_fp16<Bits>(packed_weights, packed_scales, shape.N, activations.data(), output_reference.data(), 1, workspace_reference, true);

    size_t mismatches = 0;
    size_t thread_mismatches = 0;
    size_t kernel_mismatches = 0;
    double max_error = 0.0;
    double mean_error = 0.0;
    double max_ratio = 0.0;

    for (size_t index = 0; index < reference.size(); ++index)
    {
        if (output1[index] != output2[index] || output1[index] != output4[index] || output1[index] != output8[index]) ++thread_mismatches;
        if (output1[index] != output_reference[index]) ++kernel_mismatches;
        const int row = static_cast<int>(index / shape.M);
        const int m = static_cast<int>(index % shape.M);
        const double actual = fp16_to_fp32(output1[index]);
        const double expected = fp16_to_fp32(reference[index]);
        const double error = std::fabs(actual - expected);
        const double bound = error_bound_for_element<Bits>(row, m, shape.K, activations, scales, workspace1);
        const double final_fp16_margin = 0.002 * (1.0 + std::fabs(expected));
        const double tolerance = bound + final_fp16_margin;
        if (error > tolerance) ++mismatches;
        max_error = std::max(max_error, error);
        mean_error += error;
        max_ratio = std::max(max_ratio, tolerance > 0.0 ? error / tolerance : error);
    }

    mean_error /= reference.size();
    const bool passed = packing_ok && mismatches == 0 && thread_mismatches == 0 && kernel_mismatches == 0;
    std::cout << "VERIFY bit=W" << Bits << " shape=" << shape.M << "x" << shape.K << "x" << shape.N << " seed=" << seed << " activation_mode=" << activation_mode << " weight_mode=" << weight_mode << " scale_mode=" << scale_mode << " bm=" << schedule.bm << " bn=" << schedule.bn << " kfactor=" << schedule.kfactor << " packing=" << (packing_ok ? "PASS" : "FAIL") << " mismatches=" << mismatches << "/" << reference.size() << " thread_mismatches=" << thread_mismatches << " kernel_mismatches=" << kernel_mismatches << " max_abs_err=" << std::fixed << std::setprecision(7) << max_error << " mean_abs_err=" << mean_error << " max_bound_ratio=" << max_ratio << " result=" << (passed ? "PASS" : "FAIL") << "\n";
    return passed;
}

template <int Bits>
bool verify_schedule_candidates()
{
    const VerificationShape shape{256, 256, 16};
    std::vector<uint8_t> qweights;
    std::vector<fp16_t> activations;
    std::vector<fp16_t> scales;
    generate_inputs<Bits>(shape.M, shape.K, shape.N, qweights, activations, scales, 777);
    std::vector<fp16_t> reference;
    naive_gemm_fp16<Bits>(shape.M, shape.K, shape.N, qweights, activations, scales, reference);
    bool passed = true;

    for (const ScheduleConfig& schedule : official_schedule_candidates<Bits>(shape.M, shape.N, shape.K))
    {
        const PackedWeights packed_weights = pack_weights_official<Bits>(qweights, shape.M, shape.K, schedule);
        const PackedScales packed_scales = pack_scales_official<Bits>(scales, shape.M, shape.K, schedule);
        std::vector<fp16_t> output(reference.size());
        TMacWorkspace workspace;
        tmac_gemm_fp16<Bits>(packed_weights, packed_scales, shape.N, activations.data(), output.data(), 4, workspace);
        size_t mismatches = 0;

        for (size_t i = 0; i < output.size(); ++i)
        {
            const double error = std::fabs(fp16_to_fp32(output[i]) - fp16_to_fp32(reference[i]));
            const int row = static_cast<int>(i / shape.M);
            const int m = static_cast<int>(i % shape.M);
            const double tolerance = error_bound_for_element<Bits>(row, m, shape.K, activations, scales, workspace) + 0.002 * (1.0 + std::fabs(fp16_to_fp32(reference[i])));
            if (error > tolerance) ++mismatches;
        }

        const bool case_passed = verify_packed_weights<Bits>(qweights, packed_weights) && mismatches == 0;
        passed &= case_passed;
        std::cout << "SCHEDULE bit=W" << Bits << " bm=" << schedule.bm << " bn=" << schedule.bn << " kfactor=" << schedule.kfactor << " mismatches=" << mismatches << " result=" << (case_passed ? "PASS" : "FAIL") << "\n";
    }

    return passed;
}

template <int Bits>
bool run_suite()
{
    bool passed = true;
    const std::vector<VerificationShape> random_shapes = {{256, 256, 1}, {256, 256, 3}, {512, 512, 8}, {1024, 1024, 16}, {1024, 2048, 3}, {2048, 1024, 8}};
    int seed = 42;
    for (const VerificationShape& shape : random_shapes) passed &= run_case<Bits>(shape, seed++, 0, 0, 0);
    passed &= run_case<Bits>({256, 256, 3}, 100, 1, 0, 0);
    passed &= run_case<Bits>({256, 256, 3}, 101, 2, 1, 0);
    passed &= run_case<Bits>({256, 256, 3}, 102, 3, 2, 0);
    passed &= run_case<Bits>({256, 256, 3}, 103, 4, 3, 0);
    passed &= run_case<Bits>({256, 256, 3}, 104, 0, 0, 1);
    passed &= run_case<Bits>({256, 256, 3}, 105, 0, 0, 2);
    passed &= verify_schedule_candidates<Bits>();
    return passed;
}

int main()
{
    std::cout << "==================================================================\n";
    std::cout << "T-MAC W2/W3/W4A16 Independent Verification\n";
    std::cout << "FP16 activation, official tiling candidates, permutation and thread determinism\n";
    std::cout << "==================================================================\n";
    const bool w2 = run_suite<2>();
    const bool w3 = run_suite<3>();
    const bool w4 = run_suite<4>();
    std::cout << "FINAL W2=" << (w2 ? "PASS" : "FAIL") << " W3=" << (w3 ? "PASS" : "FAIL") << " W4=" << (w4 ? "PASS" : "FAIL") << "\n";
    return (w2 && w3 && w4) ? 0 : 1;
}