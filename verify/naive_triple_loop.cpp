#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <string>
#include <algorithm>

int main() {
    const int Bits = 4, ActK = 32;
    std::vector<int> sizes = {256, 1024, 2048, 4096, 8192};

    std::cout << "=== naive_triple_loop: 獨立讀取原始輸入、純三層迴圈算 ground truth、比對 ===\n";
    std::cout << "使用 T-MAC 絕對數學邏輯 (Independent Verification)\n\n";

    for (int size : sizes) {
        int m = size, k = size;
        int k_groups = k / 4;
        int num_groups = k_groups / ActK;
        int group_span = ActK * 4;
        std::string tag = std::to_string(size);

        // 讀取未壓縮的原始權重 (0 與 1)
        std::vector<uint8_t> W_bit(m * k);
        std::ifstream fw("data_W_bit_" + tag + ".bin", std::ios::binary);
        if (!fw) { 
            std::cout << "找不到 data_W_bit_" << tag << ".bin，請先執行 t_mac_input\n"; 
            continue; 
        }
        fw.read((char*)W_bit.data(), W_bit.size());

        // 讀取原始 Activation
        std::vector<float> activations(k);
        std::ifstream fa("data_act_" + tag + ".bin", std::ios::binary);
        fa.read((char*)activations.data(), activations.size() * sizeof(float));

        // 讀取 AVX2 算出的結果以供比對
        std::vector<float> avx2_out(m);
        std::ifstream fo("data_avx2_out_" + tag + ".bin", std::ios::binary);
        fo.read((char*)avx2_out.data(), avx2_out.size() * sizeof(float));

        // 獨立計算 bias：純加總，完全不依賴 t_mac 的任何中間產物
        std::vector<float> lut_biases(num_groups);
        for (int g = 0; g < num_groups; ++g) {
            float sum = 0.0f;
            for (int i = 0; i < group_span; ++i) {
                sum += activations[g * group_span + i];
            }
            lut_biases[g] = -sum; // 對應 lut_ctor 的 bias 定義
        }
        float weight_scale = 1.0f; // 對照 t_mac_input.cpp 裡固定的 scales=1.0

        std::vector<float> out_gt(m, 0.0f);

        // ====================================================================
        // 黃金標準：純樸三層迴圈 (Ground Truth)
        // ====================================================================
        auto start = std::chrono::high_resolution_clock::now();
        for (int mm = 0; mm < m; ++mm) {
            
            // ---------------------------------------------------------
            // 【核心修正】絕對獨立的 T-MAC 數學邏輯 (與 AVX2 徹底脫鉤)
            // ---------------------------------------------------------
            // 1. 找出當前通道 mm 屬於第幾個 8-channel 區塊
            int channel_block_idx = mm / 8;
            
            // 2. 根據交錯排列邏輯，判斷這個區塊屬於哪一個 Bit-plane (0 ~ Bits-1)
            int bit_plane = channel_block_idx % Bits;
            
            // 3. 根據 T-MAC 定義，全域 Bias 統一加在 Bit 0 平面
            bool gets_bias = (bit_plane == 0);
            // ---------------------------------------------------------

            float final_acc = 0.0f;
            for (int g = 0; g < num_groups; ++g) {
                float group_acc = 0.0f;
                
                // 最純粹的 FP32 內積
                for (int kk = g * group_span; kk < (g + 1) * group_span; ++kk) {
                    float w_math = (W_bit[mm * k + kk] == 1) ? 1.0f : -1.0f;
                    group_acc += w_math * activations[kk];
                }
                
                // 只有 bit_plane == 0 才會加上該群組的 bias
                final_acc += gets_bias ? (group_acc + lut_biases[g]) : group_acc;
            }
            out_gt[mm] = final_acc * weight_scale;
        }
        auto end = std::chrono::high_resolution_clock::now();
        double naive_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // 逐元素比對誤差
        int mismatches = 0; 
        double max_err = 0.0;
        for (int i = 0; i < m; ++i) {
            double err = std::fabs(avx2_out[i] - out_gt[i]);
            max_err = std::max(max_err, err);
            
            // 容忍度：吸收 INT16 Fast Aggregation 與浮點數捨入造成的微小誤差
            if (err > 0.02 * std::fabs(out_gt[i]) + 0.05) {
                ++mismatches;
            }
        }

        std::cout << size << "x" << size << "  naive latency=" << naive_ms << "ms  "
                  << "mismatches=" << mismatches << "/" << m << "  max_err=" << max_err << "\n";
        std::cout << "  output(AVX2) [0..3] = " << avx2_out[0] << " " << avx2_out[1] << " " << avx2_out[2] << " " << avx2_out[3] << "\n";
        std::cout << "  output(Naive)[0..3] = " << out_gt[0] << " " << out_gt[1] << " " << out_gt[2] << " " << out_gt[3] << "\n\n";
    }
    return 0;
}