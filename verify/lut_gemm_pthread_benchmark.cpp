#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <pthread.h>

// ==== LUT 建表：與單執行緒版完全相同 ====
void build_lut(int K, float* input, float* lut) {
    int num_chunks = K / 8;
    for (int kt = 0; kt < num_chunks; ++kt) {
        float* chunk_input = input + kt * 8;
        float* chunk_lut = lut + kt * 256;

        float base = 0;
        for (int i = 0; i < 8; ++i) base += (-1.0f) * chunk_input[i];
        chunk_lut[0] = base;

        for (int s = 0; s < 8; ++s) {
            float iValue = 2.0f * chunk_input[s];
            int step = 1 << s;
            for (int i = 0; i < step; ++i) {
                chunk_lut[step + i] = chunk_lut[i] + iValue;
            }
        }
    }
}

// ==== 單執行緒版：正確性對照基準（完全不變）====
void lut_gemm_cpu_avx2(int M, int K, int NUM_BITS, uint32_t* W, float* alpha,
                       float* q_bias, float* input, float* output, float* lut) {

    build_lut(K, input, lut);

    int num_chunks = K / 8;
    int k32 = K / 32;

    float bias_acc = 0.0f;
    for (int kt = 0; kt < num_chunks; ++kt) bias_acc += lut[kt * 256 + 255];
    __m256 vec_bias_acc = _mm256_set1_ps(bias_acc);

    const __m256i mask255 = _mm256_set1_epi32(255);

    for (int m = 0; m < M; m += 8) {
        __m256 reg_o = _mm256_setzero_ps();

        if (q_bias != nullptr) {
            __m256 vec_qbias = _mm256_loadu_ps(q_bias + m);
            reg_o = _mm256_fmadd_ps(vec_qbias, vec_bias_acc, reg_o);
        }

        for (int b = 0; b < NUM_BITS; ++b) {
            __m256 reg_t_o = _mm256_setzero_ps();

            for (int kt32 = 0; kt32 < k32; ++kt32) {
                __m256i reg_w = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&W[kt32 * NUM_BITS * M + b * M + m]));

                for (int j = 0; j < 4; ++j) {
                    __m256i idx = _mm256_and_si256(
                        _mm256_srli_epi32(reg_w, 8 * j), mask255);
                    __m256 vals = _mm256_i32gather_ps(
                        lut + (kt32 * 4 + j) * 256, idx, 4);
                    reg_t_o = _mm256_add_ps(reg_t_o, vals);
                }
            }

            __m256 vec_alpha = _mm256_loadu_ps(alpha + b * M + m);
            reg_o = _mm256_fmadd_ps(vec_alpha, reg_t_o, reg_o);
        }

        _mm256_storeu_ps(output + m, reg_o);
    }
}

// ==== 多執行緒版：worker_thread 接受 start_m/end_m，只計算指定區間 ====
// 注意：W/alpha/q_bias/lut 是全域共享的唯讀資料，所有 thread 讀同一份
// 只有 m 維度（output 的寫入範圍）依 start_m/end_m 切分；bias_acc 與 m 無關，外部算好傳入避免重算
struct ThreadArgs {
    int start_m, end_m, K, NUM_BITS, M;
    uint32_t* W;
    float* alpha;
    float* q_bias;
    float* output;
    float* lut;
    float bias_acc;
};

void* worker_thread(void* arg) {
    ThreadArgs* a = (ThreadArgs*)arg;
    int k32 = a->K / 32;
    __m256 vec_bias_acc = _mm256_set1_ps(a->bias_acc);
    const __m256i mask255 = _mm256_set1_epi32(255);

    for (int m = a->start_m; m < a->end_m; m += 8) {
        __m256 reg_o = _mm256_setzero_ps();

        if (a->q_bias != nullptr) {
            __m256 vec_qbias = _mm256_loadu_ps(a->q_bias + m);
            reg_o = _mm256_fmadd_ps(vec_qbias, vec_bias_acc, reg_o);
        }

        for (int b = 0; b < a->NUM_BITS; ++b) {
            __m256 reg_t_o = _mm256_setzero_ps();

            for (int kt32 = 0; kt32 < k32; ++kt32) {
                __m256i reg_w = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&a->W[kt32 * a->NUM_BITS * a->M + b * a->M + m]));

                for (int j = 0; j < 4; ++j) {
                    __m256i idx = _mm256_and_si256(
                        _mm256_srli_epi32(reg_w, 8 * j), mask255);
                    __m256 vals = _mm256_i32gather_ps(
                        a->lut + (kt32 * 4 + j) * 256, idx, 4);
                    reg_t_o = _mm256_add_ps(reg_t_o, vals);
                }
            }

            __m256 vec_alpha = _mm256_loadu_ps(a->alpha + b * a->M + m);
            reg_o = _mm256_fmadd_ps(vec_alpha, reg_t_o, reg_o);
        }

        _mm256_storeu_ps(a->output + m, reg_o);
    }
    return nullptr;
}

// 將 m 切成 num_threads 份，每份對齊到 8 的倍數（lut_gemm_cpu_avx2 一次處理 8 個 m）
void compute_chunks(int m, int num_threads, std::vector<int>& starts, std::vector<int>& ends) {
    int base_chunk = (m / num_threads / 8) * 8;
    if (base_chunk == 0) base_chunk = 8;
    starts.resize(num_threads);
    ends.resize(num_threads);
    int cur = 0;
    for (int t = 0; t < num_threads; ++t) {
        starts[t] = cur;
        if (t == num_threads - 1) ends[t] = m;
        else ends[t] = cur + base_chunk;
        cur = ends[t];
    }
}

int main() {
    std::cout << "LUT-GEMM AVX2 Pthread 多執行緒 Benchmark\n";
    std::cout << "------------------------------------------------------\n";

    std::vector<int> sizes = {1024, 2048, 4096, 8192};
    std::vector<int> thread_counts = {1, 2, 4, 8};
    const int NUM_BITS = 4;

    for (int size : sizes) {
        int M = size, K = size;

        uint32_t* W = new uint32_t[(K / 32) * NUM_BITS * M];
        float* alpha = new float[NUM_BITS * M];
        float* q_bias = new float[M];
        float* input = new float[K];
        float* output = new float[M];
        float* output_single = new float[M];
        float* lut = new float[(K / 8) * 256];

        std::fill_n(W, (K / 32) * NUM_BITS * M, 0x11111111u);
        std::fill_n(alpha, NUM_BITS * M, 1.0f);
        std::fill_n(q_bias, M, 0.0f);
        std::fill_n(input, K, 1.0f);
        std::fill_n(output, M, 0.0f);
        std::fill_n(output_single, M, 0.0f);

        // 單執行緒基準（同時建好 lut，供多執行緒版本共用讀取）
        lut_gemm_cpu_avx2(M, K, NUM_BITS, W, alpha, q_bias, input, output_single, lut);

        int num_chunks = K / 8;
        float bias_acc = 0.0f;
        for (int kt = 0; kt < num_chunks; ++kt) bias_acc += lut[kt * 256 + 255];

        std::cout << "\n=== Matrix Size " << M << "x" << K << " ===\n";

        for (int nt : thread_counts) {
            std::fill_n(output, M, 0.0f);

            std::vector<int> starts, ends;
            compute_chunks(M, nt, starts, ends);

            std::vector<pthread_t> threads(nt);
            std::vector<ThreadArgs> args(nt);

            int iterations = 20;
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int it = 0; it < iterations; ++it) {
                std::fill_n(output, M, 0.0f);
                for (int t = 0; t < nt; ++t) {
                    args[t] = {starts[t], ends[t], K, NUM_BITS, M, W, alpha, q_bias, output, lut, bias_acc};
                    pthread_create(&threads[t], nullptr, worker_thread, &args[t]);
                }
                for (int t = 0; t < nt; ++t) pthread_join(threads[t], nullptr);
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            double avg_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count() / iterations;
            double gflops = (2.0 * M * K) / (avg_ms / 1000.0) / 1e9;

            int mismatches = 0;
            double max_err = 0.0;
            bool has_nan_or_inf = false;
            for (int i = 0; i < M; ++i) {
                if (std::isnan(output[i]) || std::isinf(output[i])) { has_nan_or_inf = true; break; }
                double err = std::fabs(output[i] - output_single[i]);
                max_err = std::max(max_err, err);
                if (err > 1e-4 * std::fabs(output_single[i]) + 1e-4) ++mismatches;
            }

            float expected = -2.0f * K;
            bool analytical_ok = !has_nan_or_inf && std::fabs(output[0] - expected) < std::fabs(expected) * 0.01f;

            std::cout << "  threads=" << nt << "  latency=" << avg_ms << "ms  "
                       << gflops << " GFLOPS  "
                       << (mismatches == 0 && !has_nan_or_inf ? "OK" : "MISMATCH x" + std::to_string(mismatches))
                       << "  max_err=" << max_err
                       << "  analytical=" << (analytical_ok ? "OK" : "MISMATCH")
                       << (has_nan_or_inf ? "  [NaN/Inf detected]" : "") << "\n";
        }

        delete[] W; delete[] alpha; delete[] q_bias; delete[] input;
        delete[] output; delete[] output_single; delete[] lut;
    }
    return 0;
}
