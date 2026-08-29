# cuBLAS HackMD Benchmark

此資料夾提供 NVIDIA GPU 的 cuBLAS dense FP16 GEMM benchmark，並將結果與實驗室 Ryzen CPU 的 T-MAC W2/W3/W4A16 比較。

目前 GPU 端是 **cuBLAS reference**，不是 GPU T-MAC。

## 主要檔案

| 檔案 | 用途 |
|---|---|
| `cublas_fp16_hackmd_benchmark.cu` | cuBLAS FP16 GEMM benchmark |
| `summarize_cublas.py` | 統整 cuBLAS 五輪結果，並與 Ryzen T-MAC 比較 |
| `results/lab_rtx4070super/` | RTX 4070 SUPER 五輪測試環境與 raw logs |
| `results/cublas_tmac_compare/` | cuBLAS vs T-MAC 統計與報告 |

Shape 一律以 `M×N×K` 顯示。

---

## 1. CUDA 環境

本次正式測試使用：

```text
GPU: NVIDIA GeForce RTX 4070 SUPER
Compute Capability: 8.9
CUDA Toolkit: 12.3
```

若 `/usr/local/cuda` 已指向正確 CUDA，可在目前 shell 使用：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

確認：

```bash
which nvcc
nvcc --version
nvidia-smi
```

---

## 2. 編譯 cuBLAS benchmark

在 repository root 執行：

```bash
/usr/local/cuda/bin/nvcc \
    -O3 \
    -std=c++17 \
    -arch=sm_89 \
    gpu_GEMM/hackmd_compare/cublas_fp16_hackmd_benchmark.cu \
    -lcublas \
    -o gpu_GEMM/hackmd_compare/cublas_fp16_hackmd_benchmark
```

`sm_89` 對應 RTX 4070 SUPER 的 Compute Capability 8.9。

編譯出的：

```text
gpu_GEMM/hackmd_compare/cublas_fp16_hackmd_benchmark
```

是本機 executable，不建議提交 Git。

---

## 3. 執行 cuBLAS

單次測試：

```bash
./gpu_GEMM/hackmd_compare/cublas_fp16_hackmd_benchmark
```

每輪包含 5 個 shape，因此應輸出：

```text
5 RESULT rows
```

### 五輪正式測試

```bash
RESULT_DIR=gpu_GEMM/hackmd_compare/results/lab_rtx4070super
mkdir -p "$RESULT_DIR"

for run in 1 2 3 4 5
do
    ./gpu_GEMM/hackmd_compare/cublas_fp16_hackmd_benchmark \
        | tee "$RESULT_DIR/cublas_fp16_run${run}.log"
done
```

完成後應有：

```text
5 logs
25 RESULT rows
```

---

## 4. Benchmark 定義

cuBLAS 使用：

```text
Input  : FP16
Weight : FP16
Output : FP16
Compute accumulation : FP32
```

核心呼叫為 `cublasGemmEx`。

計時使用 CUDA Event，只量 warmed-up GPU GEMM：

```text
CPU RAM
  ↓ H2D       不計時
GPU VRAM
  ↓
cuBLAS GEMM   計時
  ↓
GPU VRAM
  ↓ D2H       不計時
CPU RAM
```

因此這是 GPU GEMM latency，不是完整 application end-to-end latency。

---

## 5. 統整 cuBLAS + T-MAC

執行：

```bash
python3 \
    gpu_GEMM/hackmd_compare/summarize_cublas.py
```

腳本讀取：

```text
GPU：
gpu_GEMM/hackmd_compare/results/lab_rtx4070super/
    cublas_fp16_run1.log
    ...
    cublas_fp16_run5.log

CPU：
x86_GEMM/hackmd_compare/results/cpu_tmac_compare/
    lab_tmac_statistics.csv
```

輸出：

```text
gpu_GEMM/hackmd_compare/results/cublas_tmac_compare/
├── validation_report.txt
├── cublas_statistics.csv
├── cublas_tmac_all_threads.csv
├── cublas_tmac_primary_comparison.csv
└── cublas_tmac_comparison.md
```

---

## 6. Primary CPU baseline

GPU 沒有 CPU thread count，因此主要比較不固定使用某個 T-MAC thread 數。

每個：

```text
bit-width + shape
```

都從：

```text
1T / 2T / 4T / 8T
```

中選 latency 最低的 T-MAC，作為 Primary CPU baseline。

16T 仍保留在詳細 CSV，但不參與 primary selection，因為 Ryzen 5 8500G 只有 12 logical CPUs，16T 已屬 oversubscription。

倍率定義：

```text
GPU speedup = T-MAC CPU latency / cuBLAS GPU latency
```

---

## 7. 正確性與統計

cuBLAS 每個 shape 做 sampled numerical verification，輸出：

```text
verify_max_abs_error
checksum
```

正式統計使用五輪 run-level median。

GFLOPs 使用：

```text
2 × M × N × K / latency
```

屬於 dense GEMM logical throughput。

---

## 8. 解讀限制

cuBLAS 與 T-MAC 的結果不是同硬體、同 precision representation、同 execution model：

```text
T-MAC:
AMD Ryzen CPU
W2 / W3 / W4 low-bit weight
AVX2/F16C/OpenMP
total_ms = preprocessing + kernel

cuBLAS:
NVIDIA RTX GPU
dense FP16
FP32 accumulation
CUDA Event GEMM timing
H2D/D2H excluded
```

因此 GPU speedup 應解讀為：

> RTX 4070 SUPER 的 dense FP16 GEMM reference 相對 Ryzen CPU T-MAC 的 latency 差距。

不能解讀為：

> cuBLAS 演算法本身比 T-MAC 演算法快相同倍率。
