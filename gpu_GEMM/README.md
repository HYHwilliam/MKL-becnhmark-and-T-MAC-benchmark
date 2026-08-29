# GPU GEMM Benchmark

此目錄放置 GPU GEMM 相關實驗。

目前主要實作為：

```text
gpu_GEMM/
└── hackmd_compare/
    ├── cublas_fp16_hackmd_benchmark.cu
    ├── summarize_cublas.py
    └── results/
```

GPU baseline 使用 NVIDIA cuBLAS dense FP16 GEMM，目的是提供 CPU T-MAC 之外的 GPU performance reference。

## 快速使用

### 編譯

RTX 4070 SUPER / Compute Capability 8.9：

```bash
/usr/local/cuda/bin/nvcc \
    -O3 \
    -std=c++17 \
    -arch=sm_89 \
    gpu_GEMM/hackmd_compare/cublas_fp16_hackmd_benchmark.cu \
    -lcublas \
    -o gpu_GEMM/hackmd_compare/cublas_fp16_hackmd_benchmark
```

### 執行

```bash
./gpu_GEMM/hackmd_compare/cublas_fp16_hackmd_benchmark
```

### 統整五輪正式結果

```bash
python3 \
    gpu_GEMM/hackmd_compare/summarize_cublas.py
```

詳細 benchmark 定義、五輪測試方式、H2D/D2H 計時範圍與 T-MAC 比較方式請見：

```text
gpu_GEMM/hackmd_compare/README.md
```

> 目前 GPU 端只有 cuBLAS baseline，並沒有實作 GPU T-MAC。
