# LUT-GEMM CPU GEMV Experiments

這個資料夾保留 LUT-GEMM 在 CPU 上的兩個 GEMV microbenchmark，主要用於比較 scalar lookup 與 AVX2 gather lookup。

這不是目前正式的 T-MAC benchmark，也不是完整重現 LUT-GEMM GPU kernel。

## Files

```text
lut_gemm_scalar_gemv.cpp
lut_gemm_avx2_gemv.cpp
```

`lut_gemm_scalar_gemv.cpp` 使用 scalar C++ 完成 LUT 建表與查表累加。

`lut_gemm_avx2_gemv.cpp` 使用 AVX2，一次處理多個 output，主要 lookup 指令為 `_mm256_i32gather_ps`。

## Test Setup

```text
M = K = 256, 1024, 2048, 4096, 8192
N = 1
NUM_BITS = 4
group size = 8
LUT entries = 2^8 = 256
```

測試資料為 deterministic synthetic input，目的是觀察 CPU LUT lookup / accumulation 行為。

## Compile

```bash
cd ~/benchmark_project/experiments/LUT_GEMM_CPU
mkdir -p build

g++ -O3 -std=c++17     lut_gemm_scalar_gemv.cpp     -o build/lut_gemm_scalar_gemv

g++ -O3 -std=c++17 -mavx2 -mfma     lut_gemm_avx2_gemv.cpp     -o build/lut_gemm_avx2_gemv
```

## Run

```bash
./build/lut_gemm_scalar_gemv
./build/lut_gemm_avx2_gemv
```

`build/` 已由 repository `.gitignore` 排除。

## Correctness

目前 synthetic input 的設定可推得解析解，程式會檢查 NaN/Inf 並比較輸出。

這裡的 correctness 只代表此 microbenchmark 在指定測資下的 lookup / accumulation 流程一致，不是完整低位元模型 correctness proof。

正式 W2/W3/W4-A16 ARM T-MAC correctness 與 benchmark 請使用 `tmac_ARM/`。
