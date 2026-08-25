# T-MAC ARM WnA16 Benchmark

本資料夾用於 ARM CPU 上的 T-MAC W2/W3/W4-A16 GEMV / GEMM benchmark，並預留與群聯板 NPU GEMM 比較的流程。

## 實作

```text
src/math_scalar_wxa16.cpp
src/tmac_scalar_wxa16.cpp
src/tmac_neon_st_wxa16.cpp
src/tmac_neon_mt_fixed_wxa16.cpp
src/tmac_neon_mt_tuned_wxa16.cpp
```

- `math_scalar_wxa16`：純數學 scalar baseline
- `tmac_scalar_wxa16`：T-MAC scalar
- `tmac_neon_st_wxa16`：T-MAC NEON single-thread
- `tmac_neon_mt_fixed_wxa16`：T-MAC NEON multi-thread fixed
- `tmac_neon_mt_tuned_wxa16`：T-MAC NEON multi-thread tuned

支援：

```text
W2A16
W3A16
W4A16
```

矩陣定義：

```text
W[M,K]
X[N,K]
Y[N,M] = X * W^T

N = 1  -> GEMV
N > 1  -> GEMM
```

---

## 1. WSL：Cross Compile 與 QEMU 驗證

此步驟用於在 x86 / WSL 上先確認 ARM binary correctness，不作為正式效能測試。

### 安裝工具

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    make \
    g++ \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    qemu-user \
    python3
```

### Cross Compile

```bash
cd ~/benchmark_project/tmac_ARM/CPU

make cross
make -f Makefile.tuned tuned-cross
```

輸出：

```text
build/arm64/math_scalar_wxa16
build/arm64/tmac_scalar_wxa16
build/arm64/tmac_neon_st_wxa16
build/arm64/tmac_neon_mt_fixed_wxa16
build/arm64/tmac_neon_mt_tuned_wxa16
```

### Correctness Verification

第一次建立環境、修改 kernel 或 compiler flags 後建議執行：

```bash
make qemu-verify BITS=2 THREADS=4
make qemu-verify BITS=3 THREADS=4
make qemu-verify BITS=4 THREADS=4

python3 scripts/verify_tuned_freeze.py
```

QEMU 只驗證 correctness，不使用 QEMU latency 作為 benchmark 結果。

---

# 2. 群聯 Ubuntu 板：CPU T-MAC

## 環境確認

```bash
uname -m
lscpu
```

`uname -m` 應為：

```text
aarch64
```

目前 NEON implementation 需要 ARMv8.2-A FP16 vector arithmetic，編譯旗標為：

```text
-march=armv8.2-a+fp16
```

## 安裝編譯工具

```bash
sudo apt update
sudo apt install -y build-essential make g++ git python3
```

## 取得專案

若板子尚未有 repository：

```bash
cd ~

git clone https://github.com/HYHwilliam/MKL-becnhmark-and-T-MAC-benchmark.git

cd MKL-becnhmark-and-T-MAC-benchmark/tmac_ARM/CPU
```

若已放在 `~/benchmark_project`：

```bash
cd ~/benchmark_project/tmac_ARM/CPU
```

## Native Compile

```bash
make clean
make
make -f Makefile.tuned tuned
```

輸出：

```text
build/native/math_scalar_wxa16
build/native/tmac_scalar_wxa16
build/native/tmac_neon_st_wxa16
build/native/tmac_neon_mt_fixed_wxa16
build/native/tmac_neon_mt_tuned_wxa16
```

---

## 3. CPU Correctness Verification

`--verify-only` 不是每次 benchmark 前都必須執行。

建議在以下情況執行：

```text
第一次在新板子上執行
修改 T-MAC kernel
修改 packing / LUT / scale 邏輯
修改 compiler flags
更換 CPU / compiler
```

例如 W2：

```bash
./build/native/math_scalar_wxa16 \
    --bits 2 \
    --verify-only

./build/native/tmac_scalar_wxa16 \
    --bits 2 \
    --verify-only

./build/native/tmac_neon_st_wxa16 \
    --bits 2 \
    --verify-only

./build/native/tmac_neon_mt_fixed_wxa16 \
    --bits 2 \
    --threads 4 \
    --verify-only
```

W3 / W4 將：

```text
--bits 2
```

改為：

```text
--bits 3
--bits 4
```

已確認 correctness 且程式未修改時，可直接執行 benchmark。

---

# 4. CPU Benchmark

W2：

```bash
./build/native/math_scalar_wxa16 \
    --bits 2 \
    --max-size 8192

./build/native/tmac_scalar_wxa16 \
    --bits 2 \
    --max-size 8192

./build/native/tmac_neon_st_wxa16 \
    --bits 2 \
    --max-size 8192

./build/native/tmac_neon_mt_fixed_wxa16 \
    --bits 2 \
    --threads 4 \
    --max-size 8192

./build/native/tmac_neon_mt_tuned_wxa16 \
    --bits 2 \
    --threads 4 \
    --max-size 8192
```

W3 / W4：

```text
--bits 3
--bits 4
```

Tuned schedule 必須在實際群聯 ARM CPU 上重新搜尋，不使用 QEMU 或其他 CPU 的 tuning 結果。

---

# 5. 群聯 NPU GEMM

Ubuntu 只提供 CPU 編譯環境。NPU GEMM 仍需要群聯板實際提供的：

```text
NPU SDK
NPU runtime
NPU compiler / toolchain
driver
```

因此 NPU 的正式編譯指令必須依板子上的 SDK 決定。

## 確認 NPU SDK

先確認板子是否已安裝 NPU SDK：

```bash
ls /opt
env | grep -Ei 'npu|sdk|runtime|phison'
```

若不知道位置，可搜尋：

```bash
find /opt /usr/local \
    \( -iname '*npu*' -o -iname '*sdk*' -o -iname '*runtime*' \) \
    2>/dev/null | head -100
```

找到 SDK 後，優先閱讀 SDK 的 README / sample：

```bash
find <NPU_SDK_PATH> \
    \( -iname 'README*' -o -iname '*sample*' -o -iname '*gemm*' -o -iname '*matmul*' \) \
    2>/dev/null | head -100
```

---

## 6. NPU Build

NPU SDK 常見有三種方式，使用板子實際提供的方式即可。

### A. SDK 提供 NPU Compiler

```bash
source <NPU_SDK_PATH>/env.sh

mkdir -p build/npu

<NPU_COMPILER> \
    <NPU_FLAGS> \
    <NPU_GEMM_SOURCE> \
    -o build/npu/gemm_npu
```

### B. SDK 使用 CMake / Toolchain

```bash
source <NPU_SDK_PATH>/env.sh

cmake -S <NPU_GEMM_PROJECT> \
      -B build/npu \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=<NPU_TOOLCHAIN_FILE>

cmake --build build/npu -j$(nproc)
```

### C. SDK 使用 ONNX / Model Compiler

若 NPU 不直接編譯 C++ GEMM，而是使用 MatMul / GEMM model：

```text
GEMM / MatMul
    ↓
ONNX / Vendor IR
    ↓
NPU Compiler
    ↓
Compiled Model
    ↓
NPU Runtime
```

典型形式：

```bash
<NPU_MODEL_COMPILER> \
    --model gemm.onnx \
    --output build/npu/gemm_model
```

執行：

```bash
<NPU_RUNTIME_RUNNER> \
    --model build/npu/gemm_model
```

`<...>` 必須依群聯板實際 NPU SDK 文件替換，不應使用未確認的 compiler 或參數。

---

# 7. NPU Correctness 與 Benchmark

NPU GEMM 建議先使用小矩陣確認 correctness：

```text
M = 256
K = 256
N = 1
```

再測：

```text
M = 256
K = 256
N = 8
```

correctness 確認後即可進行正式 benchmark，不需要每次重複驗證。

---

# 8. CPU / NPU 比較

CPU 與 NPU benchmark 應使用相同：

```text
M
K
N
warmup
repeat
資料格式
計時範圍
```

至少分開記錄：

```text
CPU:
preprocess_ms
kernel_ms
total_ms

NPU:
upload_ms
kernel_ms
download_ms
total_ms
```

比較時應使用：

```text
kernel-only vs kernel-only
end-to-end vs end-to-end
```

避免將 CPU end-to-end latency 與 NPU kernel-only latency 直接比較。

如果 CPU 使用 W2A16、W3A16、W4A16，而 NPU 使用 INT8 / FP16，也必須在結果中明確標示資料格式差異。

---

# 9. 建議實際操作流程

第一次在群聯板上：

```text
1. 確認 Ubuntu / AArch64 / CPU ISA
2. 安裝 g++ / make
3. Native compile CPU T-MAC
4. 執行一次 correctness verification
5. 執行 CPU benchmark
6. 確認 NPU SDK / runtime
7. 先成功編譯並執行官方 NPU sample
8. 編譯自己的 NPU GEMM
9. 執行一次 NPU correctness verification
10. 執行 NPU benchmark
11. 比較 CPU / NPU
```

之後程式與環境沒有修改時：

```text
CPU benchmark
NPU benchmark
比較結果
```

不需要每次重新執行 `--verify-only`。

---

## Build Output

CPU：

```text
tmac_ARM/CPU/build/native/
```

NPU 建議：

```text
tmac_ARM/NPU/build/
```

`build/`、binary、log 與 generated benchmark data 不應提交至 GitHub。
