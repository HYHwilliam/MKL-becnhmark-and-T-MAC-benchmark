# T-MAC ARM WnA16 Benchmark

本資料夾用於 ARM CPU 上的 T-MAC W2/W3/W4-A16 GEMV / GEMM benchmark，並預留與群聯板 NPU T-MAC / GEMM 比較的流程。

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

此步驟用於在 x86 / WSL 上確認 ARM binary correctness，不作為正式效能測試。

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

第一次建立環境、修改 kernel、packing / LUT 邏輯或 compiler flags 後建議執行：

```bash
make qemu-verify BITS=2 THREADS=4
make qemu-verify BITS=3 THREADS=4
make qemu-verify BITS=4 THREADS=4

python3 scripts/verify_tuned_freeze.py
```

Multi-thread hardening：

```bash
./scripts/verify_mt_hardening.sh
```

QEMU 只用於 correctness 與 smoke test，不使用 QEMU latency 或 tuning winner 作為正式 benchmark 結果。

---

## 2. 群聯 Ubuntu 板：CPU T-MAC

### 環境確認

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

### 安裝編譯工具

```bash
sudo apt update
sudo apt install -y build-essential make g++ git python3
```

### 取得專案

若板子尚未有 repository：

```bash
cd ~

git clone https://github.com/HYHwilliam/MKL-becnhmark-and-T-MAC-benchmark.git

cd MKL-becnhmark-and-T-MAC-benchmark/tmac_ARM/CPU
```

若專案已放在 `~/benchmark_project`：

```bash
cd ~/benchmark_project/tmac_ARM/CPU
```

### Native Compile

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

Tuned 版本可另外確認 frozen compute：

```bash
python3 scripts/verify_tuned_freeze.py
```

已確認 correctness 且程式、compiler 與平台未修改時，可直接執行 benchmark。

---

## 4. CPU Benchmark

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

正式效能數據只在實際 ARM target 上量測。

---

## 5. 群聯 NPU T-MAC / GEMM

NPU implementation 位於：

```text
tmac_ARM/NPU/
```

進行 NPU 開發或編譯前，先切換至 NPU 目錄。

若 repository 位於 `~/benchmark_project`：

```bash
cd ~/benchmark_project/tmac_ARM/NPU
```

若是在群聯板上直接 clone：

```bash
cd ~/MKL-becnhmark-and-T-MAC-benchmark/tmac_ARM/NPU
```

Ubuntu 本身只提供一般 CPU 開發環境。NPU T-MAC / GEMM 仍需要群聯板實際提供的：

```text
NPU SDK
NPU runtime
NPU compiler / toolchain
driver
```

因此 NPU 的正式編譯指令必須依板子實際提供的 SDK 決定。

### 確認 NPU SDK

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

在撰寫自己的 NPU T-MAC 前，建議先成功編譯並執行 SDK 官方 sample，以確認 driver、runtime 與 compiler/toolchain 都能正常工作。

---

## 6. NPU Build

NPU build output 統一放在：

```text
tmac_ARM/NPU/build/
```

以下指令都假設目前位於：

```text
tmac_ARM/NPU/
```

NPU SDK 常見有三種使用方式，實際使用哪一種需依群聯板提供的 SDK 決定。

### A. SDK 提供 NPU Compiler

```bash
cd ~/benchmark_project/tmac_ARM/NPU

source <NPU_SDK_PATH>/env.sh

mkdir -p build

<NPU_COMPILER> \
    <NPU_FLAGS> \
    <NPU_GEMM_SOURCE> \
    -o build/tmac_npu
```

其中：

```text
<NPU_SDK_PATH>
<NPU_COMPILER>
<NPU_FLAGS>
<NPU_GEMM_SOURCE>
```

必須依實際 SDK 文件替換。

### B. SDK 使用 CMake / Toolchain

```bash
cd ~/benchmark_project/tmac_ARM/NPU

source <NPU_SDK_PATH>/env.sh

cmake -S . \
      -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=<NPU_TOOLCHAIN_FILE>

cmake --build build -j$(nproc)
```

實際 CMake project 結構與 toolchain file 位置需依 SDK 決定。

### C. SDK 使用 ONNX / Model Compiler

若 NPU 不直接編譯 C/C++ GEMM，而是使用 MatMul / GEMM graph：

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
cd ~/benchmark_project/tmac_ARM/NPU

mkdir -p build

<NPU_MODEL_COMPILER> \
    --model gemm.onnx \
    --output build/gemm_model
```

執行：

```bash
<NPU_RUNTIME_RUNNER> \
    --model build/gemm_model
```

`<...>` 皆為 placeholder，取得群聯板實際 NPU SDK 後才能替換為正式 compiler、runtime 與參數，不應自行假設。

---

## 7. NPU Correctness 與 Benchmark

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

若後續實作 W2A16 / W3A16 / W4A16 NPU T-MAC，correctness test 應使用與 CPU 相同的：

```text
weight
activation
scale
M
K
N
bit-width
```

確認 correctness 後，再進行正式 benchmark。

程式、compiler、runtime 與硬體環境沒有修改時，不需要每次 benchmark 都重新執行完整 correctness verification。

---

## 8. CPU / NPU Benchmark 比較

CPU 與 NPU benchmark 應盡量使用相同：

```text
M
K
N
warmup
repeat
資料格式
輸入資料
計時範圍
```

共用 benchmark 設定與結果建議集中在：

```text
tmac_ARM/benchmark/
```

例如：

```text
tmac_ARM/benchmark/
├── configs/
├── scripts/
└── results/
    ├── CPU/
    ├── NPU/
    └── comparison/
```

CPU 至少分開記錄：

```text
preprocess_ms
kernel_ms
total_ms
```

NPU 至少分開記錄：

```text
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

避免將：

```text
CPU end-to-end latency
```

直接與：

```text
NPU kernel-only latency
```

比較。

如果 CPU 使用：

```text
W2A16
W3A16
W4A16
```

而 NPU 實際只能使用：

```text
INT8
FP16
或其他格式
```

則必須在 benchmark 結果中明確標示資料格式差異，不能視為完全相同的運算條件。

---

## 9. 建議實際操作流程

第一次在群聯板上：

```text
1. 確認 Ubuntu / AArch64 / CPU ISA
2. 安裝 g++ / make
3. 進入 tmac_ARM/CPU
4. Native compile CPU T-MAC
5. 執行一次 CPU correctness verification
6. 執行 CPU benchmark
7. 進入 tmac_ARM/NPU
8. 確認 NPU SDK / runtime / driver
9. 成功編譯並執行官方 NPU sample
10. 建立或編譯自己的 NPU T-MAC / GEMM
11. 執行一次 NPU correctness verification
12. 執行 NPU benchmark
13. 將結果保存至 tmac_ARM/benchmark/results/
14. 比較 CPU / NPU
```

之後程式與環境沒有修改時：

```text
CPU benchmark
NPU benchmark
比較結果
```

不需要每次重新執行完整 `--verify-only`。

---

## Build Output

CPU cross-compile：

```text
tmac_ARM/CPU/build/arm64/
```

CPU native：

```text
tmac_ARM/CPU/build/native/
```

NPU：

```text
tmac_ARM/NPU/build/
```

CPU / NPU benchmark 結果：

```text
tmac_ARM/benchmark/results/
```

`build/`、binary、log 與其他 generated artifacts 不應提交至 GitHub。

正式需要保存的 benchmark 結果則集中管理於：

```text
tmac_ARM/benchmark/results/
```