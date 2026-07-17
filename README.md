# T-MAC AVX2 vs. LUT-GEMM (CPU) vs. MKL FP16 GEMV Benchmark

比較 T-MAC（4-bit 量化 + Lookup Table）、LUT-GEMM（官方 CUDA kernel 的 CPU 移植版）與 Intel MKL（FP16 傳統浮點運算）在單執行緒 GEMV（矩陣乘向量，`M x K x 1`）情境下的效能表現。

## 概述

- **T-MAC**：以 4-bit 量化權重搭配預建查表（LUT，μ=4），用查表取代乘法運算，加速低位元推論。
- **LUT-GEMM**：移植自 [NAVER LUT-GEMM 官方 CUDA 核心](https://github.com/naver-aics/lut-gemm)（`mv_fp16_bias.hpp` 的 `_nqmv_bias`），採用 μ=8 的查表設計，提供純量（Scalar）與 AVX2 兩種 CPU 版本。
- **MKL**：使用 Intel MKL 的 `cblas_hgemm`，做為傳統 FP16 浮點乘加的效能基準。
- 三者皆固定 `N=1`（GEMV），對應語言模型推論階段每次生成一個 token 的實際運算情境。
- 皆附上正確性驗證（Checksum 或解析解比對），避免比較到「跑得快但算錯」的無意義數字。

## 測試環境

### 硬體規格

| 項目 | 規格 |
|---|---|
| 處理器 | Intel(R) Core(TM) Ultra 7 258V @ 2.20GHz |
| 記憶體 | 32GB RAM |
| 顯示卡 | （本測試未使用 GPU） |
| 作業系統 | Windows 11 家用版，64 位元 |
| 測試環境 | WSL2（Windows Subsystem for Linux） |

> 本測試為 CPU 單執行緒 Benchmark，未使用 GPU 加速。LUT-GEMM 官方設計目標平台為 GPU，本專案僅測試其移植至 CPU 後的表現。

### 軟體環境

| 項目 | 版本 / 說明 |
|---|---|
| WSL | WSL2 |
| Linux 發行版 | Ubuntu 24.04.4 LTS (Noble Numbat) |
| 編譯器 | GCC 13（g++） |
| 指令集 | AVX2 + FMA + F16C |
| 數學函式庫 | Intel oneAPI MKL |

## 環境建置

### 1. 啟用並進入 WSL

```powershell
wsl --install -d Ubuntu-24.04
wsl
```

### 2. 安裝編譯工具

```bash
sudo apt update
sudo apt install -y build-essential g++
```

### 3. 安裝 Intel oneAPI MKL

依照 [Intel oneMKL 官方安裝指南（Linux / apt）](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=linux&linux-install=apt) 進行安裝：

```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor \
  | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update
sudo apt install -y intel-oneapi-mkl intel-oneapi-mkl-devel
```

安裝完成後，設定環境變數（每次開啟新終端機皆需執行，或加入 `~/.bashrc`）：

```bash
source /opt/intel/oneapi/setvars.sh
```

### 4. 確認 CPU 支援指令集

```bash
cat /proc/cpuinfo | grep -o 'avx2\|fma\|f16c' | sort -u
```

需同時看到 `avx2`、`fma`、`f16c` 三項，才能編譯與執行本專案。

## 建置與執行

### T-MAC AVX2 Benchmark

```bash
g++ -O2 -mavx2 -mfma t_mac_benchmark.cpp -o t_mac_benchmark
./t_mac_benchmark
```

### LUT-GEMM CPU Benchmark（純量版）

```bash
g++ -O3 lut_gemm_cpu_benchmark.cpp -o lut_gemm_cpu_benchmark
./lut_gemm_cpu_benchmark
```

> 純量版未使用任何 SIMD intrinsics，不需要額外的 `-m` 指令集旗標。

### LUT-GEMM CPU Benchmark（AVX2 優化版）

```bash
g++ -O3 -mavx2 -mfma lut_gemm_avx2_benchmark.cpp -o lut_gemm_avx2_benchmark
./lut_gemm_avx2_benchmark
```

> 必須同時加上 `-mavx2`（`_mm256_i32gather_ps` 為 AVX2 指令）與 `-mfma`（`_mm256_fmadd_ps` 為 FMA3 指令），缺一都會導致編譯期的 `inlining failed... target specific option mismatch` 錯誤。

### MKL FP16 Benchmark

```bash
g++ -O2 -mavx2 -mfma -mf16c \
    -I${MKLROOT}/include \
    mkl_benchmark.cpp \
    -L${MKLROOT}/lib/intel64 \
    -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl \
    -o mkl_benchmark

export LD_LIBRARY_PATH=${MKLROOT}/lib/intel64:$LD_LIBRARY_PATH
./mkl_benchmark
```

### （選用）記憶體安全性檢查

正式量測前，建議先以 AddressSanitizer 確認無記憶體越界問題（LUT-GEMM 的 gather 索引運算尤其容易出現越界，務必先跑過一次）：

```bash
g++ -g -O1 -mavx2 -mfma -fsanitize=address,undefined lut_gemm_avx2_benchmark.cpp -o lut_gemm_avx2_debug
./lut_gemm_avx2_debug
```

## Benchmark 結果

測試規模：`M = K = {256, 1024, 2048, 4096, 8192}`，`N = 1`，單執行緒，T-MAC/MKL 每組取 100 次迭代平均，LUT-GEMM 純量版因速度較慢取 10 次迭代平均。

### T-MAC AVX2 GEMV

| Matrix Size | Latency (ms) | Performance (GFLOPS) | Checksum |
|---|---:|---:|---|
| 256×256×1 | 0.0026 | 50.13 | 無 NaN/Inf |
| 1024×1024×1 | 0.0316 | 66.34 | 無 NaN/Inf |
| 2048×2048×1 | 0.1319 | 63.59 | 無 NaN/Inf |
| 4096×4096×1 | 0.4557 | 73.64 | 無 NaN/Inf |
| 8192×8192×1 | 1.4860 | 90.32 | 無 NaN/Inf |

### LUT-GEMM CPU AVX2 版

| Matrix Size | Latency (ms) | Performance (GFLOPS) | 解析解驗證 |
|---|---:|---:|---|
| 256×256×1 | 0.0246 | 5.32 | output[0]=-512  OK |
| 1024×1024×1 | 0.5134 | 4.09 | output[0]=-2048  OK |
| 2048×2048×1 | 1.6110 | 5.21 | output[0]=-4096  OK |
| 4096×4096×1 | 10.4368 | 3.22 | output[0]=-8192  OK |
| 8192×8192×1 | 118.772 | 1.13 | output[0]=-16384  OK |

### MKL FP16 GEMV

| Matrix Size | Latency (ms) | Performance (GFLOPS) | Checksum |
|---|---:|---:|---|
| 256×256×1 | 0.0111 | 11.85 |  OK |
| 1024×1024×1 | 0.1843 | 11.38 |  OK |
| 2048×2048×1 | 0.8261 | 10.15 |  OK |
| 4096×4096×1 | 2.3248 | 14.43 |  OK |
| 8192×8192×1 | 7.9381 | 16.91 |  OK |

## 重點結論

- **T-MAC 全面領先**：8192 規模下，T-MAC 是 MKL 的約 5.3 倍、是 LUT-GEMM（AVX2）的近 80 倍。
- **矩陣越大，T-MAC 越快，LUT-GEMM 卻越慢**：T-MAC 的查表成本能被更多輸出攤提，規模越大優勢越明顯；LUT-GEMM 因 μ=8 導致 LUT 記憶體膨脹（K=8192 時約 1MB，遠超 CPU L1 快取容量），cache miss 隨規模增大而加劇，效能反而下滑。
- **關鍵差異在查表指令**：T-MAC 的 16 格 LUT（μ=4）恰好落在 `_mm256_shuffle_epi8` 的定址範圍內，可用暫存器內單週期操作查表；LUT-GEMM 的 256 格 LUT（μ=8）超出此範圍，被迫使用 `_mm256_i32gather_ps`，該指令在多數 x86 CPU 上是拆解成多次獨立記憶體存取執行，效能接近純量版本。
- **這是架構選擇的落差，而非移植錯誤**：LUT-GEMM 的 μ=8 設計是針對 GPU shared memory 高頻寬、低延遲特性最佳化的結果，移植到記憶體階層特性不同的 CPU 後，無法重現同等加速效果。
- **MKL 為穩定基準線**：不使用量化或查表，效能不受矩陣結構影響，介於 10～17 GFLOPS，代表傳統浮點運算的效能水準；T-MAC 成功超越此基準，LUT-GEMM（CPU 版）則未能超越。
- 所有測試皆為單執行緒、未做深度硬體調校，目的在於公平比較，非各方法的硬體效能極限。

## 正確性驗證方式

| 項目 | T-MAC | LUT-GEMM | MKL |
|---|---|---|---|
| 驗證方法 | Checksum 加總 + NaN/Inf 檢查 | Checksum + 解析解比對（`output[0]=-2×K`） | 已知輸入反推理論值（`C[0]=2×K`）比對 |
| 驗證強度 | 可檢測記憶體污染、數值爆炸等明顯異常 | 可精確驗證單點數值正確性 | 可精確驗證單點數值正確性（5% 容忍度） |

## 已知限制與注意事項

- 本測試固定使用單執行緒，未測試多執行緒平行效能。
- T-MAC 核心邏輯（`lut_ctor`、`tbl_update`）基於 [T-MAC 官方原始碼](https://github.com/microsoft/T-MAC) 改寫；LUT-GEMM 核心邏輯（LUT 建表、bias 讀取、查表累加）基於 [LUT-GEMM 官方 CUDA kernel](https://github.com/naver-aics/lut-gemm) 改寫，兩者運算邏輯與數值計算方式皆未經更動。
- LUT-GEMM 的 μ=8 為官方原始設計，本專案未修改此參數；若改用較小的 μ（例如 μ=4）搭配 `shuffle_epi8`，可能可改善 CPU 效能，但將偏離官方原始設計，非本專案測試範圍。
- 測試資料為固定合成數值，非真實模型權重，僅用於效能與正確性驗證，不代表真實推論任務下的表現。
- WSL2 環境可能因虛擬化層開銷與原生 Linux 環境有些微效能落差，非本測試控制範圍。

## 授權

原始 T-MAC 演算法版權歸屬 Microsoft（詳見 [T-MAC 官方 Repository](https://github.com/microsoft/T-MAC)）。
原始 LUT-GEMM 演算法版權歸屬 NAVER Cloud Corp.，採用 Apache License 2.0（詳見 [LUT-GEMM 官方 Repository](https://github.com/naver-aics/lut-gemm)）。
本專案為效能驗證與教學用途改寫。
