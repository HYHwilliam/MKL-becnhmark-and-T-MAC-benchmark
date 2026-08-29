# x86 HackMD Benchmark

此資料夾用來重現 HackMD 中的 x86 GEMM 測試，包含 T-MAC W2/W3/W4A16、oneMKL FP16，以及不同 CPU 間的 T-MAC 結果比較。

## 主要檔案

| 檔案 | 用途 |
|---|---|
| `tmac_hackmd_benchmark.cpp` | T-MAC W2/W3/W4A16 benchmark |
| `mkl_fp16_hackmd_benchmark.cpp` | oneMKL dense FP16 benchmark |
| `summarize_results.py` | 統整原始 MKL + T-MAC 五輪結果 |
| `compare_tmac_cpus.py` | 比較 Intel 與 Ryzen 的 T-MAC 五輪統計 |
| `results/` | 統計結果與正式報告 |

Shape 一律以 `M×N×K` 顯示。

---

## 1. 編譯 T-MAC

需求：

- x86-64 CPU
- AVX2
- F16C
- FMA
- OpenMP
- g++

在 repository root 執行：

```bash
g++ -O3 -std=c++17 \
    -mavx2 -mfma -mf16c -fopenmp \
    x86_GEMM/hackmd_compare/tmac_hackmd_benchmark.cpp \
    -o x86_GEMM/hackmd_compare/tmac_hackmd_benchmark
```

### OpenMP 環境

正式測試前建議：

```bash
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=ACTIVE
```

這些設定只影響目前 shell。

---

## 2. 執行 T-MAC

單獨測 W2：

```bash
./x86_GEMM/hackmd_compare/tmac_hackmd_benchmark \
    --bits 2 \
    --threads 1,2,4,8,16 \
    --autotune
```

W3 / W4：

```bash
./x86_GEMM/hackmd_compare/tmac_hackmd_benchmark \
    --bits 3 \
    --threads 1,2,4,8,16 \
    --autotune

./x86_GEMM/hackmd_compare/tmac_hackmd_benchmark \
    --bits 4 \
    --threads 1,2,4,8,16 \
    --autotune
```

`--autotune` 會在官方風格的 `bm / bn / kfactor` candidate 中實測並選擇最快 schedule。Autotuning 所花時間不計入正式 `total_ms`。

### 五輪正式測試

例如 Ryzen 實驗：

```bash
RESULT_DIR=x86_GEMM/hackmd_compare/results/lab_ryzen8500g
mkdir -p "$RESULT_DIR"

for bit in 2 3 4
do
    for run in 1 2 3 4 5
    do
        ./x86_GEMM/hackmd_compare/tmac_hackmd_benchmark \
            --bits "$bit" \
            --threads 1,2,4,8,16 \
            --autotune \
            | tee "$RESULT_DIR/tmac_w${bit}_run${run}.log"
    done
done
```

每個 log 應有：

```text
5 shapes × 5 thread settings = 25 RESULT rows
```

W2/W3/W4 × 5 runs 共：

```text
15 logs
375 RESULT rows
```

---

## 3. 編譯 oneMKL FP16

此程式需要已安裝 oneMKL，並且目前 shell 已有正確的 `MKLROOT`。

使用 g++ 的範例：

```bash
g++ -O3 -std=c++17 -fopenmp \
    -I"$MKLROOT/include" \
    x86_GEMM/hackmd_compare/mkl_fp16_hackmd_benchmark.cpp \
    -L"$MKLROOT/lib/intel64" \
    -Wl,-rpath,"$MKLROOT/lib/intel64" \
    -lmkl_intel_lp64 \
    -lmkl_gnu_thread \
    -lmkl_core \
    -lpthread -lm -ldl \
    -o x86_GEMM/hackmd_compare/mkl_fp16_hackmd_benchmark
```

若目前環境沒有 oneMKL，請不要直接安裝或修改系統套件；先確認既有 MKL 環境。

---

## 4. 統整 Intel 原始結果

原始 MKL + T-MAC 五輪結果由：

```bash
python3 \
    x86_GEMM/hackmd_compare/summarize_results.py
```

主要輸出位於：

```text
x86_GEMM/hackmd_compare/results/
```

其中 `summary_statistics.csv` 是後續跨 CPU 比較使用的正式統計來源。

---

## 5. Intel vs Ryzen T-MAC 比較

`compare_tmac_cpus.py` 讀取：

```text
原始 Intel：
results/summary_statistics.csv

實驗室 Ryzen：
results/lab_ryzen8500g/tmac_w{2,3,4}_run{1..5}.log
```

執行：

```bash
python3 \
    x86_GEMM/hackmd_compare/compare_tmac_cpus.py
```

輸出：

```text
results/cpu_tmac_compare/
├── validation_report.txt
├── lab_tmac_statistics.csv
├── tmac_cpu_comparison.csv
└── tmac_cpu_comparison.md
```

其中：

```text
Ryzen speedup = Intel total_ms / Ryzen total_ms
```

大於 `1.0×` 表示 Ryzen 較快。

主要跨 CPU 解讀使用 `1T / 2T / 4T / 8T`；16T 保留作 HackMD thread sweep 對齊，但需注意 oversubscription。

---

## 6. 統計方式

正式結果使用 5 次獨立 run。

同一 configuration 的主值為：

```text
median(run1, run2, run3, run4, run5)
```

報告另外保留 Mean、Std、CV、Min/Max 與 schedule stability。

T-MAC 的 GFLOPs 使用：

```text
2 × M × N × K / latency
```

這是完成相同 logical GEMM 的 dense-equivalent throughput，不代表 CPU 真正執行相同數量的 FP16 FMA。

---

## 7. 注意事項

- `total_ms` 為 T-MAC preprocessing + kernel。
- weight packing 與 autotuning time 不計入正式 latency。
- `active_threads` 可能小於 requested threads，尤其是小 shape。
- 16T 在硬體 thread 數不足時屬於 oversubscription。
- raw `*.log` 目前由 repository `.gitignore` 忽略；正式 CSV / Markdown 統計結果可提交 Git。
