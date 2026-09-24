# Benchmark Comparison

> **Matrix notation：M×N×K。Latency 越低越好。**

這份文件整理本專案 x86 T-MAC、Microsoft Official T-MAC、oneMKL 與 RTX 4070 SUPER cuBLAS 的 benchmark。主比較以 **kernel-level latency** 為基準：本專案使用原始 log 中直接量測的 `kernel_ms`，Official 使用 `qgemm_lut` operator latency。

名稱統一如下：

- **Local T-MAC**：本 repository 自行實作的 x86 AVX2/F16C/OpenMP W2/W3/W4 T-MAC。
- **Microsoft Official W2/W3/W4**：Microsoft T-MAC commit `7042f8f73330bd083bc1e4bc5ccb3f88a4904aee`，TVM submodule commit `36b9535ff364c484d04b384555106731049f44cd`，以 TVM/LLVM generated `qgemm_lut` kernel + AutoTVM tuning 執行。
- **MKL FP16**：Intel Core Ultra 7 258V 上實際執行的 oneMKL dense FP16 GEMM。
- **HackMD MKL reference**：原 HackMD 保存的歷史 reference，僅作參考，不參與 CPU winner 判定。
- **RTX 4070 SUPER cuBLAS**：GPU dense FP16 GEMM reference，不是 GPU T-MAC，只作 cross-hardware reference。

<mark>本文比較的是兩套 T-MAC implementation 的 kernel/qgemm latency。Local 與 Official 的 dtype、統計方法、tuning objective、thread runtime 與 numerical input 並非完全相同，因此不把所有倍率解讀成純演算法 speedup。</mark>

## 1. Benchmark 定義與資料驗證

### Local T-MAC

Local 正式資料直接由 `x86_GEMM/hackmd_compare/results/tmac_w{2,3,4}_run1~5.log` 重建：

- 15 個 T-MAC logs。
- 75 個 `bit × shape × Threads` configurations。
- 每個 configuration 有 5 個完整 runs。
- 每列直接讀取 `kernel_ms`，不是由其他欄位相減推算。
- 每組五輪 checksum 完全一致。
- 正式 configurations 均為 `autotuned=1`。
- 最終 Local kernel latency = 五輪 `kernel_ms` 的 median。
- Audit 結果：**`LOCAL AUDIT: PASS`**。

Local timing 定義：

```text
activation
   ↓
LUT preprocessing
   ↓
preprocess_end
   ↓
T-MAC kernel
   ↓
kernel_end
```

因此：

```text
preprocess_ms = LUT preprocessing
kernel_ms     = T-MAC compute kernel
total_ms      = preprocess_ms + kernel_ms 的完整區段
```

實際統計時三個欄位各自取 median，因此表中的 `median(total)` 不要求精確等於 `median(preprocess) + median(kernel)`。

### Microsoft Official

Official adapter 將 `preprocessor` 與 `qgemm_lut` 分成獨立 operator。`qgemm_lut` 的 `evaluate()` 在 compile、input 建立、warm-up 與 correctness verification 完成後，才以 TVM `time_evaluator` 計時 compiled qgemm function。

因此 Official latency：

```text
不包含 activation → LUT preprocessing
包含 qgemm operator 內的：
LUT lookup
aggregation
scale
output conversion
```

保存的 Official raw logs另外檢查：

- 未偵測到 `NMSE` / `tvm_arrays not close` correctness warning。
- fallback configurations 明確排除，不作 tuned latency comparison。
- W2/W3 `4096³` 1/2/4/8 Threads 均無可信 tuned result。
- W3/W4 `4096×1024×2048` 1T 為 fallback。
- W4 `1024×1024×512` 1T 沒有執行到。
- W2 `4096×1024×2048` 1T 有正式 qgemm result：**136.949 ms**。

### 兩邊仍存在的方法差異

| 項目 | Local T-MAC | Microsoft Official |
|---|---|---|
| 主要比較 latency | `kernel_ms` | `qgemm_lut` |
| LUT preprocessing | 不包含 | 不包含 |
| LUT dtype | INT8 | INT8 |
| Scale / output | FP16 storage；kernel 內使用 FP32 SIMD conversion/accumulation | Intel Linux default：FP32 scale/LUT-scale/bias/output |
| Input | Local preprocessing 產生 LUT | synthetic LUT / scale tensors |
| Schedule knobs | BM / BN / KFactor | BM / BN / KFactor |
| Tuning | 手寫 AVX2/OpenMP kernel 上實測 candidates | AutoTVM + TVM/LLVM generated kernel |
| Tuning objective | `total_ms` | qgemm measurement |
| 統計 | 多 sample median + 5-run median | `number=10, repeat=10`，最後取 repeat averages 的 minimum |
| Thread runtime | OpenMP | TVM threadpool + affinity |

---

## 2. 完整 Kernel Latency 結果

`Threads 設定` 是 benchmark 呼叫時指定的 thread 數。Local T-MAC latency 後方的 `A#` 表示該次 kernel 實際啟用的 `active_threads`；例如 `0.171 (A2)` 代表設定 8 Threads，但實際只啟用 2 threads。若五輪正式 runs 出現不同 active thread 數，使用範圍表示，例如 `A2–4`。

同一列 **粗體** 為有效 CPU backend 中最低 latency。HackMD reference 與 GPU 不參與 CPU winner 判定。

| Matrix (M×N×K) | Threads 設定 | MKL FP16 (ms) | HackMD MKL ref (ms) | Local W2 kernel | Local W3 kernel | Local W4 kernel | Official W2 | Official W3 | Official W4 | RTX 4070S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **256×256×256** | 1 | **0.136** | 0.2365 | 0.341 (A1) | 0.531 (A1) | 0.725 (A1) | 0.311 | 0.475 | 0.603 | 0.006912 |
| **256×256×256** | 2 | **0.128** | 0.2465 | 0.178 (A2) | 0.270 (A2) | 0.353 (A2) | 0.350 | 0.251 | 0.681 | 0.006912 |
| **256×256×256** | 4 | **0.124** | 0.1462 | 0.213 (A4) | 0.311 (A4) | 0.180 (A4) | 0.185 | 0.130 | 0.156 | 0.006912 |
| **256×256×256** | 8 | 0.114 | 0.3258 | 0.171 (A2) | 0.261 (A2–4) | 0.149 (A8) | 0.104 | **0.094** | 0.164 | 0.006912 |
| **256×256×256** | 16 | **0.129** | 0.4081 | 0.173 (A2) | 0.265 (A2–4) | 0.151 (A4–8) | — | — | — | 0.006912 |
| **1024×1024×1024** | 1 | **7.219** | 16.7570 | 20.795 (A1) | 31.373 (A1) | 47.818 (A1) | 17.296 | 26.897 | 39.251 | 0.055104 |
| **1024×1024×1024** | 2 | 9.510 | 12.3002 | 10.838 (A2) | 15.958 (A2) | 19.396 (A2) | **8.827** | 13.796 | 19.030 | 0.055104 |
| **1024×1024×1024** | 4 | 7.205 | 6.4828 | 5.665 (A4) | 8.415 (A4) | 11.332 (A4) | **4.366** | 6.625 | 8.998 | 0.055104 |
| **1024×1024×1024** | 8 | 9.025 | 3.5757 | 6.466 (A8) | 10.330 (A8) | 10.553 (A8) | **6.004** | 8.359 | 11.984 | 0.055104 |
| **1024×1024×1024** | 16 | 7.304 | 5.9278 | **5.829 (A16)** | 8.255 (A16) | 9.763 (A16) | — | — | — | 0.055104 |
| **4096×4096×4096** | 1 | **331.529** | 2071.9777 | 1628.579 (A1) | 2189.468 (A1) | 2209.048 (A1) | — | — | — | 1.985536 |
| **4096×4096×4096** | 2 | **320.147** | 1048.0653 | 689.326 (A2) | 1087.433 (A2) | 1365.033 (A2) | — | — | — | 1.985536 |
| **4096×4096×4096** | 4 | **344.963** | 527.3677 | 348.441 (A4) | 594.978 (A4) | 786.572 (A4) | — | — | — | 1.985536 |
| **4096×4096×4096** | 8 | 324.934 | 266.5019 | **288.348 (A8)** | 482.432 (A8) | 668.103 (A8) | — | — | — | 1.985536 |
| **4096×4096×4096** | 16 | **325.085** | 260.4321 | 404.240 (A16) | 567.622 (A16) | 628.020 (A16) | — | — | — | 1.985536 |
| **4096×1024×2048** | 1 | **51.362** | 260.9551 | 181.142 (A1) | 270.854 (A1) | 270.520 (A1) | 136.949 | — (fallback) | — (fallback) | 0.277504 |
| **4096×1024×2048** | 2 | **59.344** | 130.7603 | 87.602 (A2) | 127.550 (A2) | 170.931 (A2) | 68.567 | 105.593 | 148.135 | 0.277504 |
| **4096×1024×2048** | 4 | 49.866 | 65.7123 | 44.699 (A4) | 70.022 (A4) | 99.341 (A4) | **40.510** | 60.836 | 81.747 | 0.277504 |
| **4096×1024×2048** | 8 | 51.740 | 33.2634 | **41.370 (A8)** | 66.553 (A8) | 86.023 (A8) | 43.115 | 65.414 | 75.084 | 0.277504 |
| **4096×1024×2048** | 16 | 50.053 | 72.7828 | **39.337 (A16)** | 61.349 (A16) | 81.247 (A16) | — | — | — | 0.277504 |
| **1024×1024×512** | 1 | **4.775** | 3.410722 | 11.573 (A1) | 16.807 (A1) | 16.916 (A1) | 9.095 | 14.137 | — | 0.032768 |
| **1024×1024×512** | 2 | **3.488** | 3.164299 | 5.796 (A2) | 8.488 (A2) | 10.693 (A2) | 4.600 | 7.259 | 10.599 | 0.032768 |
| **1024×1024×512** | 4 | 3.465 | 3.135987 | **3.243 (A4)** | 5.052 (A4) | 7.156 (A4) | 3.771 | 4.240 | 6.350 | 0.032768 |
| **1024×1024×512** | 8 | 3.475 | 3.123567 | 3.384 (A8) | 5.182 (A8) | 4.684 (A8) | **3.045** | 4.562 | 4.710 | 0.032768 |
| **1024×1024×512** | 16 | 5.069 | 3.145073 | **3.375 (A16)** | 4.604 (A16) | 4.783 (A16) | — | — | — | 0.032768 |

有效 matched kernel comparisons：

- **W2：Official 13/16 較快**；Local 勝出 `256³ 2T`、`4096×1024×2048 8T`、`1024×1024×512 4T`。
- **W3：Official 15/15 較快**。
- **W4：Official 10/14 較快**；Local 勝出 `256³ 2T`、`256³ 8T`、`1024³ 8T`、`1024×1024×512 8T`。

<mark>W3 是目前最一致的 kernel-level gap；W2 與 W4 則已出現多個 Local 較快的 shape/thread configuration。</mark>

---

## 3. Local `total_ms` 與 `kernel_ms`

`total_ms` 仍有意義：它代表 Local T-MAC 從 LUT preprocessing 到 kernel 完成的完整執行區段。因為 Official 主比較只量 `qgemm_lut`，所以 `total_ms` 不放進主表，而獨立保留作 Local pipeline 參考。

以下以 W2 的代表性 configuration 說明：

| Matrix | Threads | total_ms | preprocess_ms | kernel_ms | preprocessing 約占 total |
|---|---:|---:|---:|---:|---:|
| **256×256×256** | 8 | 0.288 | 0.114 | 0.171 | 39% |
| **1024×1024×1024** | 4 | 7.517 | 1.871 | 5.665 | 25% |
| **4096×4096×4096** | 8 | 320.574 | 32.480 | 288.348 | 10% |
| **4096×1024×2048** | 8 | 45.987 | 4.567 | 41.370 | 10% |
| **1024×1024×512** | 4 | 4.185 | 0.998 | 3.243 | 24% |

各欄均由五輪資料各自取 median，因此不要求三個 median 精確相加。

這張表顯示 preprocessing 對小、中型矩陣影響較明顯；對大型矩陣則主要時間仍在 compute kernel。因此：

```text
Local end-to-end-like timing → 看 total_ms
Local vs Official qgemm     → 看 kernel_ms
```

兩者回答的是不同問題，不應混在同一欄比較。

---

## 4. 各 Shape 的主要結果

這裡統一比較 **1–8 Threads**，因為本機只有 8 個 hardware threads；16T 屬於 oversubscription reference，不拿來決定主要 CPU winner。

| Matrix | MKL 最佳 | Local T-MAC 最佳 | Official 最佳 | 最快 CPU | GPU cuBLAS | Fastest CPU / GPU |
|---|---:|---:|---:|---:|---:|---:|
| **256×256×256** | 0.114 ms (8T) | 0.149 ms (W4, 8T) | **0.094 ms (W3, 8T)** | **Official W3 0.094 ms** | 0.006912 ms | 13.6× |
| **1024×1024×1024** | 7.205 ms (4T) | 5.665 ms (W2, 4T) | **4.366 ms (W2, 4T)** | **Official W2 4.366 ms** | 0.055104 ms | 79.2× |
| **4096×4096×4096** | 320.147 ms (2T) | **288.348 ms (W2, 8T)** | — | **Local W2 288.348 ms** | 1.985536 ms | 145.2× |
| **4096×1024×2048** | 49.866 ms (4T) | 41.370 ms (W2, 8T) | **40.510 ms (W2, 4T)** | **Official W2 40.510 ms** | 0.277504 ms | 146.0× |
| **1024×1024×512** | 3.465 ms (4T) | 3.243 ms (W2, 4T) | **3.045 ms (W2, 8T)** | **Official W2 3.045 ms** | 0.032768 ms | 92.9× |

這張表把 MKL 也納入主要比較，因此可以直接看出：

- **256³**：Official W3 最快，但 Local W2/W3 在高 Threads 設定時實際 active threads 未必等於設定值，因此不適合作為純 thread scaling 證據。
- **1024³**：Official W2 4T = 4.366 ms，Local W2 4T = 5.665 ms，MKL 4T = 7.205 ms。Local 已快於 MKL，但仍慢於 Official。
- <mark>**4096³：Local W2 8T = 288.348 ms，是目前可用 CPU 結果中最快，也低於 MKL 最佳 320.147 ms。**</mark> Official W2/W3 因 tuning measurement timeout 沒有可信 tuned result，因此這個 shape 不能做 Local vs Official 的完整比較。
- **4096×1024×2048**：Official W2 4T = 40.510 ms、Local W2 8T = 41.370 ms、MKL 4T = 49.866 ms；三者中 Official 最快，但 Local 與 Official 只差約 2.1%。如果額外看 16T oversubscription，Local W2 可到 39.337 ms，但不列入主要 winner。
- **1024×1024×512**：Official W2 8T = 3.045 ms、Local W2 4T = 3.243 ms、MKL 4T = 3.465 ms；Local 同樣已快於 MKL，但 Official 仍最低。

## 5. Local 與 Official 的效能差距

先把最容易混淆的一點講清楚：**Local `kernel_ms` 與 Official `qgemm_lut` 都是在 LUT 已經存在之後量 compute，但它們不是同一支 kernel。**

### Local kernel 與 Official qgemm 到底差在哪裡？

兩者都屬於 T-MAC 的 lookup-based matrix multiplication，概念上都是：

```text
low-bit packed weights
        +
precomputed activation LUT
        ↓
LUT lookup
        ↓
integer / partial accumulation
        ↓
scale / bit-plane combination
        ↓
output
```

但實作方式不同：

| 項目 | Local `kernel_ms` | Official `qgemm_lut` |
|---|---|---|
| Kernel來源 | 手寫 C++ AVX2/F16C/OpenMP | TVM schedule → LLVM generated kernel |
| LUT來源 | Local preprocessing 由 activation 建立後交給 kernel | benchmark 直接建立 synthetic INT8 LUT tensor |
| Weight layout | Local offline packing / interleave layout | Official codegen 對應的 packed layout |
| LUT lookup | 手寫 AVX2 shuffle / widening path | TVM tensorized lookup / generated vector code |
| Scale path | FP16 storage，載入後轉 FP32 SIMD 計算 | Intel Linux default 使用 FP32 scale / LUT-scale / bias |
| Output | FP16 storage | FP32 output |
| Parallel execution | OpenMP | TVM threadpool + affinity |
| Schedule | Local 從 BM/BN/KFactor candidates 中實測選擇 | AutoTVM schedule + code generation 一起決定 |
| 計時起點 | LUT preprocessing 完成後 | qgemm input tensors 已存在後 |
| 計時終點 | Local output store 完成 | Official qgemm output 完成 |

所以兩邊比較的是：

```text
Local：
已經建好 LUT
→ 進入手寫 AVX2/OpenMP T-MAC kernel
→ FP16 output

Official：
已經有 LUT / scale tensors
→ 進入 TVM/LLVM generated qgemm
→ FP32 output
```

<mark>因此現在的比較已經比 `total_ms vs qgemm` 公平很多，但仍然是在比較「兩套不同 implementation 的 qgemm/kernel」，不是逐指令完全相同的 microbenchmark。</mark>

### 目前看起來影響較大的原因

下面的排序是根據目前 benchmark 行為與程式碼路徑做的 **證據優先級**，不是已經量化好的百分比貢獻。

#### 1. Schedule × 實際 thread partition

這是目前最明顯、而且能直接從結果看到的因素。

BM / BN 不只決定 cache blocking，也會決定：

```text
M / N 被切成多少 tiles
        ↓
有多少工作可以平行
        ↓
實際 active threads
        ↓
每個 thread 做多少 tile
        ↓
cache reuse / contention / synchronization
```

所以同樣設定 8 Threads，不代表一定真的有 8 個 workers 在有效工作。例如 `256³ W2` 設定 8T 時實際只有 `A2`；VLA 中也有多個設定 8T 但實際只有 A2/A4/A5 的 case。

更重要的是，即使真正從 A4 增加到 A8，也會出現很大的 regression。例如：

```text
VLA W4 Action Residual
A4: 0.563 ms
A8: 5.767 ms
```

以及：

```text
VLA W2 Vision Attention
A4: 1.604 ms
A8: 9.016 ms
```

這種數倍差距不太可能只靠統計誤差或 FP16/FP32 解釋，代表某些 schedule 在高 active-thread 下會造成非常差的 work partition、cache behavior 或 memory contention。

因此目前最值得優先研究的是：

> **Local tuner 選到的 BM/BN/KFactor，到了不同 thread 數後，是否讓 tile 數、parallel axis 或每-thread workload 變得不合理。**

#### 2. Generated kernel 與手寫 kernel 的低階 code quality

Thread 問題不是全部原因，因為 1T 仍然存在差距。

例如：

```text
1024³ W2 1T
Local    20.795 ms
Official 17.296 ms
```

只有一個 thread 時，已經沒有 thread partition 問題，但 Official 仍快約 20%。

這時較可能的差異包括：

- register allocation 是否更有效；
- load/store 次數；
- LUT lookup 指令排列；
- widening / accumulation 是否產生多餘指令；
- loop unrolling；
- vectorization；
- instruction scheduling；
- register spill；
- cache line 使用方式。

Official 的 schedule 最後會直接參與 TVM tensorization / vectorization / LLVM code generation；Local 則是固定的手寫 AVX2/F16C kernel，再套不同 BM/BN/KFactor。

所以即使兩邊：

```text
BM = 相近
BN = 相近
KFactor = 相同
```

最後 assembly 仍可能非常不同。

不過這一點目前還不能直接說「Official assembly 一定比較好」，因為 Local 在：

```text
W2 4096×1024×2048 8T
W2 1024×1024×512 4T
部分 W4 cases
```

已經能低於 Official。

因此更精確的結論是：

> **Official generated code 在部分 shape 上具有明顯優勢，但這個優勢不是固定倍率，而是 workload-dependent。**

要真正確認這一層，需要固定同一組 BM/BN/KFactor、同一 thread 數，再比較 assembly 與 hardware counters。

#### 3. Local autotune 的目標和最後比較指標不同

Local autotune 現在是：

```text
每個 candidate
→ 跑 total_ms
→ 取 median
→ 選 total_ms 最低的 schedule
```

但本文最後比較的是：

```text
kernel_ms
```

所以 Local 實際做的是：

```text
argmin(total_ms)
→ 再看這個 schedule 的 kernel_ms
```

而不是：

```text
argmin(kernel_ms)
```

Official AutoTVM 則是直接以 qgemm measurement 選 schedule。

這表示 Local 可能選到：

```text
preprocessing + kernel 合計最好
```

但 kernel 本身不是最快的 schedule。

這個問題很可能會讓 Local kernel-only comparison 吃虧，但目前還沒有量化到底影響多少。最簡單的 controlled experiment 是把 Local tuner 暫時改成用 `kernel_ms` 選 schedule，再看主表是否明顯變化。

#### 4. dtype / output path 不同

Local 與 Official 不是完全相同 precision path：

```text
Local：
FP16 scale/output storage
→ F16C 轉 FP32 SIMD 計算
→ FP16 store

Official：
FP32 scale / LUT-scale / bias
→ FP32 output
```

這可能影響：

- memory traffic；
- load/store bandwidth；
- conversion 指令；
- register pressure；
- output store size。

但目前證據看起來它**不像主要原因**，因為如果 FP32 output 本身就是決定性劣勢，Official 不應該在 W3 15/15 matched cases 都比 Local 快。

所以 dtype 比較適合視為：

> **會影響 latency，但目前不像能解釋數倍差距的第一主因。**

#### 5. Thread runtime / affinity

Local 使用 OpenMP；Official 使用 TVM threadpool，且 benchmark 設定 `thread_affinity=1`。

這會影響：

- worker 啟動與 reuse；
- core binding；
- scheduling overhead；
- cache locality；
- heterogeneous-core placement。

在 Intel Core Ultra 7 258V 這類 CPU 上，thread placement 可能很重要。

不過目前這一點和前面的 schedule / active-thread partition 很難完全拆開，因此比較合理的做法是把它視為第二層 parallelization 問題，而不是單獨宣稱它就是主因。

#### 6. 統計 estimator

Local 最終是 median-based：

```text
internal samples median
→ 5 complete runs
→ outer median
```

Official 是：

```text
10 calls / repeat
× 10 repeats
→ 取 10 個 repeat-average 中最小值
```

所以 Official 比較接近「best stable repeat」，Local 比較接近「typical run」。

這可能讓 Official 的 latency略低一些，尤其是非常接近的 case。例如：

```text
Local W4 1024×1024×512 8T = 4.684 ms
Official W4                    = 4.710 ms
```

這種 0.x% 的差距不值得解讀成明確勝負。

但統計方式不可能合理解釋：

```text
0.563 ms vs 5.767 ms
```

這種數倍差距，因此它的優先級較低。

#### 7. Numerical input 不同

Local LUT 是由實際 random activation preprocessing 產生；Official qgemm microbenchmark 直接產生 synthetic LUT/scale tensor。

兩邊 tensor shape 與資料量對齊，但值本身不同。

對這種固定 control-flow 的 LUT kernel，數值內容通常不像 shape、layout、schedule、thread partition 那樣會大幅改變執行時間，所以目前把它視為較小的 methodology 差異，而不是主要 performance 原因。

### 目前最合理的優先順序

綜合目前證據：

```text
1. Schedule × active-thread work partition
           ↓
2. Generated code / instruction-level implementation
           ↓
3. Local autotune objective 使用 total_ms 而不是 kernel_ms
           ↓
4. dtype / output path
           ↓
5. thread runtime / affinity
           ↓
6. statistical estimator
           ↓
7. numerical input difference
```

<mark>最值得先驗證的是前三項。</mark> 因為它們可以同時解釋「Local 有些 shape 接近甚至超過 Official」以及「某些 VLA/high-thread case 卻突然慢數倍」這兩種看似矛盾的結果。若先固定 BM/BN/KFactor 與 active threads，再比較 1T assembly，接著把 Local autotune objective 改成 `kernel_ms`，就可以逐步把這三個因素拆開。

---

## 6. VLA Matrix Benchmark

這一節比較 VLA 代表性矩陣上的 **MKL FP16、Local T-MAC W2/W4、Microsoft Official T-MAC W2/W4**。Pi0 Action Expert 使用 **N=32 proxy**，不是原始 N=33 exact shape；這是為了符合 Official T-MAC legal BN 的限制。Local 欄位中的 `A#` 表示實際 `active_threads`。

### MKL VLA 結果怎麼算

MKL VLA 使用與目前 VLA benchmark 相同的 8 個 shapes，threads 固定為 `1,4,8`。每個完整 benchmark 重跑 5 次，每次執行前先 warm-up 10 次，計時區間只包含 `cblas_hgemm()`。

最終數值使用兩層 median：

```text
對每個 (workload, threads)：

第 1 層：單次完整 run
cblas_hgemm 重複量測 samples 次
→ 取 samples 的 total_ms median
→ 得到該 run latency

第 2 層：完整 benchmark 重跑 5 次
run1 latency
run2 latency
run3 latency
run4 latency
run5 latency
→ 再取 5 個 latency 的 median
→ 最終 MKL VLA latency
```

`sample_count()` 與 Local VLA harness 相同：依 logical FLOPs 使用 21 / 11 / 7 / 5 samples。本次 VLA shapes 實際使用 11 samples，只有 Pi0 KV 使用 21 samples。五輪共取得 **120 筆 RESULT = 8 workloads × 3 thread settings × 5 runs**；每個 `(workload, threads)` 都完整存在，且同一 workload 的五輪 checksum 一致。

例如 Action Residual 1T 五輪為：

```text
5.803760
6.061880
6.530492
5.435410
5.678600
```

排序後中位數為 **5.803760 ms**，因此正式表使用 `5.804 ms`。

`p90_ms` 只用來觀察單次 run 內的波動，不作為最後 latency；checksum 只用來確認輸出一致，不參與效能計算。

需要注意統計方法仍不完全相同：

- **MKL VLA**：internal sample median + 5-run outer median。
- **Local VLA**：單次 benchmark invocation 的 internal median，目前沒有 5-run outer median。
- **Official VLA**：TVM `number=10, repeat=10`，取 repeat averages 的 minimum。

因此數%的小差距不應過度解讀；數倍差距則仍具有明顯意義。

### Threads = 1

| Workload | Matrix | MKL FP16 | Local W2 | Official W2 | Local W4 | Official W4 |
|---|---:|---:|---:|---:|---:|---:|
| Action Residual | **2560×8×2560** | 5.804 | 0.971 (A1) | **0.819** | 2.073 (A1) | 1.623 |
| Action FC1 | **2560×8×17920** | 33.127 | 8.993 (A1) | **6.187** | 16.397 (A1) | 11.731 |
| Vision Attention | **1152×256×1152** | **1.907** | 6.370 (A1) | 5.470 | 13.638 (A1) | 11.006 |
| Pi0 Q | **2048×32×1024** | 1.604 | 1.219 (A1) | **1.077** | 2.429 (A1) | 2.145 |
| Pi0 KV | **256×32×1024** | **0.081** | 0.155 (A1) | 0.136 | 0.311 (A1) | 0.267 |
| Pi0 O | **1024×32×2048** | 1.532 | 1.179 (A1) | **1.057** | 2.374 (A1) | 2.092 |
| Pi0 GateUp | **4096×32×1024** | 3.779 | 3.027 (A1) | **2.184** | 6.046 (A1) | 4.315 |
| Pi0 Down | **1024×32×4096** | 3.753 | 2.508 (A1) | **2.064** | 4.956 (A1) | 4.306 |

### Threads = 4

| Workload | Matrix | MKL FP16 | Local W2 | Official W2 | Local W4 | Official W4 |
|---|---:|---:|---:|---:|---:|---:|
| Action Residual | **2560×8×2560** | 5.412 | 0.673 (A4) | **0.217** | 0.563 (A4) | 0.411 |
| Action FC1 | **2560×8×17920** | 34.397 | 11.103 (A4) | **1.485** | 10.301 (A4) | 3.000 |
| Vision Attention | **1152×256×1152** | 1.826 | 1.604 (A4) | **1.384** | 8.188 (A4) | 2.788 |
| Pi0 Q | **2048×32×1024** | 1.571 | 0.816 (A4) | **0.280** | 0.752 (A4) | 0.558 |
| Pi0 KV | **256×32×1024** | 0.080 | 0.080 (A2) | 0.133 | **0.080 (A4)** | 0.267 |
| Pi0 O | **1024×32×2048** | 1.521 | 0.615 (A2) | **0.281** | 0.608 (A4) | 1.056 |
| Pi0 GateUp | **4096×32×1024** | 3.909 | 1.707 (A4) | **0.558** | 1.287 (A4) | 1.109 |
| Pi0 Down | **1024×32×4096** | 3.666 | 0.609 (A4) | **0.545** | 1.285 (A4) | 1.068 |

### Threads = 8

| Workload | Matrix | MKL FP16 | Local W2 | Official W2 | Local W4 | Official W4 |
|---|---:|---:|---:|---:|---:|---:|
| Action Residual | **2560×8×2560** | 5.586 | 0.560 (A5) | **0.211** | 5.767 (A8) | 0.413 |
| Action FC1 | **2560×8×17920** | 33.208 | 12.654 (A8) | **1.519** | 13.983 (A8) | 3.053 |
| Vision Attention | **1152×256×1152** | 1.867 | 9.016 (A8) | **1.388** | 4.850 (A8) | 2.803 |
| Pi0 Q | **2048×32×1024** | 1.562 | 0.801 (A4) | **0.288** | 5.434 (A8) | 0.573 |
| Pi0 KV | **256×32×1024** | 0.096 | **0.081 (A2)** | 0.133 | 0.082 (A4) | 0.142 |
| Pi0 O | **1024×32×2048** | 1.560 | 0.620 (A2) | **0.281** | 1.588 (A4) | 0.548 |
| Pi0 GateUp | **4096×32×1024** | 3.803 | 5.316 (A8) | **0.578** | 2.999 (A8) | 2.282 |
| Pi0 Down | **1024×32×4096** | 4.198 | 1.291 (A2) | **0.541** | 9.214 (A8) | 1.074 |

### VLA 結果整理

從 24 個 `(workload, threads)` configurations 比較：

| 比較 | 較快 configurations |
|---|---:|
| Local W2 vs MKL | **19 / 24** |
| Official W2 vs MKL | **20 / 24** |
| Local W4 vs MKL | **12 / 24** |
| Official W4 vs MKL | **14 / 24** |
| Local W2 vs Official W2 | **2 / 24** |
| Local W4 vs Official W4 | **3 / 24** |

若直接在五個 backends 中選每個 configuration 的最低 latency：

- **Official W2：20 / 24 最快**
- **MKL：2 / 24 最快**（Vision Attention 1T、Pi0 KV 1T）
- **Local W4：1 / 24 最快**（Pi0 KV 4T）
- **Local W2：1 / 24 最快**（Pi0 KV 8T）
- **Official W4：0 / 24 最快**

這些結果顯示出幾個很清楚的 pattern：

1. **Official W2 是目前 VLA 測試中整體最強的 backend。** 它不只是比 Local W2 穩定，在 Action Residual、Action FC1、Q、O、GateUp、Down 等多數 workload 上也明顯低於 MKL。
2. **Local W2 雖然離 Official W2 還有明顯差距，但不是沒有實用價值。** 它在 19/24 configurations 低於 MKL，代表 low-bit LUT kernel 對不少 VLA shape 的確有 CPU latency 優勢。
3. **Local W2 最大的問題不是所有 shape 都慢，而是穩定性與 schedule sensitivity。** 例如 Vision Attention 4T 為 1.604 ms，已低於 MKL 1.826 ms；但 8T 卻惡化到 9.016 ms，而 MKL 仍是 1.867 ms。GateUp 也有類似 4T 好、8T 明顯 regression 的現象。
4. **W4 的優勢不夠穩定。** Local W4 對 MKL 剛好 12/24 勝、12/24 敗，而且在 Vision Attention、Q 8T、Down 8T 等 workload 出現明顯 regression。以目前結果，W4 不適合被描述成普遍優於 dense FP16 baseline。
5. **MKL 在部分矩陣仍是合理甚至最佳選擇。** Vision Attention 1T 的 MKL 為 1.907 ms，Official W2 為 5.470 ms、Local W2 為 6.370 ms；Pi0 KV 1T 也是 MKL 0.081 ms 最快。這表示沒有必要強迫所有 VLA GEMM 都走 T-MAC。
6. **Pi0 KV 很接近 latency floor。** 4T 時 MKL 0.080023 ms、Local W4 0.079933 ms，差距只有約 0.1%，在目前不同統計 estimator 下不應宣稱有實質勝負；8T Local W2 0.080595 ms 則比 MKL 0.096187 ms 低約 16%，但仍應以更多重複資料確認。

<mark>VLA 的結果支持「依 workload dispatch」，而不是「單一 backend 全包」。Official W2 已證明 T-MAC 在不少 VLA shape 上可以大幅低於 MKL，但 Local implementation 還沒有把這個優勢穩定地重現到所有 shape / thread configuration。</mark>

---

## 7. 結論與後續研究方向

整體結果可以分成兩件事看。第一，**T-MAC 這條路本身是有價值的**：在一般矩陣 benchmark 中，Local W2 已在部分 configuration 接近或低於 Official，`4096³` 8T kernel 也低於本機 MKL FP16；在 VLA benchmark 中，Official W2 更是 24 個 configuration 中有 20 個為五個 backends 的最低 latency，Local W2 也有 19/24 低於 MKL。這表示 low-bit LUT-based GEMM 並不是只在理論上有優勢，它確實能在部分 CPU workload，尤其是 narrow-N / 大 K 或 M 的矩陣上降低 latency。

第二，**本專案目前和 Microsoft Official 之間仍有明顯差距，而且不能用「資料型別不同」或「thread 不一樣」來掩飾。** 在 VLA W2，Local 只有 2/24 低於 Official；W4 也只有 3/24。Action FC1、Vision Attention、GateUp 等 workload 在部分 thread 設定下甚至出現數倍差距。這表示 Local 的 schedule selection、active-thread partition 與 hand-written kernel code quality 還有實質改善空間。尤其高 thread 數並不穩定：有些 case 沒有真正使用滿設定的 threads，有些則在 A4→A8 後直接 regression。這不是「多執行緒一定沒有用」，而是目前 Local implementation 對不同 shape 的 parallel decomposition 還不夠成熟。

MKL 的角色也更清楚了。它不是所有 VLA matrix 的最佳選擇，但在 Vision Attention 1T、Pi0 KV 1T，以及部分 W4 case 上仍具有優勢。<mark>因此合理的工程目標不是把 MKL 全部換掉，而是找出 T-MAC 穩定有利的 workload region，其他情況保留 MKL。</mark> 對 VLA 特別適合採用 hybrid dispatch：根據 `M/N/K、bit-width、thread count` 選擇 T-MAC 或 MKL，而不是假設一種 kernel 可以處理所有矩陣。

對研究方向而言，**W2 最值得優先投入**。原因不是它在所有地方都最快，而是它同時具備兩個條件：Official W2 已經證明 VLA 上有很大的潛在上限，而 Local W2 也已在 19/24 configurations 低於 MKL，表示本專案距離「實際可用」不是完全沒有基礎。相對地，W4 目前對 MKL 只有 12/24 優勢，且 regression 較多；在沒有先解決 schedule / threading 問題前，不適合把 W4 當成主要優化方向。

目前 VLA 測試中的 N 主要是 8、32、256，因此可以稱為 **narrow-N / decode-like**，但還不能直接宣稱已證明 autoregressive decode 的優勢。真正要回答「T-MAC 是否適合 decode」還需要固定代表性的 M/K，系統性測 `N=1, 8, 32`，並同時比較 Local、Official、MKL。若 W2 在 N=1/8/32 都能形成穩定優勢，才有足夠證據說 T-MAC 特別適合 decode-like CPU inference；如果只有某些 N 或某些 shape 有利，就應該把它描述成 workload-specific optimization，而不是泛化成 decode 優勢。

下一步最有價值的實驗不是再增加大量隨機 shape，而是做 controlled experiments：先固定相同 `BM/BN/KFactor` 比較 Local 與 Official 的 1T assembly，再固定 schedule 比較 A1/A2/A4/A8，確認 regression 是來自 tile partition、cache、memory bandwidth、register spill 還是 thread runtime；同時把 Local autotune objective 從 `total_ms` 改成 `kernel_ms` 做一次 A/B test。最後再建立 `M/N/K × bit-width × threads` 的 performance map，決定哪些區域使用 T-MAC、哪些直接交給 MKL。

<mark>目前最客觀的結論是：Microsoft Official T-MAC 已經展現出很強、且相對穩定的 VLA CPU kernel 效能；本專案 Local W2 在不少 VLA shape 上已能勝過 MKL，但與 Official 仍有明顯 implementation gap；Local W4 的表現則較不穩定。T-MAC 值得繼續做，但目標應該是把 Official 已證明存在的優勢轉化成 Local 可穩定重現的 workload-specific backend，而不是追求所有矩陣全面取代 MKL。</mark>
