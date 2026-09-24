# Benchmark Comparison

> **Matrix notation：M×N×K。Latency 越低越好。**  
> 本文件以 **kernel-level latency** 為主要比較基準，並把 2026-09-24 完成的資料來源、correctness、fallback、thread 與 timing audit 一併納入。

這份文件整理本專案 x86 T-MAC、Microsoft Official T-MAC、oneMKL 與 RTX 4070 SUPER cuBLAS 的 benchmark。<mark>本版不再使用本專案 `total_ms` 與 Official `qgemm_lut` 直接比較；本專案一律改用原始 log 中直接量測的 `kernel_ms`。</mark>

名稱統一如下：

- **本專案 T-MAC**：本 repository 自行實作的 x86 AVX2/F16C/OpenMP W2/W3/W4 T-MAC。
- **Microsoft Official W2 / W3 / W4**：Microsoft T-MAC commit `7042f8f73330bd083bc1e4bc5ccb3f88a4904aee`，TVM submodule commit `36b9535ff364c484d04b384555106731049f44cd`，以 TVM/LLVM generated `qgemm_lut` kernel + AutoTVM tuning 執行。
- **MKL FP16（本機）**：Intel Core Ultra 7 258V 上實際執行的 oneMKL dense FP16 GEMM。
- **學長 HackMD MKL reference**：原 HackMD 的歷史 reference，與本機 MKL 分開解讀，不參與 row winner。
- **RTX 4070 SUPER cuBLAS**：GPU dense FP16 GEMM reference，不是 GPU T-MAC，只作 cross-hardware reference。

<mark>重要：目前 Local 與 Official 已對齊到「compute/qgemm operator」層級，但仍不是完全 apples-to-apples。</mark> Local 與 Official 在 output/scale dtype、統計 estimator、tuning objective、thread runtime/affinity 與實際 numerical input 上仍存在差異；因此本文可用來比較 **implementation-level kernel latency**，不能把所有倍率直接解讀成純演算法或單一 codegen optimization 的 speedup。

## 1. 資料驗證與比較規則

### Local kernel audit

2026-09-24 直接從 `x86_GEMM/hackmd_compare/results/tmac_w{2,3,4}_run1~5.log` 重算：

- 15 個 T-MAC source logs。
- 75 個 `bit × shape × thread-budget` configurations。
- 每個 configuration **5 個完整 runs**。
- 每列直接讀取 `kernel_ms`，不是用 `total_ms - preprocess_ms` 反推。
- 每組五輪 `checksum` 差皆為 `0.000e+00`。
- 所有正式 configuration 均為 `autotuned=1`。
- 最終 kernel latency = **五個 run-level `kernel_ms` medians 的 median**。
- Audit 結果：**`LOCAL AUDIT: PASS`**。

因此本文 Local HackMD `kernel_ms` 已能由原始 Intel logs 完整重建。

### Official qgemm audit

Official adapter 的 `qgemm_lut` 與 `preprocessor` 是獨立 operator；`evaluate()` 在 compile、array 建立、warm-up 與 verify 完成後才以 TVM `time_evaluator` 計時 compiled `qgemm_lut`。所以 Official latency 可視為 **完整 qgemm operator kernel timing**，不包含 activation→LUT preprocessor；但 qgemm operator 本身仍包含 lookup、aggregation、scale 與 output conversion。

2026-09-24 對保存的 `official_compare/results/raw` 掃描：

- `tvm_arrays not close / NMSE / nmse`：**沒有找到 correctness warning**。
- 已知 fallback 仍可在 raw logs 中明確追溯，未混入正式 tuned comparison。
- W2 `4096×1024×2048` 1T 的 `136.949 ms` **存在正式結果，且 fallback scan 未命中該 workload，因此本版恢復為有效結果**；舊版文件將它排除是過度保守。

### Timing / dtype / statistics 差異

| 項目 | 本專案 T-MAC | Microsoft Official qgemm | 解讀 |
|---|---|---|---|
| 計時範圍 | `kernel_ms = preprocess_end → kernel_end` | `qgemm_lut` compiled operator | 已大致對齊 compute/qgemm 層級 |
| LUT preprocessing | 不含於 `kernel_ms` | 不含，為獨立 preprocessor operator | 不再有原本 `total_ms vs qgemm` 的 timing-boundary 偏差 |
| LUT | INT8 | INT8 | CSV 的 `dtype=int8` 指 LUT dtype |
| Scale / output path | FP16 storage，kernel 內有 FP32 SIMD conversion/accumulation | Intel Linux default 為 FP32 scale/LUT-scale/bias/output，aggregation 可為 INT32/FP32 | **precision path 不完全相同** |
| Local/Official input | Local LUT 由本專案 activation preprocessing 產生 | qgemm microbenchmark 直接建立 synthetic LUT / scale tensors | shape/bit/group 對齊，但不是同一 numerical input |
| Tuning candidates | 使用 Official 對應的 BM/BN/KFactor candidate space | AutoTVM GridSearch | 搜尋參數方向一致 |
| Tuning objective | 依 **`total_ms`** 選 Local schedule | 直接依 qgemm measurement tune | Local 被選中的 schedule 不保證是 `kernel_ms` 最佳 |
| 最終統計 | internal median，再做 5-run outer median | `number=10, repeat=10`，最後取 repeat averages 的 `min` | Official 偏 best-stable case，Local 偏 typical case |
| Thread runtime | OpenMP；requested budget 可能大於 `active_threads` | TVM threadpool，`thread_affinity=1` | thread 數不可機械視為完全相同 runtime |
| Correctness | formal verifier + 五輪 checksum 穩定 | qgemm verifier；保存 raw logs 未見 NMSE warning | 目前採用結果未發現 correctness anomaly |

> <mark>因此本文最準確的定位是：**verified implementation-level kernel benchmark**，不是完全控制 dtype、statistics 與 runtime 後的純演算法對決。</mark>

主表中的 `T` 代表 **requested thread budget**。Local 大多數中大型 HackMD shape 都有 `active_threads == requested T`；例外主要集中於 `256³`：

| Local configuration | Requested T | 五輪 observed `active_threads` |
|---|---:|---:|
| W2 `256³` | 8 | 2 |
| W2 `256³` | 16 | 2 |
| W3 `256³` | 8 | 2 或 4 |
| W3 `256³` | 16 | 2 或 4 |
| W4 `256³` | 16 | 4 或 8 |

因此 `256³` 的高-T結果不適合直接解讀成純 4→8→16 thread scaling；其餘四個主表 shape 的 Local formal runs 均可觀察到 requested T 與 active T 對應。

---

## 2. 完整結果總表

同一列 **粗體** 表示有效 CPU backend 中最低 latency；學長 HackMD reference 與 GPU 不參與 row winner。Local T-MAC 欄全部是已重新 audit 的 `kernel_ms` 五輪 median。

| Category | Matrix (M×N×K) | T budget | MKL FP16 本機 (ms) | 學長 HackMD MKL ref (ms) | Local W2 kernel (ms) | Local W3 kernel (ms) | Local W4 kernel (ms) | Official W2 qgemm (ms) | Official W3 qgemm (ms) | Official W4 qgemm (ms) | RTX 4070S cuBLAS (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Small Square | **256×256×256** | 1 | **0.136** | 0.2365 | 0.341 | 0.531 | 0.725 | 0.311 | 0.475 | 0.603 | 0.006912 |
| Small Square | **256×256×256** | 2 | **0.128** | 0.2465 | 0.178 | 0.270 | 0.353 | 0.350 | 0.251 | 0.681 | 0.006912 |
| Small Square | **256×256×256** | 4 | **0.124** | 0.1462 | 0.213 | 0.311 | 0.180 | 0.185 | 0.130 | 0.156 | 0.006912 |
| Small Square | **256×256×256** | 8 | 0.114 | 0.3258 | 0.171 | 0.261 | 0.149 | 0.104 | **0.094** | 0.164 | 0.006912 |
| Small Square | **256×256×256** | 16 | **0.129** | 0.4081 | 0.173 | 0.265 | 0.151 | — | — | — | 0.006912 |
| Medium Square | **1024×1024×1024** | 1 | **7.219** | 16.7570 | 20.795 | 31.373 | 47.818 | 17.296 | 26.897 | 39.251 | 0.055104 |
| Medium Square | **1024×1024×1024** | 2 | 9.510 | 12.3002 | 10.838 | 15.958 | 19.396 | **8.827** | 13.796 | 19.030 | 0.055104 |
| Medium Square | **1024×1024×1024** | 4 | 7.205 | 6.4828 | 5.665 | 8.415 | 11.332 | **4.366** | 6.625 | 8.998 | 0.055104 |
| Medium Square | **1024×1024×1024** | 8 | 9.025 | 3.5757 | 6.466 | 10.330 | 10.553 | **6.004** | 8.359 | 11.984 | 0.055104 |
| Medium Square | **1024×1024×1024** | 16 | 7.304 | 5.9278 | **5.829** | 8.255 | 9.763 | — | — | — | 0.055104 |
| Large Square | **4096×4096×4096** | 1 | **331.529** | 2071.9777 | 1628.579 | 2189.468 | 2209.048 | — (timeout) | — (timeout) | — (not in suite) | 1.985536 |
| Large Square | **4096×4096×4096** | 2 | **320.147** | 1048.0653 | 689.326 | 1087.433 | 1365.033 | — (timeout) | — (timeout) | — (not in suite) | 1.985536 |
| Large Square | **4096×4096×4096** | 4 | **344.963** | 527.3677 | 348.441 | 594.978 | 786.572 | — (timeout) | — (timeout) | — (not in suite) | 1.985536 |
| Large Square | **4096×4096×4096** | 8 | 324.934 | 266.5019 | **288.348** | 482.432 | 668.103 | — (timeout) | — (timeout) | — (not in suite) | 1.985536 |
| Large Square | **4096×4096×4096** | 16 | **325.085** | 260.4321 | 404.240 | 567.622 | 628.020 | — | — | — | 1.985536 |
| Rectangular | **4096×1024×2048** | 1 | **51.362** | 260.9551 | 181.142 | 270.854 | 270.520 | 136.949 | — (fallback) | — (fallback) | 0.277504 |
| Rectangular | **4096×1024×2048** | 2 | **59.344** | 130.7603 | 87.602 | 127.550 | 170.931 | 68.567 | 105.593 | 148.135 | 0.277504 |
| Rectangular | **4096×1024×2048** | 4 | 49.866 | 65.7123 | 44.699 | 70.022 | 99.341 | **40.510** | 60.836 | 81.747 | 0.277504 |
| Rectangular | **4096×1024×2048** | 8 | 51.740 | 33.2634 | **41.370** | 66.553 | 86.023 | 43.115 | 65.414 | 75.084 | 0.277504 |
| Rectangular | **4096×1024×2048** | 16 | 50.053 | 72.7828 | **39.337** | 61.349 | 81.247 | — | — | — | 0.277504 |
| Medium Rectangular | **1024×1024×512** | 1 | **4.775** | 3.410722 | 11.573 | 16.807 | 16.916 | 9.095 | 14.137 | — (not reached) | 0.032768 |
| Medium Rectangular | **1024×1024×512** | 2 | **3.488** | 3.164299 | 5.796 | 8.488 | 10.693 | 4.600 | 7.259 | 10.599 | 0.032768 |
| Medium Rectangular | **1024×1024×512** | 4 | 3.465 | 3.135987 | **3.243** | 5.052 | 7.156 | 3.771 | 4.240 | 6.350 | 0.032768 |
| Medium Rectangular | **1024×1024×512** | 8 | 3.475 | 3.123567 | 3.384 | 5.182 | 4.684 | **3.045** | 4.562 | 4.710 | 0.032768 |
| Medium Rectangular | **1024×1024×512** | 16 | 5.069 | 3.145073 | **3.375** | 4.604 | 4.783 | — | — | — | 0.032768 |

Kernel-only matched comparison（只計有效 Official tuned result）：

- **W2：Official 13/16 較快**；Local 勝出 3 組：`256³ 2T`、`4096×1024×2048 8T`、`1024×1024×512 4T`。
- **W3：Official 15/15 較快**。
- **W4：Official 10/14 較快**；Local 勝出 4 組：`256³ 2T`、`256³ 8T`、`1024³ 8T`、`1024×1024×512 8T`。

<mark>改成 kernel-only 後，Official 的優勢比舊版 `total_ms vs qgemm` 明顯縮小；但 W3 仍呈現穩定且明顯的 Official kernel advantage。</mark>

---

## 3. 每個矩陣的最快結果與重點

| Matrix | 全部已記錄 fastest CPU | Latency | Primary 1/2/4/8T fastest CPU | Latency | GPU cuBLAS | Primary CPU / GPU |
|---|---|---:|---|---:|---:|---:|
| **256×256×256** | Official W3, budget 8 | **0.094 ms** | Official W3, budget 8 | **0.094 ms** | 0.006912 ms | 13.6× |
| **1024×1024×1024** | Official W2, budget 4 | **4.366 ms** | Official W2, budget 4 | **4.366 ms** | 0.055104 ms | 79.2× |
| **4096×4096×4096** | Local W2 kernel, budget 8 | **288.348 ms** | Local W2 kernel, budget 8 | **288.348 ms** | 1.985536 ms | 145.2× |
| **4096×1024×2048** | Local W2 kernel, budget 16 | **39.337 ms** | Official W2, budget 4 | **40.510 ms** | 0.277504 ms | 146.0× |
| **1024×1024×512** | Official W2, budget 8 | **3.045 ms** | Official W2, budget 8 | **3.045 ms** | 0.032768 ms | 92.9× |

`4096×1024×2048` 的 16T 是 8-hardware-thread CPU 上的 oversubscription reference，所以主要結論以 1/2/4/8T 為主。Primary 範圍內 Official W2 4T 為 40.510 ms，本專案 W2 8T 為 41.370 ms，兩者只差約 2.1%。

幾個最值得保留的結果：

1. <mark>**Local W2 `4096³` 8T kernel = 288.348 ms，已低於本機 MKL 最佳 320.147 ms。**</mark> 以 latency 計算約快 11%，但這仍是 low-bit T-MAC 與 dense FP16 MKL 的 implementation-level performance reference，不是等 precision 演算法對決。
2. `1024³ W2 4T`：Local kernel 5.665 ms、Official 4.366 ms，Official 約快 1.30×；舊版用 Local total 7.517 ms 時看起來約 1.72×，可見 preprocessing 曾明顯放大表面差距。
3. `4096×1024×2048 W2 8T`：Local 41.370 ms、Official 43.115 ms，Local 約快 4.2%。
4. `1024×1024×512 W2 4T`：Local 3.243 ms、Official 3.771 ms，Local 約快 16.3%。
5. W3 在目前全部 15 組 valid matched HackMD cases 都由 Official 較快，這是目前最一致的 kernel-level gap。

---

## 4. Local vs Official：差距到底來自哪裡？

不能把差距簡化成「AutoTVM 比 OpenMP 好」或「thread 寫壞」。目前程式碼與 benchmark 支持的是多個因素交互作用：

> **BM / BN / KFactor → tile 數與 cache reuse → parallel axis / active work items → runtime/thread behavior → 最終 latency**

Local 也有 tuning，而且 candidate space 與 Official 的主要 BM/BN/KFactor knobs 對齊；真正差別在於 Local 在固定手寫 AVX2/F16C/OpenMP kernel 上實測 candidate，Official 則讓 schedule 與 TVM tensorization/vectorization/parallelization/LLVM code generation 綁在一起。

目前可以較有信心地分成四點：

1. **Thread 不是唯一原因。** 例如 `1024³ W2 1T` Local kernel 20.795 ms、Official 17.296 ms，即使沒有多執行緒切分，仍有約 20% latency gap；因此 blocking、instruction scheduling、register/load-store、dtype path 等因素都可能參與。
2. **Schedule × thread partition 確實是 Local 的重要問題。** `1024³` 的 Local W2/W3 在 active 4T→8T 時都 regression；VLA 也有多個真正 `A4→A8` workload 出現退化。
3. **Official generated kernel 並非永遠更快。** Local W2 在 rectangular 8T、medium rectangular 4T，以及 W4 多個 matched cases 已能低於 Official；所以不能用「TVM assembly 固定快 X 倍」解釋。
4. **Local tuning objective 目前不是 kernel-only。** Local autotuner 以 `total_ms` 選 schedule，但本文最後比較的是 `kernel_ms`；因此被選出的 Local schedule 不一定是 kernel-only 最優。Official AutoTVM 則直接 tune qgemm measurement，這會讓 kernel-only comparison 的 tuning objective 略偏向 Official。

### Precision / data path 不同，不能再寫成 FP16 對 FP16

先前文件把 Official x86 說成 `out_dtype=float16`，這是錯誤，已在本版修正。Pinned Official Intel Linux default 為 **FP32 scale/LUT-scale/bias/output path**；CSV 的 `dtype=int8` 指的是 LUT dtype。Local 則以 FP16 activation/scale/output storage 搭配 AVX2/F16C，kernel 內再進行 FP32 SIMD conversion/accumulation。

此外 Official qgemm microbenchmark 直接建立 synthetic LUT/scale inputs；Local kernel 的 LUT 則由本專案 preprocessing 產生後再進入 kernel timing。這些差異不代表任一方數字無效，但表示：

> <mark>**目前結果是「兩套 T-MAC implementation 的 qgemm/kernel performance comparison」，不是完全等 dtype、等 numerical input 的 controlled microbenchmark。**</mark>

### Statistics 也不是相同 estimator

Local：每個 configuration 先做多 sample median，正式 HackMD 再對 5 個 complete runs 取 outer median。Official Intel Linux `time_evaluator` 使用 `number=10, repeat=10`，最後回傳 10 個 repeat-average 中的 `min`。

因此 Official 更接近「best stable repeat」，Local 更接近「typical median」。這個差異可能讓 Official latency 略偏低，因此不應把很小的百分比差異過度解讀；例如 W4 `1024×1024×512 8T` Local 4.684 ms vs Official 4.710 ms 的差距只有約 0.5%，應視為非常接近，而不是強結論。

---

## 5. VLA scaling：改用 Local kernel time

VLA Local 欄同樣改用原始 standalone result 中直接保存的 `kernel_ms`。這批 Local VLA 是單次 benchmark invocation 的 internal median，**不是 HackMD 主表那種 5-run outer median**。Official VLA 已完成 AutoTVM tuning audit：W2 231 records、W4 252 records，合計 483 records，0 failed。

Pi0 Action Expert 因 Official BN/schedule 限制使用 **N=32 proxy**，不是原始 N=33 exact shape。Local cell 中 `A#` 表示該次實際 `active_threads`。

### W2

| Workload | Matrix (M×N×K) | Local 1T kernel | Official 1T | Local 4T kernel | Official 4T | Local 8T kernel | Official 8T |
|---|---:|---:|---:|---:|---:|---:|---:|
| Action Residual | **2560×8×2560** | 0.971 (A1) | **0.819** | 0.673 (A4) | **0.217** | 0.560 (A5) | **0.211** |
| Action FC1 | **2560×8×17920** | 8.993 (A1) | **6.187** | 11.103 (A4) | **1.485** | 12.654 (A8) | **1.519** |
| Vision Attention | **1152×256×1152** | 6.370 (A1) | **5.470** | 1.604 (A4) | **1.384** | 9.016 (A8) | **1.388** |
| Pi0 Expert Q proxy | **2048×32×1024** | 1.219 (A1) | **1.077** | 0.816 (A4) | **0.280** | 0.801 (A4) | **0.288** |
| Pi0 Expert KV proxy | **256×32×1024** | 0.155 (A1) | **0.136** | **0.080 (A2)** | 0.133 | **0.081 (A2)** | 0.133 |
| Pi0 Expert O proxy | **1024×32×2048** | 1.179 (A1) | **1.057** | 0.615 (A2) | **0.281** | 0.620 (A2) | **0.281** |
| Pi0 Expert GateUp proxy | **4096×32×1024** | 3.027 (A1) | **2.184** | 1.707 (A4) | **0.558** | 5.316 (A8) | **0.578** |
| Pi0 Expert Down proxy | **1024×32×4096** | 2.508 (A1) | **2.064** | 0.609 (A4) | **0.545** | 1.291 (A2) | **0.541** |

Official W2 在 **22/24** 組 matched kernel comparison 中較快；Local 兩個勝出都是 Pi0 Expert KV proxy 的 4T/8T budget。因為不少 8T budget 並沒有真的使用 8 active workers，所以不能再用「6/8 workload 4T→8T regression」直接當純 thread-scaling 證據。若只看 Local `active_threads` 真正從 A4→A8 的 W2 workload，目前可清楚比較的是 FC1、Vision Attention、GateUp，三者皆 regression。

### W4

| Workload | Matrix (M×N×K) | Local 1T kernel | Official 1T | Local 4T kernel | Official 4T | Local 8T kernel | Official 8T |
|---|---:|---:|---:|---:|---:|---:|---:|
| Action Residual | **2560×8×2560** | 2.073 (A1) | **1.623** | 0.563 (A4) | **0.411** | 5.767 (A8) | **0.413** |
| Action FC1 | **2560×8×17920** | 16.397 (A1) | **11.731** | 10.301 (A4) | **3.000** | 13.983 (A8) | **3.053** |
| Vision Attention | **1152×256×1152** | 13.638 (A1) | **11.006** | 8.188 (A4) | **2.788** | 4.850 (A8) | **2.803** |
| Pi0 Expert Q proxy | **2048×32×1024** | 2.429 (A1) | **2.145** | 0.752 (A4) | **0.558** | 5.434 (A8) | **0.573** |
| Pi0 Expert KV proxy | **256×32×1024** | 0.311 (A1) | **0.267** | **0.080 (A4)** | 0.267 | **0.082 (A4)** | 0.142 |
| Pi0 Expert O proxy | **1024×32×2048** | 2.374 (A1) | **2.092** | **0.608 (A4)** | 1.056 | 1.588 (A4) | **0.548** |
| Pi0 Expert GateUp proxy | **4096×32×1024** | 6.046 (A1) | **4.315** | 1.287 (A4) | **1.109** | 2.999 (A8) | **2.282** |
| Pi0 Expert Down proxy | **1024×32×4096** | 4.956 (A1) | **4.306** | 1.285 (A4) | **1.068** | 9.214 (A8) | **1.074** |

Official W4 在 **21/24** 組 matched kernel comparison 中較快；Local 勝出 Pi0 KV proxy 4T/8T 與 Pi0 O proxy 4T。若只看真正 A4→A8 的 W4 workload，共 6 組，其中 Residual、FC1、Q、GateUp、Down 五組 regression，只有 Vision Attention 改善。<mark>因此 high-active-thread regression 仍是真實 kernel 現象，但必須排除 requested T 與 active T 不一致的 case 後再下結論。</mark>

---

## 6. Official fallback / timeout 與 correctness 狀態

目前正式採用規則如下：

1. **W2 / W3 `4096×4096×4096`：不提供 tuned latency。** 1/2/4/8T 都無法在 Official default measurement 設定下取得有效 tuned configuration，candidate measurement 觸發預設 10 秒 RPC session timeout，之後進入 fallback；這不代表沒有合法數學 schedule，而是 default tuning measurement 無法在時限內完成。
2. **W3 / W4 `4096×1024×2048` 1T：fallback，排除。**
3. **W2 `4096×1024×2048` 1T：136.949 ms，保留。** 最新 raw fallback audit 沒有命中此 workload，且保存有正式 qgemm result。
4. **W4 `1024×1024×512` 1T：not reached。** W4 1T invocation 在前一個 rectangular fallback case 被終止後沒有跑到此 shape。
5. **W4 `4096³`：safe suite 未包含，沒有結果。**
6. Official qgemm correctness verifier 在 NMSE 超標時只 warning、不一定中止；因此本次另外掃描保存的 raw logs。结果為 **未偵測到 `NMSE / tvm_arrays not close` warning**，目前採用結果沒有發現此類 correctness anomaly。

---

## 7. 結論

1. <mark>**Local HackMD kernel timing 現在已完成完整來源閉環。**</mark> 15 個 Intel T-MAC logs、每 configuration 5 runs、直接讀取 `kernel_ms`、checksum 五輪一致，重新聚合得到 `LOCAL AUDIT: PASS`；因此主表 Local kernel 數字可視為正式 verified dataset。
2. **修正 timing boundary 後，Official 並非全面勝出。** W2 有 13/16、W3 15/15、W4 10/14 組 valid matched cases 由 Official 較快；Local 已在多個 W2/W4 configuration 取得更低 kernel latency。
3. **W3 是目前最一致的 kernel-level gap。** 所有 15 組 valid HackMD matched cases 都由 Official W3 較快，值得後續優先看 generated assembly、register usage 與 schedule。
4. **Local W2 大型 workload 已具有競爭力。** `4096³` 8T kernel 288.348 ms 低於本機 MKL 最佳 320.147 ms；`4096×1024×2048` 8T 41.370 ms 也略低於同 budget Official W2 43.115 ms。
5. **不能再把 high-thread 問題只看 requested T。** Local 的小 shape / VLA 有不少 `requested T > active_threads`；真正 active 4→8 的 workload 仍可觀察到多個 regression，因此問題存在，但應描述為 **schedule × active-thread work partition**，而不是單純「8T 一定比較差」。
6. **Local 與 Official 不是完全等 dtype / 等統計 / 等 runtime。** Local 為 FP16 storage/output path、median-based statistics、OpenMP，Official x86 qgemm 為 FP32 scale/output path、best-repeat estimator、TVM threadpool；所以本文不把小幅差距解讀成絕對勝負。
7. **下一步若要找真正原因，不需要再重跑整套 benchmark。** 最有價值的是固定相同 `BM / BN / KFactor` 做 1T controlled comparison，再固定 schedule 做 1/2/4/8 active-thread comparison；最後看 assembly、cache miss、memory bandwidth、register spill 與 thread utilization，才能把 codegen、tiling 與 parallelization 各自的影響拆開。
