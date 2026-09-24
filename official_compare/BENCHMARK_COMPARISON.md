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

`Threads 設定` 是 benchmark 呼叫時指定的 thread 數。`Local 實際 Threads` 顯示 W2/W3/W4 kernel 真正啟用的 active threads；格式為 **W2 / W3 / W4**。例如 `2 / 2–4 / 8` 代表同一個 Threads 設定下，W2 使用 2 threads、W3 五輪出現 2 或 4 threads、W4 使用 8 threads。

同一列 **粗體** 為有效 CPU backend 中最低 latency。HackMD reference 與 GPU 不參與 CPU winner 判定。

| Matrix (M×N×K) | Threads 設定 | Local 實際 Threads W2/W3/W4 | MKL FP16 (ms) | HackMD MKL ref (ms) | Local W2 kernel | Local W3 kernel | Local W4 kernel | Official W2 | Official W3 | Official W4 | RTX 4070S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **256×256×256** | 1 | 1 / 1 / 1 | **0.136** | 0.2365 | 0.341 | 0.531 | 0.725 | 0.311 | 0.475 | 0.603 | 0.006912 |
| **256×256×256** | 2 | 2 / 2 / 2 | **0.128** | 0.2465 | 0.178 | 0.270 | 0.353 | 0.350 | 0.251 | 0.681 | 0.006912 |
| **256×256×256** | 4 | 4 / 4 / 4 | **0.124** | 0.1462 | 0.213 | 0.311 | 0.180 | 0.185 | 0.130 | 0.156 | 0.006912 |
| **256×256×256** | 8 | 2 / 2–4 / 8 | 0.114 | 0.3258 | 0.171 | 0.261 | 0.149 | 0.104 | **0.094** | 0.164 | 0.006912 |
| **256×256×256** | 16 | 2 / 2–4 / 4–8 | **0.129** | 0.4081 | 0.173 | 0.265 | 0.151 | — | — | — | 0.006912 |
| **1024×1024×1024** | 1 | 1 / 1 / 1 | **7.219** | 16.7570 | 20.795 | 31.373 | 47.818 | 17.296 | 26.897 | 39.251 | 0.055104 |
| **1024×1024×1024** | 2 | 2 / 2 / 2 | 9.510 | 12.3002 | 10.838 | 15.958 | 19.396 | **8.827** | 13.796 | 19.030 | 0.055104 |
| **1024×1024×1024** | 4 | 4 / 4 / 4 | 7.205 | 6.4828 | 5.665 | 8.415 | 11.332 | **4.366** | 6.625 | 8.998 | 0.055104 |
| **1024×1024×1024** | 8 | 8 / 8 / 8 | 9.025 | 3.5757 | 6.466 | 10.330 | 10.553 | **6.004** | 8.359 | 11.984 | 0.055104 |
| **1024×1024×1024** | 16 | 16 / 16 / 16 | 7.304 | 5.9278 | **5.829** | 8.255 | 9.763 | — | — | — | 0.055104 |
| **4096×4096×4096** | 1 | 1 / 1 / 1 | **331.529** | 2071.9777 | 1628.579 | 2189.468 | 2209.048 | — | — | — | 1.985536 |
| **4096×4096×4096** | 2 | 2 / 2 / 2 | **320.147** | 1048.0653 | 689.326 | 1087.433 | 1365.033 | — | — | — | 1.985536 |
| **4096×4096×4096** | 4 | 4 / 4 / 4 | **344.963** | 527.3677 | 348.441 | 594.978 | 786.572 | — | — | — | 1.985536 |
| **4096×4096×4096** | 8 | 8 / 8 / 8 | 324.934 | 266.5019 | **288.348** | 482.432 | 668.103 | — | — | — | 1.985536 |
| **4096×4096×4096** | 16 | 16 / 16 / 16 | **325.085** | 260.4321 | 404.240 | 567.622 | 628.020 | — | — | — | 1.985536 |
| **4096×1024×2048** | 1 | 1 / 1 / 1 | **51.362** | 260.9551 | 181.142 | 270.854 | 270.520 | 136.949 | — (fallback) | — (fallback) | 0.277504 |
| **4096×1024×2048** | 2 | 2 / 2 / 2 | **59.344** | 130.7603 | 87.602 | 127.550 | 170.931 | 68.567 | 105.593 | 148.135 | 0.277504 |
| **4096×1024×2048** | 4 | 4 / 4 / 4 | 49.866 | 65.7123 | 44.699 | 70.022 | 99.341 | **40.510** | 60.836 | 81.747 | 0.277504 |
| **4096×1024×2048** | 8 | 8 / 8 / 8 | 51.740 | 33.2634 | **41.370** | 66.553 | 86.023 | 43.115 | 65.414 | 75.084 | 0.277504 |
| **4096×1024×2048** | 16 | 16 / 16 / 16 | 50.053 | 72.7828 | **39.337** | 61.349 | 81.247 | — | — | — | 0.277504 |
| **1024×1024×512** | 1 | 1 / 1 / 1 | **4.775** | 3.410722 | 11.573 | 16.807 | 16.916 | 9.095 | 14.137 | — | 0.032768 |
| **1024×1024×512** | 2 | 2 / 2 / 2 | **3.488** | 3.164299 | 5.796 | 8.488 | 10.693 | 4.600 | 7.259 | 10.599 | 0.032768 |
| **1024×1024×512** | 4 | 4 / 4 / 4 | 3.465 | 3.135987 | **3.243** | 5.052 | 7.156 | 3.771 | 4.240 | 6.350 | 0.032768 |
| **1024×1024×512** | 8 | 8 / 8 / 8 | 3.475 | 3.123567 | 3.384 | 5.182 | 4.684 | **3.045** | 4.562 | 4.710 | 0.032768 |
| **1024×1024×512** | 16 | 16 / 16 / 16 | 5.069 | 3.145073 | **3.375** | 4.604 | 4.783 | — | — | — | 0.032768 |

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

| Matrix | 主要 CPU 結果 | 重點 |
|---|---|---|
| **256×256×256** | Official W3, Threads 8：**0.094 ms** | Local W2/W3 在高 Threads 設定下未必實際啟用同樣數量的 workers，因此不適合作為純 thread scaling 證據 |
| **1024×1024×1024** | Official W2, Threads 4：**4.366 ms** | Local W2 kernel 5.665 ms，Official 約快 1.30× |
| **4096×4096×4096** | Local W2, Threads 8：**288.348 ms** | 低於本機 MKL 最佳 320.147 ms；Official W2/W3 無可信 tuned result |
| **4096×1024×2048** | 1/2/4/8 Threads 中 Official W2 4T：**40.510 ms** | Local W2 8T = 41.370 ms，只差約 2.1%；16T Local = 39.337 ms，但屬 oversubscription reference |
| **1024×1024×512** | Official W2, Threads 8：**3.045 ms** | Local W2 4T = 3.243 ms，低於同 Threads Official 3.771 ms |

<mark>Local W2 `4096³` 8T kernel = 288.348 ms，已低於本機 MKL FP16 最佳 320.147 ms。</mark> 這表示 low-bit LUT-based kernel 在該 workload 已具有很強的 CPU latency 競爭力，但因資料型別與計算內容不同，不應描述成完全等精度的 MKL speedup。

---

## 5. Local 與 Official 的效能差距

目前結果不支持「只有一個原因」。較合理的模型是：

```text
BM / BN / KFactor
        ↓
tile 數與 cache reuse
        ↓
實際 active threads / parallel axis
        ↓
runtime、affinity、memory behavior
        ↓
generated / handwritten machine code
        ↓
kernel latency
```

幾個可以直接從結果支持的判斷：

1. **Thread 不是唯一原因。** `1024³ W2 1T` Local 20.795 ms、Official 17.296 ms；沒有多執行緒切分時仍有約 20% gap，所以 blocking、instruction scheduling、register/load-store 與 dtype path 都可能參與。
2. **Schedule × active-thread partition 是 Local 的重要問題。** `1024³` W2/W3 都是在實際 active 4→8 時 latency 上升；VLA 中也有多個相同現象。
3. **Official generated kernel 不是固定更快。** Local W2 在 rectangular 8T、medium rectangular 4T，以及多個 W4 configurations 已能低於 Official。
4. **Local tuning objective 與最後比較指標不同。** Local autotuner 依 `total_ms` 選 schedule，但主表比較 `kernel_ms`；被選中的 Local schedule 不保證是 kernel-only 最佳。Official AutoTVM 直接對 qgemm measurement tuning。
5. **dtype 與 statistics 會影響小差距的解讀。** Official 使用 FP32 scale/output path與 best-repeat estimator；Local 使用 FP16 storage/output path與 median-based statistics。因此只有 0.x%～數%的差異不適合下強結論。

---

## 6. VLA Kernel Scaling

VLA Local 欄使用原始 standalone result 中直接保存的 `kernel_ms`。這批資料是單次 benchmark invocation 的 internal median，不是五輪 outer median。

Pi0 Action Expert 使用 **N=32 proxy**，不是 N=33 exact shape。`A#` 代表 Local 實際 active threads。

### W2

| Workload | Matrix | Local 1T | Official 1T | Local 4T | Official 4T | Local 8T | Official 8T |
|---|---:|---:|---:|---:|---:|---:|---:|
| Action Residual | **2560×8×2560** | 0.971 (A1) | **0.819** | 0.673 (A4) | **0.217** | 0.560 (A5) | **0.211** |
| Action FC1 | **2560×8×17920** | 8.993 (A1) | **6.187** | 11.103 (A4) | **1.485** | 12.654 (A8) | **1.519** |
| Vision Attention | **1152×256×1152** | 6.370 (A1) | **5.470** | 1.604 (A4) | **1.384** | 9.016 (A8) | **1.388** |
| Pi0 Q proxy | **2048×32×1024** | 1.219 (A1) | **1.077** | 0.816 (A4) | **0.280** | 0.801 (A4) | **0.288** |
| Pi0 KV proxy | **256×32×1024** | 0.155 (A1) | **0.136** | **0.080 (A2)** | 0.133 | **0.081 (A2)** | 0.133 |
| Pi0 O proxy | **1024×32×2048** | 1.179 (A1) | **1.057** | 0.615 (A2) | **0.281** | 0.620 (A2) | **0.281** |
| Pi0 GateUp proxy | **4096×32×1024** | 3.027 (A1) | **2.184** | 1.707 (A4) | **0.558** | 5.316 (A8) | **0.578** |
| Pi0 Down proxy | **1024×32×4096** | 2.508 (A1) | **2.064** | 0.609 (A4) | **0.545** | 1.291 (A2) | **0.541** |

Official W2 在 22/24 組 matched kernel comparison 中較快。若只看 Local 真正從 A4→A8 的 W2 workloads，FC1、Vision Attention、GateUp 都出現 regression；其餘 requested 8T case 若 active threads 沒有增加，不能直接視為 4→8 thread scaling。

### W4

| Workload | Matrix | Local 1T | Official 1T | Local 4T | Official 4T | Local 8T | Official 8T |
|---|---:|---:|---:|---:|---:|---:|---:|
| Action Residual | **2560×8×2560** | 2.073 (A1) | **1.623** | 0.563 (A4) | **0.411** | 5.767 (A8) | **0.413** |
| Action FC1 | **2560×8×17920** | 16.397 (A1) | **11.731** | 10.301 (A4) | **3.000** | 13.983 (A8) | **3.053** |
| Vision Attention | **1152×256×1152** | 13.638 (A1) | **11.006** | 8.188 (A4) | **2.788** | 4.850 (A8) | **2.803** |
| Pi0 Q proxy | **2048×32×1024** | 2.429 (A1) | **2.145** | 0.752 (A4) | **0.558** | 5.434 (A8) | **0.573** |
| Pi0 KV proxy | **256×32×1024** | 0.311 (A1) | **0.267** | **0.080 (A4)** | 0.267 | **0.082 (A4)** | 0.142 |
| Pi0 O proxy | **1024×32×2048** | 2.374 (A1) | **2.092** | **0.608 (A4)** | 1.056 | 1.588 (A4) | **0.548** |
| Pi0 GateUp proxy | **4096×32×1024** | 6.046 (A1) | **4.315** | 1.287 (A4) | **1.109** | 2.999 (A8) | **2.282** |
| Pi0 Down proxy | **1024×32×4096** | 4.956 (A1) | **4.306** | 1.285 (A4) | **1.068** | 9.214 (A8) | **1.074** |

Official W4 在 21/24 組 matched kernel comparison 中較快。真正 A4→A8 的 6 個 Local workloads 中，Residual、FC1、Q、GateUp、Down 五個 regression，Vision Attention 改善。

<mark>因此 high-active-thread regression 是確實存在的 kernel 現象，但必須用實際 active threads 判斷，而不能只看 benchmark 設定的 Threads 數。</mark>

---

## 7. 結論

1. **Local kernel timing 已完成資料閉環驗證。** 15 個 Intel logs、每 configuration 5 runs、直接讀 `kernel_ms`、checksum 五輪一致，`LOCAL AUDIT: PASS`。
2. **Official 採用的是 qgemm operator timing。** LUT preprocessor 不包含在 Official qgemm latency；保存的 raw logs 未偵測到 NMSE correctness warning，已知 fallback cases 不納入 tuned comparison。
3. **W3 是目前最一致的 Official 優勢。** 15/15 個 valid matched HackMD configurations 都由 Official W3 較快。
4. **W2 與 W4 已有多個 Local 較快的 configuration。** 特別是 W2 `4096×1024×2048` 8T 與 W2 `1024×1024×512` 4T。
5. **Local W2 `4096³` 8T kernel = 288.348 ms**，低於本機 MKL FP16 最佳 320.147 ms。
6. **Threads 必須搭配實際 active threads 解讀。** 中大型 HackMD shapes 大多能用滿設定 Threads；`256³` 與部分 VLA workloads 則可能因 tile 數不足而只啟用較少 workers。
7. **效能差距不能單獨歸因於 AutoTVM、thread 或 FP16/FP32。** 最合理的解釋是 workload-specific tiling、active-thread partition、runtime/affinity、dtype path 與 generated/handwritten machine code共同作用。
8. **後續最有價值的 controlled experiment** 是固定相同 BM/BN/KFactor 做 1T comparison，再固定 schedule 做 1/2/4/8 active-thread comparison；之後再分析 assembly、cache miss、memory bandwidth、register spill 與 thread utilization。
