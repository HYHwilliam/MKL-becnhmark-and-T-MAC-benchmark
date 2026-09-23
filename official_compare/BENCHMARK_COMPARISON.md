# Benchmark Comparison

> **Matrix notation：M×N×K。Latency 越低越好。**

這份文件整理本專案 T-MAC、Microsoft Official T-MAC、oneMKL 與 GPU cuBLAS 的 benchmark 結果。

名稱統一如下：

- **本專案 T-MAC**：本 repository 自行實作的 x86 AVX2/F16C/OpenMP T-MAC。
- **Microsoft Official W2 / W3 / W4**：pinned Microsoft T-MAC + TVM generated x86 kernel，使用 AutoTVM tuning。
- **MKL FP16（本機）**：Intel Core Ultra 7 258V 上實際執行的 oneMKL dense FP16 GEMM。
- **學長 HackMD MKL reference**：原 HackMD 提供的 MKL reference，與本機 MKL 分開解讀。
- **RTX 4070 SUPER cuBLAS**：GPU dense FP16 GEMM reference，不是 GPU T-MAC。

主表規則：

- 同一列最快的有效 CPU latency 以 **粗體** 標示；只比較本機 MKL、本專案 T-MAC 與 Microsoft Official。
- 學長 HackMD MKL 來自不同環境，只作歷史 reference，不參與 winner 判定。
- Official `4096×4096×4096` 的 W2 / W3 在 1 / 2 / 4 / 8T 都因 AutoTVM measurement 觸發預設 10 秒 RPC session timeout，之後進入 fallback，因此沒有可信 tuned latency；W4 `hackmd_safe` 不包含此 shape。
- Official `4096×1024×2048` 的 W2 / W3 / W4 1T 均為 fallback，不納入正式比較。W4 `1024×1024×512` 1T 因前一個 fallback case 被手動終止而未執行，標示為 `not reached`。
- Official W2 / W3 / W4 整理 1 / 2 / 4 / 8T；本專案 16T 在 8 hardware-thread CPU 上屬 oversubscription reference。
- 本專案與 Official 的 timing boundary 並不完全相同：本專案 `total_ms` 包含 activation/LUT preprocessing + kernel；本次 Official adapter 的 `qgemm_lut` latency 不包含獨立的 LUT preprocessor。第 4 節會進一步說明。

## 1. 完整結果總表

| Category | Matrix (M×N×K) | CPU Threads | MKL FP16 本機 258V (ms) | 學長 HackMD MKL ref (ms) | 本專案 T-MAC W2A16 (ms) | 本專案 T-MAC W3A16 (ms) | 本專案 T-MAC W4A16 (ms) | Microsoft Official W2 (ms) | Microsoft Official W3 (ms) | Microsoft Official W4 (ms) | RTX 4070 SUPER cuBLAS (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Small Square | **256×256×256** | 1 | **0.136** | 0.2365 | 0.459 | 0.651 | 0.854 | 0.311 | 0.475 | 0.603 | 0.006912 |
| Small Square | **256×256×256** | 2 | **0.128** | 0.2465 | 0.300 | 0.395 | 0.485 | 0.350 | 0.251 | 0.681 | 0.006912 |
| Small Square | **256×256×256** | 4 | **0.124** | 0.1462 | 0.341 | 0.449 | 0.304 | 0.185 | 0.130 | 0.156 | 0.006912 |
| Small Square | **256×256×256** | 8 | 0.114 | 0.3258 | 0.288 | 0.380 | 0.279 | 0.104 | **0.094** | 0.164 | 0.006912 |
| Small Square | **256×256×256** | 16 | **0.129** | 0.4081 | 0.289 | 0.380 | 0.282 | — | — | — | 0.006912 |
| Medium Square | **1024×1024×1024** | 1 | **7.219** | 16.7570 | 22.822 | 33.350 | 49.687 | 17.296 | 26.897 | 39.251 | 0.055 |
| Medium Square | **1024×1024×1024** | 2 | 9.510 | 12.3002 | 13.000 | 17.684 | 21.069 | **8.827** | 13.796 | 19.030 | 0.055 |
| Medium Square | **1024×1024×1024** | 4 | 7.205 | 6.4828 | 7.517 | 10.349 | 13.361 | **4.366** | 6.625 | 8.998 | 0.055 |
| Medium Square | **1024×1024×1024** | 8 | 9.025 | 3.5757 | 8.867 | 13.068 | 13.124 | **6.004** | 8.359 | 11.984 | 0.055 |
| Medium Square | **1024×1024×1024** | 16 | **7.304** | 5.9278 | 7.772 | 10.281 | 12.314 | — | — | — | 0.055 |
| Large Square | **4096×4096×4096** | 1 | **331.529** | 2071.9777 | 1664.725 | 2262.285 | 2230.488 | — (timeout) | — (timeout) | — (not in suite) | 1.986 |
| Large Square | **4096×4096×4096** | 2 | **320.147** | 1048.0653 | 735.940 | 1119.599 | 1387.306 | — (timeout) | — (timeout) | — (not in suite) | 1.986 |
| Large Square | **4096×4096×4096** | 4 | **344.963** | 527.3677 | 381.560 | 631.803 | 826.431 | — (timeout) | — (timeout) | — (not in suite) | 1.986 |
| Large Square | **4096×4096×4096** | 8 | 324.934 | 266.5019 | **320.574** | 529.392 | 712.419 | — (timeout) | — (timeout) | — (not in suite) | 1.986 |
| Large Square | **4096×4096×4096** | 16 | **325.085** | 260.4321 | 435.729 | 599.435 | 671.333 | — | — | — | 1.986 |
| Rectangular | **4096×1024×2048** | 1 | **51.362** | 260.9551 | 184.947 | 275.047 | 273.532 | — (fallback) | — (fallback) | — (fallback) | 0.278 |
| Rectangular | **4096×1024×2048** | 2 | **59.344** | 130.7603 | 91.515 | 131.710 | 173.782 | 68.567 | 105.593 | 148.135 | 0.278 |
| Rectangular | **4096×1024×2048** | 4 | 49.866 | 65.7123 | 49.083 | 73.907 | 104.054 | **40.510** | 60.836 | 81.747 | 0.278 |
| Rectangular | **4096×1024×2048** | 8 | 51.740 | 33.2634 | 45.987 | 73.204 | 91.422 | **43.115** | 65.414 | 75.084 | 0.278 |
| Rectangular | **4096×1024×2048** | 16 | 50.053 | 72.7828 | **43.478** | 66.658 | 86.544 | — | — | — | 0.278 |
| Medium Rectangular | **1024×1024×512** | 1 | **4.775** | 3.410722 | 12.618 | 17.832 | 17.649 | 9.095 | 14.137 | — (not reached) | 0.033 |
| Medium Rectangular | **1024×1024×512** | 2 | **3.488** | 3.164299 | 6.804 | 9.471 | 11.632 | 4.600 | 7.259 | 10.599 | 0.033 |
| Medium Rectangular | **1024×1024×512** | 4 | **3.465** | 3.135987 | 4.185 | 5.902 | 8.206 | 3.771 | 4.240 | 6.350 | 0.033 |
| Medium Rectangular | **1024×1024×512** | 8 | 3.475 | 3.123567 | 4.609 | 6.237 | 5.797 | **3.045** | 4.562 | 4.710 | 0.033 |
| Medium Rectangular | **1024×1024×512** | 16 | 5.069 | 3.145073 | **4.488** | 5.693 | 5.976 | — | — | — | 0.033 |

Official W4 在可直接對照的 14 組 HackMD configuration 中有 **13/14 組比本專案 W4 更快**；唯一例外是 `256³` 2T。本次結果也再次顯示 Official 本身並非 thread 越多越快：`256³`、`1024³` 都在 4T 最佳，而兩個 rectangular workload 則持續改善到 8T。

---

## 2. 每個矩陣的最快結果

| Matrix (M×N×K) | 最快 CPU backend | Threads | CPU latency | GPU cuBLAS latency | GPU / CPU latency ratio |
|---|---|---:|---:|---:|---:|
| **256×256×256** | Microsoft Official W3 | 8 | **0.094 ms** | 0.006912 ms | 13.6× |
| **1024×1024×1024** | Microsoft Official W2 | 4 | **4.366 ms** | 0.055 ms | 79.2× |
| **4096×4096×4096** | MKL FP16 | 2 | **320.147 ms** | 1.986 ms | 161.2× |
| **4096×1024×2048** | Microsoft Official W2 | 4 | **40.510 ms** | 0.278 ms | 146.0× |
| **1024×1024×512** | Microsoft Official W2 | 8 | **3.045 ms** | 0.033 ms | 92.9× |

結果不存在單一規律：W2 通常具有較低 latency，但 `256³` 的 Official W3 8T 反而最快；`4096³` 中本專案 W2 已能接近 MKL；不同 shape 的最佳 thread 數也不同。因此效能不能只用 bit-width 或矩陣大小解釋，**matrix shape、schedule 與 thread configuration 都是主要變因。**

### HackMD MKL reference provenance

`MKL FP16 本機 258V` 來自 `mkl_fp16_hackmd_benchmark.cpp` 五輪實測；`學長 HackMD MKL ref` 則由 `summarize_results.py` 中保存的 `HACKMD_MKL_REFERENCE` 讀入。只有 `1024×1024×512` 的 HackMD reference 明確確認為 FP16，其餘 shape 因原表沒有明確 dtype，只作歷史 cross-machine reference。

---

## 3. CPU 結果重點

| Matrix | 主要結果 | 重點 |
|---|---|---|
| **256×256×256** | Official W3 8T：**0.094 ms** | 全體最佳；Official W4 則在 4T 最佳 0.156 ms |
| **1024×1024×1024** | Official W2 4T：**4.366 ms** | Official W2 / W3 / W4 都在 4T 最佳，8T 均出現 regression |
| **4096×4096×4096** | MKL 2T：**320.147 ms**；本專案 W2 8T：320.574 ms | Official W2 / W3 全 thread tuning timeout；W4 safe suite 未包含此 shape |
| **4096×1024×2048** | Official W2 4T：**40.510 ms** | W2 / W3 最佳在 4T；W4 最佳在 8T 75.084 ms；三者 1T 均 fallback |
| **1024×1024×512** | Official W2 8T：**3.045 ms** | W2 / W4 在 8T 最佳，W3 在 4T 最佳 |

整體來看，**T-MAC 並不是矩陣越大就越差，也不是 thread 越多就越快。** `4096³` 顯示本專案可以逼近 MKL；Official W2 / W3 / W4 也會依 shape 在 4T 或 8T 達到最佳。真正需要分析的是 workload-specific schedule、thread partition 與 kernel implementation，而不是單純矩陣大小。

---

## 4. 為什麼本專案比 Official 慢？

先排除一個容易誤判的方向：**不是「本專案用 FP16、Official 用 FP32」造成目前差距。** 兩邊都屬 W2/W3/W4A16 路徑，activation / scale / output 都以 16-bit precision 為主要介面；Official profile 明確設定 `out_dtype="float16"`，本次 `dtype="int8"` 指的是量化 LUT 的資料型態，不是把整個 GEMM 改成 A8。Official 內部使用 INT8 LUT、整數 accumulation，bit-plane 合併時也會轉成 `float32` 再乘上 alpha，最後 cast 回 FP16；本專案同樣使用 INT8 LUT，而 x86 AVX2/F16C 會把 FP16 activation / scale 載入後轉成 FP32 SIMD 做 scaling 與 accumulation，再轉回 FP16 output。因此兩邊都有 mixed-precision internal path，不能把差距解釋成「FP16 對 FP32」。

### 4.1 目前已確認的 implementation 差異

| 因素 | 本專案 | Microsoft Official | 對目前結果的意義 |
|---|---|---|---|
| Activation / output | FP16 storage，AVX2/F16C 載入後以 FP32 SIMD 做部分計算 | W*A16，`out_dtype=float16` | 不是單純 FP16 vs FP32 差異 |
| LUT path | INT8 LUT + widening accumulation | `dtype=int8` LUT + integer aggregation | 核心資料路徑方向一致 |
| Schedule tuning | 會從 official-style `BM / BN / KFactor` candidates 中實測選擇，結果有 `autotuned=1` | AutoTVM 搜尋 `BM / BN / KFactor` 並生成對應 TVM/LLVM kernel | 不是「Official 有 tune、本專案完全沒 tune」 |
| Kernel implementation | 固定的手寫 AVX2/F16C/OpenMP kernel，tuner 主要選參數 | schedule 與 TVM tensorization / vectorization / code generation 綁在一起 | 同一組 BM/BN/KF 也不代表底層指令相同 |
| Thread partition | 依 `BN/BM` 形成的 N/M tiles 決定平行軸與工作數量 | generated schedule 會把 thread configuration 一起納入 codegen/tuning | schedule 與 thread scaling 是耦合問題 |
| Timing boundary | `total_ms = activation/LUT preprocessing + kernel` | 本次只執行 `qgemm_lut`；preprocessor 是獨立 operator | Official latency 與本專案 `total_ms` 不是完全 apples-to-apples |

### 4.2 Autotune 和 thread 到底哪個才是主要問題？

答案不是二選一，因為 **schedule 會直接決定 thread 怎麼切工作**。本專案的 kernel 會先由 `BM / BN` 決定 `m_tiles`、`n_tiles`，再根據 tile 數與 requested threads 選擇沿 N 或 M 平行；因此 autotune 選到的 `BM / BN` 不只是 cache blocking 參數，也會改變 parallel axis、每個 thread 分到多少 tile、同步與 cache contention。換句話說：

> **schedule 選擇 → tile 數 / parallel axis → thread work partition → 最終 scaling**

所以看到 4T → 8T regression 時，不能只說「OpenMP/thread 寫得不好」，也不能只說「autotune 選錯參數」；更精確的描述是 **本專案的 schedule × thread partition interaction 對 workload 很敏感**。

目前 benchmark 可以把差距再拆成三層：

1. **Thread 不是全部原因。** 1T 沒有多執行緒切分問題，但 Official 在多數 1T workload 仍比本專案快，例如 `1024³ W2` 為 17.296 ms vs 22.822 ms，VLA W2/W4 的 1T 也幾乎全面由 Official 較低。這表示即使拿掉 thread scaling，timing boundary、tiling 與 generated kernel 本身仍有差距。
2. **Thread / schedule interaction 是目前最明顯的可觀察弱點。** 本專案在 VLA 中 W2 有 6/8、W4 有 7/8 workload 從 4T 增加到 8T 反而變慢；Official 多數 workload 在 4T 後接近飽和，但 regression 幅度通常小很多。HackMD matrix 也能看到相同現象。
3. **差距不是固定 kernel 倍率。** `4096×1024×2048 W2` 在 8T 時本專案 45.987 ms、Official 43.115 ms，差距只有約 1.07×；但其他小矩陣或 VLA workload 差距可放大到數倍。若只是單純「Official assembly 永遠快 X 倍」，不應出現這麼強的 shape/thread dependence。

### 4.3 Timing boundary 其實是目前比較中很重要的偏差

本專案正式 latency 使用 `total_ms`，明確包含 single-thread activation/LUT preprocessing 與 T-MAC kernel；offline weight packing 與 autotuning time不計入。Official adapter 則只呼叫 `QGeMMLUTBitsCodegen` 的 `qgemm_lut` evaluation，LUT preprocessor 在 Official framework 中是另一個 `QGeMMLUTBitsPreprocessorCodegen` operator，本次沒有一起加入 latency。

因此現在的數字比較更接近：

```text
本專案：LUT preprocessing + handwritten T-MAC kernel
Official：TVM generated qgemm_lut kernel
```

這會讓 Official 具有 timing-boundary 優勢，尤其在小矩陣中 preprocessing 的固定成本占比更高。因此目前可以用這些結果判斷「implementation-level latency gap」與 scaling 趨勢，但**不能直接把全部差距都歸因於 AutoTVM 或 generated kernel 比本專案快**。

### 4.4 現階段最合理的判斷

綜合目前證據，差距來源可分成：

1. **已確認：timing boundary 不完全相同。** 這會直接放大 Official 相對本專案的表面優勢，下一輪公平比較應優先修正。
2. **已確認：schedule 與 thread partition 強烈耦合。** 本專案 high-thread regression 比較常見，是目前最明顯的 optimization 問題。
3. **高度可能：Official generated kernel 的低階 code quality 更好。** TVM/LLVM 可以依 schedule 做 tensorization、vectorization、unrolling、register allocation 與 instruction scheduling；但這一點仍需要固定相同 schedule 後比較 assembly / hardware counters 才能定量確認。
4. **可以排除：不是單純 FP16 vs FP32。** 兩邊都是 A16 / FP16 output 的 mixed-precision LUT pipeline，只是內部 conversion、aggregation 與 code generation 方式不完全相同。

因此目前最準確的結論不是「問題在 autotune」或「問題在 thread」其中一個，而是：

> **本專案與 Official 的差距主要來自 timing boundary、workload-specific schedule 與 thread work partition 的交互作用；TVM generated kernel 的低階 code quality 可能進一步拉開差距。現有證據不支持把差距歸因於 FP16 vs FP32。**

若要把原因真正拆開，下一步應依序做三組 controlled experiment：先比較本專案 `kernel_ms` 對 Official `qgemm_lut` 以對齊 timing boundary；再固定相同 `BM / BN / KFactor` 做 1T，隔離 handwritten kernel vs TVM generated kernel；最後固定 schedule 比較 1/2/4/8T，才可以單獨判斷 thread work partition 的影響。

---

## 5. VLA scaling

VLA 使用另一組實際模型 matrix shapes，可用來確認前面的現象是否只存在於 HackMD benchmark。

Official VLA AutoTVM tuning audit：

- W2：231 records，**0 failed**
- W4：252 records，**0 failed**
- 合計：483 records，**0 failed**

### W2

| Workload | Matrix (M×N×K) | 本專案 1T | Official 1T | 本專案 4T | Official 4T | 本專案 8T | Official 8T |
|---|---:|---:|---:|---:|---:|---:|---:|
| Action Residual | **2560×8×2560** | 1.009 | **0.819** | 0.712 | **0.217** | 0.627 | **0.211** |
| Action FC1 | **2560×8×17920** | 9.242 | **6.187** | 11.368 | **1.485** | 13.520 | **1.519** |
| Vision Attention | **1152×256×1152** | 6.875 | **5.470** | 2.131 | **1.384** | 9.829 | **1.388** |
| Pi0 Expert Q | **2048×32×1024** | 1.276 | **1.077** | 0.896 | **0.280** | 0.866 | **0.288** |
| Pi0 Expert KV | **256×32×1024** | 0.213 | **0.136** | 0.140 | **0.133** | 0.141 | **0.133** |
| Pi0 Expert O | **1024×32×2048** | 1.289 | **1.057** | 0.731 | **0.281** | 0.733 | **0.281** |
| Pi0 Expert GateUp | **4096×32×1024** | 3.306 | **2.184** | 1.790 | **0.558** | 5.420 | **0.578** |
| Pi0 Expert Down | **1024×32×4096** | 2.728 | **2.064** | 0.837 | **0.545** | 1.518 | **0.541** |

Official W2 在 **24/24 組 matched comparison** 中 latency 都較低。本專案 1T → 4T 有 7/8 個 workload 改善，但 4T → 8T 有 6/8 反而變慢；Official 則主要在 1T → 4T 大幅提升，4T → 8T 多數接近飽和。

### W4

| Workload | Matrix (M×N×K) | 本專案 1T | Official 1T | 本專案 4T | Official 4T | 本專案 8T | Official 8T |
|---|---:|---:|---:|---:|---:|---:|---:|
| Action Residual | **2560×8×2560** | 2.109 | **1.623** | 0.599 | **0.411** | 5.826 | **0.413** |
| Action FC1 | **2560×8×17920** | 16.647 | **11.731** | 10.604 | **3.000** | 15.129 | **3.053** |
| Vision Attention | **1152×256×1152** | 14.152 | **11.006** | 8.765 | **2.788** | 5.662 | **2.803** |
| Pi0 Expert Q | **2048×32×1024** | 2.485 | **2.145** | 0.809 | **0.558** | 5.517 | **0.573** |
| Pi0 Expert KV | **256×32×1024** | 0.369 | **0.267** | **0.144** | 0.267 | 0.146 | **0.142** |
| Pi0 Expert O | **1024×32×2048** | 2.492 | **2.092** | **0.718** | 1.056 | 1.715 | **0.548** |
| Pi0 Expert GateUp | **4096×32×1024** | 6.102 | **4.315** | 1.343 | **1.109** | 3.083 | **2.282** |
| Pi0 Expert Down | **1024×32×4096** | 5.180 | **4.306** | 1.517 | **1.068** | 10.039 | **1.074** |

Official W4 在 **24 組 matched comparison 中有 22 組較快**。本專案雖然 1T → 4T 的 8/8 workload 全部改善，但 4T → 8T 有 7/8 退化；Official 的 high-thread scaling 整體更穩定。

因此 HackMD matrix 與 VLA workload 支持相同方向：**本專案確實有 multithreading 效果，但 high-thread-count 的 schedule / work partition interaction 是主要弱點；Official 的 workload-specific generated kernel 能更穩定地利用多執行緒。** 不過因為兩邊 timing boundary 不完全相同，這裡仍應視為 implementation-level latency comparison，而不是純演算法 speedup。

---

## 6. 結論

1. **T-MAC 效能高度依賴 shape、bit-width、thread 數與 schedule。** `4096³` 中本專案 W2 已幾乎打平 MKL，而 Official W2 / W3 / W4 的最佳 thread 數也會隨 workload 改變，因此不能用「矩陣越大越慢」或「thread 越多越快」概括。
2. **不是 Official 有 autotune、本專案沒有。** 本專案也會從 official-style schedule candidates 中實測選擇 `BM / BN / KFactor`；主要差異是 Official 的 AutoTVM schedule 與 TVM/LLVM generated kernel 綁在一起，而本專案是在固定的手寫 AVX2/OpenMP kernel 上調參。
3. **本專案最明顯的 optimization 問題是 schedule × thread partition interaction。** 4T 通常有效，但增加到 8T 後經常 regression；Official 也會出現局部 regression，但 VLA 與 HackMD 結果顯示整體更穩定。
4. **目前 latency comparison 存在 timing-boundary 差異。** 本專案 `total_ms` 包含 LUT preprocessing，Official 本次只量 `qgemm_lut`。因此在完成 timing alignment 前，不應把全部倍率解讀成 Official kernel 本身的純 speedup。
5. **差距不是 FP16 vs FP32。** 兩邊都是 A16 / FP16 output 的 mixed-precision T-MAC 路徑，內部都包含 INT8 LUT、整數 accumulation 與 FP32 conversion / combination；precision path 不是目前效能差距的主要解釋。
6. **AutoTVM 也不是所有 workload 都能成功。** `4096³` 的 Official W2 / W3 在 1 / 2 / 4 / 8T 都因預設 10 秒 RPC measurement timeout 而 fallback；`4096×1024×2048` 的 W2 / W3 / W4 1T 也沒有有效 tuned result。
7. **下一步應先做 controlled comparison，再做 assembly/profile。** 先以本專案 `kernel_ms` 對 Official `qgemm_lut` 對齊 timing boundary，再固定相同 `BM / BN / KFactor` 比較 1T，最後固定 schedule 比較 1/2/4/8T；完成這三步後，再看 generated assembly、cache miss、memory bandwidth 與 thread utilization，才能真正拆出 codegen、tiling 與 parallelization 各自的影響。
