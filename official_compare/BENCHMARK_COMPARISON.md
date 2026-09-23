


# Benchmark Comparison

> **Matrix notation：M×N×K。Latency 越低越好。**

這份文件整理本專案 T-MAC、Microsoft Official T-MAC、oneMKL 與 GPU cuBLAS 的 benchmark 結果。

名稱統一如下：

* **本專案 T-MAC**：本 repository 自行實作的 x86 AVX2/F16C/OpenMP T-MAC。

* **Microsoft Official W2 / W3 / W4**：pinned Microsoft T-MAC + TVM generated x86 kernel，使用 AutoTVM tuning。

* **MKL FP16（本機）**：Intel Core Ultra 7 258V 上實際執行的 oneMKL dense FP16 GEMM。

* **學長 HackMD MKL reference**：原 HackMD 提供的 MKL reference，與本機 MKL 分開解讀。

* **RTX 4070 SUPER cuBLAS**：GPU dense FP16 GEMM reference，不是 GPU T-MAC。

主表規則：

* 同一列最快的有效 CPU latency 以 **粗體** 標示；只比較本機 MKL、本專案 T-MAC 與 Microsoft Official。

* 學長 HackMD MKL 來自不同環境，只作歷史 reference，不參與 winner 判定。

* Official 4096×4096×4096 的 W2 / W3 在 1 / 2 / 4 / 8T 下都因 AutoTVM measurement 觸發預設 10 秒 RPC session timeout 而進入 fallback，因此不納入正式 latency comparison；W4 的 hackmd_safe suite 不包含此 shape。

* Official 4096×1024×2048 的 W2 / W3 / W4 1T 均為 fallback，不納入正式比較；W4 1024×1024×512 1T 因前一個 fallback case 被手動終止而未執行，標示為 not reached。

* Official W2 / W3 / W4 整理 1 / 2 / 4 / 8T；本專案 16T 在 8 hardware-thread CPU 上屬 oversubscription reference。

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

Official W4 在可直接對照的 14 組 HackMD configuration 中有 **13/14 組比本專案 W4 更快**；唯一例外是 256³ 2T。本次結果也顯示 Official 本身並非 thread 越多越快：256³、1024³ 都在 4T 最佳，而兩個 rectangular workload 則持續改善到 8T。

---

## 2. 每個矩陣的最快結果

| Matrix (M×N×K)     | 最快 CPU backend        | Threads |    CPU latency | GPU cuBLAS latency | GPU / CPU latency ratio |

| ------------------ | --------------------- | ------: | -------------: | -----------------: | ----------------------: |

| **256×256×256**    | Microsoft Official W3 |       8 |   **0.094 ms** |        0.006912 ms |                   13.6× |

| **1024×1024×1024** | Microsoft Official W2 |       4 |   **4.366 ms** |           0.055 ms |                   79.2× |

| **4096×4096×4096** | MKL FP16              |       2 | **320.147 ms** |           1.986 ms |                  161.2× |

| **4096×1024×2048** | Microsoft Official W2 |       4 |  **40.510 ms** |           0.278 ms |                  146.0× |

| **1024×1024×512**  | Microsoft Official W2 |       8 |   **3.045 ms** |           0.033 ms |                   92.9× |

結果並不存在單一規律：W2 通常具有較低 latency，但 `256³` 的 Official W3 8T 反而最快；`4096³` 中本專案 W2 已能接近 MKL；不同 shape 的最佳 thread 數也不同。因此效能不能只用 bit-width 或矩陣大小解釋，**matrix shape、schedule 與 thread configuration 都是主要變因。**

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

整體結果沒有單一 scaling 規律。**T-MAC 並不是矩陣越大就越差，也不是 thread 越多就越快。** 4096³ 中本專案 W2 幾乎打平 MKL；而 Official W2 / W3 / W4 也會依 shape 在 4T 或 8T 達到最佳，表示 workload-specific schedule 與 thread partition 才是主要變因。

---

## 4. 為什麼本專案比 Official 慢？

Official 與本專案使用相同 T-MAC 核心概念，但 implementation strategy 不同。從目前 W2 / W3 / W4 benchmark 可以確認，**差距不是固定倍率**：不同 shape、bit-width 與 thread configuration 的差距會明顯改變；Official W4 在 14 組可直接 matched 的 HackMD configuration 中有 13 組較快。這種變化更符合 schedule / parallelization 差異，而不是核心數學算法不同。

主要差異可以整理成：

1. **Official 會針對 workload 做 AutoTVM schedule search。** Official 不是使用一組固定參數跑所有矩陣，而會依 bit-width、matrix shape 與 thread 數搜尋 `BM / BN / KFactor` 等 configuration。本專案則是手寫 AVX2/OpenMP kernel，因此無法自動替每個 shape 找到最適 blocking。

2. **Thread work partition 是目前最明顯的差異。** 本專案經常在 4T → 8T 出現明顯 regression；Official 也不是完全沒有 regression，例如 W4 的 256³、1024³ 都在 4T 最佳，但整體退化通常較小、VLA workload 也更穩定。這表示 high-thread-count 下的工作切分、同步成本、cache contention 或 memory traffic 仍是本專案最值得檢查的方向。

3. **Blocking 會直接影響 cache locality。** T-MAC 需要反覆使用 LUT、activation 與 partial result。`BM / BN` 不合適時，資料 reuse 下降、cache miss 增加，即使 SIMD 核心算法相同，也可能造成明顯 latency 差距。Official 的 workload-specific tuning 可以降低這種風險。

4. **Generated kernel 的低階 code quality 可能更好。** TVM/LLVM 可以依 schedule 進行 loop unrolling、vectorization、register allocation 與 instruction scheduling；本專案手寫 kernel 雖然已使用 AVX2/F16C，但不代表最終 assembly 在 register usage、load/store 或 instruction ordering 上已達到相同效率。**這一點目前屬合理推論，仍需要直接比較 generated assembly 才能確認。**

5. **兩邊 benchmark harness 並非完全 apples-to-apples。** 本專案與 Official 的 dtype path、kernel generation 與 timing boundary 並不完全相同，所以目前結果適合判斷 implementation-level performance gap，但不能把所有倍率都直接解讀成單一 kernel optimization 的效果。

因此目前最有證據支持的判斷是：

> **本專案與 Official 的主要差距較可能來自 workload-specific tiling、thread work partition 與 generated kernel quality，而不是 T-MAC 的數學邏輯本身。**

---

## 5. VLA scaling

VLA 使用另一組實際模型 matrix shapes，可用來確認前面的現象是否只存在於 HackMD benchmark。

Official VLA AutoTVM tuning audit：

* W2：231 records，**0 failed**

* W4：252 records，**0 failed**

* 合計：483 records，**0 failed**

### W2

| Workload          |    Matrix (M×N×K) | 本專案 1T | Official 1T | 本專案 4T | Official 4T | 本專案 8T | Official 8T |

| ----------------- | ----------------: | -----: | ----------: | -----: | ----------: | -----: | ----------: |

| Action Residual   |   **2560×8×2560** |  1.009 |   **0.819** |  0.712 |   **0.217** |  0.627 |   **0.211** |

| Action FC1        |  **2560×8×17920** |  9.242 |   **6.187** | 11.368 |   **1.485** | 13.520 |   **1.519** |

| Vision Attention  | **1152×256×1152** |  6.875 |   **5.470** |  2.131 |   **1.384** |  9.829 |   **1.388** |

| Pi0 Expert Q      |  **2048×32×1024** |  1.276 |   **1.077** |  0.896 |   **0.280** |  0.866 |   **0.288** |

| Pi0 Expert KV     |   **256×32×1024** |  0.213 |   **0.136** |  0.140 |   **0.133** |  0.141 |   **0.133** |

| Pi0 Expert O      |  **1024×32×2048** |  1.289 |   **1.057** |  0.731 |   **0.281** |  0.733 |   **0.281** |

| Pi0 Expert GateUp |  **4096×32×1024** |  3.306 |   **2.184** |  1.790 |   **0.558** |  5.420 |   **0.578** |

| Pi0 Expert Down   |  **1024×32×4096** |  2.728 |   **2.064** |  0.837 |   **0.545** |  1.518 |   **0.541** |

Official W2 在 **24/24 組 matched comparison** 中 latency 都較低。本專案 1T → 4T 有 7/8 個 workload 改善，但 4T → 8T 有 6/8 反而變慢；Official 則主要在 1T → 4T 大幅提升，4T → 8T 多數接近飽和。

### W4

| Workload          |    Matrix (M×N×K) | 本專案 1T | Official 1T |    本專案 4T | Official 4T | 本專案 8T | Official 8T |

| ----------------- | ----------------: | -----: | ----------: | --------: | ----------: | -----: | ----------: |

| Action Residual   |   **2560×8×2560** |  2.109 |   **1.623** |     0.599 |   **0.411** |  5.826 |   **0.413** |

| Action FC1        |  **2560×8×17920** | 16.647 |  **11.731** |    10.604 |   **3.000** | 15.129 |   **3.053** |

| Vision Attention  | **1152×256×1152** | 14.152 |  **11.006** |     8.765 |   **2.788** |  5.662 |   **2.803** |

| Pi0 Expert Q      |  **2048×32×1024** |  2.485 |   **2.145** |     0.809 |   **0.558** |  5.517 |   **0.573** |

| Pi0 Expert KV     |   **256×32×1024** |  0.369 |   **0.267** | **0.144** |       0.267 |  0.146 |   **0.142** |

| Pi0 Expert O      |  **1024×32×2048** |  2.492 |   **2.092** | **0.718** |       1.056 |  1.715 |   **0.548** |

| Pi0 Expert GateUp |  **4096×32×1024** |  6.102 |   **4.315** |     1.343 |   **1.109** |  3.083 |   **2.282** |

| Pi0 Expert Down   |  **1024×32×4096** |  5.180 |   **4.306** |     1.517 |   **1.068** | 10.039 |   **1.074** |

Official W4 在 **24 組 matched comparison 中有 22 組較快**。本專案雖然 1T → 4T 的 8/8 workload 全部改善，但 4T → 8T 有 7/8 退化；Official 的 high-thread scaling 明顯穩定許多。

因此 HackMD matrix 與 VLA workload 支持相同結論：**本專案確實有 multithreading 效果，但 high-thread-count 的 schedule / work partition 是主要弱點；Official 的 workload-specific generated kernel 能更穩定地利用多執行緒。**

> 本專案與 Official 的 dtype / timing harness 並非完全相同，因此此處屬 implementation-level latency comparison，不直接解讀成純演算法 speedup。

---

## 6. 結論

1. **T-MAC 效能高度依賴 shape、bit-width、thread 數與 schedule。** 4096³ 中本專案 W2 已幾乎打平 MKL，而 Official W2 / W3 / W4 的最佳 thread 數也會隨 workload 改變，因此不能用「矩陣越大越慢」或「thread 越多越快」概括。

2. **Official 的主要優勢仍是 workload-specific optimization。** AutoTVM 會依 workload 搜尋 BM / BN / KFactor 等 configuration；HackMD matrix 中 Official W4 在 14 組可 matched configuration 有 13 組較快，VLA W2 / W4 也維持明顯優勢。

3. **本專案最明顯的問題是 high-thread scaling。** 4T 通常有效，但增加到 8T 後經常 regression；Official 也會出現局部 regression，但整體較穩定，尤其 VLA workload 更明顯。

4. **AutoTVM 本身也受 benchmark configuration 限制。** 4096³ 的 Official W2 / W3 在 1 / 2 / 4 / 8T 都因預設 10 秒 RPC session timeout 而進入 fallback；4096×1024×2048 的 W2 / W3 / W4 1T 也 fallback，因此這些結果不納入 tuned latency comparison。

5. **下一步應直接比較本專案與 Official 的 BM / BN / KFactor、thread work partition 與 generated assembly。** 必要時再加入 cache / memory profiling，才能進一步確認 latency gap 來自 tiling、parallelization 還是低階 code generation。