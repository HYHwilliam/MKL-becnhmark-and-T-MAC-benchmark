# Benchmark Comparison

> **Matrix notation：M×N×K。Latency 越低越好。**

這份文件把目前正式保存的 CPU / GPU benchmark 結果集中到一張主表。

名稱統一如下：

- **本專案 T-MAC**：本 repository 自行實作的 x86 AVX2/F16C/OpenMP T-MAC。
- **Microsoft Official W2**：pinned Microsoft T-MAC + TVM generated x86 kernel。
- **MKL FP16**：CPU dense FP16 GEMM reference。
- **RTX 4070 SUPER cuBLAS**：GPU dense FP16 GEMM reference，不是 GPU T-MAC。

主表中：

- **CPU backend 同一列最快的 latency 會加粗。**
- GPU 沒有 CPU thread 數，因此同一 matrix 的 cuBLAS latency 會重複顯示，只是方便直接比較。
- Official 目前沒有可信的 `4096×4096×4096` 結果，因此保留 `—`。
- Official 目前只整理 1 / 2 / 4 / 8T，因此 16T 顯示 `—`。
- 16T 保留原 HackMD sweep 結果，但屬高 thread-count / oversubscription 參考，不當作主要結論。

## 1. 完整結果總表

| Category | Matrix (M×N×K) | CPU Threads | MKL FP16 (ms) | 本專案 T-MAC W2A16 (ms) | 本專案 T-MAC W3A16 (ms) | 本專案 T-MAC W4A16 (ms) | Microsoft Official W2 (ms) | RTX 4070 SUPER cuBLAS (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small Square | **256×256×256** | 1 | **0.136** | 0.459 | 0.651 | 0.854 | 0.311 | 0.006912 |
| Small Square | **256×256×256** | 2 | **0.128** | 0.300 | 0.395 | 0.485 | 0.350 | 0.006912 |
| Small Square | **256×256×256** | 4 | **0.124** | 0.341 | 0.449 | 0.304 | 0.185 | 0.006912 |
| Small Square | **256×256×256** | 8 | 0.114 | 0.288 | 0.380 | 0.279 | **0.104** | 0.006912 |
| Small Square | **256×256×256** | 16 | **0.129** | 0.289 | 0.380 | 0.282 | — | 0.006912 |
| Medium Square | **1024×1024×1024** | 1 | **7.219** | 22.822 | 33.350 | 49.687 | 17.296 | 0.055 |
| Medium Square | **1024×1024×1024** | 2 | 9.510 | 13.000 | 17.684 | 21.069 | **8.827** | 0.055 |
| Medium Square | **1024×1024×1024** | 4 | 7.205 | 7.517 | 10.349 | 13.361 | **4.366** | 0.055 |
| Medium Square | **1024×1024×1024** | 8 | 9.025 | 8.867 | 13.068 | 13.124 | **6.004** | 0.055 |
| Medium Square | **1024×1024×1024** | 16 | **7.304** | 7.772 | 10.281 | 12.314 | — | 0.055 |
| Large Square | **4096×4096×4096** | 1 | **331.529** | 1664.725 | 2262.285 | 2230.488 | — | 1.986 |
| Large Square | **4096×4096×4096** | 2 | **320.147** | 735.940 | 1119.599 | 1387.306 | — | 1.986 |
| Large Square | **4096×4096×4096** | 4 | **344.963** | 381.560 | 631.803 | 826.431 | — | 1.986 |
| Large Square | **4096×4096×4096** | 8 | 324.934 | **320.574** | 529.392 | 712.419 | — | 1.986 |
| Large Square | **4096×4096×4096** | 16 | **325.085** | 435.729 | 599.435 | 671.333 | — | 1.986 |
| Rectangular | **4096×1024×2048** | 1 | **51.362** | 184.947 | 275.047 | 273.532 | 136.949 | 0.278 |
| Rectangular | **4096×1024×2048** | 2 | **59.344** | 91.515 | 131.710 | 173.782 | 68.567 | 0.278 |
| Rectangular | **4096×1024×2048** | 4 | 49.866 | 49.083 | 73.907 | 104.054 | **40.510** | 0.278 |
| Rectangular | **4096×1024×2048** | 8 | 51.740 | 45.987 | 73.204 | 91.422 | **43.115** | 0.278 |
| Rectangular | **4096×1024×2048** | 16 | 50.053 | **43.478** | 66.658 | 86.544 | — | 0.278 |
| Medium Rectangular | **1024×1024×512** | 1 | **4.775** | 12.618 | 17.832 | 17.649 | 9.095 | 0.033 |
| Medium Rectangular | **1024×1024×512** | 2 | **3.488** | 6.804 | 9.471 | 11.632 | 4.600 | 0.033 |
| Medium Rectangular | **1024×1024×512** | 4 | **3.465** | 4.185 | 5.902 | 8.206 | 3.771 | 0.033 |
| Medium Rectangular | **1024×1024×512** | 8 | 3.475 | 4.609 | 6.237 | 5.797 | **3.045** | 0.033 |
| Medium Rectangular | **1024×1024×512** | 16 | 5.069 | **4.488** | 5.693 | 5.976 | — | 0.033 |

這張表就是目前最主要的結果。不要先看後面的分析，光看粗體就能知道每一組 CPU configuration 誰最快。

---

## 2. 每個矩陣的最快結果

| Matrix (M×N×K) | 最快 CPU backend | Threads | CPU latency | GPU cuBLAS latency | GPU / 最快 CPU latency speedup |
|---|---|---:|---:|---:|---:|
| **256×256×256** | **Microsoft Official W2** | 8 | **0.104 ms** | 0.006912 ms | **15.1×** |
| **1024×1024×1024** | **Microsoft Official W2** | 4 | **4.366 ms** | 0.055 ms | **79.2×** |
| **4096×4096×4096** | **MKL FP16** | 2 | **320.147 ms** | 1.986 ms | **161.2×** |
| **4096×1024×2048** | **Microsoft Official W2** | 4 | **40.510 ms** | 0.278 ms | **146.0×** |
| **1024×1024×512** | **Microsoft Official W2** | 8 | **3.045 ms** | 0.033 ms | **92.9×** |

這裡只做一層濃縮：

- **256×256×256**：CPU 最快是 Microsoft Official W2 8T，0.104 ms。
- **1024×1024×1024**：CPU 最快是 Microsoft Official W2 4T，4.366 ms。
- **4096×4096×4096**：目前有效結果中 MKL 2T 最快，320.147 ms；Official 尚無可信結果。
- **4096×1024×2048**：CPU 最快是 Microsoft Official W2 4T，40.510 ms。
- **1024×1024×512**：CPU 最快是 Microsoft Official W2 8T，3.045 ms。

所以現在不能簡化成「T-MAC 一定比 MKL 快」或「矩陣越大 T-MAC 越差」。

更準確的是：**結果高度依賴 matrix shape、bit-width、thread 數與 schedule。**

---

## 3. GPU reference

| Matrix (M×N×K) | cuBLAS latency | cuBLAS GFLOPS |
|---|---:|---:|
| **256×256×256** | 0.006912 ms | 4,854.5 |
| **1024×1024×1024** | 0.055 ms | 38,971.5 |
| **4096×4096×4096** | 1.986 ms | 69,220.1 |
| **4096×1024×2048** | 0.278 ms | 61,908.5 |
| **1024×1024×512** | 0.033 ms | 32,768.0 |

GPU 數字非常快，但要分開解讀：

- GPU：RTX 4070 SUPER
- cuBLAS dense FP16
- FP32 accumulation
- CUDA Event 只量 warmed-up GEMM
- H2D / D2H 不計時

所以 GPU 欄位的用途是給讀者一個 **跨硬體性能上限 reference**，不能把倍率直接解讀成 T-MAC 演算法和 cuBLAS 演算法的公平比較。

---

## 4. 最值得注意的 CPU 結果

### 256×256×256

小矩陣時 MKL 很有競爭力：

- 1T / 2T / 4T 都是 MKL 最快。
- 8T 時 Microsoft Official W2 以 **0.104 ms** 超過 MKL 的 0.114 ms。
- 本專案 T-MAC 在這個小 shape 明顯吃到固定 overhead。

### 1024×1024×1024

Official 優勢開始明顯：

- 4T：Official **4.366 ms**
- MKL：7.205 ms
- 本專案 W2：7.517 ms

本專案 W2 已經接近 MKL，但 official generated kernel 還有明顯差距。

### 4096×4096×4096

目前沒有可信 official result。

已有效的 CPU 結果中：

- MKL 2T：**320.147 ms**
- 本專案 W2 8T：320.574 ms

兩者幾乎打平。

因此「矩陣變大後本專案 T-MAC 一定更差」並不成立。

### 4096×1024×2048

這是本專案 T-MAC 表現相對好的大型 rectangular case：

- Official W2 4T：**40.510 ms**
- 本專案 W2 16T：43.478 ms
- 本專案 W2 8T：45.987 ms
- MKL 4T：49.866 ms

即使只看主要的 1/2/4/8T，本專案 W2 8T 也已經比 MKL 4T 快。

### 1024×1024×512

- Official W2 8T：**3.045 ms**
- MKL 4T：3.465 ms
- 本專案 W2 4T：4.185 ms

這裡 official 比本專案 W2 的 thread scaling / schedule 更有效。

---

## 5. 本專案 T-MAC 的真正問題：不是「大矩陣」，而是 thread scaling

之前容易把現象描述成：

> 矩陣越大，T-MAC 效能越差。

從現在完整結果來看，這個說法太粗。

更合理的描述是：

> **本專案 T-MAC 在部分 shape 可以接近甚至超過 MKL，但多執行緒 scaling 對 shape 與 schedule 很敏感。**

例如：

- `4096×4096×4096`：本專案 W2 8T 幾乎打平 MKL。
- `4096×1024×2048`：本專案 W2 8T / 16T 已經快過 MKL。
- 但其他 shape 增加 thread 後可能反而退化。

這和 VLA 測試看到的結果一致：**4T 通常有效，8T 常出現 regression。**

---

## 6. VLA scaling 補充

VLA 使用另一組 matrix shapes，因此不混進上面的 HackMD 主表。

### W2

| Workload | Matrix (M×N×K) | 1T | 4T | 8T | 最快 |
|---|---:|---:|---:|---:|---:|
| Action Residual | **2560×8×2560** | 1.009 | 0.712 | **0.627** | 8T |
| Action FC1 | **2560×8×17920** | **9.242** | 11.368 | 13.520 | 1T |
| Vision Attention | **1152×256×1152** | 6.875 | **2.131** | 9.829 | 4T |
| Pi0 Expert Q | **2048×32×1024** | 1.276 | 0.896 | **0.866** | 8T |
| Pi0 Expert KV | **256×32×1024** | 0.213 | **0.140** | 0.141 | 4T |
| Pi0 Expert O | **1024×32×2048** | 1.289 | **0.731** | 0.733 | 4T |
| Pi0 Expert GateUp | **4096×32×1024** | 3.306 | **1.790** | 5.420 | 4T |
| Pi0 Expert Down | **1024×32×4096** | 2.728 | **0.837** | 1.518 | 4T |

**W2：1T→4T 有 7/8 個 shape 變快，但 4T→8T 有 6/8 個 shape 反而變慢。**

### W4

| Workload | Matrix (M×N×K) | 1T | 4T | 8T | 最快 |
|---|---:|---:|---:|---:|---:|
| Action Residual | **2560×8×2560** | 2.109 | **0.599** | 5.826 | 4T |
| Action FC1 | **2560×8×17920** | 16.647 | **10.604** | 15.129 | 4T |
| Vision Attention | **1152×256×1152** | 14.152 | 8.765 | **5.662** | 8T |
| Pi0 Expert Q | **2048×32×1024** | 2.485 | **0.809** | 5.517 | 4T |
| Pi0 Expert KV | **256×32×1024** | 0.369 | **0.144** | 0.146 | 4T |
| Pi0 Expert O | **1024×32×2048** | 2.492 | **0.718** | 1.715 | 4T |
| Pi0 Expert GateUp | **4096×32×1024** | 6.102 | **1.343** | 3.083 | 4T |
| Pi0 Expert Down | **1024×32×4096** | 5.180 | **1.517** | 10.039 | 4T |

**W4：1T→4T 為 8/8 全部變快，但 4T→8T 有 7/8 個 shape 變慢。**

這是目前後續最值得研究的地方。

---

## 7. 結論

目前可以直接從結果支持的結論：

1. **Microsoft Official W2 是目前最穩定、也最常取得最佳 CPU latency 的 T-MAC reference。**
2. **本專案 T-MAC W2 並不是大矩陣一定變慢。** 在大型 square 已能接近 MKL，在大型 rectangular 甚至能超過 MKL。
3. **真正不穩定的是 thread scaling，尤其 4T→8T。**
4. W3 / W4 的 bit-serial 工作量較高，多數情況 latency 高於 W2；但 schedule 仍會造成局部反例。
5. RTX 4070 SUPER cuBLAS 是完全不同級別的 GPU dense-GEMM reference，但屬 cross-hardware comparison，不能直接當作演算法公平比較。
6. 下一步最值得做的是固定 BM / BN / KFactor，控制 schedule 變因後比較 1T / 4T / 8T，才能進一步定位 high-thread regression 來源。
