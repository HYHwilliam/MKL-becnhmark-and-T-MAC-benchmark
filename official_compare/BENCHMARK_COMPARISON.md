# Benchmark Comparison

> **Matrix notation：M×N×K。Latency 越低越好。**

這份文件把目前正式保存的 CPU / GPU benchmark 結果集中到一張主表。

名稱統一如下：

- **本專案 T-MAC**：本 repository 自行實作的 x86 AVX2/F16C/OpenMP T-MAC。
- **Microsoft Official W2**：pinned Microsoft T-MAC + TVM generated x86 kernel。
- **MKL FP16（本機）**：Intel Core Ultra 7 258V 上實際執行的 oneMKL dense FP16 GEMM。
- **學長 HackMD MKL reference**：原 HackMD 提供的 MKL reference；與本機 MKL 分開保存、分開解讀。
- **RTX 4070 SUPER cuBLAS**：GPU dense FP16 GEMM reference，不是 GPU T-MAC。

主表中：

- **CPU backend 同一列最快的 latency 會加粗。** 粗體比較只包含本機 MKL、本專案 T-MAC 與可用的 Microsoft Official；學長 HackMD reference 不參與 winner 判定。
- GPU 沒有 CPU thread 數，因此同一 matrix 的 cuBLAS latency 會重複顯示，只是方便直接比較。
- 本機 MKL 與本專案原始 x86 benchmark 的 provenance 對應 `Intel Core Ultra 7 258V`；學長 HackMD MKL reference 是另一份歷史 reference。
- 學長 HackMD 的 `1024×1024×512` MKL reference 明確標示 FP16；其餘四個 shape 來自原 HackMD 第一張 MKL reference table，原表未明確標示 dtype，因此只作 reference，不主張為 FP16 apples-to-apples comparison。
- Official `4096×4096×4096` 沒有可信 tuned result，因此保留 `—`。
- Official `4096×1024×2048` W2 1T 的所有 tuning candidates 都失敗並觸發 fallback，因此標成 `— (fallback)`，不納入正式比較。
- Official 目前只整理 1 / 2 / 4 / 8T，因此 16T 顯示 `—`。
- 16T 保留原 HackMD sweep 結果，但屬高 thread-count / oversubscription 參考，不當作主要結論。

## 1. 完整結果總表

| Category | Matrix (M×N×K) | CPU Threads | MKL FP16 本機 258V (ms) | 學長 HackMD MKL ref (ms) | 本專案 T-MAC W2A16 (ms) | 本專案 T-MAC W3A16 (ms) | 本專案 T-MAC W4A16 (ms) | Microsoft Official W2 (ms) | RTX 4070 SUPER cuBLAS (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Small Square | **256×256×256** | 1 | **0.136** | 0.2365 | 0.459 | 0.651 | 0.854 | 0.311 | 0.006912 |
| Small Square | **256×256×256** | 2 | **0.128** | 0.2465 | 0.300 | 0.395 | 0.485 | 0.350 | 0.006912 |
| Small Square | **256×256×256** | 4 | **0.124** | 0.1462 | 0.341 | 0.449 | 0.304 | 0.185 | 0.006912 |
| Small Square | **256×256×256** | 8 | 0.114 | 0.3258 | 0.288 | 0.380 | 0.279 | **0.104** | 0.006912 |
| Small Square | **256×256×256** | 16 | **0.129** | 0.4081 | 0.289 | 0.380 | 0.282 | — | 0.006912 |
| Medium Square | **1024×1024×1024** | 1 | **7.219** | 16.7570 | 22.822 | 33.350 | 49.687 | 17.296 | 0.055 |
| Medium Square | **1024×1024×1024** | 2 | 9.510 | 12.3002 | 13.000 | 17.684 | 21.069 | **8.827** | 0.055 |
| Medium Square | **1024×1024×1024** | 4 | 7.205 | 6.4828 | 7.517 | 10.349 | 13.361 | **4.366** | 0.055 |
| Medium Square | **1024×1024×1024** | 8 | 9.025 | 3.5757 | 8.867 | 13.068 | 13.124 | **6.004** | 0.055 |
| Medium Square | **1024×1024×1024** | 16 | **7.304** | 5.9278 | 7.772 | 10.281 | 12.314 | — | 0.055 |
| Large Square | **4096×4096×4096** | 1 | **331.529** | 2071.9777 | 1664.725 | 2262.285 | 2230.488 | — | 1.986 |
| Large Square | **4096×4096×4096** | 2 | **320.147** | 1048.0653 | 735.940 | 1119.599 | 1387.306 | — | 1.986 |
| Large Square | **4096×4096×4096** | 4 | **344.963** | 527.3677 | 381.560 | 631.803 | 826.431 | — | 1.986 |
| Large Square | **4096×4096×4096** | 8 | 324.934 | 266.5019 | **320.574** | 529.392 | 712.419 | — | 1.986 |
| Large Square | **4096×4096×4096** | 16 | **325.085** | 260.4321 | 435.729 | 599.435 | 671.333 | — | 1.986 |
| Rectangular | **4096×1024×2048** | 1 | **51.362** | 260.9551 | 184.947 | 275.047 | 273.532 | — (fallback) | 0.278 |
| Rectangular | **4096×1024×2048** | 2 | **59.344** | 130.7603 | 91.515 | 131.710 | 173.782 | 68.567 | 0.278 |
| Rectangular | **4096×1024×2048** | 4 | 49.866 | 65.7123 | 49.083 | 73.907 | 104.054 | **40.510** | 0.278 |
| Rectangular | **4096×1024×2048** | 8 | 51.740 | 33.2634 | 45.987 | 73.204 | 91.422 | **43.115** | 0.278 |
| Rectangular | **4096×1024×2048** | 16 | 50.053 | 72.7828 | **43.478** | 66.658 | 86.544 | — | 0.278 |
| Medium Rectangular | **1024×1024×512** | 1 | **4.775** | 3.410722 | 12.618 | 17.832 | 17.649 | 9.095 | 0.033 |
| Medium Rectangular | **1024×1024×512** | 2 | **3.488** | 3.164299 | 6.804 | 9.471 | 11.632 | 4.600 | 0.033 |
| Medium Rectangular | **1024×1024×512** | 4 | **3.465** | 3.135987 | 4.185 | 5.902 | 8.206 | 3.771 | 0.033 |
| Medium Rectangular | **1024×1024×512** | 8 | 3.475 | 3.123567 | 4.609 | 6.237 | 5.797 | **3.045** | 0.033 |
| Medium Rectangular | **1024×1024×512** | 16 | 5.069 | 3.145073 | **4.488** | 5.693 | 5.976 | — | 0.033 |

這張表就是目前最主要的結果。不要先看後面的分析，光看粗體就能知道每一組 CPU configuration 誰最快。

---

## 2. 每個矩陣的最快結果

> 此節的「最快 CPU backend」只比較本機 MKL、本專案 T-MAC 與有效的 Microsoft Official；學長 HackMD reference 因為是不同來源／硬體，而且部分 dtype 未明確標示，不參與 winner 判定。

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
- **4096×1024×2048**：CPU 最快是 Microsoft Official W2 4T，40.510 ms；Official 1T 為 fallback，不納入比較。
- **1024×1024×512**：CPU 最快是 Microsoft Official W2 8T，3.045 ms。

所以現在不能簡化成「T-MAC 一定比 MKL 快」或「矩陣越大 T-MAC 越差」。

更準確的是：**結果高度依賴 matrix shape、bit-width、thread 數與 schedule。**


---

**HackMD MKL reference provenance**

主表新增的「學長 HackMD MKL ref」與「MKL FP16 本機 258V」不是同一批結果：

- `MKL FP16 本機 258V`：由本 repository 的 `mkl_fp16_hackmd_benchmark.cpp` 五輪實測後統整。
- `學長 HackMD MKL ref`：由 `summarize_results.py` 中保存的 `HACKMD_MKL_REFERENCE` 讀入。
- `1024×1024×512` 的 HackMD reference 明確確認為 FP16。
- 其餘四個 shape 的原 HackMD 第一張 MKL table 沒有明確寫 dtype，因此只保留為歷史 reference。

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

- Official W2 1T：**tuning 全部失敗，fallback，不納入比較**
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

這和 VLA 測試看到的結果一致：**本專案 4T 通常有效，但 8T 常出現 regression；Microsoft Official VLA 的 scaling 整體明顯更穩定。**

---

## 6. VLA scaling 補充

VLA 使用另一組 matrix shapes，因此不混進上面的 HackMD 主表。

這裡把 **本機 MKL FP16、本專案 T-MAC W2/W4、Microsoft Official T-MAC W2/W4** 放在同一張表。
粗體代表該列目前記錄的 CPU latency 數值最低者；由於 MKL、本專案與 Official 的計算語意 / timing harness 並非完全相同，粗體只代表數值比較，不等同嚴格 apples-to-apples 的演算法勝負。

### VLA 完整比較

| Workload | Matrix (M×N×K) | Threads | MKL FP16 本機 258V (ms) | 本專案 W2 (ms) | Official W2 (ms) | 本專案 W4 (ms) | Official W4 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Action Residual | **2560×8×2560** | 1 | 8.544 | 1.009 | **0.819** | 2.109 | 1.623 |
| Action Residual | **2560×8×2560** | 4 | 8.531 | 0.712 | **0.217** | 0.599 | 0.411 |
| Action Residual | **2560×8×2560** | 8 | 7.911 | 0.627 | **0.211** | 5.826 | 0.413 |
| Action FC1 | **2560×8×17920** | 1 | 41.373 | 9.242 | **6.187** | 16.647 | 11.731 |
| Action FC1 | **2560×8×17920** | 4 | 42.253 | 11.368 | **1.485** | 10.604 | 3.000 |
| Action FC1 | **2560×8×17920** | 8 | 42.003 | 13.520 | **1.519** | 15.129 | 3.053 |
| Vision Attention | **1152×256×1152** | 1 | 5.630 | 6.875 | **5.470** | 14.152 | 11.006 |
| Vision Attention | **1152×256×1152** | 4 | 8.657 | 2.131 | **1.384** | 8.765 | 2.788 |
| Vision Attention | **1152×256×1152** | 8 | 4.009 | 9.829 | **1.388** | 5.662 | 2.803 |
| Pi0 Expert Q | **2048×32×1024** | 1 | 3.757 | 1.276 | **1.077** | 2.485 | 2.145 |
| Pi0 Expert Q | **2048×32×1024** | 4 | 4.547 | 0.896 | **0.280** | 0.809 | 0.558 |
| Pi0 Expert Q | **2048×32×1024** | 8 | 3.738 | 0.866 | **0.288** | 5.517 | 0.573 |
| Pi0 Expert KV | **256×32×1024** | 1 | 1.706 | 0.213 | **0.136** | 0.369 | 0.267 |
| Pi0 Expert KV | **256×32×1024** | 4 | 1.779 | 0.140 | **0.133** | 0.144 | 0.267 |
| Pi0 Expert KV | **256×32×1024** | 8 | 1.805 | 0.141 | **0.133** | 0.146 | 0.142 |
| Pi0 Expert O | **1024×32×2048** | 1 | 3.570 | 1.289 | **1.057** | 2.492 | 2.092 |
| Pi0 Expert O | **1024×32×2048** | 4 | 3.173 | 0.731 | **0.281** | 0.718 | 1.056 |
| Pi0 Expert O | **1024×32×2048** | 8 | 3.518 | 0.733 | **0.281** | 1.715 | 0.548 |
| Pi0 Expert GateUp | **4096×32×1024** | 1 | 6.321 | 3.306 | **2.184** | 6.102 | 4.315 |
| Pi0 Expert GateUp | **4096×32×1024** | 4 | 6.707 | 1.790 | **0.558** | 1.343 | 1.109 |
| Pi0 Expert GateUp | **4096×32×1024** | 8 | 5.470 | 5.420 | **0.578** | 3.083 | 2.282 |
| Pi0 Expert Down | **1024×32×4096** | 1 | 5.529 | 2.728 | **2.064** | 5.180 | 4.306 |
| Pi0 Expert Down | **1024×32×4096** | 4 | 6.297 | 0.837 | **0.545** | 1.517 | 1.068 |
| Pi0 Expert Down | **1024×32×4096** | 8 | 6.699 | 1.518 | **0.541** | 10.039 | 1.074 |

### MKL VLA 測試方式與穩定性

本機 MKL VLA 使用：

```text
CPU      : Intel Core Ultra 7 258V
oneMKL   : 2026.1
API      : cblas_hgemm
Dtype    : FP16 × FP16 → FP16
Threads  : 1 / 4 / 8
Warm-up  : 每個 configuration 10 次
Runs     : 5 次獨立完整 run
Final    : median of five run-level medians
Pi0      : N=32 proxy，與目前 T-MAC VLA comparison 對齊
```

完整性檢查：

```text
5 logs × 24 RESULT rows = 120 raw RESULT rows
24 configurations × 5 runs
checksum 在相同 shape / configuration 間一致
log 中未找到 error / failed / exception / segmentation / killed
```

因此這批 MKL VLA 結果在 **資料完整性與 correctness tracking** 上可以使用。

但它的 **run-to-run variation 很高**：

```text
CV ≥ 20% : 20 / 24 configurations
CV ≥ 30% : 15 / 24 configurations
CV ≥ 50% :  6 / 24 configurations
```

其中最極端的是 `Pi0 Expert KV 4T`，CV 約 **104.11%**。
因此 MKL VLA 欄位應視為 **本機 high-variance baseline**；目前用五輪 median-of-medians 抑制 outlier，但不宜僅靠這批資料對 MKL 的 thread scaling 做強結論。

### VLA 對照重點

Official VLA 已完成 AutoTVM tuning audit：

- W2：231 tuning records，**0 failed**
- W4：252 tuning records，**0 failed**
- 合計：483 tuning records，**0 failed**

在目前 24 個相同 shape/thread point 中：

- Microsoft Official W2 的 recorded latency 為 **24/24 數值最低**。
- 本專案 W2 相對本機 MKL FP16 為 **22/24 較低**。
- 本專案 W4 相對本機 MKL FP16 為 **19/24 較低**。
- Microsoft Official W4 相對本機 MKL FP16 為 **23/24 較低**。

本專案 W2 的 thread scaling：

- 1T→4T：7/8 個 shape 變快。
- 4T→8T：6/8 個 shape 反而變慢。

本專案 W4：

- 1T→4T：8/8 全部變快。
- 4T→8T：7/8 反而變慢。

Official W2/W4 在相同 VLA workload 下整體 scaling 明顯比本專案穩定。
另一方面，本機 MKL VLA 的 run-to-run variation 過高，因此目前不適合把 MKL 1T→4T→8T 的變化直接解讀成可靠的 thread-scaling 趨勢。

> 注意：MKL 使用 dense FP16 `cblas_hgemm`；本專案 T-MAC 是 low-bit weight + A16 implementation；Microsoft Official 的 output / aggregation dtype 與本專案也不完全相同。這張表適合做 implementation-level latency reference，而不是完全等價的 arithmetic / precision comparison。

---

## 7. 結論

目前可以直接從結果支持的結論：

1. **Microsoft Official W2 是目前最穩定、也最常取得最低 recorded CPU latency 的 T-MAC reference。**
2. **本專案 T-MAC W2 並不是大矩陣一定變慢。** 在大型 square 已能接近本機 MKL，在大型 rectangular 甚至能超過本機 MKL。
3. **本專案真正不穩定的是 thread scaling，尤其 4T→8T。** VLA 的 Official 對照後，這個差異更清楚。
4. Official VLA W2/W4 的 483 個 AutoTVM tuning records 全部通過；HackMD 則有少數 workload tuning 全失敗，因此相關 fallback 數字已排除。
5. 新增的本機 MKL VLA 已完成 5 輪、120 個 raw RESULT rows 與 24 個 configuration 的 validation；但其中 **20/24 configurations 的 CV ≥ 20%**，因此它適合作為 high-variance local baseline，不適合單獨拿來下 thread-scaling 的強結論。
6. 在目前 VLA 24 個 matched shape/thread points 中，本專案 W2 有 22 組 latency 低於本機 MKL FP16；但兩者數值語意不同，不能直接等同「低位元演算法在公平條件下必然快於 dense MKL」。
7. W3 / W4 的 bit-serial 工作量較高，多數情況 latency 高於 W2；但 schedule 仍會造成局部反例。
8. RTX 4070 SUPER cuBLAS 是完全不同級別的 GPU dense-GEMM reference，但屬 cross-hardware comparison，不能直接當作演算法公平比較。
9. 學長 HackMD MKL 數字只作歷史 cross-machine reference；正式本機 MKL baseline 仍以 Intel Core Ultra 7 258V 的實測結果為準。
10. 下一步若要定位本專案 high-thread regression，最值得做的是固定 BM / BN / KFactor 並比較 1T / 4T / 8T；若要進一步研究 MKL VLA thread scaling，則應先降低目前 run-to-run variance，再重跑一輪受控測試。
