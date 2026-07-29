# T-MAC 多執行緒驗證（Pthread / OpenMP / Thread8 壓力測試）

---

## 1. Pthread Benchmark
### 編譯指令
```bash
g++ -O2 -mavx2 -mfma t_mac_pthread_benchmark.cpp -lpthread -o t_mac_pthread_benchmark

./t_mac_pthread_benchmark
```
---
## 執行結果

| Matrix Size | threads=1 | threads=2 | threads=4 | threads=8 |
|:-----------:|----------:|----------:|----------:|----------:|
| 1024×1024 | 1.29 | 5.31 | 5.66 | 2.00 |
| 2048×2048 | 30.81 | 22.73 | 16.11 | 7.65 |
| 4096×4096 | 52.89 | 78.89 | 78.53 | 25.34 |
| 8192×8192 | 75.90 | 86.88 | 104.74 | 38.57 |

> 單位：GFLOPS  
> 所有矩陣尺寸的 `max_err` 與正確性驗證皆為 **OK**。

---

## 分析
### 1024 × 1024
- `threads=1` 效能僅 **1.29 GFLOPS**，明顯偏低。
- 增加至 `threads=2`、`threads=4` 後提升至約 **5.3~5.7 GFLOPS**。
- `threads=8` 又再次下降。
>  **原因分析**
>
> Pthread 版本每次 Benchmark 都需要重新執行 `pthread_create()` 與 `pthread_join()`。
>
> 對於小矩陣而言，Thread 建立與回收成本相對於真正的運算時間過高，因此增加 Thread 數反而造成額外負擔。

---
### 2048 × 2048
效能呈現單調下降：
```text
30.81 → 22.73 → 16.11 → 7.65 GFLOPS
```
表示 Thread 建立成本已開始大於平行化所帶來的效益。

---

### 4096 × 4096、8192 × 8192
- `threads=2`、`threads=4` 有約 **1.5~2 倍**提升。
- `threads=8` 再次出現明顯衰退。
代表矩陣變大後，平行化開始發揮作用，但過多 Thread 仍受到建立/銷毀成本影響。

## 小結
- 小矩陣主要受到 Thread 建立成本影響。
- 大矩陣可透過適量 Thread 提升效能。
- `threads=8` 在所有尺寸皆出現效能下降，符合 Pthread 每次重新建立 Thread 的預期結果。

---

## 2. OpenMP Benchmark

### 編譯指令

```bash
g++ -O2 -mavx2 -mfma -fopenmp t_mac_openmp_benchmark.cpp -o t_mac_openmp_benchmark
```
### 綁核指令說明
```bash
# 啟用綁核：強制每個 OpenMP thread 固定綁在特定實體核心，不隨 OS 排程器搬移
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# 取消綁核：讓 OpenMP thread 交由 OS 排程器自由調度
unset OMP_PROC_BIND
unset OMP_PLACES
```
---

## 綁核預想作用

T-MAC 的核心運算大量依賴 **L1 / L2 Cache** 中的 Lookup Table。
若 Thread 被 OS 排程到其他 CPU Core，原本快取中的資料需要重新載入，容易造成 Cache Miss。

啟用綁核可讓每個 Thread 固定執行於相同核心，降低 Thread 遷移造成的 Cache 失效，提高整體效能穩定性。

---

## 執行結果（未綁核）

| Matrix Size | threads=1 | threads=2 | threads=4 | threads=8 |
|:-----------:|----------:|----------:|----------:|----------:|
|1024×1024|48.70|117.33|213.83|6.46|
|2048×2048|220.12|118.91|227.46|41.67|
|4096×4096|188.92|99.91|5.75|65.09|
|8192×8192|185.26|64.78|18.06|187.02|

---

## 執行結果（已綁核）

| Matrix Size | threads=1 | threads=2 | threads=4 | threads=8 |
|:-----------:|----------:|----------:|----------:|----------:|
|1024×1024|66.25|140.70|216.56|7.21|
|2048×2048|210.76|341.95|557.74|277.62|
|4096×4096|216.96|396.81|4.93|72.87|
|8192×8192|130.10|373.64|15.20|180.06|

---

## 分析

### 綁核的正面效果

2048×2048 表現最為明顯。

|Thread|未綁核|綁核|
|:--:|--:|--:|
|4|227.46|557.74|
|8|41.67|277.62|

可觀察到約 **2.5～6.7 倍**的效能提升。
代表固定 Thread 所在 CPU Core 後，可有效減少 Cache Miss，提高多執行緒效率。

---

### 異常現象
4096×4096 的 `threads=4`：
```text
未綁核：5.75 GFLOPS
綁核：4.93 GFLOPS
```
兩者皆遠低於其他 Thread 配置。

此外：
8192×8192 的 `threads=4`

```text
18.06 → 15.20 GFLOPS
```
綁核同樣沒有改善。
代表除了 Thread 遷移外，仍存在其他效能瓶頸（例如記憶體頻寬競爭、排程器行為、或系統背景負載）。

---
### `threads=8` 效能表現較不穩定
多數矩陣尺寸下 threads=8 明顯低於 threads=4（例如綁核狀態下 2048x2048：threads=4 有 557.74 GFLOPS，threads=8 卻只有 277.62 GFLOPS），但 8192x8192 這個尺寸在兩種情境下 threads=8 反而是該尺寸表現最好或次好的設定（未綁核 187.02、綁核 180.06，均高於同尺寸 threads=4）。
顯示不同矩陣尺寸對 Thread 數量具有不同最佳配置。

## 小結

- 綁核可有效減少 Thread 遷移造成的 Cache Miss。
- 2048×2048 有最明顯的效能提升。
- 部分矩陣尺寸仍存在其他瓶頸，綁核無法完全改善。
- Thread 數量並非越多越快，需依矩陣大小調整。

---

## 3. Thread8 壓力測試

### 編譯指令

```bash
g++ -O2 -mavx2 -mfma -fopenmp t_mac_thread8_stress.cpp -o t_mac_thread8_stress

./t_mac_thread8_stress
```

### 測試條件
測試前取消綁核

固定條件：
- Matrix Size：8192 × 8192
- Thread：8
- Iteration：3000

---

## 執行結果

```text
總運行時間 : 2.37848 秒
平均延遲 : 0.792827 ms
效能 : 169.29 GFLOPS
output[0..3]
131.076
-311.614
73.097
-41.8743
```

---

## 分析

本測試主要驗證 `threads=8` 在大型矩陣下的穩定性。

測得：

```text
169.29 GFLOPS
```
與 OpenMP Benchmark 未綁核時：

```text
187.02 GFLOPS
```
屬於相同量級。

表示：
- 大矩陣
- Thread = 8
- 未綁核
此配置具有不錯的重現性與穩定性。

---

## 4. 綜合結論

### Pthread

- 小矩陣容易受到 Thread 建立成本影響。
- 大矩陣可透過適量 Thread 提升效能。
- `threads=8` 在所有矩陣尺寸皆出現衰退，符合 Thread 建立/銷毀成本累積的預期。

---

### OpenMP

- 綁核可有效降低 Thread 遷移造成的 Cache Miss。
- 2048×2048 有最明顯的效能提升（最高約 6.7 倍）。
- 4096×4096 的 `threads=4` 仍出現異常低點，表示尚有其他系統因素影響效能。

---

### Thread8 壓力測試

- 8192×8192、`threads=8` 在未綁核條件下約可維持 **170～190 GFLOPS**。
- 與 OpenMP Benchmark 結果一致，代表此配置具有良好的重現性。

---

## 5. 最終結論
本次多執行緒測試驗證了不同平行化策略對 T-MAC 的影響：
- **Pthread** 受限於 Thread 建立與回收成本，Thread 數增加並不一定帶來效能提升。
- **OpenMP** 搭配 CPU 綁核可有效改善部分矩陣尺寸的平行效率，但仍受到記憶體頻寬與系統排程等因素限制。
- **Thread8 壓力測試** 與 Benchmark 結果互相印證，確認大型矩陣下的效能具有良好的穩定性與可重現性。