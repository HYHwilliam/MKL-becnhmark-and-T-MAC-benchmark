# Official T-MAC Comparison — Results Analysis

本文件整理目前 repository 中可重現的 benchmark 結果，重點不是只看「誰比較快」，而是回答三個問題：

1. standalone x86 T-MAC 的 thread scaling 到底發生什麼事？
2. Microsoft official T-MAC 在相同 logical HackMD shapes 上呈現什麼 scaling？
3. standalone 與 official 的差距，哪些可以直接從數據支持，哪些不能過度解讀？

## 1. 資料來源與比較邊界

主要資料來源：

- `official_compare/results/summary/results_summary.txt`
  - standalone VLA W2 / W4：8 shapes × 1/4/8 threads
  - official `hackmd_safe` W2：4 shapes × 1/2/4/8 threads
- `x86_GEMM/hackmd_compare/results/summary_hackmd.csv`
  - standalone W2/W3/W4A16 與 MKL 的既有 HackMD 統計
- `official_compare/adapters/profile_compare.py`
  - official workload 定義與 logical M/N/K mapping
- `x86_GEMM/vla_compare/tmac_vla_benchmark.cpp`
  - standalone VLA timing、autotune 與 `active_threads`

Shape 在本專案一律以 `M×N×K` 顯示。

Official adapter 在呼叫 codegen 前會使用 `_MKN = [M * bits, K, N]`。這是 official bit-plane/codegen interface 所需的內部展開；CSV 中的 `shape_mnk` 仍保留 logical GEMM shape，因此不能把 `M*bits` 誤解成 benchmark 換成了不同的 logical matrix。

### 公平性限制

這裡的 direct official-vs-standalone latency ratio 只能視為 **implementation-level observation**：

- standalone：FP16 activation/storage，AVX2 + F16C + OpenMP，`total_ms = preprocessing + kernel`。
- official Intel Linux：pinned Microsoft T-MAC generated TVM kernel；官方 platform configuration 使用 `out_dtype=float32`。
- 兩邊 measurement protocol 不完全相同。
- weight packing 與 autotuning 都不納入 standalone 正式 latency；official 也使用自己的 codegen/evaluator 路徑。

所以本文不把「1.5×」寫成純 T-MAC kernel speedup，只寫成該 benchmark harness 下的 observed latency ratio。

---

## 2. 相同 HackMD W2 shapes：standalone vs official

四個 `hackmd_safe` shapes 可以和既有 standalone W2A16 HackMD 結果直接對齊 logical M/N/K。

### Exact latency

| Shape | Standalone W2 1/2/4/8T (ms) | Official W2 1/2/4/8T (ms) |
|---|---:|---:|
| 256×256×256 | 0.459 / 0.300 / 0.341 / 0.288 | 0.311 / 0.350 / 0.185 / 0.104 |
| 1024×1024×1024 | 22.822 / 13.000 / 7.517 / 8.867 | 17.296 / 8.827 / 4.366 / 6.004 |
| 4096×1024×2048 | 184.947 / 91.515 / 49.083 / 45.987 | 136.949 / 68.567 / 40.510 / 43.115 |
| 1024×1024×512 | 12.618 / 6.804 / 4.185 / 4.609 | 9.095 / 4.600 / 3.771 / 3.045 |

令：

`ratio = standalone latency / official latency`

ratio > 1 代表這次量測中 official 較快。

| Shape | 1T | 2T | 4T | 8T |
|---|---:|---:|---:|---:|
| 256×256×256 | 1.48× | 0.86× | 1.84× | 2.77× |
| 1024×1024×1024 | 1.32× | 1.47× | 1.72× | 1.48× |
| 4096×1024×2048 | 1.35× | 1.34× | 1.21× | 1.07× |
| 1024×1024×512 | 1.39× | 1.48× | 1.11× | 1.51× |
| **Geometric mean** | **1.38×** | **1.26×** | **1.44×** | **1.60×** |

### 解讀

四個 shapes × 四種 thread budget 共 16 組，official latency 在 **15/16 組較低**。唯一例外是最小的 `256³` 2T：standalone 0.300 ms、official 0.350 ms。

這表示目前 handwritten standalone 實作已經捕捉到 T-MAC 的核心運算邏輯，但和 official generated schedule 之間仍存在明顯的 implementation gap。

這個 gap 在 `1024³` 相當穩定，而 `4096×1024×2048` 到 8T 時縮到約 1.07×，表示 standalone 在某些大型 rectangular shape 的 observed latency 已經接近 official。

最小 `256³` 的結果很不規則：official 2T 比 1T 還慢，但 4T、8T 又快速下降。這也說明 sub-millisecond workload 對 thread startup、schedule、measurement overhead 特別敏感，不適合只拿單一 thread point 下結論。

---

## 3. Official W2 的 thread scaling

| Shape | 1T | 2T | 4T | 8T | 1→4 speedup | 4→8 |
|---|---:|---:|---:|---:|---:|---:|
| 256×256×256 | 0.311 | 0.350 | 0.185 | 0.104 | 1.68× | 1.78× faster |
| 1024×1024×1024 | 17.296 | 8.827 | 4.366 | 6.004 | 3.96× | 1.38× slower |
| 4096×1024×2048 | 136.949 | 68.567 | 40.510 | 43.115 | 3.38× | 1.06× slower |
| 1024×1024×512 | 9.095 | 4.600 | 3.771 | 3.045 | 2.41× | 1.24× faster |

Official 並不是「thread 越多一定越快」。對 `1024³` 與大型 rectangular，4T 已接近較好的 operating point，8T 反而小幅 regression；另外兩個 shapes 則仍受益。

較保守的結論是：

> official implementation 的 1→4T scaling 整體較穩定，而 4→8T 開始受到 workload、schedule 與 CPU 資源限制；8T 並非普遍最佳。

---

## 4. Standalone VLA：4T 通常有效，8T 明顯不穩定

### Aggregate

| Bit width | 1→4T geometric-mean speedup | 1→8T geometric-mean speedup | 1→4T 變快 | 4→8T regression |
|---|---:|---:|---:|---:|
| W2 | 1.74× | 1.16× | 7 / 8 shapes | 6 / 8 shapes |
| W4 | 2.80× | 1.07× | 8 / 8 shapes | 7 / 8 shapes |

這是目前最重要的 standalone 結果。

如果只看 1T→4T，實作其實不是「矩陣一大、多執行緒就一定變差」：W2 幾乎全部改善，W4 全部改善。

真正異常的是 **4T→8T**。W2 只有 2/8 shapes 繼續改善；W4 只剩 1/8。

因此目前問題應更精確地描述為：

> standalone 的主要問題不是「完全不會 multithreading」，而是高 thread count 下 schedule / work partition 的穩定性不足。

### 代表案例

| Case | 1T | 4T | 8T | 8T active threads | 觀察 |
|---|---:|---:|---:|---:|---|
| W2 Action Residual | 1.009 | 0.712 | 0.627 | 5 | 8T仍小幅改善，但實際只用5 threads |
| W2 Action FC1 | 9.242 | 11.368 | 13.520 | 8 | thread 越多越慢，不能用 active-thread 不足解釋 |
| W2 Vision Attention | 6.875 | 2.131 | 9.829 | 8 | 4T 很好，8T 嚴重 regression |
| W2 Pi0 Down | 2.728 | 0.837 | 1.518 | 2 | 8T request 實際只用2 threads |
| W4 Action Residual | 2.109 | 0.599 | 5.826 | 8 | full 8 threads 仍大幅變慢 |
| W4 Vision Attention | 14.152 | 8.765 | 5.662 | 8 | 少數 8T 明確繼續受益的 case |
| W4 Pi0 Q | 2.485 | 0.809 | 5.517 | 8 | full 8 threads 仍 regression |
| W4 Pi0 Down | 5.180 | 1.517 | 10.039 | 8 | 最明顯的 8T regression 之一 |

### `active_threads` 不是唯一原因

確實有不少 requested 8T 最後沒有用滿：

- W2 Action Residual：5 active
- W2 Pi0 Q：4 active
- W2 Pi0 KV：2 active
- W2 Pi0 O：2 active
- W2 Pi0 Down：2 active
- W4 Pi0 KV / O：4 active

這會直接限制 scaling。

但另一批 case 在 `active_threads=8` 時一樣大幅變慢，例如 W2 Vision Attention / GateUp，以及 W4 Action Residual / Q / GateUp / Down。

所以不能把問題只歸因於「沒有開滿 8 threads」。更可能需要一起檢查：

- autotune 選到的 BM / BN 是否適合 high-thread execution
- OpenMP parallel axis 與 tile 數量是否能平均分工
- 每 thread working set 與 cache locality
- memory bandwidth / shared-cache contention
- thread launch / synchronization overhead
- tune 時的 candidate ranking 是否能代表正式多次測量

---

## 5. Preprocessing 已開始限制小 shape

Standalone `total_ms` 包含 preprocessing + kernel。

最明顯的是 Pi0 KV：

- W2 KV：4T/8T preprocessing 約佔 total 的 **42.7%**
- W4 KV：4T/8T 約 **44%**

W2 Pi0 Down 在 4T 時 preprocessing 也約佔 **26.5%**。

因此當 kernel 被多執行緒壓得很短後，固定 preprocessing 成本會快速變成 Amdahl-like bottleneck。

這代表後續優化不能只盯 kernel GFLOPS；對 small-M / small-N workload，preprocess fusion、reuse 或降低固定成本可能和 kernel tuning 一樣重要。

---

## 6. W2 vs W4：不要只用 bit-width 判斷 scaling

在 1T 下，W4 大多比 W2 慢，符合 bit-serial 工作量增加的直覺。

但多執行緒下存在反例：

- Action Residual 4T：W4 0.599 ms，低於 W2 0.712 ms。
- Pi0 O 4T：W4 0.718 ms，和 W2 0.731 ms 接近。

這不表示 W4 的理論成本突然低於 W2，而是說目前量到的 latency 同時受到 schedule selection、thread mapping、cache behavior 與測量變異影響。

所以 bit-width 比較應優先看同一套穩定 schedule 或多輪統計，不能只抓一個最快 row 推導「W4 比 W2 更快」。

---

## 7. 這次 official 比較真正告訴我們什麼

先前 standalone 測試顯示「矩陣變大、多執行緒可能反而變差」，容易讓人懷疑是不是 T-MAC 本身不適合這些 workload。

現在 official W2 提供了重要對照：

- official 在中大型 HackMD shapes 上 1→4T 有約 2.4×～4.0× scaling。
- official 的 4→8T 也會飽和甚至小幅 regression，因此不是所有 regression 都是 standalone 獨有。
- 但 standalone VLA 在 4→8T 的 regression **更頻繁、幅度也更大**。

因此目前證據比較支持：

> 問題主要落在 standalone implementation 的 schedule / parallelization quality，而不是 T-MAC 方法本身在 CPU 上必然「矩陣越大越慢」。

這也說明直接跑 Microsoft official code 的價值：它提供了一個 generated-kernel reference，讓我們可以把「演算法限制」和「自己的 kernel 實作限制」拆開。

---

## 8. 目前不能下的結論

以下幾點目前資料不足，不能寫成確定結論：

1. **不能把 official/standalone ratio 當成純 kernel speedup。** dtype 與 timing harness 不同。
2. **不能說 8 threads 一定使 official 變慢。** 目前 official W2 四個 shapes 有 2 個改善、2 個 regression。
3. **不能說 standalone 的問題全部來自 active_threads。** full-8-thread case 也有嚴重 regression。
4. **不能正式比較 official W3/W4。** committed `results_summary.txt` 目前只有完整 official W2 HackMD-safe。
5. **不能用目前 committed summary 做完整 official-vs-standalone VLA ratio。** standalone VLA 已提交，但 official VLA final CSV 尚未整理成 tracked summary。
6. `4096³` 不在 `hackmd_safe`；先前 official tuning 曾遇到 timeout/fallback，因此在沒有有效 tuned result 前不應拿來做正式 latency comparison。

---

## 9. 下一步最有價值的驗證

若接下來要回答「為什麼 standalone 8T 掉這麼多」，最值得做的不是再盲目增加 shape，而是固定現有 workload 做 controlled experiment：

1. 對每個 regression case 固定同一 BM/BN/KFactor，比較 1/4/8T，先移除 autotune schedule 改變這個變因。
2. 同時記錄 requested_threads / active_threads / tile count / 每 thread tile 數。
3. 將 preprocessing 與 kernel latency 分開比較，尤其 Pi0 KV / O / Down。
4. 對 full-8-thread 仍 regression 的 case 優先檢查 cache miss、memory bandwidth 與 OpenMP synchronization。
5. 把 local official VLA final `results.csv` 整理成 tracked CSV，再做真正相同 VLA shapes 的 official-vs-standalone table。

這五步完成後，才能更有把握判斷差距主要來自 schedule search、parallel axis、memory behavior，還是 handwritten kernel 本身。

---

## 10. Current conclusion

目前最有證據支持的結論是：

- standalone T-MAC 的 **1→4T scaling 基本有效**。
- **4→8T 是主要異常區間**，而且 W4 比 W2 更容易出現明顯 regression。
- active thread under-utilization 是其中一個原因，但不是全部。
- preprocessing 對 small-M / small-N workload 已經成為不可忽略的固定成本。
- Microsoft official W2 在相同 HackMD-safe logical shapes 上整體 latency 較低，且中大型 shape 的 1→4T scaling 更一致。
- 因此目前優先方向不是推翻 T-MAC 的核心數學，而是改善 standalone 的 schedule selection、thread partition 與 high-thread-count behavior。
