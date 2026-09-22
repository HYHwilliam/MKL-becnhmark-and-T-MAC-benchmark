# Official T-MAC Comparison

這個目錄用來保存 **standalone x86 T-MAC** 與 **Microsoft 官方 T-MAC** 的比較流程、版本資訊與整理後結果。

## 目錄

| 路徑 | 用途 |
|---|---|
| `adapters/profile_compare.py` | 對官方 T-MAC 加上的 workload / CLI adapter，不修改官方核心 kernel |
| `scripts/tmac_env.sh` | 載入 external T-MAC、TVM、LLVM 與 `tvm-build` Conda 環境 |
| `scripts/run_vla_and_hackmd_compare.sh` | standalone VLA + official HackMD-safe benchmark |
| `scripts/run_official_overnight.sh` | official VLA / HackMD benchmark runner |
| `provenance/versions.md` | T-MAC、TVM、llama.cpp、ExecuTorch pinned revisions 與搬移紀錄 |
| `results/summary/results_summary.txt` | 目前已提交 Git 的 benchmark 摘要 |
| `RESULTS_ANALYSIS.md` | 整理後的 benchmark 比較與分析 |
| `results/raw/` | 本機保存的 raw snapshot，Git ignore |
| `results/runs/` | 之後每次新 benchmark 的獨立輸出，Git ignore |

## Official baseline

Microsoft T-MAC repository 放在本 repository 外：

`~/tmac_external/T-MAC-official-x86`

Pinned T-MAC commit：

`7042f8f73330bd083bc1e4bc5ccb3f88a4904aee`

完整 dependency revisions 見 `provenance/versions.md`。

這個架構刻意把三種東西分開：

- Microsoft upstream source / generated TVM build
- 本 repository 的 standalone x86 implementation
- comparison adapter、runner 與結果

因此跑 official benchmark 不需要修改 Microsoft 的核心 `python/t_mac/ops/*`。

## Environment

載入 comparison environment：

```bash
source official_compare/scripts/tmac_env.sh
```

目前 relocated TVM binary 已驗證能從新的 external path 正常載入。既有 `3rdparty/tvm/build` 仍含搬移前的 CMake absolute paths，因此視為 **frozen benchmark build**。

未來若要重建 TVM，應從新位置建立乾淨 build directory，不要在舊 build tree 上 incremental rebuild。

## Benchmark scope

目前 committed summary 主要包含：

1. standalone VLA workload：W2 / W4，1 / 4 / 8 threads。
2. official T-MAC `hackmd_safe`：W2，1 / 2 / 4 / 8 threads。
3. repository 既有的 standalone HackMD 統計，可用相同 logical shapes 與 official W2 做描述性比較。

VLA suite 中 BitVLA 使用實際測試 shape；π₀ Action Expert 因 official T-MAC 的 legal BN 限制，以 `N=32` 作為 real `N=33` 的 proxy。

## 重要比較限制

**Official 與 standalone 的絕對 latency 不是嚴格 apples-to-apples kernel benchmark。**

- standalone x86 路徑為 W2/W3/W4A16，FP16 storage + AVX2/F16C/OpenMP。
- pinned official T-MAC 的 Intel Linux device configuration 使用 generated TVM kernel，`out_dtype=float32`。
- standalone 報告的 `total_ms` 為 preprocessing + kernel，weight packing 與 autotuning 不計入。
- official 與 standalone 的 measurement protocol 不完全相同。

因此 `RESULTS_ANALYSIS.md` 中的 official/standalone ratio 應解讀為 **兩套 benchmark harness 下的 observed implementation latency difference**，不能直接宣稱成純 kernel 或純演算法 speedup。

## 目前結果重點

- 在 4 個可直接對齊的 HackMD-safe W2 shapes 上，official T-MAC 在 16 個 shape×thread 組合中有 **15 個 latency 較低**；唯一例外是 `256x256x256` 的 2T。
- 這 4 個 shapes 的 standalone/official geometric-mean latency ratio 約為：**1T 1.38×、2T 1.26×、4T 1.44×、8T 1.60×**。
- standalone VLA 在 **4T 通常明顯改善**：W2 有 7/8 shapes 比 1T 快，W4 為 8/8。
- 但從 4T 增加到 8T 時，standalone 出現明顯 scaling 問題：W2 有 6/8 regress，W4 有 7/8 regress。
- `active_threads < requested_threads` 能解釋部分案例，但不是唯一原因；一些 `active_threads=8` 的 case 仍大幅變慢，表示 schedule / BM-BN choice、parallel overhead、cache / bandwidth interaction 也需要檢查。
- 小 shape 在多執行緒下 preprocessing 佔比可超過 40%，kernel 即使加速也會受到固定成本限制。

完整數字與解讀見 **[RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md)**。

## Running

Combined standalone VLA + official HackMD-safe：

```bash
official_compare/scripts/run_vla_and_hackmd_compare.sh
```

Official VLA + HackMD：

```bash
official_compare/scripts/run_official_overnight.sh
```

每次新 run 會寫到：

`official_compare/results/runs/<RUN_ID>/`

不會自動刪除舊 run。
