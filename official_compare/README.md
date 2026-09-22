# Official T-MAC Comparison

這個目錄用來整理 **本專案 x86 T-MAC** 與 **Microsoft 官方 T-MAC** 的比較流程、版本與結果。

> 完整結果請直接看 **[BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md)**。

## 名稱

- **本專案 T-MAC**：本 repository 自行實作的 x86 AVX2/F16C/OpenMP T-MAC。
- **Microsoft Official T-MAC**：pinned Microsoft T-MAC + TVM generated x86 kernel。
- **MKL FP16**：CPU dense FP16 GEMM reference。
- **cuBLAS FP16**：RTX 4070 SUPER GPU reference，不是 GPU T-MAC。

## 主要檔案

| Path | 用途 |
|---|---|
| `BENCHMARK_COMPARISON.md` | **完整 benchmark 主表與分析** |
| `adapters/profile_compare.py` | Official workload / CLI adapter |
| `scripts/tmac_env.sh` | Comparison environment |
| `scripts/run_vla_and_hackmd_compare.sh` | 本專案 VLA + official HackMD-safe |
| `scripts/run_official_overnight.sh` | Official VLA / HackMD runner |
| `provenance/versions.md` | Pinned upstream revisions |
| `results/summary/results_summary.txt` | Official compare 原始摘要 |

## Official baseline

External repository：

`~/tmac_external/T-MAC-official-x86`

Pinned T-MAC commit：

`7042f8f73330bd083bc1e4bc5ccb3f88a4904aee`

完整 dependency revisions 見：

`provenance/versions.md`

## Environment

```bash
source official_compare/scripts/tmac_env.sh
```

既有 TVM build 已驗證能從 relocated path 正常執行，但 generated CMake files 仍含舊 absolute path，因此目前視為 **frozen benchmark build**。

若未來要重建 TVM，使用新的 clean build directory，不在既有 build tree 上 incremental rebuild。

## Results

完整結果不在 README 重複貼一次，避免同一份數字維護兩份。

直接閱讀：

**[BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md)**

其中第一張表已列出：

- 全部 5 個 HackMD matrix shapes
- 1 / 2 / 4 / 8 / 16 CPU threads
- MKL FP16
- 本專案 T-MAC W2 / W3 / W4A16
- Microsoft Official W2
- RTX 4070 SUPER cuBLAS reference
- 每一列最快 CPU latency 直接加粗

## Running

```bash
official_compare/scripts/run_vla_and_hackmd_compare.sh
```

或：

```bash
official_compare/scripts/run_official_overnight.sh
```

新結果寫入：

`official_compare/results/runs/<RUN_ID>/`
