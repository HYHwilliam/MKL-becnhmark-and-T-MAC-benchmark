# Official Benchmark Result Inventory

> Snapshot：`official_2026-09-22`  
> Purpose：記錄 Microsoft Official T-MAC raw benchmark 的保存狀態與可用性，避免之後把缺失、partial 或 fallback 結果誤當成正式 benchmark。

## 1. Raw data location

主要保存位置：

```text
~/benchmark_project/official_compare/results/raw/official_2026-09-22/
```

獨立備份：

```text
~/benchmark_local_backup/2026-09-22-official-external/benchmark_results/
```

搬移後驗證：

```text
RAW files    = 198
BACKUP files = 198
diff -qr     = no differences
```

因此目前可以確認：

- raw snapshot 與獨立 backup 的 **198 個檔案一致**。
- 搬移過程沒有觀察到檔案遺失。
- 本文件描述的是「benchmark 是否可作正式比較」，不是「檔案是否存在」。

---

## 2. Status definition

| Status | 定義 |
|---|---|
| **VALID** | 正式結果可用；已存在 `results.csv`，相關 AutoTVM tuning audit 沒有失敗 record。 |
| **PARTIAL** | 同一 result family 只有部分 workload / shape 可用；失敗部分不得納入正式比較。 |
| **MISSING** | 沒有正式結果；不得自行補值或由其他 thread / shape 推算。 |
| **FALLBACK** | AutoTVM 找不到有效 schedule，執行使用 fallback configuration；該 latency 不納入正式比較。 |
| **VALIDATION ONLY** | smoke / correctness / workflow 驗證用途，不作正式 performance comparison。 |
| **HISTORICAL** | 舊測試或不完整測試，保留作歷史紀錄，但不作目前主比較資料。 |

---

## 3. Full AutoTVM audit summary

對所有現存 `tune.log` 的 JSON record 逐行解析後：

```text
TOTAL AutoTVM records = 1399
FAILED records        = 44
```

44 個 failed records 全部集中在 3 個 workload：

| Result directory | Failed task | Logical matrix | Failed records | Classification |
|---|---|---:|---:|---|
| `hackmd_official_w2_t1` | `qgemm_lut_t1_int8_m8192_k4096_n4096_b2` | `4096×4096×4096` W2 1T | 16/16 | **FALLBACK** |
| `hackmd_safe_w2_t1` | `qgemm_lut_t1_int8_m8192_k2048_n1024_b2` | `4096×1024×2048` W2 1T | 16/16 | **FALLBACK** |
| `hackmd_safe_w3_t1` | `qgemm_lut_t1_int8_m12288_k2048_n1024_b3` | `4096×1024×2048` W3 1T | 12/12 | **FALLBACK** |

其餘被 audit 到的 AutoTVM tuning records 均為 `failed=0`。

---

## 4. Official VLA

VLA 使用 8 個 workload / matrix shapes：

```text
2560×8×2560
2560×8×17920
1152×256×1152
2048×32×1024
256×32×1024
1024×32×2048
4096×32×1024
1024×32×4096
```

### W2

| Directory | Results | AutoTVM audit | Status |
|---|---:|---:|---|
| `vla_compare_w2_t1` | 8 rows | 77 records, 0 failed | **VALID** |
| `vla_compare_w2_t4` | 8 rows | 77 records, 0 failed | **VALID** |
| `vla_compare_w2_t8` | 8 rows | 77 records, 0 failed | **VALID** |

W2 total：

```text
24 benchmark rows
231 tuning records
0 failed
```

### W4

| Directory | Results | AutoTVM audit | Status |
|---|---:|---:|---|
| `vla_compare_w4_t1` | 8 rows | 84 records, 0 failed | **VALID** |
| `vla_compare_w4_t4` | 8 rows | 84 records, 0 failed | **VALID** |
| `vla_compare_w4_t8` | 8 rows | 84 records, 0 failed | **VALID** |

W4 total：

```text
24 benchmark rows
252 tuning records
0 failed
```

### VLA conclusion

**Official VLA W2 / W4、1T / 4T / 8T 全部可用於正式 comparison。**

Official VLA 合計：

```text
48 benchmark rows
483 tuning records
0 failed
```

---

## 5. Official N-scale 1024

### W2

| Directory | Results | AutoTVM audit | Status |
|---|---:|---:|---|
| `nscale1024_w2_t1` | no `results.csv` | no formal result | **MISSING** |
| `nscale1024_w2_t2` | 5 rows | 52 records, 0 failed | **VALID** |
| `nscale1024_w2_t4` | 5 rows | 52 records, 0 failed | **VALID** |
| `nscale1024_w2_t8` | 5 rows | 52 records, 0 failed | **VALID** |

`nscale1024_w2_t1` 的 directory 存在，但沒有正式 `results.csv`，因此不可自行補值。

### W3

| Directory | Results | AutoTVM audit | Status |
|---|---:|---:|---|
| `nscale1024_w3_t1` | 5 rows | 39 records, 0 failed | **VALID** |
| `nscale1024_w3_t2` | 5 rows | 39 records, 0 failed | **VALID** |
| `nscale1024_w3_t4` | 5 rows | 39 records, 0 failed | **VALID** |
| `nscale1024_w3_t8` | 5 rows | 39 records, 0 failed | **VALID** |

### W4

| Directory | Results | AutoTVM audit | Status |
|---|---:|---:|---|
| `nscale1024_w4_t1` | 5 rows | 52 records, 0 failed | **VALID** |
| `nscale1024_w4_t2` | 5 rows | 52 records, 0 failed | **VALID** |
| `nscale1024_w4_t4` | 5 rows | 52 records, 0 failed | **VALID** |
| `nscale1024_w4_t8` | 5 rows | 52 records, 0 failed | **VALID** |

---

## 6. Official HackMD-safe

`hackmd_safe` 的 logical matrix set：

```text
256×256×256
1024×1024×1024
4096×1024×2048
1024×1024×512
```

### W2

| Directory | Results | AutoTVM audit | Status |
|---|---:|---:|---|
| `hackmd_safe_w2_t1` | 4 rows | 60 records, 16 failed | **PARTIAL** |
| `hackmd_safe_w2_t2` | 4 rows | 60 records, 0 failed | **VALID** |
| `hackmd_safe_w2_t4` | 4 rows | 60 records, 0 failed | **VALID** |
| `hackmd_safe_w2_t8` | 4 rows | 60 records, 0 failed | **VALID** |

`hackmd_safe_w2_t1` 中：

```text
256×256×256          VALID
1024×1024×1024       VALID
4096×1024×2048       FALLBACK
1024×1024×512        VALID
```

失敗 workload：

```text
qgemm_lut_t1_int8_m8192_k2048_n1024_b2
```

logical matrix：

```text
4096×1024×2048
```

該 workload 的 16 個 tuning candidates 全部失敗，因此：

```text
Official W2 1T 4096×1024×2048 = 不可用
```

即使 `results.csv` 中存在 fallback latency，也不得放入正式 comparison。

目前 `BENCHMARK_COMPARISON.md` 應顯示：

```text
— (fallback)
```

### W3

| Directory | Results | AutoTVM audit | Status |
|---|---:|---:|---|
| `hackmd_safe_w3_t1` | 2 rows | 36 records, 12 failed | **PARTIAL** |

失敗 workload：

```text
qgemm_lut_t1_int8_m12288_k2048_n1024_b3
```

logical matrix：

```text
4096×1024×2048
```

12/12 tuning candidates 全部失敗。

因此 `hackmd_safe_w3_t1` 不可被描述為完整 W3 HackMD result。

目前也沒有完整的：

```text
hackmd_safe_w3_t2
hackmd_safe_w3_t4
hackmd_safe_w3_t8
hackmd_safe_w4_t1
hackmd_safe_w4_t2
hackmd_safe_w4_t4
hackmd_safe_w4_t8
```

正式 raw result。

---

## 7. Older HackMD official runs

### `hackmd_official_w2_t1`

AutoTVM audit：

```text
44 records
16 failed
```

失敗 workload：

```text
qgemm_lut_t1_int8_m8192_k4096_n4096_b2
```

logical matrix：

```text
4096×4096×4096
```

16/16 tuning candidates 全失敗，AutoTVM 後續使用 fallback configuration。

因此：

```text
Official W2 1T 4096×4096×4096 = FALLBACK / INVALID FOR FORMAL COMPARISON
```

整個 directory 保留為：

**HISTORICAL / PARTIAL**

### `hackmd_official_w2_t2`

```text
28 tuning records
0 failed
```

但這是舊的、不完整 HackMD run，不作目前正式主表來源。

Status：

**HISTORICAL**

---

## 8. Smoke / validation results

下列資料只作 workflow、kernel generation、preprocessing 或 basic execution validation：

```text
official_smoke
official_smoke_median
official_smoke_preprocessor
official_smoke_tuned_w2_t1

smoke_tuned_w2_t1
smoke_tuned_w2_t2
smoke_tuned_w2_t4
smoke_tuned_w2_t8

smoke_tuned_w3_t1
smoke_tuned_w3_t2
smoke_tuned_w3_t4
smoke_tuned_w3_t8

smoke_tuned_w4_t1
smoke_tuned_w4_t2
smoke_tuned_w4_t4
smoke_tuned_w4_t8
```

其中所有存在的 tuned smoke AutoTVM records 在完整 audit 中都是：

```text
failed = 0
```

但它們的用途仍是：

**VALIDATION ONLY**

不得與正式 VLA / N-scale / HackMD benchmark 混為主要性能結論。

---

## 9. Official overnight log

保存：

```text
official_overnight.log
```

其中可以確認：

- VLA W2 T1 / T4 / T8 tuning 全部完成。
- VLA W4 T1 / T4 / T8 tuning 全部完成。
- `4096×4096×4096` W2 1T 出現 `Could not find any valid schedule`。
- 該 workload 隨後出現 `A fallback configuration is used`。

因此 `4096×4096×4096` official latency 不納入正式 comparison。

---

## 10. Usage rules

之後整理 benchmark 時固定遵守：

1. **先查本檔案，再使用 raw result。**
2. `VALID` 才能直接進正式 comparison。
3. `PARTIAL` 必須逐 shape 判斷，不可把整個 directory 當 valid。
4. `MISSING` 不得補造或插值。
5. `FALLBACK` latency 不得當作 tuned Official T-MAC performance。
6. Smoke 結果只作 validation。
7. Raw data 不直接修改；任何重新整理結果都寫到 summary / comparison 檔案。
8. 若新增正式 benchmark，先保存 raw output，再更新本 inventory。
9. 若未來重新跑同一 workload，不覆蓋此 snapshot；應建立新的 run/snapshot directory。
10. `BENCHMARK_COMPARISON.md` 中的 Official 數字必須和本 inventory 的 status 一致。

---

## 11. Current formal-data checklist

目前可安全用於正式分析的主要資料：

```text
[VALID] Official VLA W2 1T / 4T / 8T
[VALID] Official VLA W4 1T / 4T / 8T

[MISSING] Official N-scale W2 1T
[VALID]   Official N-scale W2 2T / 4T / 8T
[VALID]   Official N-scale W3 1T / 2T / 4T / 8T
[VALID]   Official N-scale W4 1T / 2T / 4T / 8T

[PARTIAL] Official HackMD-safe W2 1T
[VALID]   Official HackMD-safe W2 2T / 4T / 8T
[PARTIAL] Official HackMD-safe W3 1T

[FALLBACK] Official W2 1T 4096×4096×4096
[FALLBACK] Official W2 1T 4096×1024×2048
[FALLBACK] Official W3 1T 4096×1024×2048

[VALIDATION ONLY] smoke / tuned smoke families
[HISTORICAL] older hackmd_official runs
```

這份 inventory 是 `official_2026-09-22` raw snapshot 的狀態記錄。若未來重新執行 benchmark，應另外建立新的 run/snapshot，並更新 inventory，而不是覆寫此處的歷史判定。
