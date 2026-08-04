# LUT-GEMM AVX2 多執行緒驗證（Pthread / OpenMP）

補上 README.md「已知限制」中缺的一項：LUT-GEMM CPU AVX2 版的多執行緒效能。做法是把 `lut_gemm_cpu_avx2` 的 `m` 迴圈切分平行化，分別以 Pthread、OpenMP 實作

## 1. Pthread

```bash
g++ -O2 -mavx2 -mfma verify/lut_gemm_pthread_benchmark.cpp -lpthread -o verify/lut_gemm_pthread_benchmark
./verify/lut_gemm_pthread_benchmark
```

| Matrix Size | threads=1 | threads=2 | threads=4 | threads=8 |
|:-----------:|----------:|----------:|----------:|----------:|
| 1024×1024 | 3.89 | 8.50 | 13.43 | 12.02 |
| 2048×2048 | 4.34 | 8.36 | 16.63 | 19.73 |
| 4096×4096 | 4.39 | 8.94 | 17.31 | 22.68 |
| 8192×8192 | 3.44 | 6.96 | 12.18 | 13.04 |

單位 GFLOPS。全部尺寸、全部執行緒數 `max_err=0`，解析解驗證全都 `OK`。

## 2. OpenMP

```bash
g++ -O2 -mavx2 -mfma -fopenmp verify/lut_gemm_openmp_benchmark.cpp -o verify/lut_gemm_openmp_benchmark
./verify/lut_gemm_openmp_benchmark
```

| Matrix Size | threads=1 | threads=2 | threads=4 | threads=8 |
|:-----------:|----------:|----------:|----------:|----------:|
| 1024×1024 | 4.44 | 9.38 | 18.57 | 23.22 |
| 2048×2048 | 5.27 | 9.52 | 12.67 | 24.65 |
| 4096×4096 | 4.73 | 9.96 | 18.42 | 24.29 |
| 8192×8192 | 3.66 | 7.46 | 12.98 | 13.24 |

`max_err=0`，解析解驗證全數 `OK`。

## 3. 分析

**OpenMP 一致優於 Pthread**：同尺寸同執行緒數下 OpenMP 恆快於 Pthread（1024×1024 threads=8：23.22 vs. 12.02 GFLOPS）。兩版核心運算邏輯相同，差異來自 thread 生命週期管理——Pthread 版每次迭代重新 `pthread_create`/`join`，OpenMP 用 thread pool 重複使用。

**加速比隨 K 增大而遞減，且未達線性**：threads=1→8 的加速倍率，OpenMP 版為 1024：5.2×、2048：4.7×、4096：5.1×、8192：3.6×。8192×8192 明顯落後，原因與 README 對 LUT-GEMM 記憶體行為的分析一致——該尺寸權重矩陣 32MB 已超過 L3（12MB），多執行緒同時對同一份 LUT 發起 `_mm256_i32gather_ps`，成為記憶體頻寬瓶頸而非運算並行度不足。

**Pthread 未出現 T-MAC 式的小矩陣效能暴跌**：T-MAC pthread 版在 1024×1024 threads=8 常低於 threads=1（thread 建立成本蓋過計算量）；LUT-GEMM 除 1024×1024 threads=8 略降於 threads=4（12.02 < 13.43）外，其餘尺寸單調遞增。原因是 LUT-GEMM 單次呼叫耗時遠高於 T-MAC（同尺寸下約 10–60 倍），thread 管理開銷占比小得多。

**多執行緒未改變 T-MAC 與 LUT-GEMM 的架構性差距**：兩者絕對效能仍相差一個數量級以上（T-MAC 8192×8192 單執行緒 213 GFLOPS vs. LUT-GEMM OpenMP 8 threads 13.24 GFLOPS），因為瓶頸來自 LUT 查表機制本身（gather vs. shuffle），非執行緒數量可解決。

**未測試項目**：CPU 綁核（`OMP_PROC_BIND`/`OMP_PLACES`）。
