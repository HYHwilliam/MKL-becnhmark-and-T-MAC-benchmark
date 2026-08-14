# T-MAC ARM NEON 實作說明

## 1. 程式定位

`t_mac_neon_benchmark.cpp` 是 Microsoft T-MAC 核心演算法的獨立 AArch64 NEON 實作，用來驗證及測試 W2A16、W3A16、W4A16 的 GEMV/GEMM。它採用 T-MAC 的 bit-plane、INT8 LUT、離線權重重排與 SIMD table lookup，不是另一套 LUT-GEMM 實作。

目前固定使用 `g=4`、16-entry INT8 LUT、FP16 activation/output、activation group size 64、weight scale group size 128、`kfactor=16`、`fast_aggregation=false`、`zero_point=false`。其中 A16 代表輸入 activation 為 FP16；最後以 FP32 合併 bit planes 再轉回 FP16，這也符合官方 `qgemm.py` 的處理方式，不能因此稱為 A32。

## 2. 主要運算流程

1. `pack_weights_tmac()` 將 unsigned W2/W3/W4 權重拆成 bit planes，每四個 K 維度的 bit 組成一個 4-bit LUT index，再依官方 `[M tile][K tile][block32][kfactor][16 lanes]` 配置將兩個 index 存入同一 byte 的低、高 nibble。
2. `build_activation_luts()` 每 64 個 FP16 activation 建立 16-entry signed LUT。LUT scale 取四個 activation 絕對值和的最大值再除以 127，LUT 內容以 round-to-nearest 轉成 INT16，最後飽和縮窄成 INT8。
3. `tmac_gemm_a16()` 從 packed weight 取出低、高 nibble，使用 `vqtbl1q_s8` 查詢 NEON register 內的 16-entry LUT，再將 INT8 lookup 結果擴展成 INT16 累加。
4. lookup sum 轉成 FP16 後乘回 LUT scale。bit-plane 0 額外加入 `lut_bias=-sum(activation)`，接著乘上每 128 個 K 共用的 weight scale。
5. 最後以官方係數 `(1/2, 1, 2, 4)` 合併各 bit plane，並輸出 FP16 結果。

重要 NEON 指令包括：`vld4q_f16` 交錯載入 activation、`vabsq_f16`/`vmaxvq_f16` 計算 LUT scale、`vcvtnq_s16_f16`/`vqmovn_s16` 量化 LUT、`vqtbl1q_s8` 查表、`vmovl_s8` 與 `vaddq_s16` 執行 signed widening accumulation，以及 `vfmaq_n_f16`/`vfmaq_f16` 執行 FP16 scale 與累加。

## 3. 與官方 T-MAC 的差異

本程式對應官方預設的 W2/W3/W4 A16 路徑，但不是完整 T-MAC framework。官方透過 TVM 產生並 autotune kernel，本程式則固定 W3 使用 `bm=192`、W2/W4 使用 `bm=256`，並固定 `kfactor=16`；這些值都在官方候選值內，且預設 `g=4`、activation group 64 時，`kfactor=8` 不符合分組條件，因此應使用 16。

本程式尚未實作 W1、custom zero point、BitNet unified scale、fast aggregation、其他 group size、TVM autotuning、`bn` tiling 與多執行緒。GEMM 的 N 維度目前由外層迴圈處理，數學結果相同，但效能排程不等同官方完整 GEMM kernel。

官方 ARM `tbl.cc` 以 `a*b+c` 向量運算式表達 scaling，本實作明確使用 `vfmaq_*_f16` 固定 FMA 舍入行為。兩者數學公式一致；顯式 FMA 只做一次舍入，不是降低精度，但不屬於官方原始碼的逐字翻寫。

比對基準為 Microsoft/T-MAC commit `7042f8f73330bd083bc1e4bc5ccb3f88a4904aee`：[`qgemm.py`](https://github.com/microsoft/T-MAC/blob/7042f8f73330bd083bc1e4bc5ccb3f88a4904aee/python/t_mac/ops/qgemm.py)、[`weights.py`](https://github.com/microsoft/T-MAC/blob/7042f8f73330bd083bc1e4bc5ccb3f88a4904aee/python/t_mac/weights.py)、[`lut_ctor.cc`](https://github.com/microsoft/T-MAC/blob/7042f8f73330bd083bc1e4bc5ccb3f88a4904aee/python/t_mac/intrins/lut_ctor.cc)、[`tbl.cc`](https://github.com/microsoft/T-MAC/blob/7042f8f73330bd083bc1e4bc5ccb3f88a4904aee/python/t_mac/intrins/tbl.cc)。

## 4. 驗證方式

`--verify-only` 會同時檢查 packed layout、NEON kernel 與獨立 packed reference 的 FP16 bit pattern、非有限值，以及對 dense GEMM reference 的 NMSE。通過條件是 packing 正確、`kernel_bit_mismatches=0`、`non_finite=0` 且 `nmse<=5e-4`；相較官方只輸出 NMSE warning，本程式會在任一條件不符時回傳失敗。

```bash
qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu ./t_mac_neon_benchmark_arm64 --verify-only
```

不加 `--verify-only` 會先執行驗證，再執行 W4A16 GEMV benchmark。QEMU 適合驗證功能，輸出的 latency/GOPS 不代表真實 ARM 硬體效能。

## 5. 驗證結果分析

目前在 QEMU AArch64 上完成 W2A16、W3A16、W4A16 各 9 組、合計 27 組測試，包含 GEMV、不同 N 的 GEMM、多個 M/K tiles，以及六種邊界資料模式。27 組結果全部為 `PASS`，而且每組都是 `packing=PASS`、`kernel_bit_mismatches=0`、`signed_zero_mismatches=0`、`max_packed_ulp=0`、`max_packed_abs_error=0`、`non_finite=0`，表示 NEON kernel 與逐項讀取 packed layout 的 scalar reference 在 FP16 bit pattern 上完全相同，並非靠放寬誤差門檻通過。

| 類型 | 最大 NMSE | 最大 dense absolute error | 判定 |
|---|---:|---:|---|
| W2A16 | `6.01289e-05` | `0.132812` | PASS |
| W3A16 | `2.76115e-04` | `0.3125` | PASS |
| W4A16 | `7.87920e-05` | `0.582031` | PASS |

全體最大 NMSE 是 W3A16 mode 5 的 `2.76115e-04`，約為官方 `5e-4` 上限的 55.2%。W4A16 mode 6 雖然出現 `max_dense_abs_error=0.582031`，但該模式將所有 weight scales 提高為 1，使輸出數值及絕對誤差一起放大；其 NMSE 仍只有 `3.11612e-05`，且 kernel 與 packed reference 保持逐位元相同，因此這不是 NEON indexing、packing 或 kernel 計算錯誤，而是 INT8 LUT 量化與 FP16 累加相對於 dense reference 的預期近似誤差。若實際應用另外要求單一輸出不得超過特定 absolute error，仍應依模型輸出尺度增加獨立門檻，不能只看官方 NMSE。

六種特殊模式分別測試全零 activation、正負交錯 activation、全最小權重碼、全最大權重碼、最小/最大權重碼交錯，以及 weight scale 等於 1。這些案例涵蓋零值、符號、nibble 邊界、bit-plane 極值與放大誤差情況；另外，W2/W3/W4 的 packed weights 與 scales 也已使用官方 `weights.py::preprocess_weights()` 交叉比較，在 18 組不同 M/K 配置下達到 weight bytes 與 FP16 scale bits 完全一致。

## 6. 編譯與執行

若 WSL 主機是 Intel/AMD x86-64，不能直接用一般 `g++ -march=armv8.2-a+fp16`，必須安裝 AArch64 交叉編譯器及 QEMU：

```bash
sudo apt update
sudo apt install g++-aarch64-linux-gnu qemu-user
cd ~/benchmark_project/tmac_ARM
aarch64-linux-gnu-g++ -O3 -std=c++17 -march=armv8.2-a+fp16 t_mac_neon_benchmark.cpp -o t_mac_neon_benchmark_arm64
qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu ./t_mac_neon_benchmark_arm64 --verify-only
```

在原生 AArch64 Linux 裝置上可直接使用：

```bash
g++ -O3 -std=c++17 -march=armv8.2-a+fp16 t_mac_neon_benchmark.cpp -o t_mac_neon_benchmark
./t_mac_neon_benchmark --verify-only
```

目前輸入限制是 M 必須符合所選 expanded-M tile，K 必須是 128 的倍數，並且 CPU 必須支援 ARMv8.2-A FP16 vector arithmetic。若自行更改來源檔名，編譯指令也必須使用相同名稱。
