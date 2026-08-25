# T-MAC ARM Multi-thread Correctness Hardening

這份驗證只檢查 multi-thread correctness，不修改 production 的 T-MAC packing、activation LUT、`reconstruct_lookup()`、`compute_tile()` 或 `tmac_neon_mt()`。

主要檢查：

- **Frozen ST ↔ MT bitwise differential**
  - multi-thread output 與 frozen single-thread NEON output 逐 FP16 bit 比較
  - 必須 `STvsMT=PASS`、`mismatches=0`

- **Output coverage**
  - output 先填 sentinel
  - 執行後不得留下未寫入位置
  - 必須 `OutputCoverage=PASS`、`unwritten=0`

- **Guard / canary**
  - output 前後保留 guard elements
  - 必須 `Guard=PASS`

- **K boundary**
  - 額外驗證 `K=384`、`K=640`

- **M partition boundary**
  - 驗證 `M tiles = threads-1 / threads / threads+1`

- **N parallel boundary**
  - 固定 `BN=8`
  - `N/BN < threads` 應切 M
  - `N/BN >= threads` 應切 N

原有 packing/layout、official LUT/reference、dense NMSE、BM/BN sweep 仍保留。

---

## 執行

完整 hardening：

```bash
./verify_mt_hardening.sh
```

固定驗證 W2/W3/W4 與 threads `1,2,4,8`。

只指定 bits：

```bash
BITS_LIST="2 3 4" ./verify_mt_hardening.sh
```

單次 cross-compile：

```bash
aarch64-linux-gnu-g++ -O3 -std=c++17 -Wall -Wextra -Wpedantic \
    -march=armv8.2-a+fp16 -pthread \
    tmac_neon_mt_fixed_wxa16.cpp \
    -o build/arm64/tmac_neon_mt_fixed_wxa16
```

QEMU：

```bash
qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu \
    ./build/arm64/tmac_neon_mt_fixed_wxa16 \
    --bits 2 --threads 4 --verify-only
```

hardening case 必須看到：

```text
HARDEN STvsMT=PASS mismatches=0 OutputCoverage=PASS unwritten=0 Guard=PASS
```

QEMU 僅用於 correctness；正式 performance 必須在真實 ARM target 上量測。
