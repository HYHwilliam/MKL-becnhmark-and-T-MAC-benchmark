# T-MAC NEON MT Tuned

第五組 benchmark：`src/tmac_neon_mt_tuned_wxa16.cpp`

用途是比較：

```text
T-MAC NEON MT fixed
vs
T-MAC NEON MT tuned
```

兩者使用相同：
- T-MAC packing / LUT arithmetic
- `compute_tile()`
- `tmac_neon_mt()`
- `StaticThreadPool`
- N/M parallel rule
- FP16 reconstruction
- bit-plane reduction

Tuned 版本只額外搜尋合法 schedule，找出 target CPU 上 latency 最低的 BM/BN/KF。

目前設定下 KF 實際固定為 16，因此主要 tuning 變數是 BM 與 BN。

---

## Tuning policy

每個合法 candidate：

1. discarded warmup 1 次
2. 執行 `number × repeat`
3. 使用 repeat cost 的 mean 作為 score
4. 選 mean latency 最低者

預設：

```text
tune-number      = 10
tune-repeat      = 10
tune-cooldown-ms = 100
```

Tune 階段只量 qgemm compute；activation LUT 先建立完成。

正式 benchmark latency 仍包含：

```text
activation LUT construction
+
parallel compute
```

不包含：
- offline weight/scale packing
- tuning 本身花費時間
- thread-pool construction

---

## 編譯

WSL cross-compile：

```bash
aarch64-linux-gnu-g++ -O3 -std=c++17 -Wall -Wextra -Wpedantic \
    -march=armv8.2-a+fp16 -pthread \
    src/tmac_neon_mt_tuned_wxa16.cpp \
    -o build/arm64/tmac_neon_mt_tuned_wxa16
```

或：

```bash
make -f Makefile.tuned tuned-cross
```

真實 ARM / Phison：

```bash
g++ -O3 -std=c++17 -Wall -Wextra -Wpedantic \
    -march=armv8.2-a+fp16 -pthread \
    src/tmac_neon_mt_tuned_wxa16.cpp \
    -o build/native/tmac_neon_mt_tuned_wxa16
```

或：

```bash
make -f Makefile.tuned tuned
```

---

## 驗證

確認 tuned 版本沒有改 frozen compute：

```bash
python3 scripts/verify_tuned_freeze.py
```

QEMU smoke：

```bash
make -f Makefile.tuned qemu-smoke
```

QEMU tuning winner 只用於功能確認，不保存成 Phison schedule。

---

## 正式執行

```bash
./build/native/tmac_neon_mt_tuned_wxa16 --bits 2 --threads 4 --max-size 8192
./build/native/tmac_neon_mt_tuned_wxa16 --bits 3 --threads 4 --max-size 8192
./build/native/tmac_neon_mt_tuned_wxa16 --bits 4 --threads 4 --max-size 8192
```

快速 sanity check：

```bash
./build/native/tmac_neon_mt_tuned_wxa16 \
    --bits 2 \
    --threads 4 \
    --max-size 1024 \
    --tune-number 2 \
    --tune-repeat 2 \
    --tune-cooldown-ms 0
```

正式 Phison benchmark 使用預設 10×10 tuning；不要沿用 QEMU 或其他 CPU 的 tuned schedule。
