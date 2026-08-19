# T-MAC ARM WnA16 Benchmark

本資料夾是目前正式 ARM benchmark 主線。

## 五組實作

```text
math_scalar_wxa16.cpp
tmac_scalar_wxa16.cpp
tmac_neon_st_wxa16.cpp
tmac_neon_mt_fixed_wxa16.cpp
tmac_neon_mt_tuned_wxa16.cpp
```

支援 W2A16 / W3A16 / W4A16。`N=1` 為 GEMV，較大的 N 為 GEMM。

- `math_scalar_wxa16.cpp`：dense scalar baseline，不使用 T-MAC、NEON、thread。
- `tmac_scalar_wxa16.cpp`：T-MAC scalar，保留 packing / LUT / scale / bias / bit-plane reconstruction。
- `tmac_neon_st_wxa16.cpp`：T-MAC NEON single-thread。g=4、activation group=64、weight scale group=128、KF=16。
- `tmac_neon_mt_fixed_wxa16.cpp`：fixed MT；`N / BN >= threads` 時平行 N，否則平行 M。
- `tmac_neon_mt_tuned_wxa16.cpp`：在相同 MT compute path 上加入 target-specific BM/BN schedule search。

## WSL Cross Compile

安裝：

```bash
sudo apt update
sudo apt install -y build-essential make g++ gcc-aarch64-linux-gnu g++-aarch64-linux-gnu qemu-user
```

前四組：

```bash
make cross
```

第五組：

```bash
make -f Makefile.tuned tuned-cross
```

產生：

```text
build/arm64/math_scalar_wxa16
build/arm64/tmac_scalar_wxa16
build/arm64/tmac_neon_st_wxa16
build/arm64/tmac_neon_mt_fixed_wxa16
build/arm64/tmac_neon_mt_tuned_wxa16
```

## QEMU Correctness

```bash
make qemu-verify BITS=2 THREADS=4
make qemu-verify BITS=3 THREADS=4
make qemu-verify BITS=4 THREADS=4
```

Tuned：

```bash
python3 verify_tuned_freeze.py
make -f Makefile.tuned qemu-smoke
```

完整 MT hardening：

```bash
./verify_mt_hardening.sh
```

Hardening 必須看到：

```text
HARDEN STvsMT=PASS mismatches=0 OutputCoverage=PASS unwritten=0 Guard=PASS
```

QEMU 只驗證 correctness。

## x86 Local Scalar

```bash
make local

./build/local/math_scalar_wxa16 --bits 2 --max-size 1024
./build/local/tmac_scalar_wxa16 --bits 2 --max-size 1024
```

## Native ARM

```bash
make
make -f Makefile.tuned tuned
```

輸出：

```text
build/native/math_scalar_wxa16
build/native/tmac_scalar_wxa16
build/native/tmac_neon_st_wxa16
build/native/tmac_neon_mt_fixed_wxa16
build/native/tmac_neon_mt_tuned_wxa16
```

正式 W2 benchmark：

```bash
./build/native/math_scalar_wxa16 --bits 2 --max-size 8192
./build/native/tmac_scalar_wxa16 --bits 2 --max-size 8192
./build/native/tmac_neon_st_wxa16 --bits 2 --max-size 8192
./build/native/tmac_neon_mt_fixed_wxa16 --bits 2 --threads 4 --max-size 8192
./build/native/tmac_neon_mt_tuned_wxa16 --bits 2 --threads 4 --max-size 8192
```

Tuned 正式設定預設為 `tune-number=10`、`tune-repeat=10`、`tune-cooldown-ms=100`，並應在實際 Phison ARM CPU 上重新 tune。

所有 executable 放在 `build/local/`、`build/arm64/`、`build/native/`，名稱與 `.cpp` stem 相同。`build/` 不提交 GitHub。
