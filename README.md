# T-MAC CPU Benchmark and Verification

本 repository 記錄 T-MAC 在 x86 與 ARM CPU 上的實作、正確性驗證與 benchmark。現在主要研究主線為 `tmac_ARM/CPU/`。

## 目前比較

ARM 端比較五種 W2/W3/W4-A16 實作：

1. Dense scalar
2. T-MAC scalar
3. T-MAC NEON single-thread
4. T-MAC NEON multi-thread fixed
5. T-MAC NEON multi-thread tuned

矩陣定義：

```text
W[M x K]
X[N x K]
Y[N x M] = X * W^T
```

`N=1` 為 GEMV，`N>1` 為 GEMM。

## Repository

```text
tmac_ARM/          ARM CPU / NPU T-MAC benchmark
  CPU/             ARM CPU T-MAC implementation
  NPU/             ARM NPU T-MAC implementation
  benchmark/       CPU / NPU comparison
x86_GEMM/          x86 AVX2 W2/W3/W4-A16 GEMM 驗證
experiments/
  LUT_GEMM_CPU/    LUT-GEMM scalar / AVX2 CPU microbenchmark
  MKL_INT8/        oneMKL INT8 correctness investigation
```

## WSL 安裝 ARM cross compiler / QEMU

```bash
sudo apt update
sudo apt install -y build-essential make g++ gcc-aarch64-linux-gnu g++-aarch64-linux-gnu qemu-user
```

確認：

```bash
aarch64-linux-gnu-g++ --version
qemu-aarch64 --version
```

## ARM Cross Compile

```bash
cd tmac_ARM/CPU
make cross
make -f Makefile.tuned tuned-cross
```

輸出：

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

python3 scripts/verify_tuned_freeze.py
make -f Makefile.tuned qemu-smoke
```

完整 MT hardening：

```bash
./scripts/verify_mt_hardening.sh
```

QEMU 僅用於 correctness，不使用其 latency 當作 ARM 實機效能。

## Native ARM / Phison

```bash
cd tmac_ARM/CPU
make
make -f Makefile.tuned tuned
```

例如 W2：

```bash
./build/native/math_scalar_wxa16 --bits 2 --max-size 8192
./build/native/tmac_scalar_wxa16 --bits 2 --max-size 8192
./build/native/tmac_neon_st_wxa16 --bits 2 --max-size 8192
./build/native/tmac_neon_mt_fixed_wxa16 --bits 2 --threads 4 --max-size 8192
./build/native/tmac_neon_mt_tuned_wxa16 --bits 2 --threads 4 --max-size 8192
```

W3/W4 將 `--bits` 改成 3 / 4。Tuned schedule 必須在實際目標 ARM CPU 上重新搜尋。

Binary、`.bin`、log、QEMU build output 與本機 `tmac_ARM_test/` 由 `.gitignore` 排除，不提交到 GitHub。
