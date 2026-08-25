# CPU / NPU Benchmark

此資料夾集中管理 ARM CPU T-MAC 與 NPU T-MAC 的公平比較。

```text
benchmark/
├── configs/             共用 GEMM / GEMV shape 與 benchmark 設定
├── scripts/             CPU / NPU 執行與結果比較腳本
└── results/
    ├── CPU/             CPU 原始正式結果
    ├── NPU/             NPU 原始正式結果
    └── comparison/      CPU vs NPU 對齊後的比較結果
```

CPU 與 NPU 應使用相同的 `M / K / N`、warmup、repeat 與明確定義的計時範圍。

比較時應區分：

```text
kernel-only vs kernel-only
end-to-end vs end-to-end
```
