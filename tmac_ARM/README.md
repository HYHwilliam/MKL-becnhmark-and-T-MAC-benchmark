# T-MAC ARM CPU / NPU Benchmark

此目錄集中管理同一 ARM 平台上的 T-MAC CPU、T-MAC NPU 與兩者 benchmark。

## Structure

```text
tmac_ARM/
├── CPU/            ARM CPU T-MAC implementation
├── NPU/            ARM NPU T-MAC implementation
└── benchmark/      CPU / NPU 共用測試設定、腳本與結果
```

- `CPU/`：目前已完成的 W2/W3/W4-A16 scalar、NEON single-thread、multi-thread fixed 與 tuned 實作。
- `NPU/`：預留群聯板 NPU T-MAC implementation；取得實際 NPU SDK / runtime 後補入。
- `benchmark/`：集中管理 CPU / NPU 相同 shape 的測試設定與正式比較結果。

CPU 的完整 build、verify 與 benchmark 操作請見 `CPU/README.md`。
