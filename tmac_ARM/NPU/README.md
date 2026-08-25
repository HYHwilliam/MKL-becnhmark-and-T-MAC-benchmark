# T-MAC ARM NPU

此資料夾用於群聯 ARM 板上的 T-MAC NPU implementation。

```text
NPU/
├── src/        NPU implementation source
├── include/    project-local headers
├── scripts/    build / run / benchmark scripts
└── build/      local build output，不提交 Git
```

實際 source、compiler、runtime 與 build command 需依群聯板提供的 NPU SDK 決定。
