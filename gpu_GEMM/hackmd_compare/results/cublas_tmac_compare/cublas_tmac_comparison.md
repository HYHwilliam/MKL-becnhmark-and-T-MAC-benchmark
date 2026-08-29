# cuBLAS GPU vs T-MAC CPU

## Environment and Method

- CPU: `AMD Ryzen 5 8500G`
- GPU: `NVIDIA GeForce RTX 4070 SUPER`
- Shape notation: `M×N×K`
- T-MAC: `W2A16 / W3A16 / W4A16`, AVX2/F16C/OpenMP.
- cuBLAS: dense FP16 input/output with FP32 accumulation.
- Both summaries use the median of five independent run-level medians.
- Primary CPU comparison selects the fastest T-MAC result among `1T / 2T / 4T / 8T`.
- 16T is retained separately but excluded from the primary CPU selection because the Ryzen exposes 12 logical CPUs.
- GPU speedup = T-MAC CPU latency / cuBLAS GPU latency.

> This is a cross-hardware performance reference, not a same-hardware or equal-precision kernel comparison. T-MAC total_ms includes preprocessing + kernel, while cuBLAS timing measures the warmed-up GPU GEMM with H2D/D2H excluded.

## cuBLAS Five-Run Statistics

| Shape | Median ms | GFLOPs | CV% | P90 median ms | Max verification error |
|---|---:|---:|---:|---:|---:|
| 256x256x256 | 0.006912 | 4854.519 | 6.814 | 0.007168 | 0.000000 |
| 1024x1024x1024 | 0.055104 | 38971.466 | 0.969 | 0.055296 | 0.003906 |
| 4096x4096x4096 | 1.985536 | 69220.076 | 0.020 | 1.986560 | 0.007812 |
| 4096x1024x2048 | 0.277504 | 61908.546 | 0.000 | 0.278496 | 0.000000 |
| 1024x1024x512 | 0.032768 | 32768.000 | 0.000 | 0.033792 | 0.000000 |

## GPU vs Best Primary T-MAC

| Backend | Shape | Best CPU T | T-MAC ms | cuBLAS ms | GPU speedup |
|---|---|---:|---:|---:|---:|
| T-MAC W2A16 | 256x256x256 | 4 | 0.160773 | 0.006912 | 23.26× |
| T-MAC W2A16 | 1024x1024x1024 | 4 | 6.453994 | 0.055104 | 117.12× |
| T-MAC W2A16 | 4096x4096x4096 | 8 | 360.971359 | 1.985536 | 181.80× |
| T-MAC W2A16 | 4096x1024x2048 | 8 | 45.636620 | 0.277504 | 164.45× |
| T-MAC W2A16 | 1024x1024x512 | 4 | 3.284623 | 0.032768 | 100.24× |
| T-MAC W3A16 | 256x256x256 | 4 | 0.207024 | 0.006912 | 29.95× |
| T-MAC W3A16 | 1024x1024x1024 | 4 | 9.157900 | 0.055104 | 166.19× |
| T-MAC W3A16 | 4096x4096x4096 | 8 | 527.235345 | 1.985536 | 265.54× |
| T-MAC W3A16 | 4096x1024x2048 | 8 | 66.219955 | 0.277504 | 238.63× |
| T-MAC W3A16 | 1024x1024x512 | 4 | 4.643847 | 0.032768 | 141.72× |
| T-MAC W4A16 | 256x256x256 | 4 | 0.249835 | 0.006912 | 36.15× |
| T-MAC W4A16 | 1024x1024x1024 | 4 | 11.834163 | 0.055104 | 214.76× |
| T-MAC W4A16 | 4096x4096x4096 | 8 | 703.746787 | 1.985536 | 354.44× |
| T-MAC W4A16 | 4096x1024x2048 | 8 | 87.337454 | 0.277504 | 314.73× |
| T-MAC W4A16 | 1024x1024x512 | 8 | 5.941969 | 0.032768 | 181.33× |

## Backend Summary

| Backend | Geomean GPU speedup vs best 1/2/4/8T CPU | Minimum | Maximum |
|---|---:|---:|---:|
| T-MAC W2A16 | 96.03× | 23.26× | 181.80× |
| T-MAC W3A16 | 134.92× | 29.95× | 265.54× |
| T-MAC W4A16 | 173.46× | 36.15× | 354.44× |

## Interpretation

- cuBLAS represents an optimized NVIDIA GPU dense-GEMM baseline; it is not GPU T-MAC.
- T-MAC uses low-bit weights and FP16 activations, while cuBLAS uses dense FP16 operands with FP32 accumulation.
- Therefore the speedup values answer `how much faster this GPU dense-GEMM reference is than this CPU T-MAC implementation`, not which algorithm is intrinsically better under identical hardware and precision.
- The detailed all-thread comparison is available in `cublas_tmac_all_threads.csv`.