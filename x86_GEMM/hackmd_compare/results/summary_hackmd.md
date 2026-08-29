# HackMD Benchmark Summary

Shape notation: **M×N×K**

Final `Total (ms)` = median of five independent run-level medians.
Final `GFLOPs` is recomputed from `2×M×N×K / median latency`.

## Validation

- Status: **PASS**
- Expected logs: 20, each with 25 RESULT rows.
- Expected total RESULT rows: 500.
- WARNING: High variation: MKL 256x256x256 1T, CV=173.2%.
- WARNING: High variation: MKL 256x256x256 2T, CV=140.5%.
- WARNING: High variation: MKL 256x256x256 4T, CV=128.1%.
- WARNING: High variation: MKL 256x256x256 8T, CV=147.5%.
- WARNING: High variation: MKL 256x256x256 16T, CV=115.9%.
- WARNING: High variation: MKL 1024x1024x1024 2T, CV=23.2%.
- WARNING: High variation: MKL 1024x1024x1024 4T, CV=28.1%.
- WARNING: High variation: MKL 1024x1024x1024 8T, CV=22.9%.
- WARNING: High variation: MKL 1024x1024x1024 16T, CV=37.7%.
- WARNING: High variation: MKL 4096x1024x2048 4T, CV=26.1%.
- WARNING: High variation: MKL 4096x1024x2048 16T, CV=20.7%.
- WARNING: High variation: MKL 1024x1024x512 1T, CV=55.5%.
- WARNING: High variation: MKL 1024x1024x512 2T, CV=55.2%.
- WARNING: High variation: MKL 1024x1024x512 4T, CV=56.8%.
- WARNING: High variation: MKL 1024x1024x512 8T, CV=60.2%.
- WARNING: High variation: MKL 1024x1024x512 16T, CV=33.8%.
- WARNING: High variation: T-MAC W2A16 256x256x256 4T, CV=26.1%.
- WARNING: High variation: T-MAC W2A16 1024x1024x512 1T, CV=33.8%.
- WARNING: High variation: T-MAC W2A16 1024x1024x512 2T, CV=38.0%.
- WARNING: High variation: T-MAC W2A16 1024x1024x512 4T, CV=37.9%.
- WARNING: High variation: T-MAC W2A16 1024x1024x512 8T, CV=45.2%.
- WARNING: High variation: T-MAC W3A16 256x256x256 2T, CV=33.8%.
- WARNING: High variation: T-MAC W3A16 256x256x256 4T, CV=20.6%.
- WARNING: High variation: T-MAC W3A16 256x256x256 8T, CV=21.2%.
- WARNING: High variation: T-MAC W3A16 4096x4096x4096 1T, CV=26.4%.
- WARNING: High variation: T-MAC W3A16 1024x1024x512 1T, CV=20.3%.
- WARNING: High variation: T-MAC W4A16 1024x1024x512 8T, CV=64.3%.

## HackMD-format Results

| Backend | Dtype | Category | Shape | Threads | Total (ms) | GFLOPs |
|---|---|---|---|---:|---:|---:|
| MKL | fp16 | Small Square | 256x256x256 | 1 | 0.135663 | 247.336650 |
| MKL | fp16 | Small Square | 256x256x256 | 2 | 0.128059 | 262.023224 |
| MKL | fp16 | Small Square | 256x256x256 | 4 | 0.124377 | 269.780040 |
| MKL | fp16 | Small Square | 256x256x256 | 8 | 0.113813 | 294.820732 |
| MKL | fp16 | Small Square | 256x256x256 | 16 | 0.129488 | 259.131595 |
| MKL | fp16 | Medium Square | 1024x1024x1024 | 1 | 7.219003 | 297.476486 |
| MKL | fp16 | Medium Square | 1024x1024x1024 | 2 | 9.509686 | 225.820668 |
| MKL | fp16 | Medium Square | 1024x1024x1024 | 4 | 7.205394 | 298.038337 |
| MKL | fp16 | Medium Square | 1024x1024x1024 | 8 | 9.024728 | 237.955498 |
| MKL | fp16 | Medium Square | 1024x1024x1024 | 16 | 7.303773 | 294.023876 |
| MKL | fp16 | Large Square | 4096x4096x4096 | 1 | 331.529382 | 414.560401 |
| MKL | fp16 | Large Square | 4096x4096x4096 | 2 | 320.147220 | 429.299225 |
| MKL | fp16 | Large Square | 4096x4096x4096 | 4 | 344.963007 | 398.416499 |
| MKL | fp16 | Large Square | 4096x4096x4096 | 8 | 324.934291 | 422.974605 |
| MKL | fp16 | Large Square | 4096x4096x4096 | 16 | 325.084963 | 422.778563 |
| MKL | fp16 | Rectangular | 4096x1024x2048 | 1 | 51.362234 | 334.484462 |
| MKL | fp16 | Rectangular | 4096x1024x2048 | 2 | 59.343846 | 289.497064 |
| MKL | fp16 | Rectangular | 4096x1024x2048 | 4 | 49.865959 | 344.520982 |
| MKL | fp16 | Rectangular | 4096x1024x2048 | 8 | 51.740336 | 332.040155 |
| MKL | fp16 | Rectangular | 4096x1024x2048 | 16 | 50.053476 | 343.230292 |
| MKL | fp16 | Medium Rectangular | 1024x1024x512 | 1 | 4.775016 | 224.866644 |
| MKL | fp16 | Medium Rectangular | 1024x1024x512 | 2 | 3.488027 | 307.836443 |
| MKL | fp16 | Medium Rectangular | 1024x1024x512 | 4 | 3.464960 | 309.885778 |
| MKL | fp16 | Medium Rectangular | 1024x1024x512 | 8 | 3.475464 | 308.949200 |
| MKL | fp16 | Medium Rectangular | 1024x1024x512 | 16 | 5.068800 | 211.833535 |
| T-MAC W2A16 | W2A16 | Small Square | 256x256x256 | 1 | 0.459401 | 73.039528 |
| T-MAC W2A16 | W2A16 | Small Square | 256x256x256 | 2 | 0.299541 | 112.019496 |
| T-MAC W2A16 | W2A16 | Small Square | 256x256x256 | 4 | 0.340815 | 98.453507 |
| T-MAC W2A16 | W2A16 | Small Square | 256x256x256 | 8 | 0.287962 | 116.523819 |
| T-MAC W2A16 | W2A16 | Small Square | 256x256x256 | 16 | 0.289219 | 116.017385 |
| T-MAC W2A16 | W2A16 | Medium Square | 1024x1024x1024 | 1 | 22.822459 | 94.095191 |
| T-MAC W2A16 | W2A16 | Medium Square | 1024x1024x1024 | 2 | 12.999925 | 165.192003 |
| T-MAC W2A16 | W2A16 | Medium Square | 1024x1024x1024 | 4 | 7.517491 | 285.664944 |
| T-MAC W2A16 | W2A16 | Medium Square | 1024x1024x1024 | 8 | 8.866852 | 242.192342 |
| T-MAC W2A16 | W2A16 | Medium Square | 1024x1024x1024 | 16 | 7.772283 | 276.300239 |
| T-MAC W2A16 | W2A16 | Large Square | 4096x4096x4096 | 1 | 1664.725245 | 82.559542 |
| T-MAC W2A16 | W2A16 | Large Square | 4096x4096x4096 | 2 | 735.940318 | 186.752852 |
| T-MAC W2A16 | W2A16 | Large Square | 4096x4096x4096 | 4 | 381.560458 | 360.202297 |
| T-MAC W2A16 | W2A16 | Large Square | 4096x4096x4096 | 8 | 320.573850 | 428.727900 |
| T-MAC W2A16 | W2A16 | Large Square | 4096x4096x4096 | 16 | 435.729368 | 315.422745 |
| T-MAC W2A16 | W2A16 | Rectangular | 4096x1024x2048 | 1 | 184.947226 | 92.890656 |
| T-MAC W2A16 | W2A16 | Rectangular | 4096x1024x2048 | 2 | 91.515319 | 187.726704 |
| T-MAC W2A16 | W2A16 | Rectangular | 4096x1024x2048 | 4 | 49.083188 | 350.015349 |
| T-MAC W2A16 | W2A16 | Rectangular | 4096x1024x2048 | 8 | 45.986719 | 373.583277 |
| T-MAC W2A16 | W2A16 | Rectangular | 4096x1024x2048 | 16 | 43.477866 | 395.140580 |
| T-MAC W2A16 | W2A16 | Medium Rectangular | 1024x1024x512 | 1 | 12.618232 | 85.094475 |
| T-MAC W2A16 | W2A16 | Medium Rectangular | 1024x1024x512 | 2 | 6.804307 | 157.803260 |
| T-MAC W2A16 | W2A16 | Medium Rectangular | 1024x1024x512 | 4 | 4.185023 | 256.567724 |
| T-MAC W2A16 | W2A16 | Medium Rectangular | 1024x1024x512 | 8 | 4.609484 | 232.941870 |
| T-MAC W2A16 | W2A16 | Medium Rectangular | 1024x1024x512 | 16 | 4.487577 | 239.269838 |
| T-MAC W3A16 | W3A16 | Small Square | 256x256x256 | 1 | 0.650710 | 51.565877 |
| T-MAC W3A16 | W3A16 | Small Square | 256x256x256 | 2 | 0.394522 | 85.050851 |
| T-MAC W3A16 | W3A16 | Small Square | 256x256x256 | 4 | 0.448772 | 74.769442 |
| T-MAC W3A16 | W3A16 | Small Square | 256x256x256 | 8 | 0.379619 | 88.389759 |
| T-MAC W3A16 | W3A16 | Small Square | 256x256x256 | 16 | 0.380427 | 88.202026 |
| T-MAC W3A16 | W3A16 | Medium Square | 1024x1024x1024 | 1 | 33.349888 | 64.392530 |
| T-MAC W3A16 | W3A16 | Medium Square | 1024x1024x1024 | 2 | 17.684426 | 121.433608 |
| T-MAC W3A16 | W3A16 | Medium Square | 1024x1024x1024 | 4 | 10.349298 | 207.500417 |
| T-MAC W3A16 | W3A16 | Medium Square | 1024x1024x1024 | 8 | 13.067544 | 164.337204 |
| T-MAC W3A16 | W3A16 | Medium Square | 1024x1024x1024 | 16 | 10.281066 | 208.877528 |
| T-MAC W3A16 | W3A16 | Large Square | 4096x4096x4096 | 1 | 2262.284820 | 60.752277 |
| T-MAC W3A16 | W3A16 | Large Square | 4096x4096x4096 | 2 | 1119.598873 | 122.757317 |
| T-MAC W3A16 | W3A16 | Large Square | 4096x4096x4096 | 4 | 631.802582 | 217.534650 |
| T-MAC W3A16 | W3A16 | Large Square | 4096x4096x4096 | 8 | 529.392155 | 259.616528 |
| T-MAC W3A16 | W3A16 | Large Square | 4096x4096x4096 | 16 | 599.434801 | 229.280905 |
| T-MAC W3A16 | W3A16 | Rectangular | 4096x1024x2048 | 1 | 275.046955 | 62.461587 |
| T-MAC W3A16 | W3A16 | Rectangular | 4096x1024x2048 | 2 | 131.710203 | 130.436889 |
| T-MAC W3A16 | W3A16 | Rectangular | 4096x1024x2048 | 4 | 73.906900 | 232.452845 |
| T-MAC W3A16 | W3A16 | Rectangular | 4096x1024x2048 | 8 | 73.204279 | 234.683948 |
| T-MAC W3A16 | W3A16 | Rectangular | 4096x1024x2048 | 16 | 66.658447 | 257.729815 |
| T-MAC W3A16 | W3A16 | Medium Rectangular | 1024x1024x512 | 1 | 17.832168 | 60.213757 |
| T-MAC W3A16 | W3A16 | Medium Rectangular | 1024x1024x512 | 2 | 9.470799 | 113.373943 |
| T-MAC W3A16 | W3A16 | Medium Rectangular | 1024x1024x512 | 4 | 5.902071 | 181.926280 |
| T-MAC W3A16 | W3A16 | Medium Rectangular | 1024x1024x512 | 8 | 6.237089 | 172.154321 |
| T-MAC W3A16 | W3A16 | Medium Rectangular | 1024x1024x512 | 16 | 5.692750 | 188.615664 |
| T-MAC W4A16 | W4A16 | Small Square | 256x256x256 | 1 | 0.853521 | 39.312954 |
| T-MAC W4A16 | W4A16 | Small Square | 256x256x256 | 2 | 0.484518 | 69.253221 |
| T-MAC W4A16 | W4A16 | Small Square | 256x256x256 | 4 | 0.304221 | 110.296239 |
| T-MAC W4A16 | W4A16 | Small Square | 256x256x256 | 8 | 0.278925 | 120.299120 |
| T-MAC W4A16 | W4A16 | Small Square | 256x256x256 | 16 | 0.281838 | 119.055741 |
| T-MAC W4A16 | W4A16 | Medium Square | 1024x1024x1024 | 1 | 49.687122 | 43.220125 |
| T-MAC W4A16 | W4A16 | Medium Square | 1024x1024x1024 | 2 | 21.069338 | 101.924591 |
| T-MAC W4A16 | W4A16 | Medium Square | 1024x1024x1024 | 4 | 13.361271 | 160.724504 |
| T-MAC W4A16 | W4A16 | Medium Square | 1024x1024x1024 | 8 | 13.124263 | 163.626990 |
| T-MAC W4A16 | W4A16 | Medium Square | 1024x1024x1024 | 16 | 12.314391 | 174.388132 |
| T-MAC W4A16 | W4A16 | Large Square | 4096x4096x4096 | 1 | 2230.487506 | 61.618347 |
| T-MAC W4A16 | W4A16 | Large Square | 4096x4096x4096 | 2 | 1387.305945 | 99.068957 |
| T-MAC W4A16 | W4A16 | Large Square | 4096x4096x4096 | 4 | 826.430887 | 166.304231 |
| T-MAC W4A16 | W4A16 | Large Square | 4096x4096x4096 | 8 | 712.418761 | 192.918773 |
| T-MAC W4A16 | W4A16 | Large Square | 4096x4096x4096 | 16 | 671.332637 | 204.725565 |
| T-MAC W4A16 | W4A16 | Rectangular | 4096x1024x2048 | 1 | 273.531722 | 62.807593 |
| T-MAC W4A16 | W4A16 | Rectangular | 4096x1024x2048 | 2 | 173.782093 | 98.858685 |
| T-MAC W4A16 | W4A16 | Rectangular | 4096x1024x2048 | 4 | 104.054383 | 165.104714 |
| T-MAC W4A16 | W4A16 | Rectangular | 4096x1024x2048 | 8 | 91.422475 | 187.917349 |
| T-MAC W4A16 | W4A16 | Rectangular | 4096x1024x2048 | 16 | 86.543672 | 198.510980 |
| T-MAC W4A16 | W4A16 | Medium Rectangular | 1024x1024x512 | 1 | 17.648935 | 60.838902 |
| T-MAC W4A16 | W4A16 | Medium Rectangular | 1024x1024x512 | 2 | 11.632085 | 92.308629 |
| T-MAC W4A16 | W4A16 | Medium Rectangular | 1024x1024x512 | 4 | 8.206064 | 130.847362 |
| T-MAC W4A16 | W4A16 | Medium Rectangular | 1024x1024x512 | 8 | 5.797108 | 185.220255 |
| T-MAC W4A16 | W4A16 | Medium Rectangular | 1024x1024x512 | 16 | 5.975858 | 179.679943 |

## T-MAC Speedup vs Local MKL

| Backend | Shape | Threads | MKL ms | T-MAC ms | Speedup |
|---|---|---:|---:|---:|---:|
| T-MAC W2A16 | 256x256x256 | 1 | 0.135663 | 0.459401 | 0.295× |
| T-MAC W2A16 | 256x256x256 | 2 | 0.128059 | 0.299541 | 0.428× |
| T-MAC W2A16 | 256x256x256 | 4 | 0.124377 | 0.340815 | 0.365× |
| T-MAC W2A16 | 256x256x256 | 8 | 0.113813 | 0.287962 | 0.395× |
| T-MAC W2A16 | 256x256x256 | 16 | 0.129488 | 0.289219 | 0.448× |
| T-MAC W2A16 | 1024x1024x1024 | 1 | 7.219003 | 22.822459 | 0.316× |
| T-MAC W2A16 | 1024x1024x1024 | 2 | 9.509686 | 12.999925 | 0.732× |
| T-MAC W2A16 | 1024x1024x1024 | 4 | 7.205394 | 7.517491 | 0.958× |
| T-MAC W2A16 | 1024x1024x1024 | 8 | 9.024728 | 8.866852 | 1.018× |
| T-MAC W2A16 | 1024x1024x1024 | 16 | 7.303773 | 7.772283 | 0.940× |
| T-MAC W2A16 | 4096x4096x4096 | 1 | 331.529382 | 1664.725245 | 0.199× |
| T-MAC W2A16 | 4096x4096x4096 | 2 | 320.147220 | 735.940318 | 0.435× |
| T-MAC W2A16 | 4096x4096x4096 | 4 | 344.963007 | 381.560458 | 0.904× |
| T-MAC W2A16 | 4096x4096x4096 | 8 | 324.934291 | 320.573850 | 1.014× |
| T-MAC W2A16 | 4096x4096x4096 | 16 | 325.084963 | 435.729368 | 0.746× |
| T-MAC W2A16 | 4096x1024x2048 | 1 | 51.362234 | 184.947226 | 0.278× |
| T-MAC W2A16 | 4096x1024x2048 | 2 | 59.343846 | 91.515319 | 0.648× |
| T-MAC W2A16 | 4096x1024x2048 | 4 | 49.865959 | 49.083188 | 1.016× |
| T-MAC W2A16 | 4096x1024x2048 | 8 | 51.740336 | 45.986719 | 1.125× |
| T-MAC W2A16 | 4096x1024x2048 | 16 | 50.053476 | 43.477866 | 1.151× |
| T-MAC W2A16 | 1024x1024x512 | 1 | 4.775016 | 12.618232 | 0.378× |
| T-MAC W2A16 | 1024x1024x512 | 2 | 3.488027 | 6.804307 | 0.513× |
| T-MAC W2A16 | 1024x1024x512 | 4 | 3.464960 | 4.185023 | 0.828× |
| T-MAC W2A16 | 1024x1024x512 | 8 | 3.475464 | 4.609484 | 0.754× |
| T-MAC W2A16 | 1024x1024x512 | 16 | 5.068800 | 4.487577 | 1.130× |
| T-MAC W3A16 | 256x256x256 | 1 | 0.135663 | 0.650710 | 0.208× |
| T-MAC W3A16 | 256x256x256 | 2 | 0.128059 | 0.394522 | 0.325× |
| T-MAC W3A16 | 256x256x256 | 4 | 0.124377 | 0.448772 | 0.277× |
| T-MAC W3A16 | 256x256x256 | 8 | 0.113813 | 0.379619 | 0.300× |
| T-MAC W3A16 | 256x256x256 | 16 | 0.129488 | 0.380427 | 0.340× |
| T-MAC W3A16 | 1024x1024x1024 | 1 | 7.219003 | 33.349888 | 0.216× |
| T-MAC W3A16 | 1024x1024x1024 | 2 | 9.509686 | 17.684426 | 0.538× |
| T-MAC W3A16 | 1024x1024x1024 | 4 | 7.205394 | 10.349298 | 0.696× |
| T-MAC W3A16 | 1024x1024x1024 | 8 | 9.024728 | 13.067544 | 0.691× |
| T-MAC W3A16 | 1024x1024x1024 | 16 | 7.303773 | 10.281066 | 0.710× |
| T-MAC W3A16 | 4096x4096x4096 | 1 | 331.529382 | 2262.284820 | 0.147× |
| T-MAC W3A16 | 4096x4096x4096 | 2 | 320.147220 | 1119.598873 | 0.286× |
| T-MAC W3A16 | 4096x4096x4096 | 4 | 344.963007 | 631.802582 | 0.546× |
| T-MAC W3A16 | 4096x4096x4096 | 8 | 324.934291 | 529.392155 | 0.614× |
| T-MAC W3A16 | 4096x4096x4096 | 16 | 325.084963 | 599.434801 | 0.542× |
| T-MAC W3A16 | 4096x1024x2048 | 1 | 51.362234 | 275.046955 | 0.187× |
| T-MAC W3A16 | 4096x1024x2048 | 2 | 59.343846 | 131.710203 | 0.451× |
| T-MAC W3A16 | 4096x1024x2048 | 4 | 49.865959 | 73.906900 | 0.675× |
| T-MAC W3A16 | 4096x1024x2048 | 8 | 51.740336 | 73.204279 | 0.707× |
| T-MAC W3A16 | 4096x1024x2048 | 16 | 50.053476 | 66.658447 | 0.751× |
| T-MAC W3A16 | 1024x1024x512 | 1 | 4.775016 | 17.832168 | 0.268× |
| T-MAC W3A16 | 1024x1024x512 | 2 | 3.488027 | 9.470799 | 0.368× |
| T-MAC W3A16 | 1024x1024x512 | 4 | 3.464960 | 5.902071 | 0.587× |
| T-MAC W3A16 | 1024x1024x512 | 8 | 3.475464 | 6.237089 | 0.557× |
| T-MAC W3A16 | 1024x1024x512 | 16 | 5.068800 | 5.692750 | 0.890× |
| T-MAC W4A16 | 256x256x256 | 1 | 0.135663 | 0.853521 | 0.159× |
| T-MAC W4A16 | 256x256x256 | 2 | 0.128059 | 0.484518 | 0.264× |
| T-MAC W4A16 | 256x256x256 | 4 | 0.124377 | 0.304221 | 0.409× |
| T-MAC W4A16 | 256x256x256 | 8 | 0.113813 | 0.278925 | 0.408× |
| T-MAC W4A16 | 256x256x256 | 16 | 0.129488 | 0.281838 | 0.459× |
| T-MAC W4A16 | 1024x1024x1024 | 1 | 7.219003 | 49.687122 | 0.145× |
| T-MAC W4A16 | 1024x1024x1024 | 2 | 9.509686 | 21.069338 | 0.451× |
| T-MAC W4A16 | 1024x1024x1024 | 4 | 7.205394 | 13.361271 | 0.539× |
| T-MAC W4A16 | 1024x1024x1024 | 8 | 9.024728 | 13.124263 | 0.688× |
| T-MAC W4A16 | 1024x1024x1024 | 16 | 7.303773 | 12.314391 | 0.593× |
| T-MAC W4A16 | 4096x4096x4096 | 1 | 331.529382 | 2230.487506 | 0.149× |
| T-MAC W4A16 | 4096x4096x4096 | 2 | 320.147220 | 1387.305945 | 0.231× |
| T-MAC W4A16 | 4096x4096x4096 | 4 | 344.963007 | 826.430887 | 0.417× |
| T-MAC W4A16 | 4096x4096x4096 | 8 | 324.934291 | 712.418761 | 0.456× |
| T-MAC W4A16 | 4096x4096x4096 | 16 | 325.084963 | 671.332637 | 0.484× |
| T-MAC W4A16 | 4096x1024x2048 | 1 | 51.362234 | 273.531722 | 0.188× |
| T-MAC W4A16 | 4096x1024x2048 | 2 | 59.343846 | 173.782093 | 0.341× |
| T-MAC W4A16 | 4096x1024x2048 | 4 | 49.865959 | 104.054383 | 0.479× |
| T-MAC W4A16 | 4096x1024x2048 | 8 | 51.740336 | 91.422475 | 0.566× |
| T-MAC W4A16 | 4096x1024x2048 | 16 | 50.053476 | 86.543672 | 0.578× |
| T-MAC W4A16 | 1024x1024x512 | 1 | 4.775016 | 17.648935 | 0.271× |
| T-MAC W4A16 | 1024x1024x512 | 2 | 3.488027 | 11.632085 | 0.300× |
| T-MAC W4A16 | 1024x1024x512 | 4 | 3.464960 | 8.206064 | 0.422× |
| T-MAC W4A16 | 1024x1024x512 | 8 | 3.475464 | 5.797108 | 0.600× |
| T-MAC W4A16 | 1024x1024x512 | 16 | 5.068800 | 5.975858 | 0.848× |

## Local MKL vs HackMD MKL

| Shape | Threads | HackMD ms | Local ms | Local/HackMD | Difference | Note |
|---|---:|---:|---:|---:|---:|---|
| 256x256x256 | 1 | 0.236500 | 0.135663 | 0.574× | -42.6% | HackMD first table dtype not explicitly stated |
| 256x256x256 | 2 | 0.246500 | 0.128059 | 0.520× | -48.0% | HackMD first table dtype not explicitly stated |
| 256x256x256 | 4 | 0.146200 | 0.124377 | 0.851× | -14.9% | HackMD first table dtype not explicitly stated |
| 256x256x256 | 8 | 0.325800 | 0.113813 | 0.349× | -65.1% | HackMD first table dtype not explicitly stated |
| 256x256x256 | 16 | 0.408100 | 0.129488 | 0.317× | -68.3% | HackMD first table dtype not explicitly stated |
| 1024x1024x1024 | 1 | 16.757000 | 7.219003 | 0.431× | -56.9% | HackMD first table dtype not explicitly stated |
| 1024x1024x1024 | 2 | 12.300200 | 9.509686 | 0.773× | -22.7% | HackMD first table dtype not explicitly stated |
| 1024x1024x1024 | 4 | 6.482800 | 7.205394 | 1.111× | +11.1% | HackMD first table dtype not explicitly stated |
| 1024x1024x1024 | 8 | 3.575700 | 9.024728 | 2.524× | +152.4% | HackMD first table dtype not explicitly stated |
| 1024x1024x1024 | 16 | 5.927800 | 7.303773 | 1.232× | +23.2% | HackMD first table dtype not explicitly stated |
| 4096x4096x4096 | 1 | 2071.977700 | 331.529382 | 0.160× | -84.0% | HackMD first table dtype not explicitly stated |
| 4096x4096x4096 | 2 | 1048.065300 | 320.147220 | 0.305× | -69.5% | HackMD first table dtype not explicitly stated |
| 4096x4096x4096 | 4 | 527.367700 | 344.963007 | 0.654× | -34.6% | HackMD first table dtype not explicitly stated |
| 4096x4096x4096 | 8 | 266.501900 | 324.934291 | 1.219× | +21.9% | HackMD first table dtype not explicitly stated |
| 4096x4096x4096 | 16 | 260.432100 | 325.084963 | 1.248× | +24.8% | HackMD first table dtype not explicitly stated |
| 4096x1024x2048 | 1 | 260.955100 | 51.362234 | 0.197× | -80.3% | HackMD first table dtype not explicitly stated |
| 4096x1024x2048 | 2 | 130.760300 | 59.343846 | 0.454× | -54.6% | HackMD first table dtype not explicitly stated |
| 4096x1024x2048 | 4 | 65.712300 | 49.865959 | 0.759× | -24.1% | HackMD first table dtype not explicitly stated |
| 4096x1024x2048 | 8 | 33.263400 | 51.740336 | 1.555× | +55.5% | HackMD first table dtype not explicitly stated |
| 4096x1024x2048 | 16 | 72.782800 | 50.053476 | 0.688× | -31.2% | HackMD first table dtype not explicitly stated |
| 1024x1024x512 | 1 | 3.410722 | 4.775016 | 1.400× | +40.0% | FP16 confirmed |
| 1024x1024x512 | 2 | 3.164299 | 3.488027 | 1.102× | +10.2% | FP16 confirmed |
| 1024x1024x512 | 4 | 3.135987 | 3.464960 | 1.105× | +10.5% | FP16 confirmed |
| 1024x1024x512 | 8 | 3.123567 | 3.475464 | 1.113× | +11.3% | FP16 confirmed |
| 1024x1024x512 | 16 | 3.145073 | 5.068800 | 1.612× | +61.2% | FP16 confirmed |

## Statistics

| Backend | Shape | Threads | Runs | Mean ms | Median ms | Std ms | Min ms | Max ms | CV (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MKL | 256x256x256 | 1 | 5 | 4.511119 | 0.135663 | 7.813961 | 0.110675 | 18.155350 | 173.22 |
| MKL | 256x256x256 | 2 | 5 | 1.420900 | 0.128059 | 1.995752 | 0.111393 | 4.640257 | 140.46 |
| MKL | 256x256x256 | 4 | 5 | 1.182990 | 0.124377 | 1.515992 | 0.110058 | 3.352223 | 128.15 |
| MKL | 256x256x256 | 8 | 5 | 0.337325 | 0.113813 | 0.497422 | 0.109606 | 1.227063 | 147.46 |
| MKL | 256x256x256 | 16 | 5 | 0.574511 | 0.129488 | 0.665816 | 0.109936 | 1.583941 | 115.89 |
| MKL | 1024x1024x1024 | 1 | 5 | 8.184528 | 7.219003 | 1.633362 | 6.786995 | 10.233092 | 19.96 |
| MKL | 1024x1024x1024 | 2 | 5 | 8.944570 | 9.509686 | 2.078753 | 6.661319 | 11.405362 | 23.24 |
| MKL | 1024x1024x1024 | 4 | 5 | 8.515131 | 7.205394 | 2.393143 | 6.963149 | 12.579334 | 28.10 |
| MKL | 1024x1024x1024 | 8 | 5 | 9.037745 | 9.024728 | 2.073867 | 7.024566 | 12.331858 | 22.95 |
| MKL | 1024x1024x1024 | 16 | 5 | 9.546107 | 7.303773 | 3.602574 | 7.242579 | 15.512488 | 37.74 |
| MKL | 4096x4096x4096 | 1 | 5 | 363.612728 | 331.529382 | 55.522969 | 314.549855 | 427.407794 | 15.27 |
| MKL | 4096x4096x4096 | 2 | 5 | 324.155931 | 320.147220 | 15.828878 | 309.375185 | 350.234585 | 4.88 |
| MKL | 4096x4096x4096 | 4 | 5 | 379.288718 | 344.963007 | 63.861431 | 317.742869 | 460.727858 | 16.84 |
| MKL | 4096x4096x4096 | 8 | 5 | 330.645486 | 324.934291 | 15.359455 | 313.725752 | 353.955952 | 4.65 |
| MKL | 4096x4096x4096 | 16 | 5 | 347.065296 | 325.084963 | 55.950908 | 312.787184 | 446.497956 | 16.12 |
| MKL | 4096x1024x2048 | 1 | 5 | 53.897866 | 51.362234 | 8.582691 | 46.500183 | 67.226882 | 15.92 |
| MKL | 4096x1024x2048 | 2 | 5 | 59.285906 | 59.343846 | 11.004114 | 46.400232 | 73.963281 | 18.56 |
| MKL | 4096x1024x2048 | 4 | 5 | 60.080996 | 49.865959 | 15.672645 | 48.194049 | 82.748802 | 26.09 |
| MKL | 4096x1024x2048 | 8 | 5 | 55.428394 | 51.740336 | 9.197088 | 47.711904 | 68.572515 | 16.59 |
| MKL | 4096x1024x2048 | 16 | 5 | 55.121937 | 50.053476 | 11.409410 | 48.136572 | 75.310502 | 20.70 |
| MKL | 1024x1024x512 | 1 | 5 | 5.557506 | 4.775016 | 3.084174 | 3.364121 | 10.900643 | 55.50 |
| MKL | 1024x1024x512 | 2 | 5 | 5.190579 | 3.488027 | 2.866185 | 3.319600 | 10.017271 | 55.22 |
| MKL | 1024x1024x512 | 4 | 5 | 5.705211 | 3.464960 | 3.239441 | 3.320066 | 10.248978 | 56.78 |
| MKL | 1024x1024x512 | 8 | 5 | 5.777804 | 3.475464 | 3.480268 | 3.311028 | 10.978190 | 60.24 |
| MKL | 1024x1024x512 | 16 | 5 | 5.384142 | 5.068800 | 1.821787 | 3.471332 | 8.406744 | 33.84 |
| T-MAC W2A16 | 256x256x256 | 1 | 5 | 0.460862 | 0.459401 | 0.004932 | 0.455881 | 0.467617 | 1.07 |
| T-MAC W2A16 | 256x256x256 | 2 | 5 | 0.299887 | 0.299541 | 0.004054 | 0.293756 | 0.304120 | 1.35 |
| T-MAC W2A16 | 256x256x256 | 4 | 5 | 0.292070 | 0.340815 | 0.076256 | 0.205670 | 0.355934 | 26.11 |
| T-MAC W2A16 | 256x256x256 | 8 | 5 | 0.288118 | 0.287962 | 0.002361 | 0.285228 | 0.291163 | 0.82 |
| T-MAC W2A16 | 256x256x256 | 16 | 5 | 0.288723 | 0.289219 | 0.002023 | 0.285833 | 0.290627 | 0.70 |
| T-MAC W2A16 | 1024x1024x1024 | 1 | 5 | 23.440148 | 22.822459 | 1.339334 | 22.137084 | 25.464958 | 5.71 |
| T-MAC W2A16 | 1024x1024x1024 | 2 | 5 | 13.058409 | 12.999925 | 0.501397 | 12.460470 | 13.772543 | 3.84 |
| T-MAC W2A16 | 1024x1024x1024 | 4 | 5 | 7.727803 | 7.517491 | 0.477934 | 7.330169 | 8.539817 | 6.18 |
| T-MAC W2A16 | 1024x1024x1024 | 8 | 5 | 8.905080 | 8.866852 | 0.363133 | 8.569313 | 9.426870 | 4.08 |
| T-MAC W2A16 | 1024x1024x1024 | 16 | 5 | 7.990788 | 7.772283 | 0.590876 | 7.428365 | 8.929809 | 7.39 |
| T-MAC W2A16 | 4096x4096x4096 | 1 | 5 | 1646.081166 | 1664.725245 | 99.097157 | 1532.332630 | 1768.385888 | 6.02 |
| T-MAC W2A16 | 4096x4096x4096 | 2 | 5 | 732.219782 | 735.940318 | 15.354823 | 714.039263 | 751.177536 | 2.10 |
| T-MAC W2A16 | 4096x4096x4096 | 4 | 5 | 380.672820 | 381.560458 | 3.823495 | 375.249203 | 385.034088 | 1.00 |
| T-MAC W2A16 | 4096x4096x4096 | 8 | 5 | 323.610446 | 320.573850 | 11.455020 | 311.964201 | 340.424526 | 3.54 |
| T-MAC W2A16 | 4096x4096x4096 | 16 | 5 | 426.103571 | 435.729368 | 31.685343 | 377.715475 | 452.466038 | 7.44 |
| T-MAC W2A16 | 4096x1024x2048 | 1 | 5 | 190.229591 | 184.947226 | 20.894100 | 173.181643 | 226.499178 | 10.98 |
| T-MAC W2A16 | 4096x1024x2048 | 2 | 5 | 92.049540 | 91.515319 | 1.021779 | 91.004978 | 93.427278 | 1.11 |
| T-MAC W2A16 | 4096x1024x2048 | 4 | 5 | 50.111945 | 49.083188 | 3.141137 | 47.145566 | 54.983450 | 6.27 |
| T-MAC W2A16 | 4096x1024x2048 | 8 | 5 | 45.948308 | 45.986719 | 2.644732 | 43.054649 | 50.095524 | 5.76 |
| T-MAC W2A16 | 4096x1024x2048 | 16 | 5 | 44.140723 | 43.477866 | 2.728688 | 41.604221 | 48.259369 | 6.18 |
| T-MAC W2A16 | 1024x1024x512 | 1 | 5 | 14.634614 | 12.618232 | 4.941904 | 12.044279 | 23.464905 | 33.77 |
| T-MAC W2A16 | 1024x1024x512 | 2 | 5 | 8.206694 | 6.804307 | 3.117917 | 6.693913 | 13.778772 | 37.99 |
| T-MAC W2A16 | 1024x1024x512 | 4 | 5 | 5.330869 | 4.185023 | 2.022441 | 3.767114 | 8.215396 | 37.94 |
| T-MAC W2A16 | 1024x1024x512 | 8 | 5 | 5.567119 | 4.609484 | 2.515242 | 4.218344 | 10.049641 | 45.18 |
| T-MAC W2A16 | 1024x1024x512 | 16 | 5 | 4.465877 | 4.487577 | 0.421728 | 4.072121 | 5.128709 | 9.44 |
| T-MAC W3A16 | 256x256x256 | 1 | 5 | 0.668243 | 0.650710 | 0.054852 | 0.630734 | 0.765055 | 8.21 |
| T-MAC W3A16 | 256x256x256 | 2 | 5 | 0.462664 | 0.394522 | 0.156578 | 0.385822 | 0.742564 | 33.84 |
| T-MAC W3A16 | 256x256x256 | 4 | 5 | 0.416886 | 0.448772 | 0.085870 | 0.264304 | 0.469099 | 20.60 |
| T-MAC W3A16 | 256x256x256 | 8 | 5 | 0.330069 | 0.379619 | 0.070060 | 0.251295 | 0.383921 | 21.23 |
| T-MAC W3A16 | 256x256x256 | 16 | 5 | 0.356348 | 0.380427 | 0.057490 | 0.253931 | 0.389887 | 16.13 |
| T-MAC W3A16 | 1024x1024x1024 | 1 | 5 | 33.681020 | 33.349888 | 2.096676 | 31.045934 | 36.845400 | 6.23 |
| T-MAC W3A16 | 1024x1024x1024 | 2 | 5 | 17.883857 | 17.684426 | 0.480842 | 17.440994 | 18.658372 | 2.69 |
| T-MAC W3A16 | 1024x1024x1024 | 4 | 5 | 10.398250 | 10.349298 | 0.671520 | 9.603379 | 11.141262 | 6.46 |
| T-MAC W3A16 | 1024x1024x1024 | 8 | 5 | 14.087316 | 13.067544 | 2.729847 | 12.260599 | 18.916902 | 19.38 |
| T-MAC W3A16 | 1024x1024x1024 | 16 | 5 | 10.657669 | 10.281066 | 0.723749 | 10.034664 | 11.792128 | 6.79 |
| T-MAC W3A16 | 4096x4096x4096 | 1 | 5 | 2498.097640 | 2262.284820 | 658.911010 | 1766.245894 | 3414.028139 | 26.38 |
| T-MAC W3A16 | 4096x4096x4096 | 2 | 5 | 1094.990192 | 1119.598873 | 74.039996 | 969.488106 | 1153.652403 | 6.76 |
| T-MAC W3A16 | 4096x4096x4096 | 4 | 5 | 607.209604 | 631.802582 | 46.361415 | 546.749205 | 648.989558 | 7.64 |
| T-MAC W3A16 | 4096x4096x4096 | 8 | 5 | 520.083430 | 529.392155 | 48.452358 | 462.221219 | 585.697882 | 9.32 |
| T-MAC W3A16 | 4096x4096x4096 | 16 | 5 | 599.382172 | 599.434801 | 63.627281 | 515.974759 | 678.850297 | 10.62 |
| T-MAC W3A16 | 4096x1024x2048 | 1 | 5 | 256.461188 | 275.046955 | 43.984126 | 199.844254 | 300.893395 | 17.15 |
| T-MAC W3A16 | 4096x1024x2048 | 2 | 5 | 131.937086 | 131.710203 | 3.728919 | 126.378878 | 136.049753 | 2.83 |
| T-MAC W3A16 | 4096x1024x2048 | 4 | 5 | 75.802063 | 73.906900 | 7.112069 | 68.658202 | 84.761991 | 9.38 |
| T-MAC W3A16 | 4096x1024x2048 | 8 | 5 | 71.889389 | 73.204279 | 9.147978 | 62.962697 | 85.285641 | 12.73 |
| T-MAC W3A16 | 4096x1024x2048 | 16 | 5 | 63.985703 | 66.658447 | 7.161282 | 55.639967 | 71.886022 | 11.19 |
| T-MAC W3A16 | 1024x1024x512 | 1 | 5 | 17.654442 | 17.832168 | 3.589308 | 13.621587 | 21.576153 | 20.33 |
| T-MAC W3A16 | 1024x1024x512 | 2 | 5 | 9.319254 | 9.470799 | 0.719521 | 8.449355 | 10.320682 | 7.72 |
| T-MAC W3A16 | 1024x1024x512 | 4 | 5 | 6.080238 | 5.902071 | 0.858020 | 5.211694 | 7.316231 | 14.11 |
| T-MAC W3A16 | 1024x1024x512 | 8 | 5 | 5.952683 | 6.237089 | 0.921627 | 4.513790 | 6.850177 | 15.48 |
| T-MAC W3A16 | 1024x1024x512 | 16 | 5 | 5.512757 | 5.692750 | 0.665005 | 4.595903 | 6.262212 | 12.06 |
| T-MAC W4A16 | 256x256x256 | 1 | 5 | 0.873953 | 0.853521 | 0.105175 | 0.777416 | 1.051015 | 12.03 |
| T-MAC W4A16 | 256x256x256 | 2 | 5 | 0.505331 | 0.484518 | 0.055989 | 0.466069 | 0.600253 | 11.08 |
| T-MAC W4A16 | 256x256x256 | 4 | 5 | 0.315403 | 0.304221 | 0.040935 | 0.281577 | 0.386495 | 12.98 |
| T-MAC W4A16 | 256x256x256 | 8 | 5 | 0.284231 | 0.278925 | 0.026919 | 0.261556 | 0.329937 | 9.47 |
| T-MAC W4A16 | 256x256x256 | 16 | 5 | 0.294987 | 0.281838 | 0.049938 | 0.244657 | 0.376101 | 16.93 |
| T-MAC W4A16 | 1024x1024x1024 | 1 | 5 | 47.452775 | 49.687122 | 5.858490 | 41.385894 | 54.604235 | 12.35 |
| T-MAC W4A16 | 1024x1024x1024 | 2 | 5 | 22.533577 | 21.069338 | 3.173105 | 20.929342 | 28.199914 | 14.08 |
| T-MAC W4A16 | 1024x1024x1024 | 4 | 5 | 14.393427 | 13.361271 | 2.589663 | 12.758766 | 18.938492 | 17.99 |
| T-MAC W4A16 | 1024x1024x1024 | 8 | 5 | 14.225072 | 13.124263 | 2.721667 | 12.821653 | 19.088017 | 19.13 |
| T-MAC W4A16 | 1024x1024x1024 | 16 | 5 | 13.547838 | 12.314391 | 2.559254 | 12.037248 | 18.054988 | 18.89 |
| T-MAC W4A16 | 4096x4096x4096 | 1 | 5 | 2243.091733 | 2230.487506 | 101.454630 | 2143.633473 | 2393.420573 | 4.52 |
| T-MAC W4A16 | 4096x4096x4096 | 2 | 5 | 1374.138246 | 1387.305945 | 96.822057 | 1244.555581 | 1510.097038 | 7.05 |
| T-MAC W4A16 | 4096x4096x4096 | 4 | 5 | 884.690509 | 826.430887 | 116.508837 | 805.770237 | 1084.733384 | 13.17 |
| T-MAC W4A16 | 4096x4096x4096 | 8 | 5 | 738.030420 | 712.418761 | 78.178944 | 672.404508 | 870.806523 | 10.59 |
| T-MAC W4A16 | 4096x4096x4096 | 16 | 5 | 713.972280 | 671.332637 | 81.970216 | 654.947037 | 852.388091 | 11.48 |
| T-MAC W4A16 | 4096x1024x2048 | 1 | 5 | 271.429399 | 273.531722 | 11.163732 | 258.625180 | 285.656756 | 4.11 |
| T-MAC W4A16 | 4096x1024x2048 | 2 | 5 | 176.515994 | 173.782093 | 11.370071 | 161.455932 | 192.257197 | 6.44 |
| T-MAC W4A16 | 4096x1024x2048 | 4 | 5 | 111.260577 | 104.054383 | 14.114273 | 100.984673 | 134.886215 | 12.69 |
| T-MAC W4A16 | 4096x1024x2048 | 8 | 5 | 94.291576 | 91.422475 | 6.987389 | 90.312613 | 106.752505 | 7.41 |
| T-MAC W4A16 | 4096x1024x2048 | 16 | 5 | 90.320368 | 86.543672 | 8.922333 | 84.523193 | 105.859252 | 9.88 |
| T-MAC W4A16 | 1024x1024x512 | 1 | 5 | 18.163977 | 17.648935 | 1.380550 | 17.314691 | 20.617605 | 7.60 |
| T-MAC W4A16 | 1024x1024x512 | 2 | 5 | 11.716255 | 11.632085 | 0.412585 | 11.235057 | 12.360854 | 3.52 |
| T-MAC W4A16 | 1024x1024x512 | 4 | 5 | 8.153019 | 8.206064 | 0.509593 | 7.432346 | 8.859697 | 6.25 |
| T-MAC W4A16 | 1024x1024x512 | 8 | 5 | 8.010069 | 5.797108 | 5.146798 | 5.483733 | 17.209395 | 64.25 |
| T-MAC W4A16 | 1024x1024x512 | 16 | 5 | 6.204645 | 5.975858 | 0.667497 | 5.722364 | 7.374069 | 10.76 |

## Autotune Schedule Stability

| Backend | Shape | Threads | Schedule |
|---|---|---:|---|
| T-MAC W2A16 | 256x256x256 | 1 | unstable: mode bm=512,bn=64,kfactor=16 (2/5), 3 schedules |
| T-MAC W2A16 | 256x256x256 | 2 | unstable: mode bm=512,bn=64,kfactor=16 (2/5), 4 schedules |
| T-MAC W2A16 | 256x256x256 | 4 | unstable: mode bm=512,bn=16,kfactor=16 (2/5), 4 schedules |
| T-MAC W2A16 | 256x256x256 | 8 | stable: bm=256,bn=64,kfactor=16 (5/5) |
| T-MAC W2A16 | 256x256x256 | 16 | mostly stable: bm=256,bn=64,kfactor=16 (3/5), 2 schedules |
| T-MAC W2A16 | 1024x1024x1024 | 1 | unstable: mode bm=128,bn=64,kfactor=16 (2/5), 3 schedules |
| T-MAC W2A16 | 1024x1024x1024 | 2 | unstable: mode bm=256,bn=8,kfactor=16 (2/5), 4 schedules |
| T-MAC W2A16 | 1024x1024x1024 | 4 | unstable: mode bm=1024,bn=8,kfactor=16 (2/5), 4 schedules |
| T-MAC W2A16 | 1024x1024x1024 | 8 | unstable: mode bm=1024,bn=8,kfactor=16 (2/5), 4 schedules |
| T-MAC W2A16 | 1024x1024x1024 | 16 | mostly stable: bm=256,bn=8,kfactor=16 (3/5), 3 schedules |
| T-MAC W2A16 | 4096x4096x4096 | 1 | unstable: mode bm=128,bn=16,kfactor=16 (2/5), 4 schedules |
| T-MAC W2A16 | 4096x4096x4096 | 2 | mostly stable: bm=256,bn=32,kfactor=16 (3/5), 3 schedules |
| T-MAC W2A16 | 4096x4096x4096 | 4 | mostly stable: bm=256,bn=16,kfactor=16 (3/5), 3 schedules |
| T-MAC W2A16 | 4096x4096x4096 | 8 | unstable: mode bm=512,bn=16,kfactor=16 (1/5), 5 schedules |
| T-MAC W2A16 | 4096x4096x4096 | 16 | unstable: mode bm=512,bn=64,kfactor=16 (1/5), 5 schedules |
| T-MAC W2A16 | 4096x1024x2048 | 1 | unstable: mode bm=512,bn=16,kfactor=16 (2/5), 4 schedules |
| T-MAC W2A16 | 4096x1024x2048 | 2 | unstable: mode bm=1024,bn=16,kfactor=16 (2/5), 4 schedules |
| T-MAC W2A16 | 4096x1024x2048 | 4 | unstable: mode bm=256,bn=32,kfactor=16 (2/5), 4 schedules |
| T-MAC W2A16 | 4096x1024x2048 | 8 | unstable: mode bm=512,bn=32,kfactor=16 (1/5), 5 schedules |
| T-MAC W2A16 | 4096x1024x2048 | 16 | unstable: mode bm=1024,bn=8,kfactor=16 (2/5), 4 schedules |
| T-MAC W2A16 | 1024x1024x512 | 1 | mostly stable: bm=256,bn=8,kfactor=16 (4/5), 2 schedules |
| T-MAC W2A16 | 1024x1024x512 | 2 | unstable: mode bm=256,bn=64,kfactor=16 (2/5), 4 schedules |
| T-MAC W2A16 | 1024x1024x512 | 4 | unstable: mode bm=1024,bn=32,kfactor=16 (1/5), 5 schedules |
| T-MAC W2A16 | 1024x1024x512 | 8 | unstable: mode bm=256,bn=8,kfactor=16 (1/5), 5 schedules |
| T-MAC W2A16 | 1024x1024x512 | 16 | unstable: mode bm=256,bn=32,kfactor=16 (2/5), 4 schedules |
| T-MAC W3A16 | 256x256x256 | 1 | unstable: mode bm=768,bn=16,kfactor=16 (2/5), 3 schedules |
| T-MAC W3A16 | 256x256x256 | 2 | unstable: mode bm=384,bn=8,kfactor=16 (2/5), 4 schedules |
| T-MAC W3A16 | 256x256x256 | 4 | unstable: mode bm=384,bn=32,kfactor=16 (1/5), 5 schedules |
| T-MAC W3A16 | 256x256x256 | 8 | mostly stable: bm=384,bn=64,kfactor=16 (3/5), 2 schedules |
| T-MAC W3A16 | 256x256x256 | 16 | mostly stable: bm=384,bn=64,kfactor=16 (3/5), 3 schedules |
| T-MAC W3A16 | 1024x1024x1024 | 1 | unstable: mode bm=768,bn=64,kfactor=16 (2/5), 4 schedules |
| T-MAC W3A16 | 1024x1024x1024 | 2 | unstable: mode bm=768,bn=8,kfactor=16 (2/5), 4 schedules |
| T-MAC W3A16 | 1024x1024x1024 | 4 | unstable: mode bm=768,bn=32,kfactor=16 (2/5), 4 schedules |
| T-MAC W3A16 | 1024x1024x1024 | 8 | unstable: mode bm=768,bn=16,kfactor=16 (2/5), 4 schedules |
| T-MAC W3A16 | 1024x1024x1024 | 16 | unstable: mode bm=768,bn=32,kfactor=16 (2/5), 3 schedules |
| T-MAC W3A16 | 4096x4096x4096 | 1 | unstable: mode bm=192,bn=64,kfactor=16 (1/5), 5 schedules |
| T-MAC W3A16 | 4096x4096x4096 | 2 | unstable: mode bm=192,bn=8,kfactor=16 (2/5), 4 schedules |
| T-MAC W3A16 | 4096x4096x4096 | 4 | mostly stable: bm=192,bn=32,kfactor=16 (3/5), 3 schedules |
| T-MAC W3A16 | 4096x4096x4096 | 8 | unstable: mode bm=192,bn=64,kfactor=16 (2/5), 4 schedules |
| T-MAC W3A16 | 4096x4096x4096 | 16 | unstable: mode bm=192,bn=16,kfactor=16 (1/5), 5 schedules |
| T-MAC W3A16 | 4096x1024x2048 | 1 | mostly stable: bm=384,bn=64,kfactor=16 (4/5), 2 schedules |
| T-MAC W3A16 | 4096x1024x2048 | 2 | unstable: mode bm=192,bn=16,kfactor=16 (1/5), 5 schedules |
| T-MAC W3A16 | 4096x1024x2048 | 4 | unstable: mode bm=384,bn=8,kfactor=16 (2/5), 4 schedules |
| T-MAC W3A16 | 4096x1024x2048 | 8 | unstable: mode bm=384,bn=64,kfactor=16 (1/5), 5 schedules |
| T-MAC W3A16 | 4096x1024x2048 | 16 | unstable: mode bm=192,bn=8,kfactor=16 (2/5), 4 schedules |
| T-MAC W3A16 | 1024x1024x512 | 1 | unstable: mode bm=768,bn=64,kfactor=16 (2/5), 4 schedules |
| T-MAC W3A16 | 1024x1024x512 | 2 | unstable: mode bm=768,bn=32,kfactor=16 (1/5), 5 schedules |
| T-MAC W3A16 | 1024x1024x512 | 4 | mostly stable: bm=768,bn=16,kfactor=16 (3/5), 3 schedules |
| T-MAC W3A16 | 1024x1024x512 | 8 | unstable: mode bm=768,bn=64,kfactor=16 (2/5), 4 schedules |
| T-MAC W3A16 | 1024x1024x512 | 16 | unstable: mode bm=384,bn=64,kfactor=16 (1/5), 5 schedules |
| T-MAC W4A16 | 256x256x256 | 1 | unstable: mode bm=1024,bn=32,kfactor=16 (2/5), 3 schedules |
| T-MAC W4A16 | 256x256x256 | 2 | unstable: mode bm=1024,bn=64,kfactor=16 (2/5), 3 schedules |
| T-MAC W4A16 | 256x256x256 | 4 | unstable: mode bm=1024,bn=32,kfactor=16 (2/5), 4 schedules |
| T-MAC W4A16 | 256x256x256 | 8 | unstable: mode bm=256,bn=16,kfactor=16 (2/5), 4 schedules |
| T-MAC W4A16 | 256x256x256 | 16 | unstable: mode bm=128,bn=64,kfactor=16 (2/5), 4 schedules |
| T-MAC W4A16 | 1024x1024x1024 | 1 | unstable: mode bm=1024,bn=8,kfactor=16 (2/5), 4 schedules |
| T-MAC W4A16 | 1024x1024x1024 | 2 | unstable: mode bm=128,bn=32,kfactor=16 (2/5), 4 schedules |
| T-MAC W4A16 | 1024x1024x1024 | 4 | unstable: mode bm=128,bn=32,kfactor=16 (1/5), 5 schedules |
| T-MAC W4A16 | 1024x1024x1024 | 8 | unstable: mode bm=256,bn=8,kfactor=16 (2/5), 4 schedules |
| T-MAC W4A16 | 1024x1024x1024 | 16 | unstable: mode bm=128,bn=16,kfactor=16 (1/5), 5 schedules |
| T-MAC W4A16 | 4096x4096x4096 | 1 | mostly stable: bm=128,bn=64,kfactor=16 (3/5), 3 schedules |
| T-MAC W4A16 | 4096x4096x4096 | 2 | unstable: mode bm=256,bn=8,kfactor=16 (2/5), 4 schedules |
| T-MAC W4A16 | 4096x4096x4096 | 4 | unstable: mode bm=256,bn=64,kfactor=16 (2/5), 4 schedules |
| T-MAC W4A16 | 4096x4096x4096 | 8 | unstable: mode bm=256,bn=16,kfactor=16 (1/5), 5 schedules |
| T-MAC W4A16 | 4096x4096x4096 | 16 | unstable: mode bm=128,bn=32,kfactor=16 (2/5), 3 schedules |
| T-MAC W4A16 | 4096x1024x2048 | 1 | mostly stable: bm=128,bn=64,kfactor=16 (3/5), 3 schedules |
| T-MAC W4A16 | 4096x1024x2048 | 2 | unstable: mode bm=256,bn=8,kfactor=16 (2/5), 4 schedules |
| T-MAC W4A16 | 4096x1024x2048 | 4 | unstable: mode bm=128,bn=32,kfactor=16 (2/5), 4 schedules |
| T-MAC W4A16 | 4096x1024x2048 | 8 | mostly stable: bm=256,bn=8,kfactor=16 (3/5), 3 schedules |
| T-MAC W4A16 | 4096x1024x2048 | 16 | unstable: mode bm=1024,bn=32,kfactor=16 (1/5), 5 schedules |
| T-MAC W4A16 | 1024x1024x512 | 1 | mostly stable: bm=1024,bn=64,kfactor=16 (4/5), 2 schedules |
| T-MAC W4A16 | 1024x1024x512 | 2 | unstable: mode bm=256,bn=8,kfactor=16 (2/5), 3 schedules |
| T-MAC W4A16 | 1024x1024x512 | 4 | unstable: mode bm=256,bn=32,kfactor=16 (2/5), 4 schedules |
| T-MAC W4A16 | 1024x1024x512 | 8 | mostly stable: bm=256,bn=32,kfactor=16 (3/5), 3 schedules |
| T-MAC W4A16 | 1024x1024x512 | 16 | unstable: mode bm=256,bn=16,kfactor=16 (2/5), 4 schedules |

## Conclusion

- T-MAC W2A16: 6/25 cases beat MKL; best = 1.151× at 4096x1024x2048 16T.
- T-MAC W3A16: 0/25 cases beat MKL; best = 0.890× at 1024x1024x512 16T.
- T-MAC W4A16: 0/25 cases beat MKL; best = 0.848× at 1024x1024x512 16T.
- High-variation configurations (CV >= 20%): 27/100.
- T-MAC configurations with notably unstable autotune schedules: 59.
- 16T is oversubscribed because the current environment exposes 8 hardware threads.

## Method

- Each benchmark executable performs its own warm-up and per-run sampling.
- Each configuration is executed in five independent complete runs.
- The final latency is the median of the five run-level medians.
- Mean, sample standard deviation, min, max and CV are retained for stability analysis.
- Speedup vs MKL = `MKL median latency / T-MAC median latency`; values > 1 mean T-MAC is faster.
- The 1024×1024×512 HackMD MKL reference is explicitly labeled FP16.
- The first HackMD OpenVINO-vs-MKL table does not explicitly state dtype in the supplied table, so its MKL rows are treated as reference values but not asserted to be FP16.
- This environment exposes 8 hardware threads; 16T is oversubscribed.