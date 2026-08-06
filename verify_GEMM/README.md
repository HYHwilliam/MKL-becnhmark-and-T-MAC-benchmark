T-MAC W2/W3/W4A16 FP16 Tiled GEMM v3

此版本為 AVX2/F16C/OpenMP 的 standalone C++ 實作，對應 T-MAC 官方預設 symmetric GPTQ-like 路徑：

g=4

INT8 LUT

FP16 activation、FP16 weight scale、FP16 output

group_size=128

act_group_size=64

zero_point=false

fast_aggregation=false

W2/W3/W4 bit-plane decomposition

官方 bm/bn/kfactor 候選集合

依 bm/kfactor/SIMD 進行離線 weight permutation

單執行緒與 OpenMP 多執行緒

LUT preprocessor 保持 sequential

官方實作透過 TVM tensorization、LLVM code generation 與 AutoTVM tuning 產生 shape-specific kernel。本版本對齊 compute、資料格式、候選 schedule 與平行軸邏輯，但不宣稱機器碼與官方 generated kernel 完全相同。

v3 主要優化

上一版在每一個 K tile 都會讀寫大型 FP32 tile accumulator。v3 改為：

每次處理 32 個 expanded outputs。

使用 4 個 __m256 accumulator 保留完整 K reduction。

完成全部 K 後才將結果寫回一次。

LUT vector 在使用時即載入，不再保留 16 個 XMM LUT 暫存陣列。

保留原本官方順序 reference kernel，verifier 會逐 output 比對 optimized kernel。

在本測試環境的 quick cases 中，相較 v2：

單執行緒通常提升約 1.35～1.45 倍。

2／4 執行緒通常提升約 1.2～2.0 倍。

W2、W3、W4 的計算量仍依 bit-width 合理增加。

檔案

tmac_official_avx2.hpp：FP16、LUT、packing、reference kernel、register-blocked kernel 與 OpenMP 核心

t_mac_gemm_w234.cpp：T-MAC benchmark，支援 heuristic schedule 與 offline autotune

verify_tmac_gemm_independent.cpp：FP16 Ground Truth、edge cases、thread determinism、reference/optimized kernel regression

mkl_fp16_gemm_mt.cpp：oneMKL FP16 baseline

compare_tmac_mkl_mt.py：配對相同 bit/M/K/N/thread 並輸出 CSV

verify_official_weight_layout.py 不列入必要交付檔案。官方 layout 對照已於開發階段完成；正式驗證由 C++ packing decoder、independent Ground Truth 與 reference/optimized kernel comparison 負責。

編譯

g++ -O3 -std=c++17 -mavx2 -mfma -mf16c -fopenmp \
    -Wall -Wextra -Wpedantic \
    t_mac_gemm_w234.cpp -o t_mac_gemm_w234

g++ -O2 -std=c++17 -mavx2 -mfma -mf16c -fopenmp \
    -Wall -Wextra -Wpedantic \
    verify_tmac_gemm_independent.cpp \
    -o verify_tmac_gemm_independent

驗證

./verify_tmac_gemm_independent | tee verify_tmac_gemm_independent.log

預期：

packing=PASS
thread_mismatches=0
kernel_mismatches=0
FINAL W2=PASS W3=PASS W4=PASS

kernel_mismatches=0 代表 register-blocked optimized kernel 與原本 reference kernel 的 FP16 output 完全一致。

OpenMP 設定

export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=ACTIVE

T-MAC benchmark

使用 heuristic schedule：

./t_mac_gemm_w234 --quick --threads 1,2,4,8
./t_mac_gemm_w234 --threads 1,2,4,8 | tee tmac_fp16_mt.log

使用官方候選集合進行 offline autotune：

./t_mac_gemm_w234 --quick --threads 1,2,4,8 --autotune
./t_mac_gemm_w234 --threads 1,2,4,8 --autotune | tee tmac_fp16_mt.log

--autotune 會針對每個 bit/M/K/N/thread 測試所有有效的 bm/bn/kfactor 候選，選擇 median total latency 最低者。調優時間不計入 total_ms，但會輸出：

autotuned=1
tuning_ms=...

完整 960 組測試使用 autotune 會花較長時間，屬於與官方 AutoTVM 類似的離線工作。建議先用 --quick --autotune 確認環境。

編譯 oneMKL

source /opt/intel/oneapi/setvars.sh

g++ -O3 -std=c++17 -mavx2 -mfma -mf16c -fopenmp \
    mkl_fp16_gemm_mt.cpp \
    -I${MKLROOT}/include \
    -L${MKLROOT}/lib/intel64 \
    -Wl,--no-as-needed \
    -lmkl_intel_lp64 \
    -lmkl_gnu_thread \
    -lmkl_core \
    -lgomp -lpthread -lm -ldl \
    -o mkl_fp16_gemm_mt

./mkl_fp16_gemm_mt --threads 1,2,4,8 | tee mkl_fp16_mt.log

比較

python3 -m py_compile compare_tmac_mkl_mt.py

python3 compare_tmac_mkl_mt.py \
    --tmac-log tmac_fp16_mt.log \
    --mkl-log mkl_fp16_mt.log \
    --output comparison_mt.csv

CSV 會保留 bm、bn、kfactor、autotuned 與 tuning_ms，方便分析每個 shape 最後使用的 schedule。