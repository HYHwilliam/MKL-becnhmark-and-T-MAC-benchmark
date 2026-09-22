#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/tmac_env.sh"

ADAPTER="$PROJECT_ROOT/official_compare/adapters/profile_compare.py"
VLA_SOURCE="$PROJECT_ROOT/x86_GEMM/vla_compare/tmac_vla_benchmark.cpp"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$PROJECT_ROOT/official_compare/results/runs/$RUN_ID"

STANDALONE_DIR="$RUN_ROOT/standalone_vla"
OFFICIAL_DIR="$RUN_ROOT/official_hackmd_safe"
BIN_DIR="$RUN_ROOT/bin"

mkdir -p "$STANDALONE_DIR" "$OFFICIAL_DIR" "$BIN_DIR"

echo "============================================================"
echo " RUN ID: $RUN_ID"
echo " RESULTS: $RUN_ROOT"
echo "============================================================"

echo "============================================================"
echo " PART 1: Standalone W234 - VLA shapes"
echo "============================================================"

g++ -O3 -std=c++17 \
    -mavx2 -mfma -mf16c -fopenmp \
    "$VLA_SOURCE" \
    -o "$BIN_DIR/tmac_vla_benchmark"

for b in 2 4; do
    echo "============================================================"
    echo " STANDALONE VLA W${b}"
    echo "============================================================"

    "$BIN_DIR/tmac_vla_benchmark" \
        --bits "$b" \
        --threads 1,4,8 \
        --autotune \
        | tee "$STANDALONE_DIR/tmac_vla_w${b}.log"
done

echo "============================================================"
echo " PART 2: Official T-MAC - HackMD safe"
echo "============================================================"

cd "$TMAC_OFFICIAL_ROOT"

for b in 2 3 4; do
    for t in 1 2 4 8; do
        out="$OFFICIAL_DIR/w${b}_t${t}"

        echo "============================================================"
        echo " OFFICIAL HACKMD SAFE W${b} T${t}"
        echo "============================================================"

        mkdir -p "$out"

        python "$ADAPTER" \
            --suite hackmd_safe \
            --bits "$b" \
            --threads "$t" \
            -o "$out" \
            -k qgemm_lut \
            -t \
            -gs 128 \
            -ags 64 \
            -mg -1
    done
done

echo "============================================================"
echo " ALL BENCHMARKS DONE"
echo " RESULTS: $RUN_ROOT"
echo "============================================================"
