#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/tmac_env.sh"

ADAPTER="$PROJECT_ROOT/official_compare/adapters/profile_compare.py"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$PROJECT_ROOT/official_compare/results/runs/$RUN_ID"

VLA_DIR="$RUN_ROOT/official_vla"
HACKMD_DIR="$RUN_ROOT/official_hackmd"

mkdir -p "$VLA_DIR" "$HACKMD_DIR"

cd "$TMAC_OFFICIAL_ROOT"

echo "===== START VLA COMPARE ====="

for b in 2 4; do
    for t in 1 4 8; do
        out="$VLA_DIR/w${b}_t${t}"

        echo "========================================"
        echo " VLA W${b} T${t}"
        echo "========================================"

        mkdir -p "$out"

        python "$ADAPTER" \
            --suite vla_compare \
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

echo "===== START HACKMD COMPARE ====="

for b in 2 3 4; do
    for t in 1 2 4 8; do
        out="$HACKMD_DIR/w${b}_t${t}"

        echo "========================================"
        echo " HACKMD W${b} T${t}"
        echo "========================================"

        mkdir -p "$out"

        python "$ADAPTER" \
            --suite hackmd \
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

echo "===== ALL BENCHMARKS DONE ====="
echo "RESULTS: $RUN_ROOT"
