#!/usr/bin/env bash
set -euo pipefail

BITS_LIST="${BITS_LIST:-2 3 4}"
THREADS_LIST="1 2 4 8"

make cross

for bits in ${BITS_LIST}; do
    for threads in ${THREADS_LIST}; do
        echo
        echo "================================================================"
        echo "W${bits}A16 threads=${threads}"
        echo "================================================================"
        qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu             ./build/arm64/tmac_neon_mt_fixed_wxa16             --bits "${bits}"             --threads "${threads}"             --verify-only
    done
done
