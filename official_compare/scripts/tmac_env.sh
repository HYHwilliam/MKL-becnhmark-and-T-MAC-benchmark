#!/bin/bash

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/benchmark_project}"
TMAC_OFFICIAL_ROOT="${TMAC_OFFICIAL_ROOT:-$HOME/tmac_external/T-MAC-official-x86}"
TMAC_CONDA_ENV="${TMAC_CONDA_ENV:-tvm-build}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: conda setup not found: $CONDA_SH" >&2
    return 1 2>/dev/null || exit 1
fi

if [ ! -d "$TMAC_OFFICIAL_ROOT" ]; then
    echo "ERROR: T-MAC repository not found: $TMAC_OFFICIAL_ROOT" >&2
    return 1 2>/dev/null || exit 1
fi

source "$CONDA_SH"
conda activate "$TMAC_CONDA_ENV"

LLVM_BIN="$TMAC_OFFICIAL_ROOT/build/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04/bin"

if [ ! -x "$LLVM_BIN/llvm-config" ]; then
    echo "ERROR: LLVM not found: $LLVM_BIN/llvm-config" >&2
    return 1 2>/dev/null || exit 1
fi

export PATH="$LLVM_BIN:$PATH"

export PYTHONPATH="$TMAC_OFFICIAL_ROOT/python:$TMAC_OFFICIAL_ROOT/3rdparty/tvm/python"

export PROJECT_ROOT
export TMAC_OFFICIAL_ROOT
