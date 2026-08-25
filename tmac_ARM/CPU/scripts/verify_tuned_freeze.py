#!/usr/bin/env python3
from pathlib import Path
import sys

BASE = Path("src/tmac_neon_mt_fixed_wxa16.cpp")
TUNED = Path("src/tmac_neon_mt_tuned_wxa16.cpp")

def extract_braced(text: str, marker: str) -> str:
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"missing marker: {marker}")
    brace = text.find("{", start)
    if brace < 0:
        raise RuntimeError(f"missing opening brace: {marker}")
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise RuntimeError(f"unterminated block: {marker}")

if not BASE.exists() or not TUNED.exists():
    print("ERROR: run this script inside tmac_ARM_benchmark with both MT sources present.")
    sys.exit(2)

base = BASE.read_text(encoding="utf-8")
tuned = TUNED.read_text(encoding="utf-8")

markers = [
    "class StaticThreadPool",
    "struct Runtime",
    "inline ParallelAxis official_parallel_axis",
    "inline void update_lut_scale_from_32",
    "inline float official_horizontal_sum",
    "inline void construct_quantized_luts_64",
    "inline void build_activation_luts",
    "inline float16x8_t reconstruct_lookup",
    "template <int Bits>\ninline void compute_tile",
    "template <int Bits>\n__attribute__((noinline)) void tmac_neon_mt(",
    "template <int Bits>\n__attribute__((noinline)) void frozen_single_thread_neon",
]

ok = True
for marker in markers:
    same = extract_braced(base, marker) == extract_braced(tuned, marker)
    print(f"{marker.splitlines()[-1]:64s} {'PASS' if same else 'FAIL'}")
    ok &= same

required_tuning = [
    "tmac_neon_mt_compute_only_for_tuning",
    "official_schedule_candidates<Bits>",
    "One discarded warmup",
    "options.tune_number",
    "options.tune_repeat",
    "tune_cooldown_ms",
    "mean_ms < best.mean_ms",
]
for token in required_tuning:
    present = token in tuned
    print(f"TUNING {token:55s} {'PASS' if present else 'FAIL'}")
    ok &= present

print("TUNED FREEZE CHECK:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
