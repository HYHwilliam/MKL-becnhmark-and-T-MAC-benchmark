#!/usr/bin/env python3
"""
compare_tmac_mkl.py

Runs and compares:
  ./t_mac_gemm_w234
  ./mkl_fp16_gemm 1

Outputs:
  - Console comparison table
  - benchmark_comparison.csv
  - Raw logs:
      tmac_gemm_benchmark.log
      mkl_gemm_benchmark.log

Speedup definition:
  latency_speedup = MKL_latency_ms / T-MAC_latency_ms

Interpretation:
  > 1.0 : T-MAC is faster
  = 1.0 : same latency
  < 1.0 : MKL is faster
"""
from __future__ import annotations
import argparse
import csv
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SHAPE_PATTERN = re.compile(r"^\s*(\d+)x(\d+)x(\d+)\s+([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s+GFLOPS(?:.*?\biterations=(\d+))?")
TMAC_BITS_PATTERN = re.compile(r"T-MAC\s+W([234])A16", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"^===\s*(.*?)\s*===\s*$")

@dataclass(frozen=True)
class Shape:
    m: int
    k: int
    n: int

    def text(self) -> str:
        return f"{self.m}x{self.k}x{self.n}"

@dataclass
class BenchmarkResult:
    implementation: str
    bits: Optional[int]
    section: str
    shape: Shape
    latency_ms: float
    gflops: float
    iterations: Optional[int]

@dataclass
class ComparisonRow:
    section: str
    shape: Shape
    bits: int
    mkl_latency_ms: float
    tmac_latency_ms: float
    latency_speedup: float
    mkl_gflops: float
    tmac_gflops: float
    gflops_ratio: float
    tmac_iterations: Optional[int]

def run_command(command: List[str], timeout_seconds: Optional[int]) -> str:
    print(f"$ {' '.join(command)}", flush=True)
    try:
        completed = subprocess.run(
            command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"找不到執行檔：{command[0]}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"執行逾時：{' '.join(command)}") from error

    if completed.returncode != 0:
        print(completed.stdout)
        raise RuntimeError(f"執行失敗，return code={completed.returncode}: {' '.join(command)}")

    return completed.stdout

def parse_tmac_log(text: str) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    current_bits: Optional[int] = None
    current_section = "Unknown"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        bits_match = TMAC_BITS_PATTERN.search(line)
        if bits_match:
            current_bits = int(bits_match.group(1))
            continue

        section_match = SECTION_PATTERN.match(line)
        if section_match:
            current_section = section_match.group(1)
            continue

        shape_match = SHAPE_PATTERN.match(line)
        if not shape_match:
            continue

        if current_bits is None:
            raise ValueError("解析到 T-MAC 數據前尚未找到 W2/W3/W4 標題。")

        m, k, n = map(int, shape_match.group(1, 2, 3))
        results.append(
            BenchmarkResult(
                implementation=f"T-MAC W{current_bits}A16",
                bits=current_bits,
                section=current_section,
                shape=Shape(m, k, n),
                latency_ms=float(shape_match.group(4)),
                gflops=float(shape_match.group(5)),
                iterations=int(shape_match.group(6)) if shape_match.group(6) else None,
            )
        )

    if not results:
        raise ValueError("沒有在 T-MAC 日誌中解析到任何測試結果。")
    return results

def parse_mkl_log(text: str) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    current_section = "Unknown"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        section_match = SECTION_PATTERN.match(line)
        if section_match:
            current_section = section_match.group(1)
            continue

        shape_match = SHAPE_PATTERN.match(line)
        if not shape_match:
            continue

        m, k, n = map(int, shape_match.group(1, 2, 3))
        results.append(
            BenchmarkResult(
                implementation="oneMKL W16A16",
                bits=None,
                section=current_section,
                shape=Shape(m, k, n),
                latency_ms=float(shape_match.group(4)),
                gflops=float(shape_match.group(5)),
                iterations=int(shape_match.group(6)) if shape_match.group(6) else None,
            )
        )

    if not results:
        raise ValueError("沒有在 MKL 日誌中解析到任何測試結果。")
    return results

def build_comparisons(tmac_results: Iterable[BenchmarkResult], mkl_results: Iterable[BenchmarkResult]) -> Tuple[List[ComparisonRow], List[str]]:
    mkl_map: Dict[Shape, BenchmarkResult] = {result.shape: result for result in mkl_results}
    rows: List[ComparisonRow] = []
    missing: List[str] = []

    for tmac in tmac_results:
        mkl = mkl_map.get(tmac.shape)
        if mkl is None:
            missing.append(f"{tmac.implementation}: {tmac.shape.text()}")
            continue

        speedup = mkl.latency_ms / tmac.latency_ms if tmac.latency_ms > 0.0 else float("inf")
        latency_reduction = ((mkl.latency_ms - tmac.latency_ms) / mkl.latency_ms * 100.0) if mkl.latency_ms > 0.0 else 0.0
        gflops_ratio = tmac.gflops / mkl.gflops if mkl.gflops > 0.0 else float("inf")

        rows.append(
            ComparisonRow(
                section=tmac.section,
                shape=tmac.shape,
                bits=tmac.bits or 0,
                mkl_latency_ms=mkl.latency_ms,
                tmac_latency_ms=tmac.latency_ms,
                latency_speedup=speedup,
                mkl_gflops=mkl.gflops,
                tmac_gflops=tmac.gflops,
                gflops_ratio=gflops_ratio,
                tmac_iterations=tmac.iterations,
            )
        )
    return rows, missing

def print_table(rows: List[ComparisonRow]) -> None:
    header = (
        f"{'Shape':<20}{'Bit':>5}{'MKL ms':>12}{'T-MAC ms':>12}{'Speedup':>11}"
        f"{'MKL GF':>12}{'T-MAC GF':>12}{'GF Ratio':>11}{'Iter':>7}"
    )
    separator = "-" * len(header)
    current_section: Optional[str] = None

    for row in rows:
        if row.section != current_section:
            current_section = row.section
            print(f"\n=== {current_section} ===")
            print(header)
            print(separator)

        iteration_text = str(row.tmac_iterations) if row.tmac_iterations is not None else "-"
        print(
            f"{row.shape.text():<20}{( 'W' + str(row.bits) ):>5}{row.mkl_latency_ms:>12.6f}"
            f"{row.tmac_latency_ms:>12.6f}{row.latency_speedup:>10.3f}x"
            f"{row.mkl_gflops:>12.3f}"
            f"{row.tmac_gflops:>12.3f}{row.gflops_ratio:>10.3f}x{iteration_text:>7}"
        )

def print_summary(rows: List[ComparisonRow]) -> None:
    print("\n==================================================================")
    print("Summary")
    print("==================================================================")

    for bits in (2, 3, 4):
        bit_rows = [row for row in rows if row.bits == bits]
        if not bit_rows:
            continue

        speedups = [row.latency_speedup for row in bit_rows]
        faster_count = sum(row.latency_speedup > 1.0 for row in bit_rows)
        best = max(bit_rows, key=lambda row: row.latency_speedup)
        worst = min(bit_rows, key=lambda row: row.latency_speedup)
        geometric_mean = statistics.geometric_mean(speedups) if all(value > 0.0 for value in speedups) else float("nan")

        print(
            f"W{bits}: cases={len(bit_rows)}, T-MAC faster={faster_count}/{len(bit_rows)}, "
            f"median speedup={statistics.median(speedups):.3f}x, "
            f"geomean speedup={geometric_mean:.3f}x, "
        )
        print(f"    best : {best.shape.text()} {best.latency_speedup:.3f}x")
        print(f"    worst: {worst.shape.text()} {worst.latency_speedup:.3f}x")

def write_csv(rows: List[ComparisonRow], path: Path) -> None:
    fieldnames = [
        "section", "M", "K", "N", "tmac_bits", "mkl_precision", "tmac_precision",
        "mkl_latency_ms", "tmac_latency_ms", "latency_speedup_mkl_over_tmac",
        "mkl_gflops", "tmac_gflops",
        "tmac_over_mkl_gflops_ratio", "tmac_iterations"
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "section": row.section,
                "M": row.shape.m,
                "K": row.shape.k,
                "N": row.shape.n,
                "tmac_bits": row.bits,
                "mkl_precision": "W16A16",
                "tmac_precision": f"W{row.bits}A16",
                "mkl_latency_ms": row.mkl_latency_ms,
                "tmac_latency_ms": row.tmac_latency_ms,
                "latency_speedup_mkl_over_tmac": row.latency_speedup,
                "mkl_gflops": row.mkl_gflops,
                "tmac_gflops": row.tmac_gflops,
                "tmac_over_mkl_gflops_ratio": row.gflops_ratio,
                "tmac_iterations": row.tmac_iterations,
            })

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and compare T-MAC W2/W3/W4 against oneMKL FP16 GEMM.")
    parser.add_argument("--tmac", default="./t_mac_gemm_w234", help="T-MAC benchmark executable")
    parser.add_argument("--mkl", default="./mkl_fp16_gemm", help="MKL benchmark executable")
    parser.add_argument("--mkl-threads", type=int, default=1, help="Number of MKL threads")
    parser.add_argument("--tmac-log", type=Path, help="Use an existing T-MAC log instead of running the executable")
    parser.add_argument("--mkl-log", type=Path, help="Use an existing MKL log instead of running the executable")
    parser.add_argument("--output", type=Path, default=Path("benchmark_comparison.csv"), help="Output CSV path")
    parser.add_argument("--timeout", type=int, default=None, help="Per-command timeout in seconds")
    return parser.parse_args()

def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.tmac_log:
            tmac_text = arguments.tmac_log.read_text(encoding="utf-8", errors="replace")
        else:
            tmac_text = run_command([arguments.tmac], arguments.timeout)
            Path("tmac_gemm_benchmark.log").write_text(tmac_text, encoding="utf-8")

        if arguments.mkl_log:
            mkl_text = arguments.mkl_log.read_text(encoding="utf-8", errors="replace")
        else:
            mkl_text = run_command([arguments.mkl, str(arguments.mkl_threads)], arguments.timeout)
            Path("mkl_gemm_benchmark.log").write_text(mkl_text, encoding="utf-8")

        tmac_results = parse_tmac_log(tmac_text)
        mkl_results = parse_mkl_log(mkl_text)

        rows, missing = build_comparisons(tmac_results, mkl_results)
        expected_mkl_results = 80
        expected_tmac_results = 240
        expected_comparisons = 240

        if len(mkl_results) != expected_mkl_results:
            raise RuntimeError(
                f"MKL result count mismatch: "
                f"expected={expected_mkl_results}, "
                f"actual={len(mkl_results)}"
            )

        if len(tmac_results) != expected_tmac_results:
            raise RuntimeError(
                f"T-MAC result count mismatch: "
                f"expected={expected_tmac_results}, "
                f"actual={len(tmac_results)}"
            )

        if len(rows) != expected_comparisons:
            raise RuntimeError(
                f"Comparison count mismatch: "
                f"expected={expected_comparisons}, "
                f"actual={len(rows)}"
            )
        rows.sort(key=lambda row: (row.bits, row.section, row.shape.n, row.shape.m, row.shape.k))

        if not rows:
            raise RuntimeError("T-MAC 與 MKL 沒有任何相同 shape 可比較。")

        print_table(rows)
        print_summary(rows)
        write_csv(rows, arguments.output)

        print(f"\nCSV written to: {arguments.output}")

        if missing:
            print("\n以下 T-MAC shape 在 MKL 結果中不存在：", file=sys.stderr)
            for item in missing:
                print(f"  - {item}", file=sys.stderr)
            return 2

        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())