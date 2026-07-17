#!/usr/bin/env python3
"""
Usage:
    ./t_mac_benchmark > /tmp/tmac.log
    ./mkl_benchmark    > /tmp/mkl.log
    python3 scripts/parse_benchmark_log.py --mkl /tmp/mkl.log --tmac /tmp/tmac.log \
        -o benchmark_results.csv
"""
import argparse
import csv
import re
import sys

LINE_RE = re.compile(r"^(\d+)x\d+x1\s+([\d.]+)\s+([\d.]+)\s+GFLOPS")


def parse_log(path):
    results = {}
    with open(path) as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if m:
                size, latency_ms, gflops = m.groups()
                results[int(size)] = (float(latency_ms), float(gflops))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mkl", required=True, help="path to mkl_benchmark stdout log")
    ap.add_argument("--tmac", required=True, help="path to t_mac_benchmark stdout log")
    ap.add_argument("-o", "--output", default="benchmark_results.csv")
    args = ap.parse_args()

    mkl = parse_log(args.mkl)
    tmac = parse_log(args.tmac)

    sizes = sorted(set(mkl) & set(tmac))
    if not sizes:
        sys.exit("No matching matrix sizes found in both logs — check the input files.")
    missing = (set(mkl) ^ set(tmac))
    if missing:
        print(f"warning: sizes present in only one log, skipped: {sorted(missing)}", file=sys.stderr)

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["matrix_size", "mkl_latency_ms", "mkl_gflops", "tmac_latency_ms", "tmac_gflops"])
        for size in sizes:
            mkl_lat, mkl_gf = mkl[size]
            tmac_lat, tmac_gf = tmac[size]
            w.writerow([size, mkl_lat, mkl_gf, tmac_lat, tmac_gf])

    print(f"wrote {len(sizes)} rows to {args.output}")


if __name__ == "__main__":
    main()
