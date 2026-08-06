#!/usr/bin/env python3
import argparse
import csv
import math
import re
import statistics
from pathlib import Path

RESULT_PATTERN = re.compile(
    r'RESULT category="(?P<category>[^"]+)" '
    r'bit=W(?P<bit>[234]) '
    r'shape=(?P<M>\d+)x(?P<K>\d+)x(?P<N>\d+) '
    r'threads=(?P<threads>\d+) '
    r'(?:active_threads=(?P<active_threads>\d+) )?'
    r'(?P<rest>.*)'
)

NUMBER_PATTERN = re.compile(r'([A-Za-z0-9_]+)=([0-9.eE+-]+)')


def parse_log(path: str) -> dict:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Log file not found: {file_path}")

    rows = {}
    for line_number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        match = RESULT_PATTERN.search(line)
        if not match:
            continue

        fields = match.groupdict()
        numbers = {name: float(value) for name, value in NUMBER_PATTERN.findall(fields["rest"])}
        required = {"total_ms", "gflops", "checksum"}
        missing = required - numbers.keys()
        if missing:
            raise ValueError(f"{file_path}:{line_number}: missing fields {sorted(missing)}")

        row = {
            "category": fields["category"],
            "bit": int(fields["bit"]),
            "M": int(fields["M"]),
            "K": int(fields["K"]),
            "N": int(fields["N"]),
            "threads": int(fields["threads"]),
            "active_threads": int(fields["active_threads"]) if fields["active_threads"] is not None else int(fields["threads"]),
            "total_ms": numbers["total_ms"],
            "gflops": numbers["gflops"],
            "checksum": numbers["checksum"],
            "preprocess_ms": numbers.get("preprocess_ms", math.nan),
            "kernel_ms": numbers.get("kernel_ms", math.nan),
            "p90_ms": numbers.get("p90_ms", math.nan),
            "samples": int(numbers.get("samples", 0)),
            "bm": int(numbers.get("bm", 0)),
            "bn": int(numbers.get("bn", 0)),
            "kfactor": int(numbers.get("kfactor", 0)),
            "autotuned": int(numbers.get("autotuned", 0)),
            "tuning_ms": numbers.get("tuning_ms", 0.0),
        }

        key = (row["bit"], row["M"], row["K"], row["N"], row["threads"])
        if key in rows:
            raise ValueError(f"{file_path}:{line_number}: duplicate RESULT for {key}")
        rows[key] = row

    if not rows:
        raise ValueError(f"No RESULT records found in {file_path}")
    return rows


def geomean(values: list[float]) -> float:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    return math.exp(sum(math.log(value) for value in positive) / len(positive)) if positive else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare T-MAC and oneMKL FP16 benchmark logs.")
    parser.add_argument("--tmac-log", required=True)
    parser.add_argument("--mkl-log", required=True)
    parser.add_argument("--output", default="comparison_mt.csv")
    args = parser.parse_args()

    tmac = parse_log(args.tmac_log)
    mkl = parse_log(args.mkl_log)
    common_keys = sorted(set(tmac) & set(mkl))

    if not common_keys:
        raise RuntimeError("No matching bit/M/K/N/thread cases were found.")

    missing_tmac = sorted(set(mkl) - set(tmac))
    missing_mkl = sorted(set(tmac) - set(mkl))
    if missing_tmac:
        print(f"Warning: {len(missing_tmac)} MKL cases have no T-MAC match.")
    if missing_mkl:
        print(f"Warning: {len(missing_mkl)} T-MAC cases have no MKL match.")

    rows = []
    for key in common_keys:
        t = tmac[key]
        m = mkl[key]
        if t["active_threads"] != t["threads"]:
            print(f"Warning: T-MAC requested {t['threads']} threads but used {t['active_threads']} for W{t['bit']} {t['M']}x{t['K']}x{t['N']}.")
        checksum_abs_diff = abs(t["checksum"] - m["checksum"])
        checksum_scale = max(1.0, abs(m["checksum"]))
        rows.append({
            "category": t["category"],
            "bit": t["bit"],
            "M": t["M"],
            "K": t["K"],
            "N": t["N"],
            "threads": t["threads"],
            "tmac_active_threads": t["active_threads"],
            "tmac_total_ms": t["total_ms"],
            "tmac_bm": t["bm"],
            "tmac_bn": t["bn"],
            "tmac_kfactor": t["kfactor"],
            "tmac_autotuned": t["autotuned"],
            "tmac_tuning_ms": t["tuning_ms"],
            "tmac_preprocess_ms": t["preprocess_ms"],
            "tmac_kernel_ms": t["kernel_ms"],
            "mkl_total_ms": m["total_ms"],
            "speedup": m["total_ms"] / t["total_ms"],
            "tmac_gflops": t["gflops"],
            "mkl_gflops": m["gflops"],
            "tmac_p90_ms": t["p90_ms"],
            "mkl_p90_ms": m["p90_ms"],
            "tmac_checksum": t["checksum"],
            "mkl_checksum": m["checksum"],
            "checksum_abs_diff": checksum_abs_diff,
            "checksum_rel_diff": checksum_abs_diff / checksum_scale,
        })

    fieldnames = list(rows[0].keys())
    with Path(args.output).open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Matched cases: {len(rows)}")
    thread_values = sorted({row["threads"] for row in rows})
    for bit in (2, 3, 4):
        for threads in thread_values:
            values = [row["speedup"] for row in rows if row["bit"] == bit and row["threads"] == threads]
            if values:
                print(
                    f"W{bit} threads={threads}: cases={len(values)} "
                    f"median={statistics.median(values):.3f}x "
                    f"geomean={geomean(values):.3f}x "
                    f"faster={sum(value > 1.0 for value in values)}/{len(values)}"
                )

    print(f"CSV written to: {args.output}")


if __name__ == "__main__":
    main()