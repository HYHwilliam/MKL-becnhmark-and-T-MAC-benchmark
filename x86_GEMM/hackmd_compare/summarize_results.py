from pathlib import Path
import csv
import re
import statistics
from collections import defaultdict, Counter


RESULT_DIR = Path(__file__).resolve().parent / "results"

EXPECTED_RUNS = 5
EXPECTED_ROWS_PER_LOG = 25
HARDWARE_THREADS = 8

SHAPES = {
    "256x256x256": {
        "category": "Small Square",
        "M": 256,
        "N": 256,
        "K": 256,
        "order": 0,
    },
    "1024x1024x1024": {
        "category": "Medium Square",
        "M": 1024,
        "N": 1024,
        "K": 1024,
        "order": 1,
    },
    "4096x4096x4096": {
        "category": "Large Square",
        "M": 4096,
        "N": 4096,
        "K": 4096,
        "order": 2,
    },
    "4096x1024x2048": {
        "category": "Rectangular",
        "M": 4096,
        "N": 1024,
        "K": 2048,
        "order": 3,
    },
    "1024x1024x512": {
        "category": "Medium Rectangular",
        "M": 1024,
        "N": 1024,
        "K": 512,
        "order": 4,
    },
}

THREADS = [1, 2, 4, 8, 16]

BACKEND_ORDER = {
    "MKL": 0,
    "T-MAC W2A16": 1,
    "T-MAC W3A16": 2,
    "T-MAC W4A16": 3,
}

EXPECTED_FILES = [
    *(f"mkl_fp16_run{i}.log" for i in range(1, 6)),
    *(f"tmac_w2_run{i}.log" for i in range(1, 6)),
    *(f"tmac_w3_run{i}.log" for i in range(1, 6)),
    *(f"tmac_w4_run{i}.log" for i in range(1, 6)),
]

FILE_BACKEND = {}
for i in range(1, 6):
    FILE_BACKEND[f"mkl_fp16_run{i}.log"] = "MKL"
    FILE_BACKEND[f"tmac_w2_run{i}.log"] = "T-MAC W2A16"
    FILE_BACKEND[f"tmac_w3_run{i}.log"] = "T-MAC W3A16"
    FILE_BACKEND[f"tmac_w4_run{i}.log"] = "T-MAC W4A16"

HACKMD_MKL_REFERENCE = {
    ("256x256x256", 1): 0.2365,
    ("256x256x256", 2): 0.2465,
    ("256x256x256", 4): 0.1462,
    ("256x256x256", 8): 0.3258,
    ("256x256x256", 16): 0.4081,

    ("1024x1024x1024", 1): 16.7570,
    ("1024x1024x1024", 2): 12.3002,
    ("1024x1024x1024", 4): 6.4828,
    ("1024x1024x1024", 8): 3.5757,
    ("1024x1024x1024", 16): 5.9278,

    ("4096x4096x4096", 1): 2071.9777,
    ("4096x4096x4096", 2): 1048.0653,
    ("4096x4096x4096", 4): 527.3677,
    ("4096x4096x4096", 8): 266.5019,
    ("4096x4096x4096", 16): 260.4321,

    ("4096x1024x2048", 1): 260.9551,
    ("4096x1024x2048", 2): 130.7603,
    ("4096x1024x2048", 4): 65.7123,
    ("4096x1024x2048", 8): 33.2634,
    ("4096x1024x2048", 16): 72.7828,

    ("1024x1024x512", 1): 3.410722,
    ("1024x1024x512", 2): 3.164299,
    ("1024x1024x512", 4): 3.135987,
    ("1024x1024x512", 8): 3.123567,
    ("1024x1024x512", 16): 3.145073,
}

HACKMD_FP16_CONFIRMED_SHAPE = "1024x1024x512"


def parse_value(line, key):
    match = re.search(rf'{re.escape(key)}=("[^"]*"|\S+)', line)
    if not match:
        return None

    value = match.group(1)
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]

    return value


def parse_result_line(line, filename):
    if not line.startswith("RESULT "):
        return None

    category = parse_value(line, "category")
    shape = parse_value(line, "shape")
    threads = parse_value(line, "threads")
    total_ms = parse_value(line, "total_ms")
    checksum = parse_value(line, "checksum")

    if None in (category, shape, threads, total_ms, checksum):
        return None

    backend_raw = parse_value(line, "backend")
    dtype_raw = parse_value(line, "dtype")
    bit = parse_value(line, "bit")

    if backend_raw == "MKL":
        backend = "MKL"
        dtype = dtype_raw or "fp16"
    elif bit in ("W2", "W3", "W4"):
        backend = f"T-MAC {bit}A16"
        dtype = f"{bit}A16"
    else:
        backend = "UNKNOWN"
        dtype = "UNKNOWN"

    result = {
        "file": filename,
        "backend": backend,
        "dtype": dtype,
        "category": category,
        "shape": shape,
        "threads": int(threads),
        "total_ms": float(total_ms),
        "checksum": float(checksum),
        "reported_gflops": None,
        "p90_ms": None,
        "preprocess_ms": None,
        "kernel_ms": None,
        "autotuned": None,
        "bm": None,
        "bn": None,
        "kfactor": None,
    }

    gflops = parse_value(line, "gflops")
    if gflops is not None:
        result["reported_gflops"] = float(gflops)

    for key in ("p90_ms", "preprocess_ms", "kernel_ms"):
        value = parse_value(line, key)
        if value is not None:
            result[key] = float(value)

    for key in ("bm", "bn", "kfactor", "autotuned"):
        value = parse_value(line, key)
        if value is not None:
            result[key] = int(value)

    return result


def logical_flops(shape):
    info = SHAPES[shape]
    return 2.0 * info["M"] * info["N"] * info["K"]


def calculated_gflops(shape, total_ms):
    return logical_flops(shape) / (total_ms / 1000.0) / 1.0e9


def expected_combinations():
    return {
        (shape, threads)
        for shape in SHAPES
        for threads in THREADS
    }


def load_logs():
    files = []
    rows = []

    for filename in EXPECTED_FILES:
        path = RESULT_DIR / filename
        if not path.exists():
            continue

        files.append(path)

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                parsed = parse_result_line(line.strip(), filename)
                if parsed is not None:
                    rows.append(parsed)

    return files, rows


def validate_inputs(files, rows):
    errors = []
    warnings = []

    found_names = {path.name for path in files}
    expected_names = set(EXPECTED_FILES)

    missing_files = sorted(expected_names - found_names)
    extra_matching_files = sorted(
        path.name
        for pattern in (
            "mkl_fp16_run*.log",
            "tmac_w2_run*.log",
            "tmac_w3_run*.log",
            "tmac_w4_run*.log",
        )
        for path in RESULT_DIR.glob(pattern)
        if path.name not in expected_names
    )

    if missing_files:
        errors.append(
            "Missing expected log files: " + ", ".join(missing_files)
        )

    if extra_matching_files:
        warnings.append(
            "Extra run-like log files ignored: "
            + ", ".join(extra_matching_files)
        )

    rows_by_file = defaultdict(list)
    for row in rows:
        rows_by_file[row["file"]].append(row)

    expected_pairs = expected_combinations()

    for filename in EXPECTED_FILES:
        if filename not in found_names:
            continue

        file_rows = rows_by_file[filename]

        if len(file_rows) != EXPECTED_ROWS_PER_LOG:
            errors.append(
                f"{filename}: expected {EXPECTED_ROWS_PER_LOG} RESULT rows, "
                f"found {len(file_rows)}."
            )

        expected_backend = FILE_BACKEND[filename]

        for row in file_rows:
            if row["backend"] != expected_backend:
                errors.append(
                    f"{filename}: backend mismatch. Expected {expected_backend}, "
                    f"found {row['backend']}."
                )

            if row["shape"] not in SHAPES:
                errors.append(
                    f"{filename}: unexpected shape {row['shape']}."
                )
                continue

            expected_category = SHAPES[row["shape"]]["category"]
            if row["category"] != expected_category:
                errors.append(
                    f"{filename}: category mismatch for {row['shape']}. "
                    f"Expected '{expected_category}', found '{row['category']}'."
                )

            if row["threads"] not in THREADS:
                errors.append(
                    f"{filename}: unexpected thread count {row['threads']}."
                )

            if row["backend"].startswith("T-MAC") and row["autotuned"] != 1:
                errors.append(
                    f"{filename}: T-MAC {row['shape']} {row['threads']}T "
                    "is not autotuned=1."
                )

            if row["reported_gflops"] is not None:
                expected_gflops = calculated_gflops(
                    row["shape"], row["total_ms"]
                )
                relative_error = abs(
                    row["reported_gflops"] - expected_gflops
                ) / max(abs(expected_gflops), 1e-12)

                if relative_error > 1e-5:
                    errors.append(
                        f"{filename}: GFLOPs mismatch for {row['shape']} "
                        f"{row['threads']}T. Reported={row['reported_gflops']:.6f}, "
                        f"recomputed={expected_gflops:.6f}."
                    )

        actual_pairs = [
            (row["shape"], row["threads"])
            for row in file_rows
            if row["shape"] in SHAPES and row["threads"] in THREADS
        ]

        pair_counter = Counter(actual_pairs)

        duplicates = sorted(
            pair
            for pair, count in pair_counter.items()
            if count > 1
        )
        missing_pairs = sorted(
            expected_pairs - set(actual_pairs),
            key=lambda pair: (
                SHAPES[pair[0]]["order"],
                THREADS.index(pair[1]),
            ),
        )

        if duplicates:
            errors.append(
                f"{filename}: duplicate shape/thread rows: "
                + ", ".join(f"{s}/{t}T" for s, t in duplicates)
            )

        if missing_pairs:
            errors.append(
                f"{filename}: missing shape/thread rows: "
                + ", ".join(f"{s}/{t}T" for s, t in missing_pairs)
            )

    if len(rows) != 20 * EXPECTED_ROWS_PER_LOG:
        errors.append(
            f"Expected 500 RESULT rows across 20 logs, found {len(rows)}."
        )

    return errors, warnings


def calculate_statistics(rows):
    groups = defaultdict(list)

    for row in rows:
        key = (
            row["backend"],
            row["dtype"],
            row["category"],
            row["shape"],
            row["threads"],
        )
        groups[key].append(row)

    summary = []

    for key, group in groups.items():
        backend, dtype, category, shape, threads = key

        total_values = [row["total_ms"] for row in group]
        total_mean = statistics.mean(total_values)
        total_median = statistics.median(total_values)
        total_std = statistics.stdev(total_values) if len(total_values) >= 2 else 0.0

        cv = (
            total_std / total_mean * 100.0
            if total_mean != 0.0
            else 0.0
        )

        entry = {
            "backend": backend,
            "dtype": dtype,
            "category": category,
            "shape": shape,
            "threads": threads,
            "runs": len(group),
            "total_mean_ms": total_mean,
            "total_median_ms": total_median,
            "total_std_ms": total_std,
            "total_min_ms": min(total_values),
            "total_max_ms": max(total_values),
            "cv_percent": cv,
            "gflops_mean": statistics.mean(
                calculated_gflops(shape, value)
                for value in total_values
            ),
            "gflops_median": calculated_gflops(
                shape, total_median
            ),
            "checksum_min": min(row["checksum"] for row in group),
            "checksum_max": max(row["checksum"] for row in group),
            "autotuned": None,
            "schedule": "",
            "schedule_unique_count": 0,
            "schedule_mode_count": 0,
        }

        if backend.startswith("T-MAC"):
            autotuned_values = [
                row["autotuned"]
                for row in group
                if row["autotuned"] is not None
            ]

            entry["autotuned"] = (
                bool(autotuned_values)
                and all(value == 1 for value in autotuned_values)
            )

            schedules = [
                (row["bm"], row["bn"], row["kfactor"])
                for row in group
                if (
                    row["bm"] is not None
                    and row["bn"] is not None
                    and row["kfactor"] is not None
                )
            ]

            if schedules:
                counter = Counter(schedules)
                schedule, count = counter.most_common(1)[0]
                unique_count = len(counter)

                entry["schedule_unique_count"] = unique_count
                entry["schedule_mode_count"] = count

                if count == len(schedules):
                    entry["schedule"] = (
                        f"stable: bm={schedule[0]},bn={schedule[1]},"
                        f"kfactor={schedule[2]} ({count}/{len(schedules)})"
                    )
                elif count >= 3:
                    entry["schedule"] = (
                        f"mostly stable: bm={schedule[0]},bn={schedule[1]},"
                        f"kfactor={schedule[2]} ({count}/{len(schedules)}), "
                        f"{unique_count} schedules"
                    )
                else:
                    entry["schedule"] = (
                        f"unstable: mode bm={schedule[0]},bn={schedule[1]},"
                        f"kfactor={schedule[2]} ({count}/{len(schedules)}), "
                        f"{unique_count} schedules"
                    )

        summary.append(entry)

    summary.sort(
        key=lambda row: (
            BACKEND_ORDER.get(row["backend"], 99),
            SHAPES.get(row["shape"], {}).get("order", 99),
            THREADS.index(row["threads"])
            if row["threads"] in THREADS
            else 99,
        )
    )

    return summary



def validate_summary(summary):
    errors = []
    warnings = []

    expected_groups = (
        len(BACKEND_ORDER)
        * len(SHAPES)
        * len(THREADS)
    )

    if len(summary) != expected_groups:
        errors.append(
            f"Expected {expected_groups} aggregated configurations, "
            f"found {len(summary)}."
        )

    for row in summary:
        if row["runs"] != EXPECTED_RUNS:
            errors.append(
                f"{row['backend']} {row['shape']} {row['threads']}T: "
                f"expected {EXPECTED_RUNS} runs, found {row['runs']}."
            )

        checksum_delta = abs(
            row["checksum_max"] - row["checksum_min"]
        )

        if checksum_delta > 1e-6:
            errors.append(
                f"{row['backend']} {row['shape']} {row['threads']}T: "
                f"checksum changed across runs "
                f"({row['checksum_min']:.6f} to "
                f"{row['checksum_max']:.6f})."
            )

        if (
            row["backend"].startswith("T-MAC")
            and row["autotuned"] is not True
        ):
            errors.append(
                f"{row['backend']} {row['shape']} {row['threads']}T: "
                "not all five runs are autotuned=1."
            )

        if row["cv_percent"] >= 20.0:
            warnings.append(
                f"High variation: {row['backend']} "
                f"{row['shape']} {row['threads']}T, "
                f"CV={row['cv_percent']:.1f}%."
            )

    return errors, warnings

def build_speedup(summary):
    lookup = {
        (row["backend"], row["shape"], row["threads"]): row
        for row in summary
    }

    speedups = []

    for backend in (
        "T-MAC W2A16",
        "T-MAC W3A16",
        "T-MAC W4A16",
    ):
        for shape in SHAPES:
            for threads in THREADS:
                mkl = lookup.get(("MKL", shape, threads))
                tmac = lookup.get((backend, shape, threads))

                if mkl is None or tmac is None:
                    continue

                speedup = (
                    mkl["total_median_ms"]
                    / tmac["total_median_ms"]
                )

                speedups.append(
                    {
                        "backend": backend,
                        "shape": shape,
                        "threads": threads,
                        "mkl_ms": mkl["total_median_ms"],
                        "tmac_ms": tmac["total_median_ms"],
                        "speedup": speedup,
                    }
                )

    return speedups


def build_hackmd_comparison(summary):
    lookup = {
        (row["backend"], row["shape"], row["threads"]): row
        for row in summary
    }

    comparisons = []

    for shape in SHAPES:
        for threads in THREADS:
            row = lookup.get(("MKL", shape, threads))
            reference_ms = HACKMD_MKL_REFERENCE.get((shape, threads))

            if row is None or reference_ms is None:
                continue

            local_ms = row["total_median_ms"]
            ratio = local_ms / reference_ms
            difference_percent = (ratio - 1.0) * 100.0

            comparisons.append(
                {
                    "shape": shape,
                    "threads": threads,
                    "reference_ms": reference_ms,
                    "local_ms": local_ms,
                    "local_over_reference": ratio,
                    "difference_percent": difference_percent,
                    "dtype_note": (
                        "FP16 confirmed"
                        if shape == HACKMD_FP16_CONFIRMED_SHAPE
                        else "HackMD first table dtype not explicitly stated"
                    ),
                }
            )

    return comparisons


def write_csv(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def write_hackmd_csv(summary):
    path = RESULT_DIR / "summary_hackmd.csv"

    rows = []
    for row in summary:
        rows.append(
            [
                row["backend"],
                row["dtype"],
                row["category"],
                row["shape"],
                row["threads"],
                f'{row["total_median_ms"]:.6f}',
                f'{row["gflops_median"]:.6f}',
            ]
        )

    write_csv(
        path,
        [
            "Backend",
            "Dtype",
            "Category",
            "Shape",
            "Threads",
            "Total (ms)",
            "GFLOPs",
        ],
        rows,
    )

    return path


def write_statistics_csv(summary):
    path = RESULT_DIR / "summary_statistics.csv"

    rows = []
    for row in summary:
        rows.append(
            [
                row["backend"],
                row["dtype"],
                row["category"],
                row["shape"],
                row["threads"],
                row["runs"],
                f'{row["total_mean_ms"]:.6f}',
                f'{row["total_median_ms"]:.6f}',
                f'{row["total_std_ms"]:.6f}',
                f'{row["total_min_ms"]:.6f}',
                f'{row["total_max_ms"]:.6f}',
                f'{row["cv_percent"]:.6f}',
                f'{row["gflops_mean"]:.6f}',
                f'{row["gflops_median"]:.6f}',
                f'{row["checksum_min"]:.6f}',
                f'{row["checksum_max"]:.6f}',
                row["autotuned"],
                row["schedule"],
            ]
        )

    write_csv(
        path,
        [
            "backend",
            "dtype",
            "category",
            "shape",
            "threads",
            "runs",
            "total_mean_ms",
            "total_median_ms",
            "total_std_ms",
            "total_min_ms",
            "total_max_ms",
            "cv_percent",
            "gflops_mean",
            "gflops_median",
            "checksum_min",
            "checksum_max",
            "autotuned",
            "schedule",
        ],
        rows,
    )

    return path


def write_speedup_csv(speedups):
    path = RESULT_DIR / "speedup_vs_mkl.csv"

    rows = [
        [
            row["backend"],
            row["shape"],
            row["threads"],
            f'{row["mkl_ms"]:.6f}',
            f'{row["tmac_ms"]:.6f}',
            f'{row["speedup"]:.4f}',
        ]
        for row in speedups
    ]

    write_csv(
        path,
        [
            "Backend",
            "Shape",
            "Threads",
            "MKL Median (ms)",
            "T-MAC Median (ms)",
            "Speedup vs MKL",
        ],
        rows,
    )

    return path


def write_hackmd_reference_csv(comparisons):
    path = RESULT_DIR / "hackmd_mkl_comparison.csv"

    rows = [
        [
            row["shape"],
            row["threads"],
            f'{row["reference_ms"]:.6f}',
            f'{row["local_ms"]:.6f}',
            f'{row["local_over_reference"]:.4f}',
            f'{row["difference_percent"]:.2f}',
            row["dtype_note"],
        ]
        for row in comparisons
    ]

    write_csv(
        path,
        [
            "Shape",
            "Threads",
            "HackMD MKL (ms)",
            "Local MKL Median (ms)",
            "Local / HackMD",
            "Latency Difference (%)",
            "Reference Note",
        ],
        rows,
    )

    return path


def generate_conclusion(summary, speedups):
    lines = []

    speedups_by_backend = defaultdict(list)
    for row in speedups:
        speedups_by_backend[row["backend"]].append(row)

    for backend in (
        "T-MAC W2A16",
        "T-MAC W3A16",
        "T-MAC W4A16",
    ):
        rows = speedups_by_backend.get(backend, [])

        if not rows:
            continue

        wins = [row for row in rows if row["speedup"] > 1.0]
        best = max(rows, key=lambda row: row["speedup"])

        lines.append(
            f"- {backend}: {len(wins)}/{len(rows)} cases beat MKL; "
            f"best = {best['speedup']:.3f}× at "
            f"{best['shape']} {best['threads']}T."
        )

    high_variation = [
        row
        for row in summary
        if row["cv_percent"] >= 20.0
    ]

    lines.append(
        f"- High-variation configurations (CV >= 20%): "
        f"{len(high_variation)}/{len(summary)}."
    )

    unstable_schedules = [
        row
        for row in summary
        if (
            row["backend"].startswith("T-MAC")
            and row["schedule_unique_count"] >= 3
            and row["schedule_mode_count"] <= 2
        )
    ]

    lines.append(
        f"- T-MAC configurations with notably unstable autotune schedules: "
        f"{len(unstable_schedules)}."
    )

    lines.append(
        f"- 16T is oversubscribed because the current environment exposes "
        f"{HARDWARE_THREADS} hardware threads."
    )

    return lines


def write_validation_report(errors, warnings):
    path = RESULT_DIR / "validation_report.txt"

    lines = []

    if errors:
        lines.append("VALIDATION: FAIL")
    else:
        lines.append("VALIDATION: PASS")

    lines.append("")

    if errors:
        lines.append("Errors:")
        lines.extend(f"- {item}" for item in errors)
        lines.append("")

    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in warnings)
        lines.append("")

    if not errors and not warnings:
        lines.append("No issues detected.")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_markdown(
    summary,
    speedups,
    comparisons,
    errors,
    warnings,
):
    path = RESULT_DIR / "summary_hackmd.md"

    lines = []

    lines.append("# HackMD Benchmark Summary")
    lines.append("")
    lines.append("Shape notation: **M×N×K**")
    lines.append("")
    lines.append(
        "Final `Total (ms)` = median of five independent run-level medians."
    )
    lines.append(
        "Final `GFLOPs` is recomputed from `2×M×N×K / median latency`."
    )
    lines.append("")

    lines.append("## Validation")
    lines.append("")
    lines.append(
        f"- Status: **{'FAIL' if errors else 'PASS'}**"
    )
    lines.append(
        f"- Expected logs: 20, each with 25 RESULT rows."
    )
    lines.append(
        f"- Expected total RESULT rows: 500."
    )

    if errors:
        for item in errors:
            lines.append(f"- ERROR: {item}")

    if warnings:
        for item in warnings:
            lines.append(f"- WARNING: {item}")

    lines.append("")

    lines.append("## HackMD-format Results")
    lines.append("")
    lines.append(
        "| Backend | Dtype | Category | Shape | Threads | Total (ms) | GFLOPs |"
    )
    lines.append(
        "|---|---|---|---|---:|---:|---:|"
    )

    for row in summary:
        lines.append(
            f'| {row["backend"]} '
            f'| {row["dtype"]} '
            f'| {row["category"]} '
            f'| {row["shape"]} '
            f'| {row["threads"]} '
            f'| {row["total_median_ms"]:.6f} '
            f'| {row["gflops_median"]:.6f} |'
        )

    lines.append("")
    lines.append("## T-MAC Speedup vs Local MKL")
    lines.append("")
    lines.append(
        "| Backend | Shape | Threads | MKL ms | T-MAC ms | Speedup |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")

    for row in speedups:
        lines.append(
            f'| {row["backend"]} '
            f'| {row["shape"]} '
            f'| {row["threads"]} '
            f'| {row["mkl_ms"]:.6f} '
            f'| {row["tmac_ms"]:.6f} '
            f'| {row["speedup"]:.3f}× |'
        )

    lines.append("")
    lines.append("## Local MKL vs HackMD MKL")
    lines.append("")
    lines.append(
        "| Shape | Threads | HackMD ms | Local ms | Local/HackMD | Difference | Note |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---|"
    )

    for row in comparisons:
        lines.append(
            f'| {row["shape"]} '
            f'| {row["threads"]} '
            f'| {row["reference_ms"]:.6f} '
            f'| {row["local_ms"]:.6f} '
            f'| {row["local_over_reference"]:.3f}× '
            f'| {row["difference_percent"]:+.1f}% '
            f'| {row["dtype_note"]} |'
        )

    lines.append("")
    lines.append("## Statistics")
    lines.append("")
    lines.append(
        "| Backend | Shape | Threads | Runs | Mean ms | Median ms | Std ms | Min ms | Max ms | CV (%) |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for row in summary:
        lines.append(
            f'| {row["backend"]} '
            f'| {row["shape"]} '
            f'| {row["threads"]} '
            f'| {row["runs"]} '
            f'| {row["total_mean_ms"]:.6f} '
            f'| {row["total_median_ms"]:.6f} '
            f'| {row["total_std_ms"]:.6f} '
            f'| {row["total_min_ms"]:.6f} '
            f'| {row["total_max_ms"]:.6f} '
            f'| {row["cv_percent"]:.2f} |'
        )

    lines.append("")
    lines.append("## Autotune Schedule Stability")
    lines.append("")
    lines.append(
        "| Backend | Shape | Threads | Schedule |"
    )
    lines.append("|---|---|---:|---|")

    for row in summary:
        if not row["backend"].startswith("T-MAC"):
            continue

        lines.append(
            f'| {row["backend"]} '
            f'| {row["shape"]} '
            f'| {row["threads"]} '
            f'| {row["schedule"]} |'
        )

    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.extend(generate_conclusion(summary, speedups))

    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "- Each benchmark executable performs its own warm-up and per-run sampling."
    )
    lines.append(
        "- Each configuration is executed in five independent complete runs."
    )
    lines.append(
        "- The final latency is the median of the five run-level medians."
    )
    lines.append(
        "- Mean, sample standard deviation, min, max and CV are retained for stability analysis."
    )
    lines.append(
        "- Speedup vs MKL = `MKL median latency / T-MAC median latency`; values > 1 mean T-MAC is faster."
    )
    lines.append(
        "- The 1024×1024×512 HackMD MKL reference is explicitly labeled FP16."
    )
    lines.append(
        "- The first HackMD OpenVINO-vs-MKL table does not explicitly state dtype in the supplied table, so its MKL rows are treated as reference values but not asserted to be FP16."
    )
    lines.append(
        f"- This environment exposes {HARDWARE_THREADS} hardware threads; 16T is oversubscribed."
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    files, rows = load_logs()
    errors, warnings = validate_inputs(files, rows)

    print(f"Found {len(files)} / 20 expected log files")
    print(f"Parsed {len(rows)} RESULT rows")

    if not rows:
        validation_path = write_validation_report(errors, warnings)
        print(f"Validation: {'FAIL' if errors else 'PASS'}")
        print(f"See: {validation_path}")
        print("No RESULT rows found.")
        return

    summary = calculate_statistics(rows)

    summary_errors, summary_warnings = validate_summary(summary)
    errors.extend(summary_errors)
    warnings.extend(summary_warnings)

    validation_path = write_validation_report(errors, warnings)

    print(f"Validation: {'FAIL' if errors else 'PASS'}")

    if errors:
        print("")
        print("Validation errors found.")
        print(f"See: {validation_path}")
        print("Summary files will still be generated for inspection.")
        print("")

    speedups = build_speedup(summary)
    comparisons = build_hackmd_comparison(summary)

    hackmd_csv = write_hackmd_csv(summary)
    statistics_csv = write_statistics_csv(summary)
    speedup_csv = write_speedup_csv(speedups)
    hackmd_reference_csv = write_hackmd_reference_csv(comparisons)
    markdown = write_markdown(
        summary,
        speedups,
        comparisons,
        errors,
        warnings,
    )

    print("")
    print("Generated:")
    print(hackmd_csv)
    print(statistics_csv)
    print(speedup_csv)
    print(hackmd_reference_csv)
    print(validation_path)
    print(markdown)


if __name__ == "__main__":
    main()
