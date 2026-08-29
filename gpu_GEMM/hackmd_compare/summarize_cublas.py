from pathlib import Path
import csv
import math
import re
import statistics
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent

GPU_RESULT_DIR = (
    BASE_DIR
    / "results"
    / "lab_rtx4070super"
)

CPU_STATS_FILE = (
    REPO_ROOT
    / "x86_GEMM"
    / "hackmd_compare"
    / "results"
    / "cpu_tmac_compare"
    / "lab_tmac_statistics.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "cublas_tmac_compare"
)

EXPECTED_RUNS = 5

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
PRIMARY_THREADS = [1, 2, 4, 8]

BACKENDS = [
    "T-MAC W2A16",
    "T-MAC W3A16",
    "T-MAC W4A16",
]

EXPECTED_GPU_FILES = [
    f"cublas_fp16_run{i}.log"
    for i in range(1, EXPECTED_RUNS + 1)
]


def parse_value(line, key):
    match = re.search(
        rf'{re.escape(key)}=("[^"]*"|\S+)',
        line,
    )

    if not match:
        return None

    value = match.group(1)

    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]

    return value


def logical_flops(shape):
    info = SHAPES[shape]

    return (
        2.0
        * info["M"]
        * info["N"]
        * info["K"]
    )


def calculated_gflops(shape, total_ms):
    return (
        logical_flops(shape)
        / (total_ms / 1000.0)
        / 1.0e9
    )


def expected_samples(shape):
    flops = logical_flops(shape)

    if flops < 1.0e8:
        return 21

    if flops < 1.0e9:
        return 11

    if flops < 1.0e10:
        return 7

    return 5


def geometric_mean(values):
    valid = [
        value
        for value in values
        if value > 0.0
    ]

    if not valid:
        return 0.0

    return math.exp(
        statistics.mean(
            math.log(value)
            for value in valid
        )
    )


def parse_gpu_result(line, filename):
    if not line.startswith("RESULT "):
        return None

    required_keys = [
        "backend",
        "dtype",
        "compute",
        "category",
        "shape",
        "total_ms",
        "p90_ms",
        "gflops",
        "samples",
        "verify_max_abs_error",
        "checksum",
    ]

    values = {
        key: parse_value(line, key)
        for key in required_keys
    }

    missing = [
        key
        for key, value in values.items()
        if value is None
    ]

    if missing:
        raise ValueError(
            "missing fields: "
            + ", ".join(missing)
        )

    return {
        "file": filename,
        "backend": values["backend"],
        "dtype": values["dtype"],
        "compute": values["compute"],
        "category": values["category"],
        "shape": values["shape"],
        "total_ms": float(
            values["total_ms"]
        ),
        "p90_ms": float(
            values["p90_ms"]
        ),
        "reported_gflops": float(
            values["gflops"]
        ),
        "samples": int(
            values["samples"]
        ),
        "verify_max_abs_error": float(
            values[
                "verify_max_abs_error"
            ]
        ),
        "checksum": float(
            values["checksum"]
        ),
    }


def load_gpu_logs():
    rows = []
    errors = []
    found_files = []

    for filename in EXPECTED_GPU_FILES:
        path = GPU_RESULT_DIR / filename

        if not path.exists():
            continue

        found_files.append(path)

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:
            for line_number, raw in enumerate(
                file,
                start=1,
            ):
                line = raw.strip()

                if not line.startswith("RESULT "):
                    continue

                try:
                    row = parse_gpu_result(
                        line,
                        filename,
                    )

                    if row is not None:
                        rows.append(row)

                except Exception as error:
                    errors.append(
                        f"{filename}:{line_number}: "
                        f"{error}"
                    )

    return found_files, rows, errors


def validate_gpu(
    found_files,
    rows,
    parse_errors,
):
    errors = []
    warnings = []

    errors.extend(
        "Parse error: " + error
        for error in parse_errors
    )

    found_names = {
        path.name
        for path in found_files
    }

    missing = sorted(
        set(EXPECTED_GPU_FILES)
        - found_names
    )

    if missing:
        errors.append(
            "Missing GPU logs: "
            + ", ".join(missing)
        )

    rows_by_file = defaultdict(list)

    for row in rows:
        rows_by_file[
            row["file"]
        ].append(row)

    for filename in EXPECTED_GPU_FILES:
        if filename not in found_names:
            continue

        file_rows = rows_by_file[
            filename
        ]

        if len(file_rows) != len(SHAPES):
            errors.append(
                f"{filename}: expected "
                f"{len(SHAPES)} RESULT rows, "
                f"found {len(file_rows)}."
            )

        seen_shapes = set()

        for row in file_rows:
            if row["backend"] != "cuBLAS":
                errors.append(
                    f"{filename}: unexpected "
                    f"backend={row['backend']}."
                )

            if row["dtype"] != "fp16":
                errors.append(
                    f"{filename}: unexpected "
                    f"dtype={row['dtype']}."
                )

            if row["compute"] != "fp32":
                errors.append(
                    f"{filename}: unexpected "
                    f"compute={row['compute']}."
                )

            shape = row["shape"]

            if shape not in SHAPES:
                errors.append(
                    f"{filename}: unexpected "
                    f"shape {shape}."
                )
                continue

            if shape in seen_shapes:
                errors.append(
                    f"{filename}: duplicate "
                    f"shape {shape}."
                )

            seen_shapes.add(shape)

            expected_category = (
                SHAPES[shape]["category"]
            )

            if (
                row["category"]
                != expected_category
            ):
                errors.append(
                    f"{filename}: category "
                    f"mismatch for {shape}."
                )

            if (
                row["samples"]
                != expected_samples(shape)
            ):
                errors.append(
                    f"{filename}: unexpected "
                    f"sample count for {shape}: "
                    f"{row['samples']}."
                )

            recomputed = calculated_gflops(
                shape,
                row["total_ms"],
            )

            relative_error = (
                abs(
                    row["reported_gflops"]
                    - recomputed
                )
                / max(
                    abs(recomputed),
                    1e-12,
                )
            )

            if relative_error > 1e-5:
                errors.append(
                    f"{filename}: GFLOPs "
                    f"mismatch for {shape}. "
                    f"reported="
                    f"{row['reported_gflops']:.6f}, "
                    f"recomputed="
                    f"{recomputed:.6f}."
                )

            verify_error = (
                row[
                    "verify_max_abs_error"
                ]
            )

            if (
                not math.isfinite(
                    verify_error
                )
                or verify_error < 0.0
            ):
                errors.append(
                    f"{filename}: invalid "
                    f"verification error "
                    f"for {shape}."
                )

            elif verify_error > 0.05:
                warnings.append(
                    f"{filename}: relatively "
                    f"large verification error "
                    f"for {shape}: "
                    f"{verify_error:.6f}."
                )

        missing_shapes = (
            set(SHAPES)
            - seen_shapes
        )

        if missing_shapes:
            errors.append(
                f"{filename}: missing shapes: "
                + ", ".join(
                    sorted(missing_shapes)
                )
            )

    expected_rows = (
        EXPECTED_RUNS
        * len(SHAPES)
    )

    if len(rows) != expected_rows:
        errors.append(
            f"Expected {expected_rows} GPU "
            f"RESULT rows, found {len(rows)}."
        )

    by_shape = defaultdict(list)

    for row in rows:
        if row["shape"] in SHAPES:
            by_shape[
                row["shape"]
            ].append(row)

    for shape in SHAPES:
        group = by_shape.get(
            shape,
            [],
        )

        if len(group) != EXPECTED_RUNS:
            errors.append(
                f"{shape}: expected "
                f"{EXPECTED_RUNS} runs, "
                f"found {len(group)}."
            )
            continue

        checksums = [
            row["checksum"]
            for row in group
        ]

        checksum_delta = (
            max(checksums)
            - min(checksums)
        )

        if checksum_delta > 1e-6:
            errors.append(
                f"{shape}: checksum changed "
                f"across GPU runs "
                f"(delta="
                f"{checksum_delta:.9f})."
            )

    return errors, warnings


def calculate_gpu_statistics(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[
            row["shape"]
        ].append(row)

    output = []

    for shape, group in groups.items():
        totals = [
            row["total_ms"]
            for row in group
        ]

        p90_values = [
            row["p90_ms"]
            for row in group
        ]

        mean_ms = statistics.mean(
            totals
        )

        median_ms = statistics.median(
            totals
        )

        std_ms = (
            statistics.stdev(totals)
            if len(totals) >= 2
            else 0.0
        )

        cv = (
            std_ms / mean_ms * 100.0
            if mean_ms != 0.0
            else 0.0
        )

        output.append(
            {
                "backend": "cuBLAS",
                "dtype": "fp16",
                "compute": "fp32",
                "category": (
                    SHAPES[
                        shape
                    ]["category"]
                ),
                "shape": shape,
                "runs": len(group),
                "total_mean_ms": mean_ms,
                "total_median_ms": (
                    median_ms
                ),
                "total_std_ms": std_ms,
                "total_min_ms": min(totals),
                "total_max_ms": max(totals),
                "cv_percent": cv,
                "p90_run_median_ms": (
                    statistics.median(
                        p90_values
                    )
                ),
                "gflops_median": (
                    calculated_gflops(
                        shape,
                        median_ms,
                    )
                ),
                "verify_max_abs_error": max(
                    row[
                        "verify_max_abs_error"
                    ]
                    for row in group
                ),
                "checksum_min": min(
                    row["checksum"]
                    for row in group
                ),
                "checksum_max": max(
                    row["checksum"]
                    for row in group
                ),
            }
        )

    output.sort(
        key=lambda row:
            SHAPES[
                row["shape"]
            ]["order"]
    )

    return output


def load_cpu_statistics():
    if not CPU_STATS_FILE.exists():
        raise FileNotFoundError(
            f"Missing CPU statistics: "
            f"{CPU_STATS_FILE}"
        )

    rows = []

    with CPU_STATS_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for raw in reader:
            backend = raw.get(
                "backend",
                "",
            )

            if backend not in BACKENDS:
                continue

            autotuned_text = (
                raw.get(
                    "autotuned",
                    "",
                )
                .strip()
                .lower()
            )

            row = {
                "backend": backend,
                "category": raw[
                    "category"
                ],
                "shape": raw["shape"],
                "threads": int(
                    raw["threads"]
                ),
                "runs": int(
                    raw["runs"]
                ),
                "total_median_ms": float(
                    raw[
                        "total_median_ms"
                    ]
                ),
                "gflops_median": float(
                    raw[
                        "gflops_median"
                    ]
                ),
                "cv_percent": float(
                    raw["cv_percent"]
                ),
                "autotuned": (
                    autotuned_text
                    in (
                        "true",
                        "1",
                        "yes",
                    )
                ),
                "active_threads_mode": (
                    raw.get(
                        "active_threads_mode",
                        "",
                    )
                ),
                "schedule": raw.get(
                    "schedule",
                    "",
                ),
            }

            rows.append(row)

    return rows


def validate_cpu(rows):
    errors = []
    warnings = []

    expected_rows = (
        len(BACKENDS)
        * len(SHAPES)
        * len(THREADS)
    )

    if len(rows) != expected_rows:
        errors.append(
            f"Expected {expected_rows} "
            f"CPU T-MAC configurations, "
            f"found {len(rows)}."
        )

    index = defaultdict(list)

    for row in rows:
        if row["shape"] not in SHAPES:
            errors.append(
                f"Unexpected CPU shape "
                f"{row['shape']}."
            )
            continue

        if row["threads"] not in THREADS:
            errors.append(
                f"Unexpected CPU thread "
                f"count {row['threads']}."
            )
            continue

        if row["runs"] != EXPECTED_RUNS:
            errors.append(
                f"{row['backend']} "
                f"{row['shape']} "
                f"{row['threads']}T: "
                f"expected 5 runs, found "
                f"{row['runs']}."
            )

        if not row["autotuned"]:
            errors.append(
                f"{row['backend']} "
                f"{row['shape']} "
                f"{row['threads']}T "
                f"is not autotuned."
            )

        index[
            (
                row["backend"],
                row["shape"],
                row["threads"],
            )
        ].append(row)

    for backend in BACKENDS:
        for shape in SHAPES:
            for threads in THREADS:
                count = len(
                    index.get(
                        (
                            backend,
                            shape,
                            threads,
                        ),
                        [],
                    )
                )

                if count != 1:
                    errors.append(
                        f"{backend} {shape} "
                        f"{threads}T appears "
                        f"{count} times."
                    )

    return errors, warnings


def build_all_thread_comparison(
    cpu_rows,
    gpu_stats,
):
    gpu_index = {
        row["shape"]: row
        for row in gpu_stats
    }

    output = []

    for cpu in cpu_rows:
        gpu = gpu_index[
            cpu["shape"]
        ]

        speedup = (
            cpu["total_median_ms"]
            / gpu["total_median_ms"]
        )

        output.append(
            {
                "backend": (
                    cpu["backend"]
                ),
                "category": (
                    cpu["category"]
                ),
                "shape": cpu["shape"],
                "cpu_threads": (
                    cpu["threads"]
                ),
                "cpu_total_ms": (
                    cpu[
                        "total_median_ms"
                    ]
                ),
                "gpu_total_ms": (
                    gpu[
                        "total_median_ms"
                    ]
                ),
                "gpu_speedup_x": speedup,
                "cpu_gflops": (
                    cpu[
                        "gflops_median"
                    ]
                ),
                "gpu_gflops": (
                    gpu[
                        "gflops_median"
                    ]
                ),
                "cpu_cv_percent": (
                    cpu["cv_percent"]
                ),
                "gpu_cv_percent": (
                    gpu["cv_percent"]
                ),
                "cpu_active_threads_mode": (
                    cpu[
                        "active_threads_mode"
                    ]
                ),
                "cpu_schedule": (
                    cpu["schedule"]
                ),
            }
        )

    output.sort(
        key=lambda row: (
            BACKENDS.index(
                row["backend"]
            ),
            SHAPES[
                row["shape"]
            ]["order"],
            THREADS.index(
                row["cpu_threads"]
            ),
        )
    )

    return output


def build_primary_comparison(
    cpu_rows,
    gpu_stats,
):
    gpu_index = {
        row["shape"]: row
        for row in gpu_stats
    }

    grouped = defaultdict(list)

    for row in cpu_rows:
        grouped[
            (
                row["backend"],
                row["shape"],
            )
        ].append(row)

    output = []

    for backend in BACKENDS:
        for shape in SHAPES:
            group = grouped[
                (
                    backend,
                    shape,
                )
            ]

            primary_candidates = [
                row
                for row in group
                if row["threads"]
                in PRIMARY_THREADS
            ]

            best_primary = min(
                primary_candidates,
                key=lambda row:
                    row[
                        "total_median_ms"
                    ],
            )

            best_all = min(
                group,
                key=lambda row:
                    row[
                        "total_median_ms"
                    ],
            )

            gpu = gpu_index[shape]

            gpu_ms = (
                gpu[
                    "total_median_ms"
                ]
            )

            output.append(
                {
                    "backend": backend,
                    "category": (
                        SHAPES[
                            shape
                        ]["category"]
                    ),
                    "shape": shape,
                    "best_primary_thread": (
                        best_primary[
                            "threads"
                        ]
                    ),
                    "best_primary_cpu_ms": (
                        best_primary[
                            "total_median_ms"
                        ]
                    ),
                    "gpu_ms": gpu_ms,
                    "gpu_speedup_vs_primary_x": (
                        best_primary[
                            "total_median_ms"
                        ]
                        / gpu_ms
                    ),
                    "best_all_thread": (
                        best_all[
                            "threads"
                        ]
                    ),
                    "best_all_cpu_ms": (
                        best_all[
                            "total_median_ms"
                        ]
                    ),
                    "gpu_speedup_vs_best_all_x": (
                        best_all[
                            "total_median_ms"
                        ]
                        / gpu_ms
                    ),
                    "gpu_gflops": (
                        gpu[
                            "gflops_median"
                        ]
                    ),
                }
            )

    return output


def write_csv(
    path,
    rows,
    fields,
):
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            output = {}

            for field in fields:
                value = row.get(
                    field,
                    "",
                )

                if isinstance(
                    value,
                    float,
                ):
                    output[field] = (
                        f"{value:.6f}"
                    )
                else:
                    output[field] = (
                        value
                    )

            writer.writerow(output)


def write_validation(
    path,
    gpu_errors,
    gpu_warnings,
    cpu_errors,
    cpu_warnings,
    gpu_rows,
    cpu_rows,
):
    errors = (
        gpu_errors
        + cpu_errors
    )

    warnings = (
        gpu_warnings
        + cpu_warnings
    )

    lines = [
        "cuBLAS vs T-MAC Validation",
        "=" * 40,
        "",
        "STATUS: "
        + (
            "PASS"
            if not errors
            else "FAIL"
        ),
        "",
        f"GPU logs expected: "
        f"{EXPECTED_RUNS}",
        f"GPU RESULT rows parsed: "
        f"{len(gpu_rows)}",
        f"GPU RESULT rows expected: "
        f"{EXPECTED_RUNS * len(SHAPES)}",
        f"CPU T-MAC configurations: "
        f"{len(cpu_rows)}",
        f"CPU configurations expected: "
        f"{len(BACKENDS) * len(SHAPES) * len(THREADS)}",
        "",
    ]

    if errors:
        lines.append("ERRORS")
        lines.append("-" * 40)

        for error in errors:
            lines.append(
                f"- {error}"
            )

        lines.append("")

    if warnings:
        lines.append("WARNINGS")
        lines.append("-" * 40)

        for warning in warnings:
            lines.append(
                f"- {warning}"
            )

        lines.append("")

    if not errors and not warnings:
        lines.append(
            "No validation errors "
            "or warnings."
        )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_markdown(
    path,
    gpu_stats,
    primary,
):
    lines = []

    lines.append(
        "# cuBLAS GPU vs T-MAC CPU"
    )
    lines.append("")

    lines.append(
        "## Environment and Method"
    )
    lines.append("")

    lines.append(
        "- CPU: `AMD Ryzen 5 8500G`"
    )
    lines.append(
        "- GPU: `NVIDIA GeForce RTX 4070 SUPER`"
    )
    lines.append(
        "- Shape notation: `M×N×K`"
    )
    lines.append(
        "- T-MAC: `W2A16 / W3A16 / W4A16`, "
        "AVX2/F16C/OpenMP."
    )
    lines.append(
        "- cuBLAS: dense FP16 input/output "
        "with FP32 accumulation."
    )
    lines.append(
        "- Both summaries use the median "
        "of five independent run-level medians."
    )
    lines.append(
        "- Primary CPU comparison selects "
        "the fastest T-MAC result among "
        "`1T / 2T / 4T / 8T`."
    )
    lines.append(
        "- 16T is retained separately but "
        "excluded from the primary CPU selection "
        "because the Ryzen exposes 12 logical CPUs."
    )
    lines.append(
        "- GPU speedup = T-MAC CPU latency "
        "/ cuBLAS GPU latency."
    )
    lines.append("")

    lines.append(
        "> This is a cross-hardware performance "
        "reference, not a same-hardware or "
        "equal-precision kernel comparison. "
        "T-MAC total_ms includes preprocessing "
        "+ kernel, while cuBLAS timing measures "
        "the warmed-up GPU GEMM with H2D/D2H "
        "excluded."
    )
    lines.append("")

    lines.append(
        "## cuBLAS Five-Run Statistics"
    )
    lines.append("")

    lines.append(
        "| Shape | Median ms | GFLOPs | "
        "CV% | P90 median ms | "
        "Max verification error |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    for row in gpu_stats:
        lines.append(
            f"| {row['shape']} "
            f"| {row['total_median_ms']:.6f} "
            f"| {row['gflops_median']:.3f} "
            f"| {row['cv_percent']:.3f} "
            f"| {row['p90_run_median_ms']:.6f} "
            f"| {row['verify_max_abs_error']:.6f} |"
        )

    lines.append("")

    lines.append(
        "## GPU vs Best Primary T-MAC"
    )
    lines.append("")

    lines.append(
        "| Backend | Shape | Best CPU T | "
        "T-MAC ms | cuBLAS ms | "
        "GPU speedup |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|"
    )

    for row in primary:
        lines.append(
            f"| {row['backend']} "
            f"| {row['shape']} "
            f"| {row['best_primary_thread']} "
            f"| {row['best_primary_cpu_ms']:.6f} "
            f"| {row['gpu_ms']:.6f} "
            f"| {row['gpu_speedup_vs_primary_x']:.2f}× |"
        )

    lines.append("")

    lines.append(
        "## Backend Summary"
    )
    lines.append("")

    lines.append(
        "| Backend | Geomean GPU speedup "
        "vs best 1/2/4/8T CPU | "
        "Minimum | Maximum |"
    )
    lines.append(
        "|---|---:|---:|---:|"
    )

    for backend in BACKENDS:
        rows = [
            row
            for row in primary
            if row["backend"] == backend
        ]

        values = [
            row[
                "gpu_speedup_vs_primary_x"
            ]
            for row in rows
        ]

        lines.append(
            f"| {backend} "
            f"| {geometric_mean(values):.2f}× "
            f"| {min(values):.2f}× "
            f"| {max(values):.2f}× |"
        )

    lines.append("")

    lines.append(
        "## Interpretation"
    )
    lines.append("")

    lines.append(
        "- cuBLAS represents an optimized "
        "NVIDIA GPU dense-GEMM baseline; "
        "it is not GPU T-MAC."
    )
    lines.append(
        "- T-MAC uses low-bit weights and "
        "FP16 activations, while cuBLAS uses "
        "dense FP16 operands with FP32 accumulation."
    )
    lines.append(
        "- Therefore the speedup values answer "
        "`how much faster this GPU dense-GEMM "
        "reference is than this CPU T-MAC "
        "implementation`, not which algorithm "
        "is intrinsically better under identical "
        "hardware and precision."
    )
    lines.append(
        "- The detailed all-thread comparison "
        "is available in "
        "`cublas_tmac_all_threads.csv`."
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        gpu_files,
        gpu_rows,
        gpu_parse_errors,
    ) = load_gpu_logs()

    (
        gpu_errors,
        gpu_warnings,
    ) = validate_gpu(
        gpu_files,
        gpu_rows,
        gpu_parse_errors,
    )

    try:
        cpu_rows = (
            load_cpu_statistics()
        )

        (
            cpu_errors,
            cpu_warnings,
        ) = validate_cpu(
            cpu_rows
        )

    except Exception as error:
        cpu_rows = []
        cpu_errors = [
            str(error)
        ]
        cpu_warnings = []

    validation_path = (
        OUTPUT_DIR
        / "validation_report.txt"
    )

    write_validation(
        validation_path,
        gpu_errors,
        gpu_warnings,
        cpu_errors,
        cpu_warnings,
        gpu_rows,
        cpu_rows,
    )

    errors = (
        gpu_errors
        + cpu_errors
    )

    if errors:
        print(
            "Validation FAILED"
        )
        print(
            f"See: {validation_path}"
        )
        return 1

    gpu_stats = (
        calculate_gpu_statistics(
            gpu_rows
        )
    )

    all_threads = (
        build_all_thread_comparison(
            cpu_rows,
            gpu_stats,
        )
    )

    primary = (
        build_primary_comparison(
            cpu_rows,
            gpu_stats,
        )
    )

    gpu_stats_path = (
        OUTPUT_DIR
        / "cublas_statistics.csv"
    )

    all_threads_path = (
        OUTPUT_DIR
        / "cublas_tmac_all_threads.csv"
    )

    primary_path = (
        OUTPUT_DIR
        / "cublas_tmac_primary_comparison.csv"
    )

    markdown_path = (
        OUTPUT_DIR
        / "cublas_tmac_comparison.md"
    )

    write_csv(
        gpu_stats_path,
        gpu_stats,
        [
            "backend",
            "dtype",
            "compute",
            "category",
            "shape",
            "runs",
            "total_mean_ms",
            "total_median_ms",
            "total_std_ms",
            "total_min_ms",
            "total_max_ms",
            "cv_percent",
            "p90_run_median_ms",
            "gflops_median",
            "verify_max_abs_error",
            "checksum_min",
            "checksum_max",
        ],
    )

    write_csv(
        all_threads_path,
        all_threads,
        [
            "backend",
            "category",
            "shape",
            "cpu_threads",
            "cpu_total_ms",
            "gpu_total_ms",
            "gpu_speedup_x",
            "cpu_gflops",
            "gpu_gflops",
            "cpu_cv_percent",
            "gpu_cv_percent",
            "cpu_active_threads_mode",
            "cpu_schedule",
        ],
    )

    write_csv(
        primary_path,
        primary,
        [
            "backend",
            "category",
            "shape",
            "best_primary_thread",
            "best_primary_cpu_ms",
            "gpu_ms",
            "gpu_speedup_vs_primary_x",
            "best_all_thread",
            "best_all_cpu_ms",
            "gpu_speedup_vs_best_all_x",
            "gpu_gflops",
        ],
    )

    write_markdown(
        markdown_path,
        gpu_stats,
        primary,
    )

    print(
        "Validation PASS"
    )
    print(
        f"GPU raw rows: "
        f"{len(gpu_rows)}"
    )
    print(
        f"GPU configurations: "
        f"{len(gpu_stats)}"
    )
    print(
        f"CPU T-MAC configurations: "
        f"{len(cpu_rows)}"
    )
    print(
        f"All-thread comparisons: "
        f"{len(all_threads)}"
    )
    print(
        f"Primary comparisons: "
        f"{len(primary)}"
    )
    print()
    print("Generated:")
    print(
        f"  {validation_path}"
    )
    print(
        f"  {gpu_stats_path}"
    )
    print(
        f"  {all_threads_path}"
    )
    print(
        f"  {primary_path}"
    )
    print(
        f"  {markdown_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())