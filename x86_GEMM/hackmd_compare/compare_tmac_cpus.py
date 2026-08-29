from pathlib import Path
import csv
import math
import re
import statistics
from collections import Counter, defaultdict


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

ORIGINAL_SUMMARY = RESULT_DIR / "summary_statistics.csv"
LAB_DIR = RESULT_DIR / "lab_ryzen8500g"
OUTPUT_DIR = RESULT_DIR / "cpu_tmac_compare"

ORIGINAL_CPU = "Intel Core Ultra 7 258V"
LAB_CPU = "AMD Ryzen 5 8500G"

ORIGINAL_HARDWARE_THREADS = 8
LAB_LOGICAL_CPUS = 12

EXPECTED_RUNS = 5
EXPECTED_ROWS_PER_LOG = 25

THREADS = [1, 2, 4, 8, 16]
PRIMARY_THREADS = [1, 2, 4, 8]

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

BACKENDS = [
    "T-MAC W2A16",
    "T-MAC W3A16",
    "T-MAC W4A16",
]

BACKEND_TO_BIT = {
    "T-MAC W2A16": "W2",
    "T-MAC W3A16": "W3",
    "T-MAC W4A16": "W4",
}

BIT_TO_BACKEND = {
    "W2": "T-MAC W2A16",
    "W3": "T-MAC W3A16",
    "W4": "T-MAC W4A16",
}

EXPECTED_FILES = []

for bit in (2, 3, 4):
    for run in range(1, EXPECTED_RUNS + 1):
        EXPECTED_FILES.append(
            f"tmac_w{bit}_run{run}.log"
        )


def parse_value(line, key):
    match = re.search(
        rf'{re.escape(key)}=("[^"]*"|\S+)',
        line,
    )

    if not match:
        return None

    value = match.group(1)

    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]

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


def geometric_mean(values):
    values = [
        value
        for value in values
        if value > 0.0
    ]

    if not values:
        return 0.0

    return math.exp(
        statistics.mean(
            math.log(value)
            for value in values
        )
    )


def parse_lab_result(line, filename):
    if not line.startswith("RESULT "):
        return None

    category = parse_value(line, "category")
    bit = parse_value(line, "bit")
    shape = parse_value(line, "shape")
    threads = parse_value(line, "threads")
    active_threads = parse_value(line, "active_threads")
    total_ms = parse_value(line, "total_ms")
    gflops = parse_value(line, "gflops")
    checksum = parse_value(line, "checksum")
    autotuned = parse_value(line, "autotuned")
    bm = parse_value(line, "bm")
    bn = parse_value(line, "bn")
    kfactor = parse_value(line, "kfactor")

    required = {
        "category": category,
        "bit": bit,
        "shape": shape,
        "threads": threads,
        "total_ms": total_ms,
        "checksum": checksum,
        "autotuned": autotuned,
        "bm": bm,
        "bn": bn,
        "kfactor": kfactor,
    }

    missing = [
        key
        for key, value in required.items()
        if value is None
    ]

    if missing:
        raise ValueError(
            "missing fields: "
            + ", ".join(missing)
        )

    if bit not in BIT_TO_BACKEND:
        raise ValueError(
            f"unsupported bit value: {bit}"
        )

    result = {
        "file": filename,
        "backend": BIT_TO_BACKEND[bit],
        "dtype": f"{bit}A16",
        "bit": bit,
        "category": category,
        "shape": shape,
        "threads": int(threads),
        "active_threads": (
            int(active_threads)
            if active_threads is not None
            else None
        ),
        "total_ms": float(total_ms),
        "reported_gflops": (
            float(gflops)
            if gflops is not None
            else None
        ),
        "checksum": float(checksum),
        "autotuned": int(autotuned),
        "bm": int(bm),
        "bn": int(bn),
        "kfactor": int(kfactor),
    }

    return result


def load_lab_logs():
    rows = []
    parse_errors = []
    found_files = []

    for filename in EXPECTED_FILES:
        path = LAB_DIR / filename

        if not path.exists():
            continue

        found_files.append(path)

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:
            for line_number, raw_line in enumerate(
                file,
                start=1,
            ):
                line = raw_line.strip()

                if not line.startswith("RESULT "):
                    continue

                try:
                    result = parse_lab_result(
                        line,
                        filename,
                    )

                    if result is not None:
                        rows.append(result)

                except Exception as error:
                    parse_errors.append(
                        f"{filename}:{line_number}: "
                        f"{error}"
                    )

    return found_files, rows, parse_errors


def expected_pairs():
    return {
        (shape, threads)
        for shape in SHAPES
        for threads in THREADS
    }


def validate_lab(
    found_files,
    rows,
    parse_errors,
):
    errors = []
    warnings = []

    found_names = {
        path.name
        for path in found_files
    }

    expected_names = set(EXPECTED_FILES)

    missing_files = sorted(
        expected_names - found_names
    )

    if missing_files:
        errors.append(
            "Missing lab log files: "
            + ", ".join(missing_files)
        )

    extra_files = sorted(
        path.name
        for path in LAB_DIR.glob(
            "tmac_w*_run*.log"
        )
        if path.name not in expected_names
    )

    if extra_files:
        warnings.append(
            "Extra lab run-like files ignored: "
            + ", ".join(extra_files)
        )

    if parse_errors:
        errors.extend(
            "Parse error: " + error
            for error in parse_errors
        )

    rows_by_file = defaultdict(list)

    for row in rows:
        rows_by_file[row["file"]].append(row)

    expected_shape_thread_pairs = expected_pairs()

    for filename in EXPECTED_FILES:
        if filename not in found_names:
            continue

        file_rows = rows_by_file[filename]

        if len(file_rows) != EXPECTED_ROWS_PER_LOG:
            errors.append(
                f"{filename}: expected "
                f"{EXPECTED_ROWS_PER_LOG} RESULT rows, "
                f"found {len(file_rows)}."
            )

        expected_bit = None

        if filename.startswith("tmac_w2_"):
            expected_bit = "W2"
        elif filename.startswith("tmac_w3_"):
            expected_bit = "W3"
        elif filename.startswith("tmac_w4_"):
            expected_bit = "W4"

        actual_pairs = []

        for row in file_rows:
            if row["bit"] != expected_bit:
                errors.append(
                    f"{filename}: expected bit "
                    f"{expected_bit}, found "
                    f"{row['bit']}."
                )

            shape = row["shape"]
            threads = row["threads"]

            if shape not in SHAPES:
                errors.append(
                    f"{filename}: unexpected "
                    f"shape {shape}."
                )
                continue

            expected_category = (
                SHAPES[shape]["category"]
            )

            if row["category"] != expected_category:
                errors.append(
                    f"{filename}: category mismatch "
                    f"for {shape}. Expected "
                    f"'{expected_category}', found "
                    f"'{row['category']}'."
                )

            if threads not in THREADS:
                errors.append(
                    f"{filename}: unexpected thread "
                    f"count {threads}."
                )
                continue

            actual_pairs.append(
                (shape, threads)
            )

            if row["autotuned"] != 1:
                errors.append(
                    f"{filename}: {shape} "
                    f"{threads}T is not "
                    f"autotuned=1."
                )

            if row["reported_gflops"] is not None:
                expected_gflops = (
                    calculated_gflops(
                        shape,
                        row["total_ms"],
                    )
                )

                relative_error = (
                    abs(
                        row["reported_gflops"]
                        - expected_gflops
                    )
                    / max(
                        abs(expected_gflops),
                        1e-12,
                    )
                )

                if relative_error > 1e-5:
                    errors.append(
                        f"{filename}: GFLOPs mismatch "
                        f"for {shape} {threads}T. "
                        f"reported="
                        f"{row['reported_gflops']:.6f}, "
                        f"recomputed="
                        f"{expected_gflops:.6f}."
                    )

        counter = Counter(actual_pairs)

        duplicates = [
            pair
            for pair, count in counter.items()
            if count > 1
        ]

        missing_pairs = (
            expected_shape_thread_pairs
            - set(actual_pairs)
        )

        if duplicates:
            errors.append(
                f"{filename}: duplicate "
                f"shape/thread rows: "
                + ", ".join(
                    f"{shape}/{threads}T"
                    for shape, threads
                    in duplicates
                )
            )

        if missing_pairs:
            ordered_missing = sorted(
                missing_pairs,
                key=lambda pair: (
                    SHAPES[pair[0]]["order"],
                    THREADS.index(pair[1]),
                ),
            )

            errors.append(
                f"{filename}: missing "
                f"shape/thread rows: "
                + ", ".join(
                    f"{shape}/{threads}T"
                    for shape, threads
                    in ordered_missing
                )
            )

    expected_total_rows = (
        3
        * EXPECTED_RUNS
        * EXPECTED_ROWS_PER_LOG
    )

    if len(rows) != expected_total_rows:
        errors.append(
            f"Expected {expected_total_rows} "
            f"lab RESULT rows, found "
            f"{len(rows)}."
        )

    grouped = defaultdict(list)

    for row in rows:
        if (
            row["backend"] in BACKENDS
            and row["shape"] in SHAPES
            and row["threads"] in THREADS
        ):
            key = (
                row["backend"],
                row["shape"],
                row["threads"],
            )

            grouped[key].append(row)

    expected_groups = (
        len(BACKENDS)
        * len(SHAPES)
        * len(THREADS)
    )

    if len(grouped) != expected_groups:
        errors.append(
            f"Expected {expected_groups} "
            f"lab configurations, found "
            f"{len(grouped)}."
        )

    for key, group in grouped.items():
        backend, shape, threads = key

        if len(group) != EXPECTED_RUNS:
            errors.append(
                f"{backend} {shape} {threads}T: "
                f"expected {EXPECTED_RUNS} runs, "
                f"found {len(group)}."
            )

        checksums = [
            row["checksum"]
            for row in group
        ]

        if checksums:
            checksum_delta = (
                max(checksums)
                - min(checksums)
            )

            if checksum_delta > 1e-6:
                errors.append(
                    f"{backend} {shape} {threads}T: "
                    f"checksum changed across runs "
                    f"(delta={checksum_delta:.9f})."
                )

    return errors, warnings


def schedule_description(schedules):
    if not schedules:
        return "", 0, 0

    counter = Counter(schedules)

    schedule, count = counter.most_common(1)[0]
    unique_count = len(counter)
    total = len(schedules)

    bm, bn, kfactor = schedule

    if count == total:
        label = "stable"
    elif count >= 3:
        label = "mostly stable"
    else:
        label = "unstable"

    text = (
        f"{label}: "
        f"bm={bm},bn={bn},"
        f"kfactor={kfactor} "
        f"({count}/{total}), "
        f"{unique_count} schedule(s)"
    )

    return text, unique_count, count


def calculate_lab_statistics(rows):
    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["backend"],
            row["dtype"],
            row["category"],
            row["shape"],
            row["threads"],
        )

        grouped[key].append(row)

    summary = []

    for key, group in grouped.items():
        (
            backend,
            dtype,
            category,
            shape,
            threads,
        ) = key

        total_values = [
            row["total_ms"]
            for row in group
        ]

        mean_ms = statistics.mean(
            total_values
        )

        median_ms = statistics.median(
            total_values
        )

        std_ms = (
            statistics.stdev(total_values)
            if len(total_values) >= 2
            else 0.0
        )

        cv = (
            std_ms / mean_ms * 100.0
            if mean_ms != 0.0
            else 0.0
        )

        schedules = [
            (
                row["bm"],
                row["bn"],
                row["kfactor"],
            )
            for row in group
        ]

        (
            schedule,
            schedule_unique_count,
            schedule_mode_count,
        ) = schedule_description(
            schedules
        )

        active_threads = [
            row["active_threads"]
            for row in group
            if row["active_threads"] is not None
        ]

        active_threads_mode = None

        if active_threads:
            active_threads_mode = (
                Counter(active_threads)
                .most_common(1)[0][0]
            )

        entry = {
            "backend": backend,
            "dtype": dtype,
            "category": category,
            "shape": shape,
            "threads": threads,
            "runs": len(group),
            "total_mean_ms": mean_ms,
            "total_median_ms": median_ms,
            "total_std_ms": std_ms,
            "total_min_ms": min(
                total_values
            ),
            "total_max_ms": max(
                total_values
            ),
            "cv_percent": cv,
            "gflops_mean": statistics.mean(
                calculated_gflops(
                    shape,
                    value,
                )
                for value in total_values
            ),
            "gflops_median": (
                calculated_gflops(
                    shape,
                    median_ms,
                )
            ),
            "checksum_min": min(
                row["checksum"]
                for row in group
            ),
            "checksum_max": max(
                row["checksum"]
                for row in group
            ),
            "autotuned": all(
                row["autotuned"] == 1
                for row in group
            ),
            "active_threads_mode": (
                active_threads_mode
            ),
            "schedule": schedule,
            "schedule_unique_count": (
                schedule_unique_count
            ),
            "schedule_mode_count": (
                schedule_mode_count
            ),
        }

        summary.append(entry)

    summary.sort(
        key=lambda row: (
            BACKENDS.index(
                row["backend"]
            ),
            SHAPES[
                row["shape"]
            ]["order"],
            THREADS.index(
                row["threads"]
            ),
        )
    )

    return summary


def load_original_summary():
    if not ORIGINAL_SUMMARY.exists():
        raise FileNotFoundError(
            f"Missing original summary: "
            f"{ORIGINAL_SUMMARY}"
        )

    rows = []

    with ORIGINAL_SUMMARY.open(
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

            row = {
                "backend": backend,
                "dtype": raw.get(
                    "dtype",
                    "",
                ),
                "category": raw.get(
                    "category",
                    "",
                ),
                "shape": raw.get(
                    "shape",
                    "",
                ),
                "threads": int(
                    raw["threads"]
                ),
                "runs": int(
                    raw["runs"]
                ),
                "total_mean_ms": float(
                    raw["total_mean_ms"]
                ),
                "total_median_ms": float(
                    raw["total_median_ms"]
                ),
                "total_std_ms": float(
                    raw["total_std_ms"]
                ),
                "total_min_ms": float(
                    raw["total_min_ms"]
                ),
                "total_max_ms": float(
                    raw["total_max_ms"]
                ),
                "cv_percent": float(
                    raw["cv_percent"]
                ),
                "gflops_mean": float(
                    raw["gflops_mean"]
                ),
                "gflops_median": float(
                    raw["gflops_median"]
                ),
                "checksum_min": float(
                    raw["checksum_min"]
                ),
                "checksum_max": float(
                    raw["checksum_max"]
                ),
                "autotuned": raw.get(
                    "autotuned",
                    "",
                ),
                "schedule": raw.get(
                    "schedule",
                    "",
                ),
            }

            rows.append(row)

    return rows


def validate_original(rows):
    errors = []
    warnings = []

    expected_count = (
        len(BACKENDS)
        * len(SHAPES)
        * len(THREADS)
    )

    if len(rows) != expected_count:
        errors.append(
            f"Original summary: expected "
            f"{expected_count} T-MAC rows, "
            f"found {len(rows)}."
        )

    grouped = defaultdict(list)

    for row in rows:
        backend = row["backend"]
        shape = row["shape"]
        threads = row["threads"]

        if shape not in SHAPES:
            errors.append(
                f"Original summary: unexpected "
                f"shape {shape}."
            )
            continue

        expected_category = (
            SHAPES[shape]["category"]
        )

        if row["category"] != expected_category:
            errors.append(
                f"Original summary: category "
                f"mismatch for {backend} "
                f"{shape} {threads}T."
            )

        if threads not in THREADS:
            errors.append(
                f"Original summary: unexpected "
                f"thread count {threads}."
            )
            continue

        if row["runs"] != EXPECTED_RUNS:
            errors.append(
                f"Original summary: {backend} "
                f"{shape} {threads}T has "
                f"{row['runs']} runs instead "
                f"of {EXPECTED_RUNS}."
            )

        expected_gflops = calculated_gflops(
            shape,
            row["total_median_ms"],
        )

        relative_error = (
            abs(
                row["gflops_median"]
                - expected_gflops
            )
            / max(
                abs(expected_gflops),
                1e-12,
            )
        )

        if relative_error > 1e-5:
            errors.append(
                f"Original summary: GFLOPs "
                f"mismatch for {backend} "
                f"{shape} {threads}T."
            )

        grouped[
            (
                backend,
                shape,
                threads,
            )
        ].append(row)

    for backend in BACKENDS:
        for shape in SHAPES:
            for threads in THREADS:
                key = (
                    backend,
                    shape,
                    threads,
                )

                count = len(
                    grouped.get(
                        key,
                        [],
                    )
                )

                if count != 1:
                    errors.append(
                        f"Original summary: "
                        f"{backend} {shape} "
                        f"{threads}T appears "
                        f"{count} times."
                    )

    return errors, warnings


def build_comparison(
    original_rows,
    lab_rows,
):
    original_index = {
        (
            row["backend"],
            row["shape"],
            row["threads"],
        ): row
        for row in original_rows
    }

    lab_index = {
        (
            row["backend"],
            row["shape"],
            row["threads"],
        ): row
        for row in lab_rows
    }

    comparison = []

    for backend in BACKENDS:
        for shape in SHAPES:
            for threads in THREADS:
                key = (
                    backend,
                    shape,
                    threads,
                )

                original = original_index[key]
                lab = lab_index[key]

                original_ms = (
                    original[
                        "total_median_ms"
                    ]
                )

                lab_ms = (
                    lab[
                        "total_median_ms"
                    ]
                )

                speedup = (
                    original_ms / lab_ms
                )

                lab_difference_percent = (
                    (
                        lab_ms
                        / original_ms
                    )
                    - 1.0
                ) * 100.0

                if speedup > 1.0:
                    winner = LAB_CPU
                elif speedup < 1.0:
                    winner = ORIGINAL_CPU
                else:
                    winner = "Tie"

                row = {
                    "backend": backend,
                    "category": (
                        SHAPES[
                            shape
                        ]["category"]
                    ),
                    "shape": shape,
                    "threads": threads,
                    "original_cpu": (
                        ORIGINAL_CPU
                    ),
                    "lab_cpu": LAB_CPU,
                    "original_total_ms": (
                        original_ms
                    ),
                    "lab_total_ms": (
                        lab_ms
                    ),
                    "lab_speedup_x": (
                        speedup
                    ),
                    "lab_latency_difference_percent": (
                        lab_difference_percent
                    ),
                    "original_gflops": (
                        original[
                            "gflops_median"
                        ]
                    ),
                    "lab_gflops": (
                        lab[
                            "gflops_median"
                        ]
                    ),
                    "original_cv_percent": (
                        original[
                            "cv_percent"
                        ]
                    ),
                    "lab_cv_percent": (
                        lab[
                            "cv_percent"
                        ]
                    ),
                    "lab_active_threads_mode": (
                        lab[
                            "active_threads_mode"
                        ]
                    ),
                    "original_schedule": (
                        original[
                            "schedule"
                        ]
                    ),
                    "lab_schedule": (
                        lab[
                            "schedule"
                        ]
                    ),
                    "winner": winner,
                }

                comparison.append(row)

    return comparison


def write_csv(
    path,
    rows,
    fieldnames,
):
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            output = {}

            for field in fieldnames:
                value = row.get(field, "")

                if isinstance(value, float):
                    output[field] = (
                        f"{value:.6f}"
                    )
                else:
                    output[field] = value

            writer.writerow(output)


def schedule_class(schedule_text):
    text = (
        schedule_text
        or ""
    ).strip().lower()

    if text.startswith(
        "mostly stable"
    ):
        return "mostly stable"

    if text.startswith("stable"):
        return "stable"

    if text.startswith("unstable"):
        return "unstable"

    return "unknown"


def backend_summary(
    comparison,
    backend,
):
    rows = [
        row
        for row in comparison
        if row["backend"] == backend
    ]

    primary = [
        row
        for row in rows
        if row["threads"]
        in PRIMARY_THREADS
    ]

    speedups = [
        row["lab_speedup_x"]
        for row in primary
    ]

    lab_wins = sum(
        1
        for row in rows
        if row["winner"] == LAB_CPU
    )

    original_wins = sum(
        1
        for row in rows
        if row["winner"] == ORIGINAL_CPU
    )

    ties = (
        len(rows)
        - lab_wins
        - original_wins
    )

    best = max(
        rows,
        key=lambda row:
            row["lab_speedup_x"],
    )

    worst = min(
        rows,
        key=lambda row:
            row["lab_speedup_x"],
    )

    return {
        "backend": backend,
        "configs": len(rows),
        "lab_wins": lab_wins,
        "original_wins": original_wins,
        "ties": ties,
        "primary_geomean_speedup": (
            geometric_mean(speedups)
        ),
        "best": best,
        "worst": worst,
    }


def write_markdown(
    path,
    comparison,
    lab_statistics,
):
    lines = []

    lines.append(
        "# T-MAC CPU Comparison"
    )
    lines.append("")

    lines.append(
        "## Environment"
    )
    lines.append("")

    lines.append(
        f"- Original CPU: `{ORIGINAL_CPU}`"
    )
    lines.append(
        f"- Lab CPU: `{LAB_CPU}`"
    )
    lines.append(
        "- Shape notation: `M×N×K`"
    )
    lines.append(
        "- T-MAC modes: `W2A16`, "
        "`W3A16`, `W4A16`"
    )
    lines.append(
        "- Final latency: median of five "
        "independent run-level medians."
    )
    lines.append(
        "- `Lab Speedup = Original CPU "
        "median latency / Lab CPU "
        "median latency`."
    )
    lines.append(
        "- `Lab Speedup > 1.0×` means "
        "the lab Ryzen CPU is faster."
    )
    lines.append(
        f"- Original machine exposes "
        f"{ORIGINAL_HARDWARE_THREADS} "
        "hardware threads; lab machine "
        f"exposes {LAB_LOGICAL_CPUS} "
        "logical CPUs."
    )
    lines.append(
        "- 16T is retained for HackMD "
        "parity, but is oversubscribed "
        "on both machines."
    )
    lines.append("")

    lines.append(
        "## Overall Summary"
    )
    lines.append("")

    lines.append(
        "| Backend | Lab wins | "
        "Original wins | Ties | "
        "Primary 1/2/4/8T geomean "
        "speedup |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    summaries = []

    for backend in BACKENDS:
        summary = backend_summary(
            comparison,
            backend,
        )

        summaries.append(summary)

        lines.append(
            f"| {backend} "
            f"| {summary['lab_wins']}/"
            f"{summary['configs']} "
            f"| {summary['original_wins']}/"
            f"{summary['configs']} "
            f"| {summary['ties']} "
            f"| {summary['primary_geomean_speedup']:.3f}× |"
        )

    lines.append("")

    for summary in summaries:
        backend = summary["backend"]
        best = summary["best"]
        worst = summary["worst"]

        lines.append(
            f"### {backend} summary"
        )
        lines.append("")

        lines.append(
            f"- Best lab speedup: "
            f"`{best['lab_speedup_x']:.3f}×` "
            f"at `{best['shape']}`, "
            f"`{best['threads']}T`."
        )

        lines.append(
            f"- Lowest lab speedup: "
            f"`{worst['lab_speedup_x']:.3f}×` "
            f"at `{worst['shape']}`, "
            f"`{worst['threads']}T`."
        )

        lines.append("")

    lines.append(
        "## Detailed Results"
    )
    lines.append("")

    for backend in BACKENDS:
        lines.append(
            f"### {backend}"
        )
        lines.append("")

        lines.append(
            "| Shape | T | Original ms | "
            "Lab ms | Lab speedup | "
            "Original GFLOPs | Lab GFLOPs | "
            "Original CV% | Lab CV% | Winner |"
        )

        lines.append(
            "|---|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---|"
        )

        backend_rows = [
            row
            for row in comparison
            if row["backend"] == backend
        ]

        for row in backend_rows:
            winner = (
                "Lab"
                if row["winner"] == LAB_CPU
                else (
                    "Original"
                    if row["winner"]
                    == ORIGINAL_CPU
                    else "Tie"
                )
            )

            lines.append(
                f"| {row['shape']} "
                f"| {row['threads']} "
                f"| {row['original_total_ms']:.6f} "
                f"| {row['lab_total_ms']:.6f} "
                f"| {row['lab_speedup_x']:.3f}× "
                f"| {row['original_gflops']:.3f} "
                f"| {row['lab_gflops']:.3f} "
                f"| {row['original_cv_percent']:.2f} "
                f"| {row['lab_cv_percent']:.2f} "
                f"| {winner} |"
            )

        lines.append("")

    lines.append(
        "## Lab Autotune Schedule Stability"
    )
    lines.append("")

    lines.append(
        "| Backend | Stable | "
        "Mostly stable | Unstable |"
    )
    lines.append(
        "|---|---:|---:|---:|"
    )

    for backend in BACKENDS:
        rows = [
            row
            for row in lab_statistics
            if row["backend"] == backend
        ]

        counter = Counter(
            schedule_class(
                row["schedule"]
            )
            for row in rows
        )

        lines.append(
            f"| {backend} "
            f"| {counter['stable']} "
            f"| {counter['mostly stable']} "
            f"| {counter['unstable']} |"
        )

    lines.append("")

    lines.append(
        "## High-Variation Configurations"
    )
    lines.append("")

    high_variation = [
        row
        for row in comparison
        if (
            row["original_cv_percent"] >= 20.0
            or row["lab_cv_percent"] >= 20.0
        )
    ]

    if not high_variation:
        lines.append(
            "No configuration has CV ≥ 20%."
        )
    else:
        lines.append(
            "| Backend | Shape | T | "
            "Original CV% | Lab CV% |"
        )
        lines.append(
            "|---|---|---:|---:|---:|"
        )

        for row in high_variation:
            lines.append(
                f"| {row['backend']} "
                f"| {row['shape']} "
                f"| {row['threads']} "
                f"| {row['original_cv_percent']:.2f} "
                f"| {row['lab_cv_percent']:.2f} |"
            )

    lines.append("")

    lines.append(
        "## Interpretation Notes"
    )
    lines.append("")

    lines.append(
        "- This comparison evaluates the "
        "same standalone AVX2/F16C/OpenMP "
        "T-MAC implementation on two "
        "different x86 CPUs."
    )
    lines.append(
        "- It is a cross-CPU performance "
        "comparison, not a microarchitecture-"
        "controlled experiment."
    )
    lines.append(
        "- W2/W3/W4 use different low-bit "
        "weight precisions, so comparisons "
        "between bit widths should not be "
        "interpreted as equal-precision "
        "arithmetic."
    )
    lines.append(
        "- The main latency value is "
        "`total_ms`; autotuning time is "
        "not included in the reported "
        "benchmark latency."
    )
    lines.append(
        "- 16T results are kept for "
        "compatibility with the HackMD "
        "thread sweep but should be "
        "interpreted separately because "
        "both machines are oversubscribed "
        "at 16 requested threads."
    )
    lines.append("")

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_validation_report(
    path,
    lab_errors,
    lab_warnings,
    original_errors,
    original_warnings,
    lab_rows,
    original_rows,
):
    errors = (
        lab_errors
        + original_errors
    )

    warnings = (
        lab_warnings
        + original_warnings
    )

    lines = []

    lines.append(
        "T-MAC CPU Comparison Validation"
    )
    lines.append(
        "=" * 40
    )
    lines.append("")

    lines.append(
        "STATUS: "
        + (
            "PASS"
            if not errors
            else "FAIL"
        )
    )
    lines.append("")

    lines.append(
        f"Lab expected logs: "
        f"{len(EXPECTED_FILES)}"
    )
    lines.append(
        f"Lab parsed RESULT rows: "
        f"{len(lab_rows)}"
    )
    lines.append(
        f"Expected lab RESULT rows: "
        f"{3 * EXPECTED_RUNS * EXPECTED_ROWS_PER_LOG}"
    )
    lines.append(
        f"Original T-MAC summary rows: "
        f"{len(original_rows)}"
    )
    lines.append(
        f"Expected original T-MAC "
        f"summary rows: "
        f"{len(BACKENDS) * len(SHAPES) * len(THREADS)}"
    )
    lines.append("")

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
            "No validation errors or warnings."
        )
        lines.append("")

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
        found_files,
        lab_rows,
        parse_errors,
    ) = load_lab_logs()

    (
        lab_errors,
        lab_warnings,
    ) = validate_lab(
        found_files,
        lab_rows,
        parse_errors,
    )

    try:
        original_rows = (
            load_original_summary()
        )

        (
            original_errors,
            original_warnings,
        ) = validate_original(
            original_rows
        )

    except Exception as error:
        original_rows = []
        original_errors = [
            str(error)
        ]
        original_warnings = []

    validation_path = (
        OUTPUT_DIR
        / "validation_report.txt"
    )

    write_validation_report(
        validation_path,
        lab_errors,
        lab_warnings,
        original_errors,
        original_warnings,
        lab_rows,
        original_rows,
    )

    all_errors = (
        lab_errors
        + original_errors
    )

    if all_errors:
        print(
            "Validation FAILED."
        )
        print(
            f"See: {validation_path}"
        )
        return 1

    lab_statistics = (
        calculate_lab_statistics(
            lab_rows
        )
    )

    comparison = build_comparison(
        original_rows,
        lab_statistics,
    )

    lab_statistics_path = (
        OUTPUT_DIR
        / "lab_tmac_statistics.csv"
    )

    comparison_path = (
        OUTPUT_DIR
        / "tmac_cpu_comparison.csv"
    )

    markdown_path = (
        OUTPUT_DIR
        / "tmac_cpu_comparison.md"
    )

    lab_fields = [
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
        "active_threads_mode",
        "schedule",
        "schedule_unique_count",
        "schedule_mode_count",
    ]

    comparison_fields = [
        "backend",
        "category",
        "shape",
        "threads",
        "original_cpu",
        "lab_cpu",
        "original_total_ms",
        "lab_total_ms",
        "lab_speedup_x",
        "lab_latency_difference_percent",
        "original_gflops",
        "lab_gflops",
        "original_cv_percent",
        "lab_cv_percent",
        "lab_active_threads_mode",
        "original_schedule",
        "lab_schedule",
        "winner",
    ]

    write_csv(
        lab_statistics_path,
        lab_statistics,
        lab_fields,
    )

    write_csv(
        comparison_path,
        comparison,
        comparison_fields,
    )

    write_markdown(
        markdown_path,
        comparison,
        lab_statistics,
    )

    print(
        "Validation PASS"
    )
    print(
        f"Lab raw rows: "
        f"{len(lab_rows)}"
    )
    print(
        f"Lab configurations: "
        f"{len(lab_statistics)}"
    )
    print(
        f"Comparison rows: "
        f"{len(comparison)}"
    )
    print()
    print(
        "Generated:"
    )
    print(
        f"  {validation_path}"
    )
    print(
        f"  {lab_statistics_path}"
    )
    print(
        f"  {comparison_path}"
    )
    print(
        f"  {markdown_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())