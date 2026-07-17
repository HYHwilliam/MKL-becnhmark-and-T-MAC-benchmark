#!/usr/bin/env python3
"""
Usage:
    python3 scripts/plot_benchmark.py
    python3 scripts/plot_benchmark.py --csv benchmark_results.csv --outdir assets
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

THEMES = {
    "light": dict(
        surface="#fcfcfb",
        text_primary="#0b0b0b",
        text_secondary="#52514e",
        muted="#898781",
        gridline="#e1e0d9",
        baseline="#c3c2b7",
        mkl="#2a78d6",
        tmac="#008300",
        lutgemm="#c0392b",
    ),
    "dark": dict(
        surface="#1a1a19",
        text_primary="#ffffff",
        text_secondary="#c3c2b7",
        muted="#898781",
        gridline="#2c2c2a",
        baseline="#383835",
        mkl="#3987e5",
        tmac="#008300",
        lutgemm="#e0564a",
    ),
}


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                size=int(r["matrix_size"]),
                mkl_gflops=float(r["mkl_gflops"]),
                tmac_gflops=float(r["tmac_gflops"]),
                lutgemm_gflops=float(r["lutgemm_gflops"]),
            ))
    rows.sort(key=lambda r: r["size"])
    return rows


def render(rows, theme_name, out_path):
    t = THEMES[theme_name]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    labels = [f"{r['size']}×{r['size']}×1" for r in rows]
    mkl_vals = [r["mkl_gflops"] for r in rows]
    tmac_vals = [r["tmac_gflops"] for r in rows]
    lutgemm_vals = [r["lutgemm_gflops"] for r in rows]

    x = range(len(rows))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    bars_mkl = ax.bar(
        [i - width for i in x], mkl_vals, width,
        label="MKL FP16 (cblas_hgemm)", color=t["mkl"], zorder=3,
    )
    bars_tmac = ax.bar(
        list(x), tmac_vals, width,
        label="T-MAC (4-bit LUT)", color=t["tmac"], zorder=3,
    )
    bars_lutgemm = ax.bar(
        [i + width for i in x], lutgemm_vals, width,
        label="LUT-GEMM CPU (AVX2, μ=8)", color=t["lutgemm"], zorder=3,
    )

    ax.set_ylabel("Performance (GFLOPS)", color=t["text_secondary"], fontsize=11)
    ax.set_xlabel("Matrix Size (M = K, N = 1)", color=t["text_secondary"], fontsize=11)
    ax.set_title(
        "T-MAC (4-bit LUT) vs. LUT-GEMM (μ=8) vs. MKL FP16 — Single-thread GEMV Performance",
        color=t["text_primary"], fontsize=13, fontweight="bold", pad=16,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, color=t["text_secondary"], fontsize=10)

    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.tick_params(axis="y", colors=t["muted"], labelsize=9)
    ax.tick_params(axis="x", colors=t["muted"], length=0)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(t["baseline"])
    ax.spines["bottom"].set_linewidth(0.8)

    ax.yaxis.grid(True, color=t["gridline"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    def direct_labels(bars):
        for b in bars:
            h = b.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8.5, color=t["text_secondary"],
            )

    direct_labels(bars_mkl)
    direct_labels(bars_tmac)
    direct_labels(bars_lutgemm)

    legend = ax.legend(
        loc="upper left", frameon=False, fontsize=9.5,
        labelcolor=t["text_primary"],
    )

    fig.tight_layout()
    fig.savefig(out_path, facecolor=t["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="benchmark_results.csv")
    ap.add_argument("--outdir", default="assets")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    render(rows, "light", f"{args.outdir}/benchmark_chart_light.png")
    render(rows, "dark", f"{args.outdir}/benchmark_chart_dark.png")
    print(f"wrote {args.outdir}/benchmark_chart_light.png and benchmark_chart_dark.png")


if __name__ == "__main__":
    main()
