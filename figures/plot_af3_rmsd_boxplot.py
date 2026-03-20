"""
plot_rmsd_boxplot_vertical.py — Vertical 3×1 boxplot: prediction error
stratified by ligand RMSD bins, one panel per benchmark.

Designed for a two-column paper layout where vertical figures fit better.

Usage:
    python plot_rmsd_boxplot_vertical.py \
        --input rmsd_correlation/rmsd_prediction_merged.csv \
        --output output/figures/rmsd_boxplot_vertical.pdf
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# ── Style (matching scatter grid) ────────────────────────────────────────
sns.set_theme(style="whitegrid")

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "figure.autolayout": False,
})

BENCH_COLORS = {
    "0ligandbias": "#2171B5",
    "casf2016":    "#FD8D3C",
    "oodtest":     "#9467BD",
}

BENCH_LABELS = {
    "0ligandbias": "0LigandBias",
    "casf2016":    "CASF-2016",
    "oodtest":     "OOD Test",
}

BENCH_ORDER = ["0ligandbias", "casf2016", "oodtest"]
PANEL_LABELS = ["A", "B", "C"]

RMSD_BINS = [0, 2, 5, 10, np.inf]
RMSD_BIN_LABELS = ["< 2 Å", "2–5 Å", "5–10 Å", "> 10 Å"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rmsd_correlation/rmsd_prediction_merged.csv")
    parser.add_argument("--output", default="output/figures/rmsd_boxplot_vertical.pdf")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["benchmark"] = df["benchmark"].str.strip().str.lower()
    df["ligand_rmsd"] = pd.to_numeric(df["ligand_rmsd"], errors="coerce")
    df["abs_error"] = pd.to_numeric(df["abs_error"], errors="coerce")

    fig, axes = plt.subplots(3, 1, figsize=(5.5, 12), constrained_layout=True)

    for ax, bench, panel_label in zip(axes, BENCH_ORDER, PANEL_LABELS):
        sub = df[df["benchmark"] == bench].copy()
        x = sub["ligand_rmsd"].values
        y = sub["abs_error"].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        color = BENCH_COLORS.get(bench, "#666")

        bins = np.digitize(x, RMSD_BINS) - 1
        data_by_bin = []
        labels_used = []
        for i, label in enumerate(RMSD_BIN_LABELS):
            vals = y[bins == i]
            if len(vals) > 0:
                data_by_bin.append(vals)
                labels_used.append(label)
            else:
                data_by_bin.append([])
                labels_used.append(label)

        bp = ax.boxplot(
            data_by_bin, labels=labels_used, patch_artist=True,
            widths=0.6, showfliers=True,
            flierprops=dict(marker=".", markersize=4, alpha=0.4),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for median in bp["medians"]:
            median.set_color(color)
            median.set_linewidth(2)

        # Panel label
        ax.text(0.02, 0.97, panel_label, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top")

        # Benchmark name inside panel
        ax.text(0.98, 0.97, BENCH_LABELS.get(bench, bench),
                transform=ax.transAxes, fontsize=13, fontweight="medium",
                va="top", ha="right")

        ax.set_ylabel("|pK Prediction Error|")
        ax.set_ylim(bottom=0)

        # Only bottom panel gets xlabel
        if bench == BENCH_ORDER[-1]:
            ax.set_xlabel("Ligand RMSD Bin")
        else:
            ax.set_xlabel("")

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output}")

    png_path = args.output.replace(".pdf", ".png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()