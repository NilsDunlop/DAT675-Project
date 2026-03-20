"""
plot_scatter_grid.py — Unified 3×2 scatter plot grid for paper figure.

Layout:
  Row 1:  (A) Exp — Baseline          (B) Exp — Binary
  Row 2:  (C) Exp — Distance-Binned   (D) Exp — Cutoff5 (Topology)
  Row 3:  (E) AF3 — Baseline          (F) AF3 — Binary

All 3 benchmarks overlaid per panel. No panel titles (figure caption in paper).
Style matches the AF3 scatter script: dark edges, colored by benchmark, metrics box.

Usage:
    python plot_scatter_grid.py
    python plot_scatter_grid.py --pred_dir output/predictions --output output/figures/scatter_grid.pdf
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, kendalltau


# ── Style ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "figure.autolayout": False,
})

# ── Benchmark appearance ─────────────────────────────────────────────────
PALETTE = {
    "0ligandbias": "#2171B5",
    "casf2016":    "#FD8D3C",
    "oodtest":     "#9467BD",
    "fep":         "#2CA02C",
}

LABELS = {
    "0ligandbias": "0LigandBias",
    "casf2016":    "CASF-2016",
    "oodtest":     "OOD Test",
    "fep":         "FEP",
}

PLOT_ORDER = ["0ligandbias", "casf2016", "oodtest", "fep"]
BENCHMARKS = ["casf2016", "0ligandbias", "oodtest"]
BENCHMARKS_WITH_FEP = ["casf2016", "0ligandbias", "oodtest", "fep"]

PRED_COLS = [f"preds_{i}" for i in range(10)]


# ── Panel configuration ──────────────────────────────────────────────────
# Each tuple: (panel_label, file_pattern, is_af3, include_fep)
# file_pattern uses {bench} placeholder
PANELS = [
    # Row 1 — Experimental (with FEP)
    ("A", "{bench}_predictions.csv",                 False, True),
    ("B", "{bench}_binary_predictions.csv",          False, True),
    # Row 2 — Experimental (with FEP)
    ("C", "{bench}_distance-binned_predictions.csv", False, True),
    ("D", "{bench}_cutoff5_predictions.csv",         False, True),
    # Row 3 — AF3 (no FEP)
    ("E", "af3_{bench}_predictions.csv",             True,  False),
    ("F", "af3_{bench}_binary_predictions.csv",      True,  False),
]


# ── Data loading ─────────────────────────────────────────────────────────

def load_predictions(path):
    """Load a predictions CSV, keeping pK, preds, and ensemble columns."""
    df = pd.read_csv(path)
    cols = ["unique_id", "pK", "preds"]
    for c in PRED_COLS:
        if c in df.columns:
            cols.append(c)
    return df[[c for c in cols if c in df.columns]]


def load_panel_data(pred_dir, pattern, benchmarks):
    """Load all benchmarks for one panel. Returns {bench: DataFrame}."""
    datasets = {}
    for bench in benchmarks:
        fname = pattern.format(bench=bench)
        path = f"{pred_dir}/{fname}"
        try:
            df = load_predictions(path)
            datasets[bench] = df
            print(f"    Loaded {fname}: {len(df)} entries")
        except FileNotFoundError:
            print(f"    SKIP: {fname} not found")
    return datasets


# ── Metrics ──────────────────────────────────────────────────────────────

def compute_metrics(truth, preds_df):
    """
    Compute PCC and Kendall tau.
    If ensemble columns exist, report mean ± std across members.
    Otherwise, compute from the ensemble mean prediction.
    """
    pccs, taus = [], []

    available_pred_cols = [c for c in PRED_COLS if c in preds_df.columns]

    if available_pred_cols:
        for col in available_pred_cols:
            p = preds_df[col].values
            mask = np.isfinite(truth) & np.isfinite(p)
            if mask.sum() < 3:
                continue
            r, _ = pearsonr(truth[mask], p[mask])
            t, _ = kendalltau(truth[mask], p[mask])
            pccs.append(r)
            taus.append(t)

    if not pccs:
        # Fallback: single metric from ensemble mean
        preds = preds_df["preds"].values
        mask = np.isfinite(truth) & np.isfinite(preds)
        if mask.sum() >= 3:
            r, _ = pearsonr(truth[mask], preds[mask])
            t, _ = kendalltau(truth[mask], preds[mask])
            pccs.append(r)
            taus.append(t)

    if not pccs:
        return None

    return {
        "pcc_mean": np.mean(pccs),
        "pcc_std": np.std(pccs),
        "tau_mean": np.mean(taus),
        "tau_std": np.std(taus),
    }


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_panel(ax, datasets, panel_label, row_idx, col_idx, n_rows, n_cols):
    """Draw one scatter panel with all benchmarks overlaid."""

    # Diagonal reference
    ax.plot([-2, 13], [-2, 13], "--", color="gray", linewidth=1, zorder=0)

    # Scatter each benchmark
    for bench in PLOT_ORDER:
        if bench not in datasets:
            continue
        df = datasets[bench]
        sns.scatterplot(
            x=df["pK"], y=df["preds"],
            color=PALETTE[bench],
            s=50, edgecolor="black", alpha=0.5,
            ax=ax, label=LABELS[bench],
            zorder=2,
        )

    # Legend (inside panel, upper-left below the panel label)
    ax.legend(
        loc="upper left", bbox_to_anchor=(0.02, 0.92),
        frameon=True, framealpha=0.6, edgecolor="gray", fontsize=11,
    )

    # Panel label (A–F)
    ax.text(
        0.02, 0.97, panel_label,
        transform=ax.transAxes,
        fontsize=16, fontweight="bold", verticalalignment="top",
    )

    # Metrics text box
    metric_lines = []
    max_name_len = max(
        len(LABELS[b]) for b in PLOT_ORDER if b in datasets
    ) if datasets else 0

    for bench in PLOT_ORDER:
        if bench not in datasets:
            continue
        df = datasets[bench]
        truth = df["pK"].values
        m = compute_metrics(truth, df)
        if m is None:
            continue

        name = LABELS[bench]
        if m["pcc_std"] > 0:
            line = (
                f"{name.ljust(max_name_len)}  "
                f"PCC={m['pcc_mean']:.2f} $\\pm$ {m['pcc_std']:.2f}, "
                f"τ={m['tau_mean']:.2f} $\\pm$ {m['tau_std']:.2f}"
            )
        else:
            line = (
                f"{name.ljust(max_name_len)}  "
                f"PCC={m['pcc_mean']:.2f}, "
                f"τ={m['tau_mean']:.2f}"
            )
        metric_lines.append(line)

    if metric_lines:
        metrics_text = "\n".join(metric_lines)
        ax.text(
            0.18, 0.15, metrics_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=11.5,
            bbox=dict(
                facecolor="white", alpha=0.7, edgecolor="gray",
                boxstyle="round,pad=0.5",
            ),
        )

    # Axis labels: only bottom row gets xlabel, only left column gets ylabel
    if row_idx == n_rows - 1:
        ax.set_xlabel("Experimental pK")
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)

    if col_idx == 0:
        ax.set_ylabel("Predicted pK")
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    ax.set_xlim(-2.5, 12.5)
    ax.set_ylim(-2.5, 12.5)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate 3×2 scatter grid for paper figure"
    )
    parser.add_argument(
        "--pred_dir", default="output/predictions",
        help="Directory containing prediction CSVs",
    )
    parser.add_argument(
        "--output", default="output/figures/scatter_grid.pdf",
        help="Output path (both .pdf and .png produced)",
    )
    args = parser.parse_args()

    n_rows, n_cols = 3, 2

    # ── Load data for all 6 panels ───────────────────────────────────────
    all_panel_data = []

    # For AF3 panels, filter to entries that overlap with experimental
    # (so comparisons are on the same complexes)
    exp_baseline_ids = {}  # bench -> set of unique_ids

    for panel_label, pattern, is_af3, include_fep in PANELS:
        bench_list = BENCHMARKS_WITH_FEP if include_fep else BENCHMARKS
        print(f"\nPanel {panel_label}: {pattern}"
              f"{' (+FEP)' if include_fep else ''}")
        datasets = load_panel_data(args.pred_dir, pattern, bench_list)

        # Cache experimental baseline IDs for filtering AF3 later
        if pattern == "{bench}_predictions.csv":
            for bench, df in datasets.items():
                if bench != "fep":
                    exp_baseline_ids[bench] = set(df["unique_id"].values)

        all_panel_data.append((panel_label, datasets, is_af3))

    # ── Filter AF3 panels to matched IDs ─────────────────────────────────
    for i, (panel_label, datasets, is_af3) in enumerate(all_panel_data):
        if not is_af3:
            continue
        filtered = {}
        for bench, df in datasets.items():
            if bench in exp_baseline_ids:
                matched = df[df["unique_id"].isin(exp_baseline_ids[bench])]
                n_before = len(df)
                n_after = len(matched)
                if n_before != n_after:
                    print(f"  Panel {panel_label}/{bench}: "
                          f"filtered {n_before} → {n_after} "
                          f"(matched to experimental)")
                filtered[bench] = matched
            else:
                filtered[bench] = df
        all_panel_data[i] = (panel_label, filtered, is_af3)

    # ── Create figure ────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 6 * n_rows),
        constrained_layout=True,
    )

    for i, (panel_label, datasets, is_af3) in enumerate(all_panel_data):
        row_idx = i // n_cols
        col_idx = i % n_cols
        ax = axes[row_idx, col_idx]

        if not datasets:
            ax.text(
                0.5, 0.5, f"Panel {panel_label}\n(no data)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=14, color="gray",
            )
            ax.set_xlim(-2.5, 12.5)
            ax.set_ylim(-2.5, 12.5)
            continue

        plot_panel(ax, datasets, panel_label, row_idx, col_idx, n_rows, n_cols)

    # ── Save ─────────────────────────────────────────────────────────────
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {args.output}")

    png_path = args.output.replace(".pdf", ".png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()