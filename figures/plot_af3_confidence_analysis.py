"""
Analyze correlation between AF3 confidence metrics and prediction error.

Shows whether AF3's own confidence scores predict where binding affinity
predictions degrade.

Produces a single-panel figure:
  |pred_AF3 - truth| vs ligand_iptm (colored by benchmark)

Usage:
    python plot_af3_confidence_analysis.py
    python plot_af3_confidence_analysis.py --confidence_csv af3_confidences.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


# ── Global style (matching scatter plot) ─────────────────────────────────
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

# ── Colors (matching scatter plot palette) ───────────────────────────────
PALETTE = {
    "0LigandBias": "#2171B5",
    "CASF-2016":   "#FD8D3C",
    "OOD Test":    "#9467BD",
}


def load_predictions(path):
    df = pd.read_csv(path)
    return df[["unique_id", "pK", "preds"]].rename(
        columns={"pK": "truth"}
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default="output/predictions")
    parser.add_argument("--confidence_csv", default="data/af_processed/af3_confidences.csv",
                        help="Path to af3_confidences.csv with AF3 confidence metrics")
    parser.add_argument("--tag", default="original")
    parser.add_argument("--output", default="output/predictions/af3_confidence_analysis.pdf")
    args = parser.parse_args()

    suffix = "" if args.tag == "original" else f"_{args.tag}"

    # Load confidence metrics
    conf = pd.read_csv(args.confidence_csv)
    # Normalize pdb_id column
    if "pdb_id" in conf.columns:
        conf["unique_id"] = conf["pdb_id"].str.lower().str.strip()
    elif "unique_id" not in conf.columns:
        print("ERROR: confidence CSV needs 'pdb_id' or 'unique_id' column")
        return

    print(f"Loaded {len(conf)} confidence entries")
    print(f"Columns: {list(conf.columns)}")

    # Load predictions for each benchmark
    benchmarks = {
        "casf2016": ("CASF-2016", "#FD8D3C"),
        "0ligandbias": ("0LigandBias", "#2171B5"),
        "oodtest": ("OOD Test", "#9467BD"),
    }

    all_rows = []
    for bench, (label, color) in benchmarks.items():
        try:
            exp = load_predictions(f"{args.pred_dir}/{bench}{suffix}_predictions.csv")
            af3 = load_predictions(f"{args.pred_dir}/af3_{bench}{suffix}_predictions.csv")
        except FileNotFoundError as e:
            print(f"  SKIP {bench}: {e}")
            continue

        # Merge exp and af3 predictions
        merged = exp.merge(af3, on="unique_id", suffixes=("_exp", "_af3"))

        # Filter confidence to this benchmark, then merge
        bench_map = {
            "casf2016": "casf2016",
            "0ligandbias": "0ligandbias",
            "oodtest": "oodtest",
        }
        if "benchmark" in conf.columns:
            conf_bench = conf[conf["benchmark"] == bench_map.get(bench, bench)]
        else:
            conf_bench = conf
        merged = merged.merge(conf_bench, on="unique_id", how="inner")

        merged["benchmark"] = label
        merged["color"] = color
        merged["af3_error"] = np.abs(merged["preds_af3"] - merged["truth_af3"])

        all_rows.append(merged)
        print(f"  {bench}: {len(merged)} entries matched with confidence")

    if not all_rows:
        print("No data to plot!")
        return

    df = pd.concat(all_rows, ignore_index=True)

    # Identify which confidence columns are available
    conf_candidates = ["ligand_iptm", "iptm", "ranking_score", "ligand_pae_min", "ptm"]
    available_conf = [c for c in conf_candidates if c in df.columns]

    if not available_conf:
        print(f"ERROR: No confidence columns found. Available: {list(df.columns)}")
        return

    print(f"\nAvailable confidence metrics: {available_conf}")

    # Pick the primary confidence metric
    primary_conf = "ligand_iptm" if "ligand_iptm" in available_conf else available_conf[0]
    print(f"Using primary confidence metric: {primary_conf}")

    # Overall correlations
    print(f"\nCorrelations with {primary_conf}:")
    for target, target_label in [
        ("af3_error", "|pred_AF3 - truth|"),
    ]:
        valid = df[[primary_conf, target]].dropna()
        if len(valid) > 3:
            r, p = pearsonr(valid[primary_conf], valid[target])
            rho, p_s = spearmanr(valid[primary_conf], valid[target])
            print(f"  {target_label:<30s}  PCC={r:.3f} (p={p:.1e}), Spearman={rho:.3f} (p={p_s:.1e})")

    # ── Figure: single panel ──
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5), constrained_layout=True)

    for bench_label in df["benchmark"].unique():
        sub = df[df["benchmark"] == bench_label]
        color = sub["color"].iloc[0]

        sns.scatterplot(
            x=sub[primary_conf], y=sub["af3_error"],
            color=color, s=50, edgecolor="black", alpha=0.5,
            ax=ax, label=bench_label, zorder=2,
        )

    # Trend line (overall)
    valid = df[[primary_conf, "af3_error"]].dropna()
    if len(valid) > 10:
        z = np.polyfit(valid[primary_conf], valid["af3_error"], 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(valid[primary_conf].min(), valid[primary_conf].max(), 100)
        ax.plot(x_range, p_line(x_range), '--', color='gray', linewidth=1.5, alpha=0.7)

    ax.set_xlabel("AF3 Ligand ipTM")
    ax.set_ylabel("|Predicted pK − Experimental pK|")

    # Legend
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.97), frameon=True,
              framealpha=0.6, edgecolor="gray", fontsize=11)

    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {args.output}")

    png_path = args.output.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {png_path}")

    # ── Supplementary: correlation matrix across all confidence metrics ──
    print(f"\n{'='*60}")
    print("Correlation matrix: confidence metrics vs prediction error")
    print(f"{'='*60}")

    for conf_col in available_conf:
        valid = df[[conf_col, "af3_error"]].dropna()
        if len(valid) > 3:
            r, p = pearsonr(valid[conf_col], valid["af3_error"])
            print(f"  {conf_col:<20s} vs |error|  PCC={r:+.3f} (p={p:.1e})")


if __name__ == "__main__":
    main()