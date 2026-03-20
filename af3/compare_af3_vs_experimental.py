"""
Compare AEV-PLIG predictions on AF3-predicted structures vs experimental
PDB structures across CASF-2016, 0-LigandBias, and OOD Test benchmarks.

Only compares entries present in BOTH prediction sets (matched IDs).

Usage:
    python compare_af3_vs_experimental.py
    python compare_af3_vs_experimental.py --tag original
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau


# ── Metrics ──────────────────────────────────────────────────────────────

def pcc(y_true, y_pred):
    if len(y_true) < 3:
        return np.nan
    r, _ = pearsonr(y_true, y_pred)
    return r

def ktau(y_true, y_pred):
    if len(y_true) < 3:
        return np.nan
    t, _ = kendalltau(y_true, y_pred)
    return t

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


# ── Loading ──────────────────────────────────────────────────────────────

def load_predictions(path):
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["unique_id"] = df["unique_id"]
    out["pK"] = df["pK"].astype(float)
    out["preds"] = df["preds"].astype(float)
    for i in range(10):
        c = f"preds_{i}"
        if c in df.columns:
            out[c] = df[c].astype(float)
    return out


def match_predictions(exp_df, af3_df):
    """Inner join on unique_id to get only entries present in both."""
    merged = exp_df.merge(
        af3_df, on="unique_id", suffixes=("_exp", "_af3")
    )
    return merged


# ── Per-benchmark analysis ───────────────────────────────────────────────

def analyze_benchmark(name, exp_path, af3_path):
    """Compare experimental vs AF3 predictions for one benchmark."""
    exp = load_predictions(exp_path)
    af3 = load_predictions(af3_path)

    merged = match_predictions(exp, af3)

    n_exp = len(exp)
    n_af3 = len(af3)
    n_matched = len(merged)
    n_exp_only = n_exp - n_matched
    n_af3_only = n_af3 - n_matched

    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")
    print(f"  Experimental predictions: {n_exp}")
    print(f"  AF3 predictions:          {n_af3}")
    print(f"  Matched (compared):       {n_matched}")
    if n_exp_only > 0:
        exp_only_ids = set(exp["unique_id"]) - set(af3["unique_id"])
        print(f"  Experimental only:        {n_exp_only} ({', '.join(sorted(exp_only_ids)[:5])}{'...' if n_exp_only > 5 else ''})")
    if n_af3_only > 0:
        af3_only_ids = set(af3["unique_id"]) - set(exp["unique_id"])
        print(f"  AF3 only:                 {n_af3_only} ({', '.join(sorted(af3_only_ids)[:5])}{'...' if n_af3_only > 5 else ''})")

    truth = merged["pK_exp"].values  # ground truth pK
    pred_exp = merged["preds_exp"].values
    pred_af3 = merged["preds_af3"].values

    # Metrics: predictions vs ground truth
    print(f"\n  {'Metric':<20} {'Experimental':>14} {'AF3':>14} {'Δ (AF3-Exp)':>14}")
    print(f"  {'-'*62}")

    metrics = [
        ("PCC",  pcc(truth, pred_exp),  pcc(truth, pred_af3)),
        ("Ktau", ktau(truth, pred_exp), ktau(truth, pred_af3)),
        ("RMSE", rmse(truth, pred_exp), rmse(truth, pred_af3)),
        ("MAE",  mae(truth, pred_exp),  mae(truth, pred_af3)),
    ]

    for mname, v_exp, v_af3 in metrics:
        delta = v_af3 - v_exp
        sign = "+" if delta >= 0 else ""
        print(f"  {mname:<20} {v_exp:>14.4f} {v_af3:>14.4f} {sign}{delta:>13.4f}")

    # Agreement between experimental and AF3 predictions
    pred_r = pcc(pred_exp, pred_af3)
    pred_mae = mae(pred_exp, pred_af3)
    print(f"\n  Prediction agreement (Exp vs AF3):")
    print(f"    PCC:  {pred_r:.4f}")
    print(f"    MAE:  {pred_mae:.4f} pK")

    # Per-ensemble-member analysis
    exp_cols = [c for c in merged.columns if c.startswith("preds_") and c.endswith("_exp") and c != "preds_exp"]
    af3_cols = [c for c in merged.columns if c.startswith("preds_") and c.endswith("_af3") and c != "preds_af3"]

    if exp_cols and af3_cols:
        exp_std = merged[exp_cols].std(axis=1).mean()
        af3_std = merged[af3_cols].std(axis=1).mean()
        print(f"\n  Ensemble prediction std (mean across entries):")
        print(f"    Experimental: {exp_std:.4f} pK")
        print(f"    AF3:          {af3_std:.4f} pK")

    return {
        "Benchmark": name,
        "N": n_matched,
        "PCC_exp": pcc(truth, pred_exp),
        "PCC_af3": pcc(truth, pred_af3),
        "Ktau_exp": ktau(truth, pred_exp),
        "Ktau_af3": ktau(truth, pred_af3),
        "RMSE_exp": rmse(truth, pred_exp),
        "RMSE_af3": rmse(truth, pred_af3),
        "Pred_agreement_PCC": pred_r,
        "Pred_agreement_MAE": pred_mae,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare AEV-PLIG on AF3 vs experimental structures"
    )
    parser.add_argument("--tag", type=str, default="original")
    parser.add_argument("--pred_dir", type=str, default="output/predictions")
    args = parser.parse_args()

    suffix = "" if args.tag == "original" else f"_{args.tag}"

    print("=" * 72)
    print("  AEV-PLIG: AF3 Structures vs Experimental Structures")
    print(f"  Tag: {args.tag}")
    print("=" * 72)

    benchmarks = [
        (
            "CASF-2016",
            f"{args.pred_dir}/casf2016{suffix}_predictions.csv",
            f"{args.pred_dir}/af3_casf2016{suffix}_predictions.csv",
        ),
        (
            "0-LigandBias",
            f"{args.pred_dir}/0ligandbias{suffix}_predictions.csv",
            f"{args.pred_dir}/af3_0ligandbias{suffix}_predictions.csv",
        ),
        (
            "OOD Test",
            f"{args.pred_dir}/oodtest{suffix}_predictions.csv",
            f"{args.pred_dir}/af3_oodtest{suffix}_predictions.csv",
        ),
    ]

    summary_rows = []
    for name, exp_path, af3_path in benchmarks:
        try:
            row = analyze_benchmark(name, exp_path, af3_path)
            summary_rows.append(row)
        except FileNotFoundError as e:
            print(f"\n  SKIP {name}: {e}")

    # Summary table
    if summary_rows:
        print(f"\n{'='*72}")
        print(f"  SUMMARY")
        print(f"{'='*72}")

        print(f"\n  {'Benchmark':<16} {'N':>5}  {'PCC (Exp)':>10} {'PCC (AF3)':>10} "
              f"{'Ktau (Exp)':>11} {'Ktau (AF3)':>11}")
        print(f"  {'-'*65}")
        for r in summary_rows:
            print(f"  {r['Benchmark']:<16} {r['N']:>5}  "
                  f"{r['PCC_exp']:>10.3f} {r['PCC_af3']:>10.3f} "
                  f"{r['Ktau_exp']:>11.3f} {r['Ktau_af3']:>11.3f}")

        # Save to CSV
        summary_df = pd.DataFrame(summary_rows)
        out_path = f"{args.pred_dir}/af3_vs_experimental_comparison{suffix}.csv"
        summary_df.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()