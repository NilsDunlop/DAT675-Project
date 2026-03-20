"""
Compare our retrained AEV-PLIG predictions against the authors
published predictions across CASF-2016, 0-LigandBias, OOD Test and FEP Benchmark.

Usage:
    python compare_baselines.py --tag binary
    python compare_baselines.py --tag reduced-gaussian-8
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau

PAPER_METRICS = {
    "CASF-2016": {"PCC": 0.86, "Ktau": 0.67},
    "0-LigandBias": {"PCC": 0.37, "Ktau": 0.21},
    "OOD Test": {"PCC": 0.73, "Ktau": 0.55},
    "FEP Benchmark (ligsim90)": {"wmPCC": 0.41, "wmKtau": 0.26},
}

# Helper functions

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

def mae(a, b):
    return np.mean(np.abs(np.array(a) - np.array(b)))


def metrics_row(label, truth, preds):
    return {
        "Source": label,
        "N": len(truth),
        "PCC": pcc(truth, preds),
        "Ktau": ktau(truth, preds),
        "RMSE": rmse(truth, preds),
    }


def weighted_mean_metric(groups, metric_fn, min_group_size=10):
    """Compute weighted-mean of a per-group metric (weight = group size)."""
    vals, weights = [], []
    for gid, grp in groups:
        if len(grp) < min_group_size:
            continue
        v = metric_fn(grp["truth"].values, grp["preds"].values)
        if not np.isnan(v):
            vals.append(v)
            weights.append(len(grp))
    if not vals:
        return np.nan
    return np.average(vals, weights=weights)


def fep_metrics_row(label, df, group_col="group_id"):
    groups = df.groupby(group_col)
    return {
        "Source": label,
        "N": len(df),
        "Series": groups.ngroups,
        "wmPCC": weighted_mean_metric(groups, pcc),
        "wmKtau": weighted_mean_metric(groups, ktau),
        "RMSE": rmse(df["truth"].values, df["preds"].values),
    }


def print_section(title):
    w = 72
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


# Loaders

def load_ours(path, truth_col="pK", pred_col="preds", id_col="unique_id"):
    df = pd.read_csv(path)
    out = pd.DataFrame()
    if id_col in df.columns:
        out["unique_id"] = df[id_col]
    out["truth"] = df[truth_col].astype(float)
    out["preds"] = df[pred_col].astype(float)
    for i in range(10):
        c = f"preds_{i}"
        if c in df.columns:
            out[c] = df[c].astype(float)
    if "group_id" in df.columns:
        out["group_id"] = df["group_id"]
    return out


def load_author_casf_or_ligandbias(path):
    """Author CASF / 0-LigandBias files: columns = [index, truth, preds_0..9, preds]."""
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["truth"] = df["truth"].astype(float)
    out["preds"] = df["preds"].astype(float)
    for i in range(10):
        c = f"preds_{i}"
        if c in df.columns:
            out[c] = df[c].astype(float)
    return out


def load_author_oodtest(path):
    """Author OOD-test file: has unique_id, pK."""
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["unique_id"] = df["unique_id"]
    out["truth"] = df["pK"].astype(float)
    out["preds"] = df["preds"].astype(float)
    for i in range(10):
        c = f"preds_{i}"
        if c in df.columns:
            out[c] = df[c].astype(float)
    return out


def load_fep_truth(truth_source):
    """Load FEP ground truth from fep_benchmark_test.csv (pK column)."""
    return pd.read_csv(truth_source)


def load_author_fep(path, truth_source):
    """Author FEP file: has unique_id, group_id but NO truth.
    We merge truth from truth_source (fep_benchmark_test.csv)."""
    df = pd.read_csv(path)
    truth_df = load_fep_truth(truth_source)
    out = pd.DataFrame()
    out["unique_id"] = df["unique_id"]
    out["group_id"] = df["group_id"]
    out["preds"] = df["preds"].astype(float)
    for i in range(10):
        c = f"preds_{i}"
        if c in df.columns:
            out[c] = df[c].astype(float)
    truth_map = truth_df.set_index("unique_id")["pK"].to_dict()
    out["truth"] = out["unique_id"].map(truth_map).astype(float)
    out = out.dropna(subset=["truth"])
    return out


# Per-benchmark comparison

def leakage_check(name, our_pcc, author_pcc):
    """Warn if our PCC is suspiciously higher than the authors'."""
    if np.isnan(our_pcc) or np.isnan(author_pcc):
        return
    delta = our_pcc - author_pcc
    if delta > 0.10 and our_pcc > 0.90:
        print(f"\n  *** WARNING: Possible data leakage in {name}! ***")
        print(f"      Our PCC ({our_pcc:.3f}) >> Authors ({author_pcc:.3f})")
        print(f"      Check that benchmark IDs were excluded from training data.")


def print_prediction_agreement(ours_df, author_df):
    """Print PCC/MAE between our and the authors' ensemble predictions."""
    if "unique_id" not in ours_df.columns or "unique_id" not in author_df.columns:
        return
    merged = ours_df[["unique_id", "preds"]].merge(
        author_df[["unique_id", "preds"]],
        on="unique_id", suffixes=("_ours", "_authors"),
    )
    if len(merged) == 0:
        return
    r = pcc(merged["preds_ours"], merged["preds_authors"])
    m = mae(merged["preds_ours"], merged["preds_authors"])
    print(f"\n  Prediction-level agreement ({len(merged)} matched complexes):")
    print(f"    PCC between predictions : {r:.4f}")
    print(f"    MAE between predictions : {m:.4f} pK")


def compare_simple(name, ours_df, author_df):
    """Compare metrics for benchmarks without per-group structure."""
    print_section(name)

    our_r = pcc(ours_df["truth"], ours_df["preds"])
    auth_r = pcc(author_df["truth"], author_df["preds"])

    rows = [
        metrics_row("Ours", ours_df["truth"], ours_df["preds"]),
        metrics_row("Authors", author_df["truth"], author_df["preds"]),
    ]
    summary = pd.DataFrame(rows).set_index("Source")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    paper = PAPER_METRICS.get(name)
    if paper:
        print(f"\n  Paper Table 1:  PCC = {paper['PCC']:.2f},  Ktau = {paper['Ktau']:.2f}")

    leakage_check(name, our_r, auth_r)
    print_prediction_agreement(ours_df, author_df)


def compare_fep(name, ours_df, author_df):
    """Compare FEP benchmark using weighted-mean metrics per series."""
    print_section(name)

    rows = [
        fep_metrics_row("Ours", ours_df),
        fep_metrics_row("Authors", author_df),
    ]
    summary = pd.DataFrame(rows).set_index("Source")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    paper = PAPER_METRICS.get(name)
    if paper:
        print(f"\n  Paper Table 1:  wmPCC = {paper['wmPCC']:.2f},  wmKtau = {paper['wmKtau']:.2f}")

    all_groups = sorted(set(ours_df["group_id"]) | set(author_df["group_id"]))
    series_rows = []
    for gid in all_groups:
        o = ours_df[ours_df["group_id"] == gid]
        a = author_df[author_df["group_id"] == gid]
        if len(o) < 10 and len(a) < 10:
            continue
        row = {"Series": gid, "N": len(o)}
        if len(o) >= 3:
            row["Ours_PCC"] = pcc(o["truth"], o["preds"])
            row["Ours_Ktau"] = ktau(o["truth"], o["preds"])
        if len(a) >= 3:
            row["Auth_PCC"] = pcc(a["truth"], a["preds"])
            row["Auth_Ktau"] = ktau(a["truth"], a["preds"])
        series_rows.append(row)

    if series_rows:
        print(f"\n  Per-series comparison (series with >= 10 ligands):")
        sdf = pd.DataFrame(series_rows).set_index("Series")
        print(sdf.to_string(float_format=lambda x: f"{x:.3f}"))

    print_prediction_agreement(ours_df, author_df)


def main():
    parser = argparse.ArgumentParser(description="Compare baseline predictions")
    parser.add_argument("--tag", type=str, default="original",
                        help="Encoding tag suffix (original, binary, etc.)")
    args = parser.parse_args()

    suffix = "" if args.tag == "original" else f"_{args.tag}"

    print("AEV-PLIG Baseline Comparison: Ours vs Authors")
    print(f"  Tag: {args.tag}")

    # CASF-2016
    ours_casf = load_ours(f"output/predictions/casf2016{suffix}_predictions.csv")
    auth_casf = load_author_casf_or_ligandbias(
        "data/author_baselines/AEV-PLIG_casf2016_pdbbind_predictions.csv"
    )
    compare_simple("CASF-2016", ours_casf, auth_casf)

    # 0-LigandBias
    ours_0lb = load_ours(f"output/predictions/0ligandbias{suffix}_predictions.csv")
    auth_0lb = load_author_casf_or_ligandbias(
        "data/author_baselines/AEV-PLIG_0ligandbias_pdbbind_predictions.csv"
    )
    compare_simple("0-LigandBias", ours_0lb, auth_0lb)

    # OOD Test
    ours_ood = load_ours(f"output/predictions/oodtest{suffix}_predictions.csv")
    auth_ood = load_author_oodtest(
        "data/author_baselines/AEV-PLIG_oodtest_predictions.csv"
    )
    compare_simple("OOD Test", ours_ood, auth_ood)

    # FEP Benchmark
    fep_truth_df = load_fep_truth("evaluate/fep/fep_benchmark_test.csv")
    fep_truth_map = fep_truth_df.set_index("unique_id")["pK"].to_dict()

    ours_fep = load_ours(f"output/predictions/fep{suffix}_predictions.csv")
    ours_fep["truth"] = ours_fep["unique_id"].map(fep_truth_map).astype(float)
    if "group_id" not in ours_fep.columns:
        fep_meta = fep_truth_df[["unique_id", "group_id"]]
        ours_fep = ours_fep.merge(fep_meta, on="unique_id", how="left")

    auth_fep = load_author_fep(
        "data/author_baselines/AEV-PLIG_fep_benchmark_pdbbind_ligsim90_predictions.csv",
        truth_source="evaluate/fep/fep_benchmark_test.csv",
    )
    compare_fep("FEP Benchmark (ligsim90)", ours_fep, auth_fep)

    # Summary table
    print_section("Summary Comparison")
    summary_rows = []

    def add_row(name, ours_d, auth_d, paper_d, fep=False):
        row = {"Benchmark": name}
        if fep:
            row["Metric"] = "wmPCC / wmKtau"
            o_g = ours_d.groupby("group_id")
            a_g = auth_d.groupby("group_id")
            row["Ours"] = f"{weighted_mean_metric(o_g, pcc):.3f} / {weighted_mean_metric(o_g, ktau):.3f}"
            row["Authors"] = f"{weighted_mean_metric(a_g, pcc):.3f} / {weighted_mean_metric(a_g, ktau):.3f}"
            row["Paper"] = f"{paper_d['wmPCC']:.2f} / {paper_d['wmKtau']:.2f}"
        else:
            row["Metric"] = "PCC / Ktau"
            row["Ours"] = f"{pcc(ours_d['truth'], ours_d['preds']):.3f} / {ktau(ours_d['truth'], ours_d['preds']):.3f}"
            row["Authors"] = f"{pcc(auth_d['truth'], auth_d['preds']):.3f} / {ktau(auth_d['truth'], auth_d['preds']):.3f}"
            row["Paper"] = f"{paper_d['PCC']:.2f} / {paper_d['Ktau']:.2f}"
        summary_rows.append(row)

    add_row("CASF-2016", ours_casf, auth_casf, PAPER_METRICS["CASF-2016"])
    add_row("0-LigandBias", ours_0lb, auth_0lb, PAPER_METRICS["0-LigandBias"])
    add_row("OOD Test", ours_ood, auth_ood, PAPER_METRICS["OOD Test"])
    add_row("FEP Benchmark", ours_fep, auth_fep, PAPER_METRICS["FEP Benchmark (ligsim90)"], fep=True)

    sdf = pd.DataFrame(summary_rows).set_index("Benchmark")
    print(sdf.to_string())
    print()


if __name__ == "__main__":
    main()
