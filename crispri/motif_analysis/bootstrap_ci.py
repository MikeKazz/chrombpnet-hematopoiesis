#!/usr/bin/env python
"""
bootstrap_ci.py: bootstrap confidence intervals for the motif features in the
CRISPRi has_target_gene models (univariate and L1-logistic multivariate).

The multivariate bootstrap stores the standardized LASSO coefficient of every
feature (covariates, accessibility, all motif features) in every replicate. For
each feature it reports the bootstrap median and 95% percentile CI, the fraction
of replicates in which the coefficient was non-zero (exactly 0.0 after L1
shrinkage counts as zero), and the fraction that were negative / positive, plus
the point estimate from one fit on the full data at the same fixed C.

Outputs:
    bootstrap_summary.tsv
    bootstrap_multivariate_all_features_<feature_type>.tsv
    bootstrap_multivariate_replicates_<feature_type>.tsv.gz
        (one row per valid replicate, one column per feature)

Random numbers are drawn in a fixed order (univariate block, then multivariate
block, for presence and then importance) from a fixed default seed, so results
are reproducible given the same inputs and library versions.

Usage:
    python bootstrap_ci.py \
        --features-presence motif_features/features_presence.tsv \
        --features-importance motif_features/features_importance.tsv \
        --full-csv data/crispri/ALL7_trimmed.csv \
        --region-map region_map.tsv \
        --tfs CTCF_2 GATA_0 NFE_0 \
        --n-boot 500 \
        --out-dir bootstrap_results/
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_COVARIATES = ["ATAC_SignalValue", "ABC_score", "GC_perc",
                    "closest_TSS_distance_genome_hg38", "constraint_z", "CRE_length_hg38"]
ACCESSIBILITY_COVARIATE = "pred_log_counts"
META_COLS_NON_FEATURE = (["has_target_gene", "N_validated_target_genes"]
                          + BASE_COVARIATES + [ACCESSIBILITY_COVARIATE, "pred_counts", "Target Gene Name"])


def load_full_k562(path):
    df = pd.read_csv(path)
    df = df[df["Cell_type"] == "K562"].copy()
    starts = pd.to_numeric(df["CRE_start_hg38"], errors="coerce")
    ends = pd.to_numeric(df["CRE_end_hg38"], errors="coerce")
    mask = starts.notna() & ends.notna()
    df = df[mask].copy()
    df["CRE_start_hg38_int"] = starts[mask].astype(int)
    df["CRE_end_hg38_int"] = ends[mask].astype(int)
    df["element_id"] = df["chr"] + ":" + df["CRE_start_hg38_int"].astype(str) + "-" + df["CRE_end_hg38_int"].astype(str)
    return df


def load_region_map_pred(path):
    rm = pd.read_csv(path, sep="\t")
    rm["element_id"] = rm["chrom"] + ":" + rm["bed_start_1based"].astype(str) + "-" + rm["bed_end_1based"].astype(str)
    return dict(zip(rm["element_id"], zip(rm["pred_log_counts"], rm["pred_counts"])))


def load_and_fill(features_path, full_csv, region_map_path):
    df = pd.read_csv(features_path, sep="\t")
    tf_cols = [c for c in df.columns if c not in ["element_id"] + META_COLS_NON_FEATURE]

    full = load_full_k562(full_csv)
    pred_lookup = load_region_map_pred(region_map_path)
    missing_ids = set(full["element_id"]) - set(df["element_id"])
    if missing_ids:
        fill_rows = []
        for _, r in full[full["element_id"].isin(missing_ids)].iterrows():
            plc, pc = pred_lookup.get(r["element_id"], (np.nan, np.nan))
            row = {"element_id": r["element_id"], "has_target_gene": r.get("has_target_gene"),
                   "N_validated_target_genes": r.get("N_validated_target_genes"),
                   "Target Gene Name": r.get("Target Gene Name"), "pred_log_counts": plc, "pred_counts": pc}
            for c in BASE_COVARIATES:
                row[c] = r.get(c)
            for tf in tf_cols:
                row[tf] = 0
            fill_rows.append(row)
        df = pd.concat([df, pd.DataFrame(fill_rows)], ignore_index=True)

    df["has_target_gene"] = df["has_target_gene"].astype(str).map({"True": 1, "False": 0}).astype("Int64")
    df = df.dropna(subset=["has_target_gene"]).reset_index(drop=True)
    y = df["has_target_gene"].astype(int)
    return df, y, tf_cols


def univariate_effect(x, y, feature_type):
    x = pd.to_numeric(x, errors="coerce").fillna(0).values
    y = np.asarray(y)
    if feature_type == "presence":
        xb = x > 0
        a = float(((xb) & (y == 1)).sum()) + 0.5
        b = float(((xb) & (y == 0)).sum()) + 0.5
        c = float(((~xb) & (y == 1)).sum()) + 0.5
        d = float(((~xb) & (y == 0)).sum()) + 0.5
        return (a / c) / (b / d)  # odds ratio, Haldane-Anscombe corrected
    else:
        return x[y == 1].mean() - x[y == 0].mean()


def fit_full_model(df, y, tf_cols):
    base_df = df[BASE_COVARIATES].apply(pd.to_numeric, errors="coerce")
    access_df = df[[ACCESSIBILITY_COVARIATE]].apply(pd.to_numeric, errors="coerce")
    motif_df = df[tf_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = pd.concat([base_df, access_df, motif_df], axis=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pipe = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegressionCV(cv=cv, penalty="l1", solver="liblinear", Cs=10, scoring="roc_auc", max_iter=5000),
    )
    pipe.fit(X, y)
    best_C = pipe.named_steps["logisticregressioncv"].C_[0]
    return X, best_C


def make_fixed_c_pipeline(best_C):
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(penalty="l1", solver="liblinear", C=best_C, max_iter=5000),
    )


def full_data_coefficients(X, y, best_C):
    pipe = make_fixed_c_pipeline(best_C)
    pipe.fit(X, y)
    return pd.Series(pipe.named_steps["logisticregression"].coef_[0], index=X.columns)


def bootstrap_multivariate(X, y, best_C, n_boot, rng):
    """Returns a DataFrame (valid replicates x all features) of standardized
    L1-logistic coefficients. Same resampling draw as the original script
    (one rng.integers call per replicate)."""
    n = len(y)
    feats = list(X.columns)
    coefs = np.full((n_boot, len(feats)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X.iloc[idx], y.iloc[idx]
        if yb.nunique() < 2:
            continue
        pipe = make_fixed_c_pipeline(best_C)
        pipe.fit(Xb, yb)
        coefs[b] = pipe.named_steps["logisticregression"].coef_[0]
    valid = ~np.isnan(coefs[:, 0])
    return pd.DataFrame(coefs[valid], columns=feats)


def bootstrap_univariate(df, y, tfs, feature_type, n_boot, rng):
    n = len(y)
    results = {tf: [] for tf in tfs}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y.iloc[idx].reset_index(drop=True)
        for tf in tfs:
            xb = df[tf].iloc[idx].reset_index(drop=True)
            results[tf].append(univariate_effect(xb, yb, feature_type))
    return {tf: np.array(v) for tf, v in results.items()}


def summarize(name, arr, null_value):
    lo, med, hi = np.percentile(arr, [2.5, 50, 97.5])
    crosses = lo <= null_value <= hi
    flag = "  <-- CI crosses null, not robust" if crosses else ""
    print(f"  {name:38s} median={med:10.4f}  95% CI=[{lo:10.4f}, {hi:10.4f}]{flag}")
    return {"name": name, "median": med, "ci_lo": lo, "ci_hi": hi, "crosses_null": crosses}


def coefficient_table(reps, point_estimates):
    rows = []
    for feat in reps.columns:
        v = reps[feat].values
        lo, med, hi = np.percentile(v, [2.5, 50, 97.5])
        rows.append({
            "feature": feat,
            "full_data_coef": point_estimates[feat],
            "boot_median": med, "boot_ci_lo": lo, "boot_ci_hi": hi,
            "ci_excludes_zero": not (lo <= 0 <= hi),
            "n_valid_reps": len(v),
            "n_nonzero_reps": int((v != 0).sum()),
            "frac_nonzero": float((v != 0).mean()),
            "frac_negative": float((v < 0).mean()),
            "frac_positive": float((v > 0).mean()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-presence", required=True)
    ap.add_argument("--features-importance", required=True)
    ap.add_argument("--full-csv", required=True)
    ap.add_argument("--region-map", required=True)
    ap.add_argument("--tfs", nargs="+", default=["CTCF_2", "GATA_0", "NFE_0"])
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_rows = []

    for feature_type, path in [("presence", args.features_presence), ("importance", args.features_importance)]:
        print(f"\n{'='*70}\nFeature type: {feature_type}\n{'='*70}")
        df, y, tf_cols = load_and_fill(path, args.full_csv, args.region_map)
        print(f"N = {len(df)} ({y.sum()} positive, {(1 - y).sum()} negative)")

        print("\n-- Univariate bootstrap --")
        uni = bootstrap_univariate(df, y, args.tfs, feature_type, args.n_boot, rng)
        null_uni = 1.0 if feature_type == "presence" else 0.0
        for tf in args.tfs:
            row = summarize(f"{tf} (univariate, {feature_type})", uni[tf], null_uni)
            row.update({"tf": tf, "feature_type": feature_type, "level": "univariate"})
            all_rows.append(row)

        print("\n-- Multivariate bootstrap (fixed C from full-data LogisticRegressionCV) --")
        X, best_C = fit_full_model(df, y, tf_cols)
        print(f"  selected C = {best_C:.5f}")
        reps = bootstrap_multivariate(X, y, best_C, args.n_boot, rng)
        point = full_data_coefficients(X, y, best_C)
        table = coefficient_table(reps, point)
        table.to_csv(os.path.join(args.out_dir, f"bootstrap_multivariate_all_features_{feature_type}.tsv"),
                     sep="\t", index=False)
        reps.to_csv(os.path.join(args.out_dir, f"bootstrap_multivariate_replicates_{feature_type}.tsv.gz"),
                    sep="\t", index=False, compression="gzip")
        print(f"  {len(reps)} valid replicates of {args.n_boot}")

        for tf in args.tfs:
            row = summarize(f"{tf} (multivariate coef, {feature_type})", reps[tf].values, 0.0)
            t = table[table["feature"] == tf].iloc[0]
            row.update({"tf": tf, "feature_type": feature_type, "level": "multivariate",
                        "n_valid_reps": int(t["n_valid_reps"]), "n_nonzero_reps": int(t["n_nonzero_reps"]),
                        "frac_nonzero": t["frac_nonzero"], "frac_negative": t["frac_negative"],
                        "frac_positive": t["frac_positive"], "full_data_coef": t["full_data_coef"]})
            print(f"      non-zero in {int(t['n_nonzero_reps'])}/{int(t['n_valid_reps'])} replicates "
                  f"({t['frac_nonzero']:.1%}); negative in {t['frac_negative']:.1%}")
            all_rows.append(row)

    out_path = os.path.join(args.out_dir, "bootstrap_summary.tsv")
    pd.DataFrame(all_rows).to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
