#!/usr/bin/env python
"""
Motif features vs. CRISPRi has_target_gene.

Stage 1 (univariate): Fisher's exact test for presence features, Mann-Whitney U
otherwise, per TF, with Benjamini-Hochberg correction.

Stage 2 (multivariate): L1-regularised logistic regression, evaluated by 5-fold
cross-validated AUC, for three nested models:
  (a) baseline covariates only (ATAC signal, ABC score, GC, TSS distance,
      constraint, CRE length)
  (b) baseline covariates + ChromBPNet predicted accessibility (pred_log_counts)
  (c) (b) + motif features

CRE_length_hg38 is a covariate because longer elements accumulate more hits.
Feature tables (--feature-type) can be presence, count, importance, count_norm
or soft (see build_element_motif_features.py). K562 elements with no in-CRE hit
are zero-filled for motif features but keep their real pred_log_counts/pred_counts
from region_map.tsv, since every K562 element was scored by the model.

Usage:
    python run_stats.py \
        --features motif_features/features_soft.tsv \
        --feature-type soft \
        --full-csv data/crispri/ALL7_trimmed.csv \
        --region-map region_map.tsv \
        --out-dir stats_soft/

Requires: pandas, scipy, scikit-learn, numpy
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE_COVARIATES = ["ATAC_SignalValue", "ABC_score", "GC_perc",
                    "closest_TSS_distance_genome_hg38", "constraint_z", "CRE_length_hg38"]
ACCESSIBILITY_COVARIATE = "pred_log_counts"
META_COLS_NON_FEATURE = (["has_target_gene", "N_validated_target_genes"]
                          + BASE_COVARIATES + [ACCESSIBILITY_COVARIATE, "pred_counts", "Target Gene Name"])


def bh_correct(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


def load_full_k562(path):
    df = pd.read_csv(path)
    df = df[df["Cell_type"] == "K562"].copy()
    starts = pd.to_numeric(df["CRE_start_hg38"], errors="coerce")
    ends = pd.to_numeric(df["CRE_end_hg38"], errors="coerce")
    n_bad = int((starts.isna() | ends.isna()).sum())
    if n_bad:
        print(f"Dropping {n_bad} K562 rows with non-numeric CRE_start_hg38/CRE_end_hg38 "
              f"(e.g. 'liftover failed') before zero-fill.")
    mask = starts.notna() & ends.notna()
    df = df[mask].copy()
    df["CRE_start_hg38_int"] = starts[mask].astype(int)
    df["CRE_end_hg38_int"] = ends[mask].astype(int)
    df["element_id"] = df["chr"] + ":" + df["CRE_start_hg38_int"].astype(str) + "-" + df["CRE_end_hg38_int"].astype(str)
    return df


def load_region_map_pred(path):
    """Returns dict: 'chr:start-end' (bed_start_1based/bed_end_1based) -> (pred_log_counts, pred_counts)"""
    rm = pd.read_csv(path, sep="\t")
    rm["element_id"] = rm["chrom"] + ":" + rm["bed_start_1based"].astype(str) + "-" + rm["bed_end_1based"].astype(str)
    return dict(zip(rm["element_id"], zip(rm["pred_log_counts"], rm["pred_counts"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--feature-type", required=True,
                     choices=["presence", "count", "importance", "count_norm", "soft"])
    ap.add_argument("--full-csv", required=True)
    ap.add_argument("--region-map", required=True,
                     help="region_map.tsv (with pred_log_counts/pred_counts), used to backfill "
                          "accessibility values for zero-filled elements")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-fill-missing", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.features, sep="\t")
    tf_cols = [c for c in df.columns if c not in ["element_id"] + META_COLS_NON_FEATURE]
    print(f"Loaded {len(df)} elements, {len(tf_cols)} TF motif columns.")

    if not args.no_fill_missing:
        full = load_full_k562(args.full_csv)
        pred_lookup = load_region_map_pred(args.region_map)
        missing_ids = set(full["element_id"]) - set(df["element_id"])
        if missing_ids:
            print(f"Zero-filling {len(missing_ids)} K562 elements with no in-CRE hit at all "
                  f"(motif columns set to 0; accessibility backfilled from region_map).")
            fill_rows = []
            for _, r in full[full["element_id"].isin(missing_ids)].iterrows():
                plc, pc = pred_lookup.get(r["element_id"], (np.nan, np.nan))
                row = {"element_id": r["element_id"],
                       "has_target_gene": r.get("has_target_gene"),
                       "N_validated_target_genes": r.get("N_validated_target_genes"),
                       "Target Gene Name": r.get("Target Gene Name"),
                       "pred_log_counts": plc, "pred_counts": pc}
                for c in BASE_COVARIATES:
                    row[c] = r.get(c)
                for tf in tf_cols:
                    row[tf] = 0
                fill_rows.append(row)
            df = pd.concat([df, pd.DataFrame(fill_rows)], ignore_index=True)

    df["has_target_gene"] = df["has_target_gene"].astype(str).map({"True": 1, "False": 0}).astype("Int64")
    df = df.dropna(subset=["has_target_gene"])
    y = df["has_target_gene"].astype(int)
    print(f"Final N = {len(df)} ({y.sum()} positive, {(1 - y).sum()} negative)")

    # ---------------- Stage 1: univariate ----------------
    results = []
    for tf in tf_cols:
        x = pd.to_numeric(df[tf], errors="coerce").fillna(0)
        if args.feature_type == "presence":
            xb = (x > 0)
            a = (xb & (y == 1)).sum()
            b = (xb & (y == 0)).sum()
            c = ((~xb) & (y == 1)).sum()
            d = ((~xb) & (y == 0)).sum()
            if (a + c) == 0 or (b + d) == 0:
                continue
            odds_ratio, p = stats.fisher_exact([[a, b], [c, d]])
            n_pos_with, n_neg_with, effect = a, b, odds_ratio
        else:
            pos_vals = x[y == 1]
            neg_vals = x[y == 0]
            if pos_vals.sum() == 0 and neg_vals.sum() == 0:
                continue
            try:
                stat, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
            except ValueError:
                continue
            n_pos_with = (pos_vals > 0).sum()
            n_neg_with = (neg_vals > 0).sum()
            effect = pos_vals.mean() - neg_vals.mean()
        results.append({"TF": tf, "n_pos_with_hit": n_pos_with, "n_neg_with_hit": n_neg_with,
                         "effect": effect, "p_value": p})

    res_df = pd.DataFrame(results)
    if len(res_df):
        res_df["p_adj_BH"] = bh_correct(res_df["p_value"].values)
        res_df = res_df.sort_values("p_adj_BH")
    out_path = f"{args.out_dir}/univariate_{args.feature_type}.tsv"
    res_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nStage 1 (univariate, {args.feature_type}) -> {out_path}")
    print(res_df.head(10).to_string(index=False))

    # ---------------- Stage 2: multivariate, three nested models ----------------
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    base_df = df[BASE_COVARIATES].apply(pd.to_numeric, errors="coerce")
    access_df = df[[ACCESSIBILITY_COVARIATE]].apply(pd.to_numeric, errors="coerce")
    motif_df = df[tf_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def eval_model(X, label):
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegressionCV(cv=cv, penalty="l1", solver="liblinear",
                                  Cs=10, scoring="roc_auc", max_iter=5000),
        )
        try:
            auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        except ValueError as e:
            print(f"  [{label}] skipped: {e}")
            return None, None
        pipe.fit(X, y)
        print(f"  [{label}] CV AUC: {auc.mean():.3f} +/- {auc.std():.3f}  (n_features={X.shape[1]})")
        return auc, pipe

    print("\nStage 2a: baseline covariates only (incl. CRE_length_hg38)")
    auc_a, _ = eval_model(base_df, "covariates")

    print("\nStage 2b: covariates + ChromBPNet predicted accessibility (pred_log_counts)")
    X_b = pd.concat([base_df, access_df], axis=1)
    auc_b, _ = eval_model(X_b, "covariates+accessibility")

    print(f"\nStage 2c: covariates + accessibility + motif features ({args.feature_type})")
    X_c = pd.concat([base_df, access_df, motif_df], axis=1)
    auc_c, pipe_c = eval_model(X_c, "covariates+accessibility+motifs")

    if auc_a is not None and auc_b is not None:
        print(f"\nAUC delta, accessibility alone vs covariates: {auc_b.mean() - auc_a.mean():+.3f}")
    if auc_b is not None and auc_c is not None and pipe_c is not None:
        print(f"AUC delta, motifs on top of covariates+accessibility: {auc_c.mean() - auc_b.mean():+.3f}")
        clf = pipe_c.named_steps["logisticregressioncv"]
        coefs = pd.Series(clf.coef_[0], index=X_c.columns).sort_values(key=np.abs, ascending=False)
        coefs = coefs[coefs != 0]
        coef_path = f"{args.out_dir}/multivariate_{args.feature_type}_coefs.tsv"
        coefs.to_csv(coef_path, sep="\t", header=["standardized_coefficient"])
        print(f"\nNon-zero standardized coefficients (L1-selected, full model) -> {coef_path}")
        print(coefs.to_string())


if __name__ == "__main__":
    main()
