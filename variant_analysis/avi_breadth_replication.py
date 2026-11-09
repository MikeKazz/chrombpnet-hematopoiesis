#!/usr/bin/env python
"""
avi_breadth_replication.py

Does the AlphaGenome Variant Impact (AVI) score show the same breadth trend as
ChromBPNet? Cellular breadth is taken unchanged from the ChromBPNet pipeline, so the
only thing that changes between the two analyses is the score.

Breadth and categories (identical to 03_breadth.py)
---------------------------------------------------
  breadth  = number of the 17 cell types where `called` is True
             (accessible AND within-cell-type empirical P < 0.01)
  null 0 | specific 1 | multiple 2-13 | shared 14-17
  Variants accessible in no cell type are excluded, as in magnitude_by_category().
  "Regulatory magnitude" = max |logFC| over the cell types where the variant is
  accessible (verified against the pipeline's breadth_magnitude_groups.csv).

Analyses
--------
  A. Category means (primary). Mean AVI per category, mean +/- 1.96 SEM, with the
     tests 03_breadth.py uses: OLS (default SEs) on category dummies with null as
     reference, and Welch t-tests between active categories. PHRED and raw AVI.
  B. Cutoff selection. Fraction of each category reaching PHRED >= 10 / 20 / 30
     (Wilson 95% CIs), odds ratio vs null (Fisher), and each category's share of the
     selected active set.
  C. Discrimination. AUROC per contrast (Hanley-McNeil SE).
  D. Effect-size and accessibility adjustment. For each pairwise contrast:
       1. trim both groups to their COMMON SUPPORT of regulatory magnitude;
       2. stratify on quantile bins of magnitude computed within that support
          (optionally crossed with accessibility-breadth bins);
       3. keep strata containing >= 2 variants of each group;
       4. OLS of PHRED on group + stratum fixed effects, HC3 standard errors.
     Cross-checks: a continuous model among active variants
     (PHRED ~ category + B-spline(log10 magnitude, df 6) + C(n_accessible), HC3), and
     a Mantel-Haenszel test of passing a PHRED cutoff, stratified the same way.
     Accessibility breadth = number of cell types where the variant is accessible; it
     separates breadth of the variant's EFFECT from breadth of the ELEMENT's openness.
  E. Threshold ladder. Breadth recomputed at stricter calling rules from `pval`
     (empirical P) and `qval` (BH-FDR). For raw-P rules, the number of breadth-1
     variants expected from chance calls alone is reported:
         E[chance specific] = sum_v  n_v * a * (1 - a)^(n_v - 1),
     n_v = cell types where variant v is accessible, a = per-test threshold. This is an
     upper bound that assumes every call is null and calls are independent.
  F. Sensitivity. Allele-swapped matches included; non-coding only (VEP most-severe
     consequence); promoter-proximal (|TSS distance to protein-coding gene| <= 1 kb)
     vs distal. Each subset reports the unadjusted and adjusted contrasts.

AVI lookup
----------
Variants come from variants.parquet (chr, pos, allele1, allele2; 1-based, allele1 =
ref). The local AVI file is queried with `tabix -R` (or pysam if tabix is not on the
PATH). Matches are classified as exact (allele1>allele2 == REF>ALT), swapped,
allele mismatch, absent or not an SNV. Analyses use exact matches unless stated.

Outputs (written to --out-dir, default: current directory)
---------------------------------------------------------
  avi_breadth_summary.txt               everything printed to the screen
  avi_breadth_run_manifest.json         versions, inputs (size/mtime/md5), settings
  avi_breadth_match_report.csv          lookup status by category
  avi_breadth_variants.csv.gz           per-variant table
  avi_breadth_groups.csv                A: mean/CI per category
  avi_breadth_model.csv                 A: OLS + Welch contrasts
  avi_breadth_cutoffs.csv               B: selection at PHRED cutoffs
  avi_breadth_auroc.csv                 C: AUROC per contrast
  avi_breadth_adjusted.csv              D: stratified, spline and MH estimates
  avi_breadth_by_accessibility.csv      D: mean PHRED by category x accessibility bin
  avi_breadth_matched_bins.csv          D: PHRED by category within active-magnitude bins
  avi_breadth_ladder.csv                E: every quantity at every threshold
  avi_breadth_sensitivity.csv           F: per subset
  avi_phred_by_category.svg             main figure (matches the ChromBPNet figure)
  avi_raw_by_category.svg
  chrombpnet_vs_avi_by_category.svg     side by side, on the same variants
  avi_phred_by_matched_magnitude.svg    D, as a figure
  avi_breadth_ladder.svg                E, as a figure

Usage
-----
    python avi_breadth_replication.py --pipe-dir results/variant_analysis \\
        --avi data/external/alphagenome_variant_impact_score_snvs.tsv.gz \\
        --out-dir results/avi

Requirements: python >= 3.9, numpy, pandas, pyarrow, scipy, statsmodels (+ patsy),
matplotlib; plus either the `tabix` binary (htslib) or pysam.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ──────────────────────────────────────────────────────────────────────────────
# Settings, matched to the ChromBPNet pipeline
# ──────────────────────────────────────────────────────────────────────────────
CELL_TYPES = ["HSC", "MPP", "LMPP", "CMP", "GMP", "MEP", "CLP",
              "Ery", "Mega", "Mono", "Bcell", "CD4T", "CD8T", "NK",
              "CD34_BM", "CD34_CB", "K562"]
N_CT = len(CELL_TYPES)
BREADTH_ORDER = ["null", "specific", "multiple", "shared"]
ACTIVE = ["specific", "multiple", "shared"]
BREADTH_BOUNDS = {"null": (0, 0), "specific": (1, 1),
                  "multiple": (2, 13), "shared": (14, N_CT)}
BREADTH_COLOURS = {"null": "#BDBDBD", "specific": "#9ECAE1",
                   "multiple": "#4292C6", "shared": "#08519C"}
CONTRASTS = [("specific - null", 0, 1), ("multiple - specific", 1, 2),
             ("shared - multiple", 2, 3)]
EXPECTED_N = {"null": 228_699, "specific": 7_743, "multiple": 5_788, "shared": 77}

PRIMARY_RULE = ("raw", 0.01)
LADDER = [("raw", 0.01), ("raw", 0.005), ("raw", 0.001),
          ("fdr", 0.10), ("fdr", 0.05), ("fdr", 0.01)]
MIN_LADDER_SPECIFIC = 50              # skip a ladder level with fewer specific variants

PHRED_CUTOFFS = [10, 20, 30]          # genome-wide top 10% / 1% / 0.1% of SNVs
N_STRATA_MAG = 20                     # magnitude bins within common support
MIN_PER_STRATUM = 2                   # variants of EACH group a stratum must hold
MIN_GROUP_USED = 10                   # below this a matched contrast is not estimable
ACC_EDGES = [0, 4, 8, 12, 16, N_CT]   # accessibility-breadth bins (lo, hi]
SPLINE_DF = 6
N_PLOT_BINS = 8                       # active-magnitude bins in the matched figure
MIN_PLOT_N = 10
PROMOTER_BP = 1_000
ADJ_MAG = "magnitude"
ADJ_MAG_ACC = "magnitude x accessibility breadth"
PAIRS = [("specific", "null"), ("multiple", "specific"),
         ("shared", "multiple"), ("shared", "specific")]

# AlphaGenome accessibility coverage (DNase-seq or ATAC-seq biosamples in the AlphaGenome
# track list) for each ChromBPNet cell type. AVI's AlphaGenome features are maxima
# across tracks, so a variant whose activity is confined to cell types without a
# matching track can only be scored through other biosamples.
#   direct      a primary cell or the same cell line has a DNase/ATAC track
#   CD34+ bulk  matched only by bulk CD34+ samples (AlphaGenome labels them
#               "hematopoietic multipotent progenitor cell" and "common myeloid
#               progenitor, CD34-positive")
#   none        no matching accessibility track (erythroblast has TF ChIP only;
#               megakaryocytes only through the CMK cell line)
COVERAGE_TIER = {
    "K562": "direct", "Mono": "direct", "Bcell": "direct", "CD4T": "direct",
    "CD8T": "direct", "NK": "direct",
    "CD34_BM": "CD34+ bulk", "CD34_CB": "CD34+ bulk", "HSC": "CD34+ bulk",
    "MPP": "CD34+ bulk", "CMP": "CD34+ bulk",
    "LMPP": "none", "GMP": "none", "MEP": "none", "CLP": "none",
    "Ery": "none", "Mega": "none",
}
TIER_ORDER = ["direct", "CD34+ bulk", "none"]
TIER_COLOURS = {"direct": "#08519C", "CD34+ bulk": "#6BAED6", "none": "#BDBDBD"}
COVERAGE_REF = "K562"

CODING_TERMS = {
    "transcript_ablation", "splice_acceptor_variant", "splice_donor_variant",
    "stop_gained", "frameshift_variant", "stop_lost", "start_lost",
    "inframe_insertion", "inframe_deletion", "missense_variant",
    "protein_altering_variant", "incomplete_terminal_codon_variant",
    "start_retained_variant", "stop_retained_variant", "synonymous_variant",
    "coding_sequence_variant",
}

LOG: list[str] = []


def say(*parts) -> None:
    s = " ".join(str(p) for p in parts)
    print(s, flush=True)
    LOG.append(s)


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "nan"
    return "< 1e-300" if p == 0.0 else f"{p:.3g}"


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "svg.fonttype": "none",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def rule_label(rule: str, level: float) -> str:
    return f"P < {level:g}" if rule == "raw" else f"FDR < {level:g}"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Breadth and regulatory magnitude from the ChromBPNet pipeline
# ──────────────────────────────────────────────────────────────────────────────
def categorise(breadth: np.ndarray) -> np.ndarray:
    lab = np.full(breadth.shape[0], "", dtype=object)
    for name, (lo, hi) in BREADTH_BOUNDS.items():
        lab[(breadth >= lo) & (breadth <= hi)] = name
    return lab


def load_long(pipe: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq
    path = pipe / "variant_celltype.parquet"
    have = set(pq.ParquetFile(path).schema.names)
    cols = [c for c in ("variant_idx", "cell_type", "logfc", "abs_logfc", "called",
                        "accessible", "pval", "qval") if c in have]
    long = pd.read_parquet(path, columns=cols)
    long["cell_type"] = long["cell_type"].astype(str)
    long = long[long["cell_type"].isin(CELL_TYPES)].reset_index(drop=True)
    found = set(long["cell_type"].unique())
    if len(found) != N_CT:
        say(f"WARNING: {N_CT - len(found)} cell types missing from the parquet: "
            f"{sorted(set(CELL_TYPES) - found)}")
    long["accessible"] = long["accessible"].astype("boolean").fillna(False).astype(bool)
    long["called"] = long["called"].astype("boolean").fillna(False).astype(bool)
    return long


def per_variant(long: pd.DataFrame) -> pd.DataFrame:
    # The pipeline's magnitude statistic is abs_logfc (|logFC| averaged over the five
    # folds), which is not the same as |fold-averaged logFC|. Use it when present.
    if "abs_logfc" in long.columns:
        stat = long["abs_logfc"].to_numpy(np.float64)
        say("regulatory magnitude: max abs_logfc (fold-averaged |logFC|) over "
            "accessible cell types")
    else:
        stat = np.abs(long["logfc"].to_numpy(np.float64))
        say("WARNING: abs_logfc absent; regulatory magnitude uses |logfc|, which "
            "differs from the pipeline's statistic")
    absfc = np.where(long["accessible"], stat, np.nan)
    per = (pd.DataFrame({"variant_idx": long["variant_idx"].to_numpy(),
                         "called": long["called"].to_numpy(),
                         "acc": long["accessible"].to_numpy(), "absfc": absfc})
             .groupby("variant_idx")
             .agg(breadth=("called", "sum"), n_accessible=("acc", "sum"),
                  magnitude=("absfc", "max")))
    per["breadth"] = per["breadth"].astype(int)
    per["n_accessible"] = per["n_accessible"].astype(int)
    per["category"] = categorise(per["breadth"].to_numpy())
    return per.reset_index()


def ladder_breadth(long: pd.DataFrame, rule: str, level: float) -> pd.Series | None:
    col = "pval" if rule == "raw" else "qval"
    if col not in long.columns:
        return None
    hit = long["accessible"] & (long[col].to_numpy(np.float64) < level)
    return hit.groupby(long["variant_idx"]).sum().astype(int)


def chance_specific(n_acc: np.ndarray, alpha: float) -> float:
    """Expected breadth-1 variants if every call were a chance call."""
    n = n_acc[n_acc > 0].astype(float)
    return float((n * alpha * (1 - alpha) ** (n - 1)).sum())


def chrombpnet_sanity(per: pd.DataFrame, long: pd.DataFrame, pipe: Path) -> None:
    say("\n== ChromBPNet categories, recomputed ==")
    kept = per[per["n_accessible"] > 0]
    say(f"variants in variant_celltype.parquet: {len(per):,}; "
        f"accessible in >= 1 cell type (the analysed set): {len(kept):,}")
    ref = None
    gpath = pipe / "tables" / "breadth_magnitude_groups.csv"
    if gpath.exists():
        ref = (pd.read_csv(gpath, keep_default_na=False, na_values=[""])
                 .set_index("category"))
    for c in BREADTH_ORDER:
        v = kept.loc[kept["category"] == c, "magnitude"].dropna()
        line = (f"  {c:9s} n = {len(v):>8,} (expected {EXPECTED_N[c]:>8,})"
                f"   mean max|logFC| = {v.mean():.4f}")
        if ref is not None and c in ref.index:
            line += f"   pipeline mean = {float(ref.loc[c, 'mean_magnitude']):.4f}"
        say(line)

    if "pval" in long.columns:
        recomputed = long["accessible"] & (long["pval"].to_numpy(np.float64)
                                           < PRIMARY_RULE[1])
        n_dis = int((recomputed != long["called"]).sum())
        say(f"  `called` vs accessible & pval < {PRIMARY_RULE[1]:g}: "
            f"{n_dis:,} disagreeing variant x cell-type pairs")
    acc_pairs = int(long["accessible"].sum())
    called_pairs = int(long["called"].sum())
    rate = (long.groupby("cell_type")[["called", "accessible"]].sum()
                .assign(rate=lambda t: t["called"] / t["accessible"]))
    say(f"  accessible pairs = {acc_pairs:,}; called pairs = {called_pairs:,} "
        f"({called_pairs / acc_pairs:.2%}); per-cell-type call rate "
        f"{rate['rate'].min():.2%} - {rate['rate'].max():.2%}")
    exp = chance_specific(kept["n_accessible"].to_numpy(), PRIMARY_RULE[1])
    obs = int((kept["category"] == "specific").sum())
    say(f"  breadth-1 variants expected from chance calls alone at "
        f"P < {PRIMARY_RULE[1]:g}: {exp:,.0f} (observed specific: {obs:,}; "
        f"upper bound on the chance fraction: {min(exp / obs, 1):.1%})")


# ──────────────────────────────────────────────────────────────────────────────
# 2. AVI lookup
# ──────────────────────────────────────────────────────────────────────────────
def load_variants(pipe: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq
    path = pipe / "variants.parquet"
    have = set(pq.ParquetFile(path).schema.names)
    want = ["variant_idx", "chr", "pos", "allele1", "allele2", "rsid",
            "tss_distance_pc", "nearest_gene_pc"]
    v = pd.read_parquet(path, columns=[c for c in want if c in have])
    v["chr"] = v["chr"].astype(str)
    v.loc[~v["chr"].str.startswith("chr"), "chr"] = "chr" + v["chr"]
    v["pos"] = v["pos"].astype(np.int64)
    v["allele1"] = v["allele1"].astype(str).str.upper()
    v["allele2"] = v["allele2"].astype(str).str.upper()
    v["variant_key"] = (v["chr"] + ":" + v["pos"].astype(str) + ":"
                        + v["allele1"] + ":" + v["allele2"])
    return v


def fetch_avi(regions: pd.DataFrame, avi: Path) -> pd.DataFrame:
    cols = ["chr", "pos", "REF", "ALT", "avi_raw", "avi_phred"]
    regions = regions.drop_duplicates().sort_values(["chr", "pos"])
    tabix = shutil.which("tabix")
    if tabix:
        say(f"querying {len(regions):,} sites with {tabix} -R ...")
        with tempfile.TemporaryDirectory() as tmp:
            reg, hits = Path(tmp) / "regions.tsv", Path(tmp) / "hits.tsv"
            regions.to_csv(reg, sep="\t", header=False, index=False)
            with open(hits, "w") as fh:
                res = subprocess.run([tabix, "-R", str(reg), str(avi)],
                                     stdout=fh, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                raise SystemExit(f"tabix failed:\n{res.stderr}")
            if hits.stat().st_size == 0:
                return pd.DataFrame(columns=cols)
            out = pd.read_csv(hits, sep="\t", header=None, names=cols,
                              usecols=range(6),
                              dtype={"chr": str, "REF": str, "ALT": str})
    else:
        try:
            import pysam
        except ImportError:
            raise SystemExit("neither the `tabix` binary nor pysam is available; "
                             "install one (e.g. `conda install -c bioconda htslib`)")
        say(f"querying {len(regions):,} sites with pysam ...")
        tb = pysam.TabixFile(str(avi))
        contigs = set(tb.contigs)
        rows = []
        for chrom, pos in regions.itertuples(index=False):
            if chrom in contigs:
                rows.extend(line.split("\t")[:6]
                            for line in tb.fetch(chrom, int(pos) - 1, int(pos)))
        out = pd.DataFrame(rows, columns=cols)
        out["pos"] = out["pos"].astype(np.int64)
        out["avi_raw"] = out["avi_raw"].astype(float)
        out["avi_phred"] = out["avi_phred"].astype(float)
    return out.drop_duplicates(["chr", "pos", "REF", "ALT"])


def match_avi(v: pd.DataFrame, avi: pd.DataFrame) -> pd.DataFrame:
    a = avi.rename(columns={"REF": "allele1", "ALT": "allele2"})
    exact = v.merge(a, on=["chr", "pos", "allele1", "allele2"], how="left")
    s = avi.rename(columns={"REF": "allele2", "ALT": "allele1",
                            "avi_raw": "avi_raw_swapped",
                            "avi_phred": "avi_phred_swapped"})
    out = exact.merge(s, on=["chr", "pos", "allele1", "allele2"], how="left")
    sites = set(zip(avi["chr"], avi["pos"]))
    at_site = np.array([(c, p) in sites for c, p in zip(out["chr"], out["pos"])])
    is_snv = (out["allele1"].str.len() == 1) & (out["allele2"].str.len() == 1)
    status = np.full(len(out), "absent", dtype=object)
    status[at_site] = "allele_mismatch"
    status[out["avi_phred_swapped"].notna().to_numpy()] = "swapped"
    status[out["avi_phred"].notna().to_numpy()] = "exact"
    status[~is_snv.to_numpy()] = "not_snv"
    out["avi_match"] = status
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 3. Statistics
# ──────────────────────────────────────────────────────────────────────────────
def group_means(y: np.ndarray, lab: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in BREADTH_ORDER:
        lo, hi = BREADTH_BOUNDS[c]
        v = y[lab == c]
        v = v[np.isfinite(v)]
        n = int(v.size)
        mean = float(v.mean()) if n else np.nan
        sd = float(v.std(ddof=1)) if n > 1 else np.nan
        sem = sd / np.sqrt(n) if n > 1 else np.nan
        rows.append(dict(category=c, breadth_min=lo, breadth_max=hi, n_variants=n,
                         mean=mean, median=float(np.median(v)) if n else np.nan,
                         sd=sd, sem=sem,
                         ci_lo=mean - 1.96 * sem if n > 1 else np.nan,
                         ci_hi=mean + 1.96 * sem if n > 1 else np.nan))
    return pd.DataFrame(rows)


def category_model(y: np.ndarray, lab: np.ndarray) -> pd.DataFrame:
    """OLS (default SEs) with null as reference, plus Welch t-tests between the
    active categories -- exactly as category_model() in 03_breadth.py."""
    import statsmodels.api as sm
    ok = np.isfinite(y) & (lab != "")
    y, lab = y[ok], lab[ok]
    levels = [k for k in ACTIVE if (lab == k).sum() >= 2]
    rows = []
    if (lab == "null").sum() >= 2 and levels:
        X = np.column_stack([np.ones(y.size)]
                            + [(lab == k).astype(float) for k in levels])
        fit = sm.OLS(y, X).fit()
        ci = np.asarray(fit.conf_int())
        names = ["null (reference mean)"] + [f"{k} - null" for k in levels]
        for i, nm in enumerate(names):
            rows.append(dict(term=nm, estimate=float(fit.params[i]),
                             se=float(fit.bse[i]), t=float(fit.tvalues[i]),
                             p=float(fit.pvalues[i]),
                             p_underflow=bool(fit.pvalues[i] == 0.0),
                             ci_lo=float(ci[i, 0]), ci_hi=float(ci[i, 1]),
                             n=int(y.size), model="OLS score ~ category"))
    for a, b in (("multiple", "specific"), ("shared", "specific"),
                 ("shared", "multiple")):
        va, vb = y[lab == a], y[lab == b]
        if va.size >= 2 and vb.size >= 2:
            t = stats.ttest_ind(va, vb, equal_var=False)
            d = float(va.mean() - vb.mean())
            se = float(np.sqrt(va.var(ddof=1) / va.size + vb.var(ddof=1) / vb.size))
            rows.append(dict(term=f"{a} - {b}", estimate=d, se=se,
                             t=float(t.statistic), p=float(t.pvalue),
                             p_underflow=bool(t.pvalue == 0.0),
                             ci_lo=d - 1.96 * se, ci_hi=d + 1.96 * se,
                             n=int(va.size + vb.size), model="Welch t-test"))
    return pd.DataFrame(rows)


def auroc(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float, float]:
    """AUROC = P(score_pos > score_neg), ties counted half; Hanley-McNeil SE."""
    n1, n0 = len(pos), len(neg)
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan, np.nan
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    auc = float(u.statistic) / (n1 * n0)
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2)
                  + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))
    return auc, float(se), float(u.pvalue)


def auroc_table(y: np.ndarray, lab: np.ndarray) -> pd.DataFrame:
    rows = []
    for a, b in (("specific", "null"), ("multiple", "null"), ("shared", "null"),
                 ("multiple", "specific"), ("shared", "specific"),
                 ("shared", "multiple")):
        pa, pb = y[lab == a], y[lab == b]
        auc, se, p = auroc(pa[np.isfinite(pa)], pb[np.isfinite(pb)])
        rows.append(dict(contrast=f"{a} vs {b}", n_pos=len(pa), n_neg=len(pb),
                         auroc=auc, se=se, ci_lo=auc - 1.96 * se,
                         ci_hi=auc + 1.96 * se, p_mann_whitney=p))
    return pd.DataFrame(rows)


def wilson(k: int, n: int) -> tuple[float, float]:
    from statsmodels.stats.proportion import proportion_confint
    if n == 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return float(max(lo, 0.0)), float(min(hi, 1.0))


def cutoff_table(phred: np.ndarray, lab: np.ndarray) -> pd.DataFrame:
    rows = []
    n_active = int(np.isin(lab, ACTIVE).sum())
    for t in PHRED_CUTOFFS:
        passed = phred >= t
        n_sel_active = int((passed & np.isin(lab, ACTIVE)).sum())
        null_pass = int((passed & (lab == "null")).sum())
        null_n = int((lab == "null").sum())
        for c in BREADTH_ORDER:
            m = lab == c
            n, k = int(m.sum()), int((passed & m).sum())
            lo, hi = wilson(k, n)
            row = dict(phred_cutoff=t, category=c, n_variants=n, n_pass=k,
                       frac_pass=k / n if n else np.nan, frac_pass_ci_lo=lo,
                       frac_pass_ci_hi=hi)
            if c != "null":
                row["share_of_active"] = n / n_active if n_active else np.nan
                row["share_of_selected_active"] = (k / n_sel_active
                                                   if n_sel_active else np.nan)
                if n:
                    odds, p = stats.fisher_exact([[k, n - k],
                                                  [null_pass, null_n - null_pass]])
                    row.update(odds_ratio_vs_null=odds, fisher_p_vs_null=p)
            rows.append(row)
    return pd.DataFrame(rows)


# ── D. common-support stratified adjustment ──────────────────────────────────
def prepare_pair(d: pd.DataFrame, a: str, b: str, adjust: str,
                 y: str = "avi_phred"):
    """Restrict to groups a and b, trim to the common support of magnitude, build
    strata and keep those with >= MIN_PER_STRATUM variants of each group."""
    info = dict(contrast=f"{a} - {b}", adjustment=adjust,
                n_a_total=int((d["category"] == a).sum()),
                n_b_total=int((d["category"] == b).sum()))
    sub = d.loc[d["category"].isin([a, b]),
                ["category", "magnitude", "n_accessible", y]].dropna()
    if info["n_a_total"] < 2 or info["n_b_total"] < 2:
        return None, None, None, info
    ma = sub.loc[sub["category"] == a, "magnitude"]
    mb = sub.loc[sub["category"] == b, "magnitude"]
    lo, hi = max(ma.min(), mb.min()), min(ma.max(), mb.max())
    info.update(support_lo=float(lo), support_hi=float(hi))
    sub = sub[(sub["magnitude"] >= lo) & (sub["magnitude"] <= hi)]
    if len(sub) < 2 * MIN_PER_STRATUM:
        return None, None, None, info
    k = int(min(N_STRATA_MAG, max(1, len(sub) // 20)))
    mag_bin = pd.qcut(sub["magnitude"].rank(method="first"), k, labels=False)
    cell = mag_bin.astype(int).astype(str)
    if adjust == ADJ_MAG_ACC:
        acc_bin = pd.cut(sub["n_accessible"], ACC_EDGES, labels=False)
        cell = cell + "_" + acc_bin.astype(int).astype(str)
    is_a = sub["category"] == a
    counts = pd.crosstab(cell, is_a)
    for col in (True, False):
        if col not in counts.columns:
            counts[col] = 0
    good = counts.index[(counts[True] >= MIN_PER_STRATUM)
                        & (counts[False] >= MIN_PER_STRATUM)]
    keep = cell.isin(good)
    sub, cell, is_a = sub[keep], cell[keep], is_a[keep]
    info.update(n_a_used=int(is_a.sum()), n_b_used=int((~is_a).sum()),
                n_strata=int(len(good)))
    if info["n_a_used"] < MIN_GROUP_USED or info["n_b_used"] < MIN_GROUP_USED:
        info["not_estimable"] = (f"fewer than {MIN_GROUP_USED} variants of a group "
                                 f"in matched strata")
        return None, None, None, info
    return sub, cell, is_a, info


def stratified_contrast(d: pd.DataFrame, a: str, b: str, adjust: str,
                        y: str = "avi_phred") -> dict:
    import statsmodels.api as sm
    sub, cell, is_a, info = prepare_pair(d, a, b, adjust, y)
    info.update(method="stratified OLS, HC3", estimate=np.nan, se=np.nan,
                ci_lo=np.nan, ci_hi=np.nan, p=np.nan)
    if sub is None:
        return info
    D = pd.get_dummies(cell, drop_first=True, dtype=float).to_numpy()
    X = np.column_stack([np.ones(len(sub)), is_a.to_numpy(float), D])
    yv = sub[y].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.OLS(yv, X).fit(cov_type="HC3")
        if not np.isfinite(fit.bse[1]):
            fit = sm.OLS(yv, X).fit(cov_type="HC1")
            info["method"] = "stratified OLS, HC1 (HC3 undefined)"
    ci = np.asarray(fit.conf_int())[1]
    info.update(estimate=float(fit.params[1]), se=float(fit.bse[1]),
                ci_lo=float(ci[0]), ci_hi=float(ci[1]), p=float(fit.pvalues[1]))
    return info


def mh_cutoff(d: pd.DataFrame, a: str, b: str, t: float, adjust: str) -> dict:
    from statsmodels.stats.contingency_tables import StratifiedTable
    sub, cell, is_a, info = prepare_pair(d, a, b, adjust)
    info.update(method=f"Mantel-Haenszel, PHRED >= {t:g}", estimate=np.nan,
                ci_lo=np.nan, ci_hi=np.nan, p=np.nan)
    if sub is None:
        return info
    passed = sub["avi_phred"].to_numpy() >= t
    ia = is_a.to_numpy()
    tables = []
    for c in pd.unique(cell):
        m = (cell == c).to_numpy()
        tab = np.array([[np.sum(m & ia & passed), np.sum(m & ia & ~passed)],
                        [np.sum(m & ~ia & passed), np.sum(m & ~ia & ~passed)]],
                       dtype=float)
        if tab[:, 0].sum() > 0 and tab[:, 1].sum() > 0:
            tables.append(tab)
    info["n_informative_strata"] = len(tables)
    if not tables:
        return info
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = StratifiedTable(tables)
        lo, hi = st.oddsratio_pooled_confint()
        info.update(estimate=float(st.oddsratio_pooled), ci_lo=float(lo),
                    ci_hi=float(hi), p=float(st.test_null_odds().pvalue))
    return info


def spline_model(d: pd.DataFrame) -> list[dict]:
    """PHRED ~ category + B-spline(log10 magnitude) + C(n_accessible), active only."""
    import statsmodels.formula.api as smf
    sub = d[d["category"].isin(ACTIVE)].copy()
    levels = [c for c in ACTIVE if (sub["category"] == c).sum() >= 2]
    if len(levels) < 2:
        return []
    sub = sub[sub["category"].isin(levels)]
    sub["logmag"] = np.log10(sub["magnitude"].clip(lower=1e-6))
    sub["cat"] = pd.Categorical(sub["category"], categories=levels)
    formula = f"avi_phred ~ cat + bs(logmag, df={SPLINE_DF}) + C(n_accessible)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols(formula, data=sub).fit(cov_type="HC3")
        method = f"OLS {formula}, HC3"
        if not np.all(np.isfinite(fit.bse)):
            fit = smf.ols(formula, data=sub).fit(cov_type="HC1")
            method = f"OLS {formula}, HC1 (HC3 undefined)"
    names = list(fit.params.index)

    def coef(c):
        return names.index(f"cat[T.{c}]") if c != levels[0] else None

    rows = []
    for a, b in (("multiple", "specific"), ("shared", "specific"),
                 ("shared", "multiple")):
        if a not in levels or b not in levels:
            continue
        R = np.zeros(len(names))
        if coef(a) is not None:
            R[coef(a)] += 1
        if coef(b) is not None:
            R[coef(b)] -= 1
        tt = fit.t_test(R)
        est, se = float(np.ravel(tt.effect)[0]), float(np.ravel(tt.sd)[0])
        rows.append(dict(contrast=f"{a} - {b}",
                         adjustment="spline(log magnitude) + accessibility breadth",
                         method=method, estimate=est, se=se,
                         ci_lo=est - 1.96 * se, ci_hi=est + 1.96 * se,
                         p=float(np.ravel(tt.pvalue)[0]),
                         n_a_used=int((sub["category"] == a).sum()),
                         n_b_used=int((sub["category"] == b).sum())))
    return rows


def accessibility_table(d: pd.DataFrame) -> pd.DataFrame:
    acc_bin = pd.cut(d["n_accessible"], ACC_EDGES)
    rows = []
    for (c, b), g in d.groupby([d["category"], acc_bin], observed=True):
        v = g["avi_phred"].to_numpy(float)
        sem = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
        rows.append(dict(category=c, accessible_in=f"{int(b.left) + 1}-{int(b.right)}",
                         n_variants=v.size, mean_phred=v.mean(),
                         ci_lo=v.mean() - 1.96 * sem, ci_hi=v.mean() + 1.96 * sem,
                         median_magnitude=float(g["magnitude"].median())))
    out = pd.DataFrame(rows)
    out["category"] = pd.Categorical(out["category"], BREADTH_ORDER)
    return out.sort_values(["category", "accessible_in"]).reset_index(drop=True)


def matched_bins(d: pd.DataFrame) -> pd.DataFrame:
    """Bins of equal count over ACTIVE variants' magnitude; every category, null
    included, is summarised within each bin's magnitude range."""
    act = d.loc[d["category"].isin(ACTIVE), "magnitude"].to_numpy()
    edges = np.quantile(act, np.linspace(0, 1, N_PLOT_BINS + 1))
    edges[0], edges[-1] = edges[0] - 1e-12, edges[-1] + 1e-12
    b = pd.cut(d["magnitude"], edges, labels=False)
    rows = []
    for i in range(N_PLOT_BINS):
        inb = (b == i).to_numpy()
        for c in BREADTH_ORDER:
            v = d.loc[inb & (d["category"] == c).to_numpy(), "avi_phred"].to_numpy(float)
            n = v.size
            sem = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            rows.append(dict(bin=i + 1, magnitude_lo=float(edges[i]),
                             magnitude_hi=float(edges[i + 1]), category=c,
                             n_variants=int(n),
                             mean_phred=float(v.mean()) if n else np.nan,
                             ci_lo=float(v.mean() - 1.96 * sem) if n > 1 else np.nan,
                             ci_hi=float(v.mean() + 1.96 * sem) if n > 1 else np.nan))
    return pd.DataFrame(rows)


# ── E. threshold ladder ──────────────────────────────────────────────────────
def run_ladder(m: pd.DataFrame, long: pd.DataFrame,
               n_acc_all: np.ndarray) -> pd.DataFrame:
    rows = []
    say("\n== E. Threshold ladder (breadth recomputed at each calling rule) ==")
    for rule, level in LADDER:
        br = ladder_breadth(long, rule, level)
        lab_name = rule_label(rule, level)
        if br is None:
            say(f"  {lab_name}: column {'pval' if rule == 'raw' else 'qval'} "
                f"absent, skipped")
            continue
        d = m.copy()
        d["breadth"] = br.reindex(d["variant_idx"]).fillna(0).astype(int).to_numpy()
        d["category"] = categorise(d["breadth"].to_numpy())
        lab = d["category"].to_numpy(object)
        n_spec = int((lab == "specific").sum())
        if n_spec < MIN_LADDER_SPECIFIC:
            say(f"  {lab_name}: only {n_spec} specific variants, skipped")
            continue
        base = dict(rule=rule, level=level, rule_label=lab_name)

        g = group_means(d["avi_phred"].to_numpy(float), lab)
        for r in g.itertuples():
            rows.append(dict(base, quantity="mean_phred", term=r.category,
                             n=r.n_variants, estimate=r.mean, ci_lo=r.ci_lo,
                             ci_hi=r.ci_hi, p=np.nan))
        mdl = category_model(d["avi_phred"].to_numpy(float), lab).set_index("term")
        for term in ("specific - null", "multiple - specific", "shared - multiple"):
            if term in mdl.index:
                r = mdl.loc[term]
                rows.append(dict(base, quantity="contrast_unadjusted", term=term,
                                 n=r["n"], estimate=r["estimate"], ci_lo=r["ci_lo"],
                                 ci_hi=r["ci_hi"], p=r["p"]))
        for a, b in PAIRS[:3]:
            r = stratified_contrast(d, a, b, ADJ_MAG_ACC)
            rows.append(dict(base, quantity="contrast_adjusted", term=f"{a} - {b}",
                             n=r.get("n_a_used", 0) + r.get("n_b_used", 0),
                             estimate=r["estimate"], ci_lo=r["ci_lo"],
                             ci_hi=r["ci_hi"], p=r["p"]))
        for t in (10, 20):
            passed = d["avi_phred"].to_numpy() >= t
            sel_act = passed & np.isin(lab, ACTIVE)
            for c in ACTIVE:
                mc = lab == c
                k, n = int((passed & mc).sum()), int(mc.sum())
                lo, hi = wilson(k, n)
                rows.append(dict(base, quantity=f"frac_pass_phred{t}", term=c, n=n,
                                 estimate=k / n if n else np.nan, ci_lo=lo, ci_hi=hi,
                                 p=np.nan))
            k = int((sel_act & (lab == "specific")).sum())
            lo, hi = wilson(k, int(sel_act.sum()))
            rows.append(dict(base, quantity=f"specific_share_of_selected_phred{t}",
                             term="specific", n=int(sel_act.sum()),
                             estimate=k / sel_act.sum() if sel_act.sum() else np.nan,
                             ci_lo=lo, ci_hi=hi, p=np.nan))
        n_act = int(np.isin(lab, ACTIVE).sum())
        rows.append(dict(base, quantity="specific_share_of_active", term="specific",
                         n=n_act, estimate=n_spec / n_act, ci_lo=np.nan,
                         ci_hi=np.nan, p=np.nan))
        for a, b in (("specific", "null"), ("multiple", "specific")):
            pa = d.loc[d["category"] == a, "avi_phred"].to_numpy(float)
            pb = d.loc[d["category"] == b, "avi_phred"].to_numpy(float)
            auc, se, p = auroc(pa, pb)
            rows.append(dict(base, quantity="auroc", term=f"{a} vs {b}",
                             n=len(pa) + len(pb), estimate=auc, ci_lo=auc - 1.96 * se,
                             ci_hi=auc + 1.96 * se, p=p))
        if rule == "raw":
            exp = chance_specific(n_acc_all, level)
            n_spec_all = int((ladder_breadth(long, rule, level)
                              .reindex(m["variant_idx"]) == 1).sum())
            rows.append(dict(base, quantity="expected_chance_specific",
                             term="specific", n=n_spec_all, estimate=exp,
                             ci_lo=np.nan, ci_hi=np.nan, p=np.nan))

        gi = g.set_index("category")
        adj = {r["term"]: r for r in rows if r.get("rule_label") == lab_name
               and r["quantity"] == "contrast_adjusted"}
        q = {(r["quantity"], r["term"]): r["estimate"] for r in rows
             if r.get("rule_label") == lab_name}
        pass_txt = ("; pass PHRED>=20 spec/mult "
                    f"{q[('frac_pass_phred20', 'specific')]:.1%}/"
                    f"{q[('frac_pass_phred20', 'multiple')]:.1%}; specific share "
                    f"{q[('specific_share_of_active', 'specific')]:.0%} of active -> "
                    f"{q[('specific_share_of_selected_phred20', 'specific')]:.0%} "
                    f"of passing")
        exp_txt = ""
        if rule == "raw":
            exp_txt = (f"; chance-specific bound "
                       f"{min(exp / max(n_spec, 1), 1):.0%}")
        say(f"  {lab_name:12s} n spec/mult/shared = {n_spec:,}/"
            f"{int(gi.loc['multiple', 'n_variants']):,}/"
            f"{int(gi.loc['shared', 'n_variants']):,}; mean PHRED "
            + "/".join(f"{gi.loc[c, 'mean']:.2f}" for c in BREADTH_ORDER)
            + f"; adj. specific - null {adj['specific - null']['estimate']:+.2f} "
              f"[{adj['specific - null']['ci_lo']:+.2f}, "
              f"{adj['specific - null']['ci_hi']:+.2f}]"
            + f"; adj. multiple - specific "
              f"{adj['multiple - specific']['estimate']:+.2f} "
              f"[{adj['multiple - specific']['ci_lo']:+.2f}, "
              f"{adj['multiple - specific']['ci_hi']:+.2f}]" + pass_txt + exp_txt)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Figures
# ──────────────────────────────────────────────────────────────────────────────
def _fmt_p(row) -> str:
    underflow = str(row.get("p_underflow", "False")).strip().lower() == "true"
    p = float(row["p"])
    if underflow or p == 0.0:
        return "$P$ < 10$^{-300}$"
    e = int(np.floor(np.log10(p)))
    return f"$P$ = {p / 10 ** e:.1f} $\\times$ 10$^{{{e}}}$"


def _breadth_label(cat: str, lo: int, hi: int) -> str:
    if cat == "null":
        return "Inactive\nin all"
    if lo == hi:
        return f"{lo} cell\ntype"
    if hi >= N_CT:
        return f"$\\geq${lo} of\n{N_CT}"
    return f"{lo}–{hi} cell\ntypes"


def _clean(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_category_panel(ax, groups: pd.DataFrame, model: pd.DataFrame,
                        ylabel: str, title: str | None = None) -> None:
    g = groups.set_index("category").loc[BREADTH_ORDER]
    x = np.arange(len(BREADTH_ORDER))
    mean = g["mean"].to_numpy(float)
    lo, hi = g["ci_lo"].to_numpy(float), g["ci_hi"].to_numpy(float)
    ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor="0.25",
                elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=2)
    ax.scatter(x, mean, s=34, c=[BREADTH_COLOURS[c] for c in BREADTH_ORDER],
               edgecolor="0.15", linewidth=0.6, zorder=3, clip_on=False)
    ax.set_xticks(x)
    ax.set_xticklabels([_breadth_label(c, int(g.loc[c, "breadth_min"]),
                                       int(g.loc[c, "breadth_max"]))
                        + f"\n$n$ = {int(g.loc[c, 'n_variants']):,}"
                        for c in BREADTH_ORDER])
    ax.set_xlabel("Cell types in which the variant is active")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.55, len(BREADTH_ORDER) - 0.45)
    lo_f, hi_f = lo[np.isfinite(lo)], hi[np.isfinite(hi)]
    base = 0.0 if lo_f.min() >= 0 else float(lo_f.min())
    top = float(hi_f.max())
    span = top - base if top > base else 1.0
    mdl = model.set_index("term") if len(model) else None
    ax.set_ylim(base - (0 if base == 0 else 0.05 * span),
                top + span * (0.42 if mdl is not None else 0.12))
    if mdl is not None:
        step = span * 0.115
        for k, (term, i, j) in enumerate(CONTRASTS):
            if term not in mdl.index:
                continue
            yb = top + span * 0.03 + k * step
            ax.plot([i, i, j, j], [yb - step * 0.16, yb, yb, yb - step * 0.16],
                    lw=0.6, c="0.35", solid_joinstyle="miter", clip_on=False)
            ax.annotate(_fmt_p(mdl.loc[term]), ((i + j) / 2, yb), xytext=(0, 1.2),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=5.8, color="0.25")
    if title:
        ax.set_title(title)
    _clean(ax)
    ax.tick_params(axis="x", length=0)


def save_single(groups, model, ylabel, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.46, 3.10))
    draw_category_panel(ax, groups, model, ylabel)
    fig.savefig(path, format="svg")
    plt.close(fig)
    say(f"wrote {path}")


def save_side_by_side(g_cb, m_cb, g_avi, m_avi, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.10), gridspec_kw={"wspace": 0.42})
    draw_category_panel(axes[0], g_cb, m_cb,
                        "Mean variant-effect magnitude (max |logFC|)", "ChromBPNet")
    draw_category_panel(axes[1], g_avi, m_avi, "Mean AVI score (PHRED)",
                        "AlphaGenome AVI")
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.22, 1.02, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom")
    fig.savefig(path, format="svg")
    plt.close(fig)
    say(f"wrote {path}")


def save_matched(tab: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    offsets = dict(zip(BREADTH_ORDER, (-0.21, -0.07, 0.07, 0.21)))
    for c in BREADTH_ORDER:
        t = tab[(tab["category"] == c) & (tab["n_variants"] >= MIN_PLOT_N)]
        if t.empty:
            continue
        x = t["bin"].to_numpy(float) + offsets[c]
        ax.errorbar(x, t["mean_phred"],
                    yerr=[t["mean_phred"] - t["ci_lo"], t["ci_hi"] - t["mean_phred"]],
                    fmt="o-", ms=3, lw=0.8, elinewidth=0.7, capsize=1.5,
                    color=BREADTH_COLOURS[c], mec="0.15", mew=0.4, label=c)
    edges = tab.drop_duplicates("bin").sort_values("bin")
    ax.set_xticks(edges["bin"])
    ax.set_xticklabels([f"{max(lo, 0):.3g}–{hi:.3g}" for lo, hi in
                        zip(edges["magnitude_lo"], edges["magnitude_hi"])],
                       rotation=35, ha="right", rotation_mode="anchor")
    ax.set_xlabel("ChromBPNet regulatory magnitude (max |logFC|), "
                  "equal-count bins of active variants")
    ax.set_ylabel("Mean AVI score (PHRED)")
    ax.legend(frameon=False, fontsize=6, title="breadth", title_fontsize=6,
              loc="upper left")
    ax.set_title(f"AVI by breadth at matched effect size "
                 f"(n < {MIN_PLOT_N} not drawn)")
    _clean(ax)
    fig.savefig(path, format="svg")
    plt.close(fig)
    say(f"wrote {path}")


def save_ladder(lad: pd.DataFrame, path: Path) -> None:
    if lad.empty:
        return
    order = list(dict.fromkeys(lad["rule_label"]))
    xi = {k: i for i, k in enumerate(order)}
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), gridspec_kw={"wspace": 0.45})

    ax = axes[0]
    for c in BREADTH_ORDER:
        t = lad[(lad["quantity"] == "mean_phred") & (lad["term"] == c)]
        ax.errorbar([xi[k] for k in t["rule_label"]], t["estimate"],
                    yerr=[t["estimate"] - t["ci_lo"], t["ci_hi"] - t["estimate"]],
                    fmt="o-", ms=3, lw=0.8, capsize=1.5, color=BREADTH_COLOURS[c],
                    mec="0.15", mew=0.4, label=c)
    ax.set_ylabel("Mean AVI score (PHRED)")
    ax.set_title("Category means", pad=24)
    ax.legend(frameon=False, fontsize=5.8, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 0.99), handletextpad=0.3, columnspacing=0.8,
              borderaxespad=0.1)

    ax = axes[1]
    for q, style, lab in (("contrast_unadjusted", "o--", "unadjusted"),
                          ("contrast_adjusted", "s-", "adjusted")):
        t = lad[(lad["quantity"] == q) & (lad["term"] == "multiple - specific")]
        ax.errorbar(np.array([xi[k] for k in t["rule_label"]])
                    + (-0.08 if q.endswith("unadjusted") else 0.08),
                    t["estimate"],
                    yerr=[t["estimate"] - t["ci_lo"], t["ci_hi"] - t["estimate"]],
                    fmt=style, ms=3, lw=0.8, capsize=1.5,
                    color="0.55" if q.endswith("unadjusted") else "#4292C6",
                    label=lab)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Multiple $-$ specific, PHRED")
    ax.set_title("Breadth contrast")
    ax.legend(frameon=False, fontsize=5.8)

    ax = axes[2]
    s = lad[lad["quantity"] == "specific_share_of_active"]
    ax.plot([xi[k] for k in s["rule_label"]], s["estimate"], "k:", lw=0.9,
            label="all active variants")
    for t_cut, colour, dx in ((10, "#6BAED6", -0.06), (20, "#08519C", 0.06)):
        t = lad[(lad["quantity"] == f"specific_share_of_selected_phred{t_cut}")
                & (lad["n"] > 0)]
        ax.errorbar(np.array([xi[k] for k in t["rule_label"]]) + dx, t["estimate"],
                    yerr=[t["estimate"] - t["ci_lo"], t["ci_hi"] - t["estimate"]],
                    fmt="o-", ms=3, lw=0.8, capsize=1.5, color=colour, mec="0.15",
                    mew=0.4, label=f"active, PHRED $\\geq$ {t_cut}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share that is cell-type-specific")
    ax.set_title("Who passes a cutoff")
    ax.legend(frameon=False, fontsize=5.8)

    for ax, letter in zip(axes, "abc"):
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=35, ha="right", rotation_mode="anchor")
        ax.text(-0.25, 1.03, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom")
        _clean(ax)
    fig.supxlabel("Calling rule used to define breadth (stricter to the right "
                  "within each rule type)", fontsize=7.5, y=-0.12)
    fig.savefig(path, format="svg")
    plt.close(fig)
    say(f"wrote {path}")


# ──────────────────────────────────────────────────────────────────────────────
# G. AlphaGenome biosample coverage
# ──────────────────────────────────────────────────────────────────────────────
def robust_ols(formula: str, data: pd.DataFrame, terms: list[str]):
    """OLS with HC3; HC1 if HC3 is undefined for a reported term (strata containing
    a single variant have leverage 1, where HC3 divides by zero)."""
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols(formula, data=data).fit(cov_type="HC3")
        bse = fit.bse.reindex([t for t in terms if t in fit.bse.index])
        if not np.all(np.isfinite(bse)):
            fit = smf.ols(formula, data=data).fit(cov_type="HC1")
    return fit


def coverage_counts(long: pd.DataFrame) -> pd.DataFrame:
    act = long.loc[long["called"], ["variant_idx", "cell_type"]].copy()
    act["tier"] = act["cell_type"].map(COVERAGE_TIER)
    cnt = (pd.crosstab(act["variant_idx"], act["tier"])
             .reindex(columns=TIER_ORDER, fill_value=0))
    cnt.columns = [f"n_active_{t.split('+')[0].split()[0].lower()}" for t in TIER_ORDER]
    first = act.groupby("variant_idx")["cell_type"].first().rename("active_ct")
    return cnt.join(first).reset_index()


def strata_cells(d: pd.DataFrame) -> pd.Series:
    k = int(min(N_STRATA_MAG, max(1, len(d) // 20)))
    mb = pd.qcut(d["magnitude"].rank(method="first"), k, labels=False).astype(int)
    ab = pd.cut(d["n_accessible"], ACC_EDGES, labels=False).astype(int)
    return mb.astype(str) + "_" + ab.astype(str)


def coverage_analysis(m: pd.DataFrame, long: pd.DataFrame
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    cov = coverage_counts(long)
    d = m.merge(cov, on="variant_idx", how="left")
    for c in ("n_active_direct", "n_active_cd34", "n_active_none"):
        d[c] = d[c].fillna(0).astype(int)

    say("\n== G. AlphaGenome biosample coverage ==")
    say("  accessibility coverage of each cell type: "
        + "; ".join(f"{t}: {', '.join(k for k, v in COVERAGE_TIER.items() if v == t)}"
                    for t in TIER_ORDER))

    # G1. single-cell-type variants, by the one cell type they are active in
    spec = d[d["category"] == "specific"].copy()
    spec["tier"] = spec["active_ct"].map(COVERAGE_TIER)
    spec["cell"] = strata_cells(spec)
    rows = []
    lev = f"C(active_ct, Treatment('{COVERAGE_REF}'))"
    fit = robust_ols(f"avi_phred ~ {lev} + C(cell)", spec,
                     [f"{lev}[T.{ct}]" for ct in CELL_TYPES])
    ci = fit.conf_int()
    for ct in CELL_TYPES:
        v = spec.loc[spec["active_ct"] == ct, "avi_phred"]
        term = f"C(active_ct, Treatment('{COVERAGE_REF}'))[T.{ct}]"
        est = 0.0 if ct == COVERAGE_REF else float(fit.params.get(term, np.nan))
        lo = np.nan if ct == COVERAGE_REF else float(ci.loc[term, 0]) if term in ci.index else np.nan
        hi = np.nan if ct == COVERAGE_REF else float(ci.loc[term, 1]) if term in ci.index else np.nan
        p = np.nan if ct == COVERAGE_REF else float(fit.pvalues.get(term, np.nan))
        rows.append(dict(analysis="specific variants by active cell type",
                         term=ct, tier=COVERAGE_TIER[ct], n=int(v.size),
                         mean_phred=float(v.mean()) if v.size else np.nan,
                         estimate=est, ci_lo=lo, ci_hi=hi, p=p,
                         note=f"adjusted difference vs {COVERAGE_REF}, matched on "
                              f"magnitude x accessibility, HC3"))
    per_ct = pd.DataFrame(rows)
    say(f"  single-cell-type variants: AVI PHRED by the cell type they act in "
        f"(adjusted difference vs {COVERAGE_REF}):")
    for t in TIER_ORDER:
        sub = per_ct[per_ct["tier"] == t]
        say(f"    {t:11s} " + "  ".join(
            f"{r.term} {r.mean_phred:.1f}/{r.estimate:+.2f}(n={r.n})"
            for r in sub.itertuples()))

    tier_rows = []
    d2 = spec.assign(category=spec["tier"])
    for a, b in (("CD34+ bulk", "direct"), ("none", "direct"), ("none", "CD34+ bulk")):
        r = stratified_contrast(d2, a, b, ADJ_MAG_ACC)
        r.update(analysis="specific variants, tier contrast")
        tier_rows.append(r)
        say(f"    tier {a} - {b}: {r['estimate']:+.2f} [{r['ci_lo']:+.2f}, "
            f"{r['ci_hi']:+.2f}] p = {fmt_p(r['p'])} (used {r.get('n_a_used', 0):,} vs "
            f"{r.get('n_b_used', 0):,})")

    # G2. does coverage explain the breadth gap?
    act = d[d["category"].isin(["specific", "multiple"])]
    groups = {
        "both classes active in >= 1 directly profiled cell type":
            act["n_active_direct"] > 0,
        "both classes active in no directly profiled cell type":
            act["n_active_direct"] == 0,
    }
    say("  multiple - specific within coverage strata (magnitude x accessibility, HC3):")
    for name, msk in groups.items():
        sub = act[msk.to_numpy()]
        r = stratified_contrast(sub, "multiple", "specific", ADJ_MAG_ACC)
        r.update(analysis=name)
        tier_rows.append(r)
        mh = mh_cutoff(sub, "multiple", "specific", 20, ADJ_MAG_ACC)
        mh.update(analysis=name)
        tier_rows.append(mh)
        n_s = int((sub["category"] == "specific").sum())
        n_m = int((sub["category"] == "multiple").sum())
        say(f"    {name}: n = {n_s:,} specific, {n_m:,} multiple; difference "
            f"{r['estimate']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] "
            f"p = {fmt_p(r['p'])}; OR(PHRED >= 20) = {mh['estimate']:.2f} "
            f"[{mh['ci_lo']:.2f}, {mh['ci_hi']:.2f}]")

    # G3. among multiple variants, does AVI rise with directly profiled cell types?
    mult = d[d["category"] == "multiple"].copy()
    if len(mult) > 50:
        mult["cell"] = strata_cells(mult)
        f3 = robust_ols("avi_phred ~ n_active_direct + breadth + C(cell)", mult,
                        ["n_active_direct", "breadth"])
        c3 = f3.conf_int()
        for term in ("n_active_direct", "breadth"):
            tier_rows.append(dict(analysis="multiple variants, per added active cell type",
                                  contrast=term, estimate=float(f3.params[term]),
                                  ci_lo=float(c3.loc[term, 0]), ci_hi=float(c3.loc[term, 1]),
                                  p=float(f3.pvalues[term]), n_a_used=int(len(mult)),
                                  method="OLS + magnitude x accessibility FE, HC3"))
        say(f"  multiple variants, per extra active cell type (same total breadth, "
            f"magnitude and accessibility): directly profiled "
            f"{f3.params['n_active_direct']:+.3f} [{c3.loc['n_active_direct', 0]:+.3f}, "
            f"{c3.loc['n_active_direct', 1]:+.3f}]; any "
            f"{f3.params['breadth']:+.3f} [{c3.loc['breadth', 0]:+.3f}, "
            f"{c3.loc['breadth', 1]:+.3f}]")
    return per_ct, pd.DataFrame(tier_rows)


def save_coverage(per_ct: pd.DataFrame, path: Path) -> None:
    t = per_ct.copy()
    t["order"] = t["tier"].map({k: i for i, k in enumerate(TIER_ORDER)})
    t = t.sort_values(["order", "estimate"], ascending=[True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    y = np.arange(len(t))[::-1]
    for yi, r in zip(y, t.itertuples()):
        if np.isfinite(r.ci_lo):
            ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color="0.35", lw=0.8)
        ax.scatter(r.estimate, yi, s=26, color=TIER_COLOURS[r.tier], edgecolor="0.15",
                   linewidth=0.5, zorder=3, marker="D" if r.term == COVERAGE_REF else "o")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.term} (n = {r.n:,})" for r in t.itertuples()], fontsize=6.3)
    ax.set_xlabel(f"AVI PHRED vs {COVERAGE_REF}, single-cell-type variants\n"
                  "(matched on magnitude and accessibility breadth)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=TIER_COLOURS[k],
                              markeredgecolor="0.15", label=k) for k in TIER_ORDER],
              title="AlphaGenome accessibility track", title_fontsize=6, fontsize=6,
              frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _clean(ax)
    fig.savefig(path, format="svg")
    plt.close(fig)
    say(f"wrote {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Sensitivity subsets
# ──────────────────────────────────────────────────────────────────────────────
def load_coding_keys(pipe: Path) -> set[str] | None:
    path = pipe / "vep" / "vep_most_severe.tsv"
    if not path.exists():
        say(f"WARNING: {path} not found; non-coding sensitivity check skipped")
        return None
    vep = pd.read_csv(path, sep="\t", dtype=str)
    vep.columns = [c.lstrip("#").strip() for c in vep.columns]
    key = "variant_key" if "variant_key" in vep.columns else vep.columns[0]
    cons = [c for c in vep.columns if "consequence" in c.lower()]
    if not cons:
        say(f"WARNING: no consequence column in {path} (columns: "
            f"{list(vep.columns)}); non-coding sensitivity check skipped")
        return None
    say(f"VEP: key column '{key}', consequence column '{cons[0]}'")
    terms = vep[cons[0]].fillna("").str.split(r"[,&]")
    coding = terms.apply(lambda ts: any(t.strip() in CODING_TERMS for t in ts))
    return set(vep.loc[coding, key])


def sensitivity(df: pd.DataFrame, coding: set[str] | None) -> pd.DataFrame:
    exact = df["avi_match"] == "exact"
    subsets = {"primary (exact matches)": exact,
               "exact + swapped alleles": df["avi_match"].isin(["exact", "swapped"])}
    if coding is not None:
        is_cod = df["variant_key"].isin(coding)
        df["coding"] = is_cod
        subsets["non-coding only"] = exact & ~is_cod
    if "tss_distance_pc" in df:
        # signed genome-coordinate distance; NaN is the only missing code
        t = pd.to_numeric(df["tss_distance_pc"], errors="coerce")
        valid = t.notna()
        subsets[f"promoter-proximal (<= {PROMOTER_BP} bp)"] = (
            exact & valid & (t.abs() <= PROMOTER_BP))
        subsets[f"distal (> {PROMOTER_BP} bp)"] = exact & valid & (t.abs() > PROMOTER_BP)

    phred_any = df["avi_phred"].fillna(df["avi_phred_swapped"])
    rows = []
    say("\n== F. Sensitivity: mean AVI PHRED by category, and adjusted "
        "multiple - specific ==")
    for name, msk in subsets.items():
        d = df[msk.to_numpy(bool)].assign(avi_phred=phred_any[msk])
        y, lab = d["avi_phred"].to_numpy(float), d["category"].to_numpy(object)
        g = group_means(y, lab)
        mdl = category_model(y, lab).set_index("term")
        adj = stratified_contrast(d, "multiple", "specific", ADJ_MAG_ACC)
        say(f"  {name:34s}" + "".join(
            f" {r.category}={r.mean:.2f}(n={r.n_variants:,})" for r in g.itertuples())
            + f"  | adj. mult-spec {adj['estimate']:+.2f} "
              f"[{adj['ci_lo']:+.2f}, {adj['ci_hi']:+.2f}]")
        for r in g.itertuples():
            rows.append(dict(subset=name, kind="group", term=r.category,
                             n=r.n_variants, estimate=r.mean, ci_lo=r.ci_lo,
                             ci_hi=r.ci_hi, p=np.nan))
        for term in ("specific - null", "multiple - null", "shared - null",
                     "multiple - specific", "shared - multiple"):
            if term in mdl.index:
                r = mdl.loc[term]
                rows.append(dict(subset=name, kind=r["model"], term=term, n=r["n"],
                                 estimate=r["estimate"], ci_lo=r["ci_lo"],
                                 ci_hi=r["ci_hi"], p=r["p"]))
        rows.append(dict(subset=name, kind=f"adjusted ({ADJ_MAG_ACC}), HC3",
                         term="multiple - specific",
                         n=adj.get("n_a_used", 0) + adj.get("n_b_used", 0),
                         estimate=adj["estimate"], ci_lo=adj["ci_lo"],
                         ci_hi=adj["ci_hi"], p=adj["p"]))
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Reproducibility manifest
# ──────────────────────────────────────────────────────────────────────────────
def file_info(p: Path, hash_limit: float = 2e9) -> dict:
    st = p.stat()
    info = dict(size_bytes=st.st_size,
                mtime=datetime.datetime.fromtimestamp(st.st_mtime).isoformat())
    if st.st_size <= hash_limit:
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        info["md5"] = h.hexdigest()
    else:
        info["md5"] = "not computed (file larger than 2 GB)"
    return info


def write_manifest(path: Path, args, inputs: list[Path]) -> None:
    import importlib.metadata as md
    pkgs = {}
    for p in ("numpy", "pandas", "pyarrow", "scipy", "statsmodels", "patsy",
              "matplotlib", "pysam"):
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pkgs[p] = None
    tabix = shutil.which("tabix")
    tabix_v = None
    if tabix:
        r = subprocess.run([tabix, "--version"], capture_output=True, text=True)
        tabix_v = (r.stdout or r.stderr).splitlines()[0] if (r.stdout or r.stderr) else None
    manifest = dict(
        script=str(Path(__file__).resolve()),
        script_md5=file_info(Path(__file__).resolve())["md5"],
        run_at=datetime.datetime.now().isoformat(timespec="seconds"),
        command=" ".join(sys.argv), python=sys.version, platform=platform.platform(),
        packages=pkgs, tabix=dict(path=tabix, version=tabix_v),
        inputs={str(p): file_info(p) for p in inputs if p.exists()},
        settings=dict(cell_types=CELL_TYPES, breadth_bounds=BREADTH_BOUNDS,
                      primary_rule=PRIMARY_RULE, ladder=LADDER,
                      min_ladder_specific=MIN_LADDER_SPECIFIC,
                      phred_cutoffs=PHRED_CUTOFFS, n_strata_magnitude=N_STRATA_MAG,
                      min_per_stratum=MIN_PER_STRATUM,
                      accessibility_edges=ACC_EDGES, spline_df=SPLINE_DF,
                      promoter_bp=PROMOTER_BP, coding_terms=sorted(CODING_TERMS)))
    path.write_text(json.dumps(manifest, indent=2, default=str))
    say(f"wrote {path}")


# ──────────────────────────────────────────────────────────────────────────────
def print_model(title: str, groups: pd.DataFrame, model: pd.DataFrame) -> None:
    say(f"\n== {title} ==")
    for r in groups.itertuples():
        say(f"  {r.category:9s} n = {r.n_variants:>8,}  mean = {r.mean:.4f} "
            f"[{r.ci_lo:.4f}, {r.ci_hi:.4f}]  median = {r.median:.4f}")
    for r in model.itertuples():
        say(f"  {r.term:22s} {r.estimate:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]  "
            f"p = {fmt_p(r.p)}  ({r.model})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipe-dir", type=Path, required=True,
                    help="output directory of 01_build_matrix.py / 03_breadth.py")
    ap.add_argument("--avi", type=Path, required=True,
                    help="bgzipped + tabix-indexed AVI SNV score table")
    ap.add_argument("--out-dir", type=Path, default=Path.cwd())
    args = ap.parse_args()
    pipe, avi = args.pipe_dir.expanduser().resolve(), args.avi.expanduser().resolve()
    required = [pipe / "variant_celltype.parquet", pipe / "variants.parquet", avi,
                Path(str(avi) + ".tbi")]
    for p in required:
        if not p.exists():
            raise SystemExit(f"required input not found: {p}")
    set_style()
    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ── breadth from ChromBPNet ──────────────────────────────────────────────
    long = load_long(pipe)
    per = per_variant(long)
    chrombpnet_sanity(per, long, pipe)
    per = per[per["n_accessible"] > 0]
    n_acc_all = per["n_accessible"].to_numpy()

    # ── AVI lookup ───────────────────────────────────────────────────────────
    say("\n== AVI lookup ==")
    v = load_variants(pipe)
    df = per.merge(v, on="variant_idx", how="left")
    if df["chr"].isna().any():
        say(f"WARNING: {int(df['chr'].isna().sum()):,} analysed variants have no row "
            f"in variants.parquet; they are dropped")
        df = df[df["chr"].notna()].copy()
    hits = fetch_avi(df[["chr", "pos"]], avi)
    say(f"AVI rows returned: {len(hits):,}")
    df = match_avi(df, hits)
    report = (pd.crosstab(df["category"], df["avi_match"], margins=True)
                .reindex(BREADTH_ORDER + ["All"]))
    report.to_csv(out / "avi_breadth_match_report.csv")
    say("lookup status by category (analyses use 'exact'):")
    for line in report.to_string().splitlines():
        say("  " + line)
    if "tss_distance_pc" in df:
        t = pd.to_numeric(df["tss_distance_pc"], errors="coerce")
        miss = t.isna().groupby(df["category"]).mean().reindex(BREADTH_ORDER)
        say("tss_distance_pc missing (NaN): "
            + ", ".join(f"{c} {r:.1%}" for c, r in miss.items()))

    m = df[df["avi_match"] == "exact"].copy()
    lab = m["category"].to_numpy(object)
    phred = m["avi_phred"].to_numpy(float)
    raw = m["avi_raw"].to_numpy(float)
    mag = m["magnitude"].to_numpy(float)

    # ── A. category means ───────────────────────────────────────────────────
    g_ph, m_ph = group_means(phred, lab), category_model(phred, lab)
    g_raw, m_raw = group_means(raw, lab), category_model(raw, lab)
    g_cb, m_cb = group_means(mag, lab), category_model(mag, lab)
    print_model("A. AVI PHRED by breadth category (primary)", g_ph, m_ph)
    print_model("A. AVI raw score by breadth category", g_raw, m_raw)
    print_model("A. ChromBPNet max |logFC| on the same AVI-matched variants",
                g_cb, m_cb)
    pd.concat([g_ph.assign(score="avi_phred"), g_raw.assign(score="avi_raw"),
               g_cb.assign(score="chrombpnet_max_abs_logfc")]
              ).to_csv(out / "avi_breadth_groups.csv", index=False)
    pd.concat([m_ph.assign(score="avi_phred"), m_raw.assign(score="avi_raw"),
               m_cb.assign(score="chrombpnet_max_abs_logfc")]
              ).to_csv(out / "avi_breadth_model.csv", index=False)
    rho_all = stats.spearmanr(m["breadth"], phred)
    act = (m["breadth"] > 0).to_numpy()
    rho_act = stats.spearmanr(m.loc[act, "breadth"], phred[act])
    say(f"\nSpearman rho, breadth vs AVI PHRED: all = {rho_all.statistic:+.4f} "
        f"(p = {fmt_p(rho_all.pvalue)}); active only = {rho_act.statistic:+.4f} "
        f"(p = {fmt_p(rho_act.pvalue)})")

    # ── B. cutoffs, C. discrimination ───────────────────────────────────────
    cut = cutoff_table(phred, lab)
    cut.to_csv(out / "avi_breadth_cutoffs.csv", index=False)
    say("\n== B. Selection by AVI PHRED cutoff ==")
    for t in PHRED_CUTOFFS:
        c = cut[cut["phred_cutoff"] == t].set_index("category")
        say(f"  PHRED >= {t}: " + "; ".join(
            f"{k} {c.loc[k, 'frac_pass']:.1%} [{c.loc[k, 'frac_pass_ci_lo']:.1%}, "
            f"{c.loc[k, 'frac_pass_ci_hi']:.1%}] of {c.loc[k, 'n_variants']:,}"
            for k in BREADTH_ORDER))
        say(f"      specific share: {c.loc['specific', 'share_of_active']:.1%} of "
            f"active variants, {c.loc['specific', 'share_of_selected_active']:.1%} "
            f"of active variants passing")

    auc = auroc_table(phred, lab)
    auc.to_csv(out / "avi_breadth_auroc.csv", index=False)
    say("\n== C. AUROC of AVI PHRED ==")
    for r in auc.itertuples():
        say(f"  {r.contrast:22s} AUROC = {r.auroc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
            f"  (n = {r.n_pos:,} vs {r.n_neg:,}, MWU p = {fmt_p(r.p_mann_whitney)})")

    # ── D. effect-size and accessibility adjustment ─────────────────────────
    say("\n== D. Effect size and accessibility breadth ==")
    acc_tab = accessibility_table(m)
    acc_tab.to_csv(out / "avi_breadth_by_accessibility.csv", index=False)
    say("  mean AVI PHRED by accessibility breadth (cell types where accessible):")
    for c in BREADTH_ORDER:
        t = acc_tab[acc_tab["category"] == c]
        say(f"    {c:9s} " + "  ".join(
            f"{r.accessible_in}: {r.mean_phred:.2f} (n={r.n_variants:,})"
            for r in t.itertuples()))

    adj_rows = []
    say("  stratified contrasts, common support of magnitude, HC3:")
    for a, b in PAIRS:
        for adjust in (ADJ_MAG, ADJ_MAG_ACC):
            r = stratified_contrast(m, a, b, adjust)
            adj_rows.append(r)
            if r.get("support_lo", 0) > r.get("support_hi", 1):
                say(f"    {r['contrast']:20s} | {adjust:34s} no common support of "
                    f"magnitude ({a} >= {r['support_lo']:.3f}, {b} <= "
                    f"{r['support_hi']:.3f}); not estimable")
                continue
            if "not_estimable" in r:
                say(f"    {r['contrast']:20s} | {adjust:34s} not estimable: "
                    f"{r['not_estimable']} (used {r.get('n_a_used', 0):,} vs "
                    f"{r.get('n_b_used', 0):,}; {r.get('n_strata', 0)} strata)")
                continue
            say(f"    {r['contrast']:20s} | {adjust:34s} "
                f"{r['estimate']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
                f"p = {fmt_p(r['p'])}  (used {r.get('n_a_used', 0):,} of "
                f"{r['n_a_total']:,} vs {r.get('n_b_used', 0):,} of "
                f"{r['n_b_total']:,}; {r.get('n_strata', 0)} strata)")
    for r in spline_model(m):
        adj_rows.append(r)
        say(f"    {r['contrast']:20s} | spline + accessibility (active only)   "
            f"{r['estimate']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
            f"p = {fmt_p(r['p'])}")
    say("  cutoff loss at matched effect size and accessibility "
        "(Mantel-Haenszel odds ratio of passing):")
    for t in (10, 20):
        for a, b in PAIRS[:3]:
            r = mh_cutoff(m, a, b, t, ADJ_MAG_ACC)
            adj_rows.append(r)
            say(f"    PHRED >= {t}: {r['contrast']:20s} OR = {r['estimate']:.2f} "
                f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] p = {fmt_p(r['p'])} "
                f"({r.get('n_informative_strata', 0)} informative strata)")
    pd.DataFrame(adj_rows).to_csv(out / "avi_breadth_adjusted.csv", index=False)

    mb = matched_bins(m)
    mb.to_csv(out / "avi_breadth_matched_bins.csv", index=False)

    # ── E. threshold ladder ─────────────────────────────────────────────────
    lad = run_ladder(m, long, n_acc_all)
    lad.to_csv(out / "avi_breadth_ladder.csv", index=False)

    # ── G. AlphaGenome biosample coverage ───────────────────────────────────
    per_ct, cov_tab = coverage_analysis(m, long)
    pd.concat([per_ct, cov_tab], ignore_index=True).to_csv(
        out / "avi_breadth_coverage.csv", index=False)

    # ── F. sensitivity ──────────────────────────────────────────────────────
    coding = load_coding_keys(pipe)
    sens = sensitivity(df, coding)
    sens.to_csv(out / "avi_breadth_sensitivity.csv", index=False)

    # ── outputs ─────────────────────────────────────────────────────────────
    keep = ["variant_idx", "variant_key", "rsid", "nearest_gene_pc", "tss_distance_pc",
            "breadth", "n_accessible", "category", "magnitude", "avi_match",
            "avi_raw", "avi_phred", "avi_raw_swapped", "avi_phred_swapped", "coding"]
    df[[c for c in keep if c in df.columns]].to_csv(
        out / "avi_breadth_variants.csv.gz", index=False)

    say("")
    save_single(g_ph, m_ph, "Mean AVI score (PHRED)", out / "avi_phred_by_category.svg")
    save_single(g_raw, m_raw, "Mean AVI score (raw logit)",
                out / "avi_raw_by_category.svg")
    save_side_by_side(g_cb, m_cb, g_ph, m_ph, out / "chrombpnet_vs_avi_by_category.svg")
    save_matched(mb, out / "avi_phred_by_matched_magnitude.svg")
    save_ladder(lad, out / "avi_breadth_ladder.svg")
    save_coverage(per_ct, out / "avi_phred_by_active_cell_type.svg")

    inputs = required + [pipe / "vep" / "vep_most_severe.tsv",
                         pipe / "tables" / "breadth_magnitude_groups.csv"]
    write_manifest(out / "avi_breadth_run_manifest.json", args, inputs)
    (out / "avi_breadth_summary.txt").write_text("\n".join(LOG) + "\n")
    print(f"wrote {out / 'avi_breadth_summary.txt'} and the CSV tables")


if __name__ == "__main__":
    main()
