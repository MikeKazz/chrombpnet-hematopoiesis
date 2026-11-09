#!/usr/bin/env python
"""
ccre_by_breadth.py: ENCODE cCRE annotation of the breadth categories.

Question
--------
Are particular classes of ENCODE candidate cis-regulatory element (cCRE), for example
distal enhancer-like signatures (dELS), enriched in a particular breadth category
(specific / multiple / shared)?

Population and breadth (same as 03_breadth.py / breadth_magnitude_groups.csv)
---------------------------------------------------------------------------
Variants in variant_celltype.parquet that are accessible in >= 1 of the 17 cell types
(the variants with a defined regulatory magnitude). Breadth = number of the 17 cell
types in which the variant is `called` (accessible AND empirical P < 0.01):
    inactive 0 | specific 1 | multiple 2-13 | shared 14-17   (shared = >= 80% of 17)
The category sizes are checked against tables/breadth_magnitude_groups.csv, and the script
stops if they disagree.

Annotation
----------
Each variant gets the cCRE class of the cCRE it falls in (BED 0-based half-open; variant
positions are 1-based, so the test is start <= pos-1 < end), or "no cCRE". cCREs in the
registry do not normally overlap. If a variant falls in more than one, the class is taken
by the priority PLS > pELS > dELS > CA-H3K4me3 > CA-CTCF > CA-TF > CA > TF, and the
number of such variants is reported.

Tests (2x2 Fisher's exact; OR CI = Woolf log-OR with a Haldane 0.5 floor on the SE
only, as in 07_quadrants.py; classes with < MIN_COUNT carriers in the group are
not tested; BH across every test in the output table)
  vs_inactive      each active category against the inactive variants
                   (parallels the VEP analysis, which is against "neither")
  vs_other_active  each active category against the other two active categories, which
                   holds "active" fixed. This is the direct test of whether a cCRE class
                   distinguishes one breadth category from the others.
Global: chi-square test of independence, breadth category (3 active) x cCRE class.

Caveats
  - pELS vs dELS is defined by distance to an annotated TSS (pELS within 2 kb), so a dELS
    or PLS difference partly restates a TSS-distance difference.
  - Every variant here is already in an ATAC consensus peak, so the background is
    accessible chromatin and overall cCRE overlap is high in every group.
  - shared has only ~77 variants, so its intervals are wide and some classes are untested.

Inputs
  <input-dir>/variants.parquet                      chr, pos, allele1, allele2 per variant_idx
  <input-dir>/variant_celltype.parquet              called, accessible per variant x cell type
  <input-dir>/tables/breadth_magnitude_groups.csv   category sizes, for the cross-check
  --ccre-bed                                        ENCODE cCREs (GRCh38), 6 columns, no header

Outputs (written to --out-dir, default: current directory)
  ccre_by_breadth_variants.tsv.gz      one row per variant: key, breadth, category, cCRE class, accession
  ccre_by_breadth_composition.csv      counts and fractions, category x cCRE class
  ccre_by_breadth_enrichment.csv       every test: comparison, category, class, counts, OR, CI, p, q
  ccre_by_breadth_global_test.csv      chi-square of independence among active categories
  ccre_by_breadth.svg                  (a) composition, (b) OR vs inactive, (c) OR vs other active

Usage
-----
    python ccre_by_breadth.py --input-dir results/variant_analysis \\
        --ccre-bed data/external/GRCh38-cCREs.bed --out-dir results/ccre

Requirements: python >= 3.9, numpy, pandas, pyarrow, scipy, matplotlib.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
from scipy import stats

OUT_NAMES = dict(
    OUT_VARIANTS="ccre_by_breadth_variants.tsv.gz",
    OUT_COMP="ccre_by_breadth_composition.csv",
    OUT_ENR="ccre_by_breadth_enrichment.csv",
    OUT_GLOBAL="ccre_by_breadth_global_test.csv",
    OUT_SVG="ccre_by_breadth.svg",
)
OUT_VARIANTS = OUT_COMP = OUT_ENR = OUT_GLOBAL = OUT_SVG = None  # set in main()

# ──────────────────────────────────────────────────────────────────────────────
# Settings (config.py)
# ──────────────────────────────────────────────────────────────────────────────
CELL_TYPES = ["HSC", "MPP", "LMPP", "CMP", "GMP", "MEP", "CLP",
              "Ery", "Mega", "Mono", "Bcell", "CD4T", "CD8T", "NK",
              "CD34_BM", "CD34_CB", "K562"]            # universe all17
SHARED_FRACTION = 0.80
MIN_COUNT = 5                                          # config.VEP_MIN_COUNT, as in 07
Q_THRESHOLD = 0.05

CATEGORIES = ["inactive", "specific", "multiple", "shared"]
ACTIVE = ["specific", "multiple", "shared"]
CAT_COLOUR = {"inactive": "#BDBDBD", "specific": "#9ECAE1",
              "multiple": "#4292C6", "shared": "#08519C"}

# ENCODE cCRE classes (registry V4), priority order for the rare multi-overlap case,
# with the SCREEN colour scheme. "no cCRE" = outside every cCRE.
CCRE_CLASSES = ["PLS", "pELS", "dELS", "CA-H3K4me3", "CA-CTCF", "CA-TF", "CA", "TF"]
CCRE_LABEL = {"PLS": "Promoter-like (PLS)", "pELS": "Proximal enhancer-like (pELS)",
              "dELS": "Distal enhancer-like (dELS)", "CA-H3K4me3": "CA-H3K4me3",
              "CA-CTCF": "CA-CTCF", "CA-TF": "CA-TF", "CA": "CA only", "TF": "TF only",
              "no cCRE": "No cCRE"}
CCRE_COLOUR = {"PLS": "#FF0000", "pELS": "#FFA700", "dELS": "#FFCD00",
               "CA-H3K4me3": "#FFAAAA", "CA-CTCF": "#00B0F0", "CA-TF": "#BE28E5",
               "CA": "#06DA93", "TF": "#D876EC", "no cCRE": "#E6E6E6"}


def breadth_category(b: np.ndarray, n_ct: int) -> np.ndarray:
    shared_min = math.ceil(SHARED_FRACTION * n_ct)
    return np.select([b == 0, b == 1, (b >= 2) & (b < shared_min), b >= shared_min],
                     CATEGORIES, default="")


# ──────────────────────────────────────────────────────────────────────────────
# Breadth (as 03_breadth.py)
# ──────────────────────────────────────────────────────────────────────────────
def load_breadth(inp: Path) -> pd.DataFrame:
    long = pd.read_parquet(inp / "variant_celltype.parquet",
                           columns=["variant_idx", "cell_type", "called", "accessible"])
    long = long[long["cell_type"].astype(str).isin(CELL_TYPES)]
    n_ct = long["cell_type"].astype(str).nunique()
    g = long.groupby("variant_idx", sort=False)
    df = pd.DataFrame({"breadth": g["called"].sum().astype(int),
                       "n_accessible": g["accessible"].sum().astype(int)})
    df = df[df["n_accessible"] >= 1]           # defined regulatory magnitude, as in 03
    df["category"] = breadth_category(df["breadth"].to_numpy(), n_ct)

    V = pd.read_parquet(inp / "variants.parquet",
                        columns=["variant_idx", "chr", "pos", "allele1", "allele2"])
    df = df.join(V.set_index("variant_idx"), how="left")
    if df["pos"].isna().any():
        raise SystemExit(f"{int(df['pos'].isna().sum())} variants missing from variants.parquet")
    df["chr"] = df["chr"].astype(str)
    df["pos"] = df["pos"].astype(np.int64)
    df["variant_key"] = (df["chr"] + ":" + df["pos"].astype(str) + ":"
                         + df["allele1"].astype(str) + ":" + df["allele2"].astype(str))
    print(f"{n_ct} cell types; {len(df):,} variants accessible in >= 1 cell type")
    return df.reset_index()


def check_against_03(df: pd.DataFrame, path: Path) -> None:
    if not path.exists():
        print(f"WARNING: {path} not found; category sizes NOT cross-checked against 03")
        return
    # 03 writes the inactive category as the literal string "null"; keep it as a string
    g = pd.read_csv(path, keep_default_na=False, na_values=[""])
    g["category"] = g["category"].replace({"null": "inactive"})
    ref = g.set_index("category")["n_variants"]
    got = df["category"].value_counts()
    bad = {c: (int(got.get(c, 0)), int(ref[c])) for c in CATEGORIES
           if c in ref and int(got.get(c, 0)) != int(ref[c])}
    if bad:
        raise SystemExit(f"category sizes disagree with {path.name} (here, 03): {bad}")
    print("category sizes match breadth_magnitude_groups.csv: "
          + ", ".join(f"{c} {int(got[c]):,}" for c in CATEGORIES))


# ──────────────────────────────────────────────────────────────────────────────
# cCRE overlap
# ──────────────────────────────────────────────────────────────────────────────
def load_ccre(path: Path) -> pd.DataFrame:
    c = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 4, 5],
                    names=["chr", "start", "end", "accession", "ccre_class"],
                    dtype={0: str, 1: np.int64, 2: np.int64, 4: str, 5: str})
    if not c["chr"].str.startswith("chr").all():
        c["chr"] = "chr" + c["chr"].str.removeprefix("chr")
    unknown = sorted(set(c["ccre_class"]) - set(CCRE_CLASSES))
    if unknown:
        raise SystemExit(f"unexpected cCRE classes {unknown}; add them to CCRE_CLASSES")
    print(f"cCREs: {len(c):,} ({', '.join(f'{k} {v:,}' for k, v in c['ccre_class'].value_counts().items())})")
    return c


def annotate(df: pd.DataFrame, ccre: pd.DataFrame) -> pd.DataFrame:
    """Class and accession of the cCRE containing each variant (start <= pos0 < end)."""
    prio = {k: i for i, k in enumerate(CCRE_CLASSES)}
    cls = np.full(len(df), "no cCRE", dtype=object)
    acc = np.full(len(df), "", dtype=object)
    n_hits = np.zeros(len(df), dtype=np.int16)
    pos0 = df["pos"].to_numpy(np.int64) - 1
    chrom = df["chr"].to_numpy()
    for ch, sub in ccre.groupby("chr", sort=False):
        sel = np.flatnonzero(chrom == ch)
        if not sel.size:
            continue
        sub = sub.sort_values("start")
        s, e = sub["start"].to_numpy(), sub["end"].to_numpy()
        k_cls, k_acc = sub["ccre_class"].to_numpy(), sub["accession"].to_numpy()
        runmax = np.maximum.accumulate(e)
        p = pos0[sel]
        j = np.searchsorted(s, p, side="right") - 1      # last cCRE starting <= p
        best = np.full(sel.size, len(CCRE_CLASSES), dtype=int)
        # walk left while some earlier cCRE could still cover p (handles overlaps)
        live = j >= 0
        while live.any():
            jj = j[live]
            hit = e[jj] > p[live]
            idx = np.flatnonzero(live)[hit]
            for i_loc, jk in zip(idx, j[idx]):
                n_hits[sel[i_loc]] += 1
                pr = prio[k_cls[jk]]
                if pr < best[i_loc]:
                    best[i_loc] = pr
                    cls[sel[i_loc]] = k_cls[jk]
                    acc[sel[i_loc]] = k_acc[jk]
            j = j - 1
            live = (j >= 0)
            live[live] = runmax[j[live]] > p[live]
    out = df.copy()
    out["ccre_class"], out["ccre_accession"], out["n_ccre_overlaps"] = cls, acc, n_hits
    multi = int((n_hits > 1).sum())
    print(f"variants in a cCRE: {(n_hits > 0).mean():.1%}; in more than one: {multi:,}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────
def fisher_row(a: int, b: int, c: int, d: int) -> dict:
    odds, p = stats.fisher_exact([[a, b], [c, d]])
    se = float(np.sqrt((1.0 / np.maximum(np.array([a, b, c, d], float), 0.5)).sum()))
    lo = float(np.exp(np.log(odds) - 1.96 * se)) if odds > 0 and np.isfinite(odds) else np.nan
    hi = float(np.exp(np.log(odds) + 1.96 * se)) if odds > 0 and np.isfinite(odds) else np.nan
    return dict(odds_ratio=float(odds), ci_lo=lo, ci_hi=hi, p_fisher=float(p))


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():                                   # Benjamini-Hochberg step-up
        x = p[ok]
        o = np.argsort(x)
        r = x[o] * len(x) / np.arange(1, len(x) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        out = np.empty_like(x)
        out[o] = np.minimum(r, 1.0)
        q[ok] = out
    return q


def enrichment(ann: pd.DataFrame) -> pd.DataFrame:
    classes = CCRE_CLASSES + ["no cCRE"]
    rows = []
    for comparison in ("vs_inactive", "vs_other_active"):
        for cat in ACTIVE:
            grp = ann["category"] == cat
            bg = (ann["category"] == "inactive") if comparison == "vs_inactive" \
                else ann["category"].isin([x for x in ACTIVE if x != cat])
            for k in classes:
                carrier = ann["ccre_class"] == k
                a, c = int((grp & carrier).sum()), int((bg & carrier).sum())
                b, d = int(grp.sum()) - a, int(bg.sum()) - c
                row = dict(comparison=comparison, category=cat, ccre_class=k,
                           n_in_group=a, n_group_total=a + b,
                           n_in_reference=c, n_reference_total=c + d,
                           frac_group=a / max(a + b, 1), frac_reference=c / max(c + d, 1))
                if a >= MIN_COUNT:
                    row.update(fisher_row(a, b, c, d))
                else:
                    row.update(odds_ratio=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_fisher=np.nan)
                rows.append(row)
    out = pd.DataFrame(rows)
    out["q_value"] = bh(out["p_fisher"].to_numpy())     # one family: the whole table
    out["p_underflow"] = out["p_fisher"] == 0.0
    out["tested"] = out["p_fisher"].notna()
    return out


def global_test(ann: pd.DataFrame) -> pd.DataFrame:
    act = ann[ann["category"].isin(ACTIVE)]
    tab = pd.crosstab(act["category"], act["ccre_class"]).reindex(
        index=ACTIVE, columns=CCRE_CLASSES + ["no cCRE"], fill_value=0)
    tab = tab.loc[:, tab.sum() > 0]
    chi2, p, dof, exp = stats.chi2_contingency(tab.to_numpy())
    frac_low = float((exp < 5).mean())
    res = dict(test="chi-square, active breadth category x cCRE class",
               chi2=chi2, dof=dof, p=p, n=int(tab.to_numpy().sum()),
               frac_expected_below_5=frac_low)
    if frac_low > 0.2:
        # sparse table (driven by the small shared group): add a Monte Carlo exact-style P
        rng = np.random.default_rng(1)
        r_lab = np.repeat(np.arange(tab.shape[0]), tab.sum(axis=1).to_numpy())
        c_lab = np.repeat(np.arange(tab.shape[1]), tab.sum(axis=0).to_numpy())
        obs = chi2
        n_ge, n_sim = 0, 2000
        for _ in range(n_sim):
            t = np.zeros(tab.shape)
            np.add.at(t, (r_lab, rng.permutation(c_lab)), 1)
            n_ge += stats.chi2_contingency(t, correction=False)[0] >= obs - 1e-9
        res["p_permutation"] = (1 + n_ge) / (1 + n_sim)
        res["n_permutations"] = n_sim
    return pd.DataFrame([res])


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────
def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 0, "legend.frameon": False,
        "svg.fonttype": "none", "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def _or_panel(ax, enr: pd.DataFrame, comparison: str, classes: list[str], title: str):
    sub = enr[enr["comparison"] == comparison]
    y0 = np.arange(len(classes))[::-1]
    offs = {"specific": 0.22, "multiple": 0.0, "shared": -0.22}
    finite = sub[np.isfinite(sub["odds_ratio"]) & (sub["odds_ratio"] > 0)]
    lo = min(finite["ci_lo"].min(), 0.8) if len(finite) else 0.5
    hi = max(finite["ci_hi"].max(), 1.25) if len(finite) else 2.0
    for cat in ACTIVE:
        s = sub[sub["category"] == cat].set_index("ccre_class").reindex(classes)
        y = y0 + offs[cat]
        ok = np.isfinite(s["odds_ratio"]).to_numpy() & (s["odds_ratio"] > 0).to_numpy()
        orv = s["odds_ratio"].to_numpy(float)
        sig = (s["q_value"] < Q_THRESHOLD).to_numpy()
        ax.errorbar(orv[ok], y[ok], xerr=[orv[ok] - s["ci_lo"].to_numpy(float)[ok],
                                          s["ci_hi"].to_numpy(float)[ok] - orv[ok]],
                    fmt="none", ecolor=CAT_COLOUR[cat], elinewidth=0.8, capsize=1.5, zorder=2)
        ax.scatter(orv[ok & sig], y[ok & sig], s=16, color=CAT_COLOUR[cat],
                   edgecolor="0.15", linewidth=0.4, zorder=3)
        ax.scatter(orv[ok & ~sig], y[ok & ~sig], s=16, facecolor="white",
                   edgecolor=CAT_COLOUR[cat], linewidth=0.9, zorder=3)
        for yi in y[~ok]:
            ax.text(1.0, yi, "  n < %d" % MIN_COUNT, fontsize=4.8, color="0.55",
                    va="center", ha="left")
    ax.axvline(1, color="0.3", lw=0.6, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(lo / 1.3, hi * 1.3)
    ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20) if lo / 1.3 <= t <= hi * 1.3]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks(y0)
    ax.set_ylim(-0.6, len(classes) - 0.4)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Odds ratio (log scale)")


def plot(ann: pd.DataFrame, enr: pd.DataFrame):
    classes = CCRE_CLASSES + ["no cCRE"]
    # a spacer column between (a) and (b) holds the cCRE class labels of panel (b)
    fig = plt.figure(figsize=(7.09, 2.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 0.62, 1, 1], wspace=0.12)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])]
    # (a) composition
    ax = axes[0]
    comp = (pd.crosstab(ann["category"], ann["ccre_class"], normalize="index")
              .reindex(index=CATEGORIES, columns=classes, fill_value=0))
    n = ann["category"].value_counts()
    ycat = np.arange(len(CATEGORIES))[::-1]
    left = np.zeros(len(CATEGORIES))
    for k in classes:
        ax.barh(ycat, comp[k].to_numpy(), left=left, color=CCRE_COLOUR[k], height=0.7,
                edgecolor="white", linewidth=0.3)
        left += comp[k].to_numpy()
    ax.set_yticks(ycat)
    ax.set_yticklabels([f"{c.capitalize()}\n(n = {int(n.get(c, 0)):,})" for c in CATEGORIES])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of variants")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("a  cCRE composition by breadth", loc="left")
    ax.legend([Patch(color=CCRE_COLOUR[k]) for k in classes], [CCRE_LABEL[k] for k in classes],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=5.6,
              handlelength=0.9, columnspacing=0.8)

    # (b) vs inactive, (c) vs other active
    _or_panel(axes[1], enr, "vs_inactive", classes, "b  vs inactive variants")
    axes[1].set_yticklabels([CCRE_LABEL[k] for k in classes], fontsize=6)
    _or_panel(axes[2], enr, "vs_other_active", classes, "c  vs the other active categories")
    axes[2].set_yticklabels([])
    handles = [Line2D([], [], marker="o", ls="", color=CAT_COLOUR[c], markeredgecolor="0.15",
                      markersize=4, label=c.capitalize()) for c in ACTIVE]
    handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="white",
                          markeredgecolor="0.4", markersize=4, label=f"q \u2265 {Q_THRESHOLD:g}"))
    axes[2].legend(handles=handles, loc="upper center", bbox_to_anchor=(-0.05, -0.2),
                   ncol=4, fontsize=6, handletextpad=0.3, columnspacing=1.0)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="output directory of 01_build_matrix.py / 03_breadth.py")
    ap.add_argument("--ccre-bed", type=Path, required=True,
                    help="ENCODE cCRE registry (GRCh38), 6 columns, no header")
    ap.add_argument("--out-dir", type=Path, default=Path.cwd())
    args = ap.parse_args()
    global OUT_VARIANTS, OUT_COMP, OUT_ENR, OUT_GLOBAL, OUT_SVG
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT_VARIANTS, OUT_COMP, OUT_ENR, OUT_GLOBAL, OUT_SVG = (
        out_dir / OUT_NAMES[k] for k in
        ("OUT_VARIANTS", "OUT_COMP", "OUT_ENR", "OUT_GLOBAL", "OUT_SVG"))
    inp = args.input_dir.expanduser().resolve()
    for p in (inp / "variants.parquet", inp / "variant_celltype.parquet", args.ccre_bed):
        if not Path(p).exists():
            raise SystemExit(f"required input not found: {p}")

    df = load_breadth(inp)
    check_against_03(df, inp / "tables" / "breadth_magnitude_groups.csv")
    ann = annotate(df, load_ccre(args.ccre_bed.expanduser().resolve()))

    ann[["variant_idx", "variant_key", "breadth", "category", "ccre_class",
         "ccre_accession", "n_ccre_overlaps"]].to_csv(OUT_VARIANTS, sep="\t", index=False)

    classes = CCRE_CLASSES + ["no cCRE"]
    cnt = pd.crosstab(ann["category"], ann["ccre_class"]).reindex(
        index=CATEGORIES, columns=classes, fill_value=0)
    frac = cnt.div(cnt.sum(axis=1), axis=0)
    comp = (cnt.stack().rename("n").to_frame()
               .join(frac.stack().rename("fraction")).reset_index())
    comp.columns = ["category", "ccre_class", "n", "fraction"]
    comp.to_csv(OUT_COMP, index=False)

    enr = enrichment(ann)
    enr.to_csv(OUT_ENR, index=False)
    glob = global_test(ann)
    glob.to_csv(OUT_GLOBAL, index=False)

    print("\nfraction of each category in each cCRE class:")
    print((frac * 100).round(1).to_string())
    print("\nsignificant (q < %.2g):" % Q_THRESHOLD)
    sig = enr[enr["q_value"] < Q_THRESHOLD]
    for r in sig.itertuples():
        print(f"  {r.comparison:<16s} {r.category:<9s} {r.ccre_class:<11s} "
              f"OR {r.odds_ratio:6.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]  q = {r.q_value:.2g}")
    g = glob.iloc[0]
    print(f"\nglobal chi-square (active categories x class): chi2 = {g.chi2:.1f}, "
          f"df = {int(g.dof)}, P = {g.p:.3g}"
          + (f"; permutation P = {g.p_permutation:.3g}" if "p_permutation" in glob else ""))

    set_style()
    fig = plot(ann, enr)
    fig.savefig(OUT_SVG, format="svg")
    plt.close(fig)
    for f in (OUT_VARIANTS, OUT_COMP, OUT_ENR, OUT_GLOBAL, OUT_SVG):
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
