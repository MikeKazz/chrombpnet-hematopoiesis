#!/usr/bin/env python
"""
07_quadrants.py: sequence-model evidence against fine-mapping evidence.

The question
------------
Fine-mapping and a sequence model are two independent lines of evidence about whether a
variant is causal. Partitioning variants on both at once gives four classes, and the two
disagreeing classes are the interesting ones:

  Q1  causal-probable, model-active      both agree
  Q2  causal-probable, model-inactive    statistics only
  Q3  causal-improbable, model-active    model only
  Q4  neither

The model axis uses the same definition as the other scripts: a variant is active when it
is accessible and its within-cell-type empirical p is below the primary level in at least
one cell type (config.primary_rule()). A global cut on |logFC| is not used, since it
cannot mean the same thing in cell types with different dynamic range. The causality axis
is PIP >= 0.5, the level above which a variant is the leading candidate in its credible
set.

Everything is counted at variant level except the trait-lineage test, where the unit of
analysis is the variant-trait pair.

The test that matters
---------------------
Q3 is either model error or fine-mapping that has not resolved an LD block. These make
opposite predictions about a quantity fine-mapping never sees: which cell type the variant
is active in. If Q3 is noise, the cell types should be unrelated to the trait. If Q3 is
real regulatory signal, a variant carrying a red-cell association should be active in
erythroid cells rather than in lymphocytes.

Each trait is mapped to a haematopoietic lineage (config.TRAIT_LINEAGE) and the test asks how
often an active variant is active in at least one cell type of its own trait's lineage. The
null permutes whole trait-sets between variants, which preserves each variant's activity
pattern, each variant's number of associated traits and the overall trait frequencies, and
breaks only the variant-to-trait correspondence. Q1 is the positive control: it should match
at least as well as Q3.

The mapping is coarse and fixed before the analysis, and every result is reported under both a
strict cell-type set (mature cells plus the immediate progenitor) and a loose one (adding
the upstream progenitor), so the reader can see it does not turn on where the boundary is
drawn.

Where the threshold falls
-------------------------
Splitting a posterior at 0.5 invites the objection that the result is an artefact of that
number. The same variants are therefore also summarised as a gradient: the fraction of
variants active in at least one cell type, within bins of posterior, with a test for
monotone trend across the bins. A partition that only works at one cut would show as a
step; a real relationship shows as a rise across the whole range. The gradient is reported
both over all variants and over accessible variants alone, since accessibility is the one
property that plausibly drives both axes at once.

Outputs
-------
  tables/quadrant_counts.csv          the 2x2, with Fisher odds ratio and marginals
  tables/quadrant_vep_enrichment.csv  consequence-class enrichment per quadrant vs Q4
  tables/quadrant_trait_lineage.csv   trait-lineage concordance, observed against permuted
  tables/quadrant_aaq.csv             active-allele quantile per quadrant, accessible only
  tables/quadrant_aaq_tests.csv       all six pairwise Mann-Whitney comparisons
  tables/pip_gradient.csv             fraction active against posterior, in bins
  quadrant_qc.txt                     joins, exclusions and counts

No figures are drawn here; the tables are shaped one row per plotted element.

Usage
-----
  python 07_quadrants.py --input-dir <out> --pip-bed <CS bed> [--vep <vep.tsv>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from itertools import combinations
import pandas as pd
from scipy import stats

import config as C
from s4lib import QCReport, banner, log, peek_header, variant_key


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_pairs(path: Path, qc: QCReport) -> pd.DataFrame:
    """Variant-trait rows from the credible-set file: position, trait, posterior.

    01_build_matrix.py collapses this file to one maximum posterior per variant, which
    loses the trait; the trait is needed for the lineage test, so the file is read again.

    The file lists credible-set members without a set identifier: its UKID column is a
    variant key (chr:pos_ref_alt), and posteriors within a UKID and trait do not sum to
    one. Any analysis that needs set membership therefore cannot be run from this file,
    and none is attempted.
    """
    header = peek_header(path)
    need = {"chr", "start", "PP"}
    missing = need - set(header)
    if missing:
        raise KeyError(f"{path}: missing column(s) {sorted(missing)}. Header is: {header}")
    if "trait" not in header:
        raise KeyError(f"{path}: no 'trait' column; the trait-lineage test needs one. "
                       f"Header is: {header}")

    df = pd.read_csv(path, sep="\t", usecols=["chr", "start", "PP", "trait"],
                     dtype={"chr": str}, low_memory=False)
    df["chr"] = df["chr"].astype(str)
    if not df["chr"].str.startswith("chr").all():
        df["chr"] = "chr" + df["chr"].str.removeprefix("chr")

    qc.section("credible-set file")
    qc.add("rows", f"{len(df):,}")
    qc.add("unique sites", f"{df.groupby(['chr', 'start']).ngroups:,}")
    qc.add("traits", f"{df['trait'].nunique()} ({', '.join(sorted(df['trait'].unique())[:8])}...)")
    return df


def join_pairs_to_variants(pairs: pd.DataFrame, V: pd.DataFrame,
                           qc: QCReport) -> pd.DataFrame:
    """Attach variant_idx to each variant-trait row, using 01's coordinate convention.

    The convention is not re-derived: 01 already established it against this same file and
    recorded it in config. Re-deciding it here could silently disagree with the annotation
    the rest of the pipeline carries.
    """
    key = V[["variant_idx", "chr", "pos"]].copy()
    key["chr"] = key["chr"].astype(str)
    pairs = pairs.copy()
    pairs["pos"] = pairs["start"] + (1 if C.PIP_BED_IS_ZERO_BASED else 0)
    out = pairs.merge(key, on=["chr", "pos"], how="inner")

    qc.section("join to the variant table")
    qc.add("convention", f"pos = start + {1 if C.PIP_BED_IS_ZERO_BASED else 0} "
                         f"(PIP_BED_IS_ZERO_BASED = {C.PIP_BED_IS_ZERO_BASED})")
    qc.add("variant-trait rows joined", f"{len(out):,} of {len(pairs):,}")
    qc.add("unique variants joined", f"{out['variant_idx'].nunique():,}")
    before = len(out)
    out = (out.sort_values("PP", ascending=False)
              .drop_duplicates(subset=["variant_idx", "trait"], keep="first"))
    qc.add("duplicate variant-trait rows collapsed (max posterior kept)",
           f"{before - len(out):,}")
    frac = len(out) / max(len(pairs), 1)
    if frac < 0.5:
        qc.note(f"    WARNING: only {frac:.1%} of credible-set rows matched a scored "
                f"variant. Check the coordinate convention and the chromosome naming.")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# The partition
# ──────────────────────────────────────────────────────────────────────────────

def build_quadrants(V: pd.DataFrame, pairs: pd.DataFrame, pip_high: float,
                    qc: QCReport) -> pd.DataFrame:
    """One row per variant in the analysis universe, with its quadrant.

    The universe is every scored variant that appears in the credible-set file. It is
    deliberately NOT restricted to variants accessible somewhere. Restricting it would
    remove most coding variants, since exons are frequently outside ATAC peaks, and the
    coding composition of Q2 is the whole point of that quadrant: a variant not
    accessible in any cell type is not active, and belongs in Q2 or Q4 rather than being
    dropped from the table.
    """
    bcol = C.breadth_column(V)
    keep = [c for c in ("variant_idx", "chr", "pos", "allele1", "allele2", bcol,
                        "n_accessible", "testable") if c in V.columns]
    uni = V[keep].copy()
    uni["_key"] = variant_key(uni, C.VARIANT_KEY_COLUMNS)
    uni, n_dropped = C.collapse_duplicate_variants(uni)
    uni = uni.rename(columns={bcol: "n_active"})

    pip = (pairs.groupby("variant_idx", observed=True)["PP"].max().rename("pip_max"))
    uni = uni.merge(pip, on="variant_idx", how="inner")

    uni["accessible_anywhere"] = (uni["n_accessible"] >= 1 if "n_accessible" in uni
                                  else uni["n_active"] >= 1)
    uni["active"] = uni["n_active"] >= 1
    uni["causal_probable"] = uni["pip_max"] >= pip_high
    uni["quadrant"] = np.select(
        [uni["causal_probable"] & uni["active"],
         uni["causal_probable"] & ~uni["active"],
         ~uni["causal_probable"] & uni["active"]],
        ["Q1", "Q2", "Q3"], default="Q4")

    qc.section("analysis universe")
    qc.add("scored variants", f"{len(V):,}")
    qc.add("of those, in the credible-set file", f"{len(uni):,}")
    qc.add("of those, accessible in >=1 cell type",
           f"{int(uni['accessible_anywhere'].sum()):,}")
    qc.add("rows collapsed as duplicate variant keys", n_dropped)
    n_pos = uni.groupby(["chr", "pos"]).size() if "pos" in uni else None
    if n_pos is not None:
        multi = int((n_pos > 1).sum())
        qc.add("positions carrying more than one allele pair", f"{multi:,}")
        qc.note("    the credible-set join is on position, as 01 does, so variants "
                "sharing a position share a posterior. The count above bounds how "
                "often that applies.")
    qc.note("    the universe is NOT restricted to accessible variants: exons are often "
            "outside peaks, so that restriction would remove the coding variants that "
            "define Q2.")
    qc.add("model axis", f"active = {C.primary_rule()[2]}, in >=1 cell type")
    qc.add("causality axis", f"PIP >= {pip_high:g} (maximum across traits)")
    for q in ("Q1", "Q2", "Q3", "Q4"):
        sub = uni["quadrant"] == q
        qc.add(q, f"{int(sub.sum()):,} "
                  f"({int((sub & uni['accessible_anywhere']).sum()):,} accessible somewhere)")
    return uni


def partition_table(uni: pd.DataFrame) -> pd.DataFrame:
    n = {q: int((uni["quadrant"] == q).sum()) for q in ("Q1", "Q2", "Q3", "Q4")}
    tab = np.array([[n["Q1"], n["Q2"]], [n["Q3"], n["Q4"]]])
    odds, p = stats.fisher_exact(tab)
    se = float(np.sqrt((1.0 / np.maximum(tab, 0.5)).sum()))
    lo, hi = float(np.exp(np.log(odds) - 1.96 * se)), float(np.exp(np.log(odds) + 1.96 * se))
    hi_pip, act = n["Q1"] + n["Q2"], n["Q1"] + n["Q3"]
    acc = {q: int(((uni["quadrant"] == q) & uni["accessible_anywhere"]).sum()) for q in n}
    rows = [dict(quadrant=q, n_variants=n[q], frac_of_universe=n[q] / len(uni),
                 n_accessible_anywhere=acc[q],
                 n_never_accessible=n[q] - acc[q]) for q in n]
    rows.append(dict(quadrant="total", n_variants=len(uni), frac_of_universe=1.0,
                     n_accessible_anywhere=sum(acc.values()),
                     n_never_accessible=len(uni) - sum(acc.values())))
    out = pd.DataFrame(rows)
    out["odds_ratio"] = np.nan
    out.loc[out["quadrant"] == "Q1", "odds_ratio"] = odds
    out["ci_lo"] = np.where(out["quadrant"] == "Q1", lo, np.nan)
    out["ci_hi"] = np.where(out["quadrant"] == "Q1", hi, np.nan)
    out["p_fisher"] = np.where(out["quadrant"] == "Q1", p, np.nan)
    out["frac_of_high_pip"] = np.where(
        out["quadrant"].isin(["Q1", "Q2"]), out["n_variants"] / max(hi_pip, 1), np.nan)
    out["frac_of_active"] = np.where(
        out["quadrant"].isin(["Q1", "Q3"]), out["n_variants"] / max(act, 1), np.nan)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Consequence classes
# ──────────────────────────────────────────────────────────────────────────────

def vep_enrichment(uni: pd.DataFrame, vep: pd.DataFrame, min_count: int,
                   qc: QCReport) -> pd.DataFrame:
    """Enrichment of each consequence class in Q1, Q2 and Q3 against Q4.

    Q4 alone is the background. Using the union of Q1 and Q4, as an earlier version of
    this analysis did, puts Q1 on both sides of its own comparison and shrinks the very
    contrast being estimated.

    Two backgrounds, reported side by side under a `scope` column.

    `all` uses every Q4 variant. Most of them lie in no accessible region, and Q1 and Q3
    are accessible by construction, so a regulatory-region enrichment measured this way
    is partly an accessibility effect rather than a statement about the quadrant.

    `accessible` restricts BOTH the quadrant and the background to variants accessible in
    at least one cell type. Q1 and Q3 are unchanged by that restriction; Q2 and Q4 shrink
    to their accessible members. This is the comparison that holds accessibility fixed,
    and it is the one to quote for any regulatory-annotation claim. The two scopes are
    both reported so the shrinkage is visible rather than asserted.

    Benjamini-Hochberg is applied across the whole table, so the family spans both scopes
    and all three quadrants; `q_value` must be read with that in mind.
    """
    annotated = set(vep["variant_idx"].unique())
    qc.section("consequence classes")
    qc.add("variants with any VEP consequence", f"{len(annotated):,} of {len(uni):,}")
    qc.add("distinct consequence classes", vep["consequence"].nunique())
    qc.note("    long format: a variant carrying several classes is counted under each, "
            "so classes are not mutually exclusive.")

    v = uni[uni["variant_idx"].isin(annotated)]
    carriers = {c: set(g) for c, g in vep.groupby("consequence")["variant_idx"]}
    acc_ids = set(v.loc[v["accessible_anywhere"].to_numpy(bool), "variant_idx"])
    rows = []
    for scope in ("all", "accessible"):
        keep = (lambda s: s) if scope == "all" else (lambda s: s & acc_ids)
        by_q = {q: keep(set(v.loc[v["quadrant"] == q, "variant_idx"]))
                for q in ("Q1", "Q2", "Q3", "Q4")}
        bg = by_q["Q4"]
        qc.add(f"background size, scope={scope}", f"{len(bg):,}")
        for q in ("Q1", "Q2", "Q3"):
            sub = by_q[q]
            for cons in sorted(carriers):
                a = len(sub & carriers[cons])
                if a < min_count:
                    continue
                b, c = len(sub) - a, len(bg & carriers[cons])
                dd = len(bg) - c
                odds, p = stats.fisher_exact([[a, b], [c, dd]])
                # Haldane-Anscombe on the SE only, so a zero cell widens the interval
                # instead of making it undefined; the point estimate is untouched.
                se = float(np.sqrt((1.0 / np.maximum(
                    np.array([a, b, c, dd], float), 0.5)).sum()))
                rows.append(dict(scope=scope, quadrant=q, consequence=cons,
                                 n_in_quadrant=a, n_quadrant_total=len(sub),
                                 n_in_background=c, n_background_total=len(bg),
                                 odds_ratio=odds,
                                 ci_lo=float(np.exp(np.log(odds) - 1.96 * se)),
                                 ci_hi=float(np.exp(np.log(odds) + 1.96 * se)),
                                 p_fisher=p))
    out = pd.DataFrame(rows)
    if len(out):
        from s4lib import benjamini_hochberg
        out["q_value"] = benjamini_hochberg(out["p_fisher"].to_numpy())
        out = out.sort_values(["scope", "quadrant", "odds_ratio"],
                              ascending=[True, True, False])
        qc.add("Benjamini-Hochberg family size (both scopes, all quadrants)",
               f"{len(out):,} tests")
        w = out.pivot_table(index=["quadrant", "consequence"], columns="scope",
                            values="odds_ratio").dropna()
        if len(w):
            qc.add("classes testable under both scopes", f"{len(w):,}")
            qc.add("median odds-ratio ratio, accessible / all",
                   f"{float((w['accessible'] / w['all']).median()):.2f}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Trait-lineage concordance: the test of what Q3 is
# ──────────────────────────────────────────────────────────────────────────────

def trait_lineage_concordance(uni: pd.DataFrame, pairs: pd.DataFrame,
                              active_sets: dict[int, frozenset[str]],
                              n_perm: int, seed: int, qc: QCReport) -> pd.DataFrame:
    """Is an active variant active in a cell type belonging to its trait's lineage?

    Unit of analysis is the variant-trait pair. The null permutes whole trait-sets between
    variants of the same quadrant, so each variant keeps its activity pattern and its
    number of traits, each trait keeps its frequency, and only the correspondence between
    a variant and the traits it carries is destroyed. Permuting individual pair rows
    instead would break the block structure of a variant carrying several correlated
    traits and would understate the null.
    """
    lineage = pairs["trait"].map(C.trait_lineage)
    unmapped = sorted(set(pairs.loc[lineage.isna(), "trait"].unique())
                      - set(C.TRAIT_LINEAGE_EXCLUDED))
    if unmapped:
        raise KeyError(
            "these trait codes have no lineage assignment in config.TRAIT_LINEAGE: "
            f"{unmapped}. Add each one, or list it in TRAIT_LINEAGE_EXCLUDED if it is "
            "not specific to a lineage (as whole-blood counts are not).")

    pr = pairs.assign(lineage=lineage).dropna(subset=["lineage"])
    pr = pr.merge(uni[["variant_idx", "quadrant"]], on="variant_idx", how="inner")

    qc.section("trait-lineage concordance")
    qc.add("pairs with a lineage-specific trait", f"{len(pr):,}")
    qc.add("traits excluded as non-specific", ", ".join(C.TRAIT_LINEAGE_EXCLUDED) or "none")

    rng = np.random.default_rng(seed)
    rows = []
    for scope, cts in (("strict", C.LINEAGE_CELL_TYPES),
                       ("loose", C.LINEAGE_CELL_TYPES_LOOSE)):
        for q in ("Q1", "Q3"):
            sub = pr[pr["quadrant"] == q]
            vids = sub["variant_idx"].to_numpy()
            uniq = np.unique(vids)
            if len(sub) < 50 or len(uniq) < 20:
                qc.add(f"{scope} {q}", f"skipped: {len(sub)} pairs over {len(uniq)} "
                                       f"variants is too few to permute")
                continue
            # per variant, which lineages does its activity cover
            cover = {v: frozenset(lg for lg, s in cts.items()
                                  if active_sets.get(v, frozenset()) & s)
                     for v in uniq}
            lg_arr = sub["lineage"].to_numpy()
            obs = float(np.mean([lg in cover[v] for v, lg in zip(vids, lg_arr)]))

            # permute trait-sets between variants: relabel each variant as another variant
            pos = {v: k for k, v in enumerate(uniq)}
            idx = np.array([pos[v] for v in vids])
            cover_list = [cover[v] for v in uniq]
            null = np.empty(n_perm)
            for b in range(n_perm):
                perm = rng.permutation(len(uniq))
                permuted = [cover_list[perm[k]] for k in range(len(uniq))]
                null[b] = np.mean([lg in permuted[i] for i, lg in zip(idx, lg_arr)])
            p = (1 + int((null >= obs).sum())) / (n_perm + 1)
            rows.append(dict(
                scope=scope, quadrant=q, n_pairs=len(sub), n_variants=len(uniq),
                frac_matched=obs, null_mean=float(null.mean()),
                null_q025=float(np.quantile(null, 0.025)),
                null_q975=float(np.quantile(null, 0.975)),
                enrichment=obs / null.mean() if null.mean() > 0 else np.nan,
                p_permutation=p, n_perm=n_perm))
            qc.add(f"{scope} {q}", f"{obs:.3f} matched against {null.mean():.3f} expected "
                                   f"({obs / max(null.mean(), 1e-12):.2f}x, p = {p:.3g})")
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Active-allele quantile: are these variants in real regulatory elements
# ──────────────────────────────────────────────────────────────────────────────

def aaq_by_quadrant(uni: pd.DataFrame, long: pd.DataFrame,
                    qc: QCReport) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accessibility of the element each variant sits in, by quadrant.

    A large predicted effect in a barely accessible region is the obvious alternative
    reading of Q3. The active-allele quantile places the variant's accessible allele in
    the genome-wide distribution of predicted accessibility, so it separates a large
    change in a real element from a large change in noise. Taken at the cell type where
    the variant's |logFC| is largest, which is defined for every variant whether or not
    it is active anywhere.

    Population: the quantile is read off the long table restricted to rows where the
    variant is ACCESSIBLE, so a variant accessible in no cell type has no value and is
    absent. Every group is therefore accessible-only already, and no separate
    accessible-only re-run is needed; the group sizes are the accessible members of each
    quadrant, not the full quadrants.

    All six pairs of quadrants are compared with a two-sided Mann-Whitney U test, with
    the rank-biserial correlation as the effect size and Benjamini-Hochberg across the
    six, written to tables/quadrant_aaq_tests.csv. A rank test rather than a difference
    of means because the quantile is bounded on [0, 1] and left-skewed in Q4.
    """
    col = C.ACCESSIBILITY_METRIC
    if col not in long.columns:
        qc.section("active-allele quantile")
        qc.note(f"    {col} not present in the long table; panel skipped.")
        return pd.DataFrame(), pd.DataFrame()

    acc = long.loc[long["accessible"].to_numpy(bool),
                   ["variant_idx", C.SIGNIF_METRIC, col]]
    top = acc.loc[acc.groupby("variant_idx", observed=True)[C.SIGNIF_METRIC].idxmax()]
    m = uni.merge(top[["variant_idx", col]], on="variant_idx", how="left")

    rows = []
    for q in ("Q1", "Q2", "Q3", "Q4"):
        v = m.loc[m["quadrant"] == q, col].dropna().to_numpy()
        if not v.size:
            continue
        rows.append(dict(quadrant=q, n=int(v.size), median=float(np.median(v)),
                         q25=float(np.quantile(v, 0.25)), q75=float(np.quantile(v, 0.75)),
                         q10=float(np.quantile(v, 0.10)), q90=float(np.quantile(v, 0.90)),
                         mean=float(v.mean())))
    out = pd.DataFrame(rows)

    # pairwise comparisons: six tests, one table, BH across the six
    vals = {q: m.loc[m["quadrant"] == q, col].dropna().to_numpy()
            for q in ("Q1", "Q2", "Q3", "Q4")}
    trows = []
    for qa, qb in combinations([q for q in ("Q1", "Q2", "Q3", "Q4") if vals[q].size], 2):
        x, y = vals[qa], vals[qb]
        u = stats.mannwhitneyu(x, y, alternative="two-sided")
        # rank-biserial: 2U/(n1*n2) - 1, the probability x exceeds y rescaled to [-1, 1]
        rb = 2.0 * float(u.statistic) / (x.size * y.size) - 1.0
        trows.append(dict(group_a=qa, group_b=qb, n_a=int(x.size), n_b=int(y.size),
                          median_a=float(np.median(x)), median_b=float(np.median(y)),
                          median_difference=float(np.median(x) - np.median(y)),
                          u_statistic=float(u.statistic),
                          rank_biserial=rb, p_mannwhitney=float(u.pvalue),
                          p_underflow=bool(u.pvalue == 0.0)))
    tests = pd.DataFrame(trows)
    if len(tests):
        from s4lib import benjamini_hochberg
        tests["q_value"] = benjamini_hochberg(tests["p_mannwhitney"].to_numpy())

    qc.section("active-allele quantile")
    qc.note("    population: accessible variants only, since the quantile is read from "
            "the long table where the variant is accessible. Group sizes are the "
            "accessible members of each quadrant.")
    for r in out.itertuples():
        qc.add(r.quadrant, f"median {r.median:.3f} [IQR {r.q25:.3f}, {r.q75:.3f}], n = {r.n:,}")
    return out, tests


# ──────────────────────────────────────────────────────────────────────────────
# Within credible sets
# ──────────────────────────────────────────────────────────────────────────────

def pip_gradient(uni: pd.DataFrame, edges: tuple[float, ...],
                 qc: QCReport) -> pd.DataFrame:
    """Fraction of variants active, in bins of fine-mapping posterior.

    The quadrant table splits the posterior at one value, and the obvious objection is
    that the partition is an artefact of that value. This reports the same relationship
    without a split: if model evidence tracks causal probability, the fraction of active
    variants should rise across the whole posterior range rather than stepping at 0.5.

    Reported twice. Over all variants, which is the quantity the quadrants summarise;
    and over accessible variants alone, because accessibility is the one property that
    could plausibly raise both posterior and activity at once, and holding it fixed shows
    whether the gradient survives.

    Trend is the linear-by-linear association test across ordered bins, which is the
    standard test for a monotone dose-response on a binary outcome; the bin proportions
    with their Wilson intervals are the result, and the trend statistic summarises them.
    """
    from statsmodels.stats.contingency_tables import Table
    from statsmodels.stats.proportion import proportion_confint

    labels = [f"{edges[k]:g}-{min(edges[k + 1], 1.0):g}" for k in range(len(edges) - 1)]
    b = np.digitize(uni["pip_max"].to_numpy(), edges[1:-1], right=False)

    rows = []
    qc.section("posterior gradient")
    for scope, mask in (("all", np.ones(len(uni), bool)),
                        ("accessible", uni["accessible_anywhere"].to_numpy(bool))):
        counts = []
        for k, lab in enumerate(labels):
            sel = mask & (b == k)
            n = int(sel.sum())
            if not n:
                continue
            a = int(uni.loc[sel, "active"].sum())
            lo, hi = proportion_confint(a, n, method="wilson")
            counts.append((n - a, a))
            rows.append(dict(scope=scope, pip_bin=lab, bin_index=k, n_variants=n,
                             n_active=a, frac_active=a / n,
                             ci_lo=float(lo), ci_hi=float(hi)))
        if len(counts) >= 3:
            r = Table(np.array(counts)).test_ordinal_association()
            for row in rows:
                if row["scope"] == scope:
                    row["trend_z"], row["trend_p"] = float(r.zscore), float(r.pvalue)
            qc.add(f"{scope}: trend across {len(counts)} bins",
                   f"z = {float(r.zscore):.1f}, p = {float(r.pvalue):.3g}")
        for row in rows:
            if row["scope"] == scope:
                qc.add(f"  {scope} {row['pip_bin']}",
                       f"{row['frac_active']:.4f} active "
                       f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}], n = {row['n_variants']:,}")
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="directory written by 01 (variants.parquet, variant_celltype.parquet)")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--pip-bed", type=Path, default=C.PIP_BED)
    ap.add_argument("--vep", type=Path, default=None,
                    help="VEP output: needs a variant key and a consequence column")
    ap.add_argument("--pip-high", type=float, default=C.QUADRANT_PIP_HIGH)
    ap.add_argument("--n-perm", type=int, default=C.N_TRAIT_PERMUTATIONS)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    args = ap.parse_args()

    inp = args.input_dir
    out = args.output_dir or inp
    (out / "tables").mkdir(parents=True, exist_ok=True)
    qc = QCReport("07_quadrants")

    banner("Loading")
    V = pd.read_parquet(inp / "variants.parquet")
    long = pd.read_parquet(inp / "variant_celltype.parquet")
    log(f"variants {len(V):,}; variant-cell-type rows {len(long):,}", level=1)

    pairs = join_pairs_to_variants(load_pairs(args.pip_bed, qc), V, qc)

    banner("The partition")
    uni = build_quadrants(V, pairs, args.pip_high, qc)
    counts = partition_table(uni)
    counts.to_csv(out / "tables/quadrant_counts.csv", index=False)
    for r in counts[counts["quadrant"] != "total"].itertuples():
        log(f"{r.quadrant}: {r.n_variants:>8,}  ({r.frac_of_universe:6.2%})", level=1)
    q1 = counts[counts["quadrant"] == "Q1"].iloc[0]
    log(f"Q1 odds ratio {q1.odds_ratio:.2f} [{q1.ci_lo:.2f}, {q1.ci_hi:.2f}], "
        f"p = {q1.p_fisher:.3g}", level=1)

    banner("Trait-lineage concordance")
    graph = set(C.universe("graph14"))
    act = long.loc[long[C.call_column(long)].to_numpy(bool), ["variant_idx", "cell_type"]]
    act = act[act["cell_type"].astype(str).isin(graph)]
    active_sets = {int(v): frozenset(map(str, g)) for v, g
                   in act.groupby("variant_idx", observed=True)["cell_type"]}
    tl = trait_lineage_concordance(uni, pairs, active_sets, args.n_perm, args.seed, qc)
    tl.to_csv(out / "tables/quadrant_trait_lineage.csv", index=False)
    for r in tl.itertuples():
        log(f"{r.scope:<7s} {r.quadrant}: {r.frac_matched:.3f} vs {r.null_mean:.3f} null "
            f"= {r.enrichment:.2f}x  (p = {r.p_permutation:.3g}, n = {r.n_pairs:,})", level=1)

    banner("Active-allele quantile")
    aaq, aaq_tests = aaq_by_quadrant(uni, long, qc)
    if len(aaq):
        aaq.to_csv(out / "tables/quadrant_aaq.csv", index=False)
        if len(aaq_tests):
            aaq_tests.to_csv(out / "tables/quadrant_aaq_tests.csv", index=False)
        for r in aaq.itertuples():
            log(f"{r.quadrant}: median {r.median:.3f} [{r.q25:.3f}, {r.q75:.3f}]", level=1)

    banner("Posterior gradient")
    grad = pip_gradient(uni, C.PIP_GRADIENT_BINS, qc)
    grad.to_csv(out / "tables/pip_gradient.csv", index=False)
    for r in grad.itertuples():
        log(f"{r.scope:<11s} PP {r.pip_bin:<14s} {r.frac_active:7.4f} active "
            f"[{r.ci_lo:.4f}, {r.ci_hi:.4f}]  n = {r.n_variants:>7,}", level=1)

    if args.vep is not None:
        banner("Consequence classes")
        vep = pd.read_csv(args.vep, sep="\t")
        keyc = next((c for c in ("variant_idx", "variant_key", "variant_id", "rsid", "Uploaded_variation")
                     if c in vep.columns), None)
        consc = next((c for c in ("consequence", "Consequence", "most_severe_consequence")
                      if c in vep.columns), None)
        if keyc is None or consc is None:
            raise KeyError(f"{args.vep}: need a variant key and a consequence column; "
                           f"found {list(vep.columns)[:12]}")
        vep = vep.rename(columns={consc: "consequence"})
        if keyc != "variant_idx":
            Vk = V.copy()
            if "variant_key" not in Vk.columns:
                # built the same way vep_prepare.py built the identifier it sent to VEP
                Vk["variant_key"] = variant_key(Vk, C.VARIANT_KEY_COLUMNS)
            if keyc not in Vk.columns:
                raise KeyError(f"{args.vep}: key column {keyc!r} is not in variants.parquet; "
                               f"available keys there: "
                               f"{[c for c in ('variant_key', 'variant_id', 'rsid') if c in Vk.columns]}")
            before = vep[keyc].nunique()
            vep = vep.merge(Vk[["variant_idx", keyc]], on=keyc, how="inner")
            qc.add("VEP variants joined to the variant table",
                   f"{vep['variant_idx'].nunique():,} of {before:,}")
            if vep.empty:
                raise ValueError(f"{args.vep}: no VEP row joined on {keyc!r}. The "
                                 f"identifier column written by vep_prepare.py must be "
                                 f"the one VEP echoes back as Uploaded_variation.")
        vep = vep[["variant_idx", "consequence"]].drop_duplicates()
        ve = vep_enrichment(uni, vep[["variant_idx", "consequence"]], C.VEP_MIN_COUNT, qc)
        if len(ve):
            ve.to_csv(out / "tables/quadrant_vep_enrichment.csv", index=False)
            for r in ve[ve["q_value"] < 0.05].head(12).itertuples():
                log(f"{r.quadrant} {r.consequence:<34s} OR {r.odds_ratio:6.2f} "
                    f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]  n = {r.n_in_quadrant}", level=1)
    else:
        log("no --vep given; consequence-class enrichment skipped", level=1)

    qc.write(out / "quadrant_qc.txt")
    log(f"QC -> {out / 'quadrant_qc.txt'}")
    log("done")


if __name__ == "__main__":
    sys.exit(main())
