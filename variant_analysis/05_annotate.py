#!/usr/bin/env python
"""
05_annotate.py: annotation and external validation. Builds the supplementary variant
tables and tests whether nominated variants are enriched among fine-mapped variants.

Allele frequency is carried as an annotation column only; nothing is tested against it.
A variant's allele frequency and the number of cell types it is called in both depend on
how long the allele has been segregating and how much of the genome is accessible around
it, so a correlation between them reflects that shared dependence rather than a
regulatory relationship.

Scope
-----
Both output tables are restricted to the nominated set (variants called in at least one
cell type), so a tabix-indexed VCF can be queried per variant without the runtime
becoming a problem.

Allele-frequency sources
------------------------
The source is given with --gnomad-sites (or config.GNOMAD_SITES) and can be either:

  1. A sites table (.tsv/.tsv.gz/.parquet) with chromosome, position, ref, alt and one or
     more AF columns. Bulk inner merge on the four-part key; fast, no extra dependencies.
  2. A tabix-indexed VCF/BCF. Requires pysam and one random-access fetch per nominated
     variant.

If neither is supplied the tables are written without AF columns and the QC report says
so; no result depends on allele frequency.

Not recomputed here
-------------------
Nearest gene, TSS distance, LOEUF, PIP, trait list and the quadrant label are joined in
01_build_matrix.py and read from the checkpoint.

Outputs
-------
  tables/flagship_variants.csv          the flagship tier, one row per variant
  tables/nominated_annotated.csv        the full nominated set with the same columns
  annotate_qc.txt                       what joined, what did not, and coverage rates

Usage:
  python 05_annotate.py --input-dir out
  python 05_annotate.py --input-dir out --gnomad-sites gnomad_af.tsv.gz
  python 05_annotate.py --input-dir out --gnomad-sites gnomad.genomes.sites.vcf.bgz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C                               # noqa: E402
from s4lib import QCReport, banner, log          # noqa: E402


# ── allele-frequency readers ─────────────────────────────────────────────────

AF_ALIASES = {
    "af": "gnomad_af", "AF": "gnomad_af", "af_global": "gnomad_af",
    "AF_nfe": "gnomad_af_nfe", "af_nfe": "gnomad_af_nfe",
    "AF_NFE": "gnomad_af_nfe",
}
KEY = ["chr", "pos", "allele1", "allele2"]


def _norm_chr(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    return np.where(s.str.startswith("chr"), s, "chr" + s)


def read_af_table(path: Path, qc: QCReport) -> pd.DataFrame | None:
    """Bulk read of a sites TABLE. Column names are matched case-insensitively."""
    if path.suffix == ".parquet":
        raw = pd.read_parquet(path)
    else:
        # A VCF-derived sites table carries '##' meta lines and a header line that
        # starts with a single '#' ('#CHROM'). comment='#' would eat the header and
        # promote the first data row into it, so the meta lines are counted first and
        # the leading '#' is stripped from the column name afterwards.
        import gzip
        opener = gzip.open if str(path).endswith((".gz", ".bgz")) else open
        n_meta = 0
        with opener(path, "rt") as fh:
            for line in fh:
                if line.startswith("##"):
                    n_meta += 1
                else:
                    break
        raw = pd.read_csv(path, sep="\t", skiprows=n_meta, low_memory=False)
        raw.columns = [c.lstrip("#") for c in raw.columns]
    cols = {c.lower(): c for c in raw.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_chr = pick("chr", "chrom", "chromosome", "#chrom")
    c_pos = pick("pos", "position", "start")
    c_ref = pick("ref", "allele1", "a1")
    c_alt = pick("alt", "allele2", "a2")
    if None in (c_chr, c_pos, c_ref, c_alt):
        qc.add("gnomAD table", f"REJECTED {path.name}: could not find chr/pos/ref/alt "
                               f"among {list(raw.columns)[:12]}")
        return None

    af_cols = {c: AF_ALIASES[c] for c in raw.columns if c in AF_ALIASES}
    if not af_cols:
        af_cols = {c: f"gnomad_{c.lower()}" for c in raw.columns
                   if c.lower().startswith("af")}
    if not af_cols:
        qc.add("gnomAD table", f"REJECTED {path.name}: no AF column found")
        return None

    out = pd.DataFrame({
        "chr": _norm_chr(raw[c_chr]),
        "pos": raw[c_pos].astype("int64"),
        "allele1": raw[c_ref].astype(str).str.upper(),
        "allele2": raw[c_alt].astype(str).str.upper(),
    })
    for src, dst in af_cols.items():
        out[dst] = pd.to_numeric(raw[src], errors="coerce")
    # A sites file can carry several rows per position (multiallelic split). Keep the
    # row matching the exact allele pair; duplicates on the full key are collapsed by
    # taking the first, which is what a split-multiallelic file gives anyway.
    n0 = len(out)
    out = out.drop_duplicates(subset=KEY, keep="first")
    qc.add("gnomAD sites table", f"{path.name}: {n0:,} rows, {len(out):,} unique keys, "
                                 f"AF columns {sorted(af_cols.values())}")
    return out


def read_af_vcf(path: Path, variants: pd.DataFrame, qc: QCReport) -> pd.DataFrame | None:
    """One tabix fetch per nominated variant. Requires pysam and an index."""
    try:
        import pysam
    except ImportError:
        qc.add("gnomAD VCF", f"SKIPPED {path.name}: pysam is not installed. Either "
                             "install pysam or pre-extract a sites table with "
                             "`bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\t%INFO/AF"
                             "\\t%INFO/AF_nfe\\n'` and pass that instead.")
        return None
    try:
        vcf = pysam.VariantFile(str(path))
    except Exception as exc:                       # noqa: BLE001 - reported, not raised
        qc.add("gnomAD VCF", f"SKIPPED {path.name}: could not open ({exc})")
        return None

    contigs = set(vcf.header.contigs)
    use_chr_prefix = any(c.startswith("chr") for c in contigs)
    rows, missing = [], 0
    for i, (ch, pos, a1, a2) in enumerate(
            zip(variants["chr"], variants["pos"],
                variants["allele1"], variants["allele2"])):
        contig = ch if use_chr_prefix else ch.removeprefix("chr")
        hit = None
        try:
            for rec in vcf.fetch(contig, pos - 1, pos):
                if rec.pos != pos or rec.ref.upper() != a1.upper():
                    continue
                for k, alt in enumerate(rec.alts or ()):
                    if alt.upper() == a2.upper():
                        hit = (rec, k)
                        break
                if hit:
                    break
        except ValueError:                         # contig absent from the index
            pass
        if hit is None:
            missing += 1
            continue
        rec, k = hit
        row = {"chr": ch, "pos": pos, "allele1": a1, "allele2": a2}
        for field, dst in (("AF", "gnomad_af"), ("AF_nfe", "gnomad_af_nfe")):
            val = rec.info.get(field)
            if isinstance(val, (tuple, list)):
                val = val[k] if k < len(val) else np.nan
            row[dst] = float(val) if val is not None else np.nan
        rows.append(row)
        if (i + 1) % 2000 == 0:
            log(f"gnomAD fetch {i + 1:,}/{len(variants):,}", level=1)
    vcf.close()
    qc.add("gnomAD VCF", f"{path.name}: {len(rows):,} of {len(variants):,} nominated "
                         f"variants found, {missing:,} absent")
    return pd.DataFrame(rows) if rows else None


# ── per-variant cell-type summaries ──────────────────────────────────────────

def summarise_celltypes(long: pd.DataFrame, graph: dict[str, set[str]],
                        dist: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long table to one row per variant, describing WHERE it acts."""
    sig = long[long[C.call_column(long)]].copy()
    if not len(sig):
        return pd.DataFrame(columns=["variant_idx"])

    # strongest cell type and the direction there
    sig = sig.sort_values("abs_logfc", ascending=False)
    top = sig.groupby("variant_idx").first()

    def joined(s):
        return ";".join(sorted(s))

    agg = sig.groupby("variant_idx").agg(
        n_significant_celltypes=("cell_type", "size"),
        significant_celltypes=("cell_type", joined),
        max_abs_logfc=("abs_logfc", "max"),
        min_pval=("pval", "min"),
        min_qval=("qval", "min"))
    agg["top_celltype"] = top["cell_type"]
    agg["top_logfc"] = top["logfc"]
    agg["direction"] = np.where(top["logfc"] >= 0, "gain", "loss")

    # sign concordance across the significant set: a variant that gains accessibility in
    # one cell type and loses it in another is a different object from one that does the
    # same thing everywhere, and the flagship table should say which it is.
    signs = sig.groupby("variant_idx")["logfc"].agg(
        lambda s: "concordant" if (s >= 0).all() or (s < 0).all() else "mixed")
    agg["sign_pattern"] = signs

    # where the significant set sits on the lineage graph
    nodes = set(graph)
    conn, mean_d, n_graph = [], [], []
    for vidx, grp in sig.groupby("variant_idx")["cell_type"]:
        ms = set(grp) & nodes
        n_graph.append(len(ms))
        if len(ms) <= 1:
            conn.append(len(ms) == 1)
            mean_d.append(np.nan)
            continue
        start = next(iter(ms)); seen = {start}; stack = [start]
        while stack:
            for nb in graph[stack.pop()] & ms:
                if nb not in seen:
                    seen.add(nb); stack.append(nb)
        conn.append(len(seen) == len(ms))
        ds = [dist.loc[a, b] for a in ms for b in ms if a < b]
        mean_d.append(float(np.mean(ds)) if ds else np.nan)
    agg["n_significant_graph_nodes"] = n_graph
    agg["significant_set_connected"] = conn
    agg["mean_pairwise_graph_distance"] = mean_d
    return agg.reset_index()


MATCH_VAR = "n_accessible"          # cell types in which the variant is accessible


def pip_validation(V: pd.DataFrame, long: pd.DataFrame, qc: QCReport,
                   out: Path) -> pd.DataFrame:
    """
    Orthogonal validation: does nomination enrich for fine-mapped variants?

    Fine-mapping posteriors come from GWAS and are independent of the models, so
    credible-set membership is an external label. Other analyses here are computed from the
    same score matrix that defines significance and can characterise the nominated set but
    cannot test whether it is enriched for causal variants.

    The posterior is used as an endpoint rather than a filter. Only a minority of nominated
    variants carry a posterior, so filtering on it would leave too few variants for the
    coherence and breadth analyses, whereas as an endpoint the comparison set is the whole
    remaining testable universe.

    Absence of a posterior
    ----------------------
    A variant with no posterior is absent from the PP >= 0.001 credible-set file. That
    covers three situations: it lies outside any credible set, it lies inside one but below
    that file's PP threshold, or it was fine-mapped for a trait outside the panel. Absence
    is therefore not a posterior of zero; it means "not present in a PP >= 0.001 credible
    set". The provenance breakdown reports how much of the absence is explained by variants
    that never entered the GWAS, the only case that behaves like a true zero.

    Matching on accessibility
    -------------------------
    A variant cannot be nominated unless it is accessible somewhere, and accessibility tracks
    regulatory annotation, which itself predicts credible-set membership. Without matching,
    part of any enrichment would reflect nominated variants sitting in open chromatin.
    Comparisons are therefore stratified on the number of accessible cell types, pooled with
    Mantel-Haenszel; Tarone's test checks that the odds ratio is homogeneous enough to pool.

    Comparison with a magnitude-only rule
    -------------------------------------
    Enrichment alone shows only that the rule beats a random draw from accessible chromatin.
    The rule is therefore also compared with a fixed |logFC| cut. Because that cut calls
    more variants, it is also run size-matched (top N by the same statistic, N being the
    size of the primary set) so precision is compared at equal set size.
    """
    from statsmodels.stats.contingency_tables import StratifiedTable
    from scipy import stats

    # Variants that fall only in the consensus peak set are testable but accessible in no
    # single cell type, so neither rule can call them (both operate within accessible
    # contexts). They form a stratum with no called variants, which Mantel-Haenszel
    # discards, but they would dilute the descriptive control rate. They are excluded so
    # the descriptive rates and the stratified estimate describe the same population.
    uni = V.loc[V["testable"]].copy()
    if MATCH_VAR in uni.columns:
        _elig = uni[MATCH_VAR].fillna(0) >= 1
        qc.add("validation universe: testable", f"{len(uni):,}")
        qc.add("  excluded, accessible in no cell type so never eligible",
               f"{int((~_elig).sum()):,}")
        qc.add("  eligible universe", f"{int(_elig.sum()):,}")
        uni = uni.loc[_elig].copy()
    if not len(uni) or MATCH_VAR not in uni.columns:
        log("no testable universe; skipping PIP validation", level=1)
        return pd.DataFrame()

    uni["in_credible_set"] = uni["PIP"].notna()
    uni["pip_high"] = uni["PIP"].fillna(0.0) >= C.FLAGSHIP_PIP

    # The magnitude rule is recomputed from the long table over cell types where the
    # variant is accessible, rather than read from the panel-wide maximum, which would
    # include contexts where the variant cannot act.
    mx = (long.loc[long["accessible"]].groupby("variant_idx")["abs_logfc"]
          .max().rename("max_abs_accessible"))
    uni = uni.merge(mx, left_on="variant_idx", right_index=True, how="left")

    uni["rule_primary"] = uni[C.breadth_column(uni)] >= 1
    uni["rule_magnitude"] = uni["max_abs_accessible"] >= C.MAGNITUDE_ONLY_THRESH
    n_primary = int(uni["rule_primary"].sum())
    uni["rule_magnitude_matched"] = False
    if n_primary:
        top = uni["max_abs_accessible"].nlargest(n_primary).index
        uni.loc[top, "rule_magnitude_matched"] = True

    rules = [("primary: %s" % C.primary_rule()[2], "rule_primary"),
             ("magnitude only: |logFC| >= %g" % C.MAGNITUDE_ONLY_THRESH, "rule_magnitude"),
             ("magnitude only, size-matched to the primary set",
              "rule_magnitude_matched")]
    endpoints = [("credible-set membership (PP >= 0.001)", "in_credible_set"),
                 ("PIP >= %g" % C.FLAGSHIP_PIP, "pip_high")]

    # Two universes. The default counts absence of a posterior as a non-hit, which
    # mixes "outside every credible set" with "never entered the GWAS". Restricting to
    # variants that carry a posterior removes that, at the cost of a much smaller
    # comparison set. Only the PIP threshold endpoint is meaningful there, since
    # credible-set membership holds for every member by construction.
    universes = [("all eligible", pd.Series(True, index=uni.index))]
    if "in_credible_set" in uni.columns:
        universes.append(("posterior-carrying", uni["in_credible_set"].astype(bool)))

    rows, strata_rows = [], []
    for ulab, umask in universes:
        uni_u = uni.loc[umask.to_numpy()].copy()
        log(f"universe '{ulab}': {len(uni_u):,} variants", level=1)
        for rlab, rcol in rules:
            for elab, ecol in endpoints:
                if ulab == "posterior-carrying" and ecol == "in_credible_set":
                    continue   # degenerate: every variant here is a hit
                tabs = []
                for s, g in uni_u.groupby(MATCH_VAR):
                    a = int((g[rcol] & g[ecol]).sum())
                    b = int((g[rcol] & ~g[ecol]).sum())
                    c = int((~g[rcol] & g[ecol]).sum())
                    d = int((~g[rcol] & ~g[ecol]).sum())
                    strata_rows.append(dict(
                        universe=ulab, rule=rlab, endpoint=elab, n_accessible=int(s),
                        n_called=a + b, n_called_hit=a, n_control=c + d, n_control_hit=c,
                        rate_called=a / (a + b) if a + b else np.nan,
                        rate_control=c / (c + d) if c + d else np.nan))
                    # A stratum contributes to the pooled odds ratio only if both margins are
                    # non-degenerate. Degenerate strata are kept in the per-stratum table --
                    # they are part of the design and a reader should see how many there are.
                    if (a + b) and (c + d) and (a + c) and (b + d):
                        tabs.append(np.array([[a, b], [c, d]]))

                n_called = int(uni_u[rcol].sum())
                r_called = float(uni_u.loc[uni_u[rcol], ecol].mean()) if n_called else np.nan
                r_ctrl = float(uni_u.loc[~uni_u[rcol], ecol].mean())
                rec = dict(universe=ulab, rule=rlab, endpoint=elab, match_variable=MATCH_VAR,
                           n_called=n_called, n_control=int((~uni_u[rcol]).sum()),
                           rate_called=r_called, rate_control=r_ctrl,
                           crude_fold=r_called / r_ctrl if r_ctrl else np.nan,
                           n_strata_informative=len(tabs),
                           n_strata_total=uni_u[MATCH_VAR].nunique())
                if len(tabs) >= 2:
                    st = StratifiedTable(tabs)
                    lo, hi = st.oddsratio_pooled_confint()
                    rec.update(mh_odds_ratio=float(st.oddsratio_pooled),
                               ci_lo=float(lo), ci_hi=float(hi),
                               p_mh=float(st.test_null_odds().pvalue),
                               p_mh_underflow=bool(st.test_null_odds().pvalue == 0.0),
                               p_tarone_homogeneity=float(st.test_equal_odds().pvalue))
                    log(f"{rlab} -> {elab}: MH OR = {st.oddsratio_pooled:.2f} "
                        f"[{lo:.2f}, {hi:.2f}], p = "
                        + ("< 1e-300" if st.test_null_odds().pvalue == 0.0
                           else f"{st.test_null_odds().pvalue:.3g}") + " "
                        f"({len(tabs)} strata)", level=1)
                else:
                    rec.update(mh_odds_ratio=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                               p_mh=np.nan, p_tarone_homogeneity=np.nan)
                    log(f"{rlab} -> {elab}: too few informative strata to pool", level=1)
                rows.append(rec)

    val = pd.DataFrame(rows)
    val.to_csv(out / "tables/pip_validation.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(out / "tables/pip_validation_strata.csv",
                                     index=False)

    # ── head-to-head at equal set size ───────────────────────────────────────
    h2h = []
    A, B = uni["rule_primary"], uni["rule_magnitude_matched"]
    for elab, ecol in endpoints:
        only_a = int((A & ~B & uni[ecol]).sum())
        only_b = int((~A & B & uni[ecol]).sum())
        n_disc = only_a + only_b
        p = float(stats.binomtest(only_a, n_disc, 0.5).pvalue) if n_disc else np.nan
        h2h.append(dict(endpoint=elab, n_called_each=n_primary,
                        n_called_by_both=int((A & B).sum()),
                        hits_primary_only=only_a, hits_magnitude_only=only_b,
                        p_exact_binomial=p))
        log(f"head-to-head, {elab}: {only_a} vs {only_b} among discordant calls, "
            f"p = {p:.3g}", level=1)
    pd.DataFrame(h2h).to_csv(out / "tables/pip_validation_headtohead.csv", index=False)

    # ── what absence of a posterior actually means ───────────────────────────
    prov = []
    if "is_vuckovic" in uni.columns:
        for slab, mask in (("all testable", pd.Series(True, index=uni.index)),
                           ("nominated", uni["rule_primary"])):
            g = uni.loc[mask]
            for plab, pm in (("GWAS-panel provenance", g["is_vuckovic"]),
                             ("background provenance", ~g["is_vuckovic"])):
                sub = g.loc[pm]
                prov.append(dict(subset=slab, provenance=plab, n=len(sub),
                                 frac_of_subset=len(sub) / len(g) if len(g) else np.nan,
                                 frac_in_credible_set=float(sub["in_credible_set"].mean())
                                 if len(sub) else np.nan))
        pd.DataFrame(prov).to_csv(out / "tables/pip_provenance.csv", index=False)

    # ── QC ───────────────────────────────────────────────────────────────────
    qc.section("orthogonal validation: fine-mapping posterior, accessibility-matched")
    qc.add("universe", f"{len(uni):,} testable variants")
    qc.add("matching variable", f"{MATCH_VAR}, "
           f"{uni[MATCH_VAR].nunique()} strata, Mantel-Haenszel pooled")
    qc.add("endpoint 1 prevalence in the universe",
           f"{uni['in_credible_set'].mean():.3%} in a PP >= 0.001 credible set")
    qc.add("endpoint 2 prevalence in the universe",
           f"{uni['pip_high'].mean():.4%} with PIP >= {C.FLAGSHIP_PIP:g}")
    for _, r in val.iterrows():
        if np.isfinite(r["mh_odds_ratio"]):
            qc.add(f"{r['rule']} -> {r['endpoint']}",
                   f"MH OR {r['mh_odds_ratio']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}], "
                   f"p = {r['p_mh']:.3g}; crude {r['rate_called']:.2%} vs "
                   f"{r['rate_control']:.3%}; {int(r['n_strata_informative'])}/"
                   f"{int(r['n_strata_total'])} strata informative")
            if r["p_tarone_homogeneity"] < 0.05:
                qc.add(f"  homogeneity, {r['rule']} -> {r['endpoint']}",
                       f"Tarone p = {r['p_tarone_homogeneity']:.3g}; the odds ratio "
                       f"varies across strata, so the pooled value is an average and "
                       f"the per-stratum table should be read alongside it")
    for h in h2h:
        qc.add(f"head-to-head, {h['endpoint']}",
               f"{h['hits_primary_only']} primary-only vs "
               f"{h['hits_magnitude_only']} magnitude-only among discordant calls "
               f"(exact binomial p = {h['p_exact_binomial']:.3g}); "
               f"{h['n_called_by_both']} variants called by both")
    for p in prov:
        qc.add(f"{p['subset']}, {p['provenance']}",
               f"n = {p['n']:,} ({p['frac_of_subset']:.1%}), "
               f"{p['frac_in_credible_set']:.1%} in a credible set")
    qc.note("    Absence of a posterior is not a posterior of zero. It means the variant "
            "is not in the PP >= 0.001 credible-set file, which happens for variants "
            "outside any credible set, variants inside one but below that threshold, and "
            "variants fine-mapped for a trait outside the panel. Report it as lack of "
            "credible-set membership. The provenance rows above separate the first case "
            "from the other two: background-provenance variants never entered the GWAS, "
            "so for those absence does behave like a zero.")
    qc.note("    The size-matched magnitude rule breaks ties in the magnitude statistic "
            "arbitrarily, so its membership is not unique when ties span the cutoff. It "
            "is a comparator, not a nomination.")
    qc.note("    This is an enrichment, not a claim about individual variants. A high "
            "odds ratio does not mean a nominated variant without a posterior is causal; "
            "it means the rule recovers credible-set variants at a higher rate than "
            "accessibility alone predicts.")
    return val


def plot_validation(val: pd.DataFrame, out: Path) -> None:
    """Forest of the pooled odds ratios, with the per-stratum rates beside it."""
    import matplotlib.pyplot as plt
    import s4plot
    if not len(val) or not np.isfinite(val["mh_odds_ratio"]).any():
        log("no pooled odds ratio to plot", level=1)
        return
    s4plot.apply_style()
    tick = plt.rcParams["xtick.labelsize"]
    strata = pd.read_csv(out / "tables/pip_validation_strata.csv")

    fig, axes = plt.subplots(1, 2, figsize=(s4plot.FIG_W_DOUBLE, 3.0),
                             gridspec_kw=dict(width_ratios=[1.4, 1.0], wspace=0.55))
    ax = axes[0]
    v = val[np.isfinite(val["mh_odds_ratio"])].reset_index(drop=True)
    ys = np.arange(len(v))[::-1]
    for y, (_, r) in zip(ys, v.iterrows()):
        focal = r["rule"].startswith("primary")
        col = s4plot.FOCAL if focal else s4plot.COMPARATOR
        ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=col, lw=1.5, zorder=2)
        ax.plot([r["mh_odds_ratio"]], [y], "o", color=col,
                ms=5.5 if focal else 4.5, zorder=3)
        ax.text(r["ci_hi"] * 1.2, y, f"{r['mh_odds_ratio']:.1f}x", va="center",
                ha="left", fontsize=tick, color=col)
    ax.axvline(1.0, color=s4plot.LIGHT_GREY, lw=0.9, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['rule'].split(':')[0]}\n{r['endpoint']}"
                        for _, r in v.iterrows()], fontsize=tick - 0.5)
    ax.set_ylim(-0.7, len(v) - 0.3)
    ax.set_xlabel("odds ratio vs non-nominated variants at equal\n"
                  "accessibility breadth (Mantel-Haenszel)")
    hi_lim = float(np.nanmax(v["ci_hi"])) * 4.0
    ax.set_xlim(0.6, hi_lim)
    # Odds ratios read as numbers, not as powers of ten.
    ticks = [t for t in (0.6, 1, 2, 3, 5, 10, 20, 50, 100, 200) if 0.6 <= t <= hi_lim]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    s4plot.panel_letter(ax, "a", dx=-0.62)

    ax = axes[1]
    s = strata[strata["endpoint"].str.startswith("credible-set")
               & strata["rule"].str.startswith("primary")
               & (strata["n_called"] > 0)].sort_values("n_accessible")
    ax.plot(s["n_accessible"], s["rate_control"], "o-", color=s4plot.COMPARATOR,
            ms=3.5, lw=1.2, label="not nominated")
    ax.plot(s["n_accessible"], s["rate_called"], "o-", color=s4plot.FOCAL,
            ms=4.5, lw=1.6, label="nominated")
    ax.set_xlabel("cell types in which the\nvariant is accessible")
    ax.set_ylabel("fraction in a PP $\\geq$ 0.001\ncredible set")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=tick, loc="best")
    s4plot.panel_letter(ax, "b", dx=-0.34)

    s4plot.verify_no_overlap(fig)
    for f in s4plot.save(fig, out / "figures/pip_validation"):
        log(f"figure -> {f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=C.OUTPUT_DIR,
                    help="directory holding the 01 checkpoint")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="defaults to --input-dir")
    ap.add_argument("--gnomad-sites", type=Path, default=C.GNOMAD_SITES,
                    help="sites table (.tsv/.tsv.gz/.parquet) or tabix-indexed VCF/BCF")
    args = ap.parse_args()

    inp = args.input_dir
    out = args.output_dir or inp
    (out / "tables").mkdir(parents=True, exist_ok=True)
    qc = QCReport("Annotation QC")

    banner("Reading checkpoint")
    V = pd.read_parquet(inp / "variants.parquet")
    long = pd.read_parquet(inp / "variant_celltype.parquet")
    log(f"variants {len(V):,}; variant-cell-type rows {len(long):,}", level=1)

    # ── graph, for the lineage columns ───────────────────────────────────────
    graph_nodes = list(C.universe("graph14"))
    graph = {x: set() for x in graph_nodes}
    for a, b in C.LINEAGE_EDGES_DIRECTED:
        if a in graph and b in graph:
            graph[a].add(b); graph[b].add(a)
    dist = pd.read_csv(inp / "tables/graph_distances.csv", index_col=0) \
        if (inp / "tables/graph_distances.csv").exists() else None
    if dist is None:
        log("graph_distances.csv absent (run 04 first); distance column will be NaN",
            level=1)
        dist = pd.DataFrame(np.nan, index=graph_nodes, columns=graph_nodes)

    # ── the two tiers ───────────────────────────────────────────────────────
    banner("Nomination tiers")
    nominated_idx = set(long.loc[long[C.call_column(long)], "variant_idx"])
    nom = V[V["variant_idx"].isin(nominated_idx)].copy()

    # The flagship P criterion is evaluated over accessible cell types only, matching the
    # rule that defines activity. `min_pval_panel` is the minimum over every cell type,
    # so using it would admit variants whose only sub-threshold P lies in a context where
    # the region is not open.
    min_p_acc = (long.loc[long["accessible"]]
                 .groupby("variant_idx")["pval"].min().rename("min_pval_accessible"))
    nom = nom.merge(min_p_acc, on="variant_idx", how="left")
    both = nom["PIP"] >= C.FLAGSHIP_PIP
    flag = nom[(nom["min_pval_accessible"] < C.FLAGSHIP_P) & both].copy()

    # Variants the panel-wide minimum would have admitted and the accessible minimum does
    # not: exported so the difference between the two definitions is auditable.
    dropped = nom[(nom["min_pval_panel"] < C.FLAGSHIP_P)
                  & ~(nom["min_pval_accessible"] < C.FLAGSHIP_P) & both].copy()
    if len(dropped):
        keep = [c for c in ("variant_idx", "rsid", "chr", "pos", "allele1", "allele2",
                            "PIP", "min_pval_panel", "min_pval_accessible",
                            "n_accessible", C.breadth_column(nom), "nearest_gene",
                            "tss_distance", C.CONSTRAINT_METRIC) if c in dropped.columns]
        dropped[keep].to_csv(out / "tables/flagship_excluded_inaccessible.csv", index=False)

    log(f"nominated {len(nom):,}; flagship {len(flag):,} "
        f"(accessible P < {C.FLAGSHIP_P}, PIP >= {C.FLAGSHIP_PIP}); "
        f"{len(dropped)} excluded by the accessibility restriction", level=1)
    qc.section("tiers")
    qc.add("nominated variants", f"{len(nom):,}")
    qc.add("active and PIP >= %.1f (both methods)" % C.FLAGSHIP_PIP, f"{int(both.sum()):,}")
    qc.add("flagship variants",
           f"{len(flag):,} (accessible-only P < {C.FLAGSHIP_P} and PIP >= {C.FLAGSHIP_PIP})")
    qc.add("excluded by the accessibility restriction",
           f"{len(dropped):,} (sub-threshold P only in a cell type where not accessible)")
    assert flag["variant_idx"].isin(nom.loc[both, "variant_idx"]).all(), \
        "flagship must be a subset of the variants both methods nominate"

    # ── cell-type summary ───────────────────────────────────────────────────
    banner("Per-variant cell-type summary")
    summ = summarise_celltypes(long[long["variant_idx"].isin(nominated_idx)],
                               graph, dist)
    log(f"summarised {len(summ):,} nominated variants", level=1)

    # ── allele frequency ────────────────────────────────────────────────────
    banner("Allele frequency")
    qc.section("allele frequency")
    af = None
    if args.gnomad_sites is None:
        log("no --gnomad-sites given; AF columns omitted", level=1)
        qc.add("source", "none supplied; AF columns omitted from both tables")
    elif not Path(args.gnomad_sites).exists():
        log(f"{args.gnomad_sites} not found; AF columns omitted", level=1)
        qc.add("source", f"{args.gnomad_sites} does not exist; AF columns omitted")
    else:
        path = Path(args.gnomad_sites)
        is_vcf = any(str(path).endswith(s) for s in
                     (".vcf", ".vcf.gz", ".vcf.bgz", ".bcf"))
        af = (read_af_vcf(path, nom[KEY], qc) if is_vcf
              else read_af_table(path, qc))
        if af is not None:
            log(f"allele frequencies for {len(af):,} keys", level=1)

    # ── assemble ────────────────────────────────────────────────────────────
    banner("Assembling tables")
    carried = ["variant_idx", "variant_id", "chr", "pos", "allele1", "allele2", "rsid",
               "PIP", "traits", "quadrant", "nearest_gene", "tss_distance",
               "nearest_gene_pc", "tss_distance_pc", C.CONSTRAINT_METRIC,
               "in_consensus_peak", "n_accessible", "n_significant", "n_called", "testable",
               "min_pval_accessible", "min_pval_panel"]
    carried = [c for c in carried if c in nom.columns]

    def build(base: pd.DataFrame) -> pd.DataFrame:
        t = base[carried].merge(summ, on="variant_idx", how="left")
        # Four-part key as first column; variant_id holds provenance, not an identifier.
        t.insert(0, "variant_key", t["chr"].astype(str) + ":" + t["pos"].astype(str)
                 + ":" + t["allele1"].astype(str) + ":" + t["allele2"].astype(str))
        if af is not None:
            n0 = len(t)
            t = t.merge(af, on=KEY, how="left")
            assert len(t) == n0, f"AF merge changed row count: {n0} -> {len(t)}"
        return t.sort_values(["min_pval", "PIP"], ascending=[True, False])

    nom_tab = build(nom)
    flag_tab = build(flag)
    nom_tab.to_csv(out / "tables/nominated_annotated.csv", index=False)
    flag_tab.to_csv(out / "tables/flagship_variants.csv", index=False)
    log(f"tables/nominated_annotated.csv  {len(nom_tab):,} rows x {nom_tab.shape[1]} cols")
    log(f"tables/flagship_variants.csv    {len(flag_tab):,} rows x {flag_tab.shape[1]} cols")

    # ── coverage ────────────────────────────────────────────────────────────
    qc.section("annotation coverage, flagship table")
    if len(flag_tab):
        for col in ("rsid", "nearest_gene", C.CONSTRAINT_METRIC, "gnomad_af",
                    "gnomad_af_nfe", "traits"):
            if col in flag_tab.columns:
                frac = flag_tab[col].notna().mean()
                qc.add(col, f"{frac:.1%} non-missing")
            else:
                qc.add(col, "column absent")
        qc.add("breadth (significant cell types)",
               f"median {flag_tab['n_significant_celltypes'].median():.0f}, "
               f"range {flag_tab['n_significant_celltypes'].min():.0f}"
               f"-{flag_tab['n_significant_celltypes'].max():.0f}")
        pat = flag_tab["sign_pattern"].value_counts().to_dict()
        qc.add("sign pattern across the significant set", str(pat))
        multi = flag_tab[flag_tab["n_significant_graph_nodes"] >= 2]
        if len(multi):
            qc.add("connected significant set, multi-node flagships",
                   f"{multi['significant_set_connected'].mean():.1%} of {len(multi):,}")
        qc.note("    The connectivity and distance columns are descriptive here. The "
                "tested version of that statement is in 04, against a margin-preserving "
                "null; a per-variant flag in a supplementary table is not a test and "
                "should not be quoted as one.")
    else:
        qc.add("flagship table", "empty at the current thresholds")

    banner("Orthogonal validation: fine-mapping posterior")
    val = pip_validation(V, long, qc, out)
    if len(val):
        plot_validation(val, out)
        log(f"tables/pip_validation.csv  {len(val)} rows")

    qc.note("    Allele frequency is an annotation column. No test in this section is "
            "run against it, and af_vs_sharing.py is retired: AF and breadth are both "
            "downstream of allele age and local accessibility, so their correlation is "
            "not evidence of a biological relationship.")
    qc.write(out / "annotate_qc.txt")
    log("done")


if __name__ == "__main__":
    main()
