#!/usr/bin/env python
"""
ctcf_chipseq_sanity.py: CTCF motif sanity checks against K562 CTCF ChIP-seq.

Definitions:
  * "called CTCF instance" = confident, fully-in-boundary hit labelled CTCF_2
    (Match_1), the same set behind 161/1,259 positives vs 1,545/7,946 negatives.
    Low-confidence CTCF_2 hits inside the boundary are used only as a control.
  * TSS distance = |closest_TSS_distance_Protein_coding_hg38|.
  * part (b) reports crude and study-adjusted results.
  * part (c) = strand bias only (no external annotation).

(a) OVERLAP WITH CHIP PEAKS
    An instance "overlaps" a peak if its centre lies inside a merged IDR peak
    (width-independent, so motifs of different length are comparable); any-bp
    overlap is reported too. Controls: (1) the same CTCF instances placed at
    random positions inside their own element (same width) -- accessible
    elements sit near peaks anyway, so this is the fair null; (2) confident
    instances of all other motifs; (3) low-confidence in-boundary CTCF_2 calls.
    Also: distance from instance centre to the nearest summit (+-300 bp
    profile), and an element-level 2x2 (element contains a CTCF instance vs the
    element overlaps a peak) with a length-adjusted logistic model.
(b) CTCF ASSOCIATION BY TSS DISTANCE
    Per bin (<=1, 1-2, 2-5, 5-10, 10-50, >50 kb) and cumulative distal cuts
    (>1, >2, >5, >10 kb): odds ratio (Woolf 95% CI) and Fisher p for
      - any confident CTCF motif vs none (the paper's definition),
      - ChIP-supported CTCF vs none,
      - CTCF motif present but never ChIP-supported vs none,
      - NFE2/NRF2 (NFE_0) any motif vs none, as a comparator.
    Adjusted versions: Mantel-Haenszel stratified by study, by TSS bin, and by
    study x bin, plus a logistic model with study (and bin) fixed effects.
    Adjusted estimates use only strata that contain both positives and negatives.
(c) ORIENTATION (strand only)
    '+' vs '-' fraction of CTCF instances overall, by class (positive/negative
    element) and by TSS bin; element-level orientation class (all '+', all '-',
    mixed) by class; and, for ChIP-supported instances within 250 bp of a
    summit, whether the summit lies downstream or upstream in the motif's own
    frame. Strand is the Fi-NeMo motif strand (consistent CWM orientation); it
    is NOT orientation relative to a gene.

Usage:
    python ctcf_chipseq_sanity.py \\
        --csv data/crispri/ALL7_trimmed.csv \\
        --region-map results/crispri_motifs/region_map.tsv \\
        --hits results/crispri_motifs/finemo_hits/hits.annotated.tsv \\
        --peaks data/external/ENCFF736NYC.bed.gz \\
        --out-dir results/crispri_motifs/ctcf_chipseq [--n-perm 1000] [--dpi 600]
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings("ignore")


CTCF, NFE = "CTCF_2", "NFE_0"
OFF = 10 ** 10  # chromosome offset for combined genome keys
C_CTCF, C_NFE, C_GREY, C_LIGHT = "#2166AC", "#B2182B", "#4d4d4d", "#9a9a9a"
BIN_EDGES = [0, 1e3, 2e3, 5e3, 1e4, 5e4, np.inf]
BIN_LABELS = ["≤1 kb", "1–2 kb", "2–5 kb", "5–10 kb", "10–50 kb", ">50 kb"]

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 11, "axes.linewidth": 1.0, "svg.fonttype": "none", "pdf.fonttype": 42})


# ------------------------------------------------------------------ loading
def load_elements(path):
    df = pd.read_csv(path)
    df = df[df["Cell_type"] == "K562"].copy()
    s = pd.to_numeric(df["CRE_start_hg38"], errors="coerce")
    e = pd.to_numeric(df["CRE_end_hg38"], errors="coerce")
    df = df[s.notna() & e.notna()].copy()
    df["key"] = list(zip(df["chr"], s[s.notna() & e.notna()].astype(int), e[s.notna() & e.notna()].astype(int)))
    df["has_target_gene"] = df["has_target_gene"].astype(str)
    df["Reference"] = df["Reference"].astype(str)
    df["dist"] = pd.to_numeric(df["closest_TSS_distance_Protein_coding_hg38"], errors="coerce").abs()
    return df.drop_duplicates("key")[["key", "has_target_gene", "Reference", "dist"]]


def load_region_map(path):
    rm = pd.read_csv(path, sep="\t")
    # join key equals the CSV's CRE_start/end directly (both 1-based bed-style)
    rm["key"] = list(zip(rm["chrom"], rm["bed_start_1based"].astype(int), rm["bed_end_1based"].astype(int)))
    return rm[["region_id", "chrom", "bed_start_1based", "bed_end_1based", "key"]]


def load_peaks(path):
    cols = ["chrom", "start", "end", "name", "score", "strand", "signal", "pval", "qval", "peak"]
    p = pd.read_csv(path, sep="\t", header=None, names=cols, compression="infer")
    off = p["peak"].where(p["peak"] >= 0, (p["end"] - p["start"]) // 2)
    p["summit"] = p["start"] + off
    return p


# ------------------------------------------------------------------ interval helpers
def merge_intervals(starts, ends):
    o = np.argsort(starts, kind="stable")
    s, e = starts[o], ends[o]
    cm = np.maximum.accumulate(e)
    new = np.r_[True, s[1:] > cm[:-1]]
    idx = np.flatnonzero(new)
    return s[idx], np.maximum.reduceat(e, idx)


def center_in(ms, me, pos):
    i = np.searchsorted(ms, pos, side="right") - 1
    return (i >= 0) & (pos < me[np.clip(i, 0, None)])


def any_overlap(ms, me, s, e):
    i = np.searchsorted(ms, e, side="left") - 1
    return (i >= 0) & (me[np.clip(i, 0, None)] > s)


def nearest_summit_dist(summits, pos):
    i = np.searchsorted(summits, pos)
    left = summits[np.clip(i - 1, 0, len(summits) - 1)]
    right = summits[np.clip(i, 0, len(summits) - 1)]
    dl, dr = pos - left, pos - right
    d = np.where(np.abs(dl) <= np.abs(dr), dl, dr).astype(float)
    d[np.abs(d) >= OFF / 2] = np.nan
    return d  # instance minus summit; positive = instance downstream of summit


# ------------------------------------------------------------------ stats helpers
def or_ci(a, b, c, d):
    """a,b = positives with/without motif; c,d = negatives with/without."""
    if min(a + b, c + d) == 0:
        return np.nan, np.nan, np.nan
    aa, bb, cc, dd = float(a), float(b), float(c), float(d)
    if min(aa, bb, cc, dd) == 0:
        aa, bb, cc, dd = aa + .5, bb + .5, cc + .5, dd + .5
    o = (aa * dd) / (bb * cc)
    se = np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    return o, float(np.exp(np.log(o) - 1.96 * se)), float(np.exp(np.log(o) + 1.96 * se))


def crude(sub, x):
    y = sub["y"].to_numpy()
    x = np.asarray(x).astype(bool)
    a, b = int(((y == 1) & x).sum()), int(((y == 1) & ~x).sum())
    c, d = int(((y == 0) & x).sum()), int(((y == 0) & ~x).sum())
    o, lo, hi = or_ci(a, b, c, d)
    p = stats.fisher_exact([[a, c], [b, d]])[1] if min(a + b, c + d) > 0 else np.nan
    return dict(n_pos=a + b, n_neg=c + d, pos_with=a, neg_with=c, OR=o, CI_lo=lo, CI_hi=hi, p=p)


def adjusted(sub, x):
    d = pd.DataFrame({"y": sub["y"].to_numpy(), "x": np.asarray(x).astype(int),
                      "Reference": sub["Reference"].to_numpy(),
                      "bin": sub["bin"].astype(object).where(sub["bin"].notna(), "NA").to_numpy()})
    out = {}

    def mh(key, tag):
        tabs = []
        for _, g in d.groupby(key):
            if g["y"].nunique() < 2:
                continue
            a, b = ((g.y == 1) & (g.x == 1)).sum(), ((g.y == 1) & (g.x == 0)).sum()
            c, dd = ((g.y == 0) & (g.x == 1)).sum(), ((g.y == 0) & (g.x == 0)).sum()
            tabs.append(np.array([[a, b], [c, dd]], float))
        out[f"MH_{tag}_n_strata"] = len(tabs)
        if not tabs:
            return
        try:
            st = StratifiedTable(tabs)
            lo, hi = st.oddsratio_pooled_confint()
            out[f"MH_{tag}_OR"], out[f"MH_{tag}_lo"], out[f"MH_{tag}_hi"] = st.oddsratio_pooled, lo, hi
            out[f"MH_{tag}_p"] = st.test_null_odds(correction=False).pvalue
            if len(tabs) > 1:
                out[f"MH_{tag}_BreslowDay_p"] = st.test_equal_odds(adjust=False).pvalue
        except Exception:
            pass

    d["study_bin"] = d["Reference"] + "||" + d["bin"].astype(str)
    mh("Reference", "study")
    if d["bin"].nunique() > 1:
        mh("bin", "bin")
        mh("study_bin", "studyXbin")
    ok_refs = [r for r, g in d.groupby("Reference") if g["y"].nunique() == 2]
    use = d[d["Reference"].isin(ok_refs)]
    out["FE_n_elements"] = len(use)
    try:
        X = pd.DataFrame({"x": use["x"].astype(float)})
        X = pd.concat([X, pd.get_dummies(use["Reference"], prefix="s", drop_first=True, dtype=float)], axis=1)
        if use["bin"].nunique() > 1:
            X = pd.concat([X, pd.get_dummies(use["bin"], prefix="b", drop_first=True, dtype=float)], axis=1)
        fit = sm.Logit(use["y"].astype(int), sm.add_constant(X)).fit(disp=0, maxiter=200)
        ci = np.exp(fit.conf_int().loc["x"])
        out.update(FE_OR=float(np.exp(fit.params["x"])), FE_lo=float(ci[0]), FE_hi=float(ci[1]),
                   FE_p=float(fit.pvalues["x"]))
    except Exception:
        pass
    return out


def save_fig(fig, name, out_dir, dpi):
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches=None)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=dpi, bbox_inches=None)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", required=True, help="ALL7_trimmed.csv")
    ap.add_argument("--region-map", required=True, help="region_map.tsv from build_region_map.py")
    ap.add_argument("--hits", required=True, help="hits.annotated.tsv from annotate_hits.py")
    ap.add_argument("--peaks", required=True, help="CTCF ChIP-seq IDR peaks (ENCFF736NYC.bed.gz)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    log = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.append(s)

    # ---- load
    elements = load_elements(args.csv)
    regs = load_region_map(args.region_map).merge(elements, on="key", how="inner").drop_duplicates("region_id")
    hits = pd.read_csv(args.hits, sep="\t")
    peaks = load_peaks(args.peaks)
    say(f"elements matched to a region: {len(regs):,}; hits: {len(hits):,}; peaks: {len(peaks):,}")

    chroms = sorted(set(peaks.chrom) | set(hits.chr) | set(regs.chrom))
    cid = {c: i for i, c in enumerate(chroms)}
    peaks["cid"] = peaks.chrom.map(cid).astype(np.int64)
    ms, me = merge_intervals((peaks.cid * OFF + peaks.start).to_numpy(np.int64),
                             (peaks.cid * OFF + peaks.end).to_numpy(np.int64))
    summits = np.sort((peaks.cid * OFF + peaks.summit).to_numpy(np.int64))
    say(f"peaks after merging overlaps: {len(ms):,}")

    # ---- instance table
    h = hits.merge(regs, left_on="peak_name", right_on="region_id", how="inner")
    h["inb"] = (h.start >= h.bed_start_1based - 1) & (h.end <= h.bed_end_1based)
    flags = h["low_confidence_flags"].fillna("")
    h["conf"] = (flags != "no_tomtom_match") & ~flags.str.split(";").apply(lambda l: "Match_1" in l)
    h["ok"] = h.inb & h.conf
    h["cid"] = h.chr.map(cid).astype(np.int64)
    h["pos"] = h.cid * OFF + (h.start + h.end) // 2
    h["w"] = h.end - h.start
    h["in_peak"] = center_in(ms, me, h["pos"].to_numpy())
    h["any_ov"] = any_overlap(ms, me, (h.cid * OFF + h.start).to_numpy(), (h.cid * OFF + h.end).to_numpy())
    h["d_summit"] = nearest_summit_dist(summits, h["pos"].to_numpy())
    h["bin"] = pd.cut(h["dist"], BIN_EDGES, labels=BIN_LABELS, include_lowest=True)
    is_ctcf = h.Match_1 == CTCF
    sets = {
        "CTCF confident (called instances)": h[is_ctcf & h.ok],
        "Other motifs, confident": h[~is_ctcf & h.ok & h.Match_1.notna()],
        "CTCF low-confidence, in boundary": h[is_ctcf & h.inb & ~h.conf],
    }
    S = sets["CTCF confident (called instances)"]
    say(f"called CTCF instances: {len(S):,} in {S.region_id.nunique():,} elements")

    # ---- (a) instance-level overlap
    rows = []
    for name, g in sets.items():
        for cls, gg in [("all", g), ("positive elements", g[g.has_target_gene == "True"]),
                        ("negative elements", g[g.has_target_gene == "False"])]:
            n = len(gg)
            rows.append(dict(set=name, element_class=cls, n_instances=n,
                             n_center_in_peak=int(gg.in_peak.sum()), frac_center_in_peak=gg.in_peak.mean() if n else np.nan,
                             n_any_overlap=int(gg.any_ov.sum()), frac_any_overlap=gg.any_ov.mean() if n else np.nan))
    # shuffled-in-element null for the called CTCF instances
    lo = (S.cid * OFF + S.bed_start_1based - 1).to_numpy(np.int64)
    hi = (S.cid * OFF + S.bed_end_1based).to_numpy(np.int64)
    w = S.w.to_numpy(np.int64)
    null_frac, prof_null = [], []
    pbins = np.arange(-300, 301, 10)
    for k in range(args.n_perm):
        st = rng.integers(lo, hi - w + 1)
        c = st + w // 2
        null_frac.append(center_in(ms, me, c).mean())
        if k < 100:
            dd = nearest_summit_dist(summits, c)
            prof_null.append(np.histogram(dd[~np.isnan(dd)], bins=pbins)[0] / len(c))
    null_frac = np.array(null_frac)
    obs = S.in_peak.mean()
    p_perm = (1 + (null_frac >= obs).sum()) / (1 + len(null_frac))
    rows.append(dict(set="CTCF shuffled within own element (null; mean, 2.5-97.5%)", element_class="all",
                     n_instances=len(S), frac_center_in_peak=null_frac.mean(),
                     n_center_in_peak=np.nan, frac_any_overlap=np.nan, n_any_overlap=np.nan))
    a_tab = pd.DataFrame(rows)
    a_tab["null_2.5"] = np.nan
    a_tab["null_97.5"] = np.nan
    a_tab.loc[a_tab.index[-1], ["null_2.5", "null_97.5"]] = np.percentile(null_frac, [2.5, 97.5])
    a_tab["fold_over_null"] = a_tab["frac_center_in_peak"] / null_frac.mean()
    a_tab["perm_p_vs_null"] = np.where(a_tab["set"].str.startswith("CTCF confident") & (a_tab.element_class == "all"),
                                       p_perm, np.nan)
    a_tab.to_csv(os.path.join(args.out_dir, "a_instance_overlap.tsv"), sep="\t", index=False)
    say(f"(a) CTCF called instances with centre in a ChIP peak: {S.in_peak.sum():,}/{len(S):,} = {obs:.1%}; "
        f"shuffled-in-element null {null_frac.mean():.1%} (95% {np.percentile(null_frac, 2.5):.1%}-"
        f"{np.percentile(null_frac, 97.5):.1%}); fold {obs / null_frac.mean():.2f}; permutation p = {p_perm:.4f}")
    for name in list(sets)[1:]:
        g = sets[name]
        say(f"    {name}: {g.in_peak.mean():.1%} (n={len(g):,})")
    near = S.loc[S.in_peak, "d_summit"].abs()
    say(f"    among CTCF instances in peaks: median |instance - summit| = {near.median():.0f} bp; "
        f"within 25 bp {np.mean(near <= 25):.1%}, within 50 bp {np.mean(near <= 50):.1%}")

    # summit profile
    prof = {"bin_start": pbins[:-1], "bin_end": pbins[1:]}
    for name, g in sets.items():
        dd = g.d_summit.to_numpy()
        prof[name] = np.histogram(dd[~np.isnan(dd)], bins=pbins)[0] / max(len(g), 1)
    prof["CTCF shuffled within element (mean of 100)"] = np.mean(prof_null, axis=0)
    pd.DataFrame(prof).to_csv(os.path.join(args.out_dir, "a_summit_profile.tsv"), sep="\t", index=False)

    # ---- element table
    E = regs.set_index("region_id").copy()
    E["y"] = (E.has_target_gene == "True").astype(int)
    E["length"] = E.bed_end_1based - (E.bed_start_1based - 1)
    E["bin"] = pd.cut(E["dist"], BIN_EDGES, labels=BIN_LABELS, include_lowest=True)

    def cnt(mask):
        return h.loc[mask].groupby("region_id").size().reindex(E.index).fillna(0).astype(int)

    E["n_ctcf"] = cnt(is_ctcf & h.ok)
    E["n_ctcf_bound"] = cnt(is_ctcf & h.ok & h.in_peak)
    E["n_ctcf_plus"] = cnt(is_ctcf & h.ok & (h.strand == "+"))
    E["n_ctcf_minus"] = cnt(is_ctcf & h.ok & (h.strand == "-"))
    E["n_nfe"] = cnt((h.Match_1 == NFE) & h.ok)
    E["ctcf_any"] = E.n_ctcf > 0
    E["nfe_any"] = E.n_nfe > 0
    E["ctcf_class"] = np.select([E.n_ctcf == 0, E.n_ctcf_bound > 0], ["none", "bound"], "unbound_only")
    ecid = E.chrom.map(cid).astype(np.int64)
    E["peak_overlap"] = any_overlap(ms, me, (ecid * OFF + E.bed_start_1based - 1).to_numpy(),
                                    (ecid * OFF + E.bed_end_1based).to_numpy())
    E.drop(columns=["key"]).to_csv(os.path.join(args.out_dir, "elements_ctcf_flags.tsv"), sep="\t")
    chk = crude(E, E.ctcf_any)
    say(f"check: elements with a confident in-boundary CTCF motif: {chk['pos_with']}/{chk['n_pos']} positives, "
        f"{chk['neg_with']}/{chk['n_neg']} negatives (earlier result: 161/1,259 and 1,545/7,946); OR {chk['OR']:.3f}")

    # element-level 2x2 and length-adjusted logistic
    a, b = int((E.ctcf_any & E.peak_overlap).sum()), int((E.ctcf_any & ~E.peak_overlap).sum())
    c, d = int((~E.ctcf_any & E.peak_overlap).sum()), int((~E.ctcf_any & ~E.peak_overlap).sum())
    o2, p2 = stats.fisher_exact([[a, b], [c, d]])
    try:
        X = sm.add_constant(pd.DataFrame({"ctcf_any": E.ctcf_any.astype(float), "log10_length": np.log10(E.length)}))
        f = sm.Logit(E.peak_overlap.astype(int), X).fit(disp=0)
        adj = (float(np.exp(f.params["ctcf_any"])), *[float(v) for v in np.exp(f.conf_int().loc["ctcf_any"])],
               float(f.pvalues["ctcf_any"]))
    except Exception:
        adj = (np.nan,) * 4
    pd.DataFrame([dict(elements_with_ctcf_and_peak=a, elements_with_ctcf_no_peak=b, elements_without_ctcf_with_peak=c,
                       elements_without_ctcf_no_peak=d, frac_peak_if_ctcf=a / max(a + b, 1),
                       frac_peak_if_no_ctcf=c / max(c + d, 1), fisher_OR=o2, fisher_p=p2,
                       length_adj_OR=adj[0], length_adj_lo=adj[1], length_adj_hi=adj[2], length_adj_p=adj[3])]
                 ).to_csv(os.path.join(args.out_dir, "a_element_level.tsv"), sep="\t", index=False)
    say(f"(a) element level: {a / max(a + b, 1):.1%} of elements with a CTCF motif overlap a CTCF peak vs "
        f"{c / max(c + d, 1):.1%} without (OR {o2:.2f}, p {p2:.2g}; length-adjusted OR {adj[0]:.2f})")

    # ---- (b) TSS-distance analysis
    strata = [("All elements", "all", np.ones(len(E), bool))]
    strata += [(lab, "bin", (E.bin == lab).to_numpy()) for lab in BIN_LABELS]
    strata += [(f">{k} kb" + (" (distal)" if k == 10 else ""), "cumulative", (E.dist > k * 1e3).to_numpy())
               for k in (1, 2, 5, 10)]
    specs = [("CTCF: any confident motif", "CTCF", None, lambda s: s.ctcf_any),
             ("CTCF: ChIP-supported vs none", "CTCF", ["none", "bound"], lambda s: s.ctcf_class == "bound"),
             ("CTCF: motif never ChIP-supported vs none", "CTCF", ["none", "unbound_only"],
              lambda s: s.ctcf_class == "unbound_only"),
             ("NFE2/NRF2: any confident motif", "NFE2/NRF2", None, lambda s: s.nfe_any)]
    rows = []
    for sname, gtype, smask in strata:
        base = E[smask]
        for spec, motif, classes, xf in specs:
            sub = base if classes is None else base[base.ctcf_class.isin(classes)]
            x = xf(sub)
            r = dict(analysis=spec, motif=motif, stratum=sname, stratum_type=gtype, **crude(sub, x))
            r.update(adjusted(sub, x))
            rows.append(r)
    b_tab = pd.DataFrame(rows)
    b_tab.to_csv(os.path.join(args.out_dir, "b_tss_association.tsv"), sep="\t", index=False)
    say("(b) CTCF any-motif association (crude OR [95% CI]; study-MH OR; FE-logit OR):")
    for _, r in b_tab[(b_tab.analysis == specs[0][0])].iterrows():
        say(f"    {r.stratum:14s} pos {r.pos_with}/{r.n_pos} neg {r.neg_with}/{r.n_neg}  "
            f"OR {r.OR:.2f} [{r.CI_lo:.2f},{r.CI_hi:.2f}] p={r.p:.2g} | study-MH {r.get('MH_study_OR', np.nan):.2f} "
            f"| FE {r.get('FE_OR', np.nan):.2f}")

    # ---- (c) strand
    Sx = S.copy()
    Sx["plus"] = Sx.strand == "+"
    c_rows = []
    n, k = len(Sx), int(Sx.plus.sum())
    bt = stats.binomtest(k, n, 0.5)
    ci = proportion_confint(k, n, method="wilson")
    c_rows.append(dict(test="strand: fraction '+' overall vs 0.5", n=n, k_plus=k, frac=k / n,
                       ci_lo=ci[0], ci_hi=ci[1], p=bt.pvalue))
    for cls in ("True", "False"):
        g = Sx[Sx.has_target_gene == cls]
        kk = int(g.plus.sum())
        ci = proportion_confint(kk, len(g), method="wilson")
        c_rows.append(dict(test=f"strand: fraction '+' in {'positive' if cls == 'True' else 'negative'} elements",
                           n=len(g), k_plus=kk, frac=kk / len(g), ci_lo=ci[0], ci_hi=ci[1], p=np.nan))
    gp, gn = Sx[Sx.has_target_gene == "True"], Sx[Sx.has_target_gene == "False"]
    t = [[int(gp.plus.sum()), int((~gp.plus).sum())], [int(gn.plus.sum()), int((~gn.plus).sum())]]
    c_rows.append(dict(test="strand '+' fraction: positive vs negative elements (Fisher)", n=n,
                       k_plus=np.nan, frac=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=stats.fisher_exact(t)[1]))
    for lab in BIN_LABELS:
        g = Sx[Sx.bin == lab]
        if len(g):
            kk = int(g.plus.sum())
            ci = proportion_confint(kk, len(g), method="wilson")
            c_rows.append(dict(test=f"strand: fraction '+' in TSS bin {lab}", n=len(g), k_plus=kk, frac=kk / len(g),
                               ci_lo=ci[0], ci_hi=ci[1], p=stats.binomtest(kk, len(g), 0.5).pvalue))
    Ec = E[E.ctcf_any].copy()
    Ec["orient"] = np.select([Ec.n_ctcf_minus == 0, Ec.n_ctcf_plus == 0], ["all +", "all -"], "mixed")
    ct = pd.crosstab(Ec.orient, Ec.has_target_gene)
    c_rows.append(dict(test="element orientation class vs has_target_gene (chi-square): " +
                       "; ".join(f"{i}: {ct.loc[i].get('True', 0)} pos / {ct.loc[i].get('False', 0)} neg" for i in ct.index),
                       n=len(Ec), k_plus=np.nan, frac=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                       p=stats.chi2_contingency(ct)[1] if ct.shape[0] > 1 and ct.shape[1] > 1 else np.nan))
    Sb = Sx[Sx.in_peak & (Sx.d_summit.abs() <= 250) & (Sx.d_summit != 0)].copy()
    down = np.where(Sb.strand == "+", Sb.d_summit < 0, Sb.d_summit > 0)
    kd = int(down.sum())
    ci = proportion_confint(kd, len(Sb), method="wilson") if len(Sb) else (np.nan, np.nan)
    c_rows.append(dict(test="ChIP summit downstream of motif (in motif frame), within 250 bp: fraction vs 0.5",
                       n=len(Sb), k_plus=kd, frac=kd / max(len(Sb), 1), ci_lo=ci[0], ci_hi=ci[1],
                       p=stats.binomtest(kd, len(Sb), 0.5).pvalue if len(Sb) else np.nan))
    c_tab = pd.DataFrame(c_rows)
    c_tab.to_csv(os.path.join(args.out_dir, "c_strand.tsv"), sep="\t", index=False)
    say(f"(c) '+' strand fraction {k / n:.1%} of {n:,} instances (binomial p {bt.pvalue:.2g}); "
        f"summit downstream in motif frame: {kd / max(len(Sb), 1):.1%} of {len(Sb):,}")

    # ================= figures =================
    # Figure 1: (a)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    labels = ["CTCF\n(called)", "CTCF placed\nrandomly in\nown element", "Other motifs\n(confident)",
              "CTCF\n(low confidence)"]
    vals = [obs, null_frac.mean(), sets["Other motifs, confident"].in_peak.mean(),
            sets["CTCF low-confidence, in boundary"].in_peak.mean()]
    cols = [C_CTCF, C_LIGHT, C_GREY, "#92C5DE"]
    ax[0].bar(range(4), vals, color=cols, width=0.62)
    ax[0].errorbar([1], [null_frac.mean()], yerr=[[null_frac.mean() - np.percentile(null_frac, 2.5)],
                                                  [np.percentile(null_frac, 97.5) - null_frac.mean()]],
                   color="black", capsize=4, lw=1.2)
    ns = [len(S), len(S), len(sets["Other motifs, confident"]), len(sets["CTCF low-confidence, in boundary"])]
    tops = [v for v in vals]
    tops[1] = np.percentile(null_frac, 97.5)
    for i, (v, nn) in enumerate(zip(vals, ns)):
        ax[0].text(i, tops[i] + 0.012, f"{v:.0%}\nn={nn:,}", ha="center", va="bottom", fontsize=9)
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels(labels, fontsize=9)
    ax[0].set_ylabel("Instances with centre inside a\nK562 CTCF ChIP-seq peak")
    ax[0].set_ylim(0, max(vals) * 1.3)
    ax[0].set_title("A", loc="left", fontweight="bold", fontsize=15)
    mid = (pbins[:-1] + pbins[1:]) / 2
    ax[1].plot(mid, prof["CTCF confident (called instances)"], color=C_CTCF, lw=2, label="CTCF (called)")
    ax[1].plot(mid, prof["CTCF shuffled within element (mean of 100)"], color=C_LIGHT, lw=2,
               label="CTCF placed randomly in own element")
    ax[1].plot(mid, prof["Other motifs, confident"], color=C_GREY, lw=1.6, ls="--", label="Other motifs")
    ax[1].set_xlabel("Instance centre − nearest ChIP-seq summit (bp)")
    ax[1].set_ylabel("Fraction of instances per 10-bp bin")
    ax[1].legend(frameon=False, fontsize=9, loc="upper right")
    ax[1].set_title("B", loc="left", fontweight="bold", fontsize=15)
    for a_ in ax:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.2, wspace=0.28)
    save_fig(fig, "figure_a_chip_overlap", args.out_dir, args.dpi)

    # Figure 2: (b)
    order = [s[0] for s in strata]
    ypos = {}
    y = 0
    for sname, gtype, _ in strata:
        if gtype in ("bin", "cumulative") and (sname == BIN_LABELS[0] or sname.startswith(">1 kb")):
            y += 0.6
        ypos[sname] = y
        y += 1
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True, sharex=True)
    series = {0: [(specs[0][0], "Any confident CTCF motif", C_CTCF, "o", -0.2),
                  (specs[1][0], "ChIP-supported only", "#6BAED6", "D", 0.0),
                  (specs[2][0], "Never ChIP-supported", C_LIGHT, "s", 0.2)],
              1: [(specs[3][0], "Any confident NFE2/NRF2 motif", C_NFE, "o", 0.0)]}
    for k, ax_ in enumerate(axes):
        for spec, lab, col, mk, dy in series[k]:
            t = b_tab[b_tab.analysis == spec].set_index("stratum")
            xs = [t.loc[s, "OR"] for s in order]
            ys = [ypos[s] + dy for s in order]
            lo_ = [t.loc[s, "OR"] - t.loc[s, "CI_lo"] for s in order]
            hi_ = [t.loc[s, "CI_hi"] - t.loc[s, "OR"] for s in order]
            ax_.errorbar(xs, ys, xerr=[np.nan_to_num(lo_), np.nan_to_num(hi_)], fmt=mk, color=col, ms=6.5,
                         mec="black" if mk != "D" else col, mew=0.6, lw=1.6, capsize=0, label=lab)
        ax_.axvline(1, color="#888888", ls="--", lw=1)
        ax_.set_xscale("log")
        ax_.set_xlabel("Odds ratio (95% CI)")
        ax_.set_title(["CTCF", "NFE2 / NRF2 (comparator)"][k], fontweight="bold", loc="left")
        ax_.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3 if k == 0 else 1)
        for sp in ("top", "right"):
            ax_.spines[sp].set_visible(False)
    t0 = b_tab[b_tab.analysis == specs[0][0]].set_index("stratum")
    axes[0].set_yticks([ypos[s] for s in order])
    axes[0].set_yticklabels([f"{s}  ({int(t0.loc[s, 'n_pos'])}/{int(t0.loc[s, 'n_neg'])})" for s in order], fontsize=9)
    axes[0].set_ylim(y, -0.8)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.92, bottom=0.27, wspace=0.08)
    fig.text(0.01, 0.01, "Odds ratio = motif in target-gene-positive vs negative elements. Row labels: stratum (positive/negative elements). Cumulative rows overlap; bins are disjoint.",
             fontsize=8, color="#555555")
    save_fig(fig, "figure_b_tss_distance", args.out_dir, args.dpi)

    # Figure 3: (c)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    sr = c_tab[c_tab.test.str.startswith("strand: fraction '+'")]
    names = ["All", "Positive", "Negative"] + [r.test.split("bin ")[1] for _, r in sr.iloc[3:].iterrows()]
    fr, l_, h_ = sr.frac.to_numpy(), sr.ci_lo.to_numpy(), sr.ci_hi.to_numpy()
    ax[0].bar(range(len(fr)), fr, color=C_CTCF, width=0.6)
    ax[0].errorbar(range(len(fr)), fr, yerr=[fr - l_, h_ - fr], fmt="none", color="black", capsize=3, lw=1)
    ax[0].axhline(0.5, color="#888888", ls="--", lw=1)
    ax[0].set_xticks(range(len(fr)))
    ax[0].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("Fraction of CTCF instances on '+' strand")
    ax[0].set_title("A", loc="left", fontweight="bold", fontsize=15)
    r = c_tab[c_tab.test.str.startswith("ChIP summit")].iloc[0]
    ax[1].bar([0], [r.frac], color="#6BAED6", width=0.5)
    ax[1].errorbar([0], [r.frac], yerr=[[r.frac - r.ci_lo], [r.ci_hi - r.frac]], fmt="none", color="black", capsize=4)
    ax[1].axhline(0.5, color="#888888", ls="--", lw=1)
    ax[1].set_xticks([0])
    ax[1].set_xticklabels([f"Summit downstream of motif\n(n={int(r.n):,}, within 250 bp)"], fontsize=9)
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("Fraction")
    ax[1].set_title("B", loc="left", fontweight="bold", fontsize=15)
    for a_ in ax:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.27, wspace=0.3)
    save_fig(fig, "figure_c_orientation", args.out_dir, args.dpi)

    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    print("\nAll outputs written to", args.out_dir)


if __name__ == "__main__":
    main()
