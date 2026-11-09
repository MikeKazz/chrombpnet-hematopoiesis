#!/usr/bin/env python
"""
03_breadth.py: breadth against regulatory magnitude.

Primary analysis
----------------
Follows Marderstein et al. Regulatory magnitude is the largest absolute predicted effect
a variant shows over the panel, a property of the variant that does not depend on which
contexts were called. Breadth is the number of contexts in which the variant is
accessible and called. Variants are grouped as inactive / specific / multiple / shared
and compared on mean magnitude with 95% confidence intervals, alongside a linear model
with `inactive` as the reference category and equal-count bins of magnitude against
breadth.

  tables/breadth_magnitude_groups.csv   mean magnitude per category, SEM-based 95% CI
  tables/breadth_magnitude_model.csv    OLS against the inactive category + Welch contrasts
  tables/breadth_magnitude_bins.csv     magnitude bins against breadth

Because magnitude is taken over the whole panel it does not condition on the called
contexts, so the group contrast needs no permutation null; its own standard errors are
the inference. The rank correlations below are summary statistics and show how much the
choice of magnitude definition matters; their permutation null is optional (--permute).

Threshold ladder
----------------
The primary rule is an uncorrected within-cell-type cut (see config.primary_rule), which
admits roughly 1% of accessible variants per cell type by chance. Such calls arise
independently in each cell type, so they land at breadth 1 with near-threshold effects
and can produce a positive breadth-magnitude relationship on their own. The ladder
recomputes every quantity as the cut tightens: a relationship driven by chance calls
decays toward zero, a real one plateaus. The permutation null cannot detect this
artefact, because permuting within cell type destroys the structure that produces it.

Rank-correlation estimators
---------------------------
  naive   max |score| over the called cell types only, against breadth. Circular
          (predictor and outcome come from the same selection); supplement only.
  panel   max |score| over the full panel, against breadth. Marderstein et al.'s
          definition and the one used by the primary analysis above.
  loo     |score| in cell type c, against breadth computed excluding c, over all
          variant-cell-type pairs where the variant is called in c. Predictor and
          outcome no longer share a cell type.
  access  |score| in cell type c, against the number of cell types in which the variant
          is accessible, excluding c. Breadth is defined from chromatin, never from the
          effect statistic, so it is a circularity-free comparator.

Permutation null
----------------
Leave-one-out removes the shared-cell-type dependence but not the selection effect: a
variant reaches high breadth partly by having repeatedly exceeded the threshold, which
selects upward on the effect it is correlated against. The estimators are therefore
tested against a null that permutes the statistic and its q-value jointly within each
cell type, across that cell type's accessible variants. This destroys variant-level
effect sharing while preserving:

  - each cell type's marginal distribution of effect sizes
  - each cell type's number of calls
  - which variants are accessible in which cell types, and therefore the breadth
    variation that co-accessibility alone produces

What remains above this null is effect sharing at the sequence level.

Outputs
-------
  tables/breadth_magnitude.csv        observed rho, null mean/sd/quantiles, p per estimator
  tables/breadth_profile.csv          mean effect per breadth stratum, leave-one-out
  tables/rho_null_draws.csv.gz        the null draws
  figures/breadth_magnitude.{png,pdf}

Usage
-----
  python 03_breadth.py
  python 03_breadth.py --n-perm 10000 --metric abs_logfc_x_jsd
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C            # noqa: E402
import s4plot                 # noqa: E402
from s4lib import QCReport, banner, log  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Estimators
# ──────────────────────────────────────────────────────────────────────────────

ESTIMATORS = ("naive", "panel", "loo", "access")

ESTIMATOR_LABEL = {
    "naive": "Called-context max",
    "panel": "Marderstein",
    "loo": "Leave-one-out",
    "access": "Accessibility",
}

# Threshold ladder for the chance-call diagnostic, taken from config so it tracks the
# primary rule.
LADDER_FDR = C.LADDER_BY_KIND[C.PRIMARY_RULE_KIND]

#: Permutation draws used to put a construction floor on each ladder level. Capped
#: below the headline count because the reported quantity is the null mean, not a tail
#: probability, and 500 draws pin that to about +-0.002.
LADDER_PERM_DRAWS = 500


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via ranks. Ties get average ranks, matching scipy."""
    if len(x) < 3:
        return np.nan
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else np.nan


def three_estimators(stat: np.ndarray, sig: np.ndarray,
                     acc: np.ndarray) -> dict[str, float]:
    """All three rho definitions from a dense variant x cell-type design.

    stat  (n, k) float   effect statistic, already non-negative
    sig   (n, k) bool    significant call
    acc   (n, k) bool    accessible, i.e. testable
    """
    breadth = sig.sum(axis=1)
    nominated = breadth > 0
    if nominated.sum() < 3:
        return {k: np.nan for k in ESTIMATORS}

    b = breadth[nominated]

    # naive: maximum over the selected cell types only
    masked_sig = np.where(sig[nominated], stat[nominated], -np.inf)
    naive_y = masked_sig.max(axis=1)

    # panel: maximum over every cell type where the variant is testable
    masked_acc = np.where(acc[nominated], stat[nominated], -np.inf)
    panel_y = masked_acc.max(axis=1)
    finite = np.isfinite(panel_y)

    # leave-one-out: one row per significant pair
    vi, ci = np.nonzero(sig)
    loo_x = breadth[vi] - 1
    loo_y = stat[vi, ci]

    # accessibility breadth: predictor never touches the effect statistic
    acc_breadth = acc.sum(axis=1)
    acc_x = acc_breadth[vi] - 1

    return dict(naive=spearman(b, naive_y),
                panel=spearman(b[finite], panel_y[finite]),
                loo=spearman(loo_x, loo_y),
                access=spearman(acc_x, loo_y))


def permute_within_celltype(arrays: list[np.ndarray], acc: np.ndarray,
                            rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle several per-variant columns jointly among each cell type's accessible
    variants, so a variant keeps its (statistic, p, q) triple but moves to another
    variant's slot. Returns permuted copies; `acc` is untouched by construction."""
    out = [a.copy() for a in arrays]
    for j in range(acc.shape[1]):
        rows = np.flatnonzero(acc[:, j])
        perm = rng.permutation(rows)
        for src, dst in zip(arrays, out):
            dst[rows, j] = src[perm, j]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Primary analysis, following Marderstein et al.
# ──────────────────────────────────────────────────────────────────────────────
#
# Regulatory magnitude is the largest absolute predicted effect a variant shows over
# the contexts in which it is accessible. It is a property of the variant, computed
# over the whole panel, and does not depend on which contexts were called -- so it can
# be compared against breadth directly, without the selection problem that arises when
# magnitude is taken over the called contexts only.
#
# The three functions below are the published analysis: group means with 95% CIs over
# the four breadth categories, a linear model with `inactive` as the reference category,
# and bins of magnitude against breadth, which is the analogue of their Fig. 1d.


def regulatory_magnitude(stat: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Max |effect| over accessible contexts; -inf where a variant is never accessible."""
    return np.where(acc, stat, -np.inf).max(axis=1)


def magnitude_by_category(magnitude: np.ndarray, breadth: np.ndarray,
                          n_ct: int) -> pd.DataFrame:
    """Mean regulatory magnitude per breadth category, with SEM-based 95% CIs."""
    rows = []
    for name, (lo, hi) in C.breadth_categories(n_ct).items():
        v = magnitude[(breadth >= lo) & (breadth <= hi)]
        v = v[np.isfinite(v)]
        n = int(v.size)
        mean = float(v.mean()) if n else np.nan
        sd = float(v.std(ddof=1)) if n > 1 else np.nan
        sem = sd / np.sqrt(n) if n > 1 else np.nan
        rows.append(dict(
            category=name, breadth_min=lo, breadth_max=hi, n_variants=n,
            mean_magnitude=mean, median_magnitude=float(np.median(v)) if n else np.nan,
            sd=sd, sem=sem,
            ci_lo=mean - 1.96 * sem if n > 1 else np.nan,
            ci_hi=mean + 1.96 * sem if n > 1 else np.nan))
    return pd.DataFrame(rows)


def category_model(magnitude: np.ndarray, breadth: np.ndarray,
                   n_ct: int) -> pd.DataFrame:
    """Regulatory magnitude on breadth category, `inactive` as the reference level, plus
    Welch contrasts between the called categories. Coefficients are differences in mean
    magnitude against the inactive group, in the units of the effect statistic."""
    import statsmodels.api as sm

    cats = C.breadth_categories(n_ct)
    label = np.full(magnitude.shape[0], "", dtype=object)
    for name, (lo, hi) in cats.items():
        label[(breadth >= lo) & (breadth <= hi)] = name

    ok = np.isfinite(magnitude) & (label != "")
    y, lab = magnitude[ok], label[ok]
    levels = [k for k in ("specific", "multiple", "shared") if (lab == k).sum() >= 2]

    rows = []
    if (lab == "inactive").sum() >= 2 and levels:
        X = np.column_stack([np.ones(y.size)]
                            + [(lab == k).astype(float) for k in levels])
        fit = sm.OLS(y, X).fit()
        ci = fit.conf_int()
        names = ["inactive (reference mean)"] + [f"{k} - inactive" for k in levels]
        for i, nm in enumerate(names):
            rows.append(dict(term=nm, estimate=float(fit.params[i]),
                             se=float(fit.bse[i]), t=float(fit.tvalues[i]),
                             p=float(fit.pvalues[i]),
                             p_underflow=bool(fit.pvalues[i] == 0.0),
                             ci_lo=float(ci[i, 0]), ci_hi=float(ci[i, 1]),
                             n=int(y.size), model="OLS magnitude ~ category"))
    for a, b in (("multiple", "specific"), ("shared", "specific"),
                 ("shared", "multiple")):
        va, vb = y[lab == a], y[lab == b]
        if va.size >= 2 and vb.size >= 2:
            t = stats.ttest_ind(va, vb, equal_var=False)
            d = float(va.mean() - vb.mean())
            se = float(np.sqrt(va.var(ddof=1) / va.size + vb.var(ddof=1) / vb.size))
            rows.append(dict(term=f"{a} - {b}", estimate=d, se=se, t=float(t.statistic),
                             p=float(t.pvalue), p_underflow=bool(t.pvalue == 0.0),
                             ci_lo=d - 1.96 * se, ci_hi=d + 1.96 * se,
                             n=int(va.size + vb.size), model="Welch t-test"))
    out = pd.DataFrame(rows)
    # A p that underflows to exactly 0 must never be quoted as "P = 0". Report the
    # representable bound instead; the t statistic and n carry the real precision.
    if len(out) and out["p_underflow"].any():
        out.loc[out["p_underflow"], "p_bound"] = "<1e-300"
    return out


def magnitude_bins(magnitude: np.ndarray, breadth: np.ndarray, n_ct: int,
                   n_bins: int, extra: dict[str, np.ndarray] | None = None
                   ) -> pd.DataFrame:
    """Bins of equal count over regulatory magnitude, against mean breadth and the
    fraction in each category -- the analogue of Marderstein et al. Fig. 1d. Bins are
    cut on ranks so that ties in the statistic cannot collapse a bin."""
    ok = np.isfinite(magnitude)
    m, b = magnitude[ok], breadth[ok]
    if m.size < n_bins * 2:
        return pd.DataFrame()
    r = stats.rankdata(m, method="average")
    which = np.clip(((r - 1) / r.size * n_bins).astype(int), 0, n_bins - 1)
    cats = C.breadth_categories(n_ct)
    rows = []
    for k in range(n_bins):
        sel = which == k
        n = int(sel.sum())
        if not n:
            continue
        bb = b[sel].astype(float)
        sem = bb.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        row = dict(bin=k + 1, n_variants=n,
                   magnitude_min=float(m[sel].min()), magnitude_max=float(m[sel].max()),
                   magnitude_mean=float(m[sel].mean()),
                   mean_breadth=float(bb.mean()), sem_breadth=sem,
                   ci_lo=bb.mean() - 1.96 * sem if n > 1 else np.nan,
                   ci_hi=bb.mean() + 1.96 * sem if n > 1 else np.nan,
                   frac_called=float((bb > 0).mean()))
        for name, (lo, hi) in cats.items():
            row[f"frac_{name}"] = float(((bb >= lo) & (bb <= hi)).mean())
        for col, vals in (extra or {}).items():
            v = vals[ok][sel]
            v = v[np.isfinite(v)]
            if v.size > 1:
                s = v.std(ddof=1) / np.sqrt(v.size)
                row[f"{col}_mean"] = float(v.mean())
                row[f"{col}_ci_lo"] = float(v.mean() - 1.96 * s)
                row[f"{col}_ci_hi"] = float(v.mean() + 1.96 * s)
                row[f"{col}_n"] = int(v.size)
        rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────

def figure_breadth(res: pd.DataFrame, null: dict[str, np.ndarray],
                   profile: pd.DataFrame, ladder: pd.DataFrame, path: Path,
                   metric_label: str) -> list[Path]:
    import matplotlib.pyplot as plt

    s4plot.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(s4plot.FIG_W_DOUBLE, 2.7),
                             gridspec_kw={"width_ratios": [1.25, 0.9, 0.85],
                                          "wspace": 0.46})

    # ── panel a: observed rho against the construction null ──────────────────
    ax = axes[0]
    order = list(ESTIMATORS)
    pretty = ESTIMATOR_LABEL
    xs = np.arange(len(order))
    rng = np.random.default_rng(0)

    for i, name in enumerate(order):
        d = null[name]
        d = d[np.isfinite(d)]
        if len(d):
            jitter = rng.uniform(-0.11, 0.11, len(d))
            ax.scatter(xs[i] + jitter, d, s=1.2, color=s4plot.LIGHT_GREY,
                       alpha=0.35, linewidths=0, zorder=1, rasterized=True)
            lo, hi = np.quantile(d, [0.025, 0.975])
            ax.plot([xs[i] - 0.19, xs[i] + 0.19], [hi, hi], color=s4plot.GREY, lw=0.8)
            ax.plot([xs[i] - 0.19, xs[i] + 0.19], [lo, lo], color=s4plot.GREY, lw=0.8)

    obs = res.set_index("estimator")["rho_observed"]
    colours = {"naive": s4plot.COMPARATOR, "panel": s4plot.COMPARATOR,
               "loo": s4plot.FOCAL, "access": s4plot.FOCAL}
    circular = {"naive", "panel"}
    for i, name in enumerate(order):
        if name in circular:
            # open marker: the estimator is reported but is not a valid estimate
            ax.scatter([xs[i]], [obs[name]], s=42, facecolor="none", zorder=4,
                       edgecolor=s4plot.GREY, linewidth=1.1)
        else:
            ax.scatter([xs[i]], [obs[name]], s=42, color=colours[name], zorder=4,
                       edgecolor="white", linewidth=0.6)
        ax.annotate(f"{obs[name]:.2f}", (xs[i], obs[name]),
                    textcoords="offset points", xytext=(9, 0), va="center",
                    fontsize=s4plot.SMALL, color=colours[name],
                    fontweight="bold" if name == "loo" else "regular")

    ax.axhline(0, color="black", lw=0.5, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([pretty[n] for n in order], fontsize=s4plot.SMALL,
                       rotation=28, ha="right", rotation_mode="anchor")
    ax.set_ylabel(f"Spearman $\\rho$, breadth vs {metric_label}")
    ax.set_title("Circularity by estimator")
    ax.set_xlim(-0.5, len(order) - 0.28)
    # the observed values can exceed the null range by a lot; make room for them
    finite_null = np.concatenate([null[n][np.isfinite(null[n])] for n in order])
    lo = min(float(np.min(finite_null)), float(np.nanmin(obs.to_numpy())))
    hi = max(float(np.max(finite_null)), float(np.nanmax(obs.to_numpy())))
    pad = 0.12 * (hi - lo) if hi > lo else 0.1
    ax.set_ylim(lo - pad, hi + pad)
    ax.text(0.02, 0.03, "grey cloud: permutation null, 95% interval marked\nopen marker: circular estimator, not a valid estimate",
            transform=ax.transAxes, fontsize=s4plot.SMALL, color=s4plot.GREY,
            va="bottom", ha="left")

    # ── panel b: the threshold ladder ────────────────────────────────────────
    ax = axes[1]
    if len(ladder):
        for name, colour, style in (("loo", s4plot.FOCAL, "-"),
                                    ("access", s4plot.GREY, "--")):
            ax.plot(ladder["fdr"], ladder[f"rho_{name}"], style, color=colour,
                    marker="o", markersize=3.2, label=name)
        ax.legend(["leave-one-out", "accessibility"], loc="upper right",
                  fontsize=s4plot.SMALL)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("BH-FDR threshold, stricter $\\rightarrow$")
        ax.set_ylabel("Spearman $\\rho$")
        ax.set_title("Decay under tightening\nindicates chance calls")
        ax.margins(x=0.38)

    # ── panel c: the relationship itself ─────────────────────────────────────
    ax = axes[2]
    p = profile.dropna(subset=["mean_effect"])
    ax.errorbar(p["breadth_excl"], p["mean_effect"],
                yerr=[p["mean_effect"] - p["ci_lo"], p["ci_hi"] - p["mean_effect"]],
                fmt="o", color=s4plot.FOCAL, ecolor=s4plot.FOCAL, elinewidth=0.8,
                capsize=1.6, markersize=3.5, linewidth=0)
    if len(p) > 1:
        ax.plot(p["breadth_excl"], p["mean_effect"], color=s4plot.FOCAL, lw=0.8,
                alpha=0.5)
    ax.set_xlabel("Leave-one-out breadth")
    ax.set_ylabel(f"Mean {metric_label}")
    ax.set_title("Effect size against\nleave-one-out breadth")
    ax.margins(x=0.08)

    for label, a in zip("abc", axes):
        s4plot.panel_letter(a, label, dx=-0.20 if label == "a" else -0.30)
    return s4plot.save(fig, path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=C.OUTPUT_DIR)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--metric", default=C.SIGNIF_METRIC,
                    help=f"effect statistic to correlate (default {C.SIGNIF_METRIC})")
    ap.add_argument("--n-perm", type=int, default=C.N_PERMUTATIONS)
    ap.add_argument("--rule", choices=("raw", "fdr"), default=None,
                    help="calling rule; default from config.primary_rule()")
    ap.add_argument("--level", type=float, default=None,
                    help="threshold for --rule; default from config.primary_rule()")
    ap.add_argument("--permute", action="store_true",
                    help="run the permutation null for the four rho estimators. Off by "
                         "default: the primary analysis is the category contrast, which "
                         "needs no null, and the permutation is the expensive step.")
    ap.add_argument("--fdr", type=float, default=None,
                    help=argparse.SUPPRESS)      # backward-compatible alias
    ap.add_argument("--universe", default="all17",
                    help="cell-type universe from config.universe()")
    args = ap.parse_args()

    _kind, _level, _rule_label = C.primary_rule()
    if args.fdr is not None:                     # --fdr takes precedence if given
        args.rule, args.level = "fdr", args.fdr
    args.rule = args.rule or _kind
    args.level = args.level if args.level is not None else (
        _level if args.rule == _kind else C.LADDER_BY_KIND[args.rule][1])
    rule_label = (f"empirical P < {args.level:g} within cell type" if args.rule == "raw"
                  else f"BH-FDR < {args.level:g} within cell type")

    inp = args.input_dir
    out = args.output_dir or inp
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    long = pd.read_parquet(inp / "variant_celltype.parquet")
    wanted = [ct for ct in C.universe(args.universe)
              if ct in set(long["cell_type"].astype(str))]
    long = long[long["cell_type"].astype(str).isin(wanted)]

    qc = QCReport("Breadth-magnitude QC")
    qc.section("design")
    qc.add("universe", f"{args.universe} ({len(wanted)} cell types: {', '.join(wanted)})")
    qc.add("effect statistic", args.metric)
    qc.add("significance rule", f"{rule_label}, accessible only")

    # ── dense design ─────────────────────────────────────────────────────────
    banner("Building dense design")
    stat = long.pivot(index="variant_idx", columns="cell_type", values=args.metric)
    stat = stat.reindex(columns=wanted)
    variant_idx = stat.index.to_numpy()
    qval = long.pivot(index="variant_idx", columns="cell_type",
                      values="qval").reindex(columns=wanted)
    accdf = long.pivot(index="variant_idx", columns="cell_type",
                       values="accessible").reindex(columns=wanted)

    pval = long.pivot(index="variant_idx", columns="cell_type",
                      values="pval").reindex(columns=wanted)

    STAT = np.nan_to_num(stat.to_numpy(np.float64), nan=0.0)
    QVAL = np.nan_to_num(qval.to_numpy(np.float64), nan=1.0)
    PVAL = np.nan_to_num(pval.to_numpy(np.float64), nan=1.0)
    ACC = accdf.fillna(False).to_numpy(bool)

    def call_mask(level: float, kind: str = args.rule,
                  pv: np.ndarray | None = None,
                  qv: np.ndarray | None = None) -> np.ndarray:
        """Boolean call matrix under one rule. Accessibility is required either way."""
        if kind == "raw":
            return ACC & ((PVAL if pv is None else pv) < level)
        return ACC & ((QVAL if qv is None else qv) < level)

    SIG = call_mask(args.level)

    # prefer the call column 01 published, so the rule is defined in exactly one place
    if "called" in long.columns and args.rule == _kind and args.level == _level:
        called = (long.pivot(index="variant_idx", columns="cell_type", values="called")
                      .reindex(columns=wanted).fillna(False).to_numpy(bool))
        if not np.array_equal(called, SIG):
            qc.note(f"    `called` from 01 and the rule recomputed here disagree on "
                    f"{int((called != SIG).sum()):,} variant-cell-type pairs; using "
                    f"`called`. Re-run 01 if this is unexpected.")
        SIG = called

    if (STAT < 0).any():
        raise ValueError(f"{args.metric} contains negative values; the estimators "
                         "assume a non-negative magnitude")

    breadth = SIG.sum(axis=1)
    qc.add("variants in the design", f"{STAT.shape[0]:,}")
    qc.add("nominated (breadth >= 1)", f"{int((breadth > 0).sum()):,}")
    qc.add("significant variant-cell-type pairs", f"{int(SIG.sum()):,}")
    qc.add("median breadth among nominated",
           float(np.median(breadth[breadth > 0])) if (breadth > 0).any() else 0.0)

    # ── primary analysis: the published category contrast ────────────────────
    banner("Breadth categories against regulatory magnitude")
    MAGNITUDE = regulatory_magnitude(STAT, ACC)
    # Regulatory magnitude is a maximum over accessible contexts, so a variant that is
    # testable but accessible in no single cell type has no magnitude and leaves every
    # comparison below. Those variants reach the pipeline through the consensus peak set
    # rather than through per-cell-type accessibility. Reported here so the denominator
    # of the magnitude comparison is never mistaken for the tested count.
    _finite = int(np.isfinite(MAGNITUDE).sum())
    qc.section("denominator of the magnitude comparison")
    qc.add("tested variants", f"{MAGNITUDE.size:,}")
    qc.add("with a defined regulatory magnitude (accessible in >= 1 cell type)",
           f"{_finite:,}")
    qc.add("dropped: in the consensus peak set but accessible in no cell type",
           f"{MAGNITUDE.size - _finite:,} "
           f"({(MAGNITUDE.size - _finite) / MAGNITUDE.size:.1%})")
    n_ct = len(wanted)

    groups = magnitude_by_category(MAGNITUDE, breadth, n_ct)
    groups.to_csv(out / "tables/breadth_magnitude_groups.csv", index=False)
    qc.section("regulatory magnitude by breadth category")
    for r in groups.itertuples():
        qc.add(f"{r.category} (breadth {r.breadth_min}-{r.breadth_max})",
               f"n = {r.n_variants:,}, mean = {r.mean_magnitude:.4f} "
               f"[{r.ci_lo:.4f}, {r.ci_hi:.4f}]"
               if r.n_variants > 1 else f"n = {r.n_variants:,}")
    if int(groups.loc[groups["category"] == "shared", "n_variants"].iloc[0]) == 0:
        qc.note("    the `shared` category is empty at this threshold, so the "
                "specific-versus-multiple contrast carries the comparison")

    model = category_model(MAGNITUDE, breadth, n_ct)
    model.to_csv(out / "tables/breadth_magnitude_model.csv", index=False)
    qc.section("category model")
    for r in model.itertuples():
        pstr = "< 1e-300 (underflow)" if getattr(r, "p_underflow", False) else f"{r.p:.3g}"
        qc.add(f"{r.term} ({r.model})",
               f"{r.estimate:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}], p = {pstr}")
    if len(model) and model["p_underflow"].any():
        qc.note("    one or more p-values underflow to exactly 0 in double precision. "
                "Quote them as < 1e-300, never as 0; the effect size and its CI are "
                "the reportable quantities.")

    extra = {}
    vpath_pre = inp / "variants.parquet"
    if vpath_pre.exists():
        want_cols = ["variant_idx", "tss_distance", C.CONSTRAINT_METRIC]
        have = pd.read_parquet(vpath_pre).columns
        cols = [c for c in want_cols if c in have]
        if len(cols) > 1:
            vv = (pd.read_parquet(vpath_pre, columns=cols)
                    .set_index("variant_idx").reindex(variant_idx))
            if "tss_distance" in vv:
                _t = vv["tss_distance"].to_numpy(np.float64)
                _t = np.where(_t >= 0, _t, np.nan)      # -1 marks "no nearest gene"
                with np.errstate(invalid="ignore"):
                    extra["log10_tss_distance"] = np.log10(_t + 1.0)
            if C.CONSTRAINT_METRIC in vv:
                extra[C.CONSTRAINT_METRIC] = vv[C.CONSTRAINT_METRIC].to_numpy(np.float64)

    bins = magnitude_bins(MAGNITUDE, breadth, n_ct, C.N_MAGNITUDE_BINS, extra)
    bins.to_csv(out / "tables/breadth_magnitude_bins.csv", index=False)
    qc.section("magnitude bins")
    qc.add("bins", len(bins))
    if len(bins):
        qc.add("mean breadth, lowest bin",
               f"{bins['mean_breadth'].iloc[0]:.4f} "
               f"[{bins['ci_lo'].iloc[0]:.4f}, {bins['ci_hi'].iloc[0]:.4f}]")
        qc.add("mean breadth, highest bin",
               f"{bins['mean_breadth'].iloc[-1]:.4f} "
               f"[{bins['ci_lo'].iloc[-1]:.4f}, {bins['ci_hi'].iloc[-1]:.4f}]")
        qc.add("Spearman rho, magnitude against breadth, INCLUDING inactive variants",
               f"{spearman(MAGNITUDE[np.isfinite(MAGNITUDE)], breadth[np.isfinite(MAGNITUDE)]):.4f}")
        qc.note("    Not one of the four estimators: those condition on breadth > 0, "
                "this one does not. Because it spans breadth 0 against breadth >= 1 it "
                "contains the whole active-versus-inactive contrast, which is the part "
                "of the breadth-magnitude relationship that the calling rule generates "
                "on its own. It is a descriptive summary of the binning panel and must "
                "not be quoted as a breadth-magnitude correlation.")

    # ── observed ─────────────────────────────────────────────────────────────
    banner("Observed estimators")
    observed = three_estimators(STAT, SIG, ACC)
    for k, v in observed.items():
        log(f"{k:>6}  rho = {v:.4f}", level=1)

    # ── permutation null ─────────────────────────────────────────────────────
    # Only the rho estimators need this. The category contrast above is referred to
    # its own standard errors, so the default run skips it.
    n_perm = args.n_perm if args.permute else 0
    # The same draws also give a null floor at each ladder level (one permutation can
    # be scored at every level). They are capped below n_perm because the ladder needs
    # only the null mean, whose standard error is already ~0.002 at 500 draws.
    ladder_draws = min(n_perm, LADDER_PERM_DRAWS)
    null_ladder = {lv: {k: [] for k in ESTIMATORS} for lv in LADDER_FDR}
    banner(f"Permutation null, {n_perm:,} draws"
           + ("" if args.permute else "  (skipped; pass --permute to run)"))
    rng = np.random.default_rng(C.RANDOM_SEED)
    null = {k: np.empty(n_perm) for k in observed}
    t0 = time.time()
    for i in range(n_perm):
        sp, pp, qp = permute_within_celltype([STAT, PVAL, QVAL], ACC, rng)
        sig_p = call_mask(args.level, pv=pp, qv=qp)
        r = three_estimators(sp, sig_p, ACC)
        for k in null:
            null[k][i] = r[k]
        if i < ladder_draws:
            for lv in LADDER_FDR:
                rl = three_estimators(sp, call_mask(lv, pv=pp, qv=qp), ACC)
                for k in ESTIMATORS:
                    null_ladder[lv][k].append(rl[k])
        if i == 9:
            rate = (time.time() - t0) / 10
            log(f"{rate * 1000:.0f} ms per permutation, projected "
                f"{rate * n_perm / 60:.1f} min total", level=1)
        if n_perm >= 500 and (i + 1) % max(n_perm // 10, 1) == 0:
            log(f"{i + 1:,}/{n_perm:,}", level=1)

    # ── summarise ────────────────────────────────────────────────────────────
    rows = []
    qc.section("estimators against the construction null")
    for name in ESTIMATORS:
        d = null[name][np.isfinite(null[name])]
        o = observed[name]
        if len(d) < 2:                            # no null requested, or none usable
            rows.append(dict(estimator=name, rho_observed=o, null_mean=np.nan,
                             null_sd=np.nan, null_q025=np.nan, null_q975=np.nan,
                             excess_over_null=np.nan, p_permutation=np.nan,
                             n_null_draws=len(d), n_draws_attempted=n_perm,
                             frac_draws_finite=(len(d) / n_perm) if n_perm else np.nan))
            qc.add(f"{name}: observed rho", f"{o:.4f} (no permutation null)")
            continue
        # one-sided: the question is whether the observed exceeds what construction gives
        p = (np.sum(d >= o) + 1) / (len(d) + 1)
        excess = o - float(np.mean(d))
        rows.append(dict(
            estimator=name, rho_observed=o, null_mean=float(np.mean(d)),
            null_sd=float(np.std(d, ddof=1)), null_q025=float(np.quantile(d, 0.025)),
            null_q975=float(np.quantile(d, 0.975)), excess_over_null=excess,
            p_permutation=p, n_null_draws=len(d), n_draws_attempted=n_perm,
            frac_draws_finite=len(d) / n_perm))
        qc.add(f"{name}: observed rho", f"{o:.4f}")
        qc.add(f"{name}: null mean (sd)",
               f"{np.mean(d):.4f} ({np.std(d, ddof=1):.4f})")
        qc.add(f"{name}: excess over null", f"{excess:+.4f}")
        qc.add(f"{name}: permutation p", f"{p:.2e}" if p < 0.01 else f"{p:.4f}")
        qc.add(f"{name}: usable null draws", f"{len(d):,} of {args.n_perm:,}")
        if n_perm and len(d) < 0.9 * n_perm:
            qc.note(f"    {name}: {n_perm - len(d):,} of {n_perm:,} null draws "
                    f"returned an undefined rho and were dropped, so the p floor is "
                    f"1/{len(d) + 1} rather than 1/{args.n_perm + 1}. A draw is dropped "
                    f"when the shuffled data give every nominated variant the same "
                    f"breadth, leaving the predictor constant. The surviving draws are "
                    f"therefore the shuffles that produced the MOST spurious sharing, "
                    f"which makes this p conservative rather than anti-conservative -- "
                    f"but quote the usable count, not n_perm.")

    res = pd.DataFrame(rows)
    res.to_csv(out / "tables/breadth_magnitude.csv", index=False)
    pd.DataFrame(null).to_csv(out / "tables/rho_null_draws.csv.gz", index=False)

    qc.note("    The naive estimator's null mean is the construction floor: the "
            "correlation the original code returns when variant-level effect sharing "
            "has been destroyed. Any claim about breadth and magnitude has to be stated "
            "as the excess over this floor, not as the raw correlation.")

    # ── chance-call diagnostic: the threshold ladder ───────────────────────
    banner("Threshold ladder")
    ladder_rows = []
    for level in LADDER_FDR:
        sig_l = call_mask(level)
        b_l = sig_l.sum(axis=1)
        if (b_l > 0).sum() < 10:
            log(f"{args.rule} < {level:g}: fewer than 10 nominated variants, skipped", level=1)
            continue
        est = three_estimators(STAT, sig_l, ACC)
        row = dict(fdr=level, level=level, rule=args.rule,
                   n_nominated=int((b_l > 0).sum()),
                   n_pairs=int(sig_l.sum()),
                   frac_breadth_1=float((b_l[b_l > 0] == 1).mean()))
        row.update({f"rho_{k}": v for k, v in est.items()})
        # Each level carries its own null floor. The floor depends strongly on the level:
        # a loose threshold admits many chance calls, which alone induce a positive rho,
        # while under a tight threshold a shuffle almost never calls the same variant
        # twice, breadth becomes constant and rho is undefined (a floor of zero).
        for k in ESTIMATORS:
            draws = np.asarray(null_ladder[level][k], dtype=float)
            draws = draws[np.isfinite(draws)]
            if draws.size >= 2:
                row[f"rho_{k}_null_mean"] = float(draws.mean())
                row[f"rho_{k}_null_q025"] = float(np.quantile(draws, 0.025))
                row[f"rho_{k}_null_q975"] = float(np.quantile(draws, 0.975))
                row[f"rho_{k}_excess"] = float(est[k] - draws.mean())
            else:
                # every draw gave a constant predictor, so chance sharing at this level
                # is absent rather than unmeasured
                row[f"rho_{k}_null_mean"] = 0.0 if ladder_draws else np.nan
                row[f"rho_{k}_null_q025"] = np.nan
                row[f"rho_{k}_null_q975"] = np.nan
                row[f"rho_{k}_excess"] = (est[k] if ladder_draws else np.nan)
            row[f"rho_{k}_null_draws"] = int(draws.size)
        ladder_rows.append(row)
        log(f"{args.rule} < {level:<7g} n = {row['n_nominated']:>7,}  "
            f"loo = {est['loo']:+.4f}  access = {est['access']:+.4f}", level=1)

    ladder = pd.DataFrame(ladder_rows)
    ladder.to_csv(out / "tables/breadth_threshold_ladder.csv", index=False)

    # ── the magnitude-only rule, on the same footing ─────────────────────────
    # A rule calling a variant at |metric| >= 0.5 (optionally AND PIP >= 0.5) makes
    # breadth "the number of cell types whose magnitude clears 0.5", so breadth is a
    # monotone function of magnitude and the correlation is definitional. Scoring that
    # rule with all four estimators separates the definitional part from the rest:
    # compare rho_naive with rho_loo within the magnitude-only rows, then rho_loo across
    # the two rules.
    banner("Magnitude-only rule comparison")
    mag_rows = []
    pipv = None
    vpath = inp / "variants.parquet"
    if vpath.exists():
        vdf = pd.read_parquet(vpath, columns=["variant_idx", "PIP"])
        pipv = (vdf.set_index("variant_idx")["PIP"]
                   .reindex(variant_idx).to_numpy(np.float64))
    for label, sig_x in (
        (f"primary: {rule_label}", SIG),
        (f"magnitude-only: |{args.metric}| >= {C.MAGNITUDE_ONLY_THRESH:g} (no PIP filter)",
         ACC & (STAT >= C.MAGNITUDE_ONLY_THRESH)),
    ):
        mag_rows.append(dict(rule=label, pip_filter=False,
                                n_nominated=int((sig_x.sum(axis=1) > 0).sum()),
                                n_pairs=int(sig_x.sum()),
                                **three_estimators(STAT, sig_x, ACC)))
    if pipv is not None:
        keep = np.nan_to_num(pipv, nan=0.0) >= C.MAGNITUDE_ONLY_PIP
        qc_n = int(keep.sum())
        for label, sig_x in (
            (f"magnitude-only: |{args.metric}| >= {C.MAGNITUDE_ONLY_THRESH:g} AND "
             f"PIP >= {C.MAGNITUDE_ONLY_PIP:g}",
             ACC & (STAT >= C.MAGNITUDE_ONLY_THRESH) & keep[:, None]),
            (f"primary AND PIP >= {C.MAGNITUDE_ONLY_PIP:g}", SIG & keep[:, None]),
        ):
            mag_rows.append(dict(rule=label, pip_filter=True,
                                    n_nominated=int((sig_x.sum(axis=1) > 0).sum()),
                                    n_pairs=int(sig_x.sum()),
                                    **three_estimators(STAT, sig_x, ACC)))
        qc.add("variants with PIP >= the magnitude-only cut", f"{qc_n:,}")
    else:
        qc.note("    variants.parquet absent or without a PIP column; the PIP-filtered "
                "magnitude-only rows are omitted")
    lg = pd.DataFrame(mag_rows)
    lg.to_csv(out / "tables/breadth_magnitude_only_rule.csv", index=False)
    qc.section("magnitude-only rule comparison")
    for r in mag_rows:
        log(f"{r['rule'][:52]:<54} n = {r['n_nominated']:>7,}  "
            f"naive = {r['naive']:+.4f}  loo = {r['loo']:+.4f}", level=1)
        qc.add(f"{r['rule']}: n nominated", r["n_nominated"])
        qc.add(f"{r['rule']}: rho naive / loo",
               f"{r['naive']:+.4f} / {r['loo']:+.4f}")
    qc.note("    A large naive-minus-loo gap on a magnitude-threshold rule is the "
            "definitional component: it measures how much of the correlation comes from "
            "breadth being a function of magnitude by construction.")

    qc.section("chance-call diagnostic")
    qc.add("thresholds evaluated", len(ladder))
    if len(ladder) >= 2:
        span = ladder["rho_loo"].max() - ladder["rho_loo"].min()
        strictest = ladder.iloc[ladder["fdr"].argmin()]
        qc.add("leave-one-out rho at the primary threshold",
               f"{ladder.loc[ladder['fdr'] == args.level, 'rho_loo'].iloc[0]:.4f}"
               if (ladder["fdr"] == args.level).any() else "primary not on the ladder")
        qc.add("leave-one-out rho at the strictest threshold",
               f"{strictest['rho_loo']:.4f} ({args.rule} < {strictest['fdr']:g}, "
               f"n = {int(strictest['n_nominated']):,})")
        qc.add("range of leave-one-out rho across the ladder", f"{span:.4f}")
        qc.note("    A correlation that decays toward zero as the threshold tightens "
                "reflects chance calls, which concentrate at breadth 1 with "
                "near-threshold effect sizes, rather than effect sharing. The "
                "accessibility-breadth estimator is not affected by this or by the "
                "order-statistic artefact, so agreement between the plateau of rho_loo "
                "and rho_access supports a real relationship.")

    # ── leave-one-out profile for panel b ────────────────────────────────────
    vi, ci = np.nonzero(SIG)
    prof = pd.DataFrame(dict(variant_idx=variant_idx[vi],
                             cell_type=[wanted[c] for c in ci],
                             breadth_excl=breadth[vi] - 1,
                             effect=STAT[vi, ci]))
    g = prof.groupby("breadth_excl")["effect"]
    profile = pd.DataFrame(dict(n_pairs=g.size(), mean_effect=g.mean(),
                                sd=g.std(ddof=1))).reset_index()
    # variant-clustered standard error: pairs from one variant are not independent, so
    # the effective sample size is the number of distinct variants in the stratum
    n_var = prof.groupby("breadth_excl")["variant_idx"].nunique()
    profile["n_variants"] = n_var.to_numpy()
    se = profile["sd"] / np.sqrt(profile["n_variants"].clip(lower=1))
    profile["ci_lo"] = profile["mean_effect"] - 1.96 * se
    profile["ci_hi"] = profile["mean_effect"] + 1.96 * se
    profile.to_csv(out / "tables/breadth_profile.csv", index=False)

    qc.section("leave-one-out profile")
    qc.add("breadth strata", len(profile))
    qc.note("    Confidence intervals use the number of distinct variants in each "
            "stratum, not the number of pairs; one variant contributes up to "
            f"{len(wanted)} strongly correlated pairs.")

    # the rho panel plots the permutation cloud, so it needs a null to draw
    if any(np.isfinite(v).sum() >= 2 for v in null.values()):
        figs = figure_breadth(res, null, profile, ladder,
                              out / "figures/breadth_magnitude",
                              metric_label=args.metric.replace("_", " "))
        for f in figs:
            log(f"figure -> {f}")
    else:
        log("breadth_magnitude figure skipped: no permutation null "
            "(pass --permute to draw it)")
        qc.note("    breadth_magnitude figure not drawn: it plots the permutation "
                "cloud, which this run did not compute. The primary analysis is in "
                "breadth_magnitude_groups.csv, _model.csv and _bins.csv.")

    qc.write(out / "breadth_qc.txt")
    _spec = groups.loc[groups["category"] == "specific", "mean_magnitude"]
    _mult = groups.loc[groups["category"] == "multiple", "mean_magnitude"]
    if len(_spec) and len(_mult) and np.isfinite(_spec.iloc[0]) and np.isfinite(_mult.iloc[0]):
        log(f"primary: mean magnitude specific = {_spec.iloc[0]:.4f}, "
            f"multiple = {_mult.iloc[0]:.4f}")
    _exc = res.set_index("estimator").loc["loo", "excess_over_null"]
    log(f"rank-correlation summary (leave-one-out) rho = {observed['loo']:.4f}"
        + (f", excess over null {_exc:+.4f}" if np.isfinite(_exc) else ""))
    log("done")


if __name__ == "__main__":
    main()
