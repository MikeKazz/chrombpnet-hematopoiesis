#!/usr/bin/env python
"""
04_hierarchy.py: does effect sharing follow the haematopoietic lineage?

The panel's lineage structure allows statistics over a graph relating the cell types.
Two tests are run.

1. Variant level. For each variant called in k >= 2 graph nodes, test whether its called
   set is a connected subgraph of the lineage graph and compute the mean pairwise
   shortest-path distance within the set. Compare against a null that preserves both
   margins (each variant's breadth and each cell type's number of called variants) by
   curveball bipartite rewiring (Strona et al. 2014). Curveball removes cell-cell
   correlation entirely, and adjacent cell types have near-duplicate models (HSC and MPP
   share most peaks and predictions), so this null makes almost any structure look
   significant. These numbers are an upper bound on lineage coherence.

2. Pair level (adjusted for model similarity). Each pair's number of co-called variants
   is modelled as a function of graph distance, with the pair's score correlation as a
   covariate, so that distance retaining an effect indicates sharing that tracks lineage
   rather than model redundancy.

Pair-level model (primary)
--------------------------
A quasi-Poisson count model for the number of co-called variants with an offset for the
independence expectation. For each pair, expected = k_a * k_b / n_ab, where n_ab is the
number of variants accessible in both members and k_x the number of those called in x:
the independence expectation of the 2 x 2 table of activity among shared-accessible
variants. Pairs whose expectation is exactly zero are excluded rather than imputed.
Standard errors are Pearson-scaled, as these are counts with mild overdispersion.

A logit-rate regression on co-call rate is also fitted and labelled "confounded
sensitivity". It is confounded because a pair's co-call rate is exactly zero whenever
either member calls nothing, and cell types that call nothing are mostly progenitors,
which lie close together on the graph; the forced zeros therefore concentrate at short
distance and generate a positive distance slope from discovery-rate heterogeneity alone.

An offset built from each cell type's panel-wide called rate (n_ab * rate_a * rate_b) is
also fitted as a sensitivity. It assumes the rate holds within the shared-accessible set,
which is non-random and shrinks with distance. In coherence_pair_level.csv,
expected_cosignificant and obs_over_expected are the within-pair values; the
*_panelwide columns hold the sensitivity version.

Similarity covariate
--------------------
The primary covariate is the score correlation over variants accessible in both cell
types and called in neither: model agreement on the null background, which measures
redundancy without including the shared signal. A correlation over all variants
accessible in both includes the co-called variants and so partly measures the sharing
being explained; adjusting for it removes real signal along with the confound. The
all-variant version is fitted too and reported as a labelled sensitivity row.

Why the graph is not a tree
---------------------------
GMP has two parents in the encoded hierarchy, CMP and LMPP. The 14-node graph therefore
contains exactly one cycle and is a DAG. Distances are all-pairs shortest paths on the
undirected graph, and connectivity is ordinary graph connectivity. The flipping analysis
uses the directed edges.

Direction flipping
------------------
Variants called in both a progenitor and its immediate descendant with opposite sign,
reported per directed edge with a binomial interval. A sign flip in a model's prediction
is not by itself evidence of biological reversal; it can also mean the variant sits in a
motif whose surrounding context differs between the two cell types.

Universe
--------
The 14 graph nodes only. K562 is a leukaemia line with no position in normal
haematopoiesis, and the two CD34+ pools span several graph nodes, so they cannot enter a
graph statistic.

Outputs
-------
  tables/coherence_variant_level.csv    observed and null, connectivity and distance
  tables/coherence_pair_level.csv       all pairs: distance, expected and observed
                                        co-called counts, informative flag
  tables/coherence_regression.csv       primary model (within-pair expected count):
                                        distance coefficient, adjacent contrast,
                                        5,000-relabelling nulls, similarity adjustment and
                                        partial Mantel, edge removal; then the same set
                                        under panel-wide rates (10,000 relabellings), then
                                        the logit-rate fits as confounded-sensitivity rows
  tables/per_celltype_counts.csv        called and testable counts per cell type
  tables/direction_flips.csv            per directed edge
  tables/graph_distances.csv            all-pairs shortest path
  figures/hierarchy_coherence.{png,pdf}
  hierarchy_qc.txt

Usage
-----
  python 04_hierarchy.py --input-dir <01 output> [--n-perm 1000]
"""

from __future__ import annotations

import argparse
from collections import deque
from itertools import combinations
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C                               # noqa: E402
import s4plot                                    # noqa: E402
from s4lib import QCReport, banner, log          # noqa: E402


# ── graph ────────────────────────────────────────────────────────────────────

ADJACENT_MAX_DISTANCE = 2   # steps in the lineage graph counted as adjacent

#: Relabellings of the cell types on the graph, used to refer the distance coefficient
#: to a null that preserves the pair dependence structure. PRIMARY applies to the
#: within-pair model; the panel-wide-rate sensitivity uses 10,000 relabellings.
PRIMARY_LABEL_PERMUTATIONS = 5_000
NODE_LABEL_PERMUTATIONS = 10_000

#: GMP is the only node with two parents, so these are the two edges forming the single
#: cycle. Each is dropped in turn as a sensitivity on every graph distance.
NON_TREE_EDGE_SENSITIVITY = (("LMPP", "GMP"), ("CMP", "GMP"))

def build_graph(nodes: list[str]) -> dict[str, set[str]]:
    """Undirected adjacency over the graph nodes present in `nodes`."""
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in C.LINEAGE_EDGES_DIRECTED:
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def all_pairs_distance(adj: dict[str, set[str]]) -> pd.DataFrame:
    """BFS from every node. 14 nodes, so an O(V*E) loop is the right implementation."""
    nodes = list(adj)
    D = pd.DataFrame(np.inf, index=nodes, columns=nodes, dtype=float)
    for src in nodes:
        D.loc[src, src] = 0.0
        seen = {src}
        q = deque([(src, 0)])
        while q:
            node, d = q.popleft()
            for nb in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    D.loc[src, nb] = d + 1
                    q.append((nb, d + 1))
    return D


def is_connected(members: np.ndarray, adj_idx: list[set[int]]) -> bool:
    """Is the induced subgraph on `members` (integer node indices) connected?"""
    if len(members) <= 1:
        return True
    member_set = set(members.tolist())
    start = members[0]
    seen = {start}
    q = deque([start])
    while q:
        node = q.popleft()
        for nb in adj_idx[node]:
            if nb in member_set and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(member_set)


# ── coherence statistics ─────────────────────────────────────────────────────

def coherence_stats(SIG: np.ndarray, D: np.ndarray,
                    adj_idx: list[set[int]]) -> dict[str, float]:
    """
    Fraction of multi-context variants whose significant set is connected, and the mean
    within-set pairwise graph distance. Both computed over variants with breadth >= 2;
    breadth-1 variants are trivially connected and carry no information about structure.
    """
    breadth = SIG.sum(axis=1)
    multi = np.nonzero(breadth >= 2)[0]
    if len(multi) == 0:
        return dict(n_multi=0, frac_connected=np.nan, mean_pair_distance=np.nan)

    n_conn = 0
    dist_sum = 0.0
    dist_n = 0
    for v in multi:
        members = np.nonzero(SIG[v])[0]
        if is_connected(members, adj_idx):
            n_conn += 1
        sub = D[np.ix_(members, members)]
        iu = np.triu_indices(len(members), k=1)
        vals = sub[iu]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            dist_sum += float(vals.mean())
            dist_n += 1

    return dict(n_multi=int(len(multi)),
                frac_connected=n_conn / len(multi),
                mean_pair_distance=dist_sum / dist_n if dist_n else np.nan)


def curveball(SIG: np.ndarray, rng: np.random.Generator,
              n_rounds: int | None = None) -> np.ndarray:
    """
    Curveball bipartite rewiring (Strona et al. 2014, Nat Commun 5:4114). Preserves both
    margins exactly: every variant keeps its breadth and every cell type keeps its number
    of significant variants. Implemented on the row-set representation, which is what the
    algorithm is written for and is far faster than swapping in the dense matrix.
    """
    n_rows, n_cols = SIG.shape
    rows = [set(np.nonzero(SIG[i])[0].tolist()) for i in range(n_rows)]
    if n_rounds is None:
        # Strona's guidance is ~5x the number of rows for mixing; the matrices here are
        # tall and sparse, so this is cheap.
        n_rounds = 5 * n_rows

    for _ in range(n_rounds):
        i, j = rng.integers(0, n_rows, size=2)
        if i == j:
            continue
        a, b = rows[i], rows[j]
        shared = a & b
        only_a = list(a - shared)
        only_b = list(b - shared)
        n_a = len(only_a)
        if n_a == 0 or len(only_b) == 0:
            continue
        pool = only_a + only_b
        rng.shuffle(pool)
        rows[i] = shared | set(pool[:n_a])
        rows[j] = shared | set(pool[n_a:])

    out = np.zeros_like(SIG)
    for i, s in enumerate(rows):
        if s:
            out[i, list(s)] = True
    return out


# ── figure ───────────────────────────────────────────────────────────────────

def figure_hierarchy(var_res: pd.DataFrame, null_draws: dict[str, np.ndarray],
                     pairs: pd.DataFrame, flips: pd.DataFrame,
                     regression: pd.DataFrame | None,
                     path: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    s4plot.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(s4plot.FIG_W_DOUBLE, 3.0),
                             gridspec_kw={"width_ratios": [0.85, 1.15, 1.0],
                                          "wspace": 0.5})

    # ── panel a: variant-level coherence against the margin-preserving null ──
    ax = axes[0]
    obs = var_res.set_index("statistic")["observed"]
    names = ["frac_connected", "mean_pair_distance"]
    labels = ["Connected\nsubgraph", "Mean pairwise\ndistance"]
    # two statistics on different scales: plot each as a z-score against its own null
    zs, texts = [], []
    _vr = var_res.set_index("statistic")
    for n in names:
        draws = null_draws.get(n)
        if draws is not None and np.isfinite(draws).sum() > 1:
            draws = draws[np.isfinite(draws)]
            mu, sd = float(draws.mean()), float(draws.std())
        elif n in _vr.index:
            # --pairs-only: the draws were not regenerated, but the saved table carries
            # the null mean and sd, which is all a z-score needs
            mu, sd = float(_vr.loc[n, "null_mean"]), float(_vr.loc[n, "null_sd"])
        else:
            mu, sd = np.nan, np.nan
        zs.append((obs[n] - mu) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan)
        texts.append(f"{obs[n]:.2f}")
    xs = np.arange(len(names))
    ax.bar(xs, zs, color=s4plot.FOCAL, width=0.55)
    ax.axhline(0, color="black", lw=0.6)
    for x, z, t in zip(xs, zs, texts):
        ax.annotate(t, (x, z), textcoords="offset points",
                    xytext=(0, 4 if z >= 0 else -11), ha="center",
                    fontsize=s4plot.SMALL, color=s4plot.FOCAL)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=s4plot.SMALL)
    ax.set_ylabel("z against margin-preserving null")
    ax.set_title("Variant-level coherence\n(upper bound)")
    ax.margins(y=0.25)

    # ── panel b: what the primary model fits, observed over the independence
    # expectation, against graph distance. The confounded rate-vs-distance view this
    # panel used to show is retained in the tables as a sensitivity row only.
    ax = axes[1]
    inf = pairs[pairs["informative"]] if "informative" in pairs.columns else pairs
    n_excl = int((~pairs["informative"]).sum()) if "informative" in pairs.columns else 0
    if len(inf):
        adj = inf["graph_distance"] <= ADJACENT_MAX_DISTANCE
        for m, col, lab in ((adj, s4plot.FOCAL, f"$\\leq${ADJACENT_MAX_DISTANCE} "
                             "(lineage-adjacent)"),
                            (~adj, s4plot.COMPARATOR, "> "
                             f"{ADJACENT_MAX_DISTANCE}")):
            if m.any():
                ax.scatter(inf.loc[m, "graph_distance"],
                           inf.loc[m, "obs_over_expected"],
                           color=col, s=30, edgecolor="white", linewidth=0.4,
                           zorder=3, label=lab)
        ax.axhline(1.0, color="black", lw=0.7, ls=(0, (4, 2)), zorder=1)
        ax.annotate("independence", (ax.get_xlim()[1], 1.0),
                    textcoords="offset points", xytext=(-2, 4), ha="right",
                    fontsize=s4plot.TICK, color=s4plot.GREY)
        ax.set_yscale("log")
        # ratios read as numbers, not as powers of ten
        lo, hi = ax.get_ylim()
        ticks = [t for t in (0.25, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)
                 if lo <= t <= hi]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Lineage graph distance")
        ax.set_ylabel("Observed / expected under\nindependent nomination")
        ax.set_xticks(sorted(inf["graph_distance"].unique()))
        ax.legend(frameon=False, fontsize=s4plot.TICK, loc="lower left",
                  handletextpad=0.3, borderaxespad=0.2)
        if regression is not None and len(regression):
            h = regression[regression["model"].str.startswith("primary: count")]
            if len(h):
                r = h.iloc[0]
                ax.text(0.97, 0.97,
                        f"$\\beta_{{dist}}$ = {r['beta_distance']:+.3f}"
                        f"\n$p$ = {r['p_distance']:.1g}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=s4plot.TICK, color=s4plot.GREY, linespacing=1.5)
        ax.set_title(f"Pair-level coherence\n({len(inf)} informative pairs, "
                     f"{n_excl} excluded)")

    # ── panel c: direction flips per edge ────────────────────────────────────
    ax = axes[2]
    f = flips[flips["n_both"] >= 5].sort_values("flip_rate")
    if len(f):
        ys = np.arange(len(f))
        ax.barh(ys, f["flip_rate"], color=s4plot.COMPARATOR, height=0.68)
        err = np.vstack([(f["flip_rate"] - f["ci_lo"]).to_numpy(),
                         (f["ci_hi"] - f["flip_rate"]).to_numpy()]).clip(min=0)
        ax.errorbar(f["flip_rate"], ys, xerr=err,
                    fmt="none", ecolor=s4plot.GREY, elinewidth=0.8, capsize=1.6)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{a}$\\rightarrow${b}" for a, b in
                            zip(f["parent"], f["child"])], fontsize=s4plot.TICK)
        ax.set_xlabel("Fraction with opposite sign")
        ax.set_title("Direction flips across\ndifferentiation")
    else:
        ax.text(0.5, 0.5, "no edge with $\\geq$5\nvariants significant in both",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=s4plot.SMALL, color=s4plot.GREY)
        ax.set_xticks([]); ax.set_yticks([])

    for label, a in zip("abc", axes):
        s4plot.panel_letter(a, label, dx=-0.30)
    return s4plot.save(fig, path)


# ── main ─────────────────────────────────────────────────────────────────────


def fit_pair_models(inf: pd.DataFrame, expected: np.ndarray, D: np.ndarray,
                    nodes: list[str], adj: dict[str, set[str]], prefix: str,
                    n_perm: int, seed: int, n_zero: int,
                    null_contrast: bool = True) -> tuple[list[dict], dict, float]:
    """Every pair-level fit for one choice of expected count.

    Poisson GLM, n_cosignificant ~ covariate, offset log(expected), Pearson-scaled SEs
    (quasi-Poisson; scale = X2). Fits, in order:
      count model on graph distance             (row label: f"{prefix}: count model ...")
      lineage-adjacent (distance <= 2) contrast
      cell-type label permutation null for the distance coefficient (n_perm relabellings)
      the same null for the adjacent contrast                       (if null_contrast)
      + background score correlation, and its partial-Mantel null
      the distance model with each edge into GMP removed
    Relabelling: the 14 node labels are permuted on the graph, each pair takes the
    distance between its permuted labels, and its count and offset stay attached.
    Permutation refits are plain Poisson (scaling changes the SE, not the coefficient).
    Returns (rows, fold enrichment adjacent/distant, distance coefficient).
    """
    import statsmodels.api as sm
    rows: list[dict] = []
    off = np.log(np.asarray(expected, float))
    yc = inf["n_cosignificant"].to_numpy(float)
    dist = inf["graph_distance"].to_numpy(float)
    near = dist <= ADJACENT_MAX_DISTANCE
    base = dict(n_pairs=len(inf), n_pairs_excluded_zero_expectation=n_zero)

    def fit(cols, scaled=True):
        X = sm.add_constant(np.column_stack(cols) if len(cols) > 1 else cols[0])
        m = sm.GLM(yc, X, family=sm.families.Poisson(), offset=off)
        return m.fit(scale="X2") if scaled else m.fit()

    def row(label, g, **extra):
        ci = g.conf_int()
        r = dict(model=label, beta_distance=float(g.params[1]), se_distance=float(g.bse[1]),
                 ci_lo=float(ci[1, 0]), ci_hi=float(ci[1, 1]), p_distance=float(g.pvalues[1]),
                 beta_score_corr=np.nan, p_score_corr=np.nan, r_squared=np.nan,
                 rate_ratio=float(np.exp(g.params[1])),
                 dispersion=float(g.pearson_chi2 / g.df_resid), **base)
        r.update(extra)
        return r

    g_d = fit([dist]);  beta = float(g_d.params[1])
    g_c = fit([near.astype(float)]); beta_c = float(g_c.params[1])
    rows.append(row(f"{prefix}: count model with independence offset", g_d))
    rows.append(row(f"{prefix}: lineage-adjacent (distance <= {ADJACENT_MAX_DISTANCE}) contrast", g_c))
    for lab, g in (("distance", g_d), ("adjacent contrast", g_c)):
        ci = g.conf_int()
        log(f"{prefix} {lab}: beta = {g.params[1]:+.4f} ({np.exp(g.params[1]):.3f}x) "
            f"[{ci[1,0]:+.4f}, {ci[1,1]:+.4f}], p = {g.pvalues[1]:.3g}, "
            f"dispersion = {g.pearson_chi2 / g.df_resid:.2f} on {int(g.df_resid)} df", level=1)

    fold = {}
    for lab, m in (("adjacent", near), ("distant", ~near)):
        fold[lab] = yc[m].sum() / np.asarray(expected, float)[m].sum()
        fold[f"expected_{lab}"] = float(np.asarray(expected, float)[m].sum())
        fold[f"observed_{lab}"] = float(yc[m].sum())

    # ── relabelling null ─────────────────────────────────────────────────────────
    npos = {ct: k for k, ct in enumerate(nodes)}
    ia = np.array([npos[a] for a in inf["cell_type_a"]])
    ib = np.array([npos[b] for b in inf["cell_type_b"]])
    assert np.allclose(D[ia, ib], dist), "node index map disagrees with the stored pair distances"
    sc = inf["score_corr"].to_numpy(float) if "score_corr" in inf.columns else None
    has_sc = sc is not None and np.isfinite(sc).all()

    def coef(cols):
        try:
            return float(fit(cols, scaled=False).params[1])
        except Exception:
            return np.nan

    rng = np.random.default_rng(seed)
    nd, nc, na = [], [], []
    for _ in range(n_perm):
        p = rng.permutation(len(nodes))
        dp = D[p[ia], p[ib]]
        nd.append(coef([dp]))
        if null_contrast:
            nc.append(coef([(dp <= ADJACENT_MAX_DISTANCE).astype(float)]))
        if has_sc:
            na.append(coef([dp, sc]))
    nd, nc, na = (np.array(x, float) for x in (nd, nc, na))

    def null_row(label, obs, null, tail):
        null = null[np.isfinite(null)]
        p = (1 + ((null <= obs) if tail == "less" else (null >= obs)).sum()) / (1 + len(null))
        # two-sided, about the null mean: for coefficients whose sign is not predicted
        # (the similarity-adjusted model)
        p2 = (1 + (np.abs(null - null.mean()) >= abs(obs - null.mean())).sum()) / (1 + len(null))
        rows.append(dict(model=label, beta_distance=obs, se_distance=np.nan,
                         ci_lo=float(np.quantile(null, 0.025)), ci_hi=float(np.quantile(null, 0.975)),
                         p_distance=float(p), p_two_sided=float(p2), tail=tail,
                         beta_score_corr=np.nan, p_score_corr=np.nan,
                         r_squared=np.nan, rate_ratio=float(np.exp(obs)), dispersion=np.nan,
                         null_mean=float(null.mean()), n_null_draws=len(null), **base))
        log(f"{label}: observed {obs:+.4f}; null mean {null.mean():+.4f} "
            f"[{np.quantile(null, .025):+.4f}, {np.quantile(null, .975):+.4f}]; "
            f"one-sided ({tail}) p = {p:.3g}, two-sided p = {p2:.3g} on {len(null):,} relabellings",
            level=1)

    # one-sided: the hypothesis is that sharing DECREASES with distance (and is higher
    # for adjacent pairs)
    null_row(f"{prefix}: cell-type label permutation null", beta, nd, "less")
    if null_contrast:
        null_row(f"{prefix}: adjacent-contrast label permutation null", beta_c, nc, "greater")

    # ── background model similarity as a covariate ───────────────────────────────
    if has_sc:
        g_a = fit([dist, sc])
        rows.append(row(f"{prefix} + background score correlation", g_a,
                        beta_score_corr=float(g_a.params[2]), p_score_corr=float(g_a.pvalues[2])))
        log(f"{prefix} + background score correlation: beta = {g_a.params[1]:+.4f}", level=1)
        null_row(f"{prefix} adjusted: label permutation null (partial Mantel)",
                 float(g_a.params[1]), na, "less")

    # ── the single non-tree edge ─────────────────────────────────────────────────
    for a_, b_ in NON_TREE_EDGE_SENSITIVITY:
        if a_ not in adj or b_ not in adj[a_]:
            continue
        adj2 = {k: set(v) for k, v in adj.items()}
        adj2[a_].discard(b_); adj2[b_].discard(a_)
        D2 = all_pairs_distance(adj2).reindex(index=nodes, columns=nodes).to_numpy(float)
        d2 = D2[ia, ib]
        if not np.isfinite(d2).all():
            log(f"dropping {a_}->{b_} disconnects the graph; skipped", level=1)
            continue
        g2 = fit([d2])
        rows.append(row(f"{prefix} sensitivity: graph without {a_}->{b_}", g2))
        log(f"{prefix} without {a_}->{b_}: beta = {g2.params[1]:+.4f}", level=1)
    return rows, fold, beta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path,
                    help="output directory of 01_build_matrix.py")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--n-perm", type=int, default=C.N_PERMUTATIONS)
    ap.add_argument("--rule", choices=("raw", "fdr"), default=None,
                    help="calling rule; default from config.primary_rule()")
    ap.add_argument("--level", type=float, default=None,
                    help="threshold for --rule; default from config.primary_rule()")
    ap.add_argument("--fdr", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--metric", default=C.SIGNIF_METRIC)
    ap.add_argument("--pairs-only", action="store_true",
                    help="skip the variant-level curveball null and leave "
                         "tables/coherence_variant_level.csv untouched. The "
                         "pair-level work takes seconds; the curveball null "
                         "takes hours and does not depend on anything below it.")
    args = ap.parse_args()

    _kind, _level, _rule_label = C.primary_rule()
    if args.fdr is not None:
        args.rule, args.level = "fdr", args.fdr
    args.rule = args.rule or _kind
    args.level = args.level if args.level is not None else _level
    rule_label = (f"empirical P < {args.level:g} within cell type" if args.rule == "raw"
                  else f"BH-FDR < {args.level:g} within cell type")

    inp = args.input_dir
    out = args.output_dir or inp
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    qc = QCReport("Lineage-coherence QC")
    rng = np.random.default_rng(C.RANDOM_SEED)

    banner("Loading")
    long = pd.read_parquet(inp / "variant_celltype.parquet")
    sim_path = inp / "celltype_similarity.parquet"
    sim = pd.read_parquet(sim_path) if sim_path.exists() else None
    if sim is None:
        log("celltype_similarity.parquet not found; the pair-level regression cannot "
            "adjust for model similarity and will be reported unadjusted only", level=1)

    nodes = [ct for ct in C.universe("graph14") if ct in set(long["cell_type"])]
    adj = build_graph(nodes)
    Dd = all_pairs_distance(adj)
    Dd.to_csv(out / "tables/graph_distances.csv")
    idx = {n: i for i, n in enumerate(nodes)}
    adj_idx = [{idx[m] for m in adj[n]} for n in nodes]
    D = Dd.to_numpy(float)

    n_edges = sum(len(v) for v in adj.values()) // 2
    log(f"graph: {len(nodes)} nodes, {n_edges} undirected edges", level=1)
    if np.isinf(D).any():
        n_unreach = int(np.isinf(D).sum() // 2)
        log(f"WARNING: {n_unreach} node pair(s) unreachable, the graph is disconnected; "
            f"distance statistics are computed over reachable pairs only", level=1)

    qc.section("design")
    qc.add("universe", f"graph14 ({len(nodes)} nodes, {n_edges} edges)")
    qc.add("nodes", ", ".join(nodes))
    qc.add("excluded from all17",
           ", ".join(ct for ct in C.universe("all17") if ct not in set(nodes)))
    qc.note("    K562 is a leukaemia line with no position in normal haematopoiesis; the "
            "remaining exclusions have no graph node. Both are excluded by construction, "
            "not by threshold.")
    qc.add("significance rule",
           f"{rule_label}, accessible only")
    qc.add("effect statistic", args.metric)

    # ── significance matrix on the graph universe ────────────────────────────
    sub = long[long["cell_type"].isin(nodes)].copy()
    piv = lambda col, fill: (sub.pivot(index="variant_idx", columns="cell_type",
                                       values=col)
                             .reindex(columns=nodes).fillna(fill))
    QV = piv("qval", 1.0).to_numpy(float)
    ACC = piv("accessible", False).to_numpy(bool)
    LFC = piv("logfc", np.nan).to_numpy(float)
    if "called" in long.columns and args.rule == _kind and args.level == _level:
        SIG = piv("called", False).to_numpy(bool)
    elif args.rule == "raw":
        SIG = ACC & (piv("pval", 1.0).to_numpy(float) < args.level)
    else:
        SIG = ACC & (QV < args.level)

    breadth = SIG.sum(axis=1)
    log(f"variants with breadth >= 1: {(breadth >= 1).sum():,}; "
        f">= 2: {(breadth >= 2).sum():,}", level=1)

    # ── test 1: variant-level coherence ─────────────────────────────────────
    if args.pairs_only:
        banner("Variant-level coherence  (skipped: --pairs-only)")
        log("coherence_variant_level.csv left as it stands", level=1)
        qc.section("test 1: variant-level coherence")
        qc.add("status", "skipped (--pairs-only); existing table not overwritten")
        # Downstream the figure needs the variant-level result, so the existing table is
        # read back rather than recomputed. null_draws is empty in this mode, so panel a
        # plots the observed values without its null cloud.
        _vl = out / "tables/coherence_variant_level.csv"
        var_res = pd.read_csv(_vl) if _vl.exists() else pd.DataFrame()
        null_draws = {}
        if not len(var_res):
            log("no existing coherence_variant_level.csv; panel a will be empty", level=1)
    else:
        banner("Variant-level coherence")
        obs = coherence_stats(SIG, D, adj_idx)
        log(f"observed: {obs['n_multi']:,} multi-context variants, "
            f"{obs['frac_connected']:.3f} connected, "
            f"mean pairwise distance {obs['mean_pair_distance']:.3f}", level=1)

        if obs["n_multi"] < 10:
            log("fewer than 10 multi-context variants; the null is not worth computing "
                "and the variant-level test is reported as underpowered", level=1)
            null_draws = {k: np.array([]) for k in ("frac_connected", "mean_pair_distance")}
        else:
            # Rewire only the rows that can move: breadth-0 rows are inert, and restricting
            # to accessible-somewhere variants keeps the null on the same support.
            movable = breadth > 0
            SIG_m = SIG[movable]
            null_draws = {"frac_connected": [], "mean_pair_distance": []}
            for it in range(args.n_perm):
                SIG_p = curveball(SIG_m, rng)
                st = coherence_stats(SIG_p, D, adj_idx)
                null_draws["frac_connected"].append(st["frac_connected"])
                null_draws["mean_pair_distance"].append(st["mean_pair_distance"])
                if (it + 1) % max(1, args.n_perm // 10) == 0:
                    log(f"  {it + 1}/{args.n_perm}", level=1)
            null_draws = {k: np.asarray(v, float) for k, v in null_draws.items()}

        rows = []
        for name, tail in (("frac_connected", "greater"), ("mean_pair_distance", "less")):
            draws = null_draws[name][np.isfinite(null_draws[name])]
            o = obs[name]
            if len(draws) == 0 or not np.isfinite(o):
                p = np.nan
            elif tail == "greater":
                p = (1 + (draws >= o).sum()) / (1 + len(draws))
            else:
                p = (1 + (draws <= o).sum()) / (1 + len(draws))
            rows.append(dict(statistic=name, observed=o,
                             null_mean=draws.mean() if len(draws) else np.nan,
                             null_sd=draws.std() if len(draws) else np.nan,
                             n_draws=len(draws), tail=tail, p_permutation=p,
                             n_multi_context=obs["n_multi"]))
        var_res = pd.DataFrame(rows)
        var_res.to_csv(out / "tables/coherence_variant_level.csv", index=False)

        qc.section("test 1: variant-level coherence (upper bound)")
        qc.add("multi-context variants (breadth >= 2)", f"{obs['n_multi']:,}")
        for _, r in var_res.iterrows():
            qc.add(f"{r['statistic']} observed", f"{r['observed']:.4f}")
            qc.add(f"{r['statistic']} null mean (sd)",
                   f"{r['null_mean']:.4f} ({r['null_sd']:.4f})")
            qc.add(f"{r['statistic']} permutation p ({r['tail']} tail)",
                   f"{r['p_permutation']:.3g}")
        qc.note("    The curveball null preserves each variant's breadth and each cell "
                "type's number of significant variants, but destroys cell-cell correlation "
                "entirely. Adjacent cell types have near-duplicate models, so this null "
                "makes almost any observed structure look significant. These numbers are an "
                "upper bound on lineage coherence; test 2 adjusts for model similarity.")

    # ── test 2: pair-level, against the independence expectation ────────────
    banner("Pair-level coherence against the independence expectation")
    pair_rows = []
    for a, b in combinations(range(len(nodes)), 2):
        both_acc = ACC[:, a] & ACC[:, b]
        n_both_acc = int(both_acc.sum())
        if n_both_acc == 0:
            continue
        n_cosig = int((SIG[:, a] & SIG[:, b]).sum())
        n_either = int((SIG[:, a] | SIG[:, b])[both_acc].sum())
        pair_rows.append(dict(
            cell_type_a=nodes[a], cell_type_b=nodes[b],
            graph_distance=D[a, b],
            n_accessible_both=n_both_acc,
            n_cosignificant=n_cosig,
            # active counts restricted to the pair's shared-accessible set, so the
            # expectation can be built from rates conditional on that set rather than
            # from each cell type's marginal rate
            n_active_a_both=int(SIG[both_acc, a].sum()),
            n_active_b_both=int(SIG[both_acc, b].sum()),
            cosig_rate=n_cosig / n_both_acc,
            jaccard=n_cosig / n_either if n_either else np.nan,
        ))
    pairs = pd.DataFrame(pair_rows)

    # The covariate has to measure MODEL REDUNDANCY, not shared signal. A correlation
    # computed over all accessible variants includes the co-significant ones, so it
    # partly measures the very sharing the regression is trying to explain, and
    # adjusting for it would subtract the signal along with the confound. The primary
    # covariate is therefore the score correlation over variants accessible in both and
    # significant in NEITHER, model agreement on the null background. The all-variant
    # version from 01 is carried alongside as a sensitivity check.
    STAT_all = piv(args.metric, np.nan).to_numpy(float)
    corr_bg, corr_all = [], []
    for a, b in zip(pairs["cell_type_a"], pairs["cell_type_b"]):
        i, j = idx[a], idx[b]
        both = ACC[:, i] & ACC[:, j] & np.isfinite(STAT_all[:, i]) & np.isfinite(STAT_all[:, j])
        bg = both & ~SIG[:, i] & ~SIG[:, j]
        corr_all.append(np.corrcoef(STAT_all[both, i], STAT_all[both, j])[0, 1]
                        if both.sum() > 10 else np.nan)
        corr_bg.append(np.corrcoef(STAT_all[bg, i], STAT_all[bg, j])[0, 1]
                       if bg.sum() > 10 else np.nan)
    pairs["score_corr"] = corr_bg              # primary covariate
    pairs["score_corr_all_variants"] = corr_all
    pairs["n_background_both"] = [
        int((ACC[:, idx[a]] & ACC[:, idx[b]] & ~SIG[:, idx[a]] & ~SIG[:, idx[b]]).sum())
        for a, b in zip(pairs["cell_type_a"], pairs["cell_type_b"])]

    if sim is not None:
        key = {tuple(sorted((r["cell_type_a"], r["cell_type_b"]))):
               r.get("score_corr_accessible_both", np.nan)
               for _, r in sim.iterrows()}
        pairs["score_corr_from_01"] = [
            key.get(tuple(sorted((a, b))), np.nan)
            for a, b in zip(pairs["cell_type_a"], pairs["cell_type_b"])]

    pairs = pairs[np.isfinite(pairs["graph_distance"])].copy()
    # expected_cosignificant and informative are assigned by the regression below,
    # so the table is written after it, a reader inspecting the excluded pairs needs
    # both columns, and the figure reads them from this frame.
    _pair_csv = out / "tables/coherence_pair_level.csv"
    log(f"pairs: {len(pairs)}; with a score correlation: "
        f"{int(pairs['score_corr'].notna().sum())}", level=1)

    # ── primary specification: co-significance against independence ──────────
    # The logit-rate regression below is confounded and is kept only as a labelled
    # sensitivity row. The confound is structural, not statistical: a pair contributes a
    # co-significance rate of exactly zero whenever EITHER member nominates nothing at
    # all, and those cell types are not distributed evenly over the graph -- in this
    # design the non-nominating types are mostly progenitors, which sit close together,
    # so the forced zeros concentrate at SHORT distance and manufacture a positive
    # distance slope out of nothing. Worse, the continuity correction gives each forced
    # zero a pseudo-rate set by its accessibility denominator and then weights it by that
    # same denominator, so the largest weights in the fit carry no information about
    # co-significance. Adjusting for background score correlation appears to rescue the
    # sign only because that covariate is itself strongly anti-correlated with distance,
    # so it absorbs the artefact; the sign flip is a coincidence, not an adjustment.
    #
    # The primary model instead asks the question directly: is co-significance more
    # enriched over its independence expectation for lineage-adjacent pairs than for
    # distant ones? Pairs where either member never nominates have an independence
    # expectation of exactly zero, carry no information about distance, and are excluded
    # by that criterion rather than imputed. The offset makes per-cell-type discovery
    # rate part of the expectation instead of a nuisance covariate.
    reg_rows = []
    import statsmodels.api as sm
    # Two expected counts. PRIMARY: within-pair rates,
    #     expected = k_a * k_b / n_ab,
    # where n_ab is the number of variants accessible in both members and k_x the number
    # of those active in x -- the independence expectation of the 2 x 2 table of activity
    # among the shared-accessible variants. SENSITIVITY: each cell type's panel-wide
    # active rate, n_ab * rate_a * rate_b, which assumes the rate among shared-accessible
    # variants equals the rate over the cell type's whole accessible set. That set is
    # non-random and shrinks with distance, so the panel-wide version overstates what
    # distant pairs should share.
    rate = np.array([SIG[:, k].sum() / max(ACC[:, k].sum(), 1)
                     for k in range(SIG.shape[1])], float)
    pairs["rate_a"] = [rate[idx[a]] for a in pairs["cell_type_a"]]
    pairs["rate_b"] = [rate[idx[b]] for b in pairs["cell_type_b"]]
    pairs["expected_cosignificant"] = np.where(
        pairs["n_accessible_both"] > 0,
        pairs["n_active_a_both"] * pairs["n_active_b_both"]
        / pairs["n_accessible_both"].clip(lower=1), 0.0)
    pairs["expected_cosignificant_panelwide"] = (pairs["n_accessible_both"]
                                                 * pairs["rate_a"] * pairs["rate_b"])
    pairs["informative"] = pairs["expected_cosignificant"] > 0
    n_zero = int((~pairs["informative"]).sum())
    log(f"{n_zero} of {len(pairs)} pairs have a zero independence expectation and are "
        f"excluded from the pair model", level=1)
    for col, e in (("obs_over_expected", "expected_cosignificant"),
                   ("obs_over_expected_panelwide", "expected_cosignificant_panelwide")):
        pairs[col] = np.where(pairs[e] > 0, pairs["n_cosignificant"] / pairs[e].replace(0, np.nan),
                              np.nan)
    pairs.to_csv(_pair_csv, index=False)
    if (~pairs["informative"]).any() and pairs.loc[~pairs["informative"], "n_cosignificant"].sum():
        raise ValueError("a pair with zero independence expectation carries co-significant "
                         "variants; the activity and accessibility matrices disagree")
    inf = pairs.loc[pairs["informative"]].copy()

    pairs_fold = None
    if len(inf) >= 6:
        rows_p, pairs_fold, _ = fit_pair_models(
            inf, inf["expected_cosignificant"].to_numpy(float), D, nodes, adj,
            prefix="primary", n_perm=PRIMARY_LABEL_PERMUTATIONS, seed=C.RANDOM_SEED,
            n_zero=n_zero, null_contrast=True)
        reg_rows += rows_p
        pw = inf["expected_cosignificant_panelwide"].to_numpy(float) > 0
        if pw.sum() >= 6:
            rows_s, fold_s, _ = fit_pair_models(
                inf.loc[pw], inf.loc[pw, "expected_cosignificant_panelwide"].to_numpy(float),
                D, nodes, adj, prefix="panel-wide rates (sensitivity)",
                n_perm=NODE_LABEL_PERMUTATIONS, seed=C.RANDOM_SEED + 2,
                n_zero=int(len(pairs) - pw.sum()), null_contrast=False)
            reg_rows += rows_s
            qc.add("panel-wide-rate expectation (sensitivity): observed over expected",
                   f"adjacent {fold_s['adjacent']:.1f}-fold, distant {fold_s['distant']:.1f}-fold")
        for lab in ("adjacent", "distant"):
            log(f"  {lab}: {pairs_fold['observed_' + lab]:,.0f} observed / "
                f"{pairs_fold['expected_' + lab]:,.1f} expected = {pairs_fold[lab]:.1f}-fold", level=1)
    # Logit-scale outcome: co-significance rates are proportions bounded near zero, and
    # OLS on the raw rate would be heteroscedastic and could predict negative rates.
    if len(pairs) >= 6:

        y = pairs["cosig_rate"].to_numpy(float)
        eps = 0.5 / pairs["n_accessible_both"].to_numpy(float)  # continuity correction
        y = np.clip(y, eps, 1 - eps)
        logit_y = np.log(y / (1 - y))
        w = pairs["n_accessible_both"].to_numpy(float)

        specs = [("confounded sensitivity, unadjusted", ["graph_distance"])]
        if pairs["score_corr"].notna().sum() >= 6:
            specs.append(("confounded sensitivity, adjusted for background "
                          "score correlation",
                          ["graph_distance", "score_corr"]))
        if pairs["score_corr_all_variants"].notna().sum() >= 6:
            specs.append(("confounded sensitivity, adjusted for all-variant "
                          "score correlation",
                          ["graph_distance", "score_corr_all_variants"]))

        for label, terms in specs:
            keep = pairs[terms].notna().all(axis=1).to_numpy()
            X = sm.add_constant(pairs.loc[keep, terms].to_numpy(float))
            fit = sm.WLS(logit_y[keep], X, weights=w[keep]).fit()
            ci = fit.conf_int()
            reg_rows.append(dict(
                model=label, n_pairs=int(keep.sum()),
                beta_distance=float(fit.params[1]),
                se_distance=float(fit.bse[1]),
                ci_lo=float(ci[1, 0]), ci_hi=float(ci[1, 1]),
                p_distance=float(fit.pvalues[1]),
                beta_score_corr=float(fit.params[2]) if len(terms) > 1 else np.nan,
                p_score_corr=float(fit.pvalues[2]) if len(terms) > 1 else np.nan,
                r_squared=float(fit.rsquared)))
            log(f"{label}: beta(distance) = {fit.params[1]:+.4f} "
                f"[{ci[1,0]:+.4f}, {ci[1,1]:+.4f}], p = {fit.pvalues[1]:.3g}", level=1)
    regression = pd.DataFrame(reg_rows)
    regression.to_csv(out / "tables/coherence_regression.csv", index=False)

    qc.section("test 2: pair-level co-significance against independence")
    qc.add("cell-type pairs", len(pairs))
    qc.add("cell types nominating at least one variant",
           f"{int((rate > 0).sum())} of {len(rate)}")
    qc.add("pairs excluded, zero independence expectation", n_zero)
    qc.add("pairs in the primary model", len(inf))
    if pairs_fold is not None:
        qc.add("enrichment over independence (within-pair), lineage-adjacent",
               f"{pairs_fold['adjacent']:.1f}-fold ({pairs_fold['observed_adjacent']:,.0f} observed, "
               f"{pairs_fold['expected_adjacent']:,.0f} expected)")
        qc.add("enrichment over independence (within-pair), distant",
               f"{pairs_fold['distant']:.1f}-fold ({pairs_fold['observed_distant']:,.0f} observed, "
               f"{pairs_fold['expected_distant']:,.0f} expected)")
    qc.note("    The primary model is the count model with an independence offset built "
            "from within-pair called rates; the panel-wide-rate offset is a labelled "
            "sensitivity. The logit-rate rows are labelled 'confounded sensitivity': "
            "pairs whose independence expectation is exactly zero are imputed a "
            "pseudo-rate from their accessibility denominator and weighted by it, and "
            "because the cell types that call nothing cluster in one region of the "
            "graph, those forced zeros carry a distance signal of their own.")
    qc.add("outcome", "logit of co-significance rate among variants accessible in both, "
                      "weighted by that denominator")
    if len(regression):
        for _, r in regression.iterrows():
            qc.add(f"{r['model']}: beta(distance)",
                   f"{r['beta_distance']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], "
                   f"p = {r['p_distance']:.3g}, n = {int(r['n_pairs'])}")
            if np.isfinite(r["beta_score_corr"]):
                qc.add(f"{r['model']}: beta(score correlation)",
                       f"{r['beta_score_corr']:+.4f}, p = {r['p_score_corr']:.3g}")
        base = regression.iloc[0]["beta_distance"]
        for _, r in regression.iloc[1:].iterrows():
            shrink = 1 - abs(r["beta_distance"]) / abs(base) if base else np.nan
            qc.add(f"attenuation of beta(distance), {r['model']}",
                   f"{shrink:.1%}" if np.isfinite(shrink) else "not computable")
        qc.note("    A negative beta(distance) means co-significance falls with graph "
                "distance, which is lineage coherence. The comparison that matters is "
                "between the unadjusted row and the BACKGROUND-adjusted row. The "
                "background covariate is the score correlation over variants accessible "
                "in both cell types and significant in neither, so it measures model "
                "redundancy on the null background and cannot absorb the shared signal "
                "itself. If the distance effect survives that adjustment, sharing tracks "
                "the lineage and not merely model redundancy; if it collapses, the "
                "honest reading is that adjacent cell types have similar models and "
                "nothing further has been shown.")
        qc.note("    The all-variant row is a sensitivity check and is the WRONG "
                "covariate to draw the primary conclusion from: because it is computed "
                "over the co-significant variants as well, it is partly a measure of the "
                "sharing being explained, and adjusting for it removes real signal along "
                "with the confound. Expect it to attenuate the distance effect more than "
                "the background covariate does; a large gap between the two adjusted "
                "rows is that absorption, not a finding.")
    else:
        qc.add("regression", "not fitted (fewer than 6 usable pairs)")

    # ── direction flips along directed edges ─────────────────────────────────
    banner("Direction flips")
    from scipy import stats as sps

    flip_rows = []
    for parent, child in C.LINEAGE_EDGES_DIRECTED:
        if parent not in idx or child not in idx:
            continue
        a, b = idx[parent], idx[child]
        both = SIG[:, a] & SIG[:, b] & np.isfinite(LFC[:, a]) & np.isfinite(LFC[:, b])
        n_both = int(both.sum())
        if n_both == 0:
            flip_rows.append(dict(parent=parent, child=child, n_both=0, n_flip=0,
                                  flip_rate=np.nan, ci_lo=np.nan, ci_hi=np.nan))
            continue
        flip = np.sign(LFC[both, a]) != np.sign(LFC[both, b])
        n_flip = int(flip.sum())
        # Jeffreys interval, with the conventional boundary corrections: the equal-tailed
        # interval does not contain the estimate at x = 0 or x = n, so those ends are
        # pinned to 0 and 1 respectively.
        lo, hi = sps.beta.ppf([0.025, 0.975],
                              n_flip + 0.5, n_both - n_flip + 0.5)
        lo = 0.0 if n_flip == 0 else float(lo)
        hi = 1.0 if n_flip == n_both else float(hi)
        flip_rows.append(dict(parent=parent, child=child, n_both=n_both, n_flip=n_flip,
                              flip_rate=n_flip / n_both, ci_lo=lo, ci_hi=hi))
    flips = pd.DataFrame(flip_rows)
    flips.to_csv(out / "tables/direction_flips.csv", index=False)

    usable = flips[flips["n_both"] >= 5]
    log(f"edges with >= 5 variants significant in both: {len(usable)}/{len(flips)}",
        level=1)
    qc.section("direction flips along directed edges")
    qc.add("directed edges evaluated", len(flips))
    qc.add("edges with >= 5 variants significant in both", len(usable))
    if len(usable):
        tot_both = int(usable["n_both"].sum())
        tot_flip = int(usable["n_flip"].sum())
        qc.add("pooled flip rate", f"{tot_flip}/{tot_both} = {tot_flip/tot_both:.3f}")
        hi = usable.loc[usable["flip_rate"].idxmax()]
        qc.add("highest-flipping edge",
               f"{hi['parent']} -> {hi['child']}: {int(hi['n_flip'])}/"
               f"{int(hi['n_both'])} = {hi['flip_rate']:.3f}")
    qc.note("    A sign flip in a model's prediction is not by itself evidence of a "
            "biological reversal: it can also arise when the variant sits in a motif "
            "whose flanking context or cofactor availability differs between the two "
            "cell types. Reported as a hypothesis-generating list, not a claim.")

    # ── figure ───────────────────────────────────────────────────────────────
    figs = figure_hierarchy(var_res, null_draws, pairs, flips, regression,
                            out / "figures/hierarchy_coherence")
    for f in figs:
        log(f"figure -> {f}")

    qc.write(out / "hierarchy_qc.txt")
    if len(regression):
        # the headline is the primary count model, not the last one fitted
        hit = regression[regression["model"].str.startswith("primary: count")]
        r = (hit.iloc[0] if len(hit) else regression.iloc[0])
        log(f"headline, {r['model']}: beta(distance) = {r['beta_distance']:+.4f}, "
            f"p = {r['p_distance']:.3g}")
    log("done")


if __name__ == "__main__":
    main()