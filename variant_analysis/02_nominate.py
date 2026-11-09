#!/usr/bin/env python
"""
02_nominate.py: variant nomination, flagship tier and threshold sensitivity.

A variant is nominated on model evidence alone: it is called (accessible and
significant under the primary rule in config.py) in at least one cell type. The
fine-mapping posterior enters as an annotation and as part of the flagship tier, not
as a requirement for nomination. Requiring PP >= 0.5 for a variant to be called
active would conflate two independent questions (does the model predict a regulatory
effect, and is the variant causal for a blood trait) and would make it impossible for
the nominated set to contain a model-predicted effect at a variant the fine-mapping
missed.

Significance, not an absolute magnitude cut, defines activity: a fixed |score|
threshold means something different in each cell type because cell types differ in
dynamic range, whereas an empirical P against each cell type's own shuffled null puts
all 17 on a common footing.

Tiers
-----
  nominated   called in >= 1 cell type under the primary rule
  flagship    nominated AND empirical P < 0.001 somewhere AND PP >= 0.5
  model-only  nominated AND empirical P < 0.001 somewhere AND (PP < 0.5 or absent)

Outputs
-------
  nominated_variants.parquet     nominated set with tier, breadth and annotations
  tables/nomination_summary.csv  counts at every tier
  tables/threshold_sensitivity.csv
  tables/per_celltype_counts.csv per-cell-type counts and called rates
  figures/threshold_sensitivity.{png,pdf}

Usage
-----
  python 02_nominate.py --input-dir <01 output>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C            # noqa: E402
import s4plot                 # noqa: E402
from s4lib import QCReport, banner, log  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Rule application
# ──────────────────────────────────────────────────────────────────────────────

def apply_rule(long: pd.DataFrame, kind: str, level: float,
               metric_p: str = "pval") -> pd.Series:
    """Boolean call per variant-cell-type row under one thresholding rule.

    ``kind`` is 'fdr' (BH within cell type, recomputed at this level from the stored
    q-values) or 'raw' (uncorrected empirical p). Accessibility is required either way:
    a variant in closed chromatin has no predicted effect to interpret.
    """
    acc = long["accessible"].to_numpy()
    if kind == "fdr":
        q = long["qval"].to_numpy()
        return pd.Series(acc & np.isfinite(q) & (q < level), index=long.index)
    if kind == "raw":
        return pd.Series(acc & (long[metric_p].to_numpy() < level), index=long.index)
    raise ValueError(f"unknown rule kind: {kind}")


def breadth_from_calls(long: pd.DataFrame, called: pd.Series,
                       n_variants_index: pd.Index) -> pd.Series:
    b = (long.loc[called.to_numpy()]
             .groupby("variant_idx", observed=True)
             .size())
    return b.reindex(n_variants_index).fillna(0).astype(np.int16)


def classify_breadth(breadth: pd.Series, n_contexts: int) -> pd.Series:
    cats = C.breadth_categories(n_contexts)
    out = pd.Series(pd.NA, index=breadth.index, dtype="object")
    for name, (lo, hi) in cats.items():
        out[(breadth >= lo) & (breadth <= hi)] = name
    return pd.Series(pd.Categorical(out, categories=list(cats), ordered=True),
                     index=breadth.index)


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────

def figure_sensitivity(sens: pd.DataFrame, comp: pd.DataFrame, path: Path,
                       primary_label: str) -> list[Path]:
    """Two panels: how many variants each rule nominates, and how the sharing
    composition responds to the threshold."""
    import matplotlib.pyplot as plt

    s4plot.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(s4plot.FIG_W_DOUBLE, 2.5),
                             gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.08})

    # ── panel a: nominated count per rule ────────────────────────────────────
    ax = axes[0]
    order = sens.sort_values("n_nominated").reset_index(drop=True)
    y = np.arange(len(order))
    is_primary = order["label"] == primary_label
    colours = np.where(is_primary, s4plot.FOCAL, s4plot.COMPARATOR)

    ax.hlines(y, 0, order["n_nominated"], color=s4plot.LIGHT_GREY, lw=0.8, zorder=1)
    ax.scatter(order["n_nominated"], y, s=22, color=colours, zorder=3,
               edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(order["label"])
    for tick, prim in zip(ax.get_yticklabels(), is_primary):
        if prim:
            tick.set_color(s4plot.FOCAL)
            tick.set_fontweight("bold")
    ax.set_xlabel("Variants nominated in at least one cell type")
    ax.set_title("Nominated set size across thresholds")
    ax.margins(x=0.16, y=0.06)

    headline = order.loc[is_primary, "n_nominated"]
    if len(headline):
        yi = int(np.flatnonzero(is_primary.to_numpy())[0])
        ax.annotate(f"{int(headline.iloc[0]):,}",
                    (headline.iloc[0], yi), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=s4plot.SMALL,
                    color=s4plot.FOCAL, fontweight="bold")

    # ── panel b: sharing composition per rule ────────────────────────────────
    ax = axes[1]
    cats = [c for c in comp.columns if c != "label"]
    comp_o = comp.set_index("label").loc[order["label"]]
    frac = comp_o[cats].div(comp_o[cats].sum(axis=1), axis=0).fillna(0.0)
    left = np.zeros(len(frac))
    ramp = s4plot.ORDINAL[-len(cats):] if len(cats) <= len(s4plot.ORDINAL) else None
    for i, c in enumerate(cats):
        colour = ramp[i] if ramp else plt.cm.Blues((i + 1) / len(cats))
        ax.barh(y, frac[c].to_numpy(), left=left, height=0.62, color=colour,
                edgecolor="white", linewidth=0.5, label=c)
        left = left + frac[c].to_numpy()
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of nominated variants")
    ax.set_title("Sharing composition across thresholds")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=len(cats),
              fontsize=s4plot.SMALL)

    s4plot.panel_letter(axes[0], "a", dx=-0.52)
    s4plot.panel_letter(axes[1], "b", dx=-0.04)
    return s4plot.save(fig, path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=C.OUTPUT_DIR)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    inp = args.input_dir
    out = args.output_dir or inp
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    var = pd.read_parquet(inp / "variants.parquet")
    long = pd.read_parquet(inp / "variant_celltype.parquet")
    cell_types = [c for c in C.CELL_TYPES if c in set(long["cell_type"].astype(str))]
    n_ct = len(cell_types)

    qc = QCReport("Nomination QC")
    qc.section("inputs")
    qc.add("variants", len(var))
    qc.add("variant x cell-type rows", len(long))
    qc.add("cell types present", ", ".join(cell_types))

    banner("Primary nomination")
    idx = var.set_index("variant_idx").index

    _kind, _level, _rule_label = C.primary_rule()
    if "called" in long.columns:
        primary_called = long["called"].astype(bool)
    else:
        primary_called = apply_rule(long, _kind, _level)
    breadth = breadth_from_calls(long, primary_called, idx)

    nom = var.set_index("variant_idx").copy()
    nom["breadth"] = breadth
    nom["nominated"] = nom["breadth"] > 0
    nom["sharing"] = classify_breadth(nom["breadth"], n_ct)

    # Flagship: model evidence strong AND fine-mapping supports causality.
    has_pip = nom["PIP"].notna()
    nom["flagship"] = (nom["nominated"]
                       & (nom["min_pval_panel"] < C.FLAGSHIP_P)
                       & has_pip & (nom["PIP"] >= C.FLAGSHIP_PIP))
    # Strong model evidence, weak or absent posterior (the discordant set).
    nom["model_only"] = (nom["nominated"]
                         & (nom["min_pval_panel"] < C.FLAGSHIP_P)
                         & (~has_pip | (nom["PIP"] < C.FLAGSHIP_PIP)))

    qc.section("tiers")
    qc.add("nominated (%s, in >=1 cell type)" % _rule_label,
           int(nom["nominated"].sum()))
    qc.add("  of which carry a fine-mapping posterior",
           int((nom["nominated"] & has_pip).sum()))
    qc.add("  of which PP >= %.1f" % C.FLAGSHIP_PIP,
           int((nom["nominated"] & has_pip & (nom["PIP"] >= C.FLAGSHIP_PIP)).sum()))
    qc.add("flagship (P < %g and PP >= %.1f)" % (C.FLAGSHIP_P, C.FLAGSHIP_PIP),
           int(nom["flagship"].sum()))
    qc.add("model-only (P < %g, PP < %.1f or absent)" % (C.FLAGSHIP_P, C.FLAGSHIP_PIP),
           int(nom["model_only"].sum()))
    for name in nom["sharing"].cat.categories:
        qc.add(f"sharing {name}", int((nom["sharing"] == name).sum()))

    # ── magnitude-only comparator ────────────────────────────────────────────
    qc.section("comparison with a magnitude-only rule")
    mag = (nom["max_abs_panel"] >= C.MAGNITUDE_ONLY_THRESH) & has_pip & \
          (nom["PIP"] >= C.MAGNITUDE_ONLY_PIP)
    qc.add(f"magnitude-only rule (|score| >= {C.MAGNITUDE_ONLY_THRESH}, "
           f"PP >= {C.MAGNITUDE_ONLY_PIP})", int(mag.sum()))
    qc.add("magnitude-only AND flagship", int((mag & nom["flagship"]).sum()))
    qc.add("magnitude-only but NOT nominated", int((mag & ~nom["nominated"]).sum()))
    qc.add("flagship but NOT magnitude-only", int((nom["flagship"] & ~mag).sum()))
    nom["magnitude_only"] = mag

    # ── per-cell-type counts, power-normalised ───────────────────────────────
    banner("Per-cell-type counts")
    qc.section("per-cell-type detection")
    long["_called"] = primary_called.to_numpy()
    per_ct = (long.groupby("cell_type", observed=True)
                  .agg(n_testable=("accessible", "sum"),
                       n_called=("_called", "sum"))
                  .reindex(cell_types).reset_index())
    per_ct["rate"] = per_ct["n_called"] / per_ct["n_testable"]
    per_ct["lineage"] = [C.CELL_TYPE_META[ct]["lineage"] for ct in per_ct["cell_type"]]
    per_ct["dev_class"] = [C.CELL_TYPE_META[ct]["dev_class"] for ct in per_ct["cell_type"]]
    per_ct.to_csv(out / "tables/per_celltype_counts.csv", index=False)
    qc.add("accessible-variant count range across cell types",
           f"{per_ct['n_testable'].min():,} to {per_ct['n_testable'].max():,}")
    qc.add("nomination rate range",
           f"{per_ct['rate'].min():.4f} to {per_ct['rate'].max():.4f}")
    qc.note("    Raw per-cell-type counts partly reflect how many variants each cell type "
            "can test at all; compare the rate column across cell types.")

    # ── threshold sensitivity ────────────────────────────────────────────────
    banner("Threshold sensitivity")
    rules = ([("fdr", lv, f"FDR < {lv:g}") for lv in C.SENSITIVITY_FDR]
             + [("raw", lv, f"P < {lv:g}") for lv in C.SENSITIVITY_RAW_P])
    primary_label = (f"P < {_level:g}" if _kind == "raw" else f"FDR < {_level:g}")

    sens_rows, comp_rows = [], []
    for kind, level, label in rules:
        for metric_label, pcol in (("logfc", "pval"),
                                   ("logfc x JSD", "pval_secondary")):
            if kind == "fdr" and pcol != "pval":
                continue   # q-values are stored for the primary statistic only
            called = apply_rule(long, kind, level, metric_p=pcol)
            b = breadth_from_calls(long, called, idx)
            cls = classify_breadth(b, n_ct)
            nominated = b > 0
            full_label = label if metric_label == "logfc" else f"{label} ({metric_label})"
            sens_rows.append(dict(
                rule=kind, level=level, metric=metric_label, label=full_label,
                n_nominated=int(nominated.sum()),
                median_breadth=float(b[nominated].median()) if nominated.any() else 0.0,
                n_flagship=int((nominated
                                & (nom["min_pval_panel"] < C.FLAGSHIP_P)
                                & has_pip & (nom["PIP"] >= C.FLAGSHIP_PIP)).sum()),
                frac_with_pip=float(has_pip[nominated].mean()) if nominated.any() else np.nan,
            ))
            row = {"label": full_label}
            for name in cls.cat.categories:
                if name == "null":
                    continue
                row[name] = int(((cls == name) & nominated).sum())
            comp_rows.append(row)

    sens = pd.DataFrame(sens_rows)
    comp = pd.DataFrame(comp_rows).fillna(0)
    sens.to_csv(out / "tables/threshold_sensitivity.csv", index=False)
    qc.add("rules evaluated", len(sens))
    qc.add("nominated-set size range across rules",
           f"{sens['n_nominated'].min():,} to {sens['n_nominated'].max():,}")

    figs = figure_sensitivity(sens, comp, out / "figures/threshold_sensitivity",
                              primary_label)
    for f in figs:
        log(f"figure -> {f}")

    # ── write ────────────────────────────────────────────────────────────────
    keep = nom[nom["nominated"]].reset_index()
    keep.to_parquet(out / "nominated_variants.parquet", index=False)

    summary = pd.DataFrame([
        dict(tier="scored", n=len(var)),
        dict(tier="accessible in >=1 cell type", n=int(var["accessible_any"].sum())),
        dict(tier="nominated", n=int(nom["nominated"].sum())),
        dict(tier="flagship", n=int(nom["flagship"].sum())),
        dict(tier="model-only", n=int(nom["model_only"].sum())),
        dict(tier="magnitude-only rule", n=int(mag.sum())),
    ])
    summary.to_csv(out / "tables/nomination_summary.csv", index=False)

    qc.section("outputs")
    for f in ("nominated_variants.parquet", "tables/nomination_summary.csv",
              "tables/threshold_sensitivity.csv", "tables/per_celltype_counts.csv",
              "figures/threshold_sensitivity.png"):
        qc.add(f, f"{(out / f).stat().st_size / 1e3:.0f} kB")
    qc.write(out / "nomination_qc.txt")

    log(f"nominated {int(nom['nominated'].sum()):,} variants, "
        f"{int(nom['flagship'].sum()):,} flagship")
    log("done")


if __name__ == "__main__":
    main()
