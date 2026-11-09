#!/usr/bin/env python
"""
score_all_variants.py: per-variant model evidence for every scored variant, not only
the accessible ones.

`01_build_matrix.py` computes a calibrated empirical p for every variant in every cell
type, but it keeps only the testable subset in the long table, so `min_pval_panel` exists
for 302,830 variants rather than all 1,565,539. The enrichment replication needs the
whole set, because the variants outside every peak are the denominator that makes an
enrichment an enrichment.

This recomputes the same quantity for everything and nothing else:

  all_variant_scores.parquet
      variant_key, min_pval, max_abs_logfc, n_celltypes_scored

The empirical p is not re-derived here. `average_across_folds` and `empirical_pvalues`
are imported from the pipeline itself, so the p a variant gets from this script is the p
it would have got from `01`, by construction rather than by agreement. Averaging the
statistic across folds and testing it against the fold-averaged shuffled arm is what
makes the p calibrated; both arms carry the same inter-fold correlation and it cancels.

One cell type per worker, each writing `celltype_scores/<cell_type>.parquet`, and a cell
type whose file exists is skipped. An interrupted run costs one cell type.

Usage
-----
  python score_all_variants.py \
    --score-dir  <directory holding the per-cell-type fold score files> \
    --input-dir  <01_build_matrix.py output dir> \
    --threads    17
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from s4lib import QCReport, banner, empirical_pvalues, log, variant_key

_HERE = Path(__file__).resolve().parent


def load_pipeline_module():
    """Import 01_build_matrix.py, whose name is not a legal identifier.

    Its fold reader is reused rather than rewritten, so the p-values cannot diverge
    from those computed in 01_build_matrix.py.
    """
    path = _HERE / "01_build_matrix.py"
    if not path.exists():
        raise FileNotFoundError(f"expected {path} next to this script")
    spec = importlib.util.spec_from_file_location("_build_matrix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_celltype(cell_type: str, out_dir: Path, score_dir: Path | None = None) -> dict:
    if score_dir is not None:
        C.SCORE_DIR = score_dir
    dest = out_dir / f"{cell_type}.parquet"
    if dest.exists():
        return dict(cell_type=cell_type, status="skipped (output present)", n=-1,
                    n_dup=0, seconds=0.0)

    t0 = time.time()
    bm = load_pipeline_module()
    qc = QCReport(f"score_all:{cell_type}")
    metrics = [C.SIGNIF_METRIC, C.EFFECT_METRIC]

    ids, obs = bm.average_across_folds(cell_type, False, metrics, None, qc)
    _, nul = bm.average_across_folds(cell_type, True, [C.SIGNIF_METRIC], ids, qc)

    pval = empirical_pvalues(obs[C.SIGNIF_METRIC], nul[C.SIGNIF_METRIC])
    out = pd.DataFrame({
        "variant_key": variant_key(ids, C.VARIANT_KEY_COLUMNS).to_numpy(dtype=object),
        "pval": pval.astype(np.float32),
        "abs_logfc": np.asarray(obs[C.SIGNIF_METRIC], dtype=np.float32),
    })

    # A variant present in both source lists is scored twice under one key. The two rows
    # are the same variant through the same model, so their statistics must agree
    # exactly; this is checked, since differing scores under one key would mean the key
    # does not identify a variant.
    n_dup = 0
    dup = out["variant_key"].duplicated(keep=False)
    if dup.any():
        spread = out.loc[dup].groupby("variant_key")[["pval", "abs_logfc"]].nunique().max()
        differing = spread[spread > 1]
        if len(differing):
            examples = (out.loc[dup].groupby("variant_key")[["pval", "abs_logfc"]]
                        .nunique().query("pval > 1 or abs_logfc > 1").index[:3].tolist())
            raise ValueError(
                f"{cell_type}: rows sharing a variant key differ in "
                f"{list(differing.index)}, so the key does not identify a variant. "
                f"Examples: {examples}")
        before = len(out)
        out = out.drop_duplicates(subset=["variant_key"], keep="first")
        n_dup = before - len(out)

    out.to_parquet(dest, index=False)
    return dict(cell_type=cell_type, status="ok", n=len(out), n_dup=n_dup,
                seconds=round(time.time() - t0, 1))


def reduce_celltypes(cell_types: list[str], out_dir: Path,
                     variants: pd.DataFrame) -> pd.DataFrame:
    """Minimum p and maximum |logFC| across cell types, on the full variant table.

    Reduced against variants.parquet rather than against the union of the score files, so
    the output is one row per scored variant in the pipeline's own ordering and a variant
    missing from a cell type simply does not contribute to its minimum.
    """
    keys = variant_key(variants, C.VARIANT_KEY_COLUMNS)
    pos = pd.Series(np.arange(len(keys), dtype=np.int64), index=keys.to_numpy())
    assert not pos.index.has_duplicates, (
        "duplicate chr:pos:allele1:allele2 reached the reduction; main() collapses "
        "them, so this means the frame was built elsewhere")

    min_p = np.full(len(keys), np.inf, dtype=np.float64)
    max_a = np.full(len(keys), -np.inf, dtype=np.float64)
    n_ct = np.zeros(len(keys), dtype=np.int16)

    for ct in cell_types:
        df = pd.read_parquet(out_dir / f"{ct}.parquet")
        idx = pos.reindex(df["variant_key"].to_numpy()).to_numpy()
        ok = np.isfinite(idx)
        if not ok.all():
            log(f"  {ct}: {int((~ok).sum()):,} scored keys absent from variants.parquet, "
                f"ignored", level=1)
        i = idx[ok].astype(np.int64)
        p = df["pval"].to_numpy(np.float64)[ok]
        a = np.abs(df["abs_logfc"].to_numpy(np.float64))[ok]
        np.minimum.at(min_p, i, p)
        np.maximum.at(max_a, i, a)
        np.add.at(n_ct, i, 1)

    return pd.DataFrame({
        "variant_key": keys.to_numpy(dtype=object),
        "min_pval": np.where(n_ct > 0, min_p, np.nan).astype(np.float32),
        "max_abs_logfc": np.where(n_ct > 0, max_a, np.nan).astype(np.float32),
        "n_celltypes_scored": n_ct,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--score-dir", type=Path, default=None,
                    help="overrides config.SCORE_DIR if the scores have moved")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default <input-dir>/celltype_scores")
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()

    if args.score_dir is not None:
        C.SCORE_DIR = args.score_dir            # fold_score_file reads this at call time
    out_dir = args.out_dir or (args.input_dir / "celltype_scores")
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_types = list(C.CELL_TYPES)
    banner("Per-cell-type empirical p for every scored variant")
    log(f"{len(cell_types)} cell types, {len(C.FOLDS)} folds, observed and shuffled")

    workers = max(1, min(args.threads, len(cell_types)))

    def report(r: dict) -> None:
        log(f"  {r['cell_type']:<12s} {r['status']:<26s}"
            + (f" {r['n']:>10,} variants  {r['seconds']:>6.1f}s" if r["n"] >= 0 else "")
            + (f"  ({r['n_dup']:,} duplicate key(s) collapsed)"
               if r.get("n_dup") else ""))

    results = []
    if workers == 1:
        for ct in cell_types:
            try:
                r = run_celltype(ct, out_dir, args.score_dir)
            except Exception as exc:
                log(f"  {ct:<12s} FAILED: {exc}")
                results.append(dict(cell_type=ct, status=f"failed: {exc}", n=-1))
                continue
            results.append(r)
            report(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(run_celltype, ct, out_dir, args.score_dir): ct
                       for ct in cell_types}
            for fut in as_completed(futures):
                ct = futures[fut]
                try:
                    r = fut.result()
                except Exception as exc:
                    log(f"  {ct:<12s} FAILED: {exc}")
                    results.append(dict(cell_type=ct, status=f"failed: {exc}", n=-1))
                    continue
                results.append(r)
                report(r)

    failed = [r for r in results if str(r["status"]).startswith("failed")]
    if failed:
        raise SystemExit(f"{len(failed)} cell type(s) failed; re-run to retry only those")

    banner("Reducing across cell types")
    V = pd.read_parquet(args.input_dir / "variants.parquet")
    V, n_dup = C.collapse_duplicate_variants(
        V.assign(_key=variant_key(V, C.VARIANT_KEY_COLUMNS)), "_key")
    if n_dup:
        log(f"{n_dup:,} duplicate variant key(s) collapsed")
    table = reduce_celltypes(cell_types, out_dir, V)
    dest = args.input_dir / "all_variant_scores.parquet"
    table.to_parquet(dest, index=False)

    scored = int(table["n_celltypes_scored"].gt(0).sum())
    log(f"variants in table        {len(table):,}")
    log(f"scored in >=1 cell type  {scored:,}")
    log(f"median cell types scored {float(table['n_celltypes_scored'].median()):.0f}")
    log(f"min_pval below 0.01      {int((table['min_pval'] < 0.01).sum()):,}")
    log(f"wrote {dest}")


if __name__ == "__main__":
    sys.exit(main())
