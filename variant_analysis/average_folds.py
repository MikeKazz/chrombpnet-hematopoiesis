#!/usr/bin/env python
"""
average_folds.py

For each of the 17 blood cell type folders under the score directory,
average the `variant_scores.tsv` files across the 5 folds (fold0-fold4) and
write one summary file per cell type, following the same logic as
variant-scorer's `variant_summary_across_folds.py`:

  - Identifier columns (chr, pos, allele1, allele2, variant_id) are taken
    from fold0 after sorting all folds into a common row order.
  - For each score column in a fixed list, if present in the fold files,
    the arithmetic mean across the 5 folds is computed as `<score>.mean`.
  - If a matching `.pval` (or `_pval`) column exists for that score, the
    geometric mean across folds is computed as `<score>.mean.pval`.

Each fold's rows are sorted by the full identifier tuple (chr, pos, allele1,
allele2, variant_id) before averaging, rather than assuming the files share row
order, because `variant_id` alone is not a unique key in this dataset (it holds
a provenance label), so the full tuple is the join key.

Only `*_fold{0-4}.variant_scores.tsv` files are processed. `.peak_scores.tsv`
and `.variant_scores.shuffled.tsv` files are ignored entirely.

The `.mean.pval` columns are the geometric mean of per-fold p-values and are not
calibrated (see 01_build_matrix.py); these files are used only to cross-check the
fold-averaged statistics.

Usage:
    python average_folds.py --base-dir data/variant_scores [--dry-run]

Output:
    <base-dir>/<celltype>/<celltype>.mean.variant_scores.tsv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Hard-coded list of the 17 cell type subfolders to process.
CELL_TYPES = [
    "Bcell", "CD34_BM", "CD34_CB", "CD4T", "CD8T", "CLP", "CMP",
    "Ery", "GMP", "HSC", "K562", "LMPP", "MEP", "Mega", "Mono",
    "MPP", "NK",
]

# Number of folds expected for every cell type.
FOLDS = [0, 1, 2, 3, 4]

# Identifier columns used to align rows across folds (matches the
# `original` schema in variant-scorer's utils.io.get_variant_schema).
ID_COLUMNS = ["chr", "pos", "variant_id", "allele1", "allele2"]

# Fixed list of score columns to average, exactly as in
# variant-scorer's variant_summary_across_folds.py.
SCORE_COLUMNS = [
    "logfc", "abs_logfc", "jsd", "logfc_x_jsd", "abs_logfc_x_jsd",
    "active_allele_quantile", "logfc_x_active_allele_quantile",
    "abs_logfc_x_active_allele_quantile", "jsd_x_active_allele_quantile",
    "logfc_x_jsd_x_active_allele_quantile",
    "abs_logfc_x_jsd_x_active_allele_quantile",
    "quantile_change", "abs_quantile_change",
]


def geo_mean_overflow(iterable, axis=0):
    """Geometric mean, as in variant-scorer."""
    return np.exp(np.log(iterable).mean(axis=0))


def load_and_sort(fold_file):
    """Read a variant_scores.tsv file and sort rows by the identifier tuple."""
    df = pd.read_table(fold_file)
    missing = [c for c in ID_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{fold_file} is missing expected identifier column(s): {missing}"
        )
    df = df.sort_values(ID_COLUMNS, kind="mergesort").reset_index(drop=True)
    return df


def average_cell_type(base_dir, cell_type, dry_run=False):
    """Average the 5 fold variant_scores.tsv files for one cell type."""
    cell_dir = os.path.join(base_dir, cell_type)

    fold_files = []
    for fold in FOLDS:
        fpath = os.path.join(cell_dir, f"{cell_type}_fold{fold}.variant_scores.tsv")
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"[{cell_type}] Missing expected file: {fpath}")
        fold_files.append(fpath)

    # Load and sort every fold into a common row order.
    fold_dfs = {}
    for fold, fpath in zip(FOLDS, fold_files):
        fold_dfs[fold] = load_and_sort(fpath)

    # Use fold0 (post-sort) as the reference for identifier columns.
    ref = fold_dfs[FOLDS[0]]
    variant_scores = ref[ID_COLUMNS].copy()

    # Sanity-check that every fold has the identical, aligned identifier
    # columns after sorting (same variants present in every fold).
    for fold in FOLDS:
        df = fold_dfs[fold]
        if len(df) != len(ref):
            raise ValueError(
                f"[{cell_type}] fold{fold} has {len(df)} rows, "
                f"expected {len(ref)} (mismatch with fold{FOLDS[0]})"
            )
        for col in ID_COLUMNS:
            if not df[col].tolist() == ref[col].tolist():
                raise ValueError(
                    f"[{cell_type}] fold{fold} column '{col}' does not match "
                    f"fold{FOLDS[0]} after sorting; variant sets differ "
                    f"between folds."
                )

    n_score_cols_found = 0
    n_pval_cols_found = 0

    for score in SCORE_COLUMNS:
        if score not in ref.columns:
            continue
        n_score_cols_found += 1

        stacked = np.array([fold_dfs[fold][score].to_numpy() for fold in FOLDS])
        variant_scores.loc[:, score + ".mean"] = np.mean(stacked, axis=0)

        pval_col = None
        if score + ".pval" in ref.columns:
            pval_col = score + ".pval"
        elif score + "_pval" in ref.columns:
            pval_col = score + "_pval"

        if pval_col is not None:
            n_pval_cols_found += 1
            pval_stacked = np.array(
                [fold_dfs[fold][pval_col].to_numpy() for fold in FOLDS]
            )
            variant_scores.loc[:, score + ".mean.pval"] = geo_mean_overflow(
                pval_stacked
            )

    out_file = os.path.join(cell_dir, f"{cell_type}.mean.variant_scores.tsv")

    summary = {
        "cell_type": cell_type,
        "n_variants": len(variant_scores),
        "n_score_columns_averaged": n_score_cols_found,
        "n_pval_columns_geomeaned": n_pval_cols_found,
        "output_shape": variant_scores.shape,
        "output_file": out_file,
    }

    if not dry_run:
        variant_scores.to_csv(out_file, sep="\t", index=False)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Average variant_scores.tsv across folds for each cell type."
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help=(
            "Path to the score directory containing one "
            "subfolder per cell type."
        ),
    )
    parser.add_argument(
        "--cell-types",
        nargs="+",
        default=None,
        help=(
            "Optional subset of cell type names to process (default: all 17 "
            "hard-coded cell types)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print summaries but do not write any output files.",
    )
    args = parser.parse_args()

    cell_types = args.cell_types if args.cell_types else CELL_TYPES

    print(f"Base directory: {args.base_dir}")
    print(f"Cell types to process ({len(cell_types)}): {cell_types}")
    if args.dry_run:
        print("DRY RUN: no output files will be written.")
    print()

    results = []
    failures = []

    for cell_type in cell_types:
        print(f"--- {cell_type} ---")
        try:
            summary = average_cell_type(args.base_dir, cell_type, dry_run=args.dry_run)
            results.append(summary)
            print(f"  variants: {summary['n_variants']}")
            print(f"  score columns averaged: {summary['n_score_columns_averaged']} / {len(SCORE_COLUMNS)}")
            print(f"  pval columns geo-meaned: {summary['n_pval_columns_geomeaned']}")
            print(f"  output shape: {summary['output_shape']}")
            print(f"  output file: {summary['output_file']}")
        except Exception as e:
            failures.append((cell_type, str(e)))
            print(f"  FAILED: {e}")
        print()

    print("=" * 60)
    print(f"Done. {len(results)}/{len(cell_types)} cell types succeeded.")
    if failures:
        print(f"{len(failures)} cell type(s) FAILED:")
        for cell_type, err in failures:
            print(f"  - {cell_type}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
