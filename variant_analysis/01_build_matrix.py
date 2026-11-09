#!/usr/bin/env python
"""
01_build_matrix.py: ingest ChromBPNet variant scores and build the analysis checkpoint.

Method notes
------------
1.  Reads the 170 per-fold score files (17 cell types x 5 folds) rather than the 17
    fold-averaged files, and recomputes significance. The `<metric>.mean.pval` columns
    produced by average_folds.py are the geometric mean of the five per-fold empirical
    p-values (as in variant-scorer's variant_summary_across_folds.py). That quantity is
    a monotone transform of Fisher's combined statistic, so its null distribution
    depends on how correlated the five folds are: at a nominal 0.01 cut the realised
    null rate ranges from 5e-6 (independent folds) to 1e-2 (identical folds). Here the
    statistic is averaged across folds and a single empirical p is computed against the
    fold-averaged shuffled null, so the fold correlation is present in both arms.

2.  Two calling rules are derived: BH-FDR within cell type on the accessible subset,
    and the uncorrected empirical P < 0.01 rule of Marderstein et al. With 1,565,539
    variants scored per cell type, an uncorrected P < 0.01 on all variants would admit
    ~15,700 expected false positives per cell type; both rules are therefore applied to
    the accessible subset, and the one designated primary in config.py becomes the
    `called` column.

3.  Coordinate conversions and joins are asserted. The 1-based/0-based question is
    tested both ways and fails loudly rather than returning a near-empty join.

Outputs
-------
  variants.parquet              one row per variant (1.57M), keys + annotations
  variant_celltype.parquet      one row per testable variant x cell type, scores + calls
  celltype_similarity.parquet   pairwise peak Jaccard and score correlation (for 04)
  ingest_qc.txt                 diagnostics

Usage
-----
  python 01_build_matrix.py                       # defaults from config.py
  python 01_build_matrix.py --score-dir /path --output-dir /path
  python 01_build_matrix.py --cell-types HSC,Ery  # quick test on two cell types
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
from s4lib import (  # noqa: E402
    QCReport,
    banner,
    benjamini_hochberg,
    build_interval_index,
    empirical_pvalues,
    ids_aligned,
    log,
    nearest_tss,
    parse_gencode_tss,
    peek_header,
    points_in_intervals,
    read_tsv_subset,
    resolve_columns,
    variant_key,
)


# ──────────────────────────────────────────────────────────────────────────────
# Per-fold reading
# ──────────────────────────────────────────────────────────────────────────────

def read_fold(path: Path, metrics: list[str], with_ids: bool) -> tuple[pd.DataFrame, dict]:
    """Read one per-fold score file, resolving metric column names tolerantly."""
    header = peek_header(path)
    colmap = resolve_columns(header, metrics, context=str(path))
    wanted = list(colmap.values())
    id_cols = []
    if with_ids:
        id_cols = [c for c in C.SCORE_ID_COLUMNS if c in header]
        missing_ids = set(C.VARIANT_KEY_COLUMNS) - set(id_cols)
        if missing_ids:
            raise KeyError(
                f"{path}: missing key column(s) {sorted(missing_ids)}. Header is: {header}"
            )
    df = read_tsv_subset(path, id_cols + wanted)
    df = df.rename(columns={v: k for k, v in colmap.items()})
    return df, colmap


def average_across_folds(cell_type: str, shuffled: bool, metrics: list[str],
                         ref_ids: pd.DataFrame | None, qc: QCReport):
    """Mean of each metric across the five folds.

    The statistic is averaged before testing, rather than combining per-fold p-values
    afterwards. This suppresses fold-to-fold noise and leaves a single quantity whose
    null can be taken from the shuffled arm without an independence assumption.

    Returns (ids, {metric: mean array}). ``ids`` is None when shuffled and unaligned.
    """
    label = f"{cell_type}{' [shuffled]' if shuffled else ''}"
    acc: dict[str, np.ndarray] = {}
    ids = None
    n_rows = None

    for k in C.FOLDS:
        path = C.fold_score_file(cell_type, k, shuffled=shuffled)
        if not path.exists():
            raise FileNotFoundError(f"{label}: missing fold file {path}")
        df, _ = read_fold(path, metrics, with_ids=True)

        if ids is None:
            ids = df[[c for c in C.SCORE_ID_COLUMNS if c in df.columns]].copy()
            n_rows = len(df)
        else:
            if len(df) != n_rows:
                raise ValueError(
                    f"{label}: fold {k} has {len(df):,} rows but fold {C.FOLDS[0]} has "
                    f"{n_rows:,}. Folds must score the same variant set."
                )
            if not ids_aligned(ids, df, C.VARIANT_KEY_COLUMNS):
                # Fall back to sorting both by key, as average_folds.py does.
                log(f"{label}: fold {k} row order differs from fold 0; sorting to align",
                    level=2)
                order_ref = np.lexsort([ids[c].to_numpy() for c in C.VARIANT_KEY_COLUMNS[::-1]])
                order_new = np.lexsort([df[c].to_numpy() for c in C.VARIANT_KEY_COLUMNS[::-1]])
                ids = ids.iloc[order_ref].reset_index(drop=True)
                for m in acc:
                    acc[m] = acc[m][order_ref]
                df = df.iloc[order_new].reset_index(drop=True)
                if not ids_aligned(ids, df, C.VARIANT_KEY_COLUMNS):
                    raise ValueError(
                        f"{label}: fold {k} scores a different variant set than fold "
                        f"{C.FOLDS[0]} even after sorting. Cannot average."
                    )

        for m in metrics:
            v = df[m].to_numpy(dtype=np.float32)
            acc[m] = v if m not in acc else acc[m] + v
        del df

    n_folds = len(C.FOLDS)
    means = {m: acc[m] / n_folds for m in metrics}

    if shuffled and ref_ids is not None:
        matched = ids_aligned(ref_ids, ids, C.VARIANT_KEY_COLUMNS)
        qc.add(f"{cell_type} shuffled rows", len(ids))
        qc.add(f"{cell_type} shuffled 1:1 matched to observed", str(matched))
        if not matched:
            qc.note(
                f"    {cell_type}: shuffled arm is not row-matched to the observed arm; "
                f"used as a pooled background, which is what the empirical p requires."
            )
    return ids, means


# ──────────────────────────────────────────────────────────────────────────────
# Accessibility
# ──────────────────────────────────────────────────────────────────────────────

def load_peak_index(cell_type: str, qc: QCReport) -> dict:
    """Interval index over one cell type's ChromBPNet peak set (fold 0)."""
    path = C.peak_file(cell_type)
    if not path.exists():
        raise FileNotFoundError(f"{cell_type}: missing peak file {path}")
    header = peek_header(path)
    # peak_scores.tsv is a BED-like table; the first three columns are chrom/start/end
    # whatever they happen to be named.
    cols = header[:3]
    df = read_tsv_subset(path, cols)
    df.columns = ["chr", "start", "end"]
    df["chr"] = df["chr"].astype(str)
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    qc.add(f"{cell_type} peaks", len(df))
    return build_interval_index(df)


def load_consensus_index(path: Path, qc: QCReport) -> dict | None:
    if not path.exists():
        qc.note(f"    consensus peak file absent ({path}); testability falls back to the "
                f"union of per-cell-type peak sets")
        return None
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2],
                     names=["chr", "start", "end"], dtype={0: str})
    df["chr"] = df["chr"].astype(str)
    qc.add("consensus peaks", len(df))
    return build_interval_index(df)


# ──────────────────────────────────────────────────────────────────────────────
# Annotation sources
# ──────────────────────────────────────────────────────────────────────────────

def load_pip(path: Path, qc: QCReport) -> pd.DataFrame:
    """Per-site maximum fine-mapping posterior, with rsID and trait list.

    The credible-set file lists one row per variant per trait, so a variant appears
    repeatedly; the maximum PP across traits is the per-variant posterior used
    throughout.
    """
    df = pd.read_csv(path, sep="\t", dtype={"chr": str}, low_memory=False)
    need = {"chr", "start", "PP"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{path}: PIP bed missing column(s) {sorted(missing)}. "
                       f"Columns are: {list(df.columns)}")
    df["chr"] = df["chr"].astype(str)
    df.loc[~df["chr"].str.startswith("chr"), "chr"] = "chr" + df["chr"]
    has_rsid = "rsid" in df.columns
    has_trait = "trait" in df.columns

    agg = {"PP": "max"}
    if has_rsid:
        agg["rsid"] = "first"
    g = df.groupby(["chr", "start"], sort=False).agg(agg).reset_index()
    if has_trait:
        traits = (df.groupby(["chr", "start"], sort=False)["trait"]
                    .agg(lambda s: ",".join(sorted(set(map(str, s)))))
                    .reset_index().rename(columns={"trait": "traits"}))
        g = g.merge(traits, on=["chr", "start"], how="left")

    qc.add("PIP rows", len(df))
    qc.add("PIP unique sites", len(g))
    qc.add(f"PIP sites with PP >= {C.FLAGSHIP_PIP}", int((g["PP"] >= C.FLAGSHIP_PIP).sum()))
    return g


def load_constraint(path: Path, qc: QCReport) -> pd.DataFrame | None:
    """gnomAD gene-level constraint. LOEUF (oe_lof_upper) substitutes for s_het."""
    if not path.exists():
        qc.note(f"    constraint file absent ({path}); {C.CONSTRAINT_METRIC_LABEL} "
                f"covariate will be unavailable")
        return None
    opener = gzip.open if str(path).endswith((".gz", ".bgz")) else open
    with opener(path, "rt") as fh:
        head = fh.readline().rstrip("\n").split("\t")
    keep = [c for c in ("gene", C.CONSTRAINT_METRIC, "pLI", "oe_lof") if c in head]
    if "gene" not in keep or C.CONSTRAINT_METRIC not in keep:
        raise KeyError(f"{path}: expected columns 'gene' and '{C.CONSTRAINT_METRIC}'. "
                       f"Header is: {head}")
    df = pd.read_csv(path, sep="\t", usecols=keep, compression="gzip")
    df = df.dropna(subset=["gene"]).drop_duplicates(subset=["gene"], keep="first")
    qc.add("constraint genes", len(df))
    qc.add(f"genes with {C.CONSTRAINT_METRIC_LABEL}",
           int(df[C.CONSTRAINT_METRIC].notna().sum()))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score-dir", type=Path, default=C.SCORE_DIR)
    ap.add_argument("--consensus-peaks", type=Path, default=C.CONSENSUS_PEAKS)
    ap.add_argument("--pip-bed", type=Path, default=C.PIP_BED)
    ap.add_argument("--gencode-gtf", type=Path, default=C.GENCODE_GTF)
    ap.add_argument("--constraint", type=Path, default=C.GNOMAD_CONSTRAINT)
    ap.add_argument("--quadrant-table", type=Path, default=None,
                    help="optional quadrant assignments; must carry chr, pos "
                         "(1-based) and quadrant columns")
    ap.add_argument("--output-dir", type=Path, default=C.OUTPUT_DIR)
    ap.add_argument("--cell-types", type=str, default=None,
                    help="comma-separated subset, for smoke tests")
    ap.add_argument("--keep-all-long", action="store_true",
                    help="retain long rows for every variant, not just testable ones "
                         "(raises the checkpoint from ~0.2 GB to several GB)")
    args = ap.parse_args()

    C.SCORE_DIR = args.score_dir
    C.OUTPUT_DIR = args.output_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cell_types = args.cell_types.split(",") if args.cell_types else list(C.CELL_TYPES)
    unknown = set(cell_types) - set(C.CELL_TYPES)
    if unknown:
        raise SystemExit(f"unknown cell type(s): {sorted(unknown)}")

    C.validate()
    qc = QCReport("Ingest QC")
    qc.section("configuration")
    qc.add("cell types", ", ".join(cell_types))
    qc.add("folds", ", ".join(map(str, C.FOLDS)))
    qc.add("significance statistic", C.SIGNIF_METRIC)
    qc.add("primary rule", f"accessible AND {C.primary_rule()[2]}")
    qc.note("    average_folds.py .mean.pval columns not used "
            "(geometric mean of per-fold p-values is not calibrated)")

    metrics = [C.EFFECT_METRIC, C.SIGNIF_METRIC, C.SECONDARY_SIGNIF_METRIC,
               C.ACCESSIBILITY_METRIC]

    # ── Pass A: variant list and accessibility ────────────────────────────────
    banner("Pass A, variant list and per-cell-type accessibility")
    qc.section("variant universe")

    first = cell_types[0]
    probe_path = C.fold_score_file(first, C.FOLDS[0])
    probe_header = peek_header(probe_path)
    qc.add("per-fold score columns detected", ", ".join(probe_header))
    log(f"per-fold header ({probe_path.name}): {probe_header}")

    probe, _ = read_fold(probe_path, metrics, with_ids=True)
    ref_ids = probe[[c for c in C.SCORE_ID_COLUMNS if c in probe.columns]].copy()
    del probe

    n_var = len(ref_ids)
    ref_ids["chr"] = ref_ids["chr"].astype(str)
    ref_ids["pos"] = ref_ids["pos"].astype(np.int64)
    pos0 = ref_ids["pos"].to_numpy() - (1 if C.POS_IS_ONE_BASED else 0)
    chroms = ref_ids["chr"].to_numpy()

    qc.add("variants scored per cell type", n_var)
    if n_var != C.EXPECTED_N_VARIANTS:
        qc.note(f"    NOTE: expected {C.EXPECTED_N_VARIANTS:,} variants from the "
                f"provenance tally, found {n_var:,}")

    if C.PROVENANCE_COLUMN in ref_ids.columns:
        prov = ref_ids[C.PROVENANCE_COLUMN].value_counts()
        for k, v in prov.items():
            qc.add(f"provenance {k}", int(v))
        ref_ids["is_vuckovic"] = (ref_ids[C.PROVENANCE_COLUMN]
                                  .astype(str).str.contains("Vuckovic"))
        qc.add("Vuckovic-derived variants", int(ref_ids["is_vuckovic"].sum()))
    else:
        ref_ids["is_vuckovic"] = False
        qc.note("    provenance column absent; is_vuckovic set False throughout")

    qc.section("accessibility")
    access = {}
    for ct in cell_types:
        idx = load_peak_index(ct, qc)
        access[ct] = points_in_intervals(chroms, pos0, idx)
        qc.add(f"{ct} accessible variants", int(access[ct].sum()))
        log(f"{ct}: {access[ct].sum():,} accessible of {n_var:,}", level=1)

    any_access = np.zeros(n_var, dtype=bool)
    for m in access.values():
        any_access |= m

    cons_idx = load_consensus_index(args.consensus_peaks, qc)
    in_consensus = (points_in_intervals(chroms, pos0, cons_idx) if cons_idx is not None
                    else np.zeros(n_var, dtype=bool))
    qc.add("variants in consensus peaks", int(in_consensus.sum()))
    qc.add("variants accessible in >=1 cell type", int(any_access.sum()))
    qc.add("consensus peaks are a superset of per-cell-type accessibility",
           str(bool((~in_consensus & any_access).sum() == 0)))

    testable = any_access | in_consensus
    if args.keep_all_long:
        testable = np.ones(n_var, dtype=bool)
    n_testable = int(testable.sum())
    qc.add("testable variants retained in the long table", n_testable)

    # ── Pass B: fold averaging, empirical p, FDR ──────────────────────────────
    banner("Pass B, fold averaging and calibrated empirical p-values")
    qc.section("significance")

    n_ct = len(cell_types)
    # Dense testable-by-cell-type matrices. Building these directly during the pass is
    # both faster and more memory-frugal than assembling a 26M-row long table and
    # pivoting it back out repeatedly for the panel summaries and the similarity matrix.
    mat = {
        "eff": np.empty((n_testable, n_ct), dtype=np.float32),
        "abs": np.empty((n_testable, n_ct), dtype=np.float32),
        "sec": np.empty((n_testable, n_ct), dtype=np.float32),
        "qnt": np.empty((n_testable, n_ct), dtype=np.float32),
        "p": np.empty((n_testable, n_ct), dtype=np.float32),
        "p2": np.empty((n_testable, n_ct), dtype=np.float32),
        "q": np.empty((n_testable, n_ct), dtype=np.float32),
    }
    acc_mat = np.empty((n_testable, n_ct), dtype=bool)
    sig_mat = np.empty((n_testable, n_ct), dtype=bool)
    sigraw_mat = np.empty((n_testable, n_ct), dtype=bool)
    null_summaries = []

    for j, ct in enumerate(cell_types):
        log(f"{ct}: averaging {len(C.FOLDS)} observed folds", level=1)
        ids, obs = average_across_folds(ct, False, metrics, None, qc)
        if not ids_aligned(ref_ids, ids, C.VARIANT_KEY_COLUMNS):
            raise ValueError(
                f"{ct} scores a different variant set than {first}. All cell types must "
                f"share one variant list for the breadth analysis to be meaningful."
            )

        log(f"{ct}: averaging {len(C.FOLDS)} shuffled folds", level=1)
        _, nul = average_across_folds(ct, True, metrics, ref_ids, qc)

        p_primary = empirical_pvalues(obs[C.SIGNIF_METRIC], nul[C.SIGNIF_METRIC])
        p_secondary = empirical_pvalues(obs[C.SECONDARY_SIGNIF_METRIC],
                                        nul[C.SECONDARY_SIGNIF_METRIC])

        # BH within cell type, on the accessible subset only. Accessibility is an
        # inclusion criterion for nomination, so restricting the correction to it is
        # legitimate and materially increases power relative to correcting over 1.57M.
        acc = access[ct]
        q = np.full(n_var, np.nan, dtype=np.float64)
        q[acc] = benjamini_hochberg(p_primary[acc])

        sig_fdr = acc & np.isfinite(q) & (q < C.FDR_PRIMARY)
        sig_raw = acc & (p_primary < C.P_RAW_MARDERSTEIN)

        qc.add(f"{ct} accessible", int(acc.sum()))
        qc.add(f"{ct} significant (BH < {C.FDR_PRIMARY}, accessible)", int(sig_fdr.sum()))
        qc.add(f"{ct} significant (raw p < {C.P_RAW_MARDERSTEIN}, accessible)",
               int(sig_raw.sum()))
        qc.add(f"{ct} raw p < {C.P_RAW_MARDERSTEIN} genome-wide (uncorrected, for scale)",
               int((p_primary < C.P_RAW_MARDERSTEIN).sum()))

        null_summaries.append(dict(
            cell_type=ct,
            null_n=int(np.isfinite(nul[C.SIGNIF_METRIC]).sum()),
            null_median=float(np.nanmedian(nul[C.SIGNIF_METRIC])),
            null_p99=float(np.nanpercentile(nul[C.SIGNIF_METRIC], 99)),
            obs_median=float(np.nanmedian(obs[C.SIGNIF_METRIC])),
            obs_p99=float(np.nanpercentile(obs[C.SIGNIF_METRIC], 99)),
            p_floor=float(p_primary.min()),
            frac_p_below_01=float((p_primary < 0.01).mean()),
            frac_p_below_05=float((p_primary < 0.05).mean()),
        ))

        sel = testable
        mat["eff"][:, j] = obs[C.EFFECT_METRIC][sel]
        mat["abs"][:, j] = obs[C.SIGNIF_METRIC][sel]
        mat["sec"][:, j] = obs[C.SECONDARY_SIGNIF_METRIC][sel]
        mat["qnt"][:, j] = obs[C.ACCESSIBILITY_METRIC][sel]
        mat["p"][:, j] = p_primary[sel]
        mat["p2"][:, j] = p_secondary[sel]
        mat["q"][:, j] = q[sel]
        acc_mat[:, j] = acc[sel]
        sig_mat[:, j] = sig_fdr[sel]
        sigraw_mat[:, j] = sig_raw[sel]
        del obs, nul, p_primary, p_secondary, q

    testable_idx = np.flatnonzero(testable).astype(np.int32)
    long = pd.DataFrame({
        "variant_idx": np.tile(testable_idx, n_ct),
        "cell_type": pd.Categorical(
            np.repeat(np.array(cell_types, dtype=object), n_testable),
            categories=C.CELL_TYPES),
        C.EFFECT_METRIC: mat["eff"].T.ravel(),
        C.SIGNIF_METRIC: mat["abs"].T.ravel(),
        C.SECONDARY_SIGNIF_METRIC: mat["sec"].T.ravel(),
        C.ACCESSIBILITY_METRIC: mat["qnt"].T.ravel(),
        "pval": mat["p"].T.ravel(),
        "pval_secondary": mat["p2"].T.ravel(),
        "qval": mat["q"].T.ravel(),
        "accessible": acc_mat.T.ravel(),
        "significant": sig_mat.T.ravel(),
        "significant_raw_p": sigraw_mat.T.ravel(),
    })

    # ── the primary call, derived once here and read by every later module ────
    # `significant` and `significant_raw_p` are the two named rules; `called` is the
    # one config designates primary, so the rule is defined in one place.
    _kind, _level, _label = C.primary_rule()
    if _kind == "raw":
        long["called"] = long["accessible"].to_numpy() & (long["pval"].to_numpy() < _level)
    else:
        _q = long["qval"].to_numpy()
        long["called"] = (long["accessible"].to_numpy() & np.isfinite(_q) & (_q < _level))
    qc.add("primary call column", f"called = accessible AND {_label}")
    qc.add("called variant-cell-type pairs", int(long["called"].sum()))
    # consistency check against the pre-computed named rules
    _named = {"raw": ("significant_raw_p", C.P_RAW_MARDERSTEIN),
              "fdr": ("significant", C.FDR_PRIMARY)}[_kind]
    if abs(_level - _named[1]) < 1e-15:
        assert bool((long["called"] == long[_named[0]]).all()), (
            f"`called` disagrees with `{_named[0]}` at the same level")
    log(f"long table: {len(long):,} rows", level=1)

    # ── Variant-level table ───────────────────────────────────────────────────
    banner("Variant-level annotation")
    qc.section("annotation joins")

    var = ref_ids.copy()
    var.insert(0, "variant_idx", np.arange(n_var, dtype=np.int32))
    var["key"] = variant_key(var, C.VARIANT_KEY_COLUMNS)
    var["pos0"] = pos0
    var["in_consensus_peak"] = in_consensus
    var["accessible_any"] = any_access
    var["n_accessible"] = np.sum([access[ct] for ct in cell_types], axis=0).astype(np.int16)
    var["testable"] = testable

    # PIP. The score files are 1-based and the credible-set file is also 1-based (its
    # start and end are equal on a SNV row), so the correct join is pos == start with no
    # conversion. The consensus-peak BED *is* 0-based half-open and was converted
    # separately above: the two inputs genuinely disagree, which is why each carries its
    # own flag. Both joins are counted either way and the losing convention fails loudly,
    # because a silent off-by-one here deletes most of the fine-mapped signal.
    pip = load_pip(args.pip_bed, qc)
    pip_sites = set(zip(pip["chr"].to_numpy(), pip["start"].to_numpy()))
    hits_0based = sum(1 for c, p in zip(chroms, pos0) if (c, p) in pip_sites)
    hits_1based = sum(1 for c, p in zip(chroms, ref_ids["pos"].to_numpy())
                      if (c, p) in pip_sites)
    qc.add("PIP join using pos-1 (credible set read as 0-based BED)", hits_0based)
    qc.add("PIP join using pos (credible set read as 1-based)", hits_1based)
    join_col = "pos0" if C.PIP_BED_IS_ZERO_BASED else "pos"
    chosen, other = ((hits_0based, hits_1based) if C.PIP_BED_IS_ZERO_BASED
                     else (hits_1based, hits_0based))
    qc.add("PIP join convention used", f"{join_col} "
           f"(PIP_BED_IS_ZERO_BASED = {C.PIP_BED_IS_ZERO_BASED})")
    if chosen < other:
        raise ValueError(
            f"PIP join recovers more sites under the opposite coordinate convention "
            f"({other:,}) than the configured one ({chosen:,}). "
            f"config.PIP_BED_IS_ZERO_BASED is currently {C.PIP_BED_IS_ZERO_BASED}; flip "
            f"it and re-run. Read one SNV row of the credible-set file to decide: if "
            f"`start` == `end` it is 1-based (False); if `end` == `start` + 1 it is a "
            f"true BED (True). This flag does not affect the consensus-peak BED, which "
            f"is 0-based half-open and converted independently."
        )

    var = var.merge(
        pip.rename(columns={"start": join_col, "PP": "PIP"}),
        on=["chr", join_col], how="left",
    )
    assert len(var) == n_var, f"PIP merge changed row count: {len(var):,} vs {n_var:,}"
    qc.add("variants with a fine-mapping posterior", int(var["PIP"].notna().sum()))
    if "is_vuckovic" in var.columns and var["is_vuckovic"].any():
        recov = var.loc[var["is_vuckovic"], "PIP"].notna().mean()
        qc.add("fraction of Vuckovic-provenance variants matched to a PP", float(recov))
        if recov < 0.5:
            qc.note("    WARNING: fewer than half the Vuckovic-provenance variants "
                    "matched a credible-set site. Check that the PIP bed is the same "
                    "build and release used to generate the scored variant list.")

    # GENCODE. Two nearest-gene columns: any gene type for interpretation, and
    # protein-coding only because that is what joins to gnomAD constraint.
    if args.gencode_gtf.exists():
        tss = parse_gencode_tss(args.gencode_gtf)
        qc.add("GENCODE genes", len(tss))
        g_any, d_any = nearest_tss(chroms, pos0, tss)
        var["nearest_gene"] = g_any
        var["tss_distance"] = d_any
        pc = tss[tss["gene_type"] == "protein_coding"]
        qc.add("GENCODE protein-coding genes", len(pc))
        g_pc, d_pc = nearest_tss(chroms, pos0, pc)
        var["nearest_gene_pc"] = g_pc
        var["tss_distance_pc"] = d_pc
        var["log10_tss_distance"] = np.log10(np.abs(d_pc) + 1.0)
        qc.add("variants with a nearest protein-coding TSS",
               int(pd.notna(var["nearest_gene_pc"]).sum()))
    else:
        qc.note(f"    GENCODE GTF absent ({args.gencode_gtf}); nearest-gene and TSS "
                f"distance columns omitted, covariate model will run without them")
        for col in ("nearest_gene", "tss_distance", "nearest_gene_pc",
                    "tss_distance_pc", "log10_tss_distance"):
            var[col] = np.nan

    # Constraint
    cons = load_constraint(args.constraint, qc)
    if cons is not None and "nearest_gene_pc" in var.columns:
        var = var.merge(cons.rename(columns={"gene": "nearest_gene_pc"}),
                        on="nearest_gene_pc", how="left")
        assert len(var) == n_var, "constraint merge changed row count"
        rate = float(var[C.CONSTRAINT_METRIC].notna().mean())
        qc.add(f"variants with {C.CONSTRAINT_METRIC_LABEL}", int(
            var[C.CONSTRAINT_METRIC].notna().sum()))
        qc.add(f"{C.CONSTRAINT_METRIC_LABEL} join rate", rate)
        if rate < 0.5:
            qc.note("    NOTE: gnomAD v2.1.1 predates GENCODE v47, so some gene symbols "
                    "will not match. A join rate well below 1 is expected; below 0.5 "
                    "suggests a symbol-namespace problem worth checking.")
    else:
        var[C.CONSTRAINT_METRIC] = np.nan

    # Optional quadrant labels
    if args.quadrant_table is not None and args.quadrant_table.exists():
        qd = pd.read_csv(args.quadrant_table, sep=None, engine="python")
        need = {"chr", "pos", "quadrant"}
        if not need <= set(qd.columns):
            raise KeyError(f"{args.quadrant_table}: need columns {sorted(need)}, "
                           f"found {list(qd.columns)}")
        qd["chr"] = qd["chr"].astype(str)
        qd = qd.drop_duplicates(subset=["chr", "pos"])
        var = var.merge(qd[["chr", "pos", "quadrant"]], on=["chr", "pos"], how="left")
        assert len(var) == n_var, "quadrant merge changed row count"
        qc.add("variants with a quadrant label",
               int(var["quadrant"].notna().sum()))
    else:
        var["quadrant"] = pd.NA
        qc.note("    no quadrant table supplied (--quadrant-table); quadrant "
                "column left empty")

    # ── Panel-level summaries over all 17 cell types ──────────────────────────
    qc.section("panel summaries")
    # Magnitude is the maximum over the full panel, not over the called cell types:
    # taking the maximum over called contexts and correlating it with the number of
    # called contexts would use the same selection on both axes.
    panel = pd.DataFrame({
        "variant_idx": testable_idx,
        "n_significant": sig_mat.sum(axis=1).astype(np.int16),
        "n_significant_raw_p": sigraw_mat.sum(axis=1).astype(np.int16),
        "max_abs_panel": mat["abs"].max(axis=1).astype(np.float32),
        "mean_abs_panel": mat["abs"].mean(axis=1).astype(np.float32),
        "min_pval_panel": mat["p"].min(axis=1).astype(np.float32),
    })
    var = var.merge(panel, on="variant_idx", how="left")
    assert len(var) == n_var, "panel summary merge changed row count"
    for col, fill in (("n_significant", 0), ("n_significant_raw_p", 0)):
        var[col] = var[col].fillna(fill).astype(np.int16)

    # breadth under the primary rule, the variant-level counterpart of `called`
    _n_called = (long.loc[long["called"].to_numpy()]
                     .groupby("variant_idx", observed=True)
                     .size().rename("n_called"))
    var = var.merge(_n_called, on="variant_idx", how="left")
    var["n_called"] = var["n_called"].fillna(0).astype(np.int16)
    assert len(var) == n_var, "n_called merge changed row count"
    qc.add("variants called in >=1 cell type", int((var["n_called"] > 0).sum()))

    cats = C.breadth_categories(len(cell_types))
    qc.add("breadth categories", str(cats))
    for name, (lo, hi) in cats.items():
        _bc = C.breadth_column(var)
        n = int(((var[_bc] >= lo) & (var[_bc] <= hi)).sum())
        qc.add(f"breadth {name} [{lo}-{hi}]", n)

    # ── Cell-type similarity, for the 04 confound adjustment ──────────────────
    banner("Cell-type similarity")
    qc.section("similarity")
    sims = []
    for i, a in enumerate(cell_types):
        for j2 in range(i + 1, n_ct):
            b = cell_types[j2]
            ja, jb = access[a], access[b]
            inter, union = int((ja & jb).sum()), int((ja | jb).sum())
            # Score correlation restricted to variants accessible in BOTH cell types is
            # the quantity that confounds the co-significance rate, because that is the
            # set on which both cell types can be called significant. Two adjacent cell
            # types share most of their peaks and most of their model, so a high value
            # here explains co-significance without any lineage structure, which is why
            # 04 regresses co-significance on graph distance with this as a covariate.
            both = acc_mat[:, i] & acc_mat[:, j2]
            if both.sum() >= 1000:
                x, y = mat["eff"][both, i], mat["eff"][both, j2]
                ok = np.isfinite(x) & np.isfinite(y)
                r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= 1000 else np.nan
            else:
                r = np.nan
            sims.append(dict(cell_type_a=a, cell_type_b=b,
                             peak_jaccard=inter / union if union else np.nan,
                             n_accessible_both=inter,
                             score_corr_accessible_both=r))
    sim_df = pd.DataFrame(sims)
    qc.add("cell-type pairs", len(sim_df))
    if len(sim_df):
        qc.add("peak Jaccard range",
               f"{sim_df['peak_jaccard'].min():.3f} to {sim_df['peak_jaccard'].max():.3f}")
        qc.add("score correlation range",
               f"{sim_df['score_corr_accessible_both'].min():.3f} to "
               f"{sim_df['score_corr_accessible_both'].max():.3f}")

    # ── Write ─────────────────────────────────────────────────────────────────
    banner("Writing checkpoint")
    var = var.drop(columns=["key"])
    var.to_parquet(out / "variants.parquet", index=False)
    long.to_parquet(out / "variant_celltype.parquet", index=False)
    sim_df.to_parquet(out / "celltype_similarity.parquet", index=False)
    pd.DataFrame(null_summaries).to_csv(out / "null_summaries.csv", index=False)

    qc.section("outputs")
    for f in ("variants.parquet", "variant_celltype.parquet",
              "celltype_similarity.parquet", "null_summaries.csv"):
        p = out / f
        qc.add(f, f"{p.stat().st_size / 1e6:.1f} MB")
    qc.write(out / "ingest_qc.txt")

    log(f"variants.parquet        {len(var):,} rows x {var.shape[1]} cols")
    log(f"variant_celltype.parquet {len(long):,} rows x {long.shape[1]} cols")
    log("done")


if __name__ == "__main__":
    main()
