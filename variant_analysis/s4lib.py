"""
s4lib.py: shared helpers for the variant-analysis scripts.

Interval overlap, GENCODE parsing, empirical p-values, BH correction, tolerant column
resolution and run logging. Imported by the numbered modules; contains no thresholds and
no cell-type knowledge (those live in config.py).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

_T0 = time.time()


def log(msg: str = "", *, level: int = 0) -> None:
    """Timestamped progress line on stderr, so stdout stays reserved for data."""
    if not msg:
        print("", file=sys.stderr)
        return
    el = time.time() - _T0
    print(f"[{el / 60:6.1f}m] {'  ' * level}{msg}", file=sys.stderr, flush=True)


def banner(msg: str) -> None:
    log()
    log("=" * 78)
    log(msg)
    log("=" * 78)


# ──────────────────────────────────────────────────────────────────────────────
# Tolerant column resolution
# ──────────────────────────────────────────────────────────────────────────────

def resolve_columns(available, wanted, *, context: str = "") -> dict:
    """Map logical metric names onto the column names actually present in a file.

    variant-scorer per-fold outputs use bare metric names (``abs_logfc``); the
    fold-averaged outputs from average_folds.py use ``abs_logfc.mean``. Accept either,
    plus a ``.mean`` / ``_mean`` suffix variant, so the same reader works on both.

    Returns {logical_name: actual_column}. Raises KeyError naming every available
    column if a wanted metric cannot be found, this is the failure mode most likely to
    bite on first run, so the message lists every available column.
    """
    avail = list(available)
    lookup = {c.lower(): c for c in avail}
    out, missing = {}, []
    for w in wanted:
        for cand in (w, f"{w}.mean", f"{w}_mean", f"mean.{w}"):
            hit = lookup.get(cand.lower())
            if hit is not None:
                out[w] = hit
                break
        else:
            missing.append(w)
    if missing:
        raise KeyError(
            f"{context}: could not resolve column(s) {missing}. "
            f"Available columns are: {avail}"
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Fast TSV reading
# ──────────────────────────────────────────────────────────────────────────────

try:
    import pyarrow  # noqa: F401
    import pyarrow.csv as _pacsv

    HAVE_PYARROW = True
except ImportError:  # pragma: no cover
    HAVE_PYARROW = False


def peek_header(path: Path) -> list[str]:
    """Read only the header line of a (possibly very large) TSV."""
    import gzip

    opener = gzip.open if str(path).endswith((".gz", ".bgz")) else open
    with opener(path, "rt") as fh:
        return fh.readline().rstrip("\n").split("\t")


def read_tsv_subset(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a subset of columns from a large TSV as fast as the environment allows.

    pyarrow's multithreaded CSV reader is roughly an order of magnitude faster than the
    pandas C parser on these files and is what makes the 170-file ingest practical;
    pandas is used as a fallback so the pipeline still runs without it.
    """
    if HAVE_PYARROW:
        tbl = _pacsv.read_csv(
            str(path),
            parse_options=_pacsv.ParseOptions(delimiter="\t"),
            convert_options=_pacsv.ConvertOptions(include_columns=columns),
        )
        return tbl.to_pandas()
    return pd.read_csv(path, sep="\t", usecols=columns, engine="c")


# ──────────────────────────────────────────────────────────────────────────────
# Variant keys
# ──────────────────────────────────────────────────────────────────────────────

def variant_key(df: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    """Single string key ``chr:pos:allele1:allele2`` for joining and alignment checks."""
    parts = [df[c].astype(str) for c in key_cols]
    key = parts[0]
    for p in parts[1:]:
        key = key.str.cat(p, sep=":")
    return key


def ids_aligned(a: pd.DataFrame, b: pd.DataFrame, key_cols: list[str]) -> bool:
    """True when two frames carry the same variants in the same row order."""
    if len(a) != len(b):
        return False
    for c in key_cols:
        if not np.array_equal(a[c].to_numpy(), b[c].to_numpy()):
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Empirical p-values
# ──────────────────────────────────────────────────────────────────────────────

def empirical_pvalues(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """One-sided empirical p: P(null statistic >= observed), add-one corrected.

    ``p = (1 + #{null >= obs}) / (1 + n_null)``. The add-one keeps p strictly positive,
    which matters because these feed a BH correction and a log axis; it also encodes the
    honest statement that the resolution of a permutation p is bounded by the size of the
    null set. The floor is 1/(n_null + 1); with the 100,000-draw shuffled arm
    used here that is 1.0e-5.

    This replaces the geometric mean of per-fold p-values that average_folds.py emits.
    The geometric mean is a monotone transform of Fisher's combined statistic, so its
    null distribution depends on inter-fold correlation and its realised rate at a
    nominal 0.01 cut ranges over three orders of magnitude. Computing one empirical p on
    the fold-averaged statistic against the fold-averaged shuffled null avoids the
    combination problem: the fold correlation is present in both arms and cancels.
    """
    null_sorted = np.sort(null[np.isfinite(null)])
    n_null = null_sorted.size
    if n_null == 0:
        raise ValueError("empirical_pvalues: null distribution is empty")
    # count of null >= obs, via the insertion point of obs in the ascending null
    n_ge = n_null - np.searchsorted(null_sorted, observed, side="left")
    return (1.0 + n_ge) / (1.0 + n_null)


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (q-values), NaN-safe.

    Uses scipy.stats.false_discovery_control where available, falling back to an explicit
    implementation. NaN inputs propagate as NaN and are excluded from the ranking, so
    non-testable entries never inflate the correction.
    """
    q = np.full(pvals.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(pvals)
    if not ok.any():
        return q
    p = pvals[ok].astype(np.float64)
    try:
        from scipy.stats import false_discovery_control

        q[ok] = false_discovery_control(p, method="bh")
        return q
    except ImportError:
        pass
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    q[ok] = adj
    return q


# ──────────────────────────────────────────────────────────────────────────────
# Interval overlap
# ──────────────────────────────────────────────────────────────────────────────

def build_interval_index(bed: pd.DataFrame,
                         chrom_col: str = "chr",
                         start_col: str = "start",
                         end_col: str = "end") -> dict:
    """Index a BED-like frame for point-in-any-interval queries.

    Intervals are 0-based half-open [start, end). Per chromosome the intervals are sorted
    by start and a running maximum of end is stored, which makes a query a single
    searchsorted plus one comparison and remains correct when intervals overlap each
    other (as ChromBPNet peak sets do).
    """
    idx = {}
    for chrom, sub in bed.groupby(chrom_col, sort=False):
        s = sub[start_col].to_numpy(dtype=np.int64)
        e = sub[end_col].to_numpy(dtype=np.int64)
        order = np.argsort(s, kind="stable")
        s, e = s[order], e[order]
        idx[str(chrom)] = (s, np.maximum.accumulate(e))
    return idx


def points_in_intervals(chroms: np.ndarray, pos0: np.ndarray, index: dict) -> np.ndarray:
    """Boolean mask: is each 0-based point inside any indexed interval?"""
    out = np.zeros(pos0.shape, dtype=bool)
    chroms = np.asarray(chroms)
    for chrom in pd.unique(chroms):
        key = str(chrom)
        if key not in index:
            continue
        starts, cummax_end = index[key]
        sel = chroms == key
        p = pos0[sel]
        j = np.searchsorted(starts, p, side="right") - 1
        hit = np.zeros(p.shape, dtype=bool)
        valid = j >= 0
        hit[valid] = cummax_end[j[valid]] > p[valid]
        out[sel] = hit
    return out


# ──────────────────────────────────────────────────────────────────────────────
# GENCODE
# ──────────────────────────────────────────────────────────────────────────────

def parse_gencode_tss(gtf_path: Path, feature: str = "gene") -> pd.DataFrame:
    """Extract one TSS per gene from a GENCODE GTF.

    Returns columns chr, tss (0-based), strand, gene_name, gene_type. The TSS is the
    start coordinate on the + strand and the end coordinate on the - strand; GTF is
    1-based inclusive, so the 0-based TSS is start-1 or end-1 respectively.
    """
    import gzip

    opener = gzip.open if str(gtf_path).endswith((".gz", ".bgz")) else open
    rows = []
    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != feature:
                continue
            chrom, start, end, strand, attrs = f[0], int(f[3]), int(f[4]), f[6], f[8]
            tss0 = (start - 1) if strand == "+" else (end - 1)
            name = gtype = None
            for field in attrs.split(";"):
                field = field.strip()
                if field.startswith("gene_name "):
                    name = field.split('"')[1]
                elif field.startswith("gene_type "):
                    gtype = field.split('"')[1]
                if name and gtype:
                    break
            rows.append((chrom, tss0, strand, name, gtype))
    df = pd.DataFrame(rows, columns=["chr", "tss", "strand", "gene_name", "gene_type"])
    return df.dropna(subset=["gene_name"]).reset_index(drop=True)


def nearest_tss(chroms: np.ndarray, pos0: np.ndarray, tss_df: pd.DataFrame):
    """Nearest TSS per query point.

    Returns (gene_name array, signed distance array). Distance is ``pos0 - tss``, so it is
    negative upstream of a + strand gene; downstream analyses use its absolute value.
    Points on a chromosome absent from ``tss_df`` receive None and NaN.
    """
    genes = np.full(pos0.shape, None, dtype=object)
    dist = np.full(pos0.shape, np.nan, dtype=np.float64)
    chroms = np.asarray(chroms)
    for chrom, sub in tss_df.groupby("chr", sort=False):
        sel = chroms == str(chrom)
        if not sel.any():
            continue
        t = sub["tss"].to_numpy(dtype=np.int64)
        names = sub["gene_name"].to_numpy(dtype=object)
        order = np.argsort(t, kind="stable")
        t, names = t[order], names[order]
        p = pos0[sel]
        j = np.searchsorted(t, p, side="left")
        left = np.clip(j - 1, 0, t.size - 1)
        right = np.clip(j, 0, t.size - 1)
        dl = np.abs(p - t[left])
        dr = np.abs(p - t[right])
        pick = np.where(dl <= dr, left, right)
        genes[sel] = names[pick]
        dist[sel] = p - t[pick]
    return genes, dist


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

class QCReport:
    """Accumulates ingest diagnostics and writes them as a readable text report.

    The point of collecting these rather than printing them is that the first run happens
    on a machine the author of this pipeline cannot see; the report is the evidence that
    the joins, the coordinate convention and the shuffled null all behaved.
    """

    def __init__(self, title: str):
        self.title = title
        self.sections: list[tuple[str, list[str]]] = []
        self._cur: list[str] | None = None

    def section(self, name: str) -> "QCReport":
        self._cur = []
        self.sections.append((name, self._cur))
        return self

    def add(self, key: str, value) -> None:
        if self._cur is None:
            self.section("general")
        if isinstance(value, float):
            value = f"{value:,.6g}"
        elif isinstance(value, int):
            value = f"{value:,}"
        self._cur.append(f"{key}: {value}")  # type: ignore[union-attr]

    def note(self, text: str) -> None:
        if self._cur is None:
            self.section("general")
        self._cur.append(text)  # type: ignore[union-attr]

    def render(self) -> str:
        out = [f"# {self.title}", ""]
        for name, lines in self.sections:
            out.append(f"## {name}")
            out.extend(f"  {ln}" for ln in lines)
            out.append("")
        return "\n".join(out)

    def write(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.render())
        log(f"QC report -> {path}")
