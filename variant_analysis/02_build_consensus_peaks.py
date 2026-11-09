#!/usr/bin/env python3
"""
02_build_consensus_peaks.py: build a single consensus peak set across all cell types, for use as the shared
region list (`-r`) for chrombpnet contribs_bw / pred_bw and for Fi-NeMo.

Reimplements getPeakSet.snakeATAC.v2 (RoboATAC, code/utils/tf_funcs.R) with
type="summits_no_fw", unifWidth=500, capMaxPeaks=150000, followed by
getConsensusPeakSet(mode="no_by_score"). The manifest is a tab-separated file with
columns cell_id, peak_dir and peak_prefix (MACS2 output is <peak_dir>/<peak_prefix>_summits.bed).
Procedure:

  1. read each sample's MACS2 _summits.bed
  2. keep canonical chromosomes only
  3. percentile-normalise the score WITHIN each sample (ecdf)
  4. extend each summit to a fixed 501 bp window (summit +/- 250)
  5. drop windows trimmed by a chromosome boundary
  6. keep the top --cap summits per sample by raw score
  7. pool all samples; greedily keep the highest score_norm window and discard
     everything overlapping it, until no two windows overlap
  8. remove windows whose model input window overlaps a blacklist region
  9. write a 10-column BED: cols 7-9 = '.', col 10 = 250 (summit offset)

Step 7 is exact, not approximate. Because every window is exactly 501 bp, two
windows overlap iff their starts differ by less than 501, so bucketing starts by
//501 makes the greedy sweep linear rather than quadratic.

Usage:
    python3 02_build_consensus_peaks.py \
        --manifest    cell_types.tsv \
        --chrom-sizes hg38.chrom.sizes \
        --blacklist   blacklist.bed.gz \
        --outdir      data/variant_scores
"""

import argparse
import gzip
import os
import sys

import numpy as np
import pandas as pd

WIDTH = 501            # 500 bp uniform width, R's promoters(250, 251) -> 501
HALF = 250             # summit offset written into column 10
INPUT_WINDOW = 2114    # chrombpnet model receptive field
CANONICAL = ["chr%d" % i for i in range(1, 23)] + ["chrX", "chrY"]


def log(msg):
    print(msg, flush=True)


def read_chrom_sizes(path):
    sizes = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            f = line.split()
            sizes[f[0]] = int(f[1])
    return {c: sizes[c] for c in CANONICAL if c in sizes}


def read_blacklist(path, chrom_sizes, slop):
    """Blacklist intervals padded by `slop` on each side, clipped to chromosome."""
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(("#", "track", "browser")) or not line.strip():
                continue
            f = line.split()
            c = f[0]
            if c not in chrom_sizes:
                continue
            s = max(0, int(f[1]) - slop)
            e = min(chrom_sizes[c], int(f[2]) + slop)
            if e > s:
                rows.append((c, s, e))
    df = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    log("  blacklist intervals (slop %d): %d" % (slop, len(df)))
    return df


def load_sample(cell_id, summits_path, chrom_sizes, cap):
    """One sample -> fixed-width windows with within-sample percentile scores."""
    df = pd.read_csv(
        summits_path, sep="\t", header=None,
        names=["chrom", "start", "end", "name", "score"],
        usecols=[0, 1, 2, 3, 4],
        dtype={"chrom": str, "start": np.int64, "end": np.int64,
               "name": str, "score": np.float64},
    )
    n_raw = len(df)

    df = df[df["chrom"].isin(chrom_sizes)].copy()

    # Percentile-normalise score within this sample. R's ecdf(x)(x) gives the
    # fraction of values <= x, i.e. 'max' ranking divided by n.
    df["score_norm"] = df["score"].rank(method="max").to_numpy() / len(df)

    # Fixed-width window centred on the summit.
    summit = df["start"].to_numpy()
    df["win_start"] = summit - HALF
    df["win_end"] = summit + HALF + 1

    # Drop anything a chromosome boundary would truncate.
    chrlen = df["chrom"].map(chrom_sizes).to_numpy()
    keep = (df["win_start"].to_numpy() >= 0) & (df["win_end"].to_numpy() <= chrlen)
    n_trim = int((~keep).sum())
    df = df[keep]

    # Top `cap` by RAW score, matching capMaxPeaks in the R implementation.
    n_before_cap = len(df)
    if cap and len(df) > cap:
        df = df.nlargest(cap, "score", keep="all").head(cap)

    df = df.assign(cell_id=cell_id)
    log("    %-10s raw=%-8d canonical=%-8d trimmed=%-5d capped=%d"
        % (cell_id, n_raw, n_before_cap + n_trim, n_trim, len(df)))
    return df[["chrom", "win_start", "win_end", "score", "score_norm", "cell_id"]]


def greedy_nonoverlapping(df):
    """
    Iterative overlap removal, highest score_norm first.

    All windows are exactly WIDTH bp, so window i overlaps window j iff
    |start_i - start_j| < WIDTH. Bucketing starts by //WIDTH means any overlap
    partner lives in bucket b-1, b, or b+1, and each bucket can hold at most two
    accepted windows. That makes each test O(1).
    """
    order = np.lexsort((
        df["win_start"].to_numpy(),
        df["chrom"].to_numpy(),
        -df["score_norm"].to_numpy(),   # descending score_norm is primary
    ))

    chrom = df["chrom"].to_numpy()
    start = df["win_start"].to_numpy()

    accepted = np.zeros(len(df), dtype=bool)
    buckets = {}   # (chrom, bucket_index) -> list of accepted starts

    for i in order:
        c = chrom[i]
        s = start[i]
        b = s // WIDTH
        clash = False
        for bb in (b - 1, b, b + 1):
            for s_acc in buckets.get((c, bb), ()):
                if abs(s_acc - s) < WIDTH:
                    clash = True
                    break
            if clash:
                break
        if not clash:
            accepted[i] = True
            buckets.setdefault((c, b), []).append(s)

    return df[accepted]


def drop_blacklisted(df, bl, chrom_sizes, input_window):
    """
    Remove any window whose full model input window overlaps a blacklist region.
    The model reads input_window bp centred on the region, so the guard is the
    region padded by (input_window - WIDTH) / 2 on each side.
    """
    if bl.empty:
        return df
    pad = (input_window - WIDTH) // 2
    keep_mask = np.ones(len(df), dtype=bool)
    df = df.reset_index(drop=True)

    for c, sub in df.groupby("chrom", sort=False):
        bl_c = bl[bl["chrom"] == c]
        if bl_c.empty:
            continue
        bl_s = np.sort(bl_c["start"].to_numpy())
        bl_e = bl_c["end"].to_numpy()[np.argsort(bl_c["start"].to_numpy())]
        # running max of ends, so a binary search answers "does any earlier
        # interval reach past this point"
        bl_emax = np.maximum.accumulate(bl_e)

        q_s = sub["win_start"].to_numpy() - pad
        q_e = sub["win_end"].to_numpy() + pad

        idx = np.searchsorted(bl_s, q_e, side="left")   # intervals starting before q_e
        hit = np.zeros(len(sub), dtype=bool)
        nz = idx > 0
        hit[nz] = bl_emax[idx[nz] - 1] > q_s[nz]
        keep_mask[sub.index.to_numpy()] = ~hit

    log("  removed by blacklist (padded %d bp): %d" % (pad, int((~keep_mask).sum())))
    return df[keep_mask]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--chrom-sizes", required=True)
    p.add_argument("--blacklist", default=None)
    p.add_argument("--outdir", required=True)
    p.add_argument("--cap", type=int, default=150000,
                   help="max summits per sample by score (RoboATAC: 150000)")
    p.add_argument("--blacklist-slop", type=int, default=0,
                   help="extra bp to pad blacklist intervals before overlap test")
    p.add_argument("--input-window", type=int, default=INPUT_WINDOW,
                   help="chrombpnet input width used to guard the blacklist test")
    p.add_argument("--prefix", default="consensus_peaks_blood17")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    log("Reading chrom sizes")
    chrom_sizes = read_chrom_sizes(args.chrom_sizes)
    log("  canonical chromosomes: %d" % len(chrom_sizes))
    if len(chrom_sizes) != 24:
        log("  WARNING: expected 24, got %d" % len(chrom_sizes))

    man = pd.read_csv(args.manifest, sep="\t", dtype=str)
    log("\nLoading %d samples" % len(man))

    frames = []
    for _, r in man.iterrows():
        sp = os.path.join(r["peak_dir"], r["peak_prefix"] + "_summits.bed")
        if not os.path.isfile(sp):
            sys.exit("ERROR: missing summits file: %s" % sp)
        frames.append(load_sample(r["cell_id"], sp, chrom_sizes, args.cap))

    pooled = pd.concat(frames, ignore_index=True)
    log("\nPooled windows: %d" % len(pooled))

    log("\nIterative overlap removal by normalised score")
    cons = greedy_nonoverlapping(pooled)
    log("  non-overlapping consensus: %d" % len(cons))

    if args.blacklist:
        log("\nBlacklist filtering")
        bl = read_blacklist(args.blacklist, chrom_sizes, args.blacklist_slop)
        cons = drop_blacklisted(cons, bl, chrom_sizes, args.input_window)
        log("  after blacklist: %d" % len(cons))

    cons = cons.sort_values(
        ["chrom", "win_start"],
        key=lambda s: s.map({c: i for i, c in enumerate(CANONICAL)}) if s.name == "chrom" else s,
    ).reset_index(drop=True)

    # --- 10-column BED for chrombpnet -------------------------------------
    out10 = os.path.join(args.outdir, args.prefix + "_10col.bed")
    bed = pd.DataFrame({
        0: cons["chrom"],
        1: cons["win_start"],
        2: cons["win_end"],
        3: ["%s_%d" % (args.prefix, i + 1) for i in range(len(cons))],
        4: 0,
        5: ".",
        6: ".",
        7: ".",
        8: ".",
        9: HALF,
    })
    bed.to_csv(out10, sep="\t", header=False, index=False)
    log("\nWrote %s  (%d regions)" % (out10, len(bed)))

    # --- provenance: which cell type contributed each region ---------------
    outsrc = os.path.join(args.outdir, args.prefix + "_source.tsv")
    cons.assign(name=bed[3]).to_csv(
        outsrc, sep="\t", index=False,
        columns=["chrom", "win_start", "win_end", "name", "score",
                 "score_norm", "cell_id"],
    )
    log("Wrote %s" % outsrc)

    # --- summary ------------------------------------------------------------
    log("\nRegions contributed per cell type (winner of each merge):")
    for cid, n in cons["cell_id"].value_counts().sort_index().items():
        log("  %-10s %8d  (%5.2f%%)" % (cid, n, 100.0 * n / len(cons)))

    log("\nRegions per chromosome:")
    vc = cons["chrom"].value_counts()
    for c in CANONICAL:
        if c in vc:
            log("  %-6s %8d" % (c, vc[c]))

    widths = (cons["win_end"] - cons["win_start"]).unique()
    log("\nWindow widths present: %s" % sorted(widths))
    if len(widths) != 1 or widths[0] != WIDTH:
        log("  WARNING: expected a single width of %d" % WIDTH)

    log("\nDone. Use %s as -r for contribs_bw / pred_bw across all 17 cell types."
        % out10)


if __name__ == "__main__":
    main()
