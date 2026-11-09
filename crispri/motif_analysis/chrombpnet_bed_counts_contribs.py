#!/usr/bin/env python
"""
ChromBPNet: predicted counts + counts-head contribution scores (DeepSHAP)
for millions of genomic intervals (REF sequence only, hg38).

Designed to run as one shard of a SLURM/cluster array job.
Each shard reads a slice of the BED file, processes it in batches, and
writes results incrementally to its own HDF5 file so a crash/timeout only
loses the current batch, not the whole shard (resumable via --resume).

Input BED: chrom<TAB>start<TAB>end, 1-based start and end,
one interval per line, hg38. (Internally converted to 0-based half-open,
which is what pysam/bigwig/etc. expect.)

Typical SLURM array usage, e.g. 500 shards over ~2.3M ENCODE SCREEN CREs:
    sbatch --array=0-499 run_shard.sh
  where run_shard.sh calls:
    python chrombpnet_bed_counts_contribs.py \
        --bed screen_ccres.hg38.bed \
        --model chrombpnet_nobias.h5 \
        --fasta hg38.fa \
        --out-dir out/ \
        --shard-idx $SLURM_ARRAY_TASK_ID \
        --n-shards 500

Requires: pysam, h5py, tensorflow, the Kundaje-lab shap fork
(github.com/kundajelab/shap; plain PyPI shap does not support these functional
Keras models), chrombpnet (for losses/one_hot utils).

DeepSHAP contribution scores use shuffle_several_times and
combine_mult_and_diffref from kundajelab/variant-scorer's src/utils/shap_utils.py,
fetched locally by fetch_variant_scorer.sh, so the counts-head contribution
scores follow the variant-scorer / ChromBPNet method.
"""

import os
import sys
import argparse
import numpy as np
import h5py
import pysam

import tensorflow as tf
# shap's legacy DeepExplainer backend only supports TF1-style graph mode, not TF2
# eager functional models. variant-scorer's shap_utils.py makes the same call.
tf.compat.v1.disable_v2_behavior()

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import chrombpnet.training.utils.losses as losses
import chrombpnet.training.utils.one_hot as one_hot

import shap  # must resolve to github.com/kundajelab/shap, not PyPI shap

# Local clone of kundajelab/variant-scorer (see fetch_variant_scorer.sh). shap_utils.py
# imports a sibling `generators` package, so the whole src/ tree must be on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "variant-scorer", "src"))
from utils.shap_utils import shuffle_several_times, combine_mult_and_diffref

SEQ_LEN = 2114  # ChromBPNet input length (don't change)


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bed", required=True, help="BED file, 1-based start/end, one interval per line (chrom, start, end; extra columns ignored).")
    p.add_argument("--model", required=True, help="Path to chrombpnet_nobias.h5")
    p.add_argument("--fasta", required=True, help="Path to hg38.fa (indexed with .fai)")
    p.add_argument("--out-dir", default=None, help="Output directory; one HDF5 file per shard is written here. "
                    "Default: a 'chrombpnet_contribs_results' folder created inside the current working directory.")
    p.add_argument("--shard-idx", type=int, required=True, help="0-based index of this shard.")
    p.add_argument("--n-shards", type=int, required=True, help="Total number of shards the BED is split into.")
    p.add_argument("--batch-size", type=int, default=64, help="Sequences per prediction/DeepSHAP batch (CPU: keep modest, default 64).")
    p.add_argument("--n-shuffles", type=int, default=20, help="Unused: shuffle_several_times() uses 20 "
                    "dinucleotide shuffles internally. Edit lib/variant-scorer/src/utils/shap_utils.py to change it.")
    p.add_argument("--resume", action="store_true", help="If the shard's HDF5 already exists, skip intervals already written and continue.")
    return p.parse_args()


# ---------------- BED sharding ----------------
def load_bed_shard(bed_path, shard_idx, n_shards):
    """Read the whole BED file's coordinates (cheap: just text), then take
    every n_shards-th line offset by shard_idx (round-robin sharding, so
    shards stay balanced even if the BED is chromosome-sorted)."""
    intervals = []
    with open(bed_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split("\t")
            chrom, start_1based, end_1based = fields[0], int(fields[1]), int(fields[2])
            intervals.append((chrom, start_1based, end_1based))

    shard = intervals[shard_idx::n_shards]
    return shard


# ---------------- Model loading ----------------
def load_model_wrapper(model_h5):
    custom_objects = {"multinomial_nll": losses.multinomial_nll, "tf": tf}
    get_custom_objects().update(custom_objects)
    return load_model(model_h5, compile=False)


def softmax(x, temp=1):
    norm_x = x - np.mean(x, axis=1, keepdims=True)
    return np.exp(temp * norm_x) / np.sum(np.exp(temp * norm_x), axis=1, keepdims=True)


# ---------------- Sequence extraction ----------------
def window_for_interval(chrom, start_1based, end_1based, chrom_len):
    """Center a SEQ_LEN window on the interval midpoint. Coordinates are
    converted to 0-based half-open. If the window would run off the
    chromosome edge, it's clipped back on (shifted, not truncated) so it
    always returns exactly SEQ_LEN bp -- flagged via the returned `clipped`
    bool so those intervals can be excluded or inspected downstream."""
    start0 = start_1based - 1
    end0 = end_1based  # 1-based inclusive end -> 0-based exclusive end, same number
    mid = (start0 + end0) // 2

    win_start = mid - SEQ_LEN // 2
    win_end = win_start + SEQ_LEN

    clipped = False
    if win_start < 0:
        win_start = 0
        win_end = SEQ_LEN
        clipped = True
    if win_end > chrom_len:
        win_end = chrom_len
        win_start = chrom_len - SEQ_LEN
        clipped = True
    if win_start < 0:
        # chromosome itself shorter than SEQ_LEN -- can't happen for real hg38
        # chroms, but guard anyway.
        raise ValueError(f"Chromosome {chrom} (len {chrom_len}) shorter than SEQ_LEN={SEQ_LEN}")

    return win_start, win_end, clipped


def fetch_batch_seqs(fa, batch_intervals, chrom_lens):
    seqs = []
    windows = []
    for chrom, s1, e1 in batch_intervals:
        clen = chrom_lens[chrom]
        ws, we, clipped = window_for_interval(chrom, s1, e1, clen)
        seq = fa.fetch(chrom, ws, we).upper()
        # N's are allowed; one_hot encodes them as all-zero rows
        assert len(seq) == SEQ_LEN, f"{chrom}:{ws}-{we} returned {len(seq)}bp, expected {SEQ_LEN}"
        seqs.append(seq)
        windows.append((chrom, ws, we, clipped))
    return seqs, windows


# ---------------- Counts-head DeepSHAP (variant-scorer's own method) ----------------
def make_counts_explainer(model):
    """Build the DeepSHAP explainer once per shard, using variant-scorer's
    background and combine functions.

    shuffle_several_times is passed as the `data` argument (not a static array):
    shap calls it once per example during shap_values(), so every sequence gets
    its own dinucleotide-shuffled background while the explainer graph is built
    only once.

    The model (chrombpnet_nobias.h5) takes a single one-hot sequence input.
    shuffle_several_times has a single-input branch (len(s) != 2), so the input
    is passed as a length-1 list.

    The explained tensor is tf.reduce_sum(model.outputs[1], axis=-1), as in
    variant-scorer's counts-head construction."""
    model_inputs = [model.input]
    counts_output_tensor = tf.reduce_sum(model.outputs[1], axis=-1)
    explainer = shap.explainers.deep.TFDeepExplainer(
        (model_inputs, counts_output_tensor),
        shuffle_several_times,
        combine_mult_and_diffref=combine_mult_and_diffref,
    )
    return explainer


def compute_counts_contribs(explainer, one_hot_batch):
    """one_hot_batch: (B, SEQ_LEN, 4). Returns:
       hyp_scores: (B, SEQ_LEN, 4) hypothetical importance scores
       contrib:    (B, SEQ_LEN)    actual (projected onto observed base) scores,
                                    i.e. hyp_scores * one_hot, summed over bases
                                    -- same downstream projection convention
                                    variant-scorer's own output format uses.
    Whole batch through the already-built explainer in one call; the
    per-sequence dinucleotide-shuffled background is generated internally
    by shap calling shuffle_several_times for each example."""
    sv = explainer.shap_values([one_hot_batch.astype(np.float32)])
    if isinstance(sv, list):
        sv = sv[0]
    hyp_scores = np.array(sv)  # (B, SEQ_LEN, 4)
    contrib = (hyp_scores * one_hot_batch).sum(axis=-1)  # (B, SEQ_LEN)
    return hyp_scores, contrib


# ---------------- Main ----------------
def main():
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(os.getcwd(), "chrombpnet_contribs_results")
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"shard_{args.shard_idx:05d}_of_{args.n_shards:05d}.h5")

    print(f"[shard {args.shard_idx}/{args.n_shards}] loading BED shard from {args.bed} ...")
    shard_intervals = load_bed_shard(args.bed, args.shard_idx, args.n_shards)
    n_total = len(shard_intervals)
    print(f"[shard {args.shard_idx}] {n_total} intervals assigned to this shard")

    start_idx = 0
    if args.resume and os.path.exists(out_path):
        with h5py.File(out_path, "r") as hf:
            start_idx = int(hf.attrs.get("n_written", 0))
        print(f"[shard {args.shard_idx}] resuming from interval {start_idx}/{n_total}")

    if start_idx >= n_total:
        print(f"[shard {args.shard_idx}] already complete, nothing to do.")
        return

    print("Loading model...")
    model = load_model_wrapper(args.model)

    fa = pysam.FastaFile(args.fasta)
    chrom_lens = dict(zip(fa.references, fa.lengths))

    # Build the DeepSHAP explainer ONCE for this whole shard. No seed-
    # sequence sampling needed anymore -- shuffle_several_times generates
    # each example's own dinucleotide-shuffled background dynamically
    # inside shap_values(), so there's no static background to precompute.
    print(f"[shard {args.shard_idx}] building DeepSHAP explainer (variant-scorer's shap_utils)...")
    counts_explainer = make_counts_explainer(model)
    print(f"[shard {args.shard_idx}] explainer ready.")

    mode = "a" if (args.resume and os.path.exists(out_path)) else "w"
    with h5py.File(out_path, mode) as hf:
        if mode == "w":
            str_dt = h5py.string_dtype(encoding="utf-8")
            hf.create_dataset("chrom", shape=(n_total,), maxshape=(n_total,), dtype=str_dt)
            hf.create_dataset("bed_start_1based", shape=(n_total,), maxshape=(n_total,), dtype="i8")
            hf.create_dataset("bed_end_1based", shape=(n_total,), maxshape=(n_total,), dtype="i8")
            hf.create_dataset("input_start_0based", shape=(n_total,), maxshape=(n_total,), dtype="i8")
            hf.create_dataset("input_end_0based", shape=(n_total,), maxshape=(n_total,), dtype="i8")
            hf.create_dataset("clipped", shape=(n_total,), maxshape=(n_total,), dtype="bool")
            hf.create_dataset("pred_log_counts", shape=(n_total,), maxshape=(n_total,), dtype="f4")
            hf.create_dataset("pred_counts", shape=(n_total,), maxshape=(n_total,), dtype="f4")
            hf.create_dataset(
                "counts_contrib", shape=(n_total, SEQ_LEN), maxshape=(n_total, SEQ_LEN),
                dtype="f4", compression="gzip", compression_opts=4, chunks=(1, SEQ_LEN),
            )
            hf.create_dataset(
                "counts_hyp_scores", shape=(n_total, SEQ_LEN, 4), maxshape=(n_total, SEQ_LEN, 4),
                dtype="f4", compression="gzip", compression_opts=4, chunks=(1, SEQ_LEN, 4),
            )
            hf.attrs["n_written"] = 0
            hf.attrs["seq_len"] = SEQ_LEN
            hf.attrs["shap_method"] = "kundajelab/variant-scorer shap_utils.py (shuffle_several_times + combine_mult_and_diffref), vendored"
            hf.attrs["shap_source_url"] = "https://github.com/kundajelab/variant-scorer/blob/main/src/utils/shap_utils.py"
            hf.attrs["shap_package"] = "github.com/kundajelab/shap (not PyPI shap)"

        n_written = start_idx
        for batch_start in range(start_idx, n_total, args.batch_size):
            batch_end = min(batch_start + args.batch_size, n_total)
            batch_intervals = shard_intervals[batch_start:batch_end]

            seqs, windows = fetch_batch_seqs(fa, batch_intervals, chrom_lens)
            one_hot_batch = one_hot.dna_to_one_hot(seqs).astype(np.float32)  # (B, SEQ_LEN, 4)

            pred_logits, pred_logcts = model.predict(one_hot_batch, verbose=0)
            pred_counts = np.exp(pred_logcts[:, 0])

            hyp_scores, contrib = compute_counts_contribs(counts_explainer, one_hot_batch)

            for j, (chrom, ws, we, clipped) in enumerate(windows):
                idx = batch_start + j
                hf["chrom"][idx] = chrom
                hf["bed_start_1based"][idx] = batch_intervals[j][1]
                hf["bed_end_1based"][idx] = batch_intervals[j][2]
                hf["input_start_0based"][idx] = ws
                hf["input_end_0based"][idx] = we
                hf["clipped"][idx] = clipped
                hf["pred_log_counts"][idx] = pred_logcts[j, 0]
                hf["pred_counts"][idx] = pred_counts[j]
                hf["counts_contrib"][idx] = contrib[j]
                hf["counts_hyp_scores"][idx] = hyp_scores[j]

            n_written = batch_end
            hf.attrs["n_written"] = n_written
            hf.flush()
            print(f"[shard {args.shard_idx}] {n_written}/{n_total} intervals done", flush=True)

    fa.close()
    print(f"[shard {args.shard_idx}] complete. Wrote {out_path}")


if __name__ == "__main__":
    main()
