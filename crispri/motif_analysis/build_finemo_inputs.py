#!/usr/bin/env python
"""
Convert the ChromBPNet contribution-score shard H5s written by
chrombpnet_bed_counts_contribs.py into the plain .npy/.npz inputs that Fi-NeMo's
`finemo extract-regions-modisco-fmt` command expects, then (optionally) invoke
that command.

Why not `finemo extract-regions-chrombpnet-h5`: the shard H5s use a flat schema
(`counts_hyp_scores`, `chrom`/`bed_start_1based`, no stored one-hot sequence)
rather than ChromBPNet's own `interpret` output schema (`raw/seq`, `shap/seq`,
`coords_chrom`, ...). Reformatting to numpy plus a NarrowPeak file and using the
modisco-fmt extractor avoids imitating the chrombpnet-h5 schema.

Steps this script performs:
  1. Reads every shard_*.h5 in --shard-dir, in shard order.
  2. For each region, fetches the REF one-hot sequence for
     [input_start_0based, input_end_0based) from --fasta (must be the
     same hg38.fa used for the original contribution-score run).
  3. Writes:
       sequences.npy    (n, 4, seq_len) int8, ACGT order
       attributions.npy (n, 4, seq_len) float16, hypothetical scores
       regions.narrowPeak  (summit column points at the window midpoint,
                            i.e. the same centre used by the contribution-score
                            script)
  4. Prints (and optionally runs) the exact `finemo extract-regions-modisco-fmt`
     command to crop these down to --region-width (e.g. 1000bp) and pack
     them into the .npz Fi-NeMo actually consumes for `call-hits`.

Usage:
    python build_finemo_inputs.py \
        --shard-dir chrombpnet_contribs_results/ \
        --fasta data/external/hg38.fa \
        --out-dir finemo_inputs/ \
        --region-width 1000 \
        [--run]   # actually invoke `finemo extract-regions-modisco-fmt` at the end
"""
import argparse
import glob
import os
import subprocess
import sys

import h5py
import numpy as np
import pysam

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def seq_to_onehot(seq):
    """seq: str of length L, upper-cased. Returns (4, L) int8, ACGT order.
    Non-ACGT bases (N, etc.) become an all-zero column, matching
    chrombpnet's one_hot.dna_to_one_hot convention."""
    L = len(seq)
    oh = np.zeros((4, L), dtype=np.int8)
    for i, b in enumerate(seq):
        idx = BASE_TO_IDX.get(b)
        if idx is not None:
            oh[idx, i] = 1
    return oh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True, help="Dir containing shard_*.h5 files")
    ap.add_argument("--fasta", required=True, help="hg38.fa used for the original contribution run (must have .fai)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--region-width", type=int, default=1000)
    ap.add_argument("--run", action="store_true", help="Also invoke `finemo extract-regions-modisco-fmt` at the end")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    shard_paths = sorted(glob.glob(os.path.join(args.shard_dir, "shard_*.h5")))
    if not shard_paths:
        sys.exit(f"No shard_*.h5 files found in {args.shard_dir}")
    print(f"Found {len(shard_paths)} shard files.")

    fa = pysam.FastaFile(args.fasta)

    all_seqs = []
    all_attrs = []
    narrowpeak_rows = []
    region_idx = 0
    seq_len = None

    for sp in shard_paths:
        with h5py.File(sp, "r") as hf:
            n_written = int(hf.attrs.get("n_written", hf["chrom"].shape[0]))
            this_seq_len = int(hf.attrs["seq_len"])
            if seq_len is None:
                seq_len = this_seq_len
            elif seq_len != this_seq_len:
                sys.exit(f"seq_len mismatch across shards: {seq_len} vs {this_seq_len} in {sp}")

            chrom = hf["chrom"][:n_written]
            ws = hf["input_start_0based"][:n_written]
            we = hf["input_end_0based"][:n_written]
            clipped = hf["clipped"][:n_written]
            hyp = hf["counts_hyp_scores"][:n_written]  # (n, seq_len, 4)

            n_clipped = int(clipped.sum())
            if n_clipped:
                print(f"  [{os.path.basename(sp)}] {n_clipped}/{n_written} regions were edge-clipped "
                      f"during contribution scoring (window shifted, not centered on the true midpoint). "
                      f"Their summit position below will be off-center; consider excluding them.")

            print(f"  [{os.path.basename(sp)}] {n_written} regions")

            for i in range(n_written):
                c = chrom[i].decode() if isinstance(chrom[i], bytes) else chrom[i]
                s0, e0 = int(ws[i]), int(we[i])
                seq = fa.fetch(c, s0, e0).upper()
                if len(seq) != seq_len:
                    sys.exit(f"Fetched {len(seq)}bp for {c}:{s0}-{e0}, expected {seq_len}. "
                             f"Is --fasta the same hg38 build used for the original contribution run?")
                oh = seq_to_onehot(seq)  # (4, seq_len)
                all_seqs.append(oh)

                # hyp: (seq_len, 4) -> (4, seq_len) to match sequences' axis order
                all_attrs.append(hyp[i].T.astype(np.float16))

                summit_offset = seq_len // 2  # matches window_for_interval's centering
                narrowpeak_rows.append(
                    f"{c}\t{s0}\t{e0}\tregion_{region_idx}\t0\t.\t0\t0\t0\t{summit_offset}"
                )
                region_idx += 1

    fa.close()

    sequences = np.stack(all_seqs, axis=0)      # (n, 4, seq_len) int8
    attributions = np.stack(all_attrs, axis=0)  # (n, 4, seq_len) float16

    seq_path = os.path.join(args.out_dir, "sequences.npy")
    attr_path = os.path.join(args.out_dir, "attributions.npy")
    peak_path = os.path.join(args.out_dir, "regions.narrowPeak")

    np.save(seq_path, sequences)
    np.save(attr_path, attributions)
    with open(peak_path, "w") as f:
        f.write("\n".join(narrowpeak_rows) + "\n")

    print(f"\nWrote:")
    print(f"  {seq_path}         shape={sequences.shape} dtype={sequences.dtype}")
    print(f"  {attr_path}      shape={attributions.shape} dtype={attributions.dtype}")
    print(f"  {peak_path}  ({region_idx} regions, full {seq_len}bp span; "
          f"summit at offset {seq_len // 2} = the original contribution-scoring midpoint)")

    npz_path = os.path.join(args.out_dir, "regions.npz")
    cmd = [
        "finemo", "extract-regions-modisco-fmt",
        "-s", seq_path,
        "-a", attr_path,
        "-p", peak_path,
        "-o", npz_path,
        "-w", str(args.region_width),
    ]
    print("\nNext step (crops each region from {}bp down to {}bp, centered on the summit, "
          "and packs into Fi-NeMo's .npz):\n".format(seq_len, args.region_width))
    print("  " + " ".join(cmd))

    if args.run:
        print("\nRunning it now (--run was passed)...")
        subprocess.run(cmd, check=True)
        print(f"\nDone. Region file ready at: {npz_path}")
        print("\nNow run hit calling, e.g.:")
        print(f"  finemo call-hits -r {npz_path} -m counts_modisco.h5 "
              f"-o finemo_hits/ -p {peak_path}")


if __name__ == "__main__":
    main()
