#!/usr/bin/env python
"""
Regenerate the region_id -> original CRE coordinate + ChromBPNet
prediction mapping.

The map also carries pred_log_counts / pred_counts per region, so downstream
steps can use ChromBPNet's own predicted accessibility as a covariate.

region_N indices were assigned by build_finemo_inputs.py in exactly this
order: shard files sorted alphabetically, then rows within each shard in
on-disk order. This script replays that same order over the same shard
directory to reconstruct the mapping, so it MUST be run against the same
--shard-dir used for build_finemo_inputs.py.

Usage:
    python build_region_map.py \
        --shard-dir chrombpnet_contribs_results/ \
        --out region_map.tsv
"""
import argparse
import glob
import os
import sys

import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    shard_paths = sorted(glob.glob(os.path.join(args.shard_dir, "shard_*.h5")))
    if not shard_paths:
        sys.exit(f"No shard_*.h5 files found in {args.shard_dir}")

    region_id = 0
    with open(args.out, "w") as out:
        out.write("region_id\tchrom\tbed_start_1based\tbed_end_1based\t"
                   "input_start_0based\tinput_end_0based\tclipped\t"
                   "pred_log_counts\tpred_counts\n")
        for sp in shard_paths:
            with h5py.File(sp, "r") as hf:
                n_written = int(hf.attrs.get("n_written", hf["chrom"].shape[0]))
                chrom = hf["chrom"][:n_written]
                bs = hf["bed_start_1based"][:n_written]
                be = hf["bed_end_1based"][:n_written]
                ws = hf["input_start_0based"][:n_written]
                we = hf["input_end_0based"][:n_written]
                clipped = hf["clipped"][:n_written]
                plc = hf["pred_log_counts"][:n_written]
                pc = hf["pred_counts"][:n_written]
                for i in range(n_written):
                    c = chrom[i].decode() if isinstance(chrom[i], bytes) else chrom[i]
                    out.write(f"region_{region_id}\t{c}\t{int(bs[i])}\t{int(be[i])}\t"
                              f"{int(ws[i])}\t{int(we[i])}\t{bool(clipped[i])}\t"
                              f"{float(plc[i])}\t{float(pc[i])}\n")
                    region_id += 1

    print(f"Wrote {region_id} region mappings to {args.out}")


if __name__ == "__main__":
    main()
