#!/usr/bin/env python
"""
Join annotated Fi-NeMo hits back to the original CRISPRi CRE table and
build per-element x per-TF feature matrices, in several representations:

    features_presence.tsv     1/0 -- >=1 hit with Match_1_qvalue < --qvalue-threshold
    features_count.tsv        integer count of such hits
    features_importance.tsv   sum of hit_importance across such hits
    features_count_norm.tsv   features_count, divided by CRE length (per kb)
    features_soft.tsv         sum of hit_coefficient_global * (1 - Match_1_qvalue)
                               across all hits with a named Match_1, with no hard
                               confidence cutoff. Borderline-confidence motif
                               calls are down-weighted continuously by TomTom
                               confidence instead of being discarded.

All tables are joined with element metadata: has_target_gene,
N_validated_target_genes, the original covariate columns, CRE_length_hg38
(for length control), and pred_log_counts/pred_counts (ChromBPNet's own
predicted accessibility for that element's window, pulled from the shard
H5s via region_map.tsv), so that motif signal can be assessed after accounting
for the model's own accessibility prediction.

A hit is used only if:
  - its peak_id/peak_name maps (via region_map.tsv) to a K562 CRE row
    in the CSV (matched on chr + CRE_start_hg38 + CRE_end_hg38)
  - its [start, end) falls entirely inside that CRE's own boundary
    [bed_start_1based - 1, bed_end_1based)  (0-based half-open)
For features_presence/count/importance/count_norm, additionally:
  - its Match_1_qvalue < --qvalue-threshold (default 0.1)
features_soft has no q-value cutoff at all -- every hit with a Match_1
name contributes, weighted by (1 - Match_1_qvalue).

Usage:
    python build_element_motif_features.py \
        --csv ALL7_trimmed.csv \
        --hits finemo_hits/hits.annotated.tsv \
        --region-map region_map.tsv \
        --out-dir motif_features/ \
        --qvalue-threshold 0.1
"""
import argparse
import csv
import os
import sys
from collections import defaultdict


def load_csv_k562(path):
    """Returns dict: (chr, cre_start_int, cre_end_int) -> row dict,
    restricted to Cell_type == 'K562'."""
    rows = {}
    n_bad = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Cell_type") != "K562":
                continue
            try:
                c = row["chr"]
                s = int(float(row["CRE_start_hg38"]))
                e = int(float(row["CRE_end_hg38"]))
            except (ValueError, KeyError):
                n_bad += 1
                continue
            key = (c, s, e)
            if key in rows:
                print(f"WARNING: duplicate K562 CRE key {key} in CSV -- keeping first occurrence", file=sys.stderr)
                continue
            rows[key] = row
    if n_bad:
        print(f"Skipped {n_bad} K562 rows with non-numeric CRE_start_hg38/CRE_end_hg38 (e.g. 'liftover failed').")
    return rows


def load_region_map(path):
    """Returns dict: region_name ('region_N') ->
    (chrom, bed_start_1based, bed_end_1based, pred_log_counts, pred_counts)"""
    m = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            m[row["region_id"]] = (
                row["chrom"], int(row["bed_start_1based"]), int(row["bed_end_1based"]),
                float(row["pred_log_counts"]), float(row["pred_counts"]),
            )
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="ALL7_trimmed.csv")
    ap.add_argument("--hits", required=True, help="hits.annotated.tsv (from annotate_hits.py)")
    ap.add_argument("--region-map", required=True, help="region_map.tsv (from build_region_map.py, with pred columns)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--qvalue-threshold", type=float, default=0.1,
                     help="Hard cutoff used for presence/count/importance/count_norm tables. "
                          "Does not affect features_soft.tsv, which has no cutoff.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csv_rows = load_csv_k562(args.csv)
    print(f"Loaded {len(csv_rows)} K562 CRE rows from {args.csv}")

    region_map = load_region_map(args.region_map)
    print(f"Loaded {len(region_map)} region mappings from {args.region_map}")

    element_meta = {}
    presence = defaultdict(set)
    counts = defaultdict(lambda: defaultdict(int))
    importance = defaultdict(lambda: defaultdict(float))
    soft = defaultdict(lambda: defaultdict(float))  # no hard cutoff, confidence-weighted

    all_tfs = set()

    n_hits_total = 0
    n_no_region_map = 0
    n_no_csv_match = 0
    n_outside_cre = 0
    n_no_name = 0
    n_used_hard = 0
    n_used_soft = 0

    with open(args.hits, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = "\t" if "\t" in sample.splitlines()[0] else " "
        reader = csv.DictReader(f, delimiter=delim)
        reader.fieldnames = [c.strip() for c in reader.fieldnames]

        for row in reader:
            n_hits_total += 1
            region_name = row["peak_name"]

            rm = region_map.get(region_name)
            if rm is None:
                n_no_region_map += 1
                continue
            chrom, bed_start_1based, bed_end_1based, pred_log_counts, pred_counts = rm

            csv_key = (chrom, bed_start_1based, bed_end_1based)
            csv_row = csv_rows.get(csv_key)
            if csv_row is None:
                n_no_csv_match += 1
                continue

            element_id = f"{chrom}:{bed_start_1based}-{bed_end_1based}"
            if element_id not in element_meta:
                element_meta[element_id] = {
                    "csv_row": csv_row,
                    "pred_log_counts": pred_log_counts,
                    "pred_counts": pred_counts,
                }

            cre_start0 = bed_start_1based - 1
            cre_end0 = bed_end_1based
            hit_start, hit_end = int(row["start"]), int(row["end"])
            if not (hit_start >= cre_start0 and hit_end <= cre_end0):
                n_outside_cre += 1
                continue

            tf = row.get("Match_1")
            flags = row.get("low_confidence_flags", "")
            if not tf or flags == "no_tomtom_match":
                n_no_name += 1
                continue

            try:
                q1 = float(row["Match_1_qvalue"])
            except (ValueError, TypeError):
                n_no_name += 1
                continue

            all_tfs.add(tf)

            try:
                hit_imp = float(row["hit_importance"])
            except (ValueError, KeyError):
                hit_imp = 0.0
            try:
                hit_coef_global = float(row["hit_coefficient_global"])
            except (ValueError, KeyError):
                hit_coef_global = 0.0

            # soft-weighted: always contributes, weighted by (1 - qvalue)
            soft[element_id][tf] += hit_coef_global * (1.0 - q1)
            n_used_soft += 1

            # hard-threshold tables: only if confident enough
            if q1 < args.qvalue_threshold:
                presence[element_id].add(tf)
                counts[element_id][tf] += 1
                importance[element_id][tf] += hit_imp
                n_used_hard += 1

    print(f"\nHit accounting:")
    print(f"  total hits in hits.tsv:                {n_hits_total}")
    print(f"  dropped, region_id not in region_map:   {n_no_region_map}")
    print(f"  dropped, region not a K562 CSV row:     {n_no_csv_match}")
    print(f"  dropped, outside CRE boundary:          {n_outside_cre}")
    print(f"  dropped, no usable Match_1 name/qvalue: {n_no_name}")
    print(f"  used in soft-weighted table:             {n_used_soft}")
    print(f"  used in hard-threshold tables (q<{args.qvalue_threshold}): {n_used_hard}")

    if not element_meta:
        sys.exit("\nNo elements matched -- check that --hits/--region-map/--csv all come from the same run.")

    all_tfs = sorted(all_tfs)
    meta_cols_fixed = ["has_target_gene", "N_validated_target_genes", "ATAC_SignalValue", "ABC_score",
                        "GC_perc", "closest_TSS_distance_genome_hg38", "constraint_z",
                        "CRE_length_hg38", "Target Gene Name"]

    def meta_row(element_id):
        info = element_meta[element_id]
        r = info["csv_row"]
        out = {c: r.get(c, "") for c in meta_cols_fixed}
        out["pred_log_counts"] = info["pred_log_counts"]
        out["pred_counts"] = info["pred_counts"]
        return out

    extra_meta_cols = meta_cols_fixed + ["pred_log_counts", "pred_counts"]

    def write_table(path, value_fn):
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["element_id"] + extra_meta_cols + all_tfs)
            for element_id in element_meta:
                mr = meta_row(element_id)
                row_vals = [element_id] + [mr[c] for c in extra_meta_cols]
                row_vals += [value_fn(element_id, tf) for tf in all_tfs]
                w.writerow(row_vals)

    write_table(f"{args.out_dir}/features_presence.tsv",
                lambda eid, tf: int(tf in presence.get(eid, set())))
    write_table(f"{args.out_dir}/features_count.tsv",
                lambda eid, tf: counts.get(eid, {}).get(tf, 0))
    write_table(f"{args.out_dir}/features_importance.tsv",
                lambda eid, tf: round(importance.get(eid, {}).get(tf, 0.0), 6))
    write_table(f"{args.out_dir}/features_soft.tsv",
                lambda eid, tf: round(soft.get(eid, {}).get(tf, 0.0), 6))

    def cre_len_kb(eid):
        try:
            L = float(meta_row(eid)["CRE_length_hg38"])
            return max(L, 1.0) / 1000.0
        except (ValueError, TypeError):
            return 1.0

    write_table(f"{args.out_dir}/features_count_norm.tsv",
                lambda eid, tf: round(counts.get(eid, {}).get(tf, 0) / cre_len_kb(eid), 6))

    n_pos = sum(1 for eid in element_meta if str(meta_row(eid).get("has_target_gene")) == "True")
    print(f"\n{len(element_meta)} K562 elements with >=1 hit landing inside their CRE boundary "
          f"({n_pos} has_target_gene=True, {len(element_meta) - n_pos} False)")
    print(f"{len(all_tfs)} distinct named TF motifs found (any confidence): {all_tfs}")
    print(f"\nWrote (all in {args.out_dir}/):")
    print(f"  features_presence.tsv, features_count.tsv, features_importance.tsv,")
    print(f"  features_count_norm.tsv (count per kb), features_soft.tsv (confidence-weighted, no cutoff)")

    n_total_k562 = len(csv_rows)
    n_missing = n_total_k562 - len(element_meta)
    if n_missing:
        print(f"\nNOTE: {n_missing}/{n_total_k562} K562 CRE rows had no in-CRE-boundary hit at all "
              f"(regardless of confidence) -- these will still be zero-filled by run_stats.py.")


if __name__ == "__main__":
    main()
