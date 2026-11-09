#!/usr/bin/env python
"""
Build the input BED for chrombpnet_bed_counts_contribs.py from the CRISPRi
CRE table (ALL7_trimmed.csv), filtered to Cell_type == K562.

Coordinate handling: CRE_start_hg38 / CRE_end_hg38 in this CSV are 1-based
inclusive, so they are used as-is (no +1 conversion).

chrombpnet_bed_counts_contribs.py centres its own 2114 bp window on each
interval's midpoint (window_for_interval()), regardless of the interval's width,
so this script passes through each CRE's own start/end.

Output format: BED-plus. The first 3 columns (chrom, start_1based, end_1based)
are followed by tab-separated metadata columns that trace each interval back to
its source CSV row (Target Gene Name, Reference, Cell_type, csv_row).
chrombpnet_bed_counts_contribs.py reads only the first 3 columns.

Usage:
    python make_crispri_input_bed.py \
        --in-csv data/crispri/ALL7_trimmed.csv \
        --out-dir results/crispri_motifs/input \
        --cell-type K562
"""

import argparse
import csv
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-csv", required=True, help="Path to ALL7_trimmed.csv")
    p.add_argument("--out-dir", required=True, help="Directory to write the output BED into (created if missing).")
    p.add_argument("--cell-type", default="K562", help="Only keep rows where Cell_type equals this value (default: K562).")
    return p.parse_args()


def parse_coord(raw):
    """Coordinates in this CSV sometimes appear as '1159680.0' (float-
    formatted) and sometimes as '1120161' (plain int) -- parse via float
    first, then round to int, to handle both without erroring."""
    return int(round(float(raw)))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    out_path = os.path.join(args.out_dir, f"crispri_{args.cell_type}.hg38.1based.bed")

    n_total = 0
    n_kept = 0
    n_skipped_cell_type = 0
    n_skipped_bad_coord = 0
    n_skipped_bad_interval = 0

    with open(args.in_csv, newline="") as fin, open(out_path, "w") as fout:
        reader = csv.DictReader(fin)
        required_cols = {"chr", "CRE_start_hg38", "CRE_end_hg38", "Cell_type"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV is missing expected column(s): {missing}. Found columns: {reader.fieldnames}")

        for row_idx, row in enumerate(reader, start=2):  # start=2: row 1 is the header
            n_total += 1

            if row.get("Cell_type", "").strip() != args.cell_type:
                n_skipped_cell_type += 1
                continue

            chrom = row["chr"].strip()
            try:
                start1 = parse_coord(row["CRE_start_hg38"])
                end1 = parse_coord(row["CRE_end_hg38"])
            except (ValueError, TypeError):
                print(f"WARNING: row {row_idx} has unparseable coordinates "
                      f"(chr={row.get('chr')!r}, start={row.get('CRE_start_hg38')!r}, "
                      f"end={row.get('CRE_end_hg38')!r}) -- skipping")
                n_skipped_bad_coord += 1
                continue

            if end1 <= start1:
                print(f"WARNING: row {row_idx} has end <= start ({chrom}:{start1}-{end1}) -- skipping")
                n_skipped_bad_interval += 1
                continue

            target_gene = row.get("Target Gene Name", "").strip()
            reference = row.get("Reference", "").strip()
            cell_type = row.get("Cell_type", "").strip()

            # Extra columns (4+) trace a scored interval back to its source CSV row.
            fout.write(
                f"{chrom}\t{start1}\t{end1}\t"
                f"{target_gene}\t{reference}\t{cell_type}\t"
                f"csv_row={row_idx}\n"
            )
            n_kept += 1

    print(f"Read {args.in_csv}")
    print(f"Total data rows: {n_total}")
    print(f"Kept ({args.cell_type}): {n_kept}")
    print(f"Skipped (wrong Cell_type): {n_skipped_cell_type}")
    print(f"Skipped (unparseable coordinates): {n_skipped_bad_coord}")
    print(f"Skipped (end <= start): {n_skipped_bad_interval}")
    print(f"Wrote -> {out_path}")


if __name__ == "__main__":
    main()
