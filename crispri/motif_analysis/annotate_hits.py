#!/usr/bin/env python
"""
Annotate finemo hits.tsv with human-readable TF names from the matching
TomTom output, joining on motif_name (hits.tsv) == Pattern (tomtom file).

Adds columns:
    Match_1, Match_1_qvalue, Match_2, Match_2_qvalue, Match_3, Match_3_qvalue
    low_confidence_flags  -- semicolon-separated list of which of the three
                             matches have qvalue >= --qvalue-threshold
                             (e.g. "Match_2;Match_3"), or "" if all three
                             are confident, or "no_tomtom_match" if the
                             motif_name wasn't found in the tomtom file at all.

Usage:
    python annotate_hits.py \
        --hits finemo_hits/hits.tsv \
        --tomtom counts_tomtom_matches.tsv \
        --out finemo_hits/hits.annotated.tsv \
        [--qvalue-threshold 0.05]
"""
import argparse
import csv
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", required=True, help="finemo hits.tsv (or hits_unique.tsv)")
    ap.add_argument("--tomtom", required=True, help="counts_tomtom_matches.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--qvalue-threshold", type=float, default=0.05,
                     help="Matches with qvalue >= this are flagged low-confidence. Default 0.05.")
    args = ap.parse_args()

    # --- load tomtom matches, keyed by Pattern name ---
    tomtom = {}
    with open(args.tomtom, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        expected_cols = {"Pattern", "Match_1", "Match_1_qvalue", "Match_2",
                          "Match_2_qvalue", "Match_3", "Match_3_qvalue"}
        missing = expected_cols - set(reader.fieldnames)
        if missing:
            sys.exit(f"tomtom file is missing expected columns: {missing}\n"
                      f"Found columns: {reader.fieldnames}")
        for row in reader:
            tomtom[row["Pattern"]] = row

    n_pos = sum(1 for k in tomtom if k.startswith("pos_patterns."))
    n_neg = sum(1 for k in tomtom if k.startswith("neg_patterns."))
    print(f"Loaded {len(tomtom)} tomtom pattern rows ({n_pos} pos_patterns, {n_neg} neg_patterns).")
    if n_neg == 0:
        print("WARNING: no neg_patterns rows found in the tomtom file -- "
              "any neg_patterns.* hits will be flagged 'no_tomtom_match'.")

    # --- annotate hits ---
    new_cols = ["Match_1", "Match_1_qvalue", "Match_2", "Match_2_qvalue",
                "Match_3", "Match_3_qvalue", "low_confidence_flags"]

    n_hits = 0
    n_unmatched = 0
    unmatched_motifs = set()

    with open(args.hits, newline="") as fin, open(args.out, "w", newline="") as fout:
        # finemo's hits.tsv may be space- or tab-delimited; detect the delimiter.
        sample = fin.read(4096)
        fin.seek(0)
        delim = "\t" if "\t" in sample.splitlines()[0] else " "
        reader = csv.DictReader(fin, delimiter=delim)
        reader.fieldnames = [c.strip() for c in reader.fieldnames]

        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames + new_cols, delimiter="\t")
        writer.writeheader()

        for row in reader:
            n_hits += 1
            motif = row["motif_name"]
            tt = tomtom.get(motif)
            if tt is None:
                n_unmatched += 1
                unmatched_motifs.add(motif)
                for c in new_cols[:-1]:
                    row[c] = "NA"
                row["low_confidence_flags"] = "no_tomtom_match"
            else:
                flags = []
                for i in (1, 2, 3):
                    match_col = f"Match_{i}"
                    qval_col = f"Match_{i}_qvalue"
                    row[match_col] = tt[match_col]
                    row[qval_col] = tt[qval_col]
                    try:
                        qv = float(tt[qval_col])
                        if qv >= args.qvalue_threshold:
                            flags.append(match_col)
                    except ValueError:
                        flags.append(f"{match_col}_unparseable_qvalue")
                row["low_confidence_flags"] = ";".join(flags) if flags else ""
            writer.writerow(row)

    print(f"\nAnnotated {n_hits} hits -> {args.out}")
    if n_unmatched:
        print(f"WARNING: {n_unmatched} hits had a motif_name not found in the tomtom file: "
              f"{sorted(unmatched_motifs)}")
        print("These usually mean the tomtom file doesn't cover neg_patterns, or the "
              "hits.tsv/tomtom files came from different modisco runs -- double check.")


if __name__ == "__main__":
    main()
