#!/usr/bin/env python
"""
vep_collapse.py: reduce raw VEP --tab output to one row per variant per consequence.

VEP emits one row per variant per transcript or regulatory feature, and each row's
Consequence field can itself hold several comma-separated terms. A variant therefore
appears many times. This collapses that to the two forms the quadrant analysis needs:

  vep_consequences.tsv   long: variant_key, consequence  (one row per distinct pair)
  vep_most_severe.tsv    one row per variant: most severe consequence, gene, biotype

The long form is the one 07_quadrants.py consumes. Consequence classes are deliberately
NOT made mutually exclusive: the question asked of each class is whether a variant
carries it in any feature, which is how the counts per class are meant to be read. A
variant that is both intronic and a regulatory-region variant is counted under both,
because both facts are true of it.

The most-severe assignment uses Ensembl's own ordering of consequence terms, kept here
as an explicit list so the ranking is auditable and does not depend on the order VEP
happens to emit rows in. A term VEP produces that is absent from the list is ranked last
and reported, rather than being silently dropped or silently treated as severe.

Usage
-----
  python vep_collapse.py --vep-output vep_out.txt --out-dir <dir>
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd

from s4lib import log

#: Ensembl consequence terms, most severe first.
#: https://www.ensembl.org/info/genome/variation/prediction/predicted_data.html
SEVERITY = [
    "transcript_ablation", "splice_acceptor_variant", "splice_donor_variant",
    "stop_gained", "frameshift_variant", "stop_lost", "start_lost",
    "transcript_amplification", "feature_elongation", "feature_truncation",
    "inframe_insertion", "inframe_deletion", "missense_variant",
    "protein_altering_variant", "splice_donor_5th_base_variant", "splice_region_variant",
    "splice_donor_region_variant", "splice_polypyrimidine_tract_variant",
    "incomplete_terminal_codon_variant", "start_retained_variant", "stop_retained_variant",
    "synonymous_variant", "coding_sequence_variant", "mature_miRNA_variant",
    "5_prime_UTR_variant", "3_prime_UTR_variant", "non_coding_transcript_exon_variant",
    "intron_variant", "NMD_transcript_variant", "non_coding_transcript_variant",
    "coding_transcript_variant", "upstream_gene_variant", "downstream_gene_variant",
    "TFBS_ablation", "TFBS_amplification", "TF_binding_site_variant",
    "regulatory_region_ablation", "regulatory_region_amplification",
    "regulatory_region_variant", "intergenic_variant", "sequence_variant",
]
RANK = {t: i for i, t in enumerate(SEVERITY)}


def read_vep_tab(path: Path) -> pd.DataFrame:
    """Read VEP --tab output, skipping the ## metadata block.

    The real header line begins with a single '#', which pandas would otherwise treat as
    a comment along with the metadata, so the skip count is found by scanning.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    skip = 0
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                skip += 1
            else:
                break
    df = pd.read_csv(path, sep="\t", skiprows=skip, low_memory=False)
    df = df.rename(columns={df.columns[0]: df.columns[0].lstrip("#")})
    need = {"Uploaded_variation", "Consequence"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing {sorted(missing)}. Columns are: "
                       f"{list(df.columns)}. Re-run VEP with --tab and include those "
                       f"fields in --fields.")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vep-output", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = read_vep_tab(args.vep_output)
    log(f"VEP rows {len(df):,} over {df['Uploaded_variation'].nunique():,} variants")

    # explode the comma-separated Consequence field
    long = (df[["Uploaded_variation", "Consequence"]]
            .assign(consequence=lambda d: d["Consequence"].astype(str).str.split(","))
            .explode("consequence"))
    long["consequence"] = long["consequence"].str.strip()
    long = long[long["consequence"].ne("") & long["consequence"].ne("-")]
    long = (long.rename(columns={"Uploaded_variation": "variant_key"})
                [["variant_key", "consequence"]].drop_duplicates())

    unknown = sorted(set(long["consequence"]) - set(RANK))
    if unknown:
        log(f"WARNING: consequence terms not in the severity list, ranked last: {unknown}")

    long.to_csv(args.out_dir / "vep_consequences.tsv", sep="\t", index=False)

    # most severe per variant, with the feature it came from
    long["rank"] = long["consequence"].map(RANK).fillna(len(SEVERITY)).astype(int)
    top = long.loc[long.groupby("variant_key")["rank"].idxmin(),
                   ["variant_key", "consequence"]]
    extra = [c for c in ("SYMBOL", "Gene", "BIOTYPE", "IMPACT") if c in df.columns]
    if extra:
        first = (df.rename(columns={"Uploaded_variation": "variant_key"})
                   .drop_duplicates(subset=["variant_key"], keep="first")
                   [["variant_key", *extra]])
        top = top.merge(first, on="variant_key", how="left")
    top.to_csv(args.out_dir / "vep_most_severe.tsv", sep="\t", index=False)

    log(f"distinct consequence classes {long['consequence'].nunique()}")
    log("\n".join(f"  {c:<38s} {n:>8,}"
                  for c, n in long["consequence"].value_counts().head(15).items()))
    log(f"wrote {args.out_dir / 'vep_consequences.tsv'}")
    log(f"wrote {args.out_dir / 'vep_most_severe.tsv'}")


if __name__ == "__main__":
    main()
