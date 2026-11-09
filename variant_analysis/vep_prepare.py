#!/usr/bin/env python
"""
vep_prepare.py: write the VEP input file for the quadrant analysis.

Emits every scored variant that appears in the credible-set file, in VEP's default
("ensembl") input format:

    chrom  start  end  REF/ALT  strand  identifier

The identifier is set to the pipeline's own `variant_key` (chr:pos:ref:alt) rather than
an rsID. VEP echoes it back as `Uploaded_variation`, which makes the join to the variant
table exact and total: rsIDs are missing for a large fraction of these variants and are
not unique when present.

Two conventions matter and are asserted rather than assumed:

  * The score files, and therefore `pos` in variants.parquet, are 1-based. VEP's default
    format is also 1-based, so start = end = pos for a SNV and no shift is applied.
  * Ensembl caches name chromosomes without a `chr` prefix. The prefix is stripped here
    and restored by the identifier, so nothing downstream depends on VEP's naming.

Indels are written with VEP's coordinate convention for insertions and deletions, and
counted separately in the summary so the numbers can be checked against the run log.

Usage
-----
  python vep_prepare.py --input-dir <out> --pip-bed <CS bed> --out vep_input.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config as C
from s4lib import log, peek_header, variant_key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--pip-bed", type=Path, default=C.PIP_BED)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    V = pd.read_parquet(args.input_dir / "variants.parquet")

    header = peek_header(args.pip_bed)
    # chr is read as a string: its values are bare ("1", "X"), so chunked type
    # inference otherwise reports a mixed-type column
    cs = pd.read_csv(args.pip_bed, sep="\t", usecols=["chr", "start"],
                     dtype={"chr": str}, low_memory=False)
    cs["chr"] = cs["chr"].astype(str)
    if not cs["chr"].str.startswith("chr").all():
        cs["chr"] = "chr" + cs["chr"].str.removeprefix("chr")
    cs["pos"] = cs["start"] + (1 if C.PIP_BED_IS_ZERO_BASED else 0)
    sites = cs[["chr", "pos"]].drop_duplicates()

    v = V.copy()
    v["chr"] = v["chr"].astype(str)
    v = v.merge(sites, on=["chr", "pos"], how="inner")
    assert len(v), ("no variant matched the credible-set file; check the coordinate "
                    f"convention (PIP_BED_IS_ZERO_BASED = {C.PIP_BED_IS_ZERO_BASED}) "
                    f"and chromosome naming. Credible-set header: {header}")

    ref, alt = v["allele1"].astype(str), v["allele2"].astype(str)
    is_snv = (ref.str.len() == 1) & (alt.str.len() == 1)

    # VEP default format: 1-based, start/end span the REFERENCE bases affected.
    start, end = v["pos"].astype(int).copy(), v["pos"].astype(int).copy()
    alleles = ref + "/" + alt
    # deletion: reference bases after the anchor are removed
    dele = ref.str.len() > alt.str.len()
    start.loc[dele] = v.loc[dele, "pos"].astype(int) + 1
    end.loc[dele] = v.loc[dele, "pos"].astype(int) + ref[dele].str.len() - 1
    alleles.loc[dele] = ref[dele].str[1:] + "/-"
    # insertion: zero-length reference interval, so end = start - 1
    ins = alt.str.len() > ref.str.len()
    start.loc[ins] = v.loc[ins, "pos"].astype(int) + 1
    end.loc[ins] = v.loc[ins, "pos"].astype(int)
    alleles.loc[ins] = "-/" + alt[ins].str[1:]

    # the same helper 07_quadrants.py uses, so the two keys cannot drift apart
    key = variant_key(v, C.VARIANT_KEY_COLUMNS)
    v = v.assign(_key=key)
    v, n_dropped = C.collapse_duplicate_variants(v)
    ref, alt = v["allele1"].astype(str), v["allele2"].astype(str)
    is_snv = (ref.str.len() == 1) & (alt.str.len() == 1)
    start, end, alleles = start.loc[v.index], end.loc[v.index], alleles.loc[v.index]
    key = v["_key"]

    out = pd.DataFrame({
        "chrom": v["chr"].str.removeprefix("chr"),
        "start": start, "end": end, "allele": alleles, "strand": "+", "id": key,
    })
    assert out["id"].is_unique, "variant_key is not unique after collapsing"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", header=False, index=False)

    log(f"scored variants                 {len(V):,}")
    log(f"in the credible-set file        {len(v):,}")
    log(f"  collapsed as duplicate keys   {n_dropped:,}")
    log(f"  SNVs                          {int(is_snv.sum()):,}")
    log(f"  deletions                     {int(dele.sum()):,}")
    log(f"  insertions                    {int(ins.sum()):,}")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
