#!/usr/bin/env bash
# run_all.sh: runs the variant-analysis scripts in order, then the CRISPRi
# logistic-regression benchmark.
#
# Inputs (see data/README.md): ChromBPNet variant scores (17 cell types x 5 folds,
# observed and shuffled), the consensus peak set, the Vuckovic et al. PP >= 0.001
# credible-set file, GENCODE v47, gnomAD v2.1.1 constraint, the ENCODE cCRE registry
# (V4) and the curated K562 CRISPRi elements.
#
# Usage, from the repository root:
#     bash variant_analysis/run_all.sh
# Override DATA / OUT to use other locations.
#
# Permutation counts used for the reported results:
#   03  breadth-magnitude rho nulls             10,000 permutations (--permute --n-perm)
#       threshold-ladder nulls                  500 per rung (fixed in 03)
#   04  curveball null, variant-level coherence 10,000 (--n-perm)
#       pair-level relabelling null             5,000 primary, 10,000 panel-wide (fixed in 04)
#   07  trait-lineage concordance null          10,000 (--n-perm)
#   ccre_by_breadth  global-test permutation    2,000 (fixed)
#
# Requirements: see requirements.txt; Ensembl VEP 115.2 with the release-113 cache;
# bedtools (CRISPRi step).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:-$REPO/data}"
OUT="${OUT:-$REPO/results/variant_analysis}"

SCORES="$DATA/variant_scores"
PIP="$DATA/external/CS.ukid.ukbb_v2.PP0.001_hg38.bed"
GTF="$DATA/external/gencode.v47.basic.annotation.gtf"
CONSTRAINT="$DATA/external/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz"
CCRE="$DATA/external/GRCh38-cCREs.bed"
VEP="${VEP:-vep}"
VEP_CACHE="${VEP_CACHE:-$HOME/.vep}"
VEP_CACHE_VERSION=113

SCRIPTS="$REPO/variant_analysis"
export PYTHONPATH="$SCRIPTS${PYTHONPATH:+:$PYTHONPATH}"     # config.py, s4lib.py, s4plot.py
mkdir -p "$OUT"

# 0. Before 01 (run once, inputs described in the script docstrings):
#    02_build_consensus_peaks.py builds consensus_peaks_blood17_10col.bed from the MACS2
#    summits; average_folds.py (optional) writes fold-averaged score files used only to
#    cross-check the fold-averaged statistics.

# 1. variant x cell-type matrix: fold-averaged scores, empirical P, calls
python "$SCRIPTS/01_build_matrix.py" \
  --score-dir       "$SCORES" \
  --consensus-peaks "$SCORES/consensus_peaks_blood17_10col.bed" \
  --pip-bed         "$PIP" \
  --gencode-gtf     "$GTF" \
  --constraint      "$CONSTRAINT" \
  --output-dir      "$OUT"

# 1b. nomination tiers and threshold sensitivity
python "$SCRIPTS/02_nominate.py" --input-dir "$OUT"

# 1c. empirical P for every scored variant, including those outside all peaks
python "$SCRIPTS/score_all_variants.py" --score-dir "$SCORES" --input-dir "$OUT"

# 2. breadth against regulatory magnitude
python "$SCRIPTS/03_breadth.py" --input-dir "$OUT" --permute --n-perm 10000

# 3. lineage coherence, pair-level model, direction of effect
python "$SCRIPTS/04_hierarchy.py" --input-dir "$OUT" --n-perm 10000

# 4. annotated variant tables and PIP validation
python "$SCRIPTS/05_annotate.py" --input-dir "$OUT"

# 5. Ensembl VEP annotation
VEPDIR="$OUT/vep"
mkdir -p "$VEPDIR"
python "$SCRIPTS/vep_prepare.py" --input-dir "$OUT" --pip-bed "$PIP" --out "$VEPDIR/vep_input.txt"
"$VEP" \
  --offline --cache --dir_cache "$VEP_CACHE" --cache_version "$VEP_CACHE_VERSION" \
  --species homo_sapiens --assembly GRCh38 \
  --format ensembl --tab --no_stats --force_overwrite --fork 8 \
  --regulatory \
  --fields "Uploaded_variation,Location,Allele,Consequence,IMPACT,SYMBOL,Gene,BIOTYPE" \
  --input_file "$VEPDIR/vep_input.txt" --output_file "$VEPDIR/vep_out.txt"
python "$SCRIPTS/vep_collapse.py" --vep-output "$VEPDIR/vep_out.txt" --out-dir "$VEPDIR"

# 6. fine-mapping vs model: classes, consequence enrichment, AAQ, concordance
python "$SCRIPTS/07_quadrants.py" \
  --input-dir "$OUT" --pip-bed "$PIP" \
  --vep "$VEPDIR/vep_consequences.tsv" --n-perm 10000

# 7. ENCODE cCRE annotation by breadth
python "$SCRIPTS/ccre_by_breadth.py" \
  --input-dir "$OUT" --ccre-bed "$CCRE" --out-dir "$OUT/ccre"

# 8. CRISPRi benchmark in K562
python "$REPO/crispri/crispri_logistic_regression.py" \
  --data-dir "$DATA" --out-dir "$REPO/results/crispri_logistic_regression"

echo "done: tables in $OUT/tables, cCRE results in $OUT/ccre"
