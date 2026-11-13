# Chromatin accessibility predictions reveal lineage-structured effects of noncoding variants across hematopoiesis

Code for *"Chromatin accessibility predictions reveal lineage-structured effects of noncoding variants across hematopoiesis"* (Kazemi and Morris, Genome Biology, submitted).

The analyses use ChromBPNet models trained on ATAC-seq from 17 haematopoietic cell types
(5 folds each) to score fine-mapped and HapMap variants, then relate the predictions to
cellular breadth, the lineage hierarchy, fine-mapping posteriors, variant consequence,
ENCODE cCREs and CRISPRi results in K562.

## Repository layout

```
variant_analysis/   variant scores -> breadth, lineage coherence, annotation, quadrants, cCREs
crispri/            K562 CRISPRi benchmark and motif analysis of CRISPRi elements
data/               input locations (see data/README.md)
```

## variant_analysis/

| Script | Results analysis it reproduces |
|---|---|
| `config.py`, `s4lib.py`, `s4plot.py` | Helper modules imported by every step; nothing runs without them |
| `02_build_consensus_peaks.py` | Consensus peak set across the 17 cell types, the testable universe for variant scoring |
| `average_folds.py` | Fold-averaged score files per cell type (used to cross-check the fold-averaged statistics) |
| `01_build_matrix.py` | Combines the fold-averaged variant scores and computes the empirical P values and active calls |
| `02_nominate.py` | Nominated, flagship and model-only tiers, per-cell-type called rates and the threshold sensitivity |
| `score_all_variants.py` | Minimum empirical P and maximum \|logFC\| for every scored variant, including those outside all peaks |
| `03_breadth.py` | Breadth vs regulatory magnitude (group means, OLS/Welch contrasts, ρ estimators and permutation nulls) |
| `04_hierarchy.py` | Lineage coherence, the pair-level distance model and sign discordance |
| `05_annotate.py` | The 88 high-confidence variants and the active-variant table (Tables S1-S2) |
| `vep_prepare.py`, `vep_collapse.py` | VEP input and output for the consequence analysis |
| `07_quadrants.py` | Fine-mapping vs model: the four classes, VEP enrichment, active-allele quantile, trait-lineage concordance and the PIP gradient |
| `ccre_by_breadth.py` | ENCODE cCRE annotation by breadth |
| `avi_breadth_replication.py` | Breadth trend replicated with AlphaGenome Variant Impact scores |
| `run_all.sh` | Runs the steps above in order, then the CRISPRi benchmark |

## crispri/

| Script | Results analysis it reproduces |
|---|---|
| `crispri_logistic_regression.py` | ChromBPNet scores of variants inside K562 CRISPRi elements vs target-gene status (primary analysis and specification sweep) |

### crispri/motif_analysis/

Motif analysis of the K562 CRISPRi elements. Run in this order:

| Step | Script | What it does |
|---|---|---|
| 1 | `make_crispri_input_bed.py` | Turns the K562 rows of `TableS4.csv` into the BED that gets scored. It defines which 9,205 elements enter the analysis. |
| 2 | `fetch_variant_scorer.sh` | Fetches variant-scorer, which supplies the DeepSHAP background and combine functions the Methods cite. Pass a commit hash to pin the version. |
| 3 | `chrombpnet_bed_counts_contribs.py` | Computes the predicted counts and the counts-head DeepSHAP contribution scores (one SLURM array shard per job). It also writes the `pred_log_counts` that become the model's own accessibility covariate. |
| 4 | `build_finemo_inputs.py` | Converts the shards into Fi-NeMo inputs and crops each element to 1,000 bp. |
| 5 | `build_region_map.py` | Maps each `region_N` ID back to its element and its predicted accessibility. Run it on the same shard directory as `build_finemo_inputs.py`, because both assign IDs in the same sorted shard order. |
| 6 | `annotate_hits.py` | Attaches TomTom labels and q-values to the Fi-NeMo hits. |
| 7 | `build_element_motif_features.py` | Applies the in-boundary and confidence filters and builds the per-element feature tables. |
| 8 | `run_stats.py` | Runs the univariate tests, the Benjamini-Hochberg correction and the nested multivariate models. |
| 9 | `bootstrap_ci.py` | Produces the bootstrap intervals for CTCF and NFE2/NRF2. |
| | `ctcf_chipseq_sanity.py` | Checks the CTCF motif calls against K562 CTCF ChIP-seq. |

Motif discovery (TF-MoDISco-lite), TomTom matching and `finemo call-hits` are run with
their own command-line tools between steps 5 and 6; see the Fi-NeMo and TF-MoDISco-lite
documentation.

## Model training and variant scoring

ChromBPNet models were trained and variants were scored with the default configuration
from the official [ChromBPNet](https://github.com/kundajelab/chrombpnet) instructions
(bias model, ChromBPNet model training, and variant scoring with variant-scorer), run as
SLURM jobs on GPUs. Details are in the Methods of the paper. No custom training or scoring
code is needed; the trained models and score tables are on Zenodo (below).

## Data and models

Inputs are described in [data/README.md](data/README.md). Trained models and variant
scores are on Zenodo: **DOI 10.5281/zenodo.24091438**.
All underlying ATAC-seq and CRISPRi data are public.

## Running

```
pip install -r requirements.txt
bash variant_analysis/run_all.sh
```

Paths default to `data/` and `results/` inside the repository. Override them with
`DATA=... OUT=... bash variant_analysis/run_all.sh`, or with the `CBPNET_*` environment
variables read by `variant_analysis/config.py`. `python variant_analysis/config.py`
validates the configuration and lists any missing input files. Step 5 of `run_all.sh`
needs Ensembl VEP 115.2 with the release-113 cache; the CRISPRi steps need bedtools.

## Citation

Kazemi M, Morris JA. *Chromatin accessibility predictions reveal lineage-structured effects of noncoding variants across hematopoiesis.* https://doi.org/10.64898/2026.11.12.766027

## License

MIT, see [LICENSE](LICENSE).
