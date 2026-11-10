# Data

The analyses use public data only. Nothing here is controlled-access. Large files are
not stored in the repository; place them as below (or point the scripts at other
locations with `CBPNET_DATA_DIR` and the `--data-dir` / `--input-dir` arguments).

## Deposited on Zenodo (DOI: 10.5281/zenodo.24091438)

| Path | Contents |
|---|---|
| `variant_scores/<cell_type>/` | ChromBPNet variant scores per fold (observed and shuffled), 17 cell types x 5 folds, plus `consensus_peaks_blood17_10col.bed` |
| `crispri/ALL7_trimmed.csv` | Curated table of K562 CRISPRi elements used for the benchmark |
| `crispri/chrombpnet_k562_scores/` | K562 ChromBPNet scores for the HapMap/Vuckovic variants (`HapMap_GWAS_K562_snp_scores.tsv`) and the eQTL Catalogue variants (`eQTL_Catalogue_K562/`) |
| `crispri/K562_ATAC_peaks.narrowPeak` | K562 ATAC-seq peaks |
| trained ChromBPNet models | one set per cell type and fold |

## Public sources

| Path | Source |
|---|---|
| ATAC-seq (training data) | GEO GSE74912 (Corces et al. 2016), GSE96772 and GSE213295 |
| `external/CS.ukid.ukbb_v2.PP0.001_hg38.bed` | Vuckovic et al. fine-mapped credible sets (PP >= 0.001), lifted to hg38 |
| `external/gencode.v47.basic.annotation.gtf` | GENCODE v47 basic annotation |
| `external/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz` | gnomAD v2.1.1 gene-level LoF constraint |
| `external/GRCh38-cCREs.bed` | ENCODE cCRE registry V4 (GRCh38) |
| `external/alphagenome_variant_impact_score_snvs.tsv.gz` (+ `.tbi`) | AlphaGenome Atlas Variant Impact SNV score table, bgzipped and tabix-indexed |
| `external/ENCFF736NYC.bed.gz` | K562 CTCF ChIP-seq IDR peaks (ENCODE experiment ENCSR000DWE) |
| `external/hg38.fa` (+ `.fai`) | hg38 reference genome |
