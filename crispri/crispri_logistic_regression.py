#!/usr/bin/env python3
"""
ChromBPNet vs CRISPRi logistic regression, signed aggregation

For each K562 CRISPRi element (cCRE), the K562 ChromBPNet predictions of the fine-mapped
variants inside it are summarised, and target-gene status is regressed on that summary:

    has_target_gene ~ element score + log10(|distance to nearest protein-coding TSS| + 1)

Element summaries (signed log_counts_diff, among variants with |lcd| >= threshold):
  max_signed    : max(lcd)  -> the single strongest accessibility-increasing variant
  mean_signed   : mean(lcd) -> net directional effect per variant
  frac_positive : n(lcd > 0) / n_variants -> direction only
Elements with no variant inside them are excluded from a given fit.

PRIMARY ANALYSIS (see PRIMARY below)
  pooled variant set (Vuckovic GWAS + eQTL Catalogue, PIP >= 0.05, deduplicated keeping
  the highest posterior), no effect-size threshold, no PIP filter, no variant or element
  accessibility filter, elements > 10 kb from the nearest protein-coding TSS,
  max_signed aggregation.

SPECIFICATION SWEEP (robustness): every combination of
  variant set {vuckovic, eqtl, pooled} x variant ATAC filter {no, yes} x PIP filter
  {none, 0.5} x |lcd| threshold {0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5} x element
  ATAC filter {no, yes} x TSS boundary {1, 2, 5, 10 kb} x aggregation (3).
  Fits with a 95% CI wider than 8 log-OR units (separation) are flagged ci_sane = False.

Usage
  python crispri_logistic_regression.py --data-dir data --out-dir results/crispri_logistic_regression

Outputs (--out-dir)
  all_results.csv, significant_results.csv, sane_results.csv   every fitted specification
  primary_specification.csv     the pre-specified analysis: three aggregations, plus the
                                two variant sets analysed separately
  sweep_summary.csv             per aggregation over interpretable fits: n, fraction with
                                OR > 1, median OR; and the median strongest-variant OR by
                                TSS boundary
"""

import argparse
import pandas as pd
import numpy as np
import glob, os, subprocess, tempfile, warnings
from pathlib import Path
from itertools import product
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('--data-dir', default='data', help='data directory (see data/README.md)')
_ap.add_argument('--out-dir', default='results/crispri_logistic_regression')
_args = _ap.parse_args()

DATA_DIR        = Path(_args.data_dir)
CRISPRI_FILE    = DATA_DIR / 'crispri' / 'ALL7_trimmed.csv'
VUCKOVIC_SCORES = DATA_DIR / 'crispri' / 'chrombpnet_k562_scores' / 'HapMap_GWAS_K562_snp_scores.tsv'
PP_FILE         = DATA_DIR / 'external' / 'CS.ukid.ukbb_v2.PP0.001_hg38.bed'
EQTL_SCORES_DIR = DATA_DIR / 'crispri' / 'chrombpnet_k562_scores' / 'eQTL_Catalogue_K562'
ATAC_PEAKS      = DATA_DIR / 'crispri' / 'K562_ATAC_peaks.narrowPeak'
OUTPUT_DIR      = Path(_args.out_dir)
CACHE_DIR       = OUTPUT_DIR / 'cache'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PARAMETERS
# =============================================================================
SCORE_COL      = 'log_counts_diff'
ABS_THRESHOLDS = [0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
AGG_METHODS    = ['max_signed', 'mean_signed', 'frac_positive']
TSS_CUTOFFS    = [1000, 2000, 5000, 10000]
CRE_ATAC_OPTS  = [False, True]
CRE_ATAC_THRESH = 0.2
PP_OPTS        = [None, 0.5]
VAR_ATAC_OPTS  = [False, True]
DATASETS       = ['vuckovic', 'eqtl', 'pooled']
TSS_COL        = 'closest_TSS_distance_Protein_coding_hg38'

SEP = '=' * 70
def log(msg): print(msg, flush=True)

# =============================================================================
# STEP 1: CRISPRi
# =============================================================================
log(f'\n{SEP}'); log('STEP 1: CRISPRi'); log(SEP)

df_cre = pd.read_csv(CRISPRI_FILE)
df_cre = df_cre[df_cre['Cell_type'] == 'K562'].copy()
for col in ['CRE_start_hg38', 'CRE_end_hg38', TSS_COL, 'ATAC_peak_overlap_hg38']:
    df_cre[col] = pd.to_numeric(df_cre[col], errors='coerce')
df_cre = df_cre.dropna(subset=['CRE_start_hg38', 'CRE_end_hg38', TSS_COL]).reset_index(drop=True)
df_cre['has_target_gene'] = df_cre['has_target_gene'].astype(bool)
log(f'  K562 CREs: {len(df_cre)}  |  has_target_gene True={df_cre["has_target_gene"].sum()}')

# =============================================================================
# STEP 2: Vuckovic variants + PP
# =============================================================================
log(f'\n{SEP}'); log('STEP 2: Vuckovic variants + PP'); log(SEP)

vuc_cache = f'{CACHE_DIR}/vuckovic_variants_with_pp.parquet'
if os.path.exists(vuc_cache):
    df_vuc = pd.read_parquet(vuc_cache)
    log(f'  Loaded from cache: {len(df_vuc):,} variants')
else:
    df_scores = pd.read_csv(VUCKOVIC_SCORES, sep='\t', dtype={'CHR': str})
    df_vuc    = df_scores[df_scores['META_DATA'].isin(['Vuckovic', 'HapMap,Vuckovic'])].copy()
    df_pp     = pd.read_csv(PP_FILE, sep='\t', low_memory=False)
    df_pp['chr_norm'] = df_pp['chr'].astype(str).apply(
        lambda x: f'chr{x}' if not str(x).startswith('chr') else x)
    df_pp_max = df_pp.groupby(['chr_norm', 'start'])['PP'].max().reset_index()
    df_pp_max.columns = ['CHR', 'pos_1based', 'PP']
    df_vuc['pos_1based'] = df_vuc['POS0'] + 1
    df_vuc = df_vuc.merge(df_pp_max, on=['CHR', 'pos_1based'], how='inner')
    df_vuc = df_vuc.drop(columns=['pos_1based']).reset_index(drop=True)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    df_vuc.to_parquet(vuc_cache, index=False)
    log(f'  Vuckovic variants with PP: {len(df_vuc):,}')

df_vuc['abs_lcd'] = df_vuc[SCORE_COL].abs()

# =============================================================================
# STEP 3: eQTL variants
# =============================================================================
log(f'\n{SEP}'); log('STEP 3: eQTL variants'); log(SEP)

eqtl_cache = f'{CACHE_DIR}/eqtl_variants_deduplicated.parquet'
if os.path.exists(eqtl_cache):
    df_eqtl = pd.read_parquet(eqtl_cache)
    log(f'  Loaded from cache: {len(df_eqtl):,} variants')
else:
    chunks = sorted(glob.glob(f'{EQTL_SCORES_DIR}/*.tsv'))
    log(f'  Loading {len(chunks)} chunks...')
    dfs  = []
    keep = ['CHR','POS0','REF','ALT','META_DATA',
            'log_counts_diff','log_probs_diff_abs_sum','probs_jsd_diff']
    for i, p in enumerate(chunks):
        if (i+1) % 50 == 0: log(f'    {i+1}/{len(chunks)}...')
        df_c = pd.read_csv(p, sep='\t', dtype={'CHR': str}, usecols=keep)
        df_c['PP'] = pd.to_numeric(df_c['META_DATA'].str.split('|').str[1], errors='coerce')
        dfs.append(df_c.drop(columns=['META_DATA']))
    df_all  = pd.concat(dfs, ignore_index=True).sort_values('PP', ascending=False)
    df_eqtl = df_all.drop_duplicates(subset=['CHR','POS0','REF','ALT']).reset_index(drop=True)
    df_eqtl.to_parquet(eqtl_cache, index=False)
    log(f'  Unique eQTL variants: {len(df_eqtl):,}')

df_eqtl['abs_lcd'] = df_eqtl[SCORE_COL].abs()

# =============================================================================
# STEP 4: ATAC masks
# =============================================================================
log(f'\n{SEP}'); log('STEP 4: ATAC variant masks'); log(SEP)

def atac_filter_variants(df_variants, label):
    cache_path = f'{CACHE_DIR}/atac_mask_{label}.npy'
    if os.path.exists(cache_path):
        mask = np.load(cache_path)
        log(f'  [{label}] cache: {mask.sum():,}/{len(mask):,} ({100*mask.mean():.1f}%)')
        return mask
    log(f'  [{label}] bedtools intersect ({len(df_variants):,} variants)...')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False, dir='/tmp') as f:
        bed = f.name
        for i, (_, row) in enumerate(df_variants.iterrows()):
            f.write(f"{row['CHR']}\t{int(row['POS0'])}\t{int(row['POS0'])+1}\t{i}\n")
    res = subprocess.run(['bedtools','intersect','-a',bed,'-b',ATAC_PEAKS,'-u'],
                         capture_output=True, text=True)
    Path(bed).unlink()
    passing = {int(l.split('\t')[3]) for l in res.stdout.strip().split('\n') if l}
    mask = np.array([i in passing for i in range(len(df_variants))])
    np.save(cache_path, mask)
    log(f'  [{label}] {mask.sum():,}/{len(mask):,} ({100*mask.mean():.1f}%)')
    return mask

df_vuc['in_atac']  = atac_filter_variants(df_vuc,  'vuckovic')
df_eqtl['in_atac'] = atac_filter_variants(df_eqtl, 'eqtl')

# =============================================================================
# STEP 5: Pooled
# =============================================================================
log(f'\n{SEP}'); log('STEP 5: Pooled dataset'); log(SEP)

df_pooled = pd.concat([df_vuc, df_eqtl], ignore_index=True)
df_pooled = df_pooled.sort_values('PP', ascending=False)
df_pooled = df_pooled.drop_duplicates(subset=['CHR','POS0','REF','ALT']).reset_index(drop=True)
df_pooled['abs_lcd'] = df_pooled[SCORE_COL].abs()
log(f'  Pooled (deduplicated): {len(df_pooled):,}')

dataset_map = {'vuckovic': df_vuc, 'eqtl': df_eqtl, 'pooled': df_pooled}

# =============================================================================
# Aggregation
#
# For each CRE, collect variants with |lcd| >= threshold, then compute:
#   max_signed    = max(lcd values)        , best single opening effect
#   mean_signed   = mean(lcd values)       , net directional effect per variant
#   frac_positive = n(lcd > 0) / n_total   , proportion opening
#
# All three are length-bias-free (max/mean per variant, fraction).
# CRE coords: 1-based inclusive. Variant POS0: 0-based → 1-based = POS0+1.
# =============================================================================

def aggregate_per_cre(df_cres, df_variants, threshold):
    """
    Returns DataFrame (same length/order as df_cres, reset_index'd).
    Columns: max_signed, mean_signed, frac_positive, n_variants
    n_variants = number of variants with |lcd| >= threshold overlapping the CRE.
    """
    n   = len(df_cres)
    out = {
        'max_signed':    np.full(n, np.nan),
        'mean_signed':   np.full(n, np.nan),
        'frac_positive': np.full(n, np.nan),
        'n_variants':    np.zeros(n, dtype=int),
    }

    # Pre-build per-chromosome sorted arrays
    # Apply threshold filter here so searchsorted is on the right subset
    df_thresh = df_variants[df_variants['abs_lcd'] >= threshold].copy()

    var_by_chr = {}
    for chrom, grp in df_thresh.groupby('CHR'):
        pos1  = grp['POS0'].values + 1           # 1-based
        order = np.argsort(pos1)
        var_by_chr[chrom] = {
            'pos1': pos1[order],
            'lcd':  grp[SCORE_COL].values[order],
        }

    for i, row in df_cres.iterrows():
        chrom = row['chr']
        if chrom not in var_by_chr:
            continue
        vd    = var_by_chr[chrom]
        lo    = int(row['CRE_start_hg38'])    # 1-based inclusive
        hi    = int(row['CRE_end_hg38'])      # 1-based inclusive
        l     = np.searchsorted(vd['pos1'], lo,    side='left')
        r     = np.searchsorted(vd['pos1'], hi+1,  side='left')

        if r > l:
            vals = vd['lcd'][l:r]
            n_var = r - l
            out['n_variants'][i]    = n_var
            out['max_signed'][i]    = float(np.max(vals))
            out['mean_signed'][i]   = float(np.mean(vals))
            out['frac_positive'][i] = float(np.sum(vals > 0) / n_var)

    return pd.DataFrame(out)


# =============================================================================
# Logistic regression
# conf_int() returns numpy array in statsmodels >= 0.14
# =============================================================================

def run_logit(y, x_score, x_tss_log):
    """
    Fit: has_target_gene ~ x_score + log10(|TSS_dist| + 1)
    x_score may contain NaN (CREs with no variants after threshold), these
    are already excluded upstream via has_var mask, so NaN here is unexpected
    but we guard anyway.
    """
    n, n_pos = len(y), int(y.sum())
    if n < 10 or n_pos < 2 or (n - n_pos) < 2:
        return None
    if np.std(x_score) == 0:
        return None

    X     = np.column_stack([x_score, x_tss_log])
    valid = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
    X, y  = X[valid], y[valid]
    if len(y) < 10 or y.sum() < 2 or (len(y) - y.sum()) < 2:
        return None

    try:
        X_sm   = sm.add_constant(X, has_constant='add')
        fitted = sm.Logit(y, X_sm).fit(disp=0, maxiter=500, method='bfgs')
        coef   = float(fitted.params[1])
        ci     = np.asarray(fitted.conf_int())   # numpy array in statsmodels >= 0.14
        return {
            'coef':        coef,
            'se':          float(fitted.bse[1]),
            'OR':          float(np.exp(coef)),
            'OR_CI_lower': float(np.exp(ci[1, 0])),
            'OR_CI_upper': float(np.exp(ci[1, 1])),
            'p_value':     float(fitted.pvalues[1]),
            'n_cres':      int(len(y)),
            'n_positive':  int(y.sum()),
        }
    except Exception:
        return None


# =============================================================================
# STEP 6: Run all combinations
# =============================================================================
log(f'\n{SEP}'); log('STEP 6: Running all combinations'); log(SEP)

total = (len(DATASETS) * len(VAR_ATAC_OPTS) * len(PP_OPTS) *
         len(ABS_THRESHOLDS) * len(CRE_ATAC_OPTS) * len(TSS_CUTOFFS) * len(AGG_METHODS))
log(f'  Total combinations: {total}')

all_results = []
skip_novar  = 0
skip_logit  = 0

for dataset_name in DATASETS:
    df_var_base = dataset_map[dataset_name].copy().reset_index(drop=True)

    for var_atac, pp_t, abs_thresh in product(VAR_ATAC_OPTS, PP_OPTS, ABS_THRESHOLDS):

        # --- variant-level filters (PP and ATAC only; threshold applied inside agg) ---
        mask = pd.Series(True, index=df_var_base.index)
        if var_atac:
            mask &= df_var_base['in_atac'].astype(bool)
        if pp_t is not None:
            mask &= (df_var_base['PP'] >= pp_t)

        df_var = df_var_base[mask].copy().reset_index(drop=True)
        if len(df_var) < 5:
            continue

        for cre_atac, tss_cut in product(CRE_ATAC_OPTS, TSS_CUTOFFS):

            # --- CRE-level filters ---
            cm = np.abs(df_cre[TSS_COL]) > tss_cut
            if cre_atac:
                cm &= (df_cre['ATAC_peak_overlap_hg38'].fillna(0) >= CRE_ATAC_THRESH)
            df_cf = df_cre[cm].copy().reset_index(drop=True)
            if len(df_cf) < 10:
                continue

            # --- aggregate (threshold applied inside) ---
            agg     = aggregate_per_cre(df_cf, df_var, abs_thresh)
            has_var = agg['n_variants'] > 0

            if has_var.sum() == 0:
                skip_novar += 1
                continue

            agg_sub    = agg[has_var].reset_index(drop=True)
            df_cre_sub = df_cf[has_var].reset_index(drop=True)
            y          = df_cre_sub['has_target_gene'].values.astype(int)
            x_tss_log  = np.log10(np.abs(df_cre_sub[TSS_COL].values) + 1)

            for agg_method in AGG_METHODS:
                x_score = agg_sub[agg_method].values
                res     = run_logit(y, x_score, x_tss_log)

                if res is None:
                    skip_logit += 1
                    continue

                all_results.append({
                    'dataset':         dataset_name,
                    'agg_method':      agg_method,
                    'abs_threshold':   abs_thresh,
                    'var_pp_filter':   pp_t if pp_t is not None else 'none',
                    'var_atac_filter': var_atac,
                    'cre_atac_filter': cre_atac,
                    'tss_cutoff_bp':   tss_cut,
                    'n_cres_total':    len(df_cf),
                    'n_cres_with_vars': int(has_var.sum()),
                    **res,
                })

    log(f'  {dataset_name} done, results so far: {len(all_results)}')

log(f'\n  Skipped (no variant-CRE overlap): {skip_novar}')
log(f'  Skipped (logit failed):            {skip_logit}')

# =============================================================================
# STEP 7: Save
# =============================================================================
log(f'\n{SEP}'); log('STEP 7: Save & report'); log(SEP)

if len(all_results) == 0:
    log('ERROR: No results.'); import sys; sys.exit(1)

df_res = pd.DataFrame(all_results)

# CI sanity filter: remove complete-separation outliers (CI width > 8 log-OR units)
df_res['log_CI_width'] = (np.log(df_res['OR_CI_upper'].clip(upper=1e10)) -
                          np.log(df_res['OR_CI_lower'].clip(lower=1e-10)))
df_res['ci_sane']      = df_res['log_CI_width'] < 8
df_res = df_res.sort_values('OR', ascending=False).reset_index(drop=True)

df_res.to_csv(f'{OUTPUT_DIR}/all_results.csv', index=False)
df_sig  = df_res[df_res['p_value'] < 0.05].copy()
df_sig.to_csv(f'{OUTPUT_DIR}/significant_results.csv', index=False)
df_sane = df_res[df_res['ci_sane']].copy()
df_sane.to_csv(f'{OUTPUT_DIR}/sane_results.csv', index=False)

log(f'  Total regressions:    {len(df_res)}')
log(f'  Significant (p<0.05): {len(df_sig)}')
log(f'  Sane CI:              {len(df_sane)}')

# =============================================================================
# TOP 30 (sane CI) by OR
# =============================================================================
log(f'\n{SEP}'); log('TOP 30 (sane CI) BY ODDS RATIO'); log(SEP)

cols = ['dataset','agg_method','abs_threshold','var_pp_filter','var_atac_filter',
        'cre_atac_filter','tss_cutoff_bp','n_cres_with_vars',
        'OR','OR_CI_lower','OR_CI_upper','p_value']
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 240)
pd.set_option('display.float_format', '{:.4f}'.format)
print(df_sane.head(30)[cols].to_string(index=True))

# =============================================================================
# BEST PER dataset × agg_method
# =============================================================================
log(f'\n{SEP}'); log('BEST (sane CI) PER DATASET × AGG METHOD'); log(SEP)

for ds in DATASETS:
    for am in AGG_METHODS:
        sub = df_sane[(df_sane['dataset']==ds) & (df_sane['agg_method']==am)]
        if len(sub) == 0: continue
        b = sub.iloc[0]
        log(f'\n  {ds} | {am}')
        log(f'    OR={b["OR"]:.4f}  [{b["OR_CI_lower"]:.4f}, {b["OR_CI_upper"]:.4f}]  '
            f'p={b["p_value"]:.2e}  N={b["n_cres_with_vars"]}')
        log(f'    |lcd|>={b["abs_threshold"]}  PP={b["var_pp_filter"]}  '
            f'VarATAC={b["var_atac_filter"]}  CREATAC={b["cre_atac_filter"]}  '
            f'TSS>{b["tss_cutoff_bp"]}bp')

# =============================================================================
# MARGINAL: median OR per filter level (sane CI)
# =============================================================================
log(f'\n{SEP}'); log('MEDIAN OR (sane CI) BY INDIVIDUAL FILTER'); log(SEP)

for col, label in [
    ('dataset',          'Dataset'),
    ('agg_method',       'Aggregation method'),
    ('abs_threshold',    '|lcd| threshold'),
    ('var_pp_filter',    'PP filter'),
    ('var_atac_filter',  'Var ATAC filter'),
    ('cre_atac_filter',  'CRE ATAC filter'),
    ('tss_cutoff_bp',    'TSS cutoff (bp)'),
]:
    log(f'\n  {label}:')
    g = df_sane.groupby(col)['OR'].agg(['median','mean','count'])
    g.columns = ['median_OR', 'mean_OR', 'n']
    log(g.to_string())

# =============================================================================
# PRIMARY ANALYSIS
# =============================================================================
PRIMARY = dict(dataset='pooled', abs_threshold=0, var_pp_filter='none',
               var_atac_filter=False, cre_atac_filter=False, tss_cutoff_bp=10000)

def select(df, **spec):
    """Rows matching every key; numbers compared as numbers, flags and labels as text."""
    m = pd.Series(True, index=df.index)
    for k, v in spec.items():
        if isinstance(v, bool) or isinstance(v, str):
            m &= df[k].astype(str).str.lower() == str(v).lower()
        else:
            m &= pd.to_numeric(df[k], errors='coerce') == v
    return df[m]

log(f'\n{SEP}'); log('PRE-SPECIFIED PRIMARY ANALYSIS'); log(SEP)
prim_rows = []
for am in AGG_METHODS:                                       # the three aggregations
    r = select(df_res, **PRIMARY, agg_method=am)
    if len(r): prim_rows.append(r.iloc[0].to_dict() | {'role': 'primary set'})
for ds in ('vuckovic', 'eqtl'):                              # each variant set alone
    r = select(df_res, **(PRIMARY | {'dataset': ds}), agg_method='max_signed')
    if len(r): prim_rows.append(r.iloc[0].to_dict() | {'role': 'variant set alone'})
df_prim = pd.DataFrame(prim_rows)
df_prim.to_csv(f'{OUTPUT_DIR}/primary_specification.csv', index=False)
for r in df_prim.itertuples():
    log(f'  {r.dataset:<9s} {r.agg_method:<14s} OR {r.OR:.2f} [{r.OR_CI_lower:.2f}, {r.OR_CI_upper:.2f}]  '
        f'p = {r.p_value:.3g}  n = {r.n_cres_with_vars:,} distal elements, {r.n_positive:,} positive')
log('  strongest-variant OR by TSS boundary (primary set): ' + ', '.join(
    f">{c // 1000} kb {select(df_res, **(PRIMARY | {'tss_cutoff_bp': c}), agg_method='max_signed')['OR'].iloc[0]:.2f}"
    for c in TSS_CUTOFFS
    if len(select(df_res, **(PRIMARY | {'tss_cutoff_bp': c}), agg_method='max_signed'))))

# =============================================================================
# SWEEP SUMMARY over interpretable fits (ci_sane)
# =============================================================================
log(f'\n{SEP}'); log(f'SPECIFICATION SWEEP: {len(df_sane):,} interpretable fits of {len(df_res):,}'); log(SEP)
summ = []
for am in AGG_METHODS:
    sub = df_sane[df_sane['agg_method'] == am]
    summ.append(dict(summary='by aggregation', agg_method=am, tss_cutoff_bp='all',
                     n_specs=len(sub), frac_OR_gt_1=(sub['OR'] > 1).mean(),
                     median_OR=sub['OR'].median()))
    log(f'  {am:<14s} n = {len(sub):4d}  OR > 1 in {100 * (sub["OR"] > 1).mean():.1f}%  '
        f'median OR {sub["OR"].median():.2f}')
for c in TSS_CUTOFFS:
    sub = df_sane[(df_sane['agg_method'] == 'max_signed') & (df_sane['tss_cutoff_bp'] == c)]
    summ.append(dict(summary='strongest variant by TSS boundary', agg_method='max_signed',
                     tss_cutoff_bp=c, n_specs=len(sub), frac_OR_gt_1=(sub['OR'] > 1).mean(),
                     median_OR=sub['OR'].median()))
log('  median strongest-variant OR by TSS boundary: ' + ', '.join(
    f'>{r["tss_cutoff_bp"] // 1000} kb {r["median_OR"]:.2f}' for r in summ
    if r['summary'].startswith('strongest')))
pd.DataFrame(summ).to_csv(f'{OUTPUT_DIR}/sweep_summary.csv', index=False)

log(f'\n{SEP}')
log(f'  {OUTPUT_DIR}/primary_specification.csv')
log(f'  {OUTPUT_DIR}/sweep_summary.csv')
log(f'  {OUTPUT_DIR}/all_results.csv')
log(f'  {OUTPUT_DIR}/significant_results.csv')
log(f'  {OUTPUT_DIR}/sane_results.csv')
log('Done.')
