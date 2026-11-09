"""
Shared configuration for the variant-analysis scripts: paths, cell-type metadata,
lineage graph, thresholds and trait-to-lineage maps. No other script defines a
threshold or a cell-type list.

Paths default to locations under the repository (data/ and results/) and can be
overridden with CBPNET_* environment variables.

Run `python config.py` to validate the configuration and print a summary.
"""

from __future__ import annotations

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CBPNET_DATA_DIR", REPO_ROOT / "data"))

#: Directory holding one subfolder per cell type, each with per-fold score files.
SCORE_DIR = Path(os.environ.get("CBPNET_SCORE_DIR", DATA_DIR / "variant_scores"))

#: Consensus peak set across all 17 blood cell types (10-column BED). Defines the
#: testable universe, the common denominator for power normalisation.
CONSENSUS_PEAKS = Path(
    os.environ.get("CBPNET_CONSENSUS_PEAKS", SCORE_DIR / "consensus_peaks_blood17_10col.bed")
)

#: Vuckovic fine-mapping credible sets. Tab-separated with a header; the columns used
#: are chr, start, PP, rsid, trait. Despite the .bed extension this is not a 0-based
#: BED: on a SNV row `start` and `end` are equal and both hold the 1-based position
#: (rs28537345 chr1 1737504 1737504), so it joins to the scores unconverted.
#: See PIP_BED_IS_ZERO_BASED below.
PIP_BED = Path(
    os.environ.get(
        "CBPNET_PIP_BED", DATA_DIR / "external" / "CS.ukid.ukbb_v2.PP0.001_hg38.bed"
    )
)

#: GENCODE annotation for nearest gene and TSS distance.
GENCODE_GTF = Path(
    os.environ.get("CBPNET_GENCODE_GTF", DATA_DIR / "external" / "gencode.v47.basic.annotation.gtf")
)

#: gnomAD v2.1.1 gene-level LoF constraint metrics (bgzipped TSV). Supplies LOEUF
#: (`oe_lof_upper`) as the constraint covariate; see CONSTRAINT_METRIC below.
GNOMAD_CONSTRAINT = Path(
    os.environ.get(
        "CBPNET_GNOMAD_CONSTRAINT",
        DATA_DIR / "external" / "gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz",
    )
)

#: gnomAD sites file for allele frequencies (annotation only; nothing is tested on AF).
#: Optional: 05_annotate.py skips the AF columns if it is absent.
GNOMAD_SITES = Path(os.environ["CBPNET_GNOMAD_SITES"]) if os.environ.get("CBPNET_GNOMAD_SITES") else None

OUTPUT_DIR = Path(os.environ.get("CBPNET_OUTPUT_DIR", REPO_ROOT / "results" / "variant_analysis"))

#: Ingest checkpoint written by 01_build_matrix.py; later scripts read this.
MATRIX_PARQUET = OUTPUT_DIR / "variant_celltype_matrix.parquet"

FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


# ──────────────────────────────────────────────────────────────────────────────
# Cell types
# ──────────────────────────────────────────────────────────────────────────────

FOLDS = [0, 1, 2, 3, 4]

#: Peak regions are identical across the five folds (confirmed by md5 of columns 1-3),
#: so fold 0 supplies each cell type's peak set.
PEAK_FOLD = 0

CELL_TYPES = [
    "HSC", "MPP", "LMPP", "CMP", "GMP", "MEP", "CLP",
    "Ery", "Mega", "Mono", "Bcell", "CD4T", "CD8T", "NK",
    "CD34_BM", "CD34_CB", "K562",
]

#: ``lineage`` is the colour/grouping label used in figures. ``dev_class`` is the
#: separate progenitor/mature axis used by the developmental contrast. The two differ
#: (MEP is Erythroid by lineage but progenitor by dev_class).
CELL_TYPE_META = {
    "HSC":     dict(lineage="Progenitor",     dev_class="progenitor", graph_node=True,  label="HSC"),
    "MPP":     dict(lineage="Progenitor",     dev_class="progenitor", graph_node=True,  label="MPP"),
    "LMPP":    dict(lineage="Progenitor",     dev_class="progenitor", graph_node=True,  label="LMPP"),
    "CMP":     dict(lineage="Progenitor",     dev_class="progenitor", graph_node=True,  label="CMP"),
    "GMP":     dict(lineage="Progenitor",     dev_class="progenitor", graph_node=True,  label="GMP"),
    "MEP":     dict(lineage="Erythroid",      dev_class="progenitor", graph_node=True,  label="MEP"),
    "CLP":     dict(lineage="Progenitor",     dev_class="progenitor", graph_node=True,  label="CLP"),
    "Ery":     dict(lineage="Erythroid",      dev_class="mature",     graph_node=True,  label="Ery"),
    "Mega":    dict(lineage="Megakaryocytic", dev_class="mature",     graph_node=True,  label="Mega"),
    "Mono":    dict(lineage="Myeloid",        dev_class="mature",     graph_node=True,  label="Mono"),
    "Bcell":   dict(lineage="Lymphoid",       dev_class="mature",     graph_node=True,  label="B cell"),
    "CD4T":    dict(lineage="Lymphoid",       dev_class="mature",     graph_node=True,  label="CD4 T"),
    "CD8T":    dict(lineage="Lymphoid",       dev_class="mature",     graph_node=True,  label="CD8 T"),
    "NK":      dict(lineage="Lymphoid",       dev_class="mature",     graph_node=True,  label="NK"),
    # Heterogeneous CD34+ pools: they span seven progenitor populations and therefore
    # cannot occupy a single graph position. Progenitors for the developmental contrast,
    # excluded from the coherence test.
    "CD34_BM": dict(lineage="Progenitor",     dev_class="progenitor", graph_node=False, label="CD34+ BM"),
    "CD34_CB": dict(lineage="Progenitor",     dev_class="progenitor", graph_node=False, label="CD34+ CB"),
    # Aneuploid CML line: not a normal developmental state. Excluded from both the
    # coherence test and the developmental contrast; reported as its own category.
    "K562":    dict(lineage="K562",           dev_class="cell_line",  graph_node=False, label="K562"),
}

#: Populations pooled within the CD34+ fractions, per the study's own definition.
CD34_COVERAGE = ["HSC", "MPP", "CMP", "LMPP", "CLP", "GMP", "MEP"]

LINEAGE_PALETTE = {
    "Progenitor":     "#76448A",
    "Erythroid":      "#A93226",
    "Megakaryocytic": "#E08A00",
    "Myeloid":        "#1F618D",
    "Lymphoid":       "#117A65",
    "K562":           "#B7860B",
}


# ──────────────────────────────────────────────────────────────────────────────
# Haematopoietic lineage graph
# ──────────────────────────────────────────────────────────────────────────────
#
# Directed differentiation edges. GMP has two parents (CMP and LMPP), so this is a DAG
# and not a tree: the undirected version contains the cycle MPP-LMPP-GMP-CMP-MPP. All
# coherence statistics are therefore defined on the undirected graph (connected
# subgraph, shortest-path distance), never on a tree.

LINEAGE_EDGES_DIRECTED = [
    ("HSC",  "MPP"),
    ("MPP",  "LMPP"),
    ("MPP",  "CMP"),
    ("CMP",  "GMP"),
    ("CMP",  "MEP"),
    ("LMPP", "CLP"),
    ("LMPP", "GMP"),
    ("GMP",  "Mono"),
    ("MEP",  "Ery"),
    ("MEP",  "Mega"),
    ("CLP",  "Bcell"),
    ("CLP",  "CD4T"),
    ("CLP",  "CD8T"),
    ("CLP",  "NK"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Cell-type universes
# ──────────────────────────────────────────────────────────────────────────────
# Each analysis declares which universe it runs on; breadth is always reported with
# its universe, because "shared in 12 of 17" and "12 of 14" are different claims.

def universe(name: str) -> list[str]:
    """Return the ordered cell-type list for a named analysis universe.

    all17  , every scored cell type. Breadth categories and the Marderstein-comparable
              numbers use this.
    graph14, the lineage-graph nodes. Coherence and direction-flipping use this.
    dev16  , all except K562. The progenitor/mature contrast uses this (9 progenitor,
              7 mature), with graph14 (7/7) available as a sensitivity analysis.
    """
    if name == "all17":
        return list(CELL_TYPES)
    if name == "graph14":
        return [c for c in CELL_TYPES if CELL_TYPE_META[c]["graph_node"]]
    if name == "dev16":
        return [c for c in CELL_TYPES if CELL_TYPE_META[c]["dev_class"] != "cell_line"]
    raise ValueError(f"unknown universe {name!r}; expected one of all17, graph14, dev16")


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
#
# Primary signed effect is `logfc`; primary significance is computed on `abs_logfc`.
# This matches Marderstein et al.'s log2 fold-change in total predicted coverage.
#
# Note: the `<metric>.mean.pval` columns in <CT>.mean.variant_scores.tsv are the
# geometric mean of the five per-fold empirical p-values (variant-scorer's
# variant_summary_across_folds.py). That is a monotone transform of Fisher's combined
# statistic, so its null distribution depends on inter-fold correlation and it is not
# calibrated: the realised rate at a nominal 0.01 cut ranges from 5e-6 (independent
# folds) to 1e-2 (identical folds). These columns are not used here.
# 01_build_matrix.py instead computes a single empirical p per variant per cell type by
# averaging the statistic across folds and comparing it with the fold-averaged shuffled
# null.

EFFECT_METRIC = "logfc"                          # signed effect size, averaged across folds
SIGNIF_METRIC = "abs_logfc"                      # statistic the empirical p is computed on
SECONDARY_SIGNIF_METRIC = "abs_logfc_x_jsd"      # reported in the sensitivity grid
ACCESSIBILITY_METRIC = "active_allele_quantile"  # continuous cross-check, not a filter

#: Columns read from each per-fold score file (kept minimal: 17 cell types x 5 folds x
#: 2 files).
SCORE_ID_COLUMNS = ["chr", "pos", "allele1", "allele2", "variant_id"]
SCORE_VALUE_COLUMNS = [EFFECT_METRIC, SIGNIF_METRIC, SECONDARY_SIGNIF_METRIC, ACCESSIBILITY_METRIC]

#: Variant scores are 1-based.
POS_IS_ONE_BASED = True

#: The consensus-peak file is a true 0-based half-open BED, the credible-set file is
#: 1-based (above). The two disagree, so the convention is a per-file flag rather than
#: one global setting. Every join converts explicitly and asserts on recovery, so a
#: wrong flag fails loudly instead of silently dropping variants.
CONSENSUS_PEAKS_ARE_ZERO_BASED = True
PIP_BED_IS_ZERO_BASED = bool(int(os.environ.get("CBPNET_PIP_BED_IS_ZERO_BASED", "0")))

#: `variant_id` holds a provenance string, not an identifier (values: "HapMap",
#: "Vuckovic", "HapMap,Vuckovic"). The join key is chr:pos:allele1:allele2.
VARIANT_KEY_COLUMNS = ["chr", "pos", "allele1", "allele2"]
PROVENANCE_COLUMN = "variant_id"

#: Expected variant count per cell type and per provenance label.
EXPECTED_N_VARIANTS = 1_565_539
EXPECTED_PROVENANCE_COUNTS = {"HapMap": 1_368_798, "HapMap,Vuckovic": 36_194, "Vuckovic": 160_547}


# ──────────────────────────────────────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────────────────────────────────────
#
# Two rule families are implemented: BH-FDR within cell type on the accessible subset,
# and the uncorrected empirical P < 0.01 used by Marderstein et al. With 1,565,539
# variants scored per cell type an uncorrected 1% cut admits ~15,700 expected false
# positives if applied to all of them, so the rule is applied to the accessible subset
# only and is read together with the threshold ladder (see PRIMARY_RULE_KIND below).

FDR_PRIMARY = 0.05               # BH within cell type, accessible subset
P_RAW_MARDERSTEIN = 0.01         # uncorrected empirical p, comparable to Marderstein et al.

#: Flagship tier for locus-level follow-up.
FLAGSHIP_P = 0.001
FLAGSHIP_PIP = 0.5

#: Threshold grid swept by 02_nominate.py.
SENSITIVITY_FDR = [0.10, 0.05, 0.01]
SENSITIVITY_RAW_P = [0.05, 0.01, 0.001]

#: Magnitude-only comparator: a fixed |logFC| cut (optionally with a PIP filter), used
#: as the reference rule in 03_breadth.py and 05_annotate.py.
MAGNITUDE_ONLY_THRESH = 0.5
MAGNITUDE_ONLY_PIP = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Which rule defines the nominated set
# ──────────────────────────────────────────────────────────────────────────────

#: A variant is called in a cell type when it is accessible there and its
#: within-cell-type empirical p is below PRIMARY_RULE_LEVEL, with no multiple-testing
#: correction. This makes the breadth categories and the specific/shared contrast
#: comparable with Marderstein et al.
#:
#: An uncorrected 0.01 cut admits roughly 1% of accessible variants per cell type by
#: chance. Such calls arise independently in each cell type and concentrate at breadth 1
#: with small effects, so breadth-magnitude relationships are read together with the
#: threshold ladder in 03_breadth.py, which shows whether they survive as the cut
#: tightens. BH-FDR is the strict end of that ladder.
#:
#: Set CBPNET_PRIMARY_RULE_KIND=fdr to use BH-FDR as the primary rule; every module
#: reads the `called` / `n_called` columns that 01_build_matrix.py derives from this
#: setting, so nothing else needs changing.
PRIMARY_RULE_KIND = os.environ.get("CBPNET_PRIMARY_RULE_KIND", "raw")

_PRIMARY_DEFAULT_LEVEL = {"raw": P_RAW_MARDERSTEIN, "fdr": FDR_PRIMARY}
PRIMARY_RULE_LEVEL = float(os.environ.get(
    "CBPNET_PRIMARY_RULE_LEVEL",
    _PRIMARY_DEFAULT_LEVEL.get(PRIMARY_RULE_KIND, P_RAW_MARDERSTEIN)))


def primary_rule() -> tuple[str, float, str]:
    """(kind, level, human label) of the rule that defines the nominated set."""
    if PRIMARY_RULE_KIND == "raw":
        return ("raw", PRIMARY_RULE_LEVEL,
                f"empirical P < {PRIMARY_RULE_LEVEL:g} within cell type")
    if PRIMARY_RULE_KIND == "fdr":
        return ("fdr", PRIMARY_RULE_LEVEL,
                f"BH-FDR < {PRIMARY_RULE_LEVEL:g} within cell type")
    raise ValueError(f"unknown PRIMARY_RULE_KIND: {PRIMARY_RULE_KIND!r}")


#: Threshold ladder for the chance-call diagnostic, one per rule kind. Spans the primary
#: level and progressively stricter cuts.
LADDER_BY_KIND = {
    "raw": (0.05, 0.01, 1e-3, 1e-4, 1e-5),
    "fdr": (0.10, 0.05, 0.01, 1e-3, 1e-4),
}

#: Regulatory magnitude, as defined by Marderstein et al.: the largest absolute
#: predicted effect a variant shows across contexts. Computed over the whole panel
#: rather than over the called contexts, so it does not condition on the selection.
MAGNITUDE_COLUMN = "max_abs_panel"

#: Number of magnitude bins for the decile analysis that mirrors their Fig. 1d.
N_MAGNITUDE_BINS = 10

#: Variant-level breadth column that 01_build_matrix.py writes under the primary rule.
#: Consumers fall back to "n_significant" (the BH count) if it is absent.
BREADTH_COLUMN = "n_called"


def call_column(df) -> str:
    """Name of the per-pair primary-call column in a long table.

    Returns "called" (written by 01_build_matrix.py from primary_rule()) when present,
    otherwise the BH column. Consumers go through this rather than naming a column, so
    switching the primary rule cannot leave one module reading a different set.
    """
    return "called" if "called" in getattr(df, "columns", ()) else "significant"


def breadth_column(df) -> str:
    """Variant-level counterpart of call_column, for a variants table."""
    return BREADTH_COLUMN if BREADTH_COLUMN in getattr(df, "columns", ()) else "n_significant"

#: Breadth categories, following Marderstein et al.: null (0 contexts), specific (1),
#: multiple (2 to the shared cut), shared (>=80% of contexts). The cut is computed from
#: the universe size, so it is 14 of 17 for all17 and 12 of 14 for graph14.
SHARED_FRACTION = 0.80



def breadth_categories(n_contexts: int) -> dict:
    """Marderstein breadth categories for a universe of ``n_contexts`` cell types."""
    import math

    shared_min = math.ceil(SHARED_FRACTION * n_contexts)
    return {
        "inactive": (0, 0),   # named "null" in Marderstein et al.; renamed because
                              # pandas reads a field literally spelled null as NaN
                              # and silently loses the category label
        "specific": (1, 1),
        "multiple": (2, shared_min - 1),
        "shared":   (shared_min, n_contexts),
    }


#: Constraint covariate. LOEUF is used in place of s_het. It is preferred over pLI from
#: the same gnomAD file because pLI saturates near 1 across most constrained genes,
#: whereas LOEUF is continuous.
CONSTRAINT_METRIC = "oe_lof_upper"
CONSTRAINT_METRIC_LABEL = "LOEUF"

#: Permutation / resampling settings.
N_PERMUTATIONS = 10_000
RANDOM_SEED = 20260914


# ──────────────────────────────────────────────────────────────────────────────
# Trait annotation
# ──────────────────────────────────────────────────────────────────────────────

TRAIT_GROUPS = {
    "platelet":            ["plt", "mpv", "pdw", "pct"],
    "mature_red_cell":     ["rbc", "mcv", "hct", "mch", "mscv", "mchc", "hgb", "rdw_cv"],
    "immature_red_cell":   ["ret", "ret_p", "irf", "hlr", "hlr_p", "mrv"],
    "myeloid_white_cell":  ["mono", "neut", "eo", "baso"],
    "lymphoid_white_cell": ["lymph"],
    "compound_white_cell": ["wbc", "mono_p", "neut_p", "eo_p", "baso_p", "lymph_p"],
}

TRAIT_TO_GROUP = {t: g for g, traits in TRAIT_GROUPS.items() for t in traits}


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────────────

def fold_score_file(cell_type: str, fold: int, shuffled: bool = False) -> Path:
    suffix = "variant_scores.shuffled.tsv" if shuffled else "variant_scores.tsv"
    return SCORE_DIR / cell_type / f"{cell_type}_fold{fold}.{suffix}"


def peak_file(cell_type: str, fold: int = PEAK_FOLD) -> Path:
    return SCORE_DIR / cell_type / f"{cell_type}_fold{fold}.peak_scores.tsv"


def mean_score_file(cell_type: str) -> Path:
    """Fold-averaged file from average_folds.py.

    Used only to cross-check the averaged statistics. Its `.mean.pval` columns are not
    used; see the metrics section above.
    """
    return SCORE_DIR / cell_type / f"{cell_type}.mean.variant_scores.tsv"


def ensure_output_dirs() -> None:
    for d in (OUTPUT_DIR, FIGURE_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Self-check
# ──────────────────────────────────────────────────────────────────────────────

def validate() -> dict:
    """Validate the configuration. Raises AssertionError on any inconsistency."""
    assert len(CELL_TYPES) == 17, f"expected 17 cell types, got {len(CELL_TYPES)}"
    assert len(set(CELL_TYPES)) == 17, "duplicate cell type in CELL_TYPES"
    assert set(CELL_TYPE_META) == set(CELL_TYPES), "CELL_TYPE_META / CELL_TYPES mismatch"

    nodes = universe("graph14")
    assert len(nodes) == 14, f"expected 14 graph nodes, got {len(nodes)}"

    edge_nodes = {n for e in LINEAGE_EDGES_DIRECTED for n in e}
    assert edge_nodes == set(nodes), (
        f"graph edges inconsistent with graph_node flags: {sorted(edge_nodes ^ set(nodes))}"
    )

    # Undirected adjacency, connectivity, and the expected single cycle.
    adj = {n: set() for n in nodes}
    for a, b in LINEAGE_EDGES_DIRECTED:
        adj[a].add(b)
        adj[b].add(a)

    seen, stack = {nodes[0]}, [nodes[0]]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur] - seen:
            seen.add(nxt)
            stack.append(nxt)
    assert seen == set(nodes), f"lineage graph disconnected; unreachable: {sorted(set(nodes) - seen)}"

    n_edges = len(LINEAGE_EDGES_DIRECTED)
    n_cycles = n_edges - len(nodes) + 1
    assert n_cycles == 1, (
        f"expected exactly one cycle (MPP-LMPP-GMP-CMP-MPP) from GMP's two parents, "
        f"found {n_cycles} with {n_edges} edges over {len(nodes)} nodes"
    )

    parents = {}
    for a, b in LINEAGE_EDGES_DIRECTED:
        parents.setdefault(b, []).append(a)
    assert sorted(parents["GMP"]) == ["CMP", "LMPP"], (
        f"GMP should have parents CMP and LMPP, found {parents.get('GMP')}"
    )
    multi_parent = {k: v for k, v in parents.items() if len(v) > 1}
    assert set(multi_parent) == {"GMP"}, f"unexpected multi-parent nodes: {multi_parent}"

    assert set(CD34_COVERAGE) <= set(nodes), "CD34_COVERAGE references a non-node cell type"
    for ct in ("CD34_BM", "CD34_CB", "K562"):
        assert not CELL_TYPE_META[ct]["graph_node"], f"{ct} must not be a graph node"

    dev = universe("dev16")
    n_prog = sum(CELL_TYPE_META[c]["dev_class"] == "progenitor" for c in dev)
    n_mat = sum(CELL_TYPE_META[c]["dev_class"] == "mature" for c in dev)
    assert n_prog + n_mat == len(dev) == 16, "dev16 universe is not fully classified"

    for ct in CELL_TYPES:
        assert CELL_TYPE_META[ct]["lineage"] in LINEAGE_PALETTE, f"no palette entry for {ct}"

    assert sum(EXPECTED_PROVENANCE_COUNTS.values()) == EXPECTED_N_VARIANTS, (
        "provenance counts do not sum to EXPECTED_N_VARIANTS"
    )
    assert 0 < FDR_PRIMARY < 1 and 0 < P_RAW_MARDERSTEIN < 1 and 0 < FLAGSHIP_P < 1
    assert PRIMARY_RULE_KIND in ("raw", "fdr"), PRIMARY_RULE_KIND
    assert 0 < PRIMARY_RULE_LEVEL < 1, PRIMARY_RULE_LEVEL
    assert PRIMARY_RULE_KIND in LADDER_BY_KIND
    assert N_MAGNITUDE_BINS >= 2

    return {
        "n_cell_types": len(CELL_TYPES),
        "graph_nodes": len(nodes),
        "graph_edges": n_edges,
        "independent_cycles": n_cycles,
        "dev16_progenitor": n_prog,
        "dev16_mature": n_mat,
        "graph14_progenitor": sum(CELL_TYPE_META[c]["dev_class"] == "progenitor" for c in nodes),
        "graph14_mature": sum(CELL_TYPE_META[c]["dev_class"] == "mature" for c in nodes),
        "breadth_all17": breadth_categories(17),
        "breadth_graph14": breadth_categories(14),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Quadrant analysis: sequence-model evidence against fine-mapping evidence
# ──────────────────────────────────────────────────────────────────────────────

#: Posterior above which a variant is treated as the leading causal candidate in its
#: credible set. The model axis is not set here: it is primary_rule(), the same
#: definition of "active" used by every other module.
QUADRANT_PIP_HIGH = 0.5

#: Bin edges for the posterior gradient. The first bin is the bulk of the fine-mapping
#: output, where the posterior carries almost no information; the last two straddle
#: QUADRANT_PIP_HIGH so the gradient shows whether that threshold sits on a step or on a
#: slope.
PIP_GRADIENT_BINS = (0.0, 0.01, 0.1, 0.5, 0.9, 1.000001)

#: Permutations for the trait-lineage concordance null.
N_TRAIT_PERMUTATIONS = 1_000

#: A consequence class is tested only if a quadrant holds at least this many variants
#: of it; below that the odds ratio is driven by single counts.
VEP_MIN_COUNT = 5

# ──────────────────────────────────────────────────────────────────────────────
# Trait to lineage
# ──────────────────────────────────────────────────────────────────────────────
#
# Blood-cell indices are measured on specific mature lineages, so a trait implies which
# cell types a causal regulatory variant should act in. The mapping is deliberately
# coarse, fixed before the analysis is run, and applied to the base trait code after any
# trailing "_p" (percentage of a parent population) is stripped: eo and eo_p are the same
# lineage. Any trait code in the credible-set file that is neither mapped here nor listed
# as excluded raises, rather than being silently dropped.

#: Base trait code -> lineage. Percentage variants ("_p") resolve to the same entry.
TRAIT_LINEAGE = {
    # red cell indices
    "rbc": "erythroid", "hgb": "erythroid", "hct": "erythroid",
    "mcv": "erythroid", "mch": "erythroid", "mchc": "erythroid",
    "rdw": "erythroid", "rdw_cv": "erythroid", "mrv": "erythroid",
    "mscv": "erythroid", "ret": "erythroid", "hlr": "erythroid",
    "irf": "erythroid",
    # platelet indices
    "plt": "megakaryocytic", "mpv": "megakaryocytic",
    "pct": "megakaryocytic", "pdw": "megakaryocytic",
    # myeloid white cells
    "neut": "myeloid", "mono": "myeloid", "eo": "myeloid", "baso": "myeloid",
    "gran": "myeloid", "eo_baso": "myeloid", "neut_eo_sum": "myeloid",
    "myeloid_wbc": "myeloid",
    # lymphoid white cells
    "lymph": "lymphoid",
}

#: Traits measured on the whole leukocyte pool, which do not imply one lineage and are
#: therefore not used in the concordance test.
TRAIT_LINEAGE_EXCLUDED = ("wbc",)

#: Lineage -> the cell types in which a causal variant for that lineage should act.
#: Strict: the mature population(s) plus the immediate committed progenitor. MEP belongs
#: to both red-cell and platelet lineages, which is what it is.
LINEAGE_CELL_TYPES = {
    "erythroid":      frozenset({"Ery", "MEP"}),
    "megakaryocytic": frozenset({"Mega", "MEP"}),
    "myeloid":        frozenset({"GMP", "Mono"}),
    "lymphoid":       frozenset({"CLP", "Bcell", "CD4T", "CD8T", "NK"}),
}

#: Loose: adds the upstream multipotent progenitor. Reported alongside the strict result
#: so the reader can see the conclusion does not depend on where the boundary is drawn.
LINEAGE_CELL_TYPES_LOOSE = {
    "erythroid":      LINEAGE_CELL_TYPES["erythroid"] | {"CMP"},
    "megakaryocytic": LINEAGE_CELL_TYPES["megakaryocytic"] | {"CMP"},
    "myeloid":        LINEAGE_CELL_TYPES["myeloid"] | {"CMP"},
    "lymphoid":       LINEAGE_CELL_TYPES["lymphoid"] | {"LMPP"},
}


def collapse_duplicate_variants(V, key_col: str = "_key"):
    """Collapse variants that share chr:pos:allele1:allele2 into one row.

    A variant reaching the pipeline from both source lists is ingested twice, with the
    same coordinates, alleles and scores but a different provenance label and
    variant_idx. 01 merges provenance for nearly all of these; a residual handful survive
    as duplicate rows. They are the same variant, so counting them twice would inflate
    set sizes and give one variant two votes in an enrichment.

    The analysis-relevant fields are checked to be identical before collapsing, so this
    can never silently merge two genuinely different records that happen to collide.
    Returns (collapsed frame, number of rows dropped).
    """
    dup = V[key_col].duplicated(keep=False)
    if not dup.any():
        return V, 0
    check = [c for c in ("chr", "pos", "allele1", "allele2", "PIP", "n_accessible",
                         "n_called", "n_significant", "testable") if c in V.columns]
    worst = V.loc[dup].groupby(key_col)[check].nunique().max()
    bad = worst[worst > 1]
    if len(bad):
        raise ValueError(
            f"duplicate variant keys differ in {list(bad.index)}, so they are not the "
            f"same variant and cannot be collapsed. Inspect them before proceeding.")
    out = V.drop_duplicates(subset=[key_col], keep="first")
    return out, int(len(V) - len(out))


def trait_lineage(trait: str) -> str | None:
    """Lineage for a trait code, or None if the trait is not lineage-specific."""
    t = str(trait).strip().lower()
    if t in TRAIT_LINEAGE_EXCLUDED:
        return None
    if t in TRAIT_LINEAGE:
        return TRAIT_LINEAGE[t]
    base = t[:-2] if t.endswith("_p") else t
    if base in TRAIT_LINEAGE_EXCLUDED:
        return None
    return TRAIT_LINEAGE.get(base)


if __name__ == "__main__":
    import json

    info = validate()
    print(json.dumps(info, indent=2, default=str))
    print("\nPaths")
    for name in ("SCORE_DIR", "CONSENSUS_PEAKS", "PIP_BED", "GENCODE_GTF",
                 "GNOMAD_CONSTRAINT", "OUTPUT_DIR"):
        p = globals()[name]
        state = "ok     " if p is not None and Path(p).exists() else "MISSING"
        print(f"  [{state}] {name}: {p}")
