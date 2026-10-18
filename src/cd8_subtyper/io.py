"""Loaders for the reference atlas, TISCH differential-expression tables and marker sets.

Every loader returns a long-format DataFrame with columns ``Cluster``, ``Gene``, ``ES``
(effect size), so reference and query data can be handled identically downstream.

Two lineages are supported: ``cd8`` (17 meta-clusters, bundled) and ``cd4``
(24 meta-clusters; add the file with ``scripts/add_cd4_reference.py``).
"""
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
REF_DIR = DATA_DIR / "reference"
MARKER_CSV = DATA_DIR / "markers" / "marker_gene_sets.csv"
TISCH_DIR = DATA_DIR / "tisch"

LINEAGES = ("cd8", "cd4")
REFERENCE_FILES = {
    "cd8": REF_DIR / "zheng2021_table_s3_cd8_signatures.csv",
    "cd4": REF_DIR / "zheng2021_table_s3_cd4_signatures.csv",
}
STATES_FILES = {
    "cd8": REF_DIR / "subtype_states.csv",
    "cd4": REF_DIR / "subtype_states_cd4.csv",
}
# Kept for backwards compatibility
REFERENCE_CSV = REFERENCE_FILES["cd8"]
STATES_CSV = STATES_FILES["cd8"]

# TISCH minor-lineage labels belonging to each lineage
CD4_TISCH_LABELS = {"CD4Tn", "CD4Tcm", "CD4Tem", "CD4Teff", "CD4Tconv", "Tfh", "Th1", "Th2", "Th17", "Th22", "Treg"}

CD4_MISSING_MSG = (
    "The CD4 reference is not bundled. Download Table S3 from Zheng et al. 2021 "
    "(https://doi.org/10.1126/science.abe6474, Supplementary Materials) and run:\n"
    "    python scripts/add_cd4_reference.py path/to/science.abe6474_table_s3.xlsx"
)


def _check_lineage(lineage):
    if lineage not in LINEAGES:
        raise ValueError(f"Unknown lineage '{lineage}'. Choose from {LINEAGES}.")


def load_reference(path=None, lineage="cd8"):
    """Signature genes for one lineage's meta-clusters (Zheng et al. 2021, Table S3).

    The file has a two-row header (a grouping row, then the real column names);
    ``comb.ES`` is the effect size combined across cancer types.
    ``sig.freq`` (fraction of cancer types in which the gene is significant for that
    subtype) is kept as ``SigFreq`` and used by the weighted scoring method.
    """
    _check_lineage(lineage)
    path = Path(path) if path is not None else REFERENCE_FILES[lineage]
    if not path.exists():
        raise FileNotFoundError(CD4_MISSING_MSG if lineage == "cd4" else f"Reference not found: {path}")
    df = pd.read_csv(path, header=1)
    df = df[["cluster.name", "geneSymbol", "comb.ES", "sig.freq"]].rename(
        columns={"cluster.name": "Cluster", "geneSymbol": "Gene", "comb.ES": "ES", "sig.freq": "SigFreq"}
    )
    for col in ("ES", "SigFreq"):
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
    df["SigFreq"] = df["SigFreq"].fillna(1.0)
    return df.dropna(subset=["ES"]).reset_index(drop=True)


def reference_available(lineage):
    return REFERENCE_FILES[lineage].exists()


def load_tisch_clusters(path):
    """Cluster-level differential expression table exported from TISCH.

    Clusters are identified by their minor-lineage annotation (e.g. ``CD8Tex``);
    ``log2FC`` is used as the effect size.
    """
    df = pd.read_csv(path, sep="\t")
    df = df[["Celltype (minor-lineage)", "Gene", "log2FC"]].rename(
        columns={"Celltype (minor-lineage)": "Cluster", "log2FC": "ES"}
    )
    df["ES"] = df["ES"].astype(float)
    return df


def list_lineage_clusters(df, lineage="cd8"):
    """Cluster labels of one lineage present in a TISCH table (e.g. CD8Tex, or Treg/Th17)."""
    _check_lineage(lineage)
    labels = df["Cluster"].astype(str).unique()
    if lineage == "cd8":
        return sorted(c for c in labels if c.startswith("CD8T"))
    return sorted(c for c in labels if c in CD4_TISCH_LABELS or c.startswith("CD4T"))


def list_cd8_clusters(df):
    """All CD8+ T cell cluster labels present in a TISCH table (e.g. CD8Tex, CD8Tn)."""
    return list_lineage_clusters(df, "cd8")


def load_marker_sets(path=MARKER_CSV):
    """Curated gene sets (exhaustion markers, cytokines/effector molecules, ...) as a dict."""
    df = pd.read_csv(path)
    return {cat: sorted(set(g["gene"])) for cat, g in df.groupby("category")}


def infer_state(subtype):
    """Coarse state from a Zheng meta-cluster name, e.g. 'CD4.c20(TNFRSF9+ Treg)' -> 'Treg'.

    Used to draft the CD4 state mapping; the CD8 mapping is curated by hand.
    """
    name = subtype.split("(", 1)[1].rstrip(")") if "(" in subtype else subtype
    rules = [
        (r"Treg", "Treg"),
        (r"Tfh.*Th1|Th1.*Tfh", "Tfh_Th1"),
        (r"Tfh", "Tfh"),
        (r"Th17", "Th17"),
        (r"Th1\b|Th1-like", "Th1"),
        (r"Th2", "Th2"),
        (r"Tex", "exhausted"),
        (r"Temra|NK-like|CTL", "effector"),
        (r"Tem\b|Tem$|early Tem", "effector_memory"),
        (r"Trm|Tcm|Tm\b|Tm$", "memory"),
        (r"Tn\b|Tn$|Tn\.", "naive"),
    ]
    for pattern, state in rules:
        if re.search(pattern, name):
            return state
    return "other"


def load_subtype_states(lineage="cd8", path=None):
    """Mapping of each reference subtype to a coarse state."""
    _check_lineage(lineage)
    path = Path(path) if path is not None else STATES_FILES[lineage]
    if not path.exists():
        if lineage == "cd4" and reference_available("cd4"):
            subtypes = load_reference(lineage="cd4")["Cluster"].unique()
            return {s: infer_state(s) for s in subtypes}
        raise FileNotFoundError(CD4_MISSING_MSG if lineage == "cd4" else f"State mapping not found: {path}")
    df = pd.read_csv(path)
    return dict(zip(df["subtype"], df["state"]))


def find_tisch_file(dataset, tisch_dir=TISCH_DIR):
    """Resolve a short dataset id like ``BRCA_GSE110686`` to its TSV file."""
    matches = sorted(Path(tisch_dir).glob(f"{dataset}*_AllDiffGenes_table.tsv"))
    if not matches:
        available = ", ".join(available_datasets(tisch_dir))
        raise FileNotFoundError(f"No TISCH table for '{dataset}'. Available: {available}")
    return matches[0]


def available_datasets(tisch_dir=TISCH_DIR):
    return sorted(
        p.name.split("_Cluster_")[0].replace("_10X", "")
        for p in Path(tisch_dir).glob("*_AllDiffGenes_table.tsv")
    )
