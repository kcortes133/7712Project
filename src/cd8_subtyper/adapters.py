"""Convert common single-cell outputs into the long ``Cluster, Gene, ES`` format.

Supported inputs
----------------
* **Scanpy / AnnData** (``.h5ad`` or an in-memory ``AnnData``): runs (or reuses)
  ``sc.tl.rank_genes_groups`` and takes ``logfoldchanges`` as the effect size.
* **Seurat** ``FindAllMarkers`` output saved as CSV/TSV (``write.csv(markers, ...)``).
  Uses ``avg_log2FC`` (Seurat ≥ 4) or ``avg_logFC`` (Seurat ≤ 3, natural log →
  converted to log2).
* **TISCH** ``*_AllDiffGenes_table.tsv`` downloads.
* **Any table** with cluster / gene / log2FC columns (names configurable).

Genes are matched to the (human) reference by symbol. For mouse data pass
``uppercase=True`` to upper-case symbols — a rough ortholog mapping that works for most
T cell genes but not all.
"""
from math import log
from pathlib import Path

import pandas as pd

from .io import load_tisch_clusters

FORMATS = ("auto", "tisch", "seurat", "h5ad", "table")


def _finish(df, max_padj=None, padj_col=None, uppercase=False):
    if max_padj is not None and padj_col is not None and padj_col in df:
        df = df[df[padj_col] <= max_padj]
    out = df[["Cluster", "Gene", "ES"]].copy()
    out["Cluster"] = out["Cluster"].astype(str)
    out["Gene"] = out["Gene"].astype(str)
    if uppercase:
        out["Gene"] = out["Gene"].str.upper()
    out["ES"] = out["ES"].astype(float)
    return out.dropna().reset_index(drop=True)


def from_seurat_markers(markers, max_padj=0.05, uppercase=False):
    """Seurat ``FindAllMarkers`` result (DataFrame or path to CSV/TSV)."""
    if not isinstance(markers, pd.DataFrame):
        sep = "\t" if str(markers).endswith((".tsv", ".txt")) else ","
        markers = pd.read_csv(markers, sep=sep)
    df = markers.copy()
    if "gene" not in df:  # write.csv keeps gene names as row names when 'gene' column is absent
        df["gene"] = df.iloc[:, 0] if df.columns[0] in ("", "Unnamed: 0") else df.index
    if "avg_log2FC" in df:
        df["ES"] = df["avg_log2FC"]
    elif "avg_logFC" in df:
        df["ES"] = df["avg_logFC"] / log(2)
    else:
        raise KeyError("Expected an 'avg_log2FC' or 'avg_logFC' column from Seurat FindAllMarkers.")
    df = df.rename(columns={"cluster": "Cluster", "gene": "Gene"})
    return _finish(df, max_padj, "p_val_adj", uppercase)


def from_anndata(adata, groupby, max_padj=0.05, method="wilcoxon", recompute=False, uppercase=False, **rank_kwargs):
    """Differential expression per ``groupby`` group from an AnnData object.

    Reuses ``adata.uns['rank_genes_groups']`` if it was computed for the same
    ``groupby`` (with fold changes); otherwise runs ``sc.tl.rank_genes_groups`` on a copy.
    ``adata.X`` should hold log-normalised expression.
    """
    import scanpy as sc

    rgg = adata.uns.get("rank_genes_groups")
    reuse = (not recompute and rgg is not None and rgg.get("params", {}).get("groupby") == groupby
             and "logfoldchanges" in rgg)
    if not reuse:
        adata = adata.copy()
        sc.tl.rank_genes_groups(adata, groupby, method=method, n_genes=adata.n_vars, **rank_kwargs)
    df = sc.get.rank_genes_groups_df(adata, group=None)
    if "group" not in df:  # single-group result
        df["group"] = list(adata.uns["rank_genes_groups"]["names"].dtype.names)[0]
    df = df.rename(columns={"group": "Cluster", "names": "Gene", "logfoldchanges": "ES"})
    return _finish(df, max_padj, "pvals_adj", uppercase)


def from_table(path_or_df, cluster_col="cluster", gene_col="gene", es_col="log2FC",
               padj_col=None, max_padj=None, uppercase=False, sep=None):
    """Generic table of per-cluster differential expression."""
    df = path_or_df
    if not isinstance(df, pd.DataFrame):
        sep = sep or ("\t" if str(path_or_df).endswith((".tsv", ".txt")) else ",")
        df = pd.read_csv(path_or_df, sep=sep)
    df = df.rename(columns={cluster_col: "Cluster", gene_col: "Gene", es_col: "ES"})
    return _finish(df, max_padj, padj_col, uppercase)


def detect_format(path):
    path = Path(path)
    if path.suffix == ".h5ad":
        return "h5ad"
    if path.name.endswith("_AllDiffGenes_table.tsv"):
        return "tisch"
    sep = "\t" if path.suffix in (".tsv", ".txt") else ","
    cols = set(pd.read_csv(path, sep=sep, nrows=0).columns)
    if {"avg_log2FC", "p_val_adj"} & cols or "avg_logFC" in cols:
        return "seurat"
    if "Celltype (minor-lineage)" in cols:
        return "tisch"
    return "table"


def read_input(path, fmt="auto", groupby=None, uppercase=False, **table_kwargs):
    """Load any supported input file as a ``Cluster, Gene, ES`` table."""
    fmt = detect_format(path) if fmt == "auto" else fmt
    if fmt == "tisch":
        return load_tisch_clusters(path)
    if fmt == "seurat":
        return from_seurat_markers(path, uppercase=uppercase)
    if fmt == "h5ad":
        if not groupby:
            raise ValueError("--groupby is required for .h5ad input (e.g. 'leiden' or 'cell_type').")
        import anndata

        return from_anndata(anndata.read_h5ad(path), groupby, uppercase=uppercase)
    if fmt == "table":
        return from_table(path, uppercase=uppercase, **table_kwargs)
    raise ValueError(f"Unknown format '{fmt}'. Choose from {FORMATS}.")
