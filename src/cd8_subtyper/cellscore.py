"""Per-cell subtype assignment, to estimate the subtype *composition* of each cluster.

A single label per cluster hides mixtures (TISCH clusters are large and coarse). With
cell-level data (AnnData) each cell is scored against every subtype signature with
``scanpy.tl.score_genes`` (mean expression of the signature minus a matched random
background), assigned to its best-scoring subtype, and the assignments are tabulated
per cluster.
"""
import numpy as np
import pandas as pd

from .signatures import DEFAULT_THRESHOLD


def top_signature_genes(ref_df, n_genes=50, threshold=DEFAULT_THRESHOLD):
    """Top-``n_genes`` signature genes per subtype, ranked by effect size × sig.freq."""
    ref = ref_df[ref_df["ES"] > threshold].copy()
    ref["rank_key"] = ref["ES"] * ref.get("SigFreq", 1.0)
    return {
        c: g.sort_values("rank_key", ascending=False)["Gene"].head(n_genes).tolist()
        for c, g in ref.groupby("Cluster")
    }


def score_cells(adata, ref_df, n_genes=50, min_genes=10, random_state=0, uppercase=False):
    """Return a cells x subtypes DataFrame of signature scores (adata.X log-normalised)."""
    import scanpy as sc

    var_names = adata.var_names.str.upper() if uppercase else adata.var_names
    lookup = dict(zip(var_names, adata.var_names))
    ad = adata.copy()
    scores = {}
    for subtype, genes in top_signature_genes(ref_df, n_genes).items():
        present = [lookup[g] for g in genes if g in lookup]
        if len(present) < min_genes:
            continue
        sc.tl.score_genes(ad, present, score_name="_s", random_state=random_state)
        scores[subtype] = ad.obs["_s"].to_numpy()
    if not scores:
        raise ValueError("No subtype signature had enough genes present in adata.var_names.")
    return pd.DataFrame(scores, index=adata.obs_names)


def assign_cells(cell_scores, min_score=0.0):
    """Best-scoring subtype per cell; 'unassigned' if no score exceeds ``min_score``."""
    best = cell_scores.idxmax(axis=1)
    best[cell_scores.max(axis=1) <= min_score] = "unassigned"
    return best


def subtype_composition(adata, groupby, ref_df, n_genes=50, min_score=0.0, uppercase=False):
    """Fraction of each cluster's cells assigned to each subtype (rows sum to 1)."""
    labels = assign_cells(score_cells(adata, ref_df, n_genes, uppercase=uppercase), min_score)
    comp = pd.crosstab(adata.obs[groupby], labels, normalize="index")
    return comp.reindex(columns=sorted(comp.columns, key=lambda c: (c == "unassigned", c)))
