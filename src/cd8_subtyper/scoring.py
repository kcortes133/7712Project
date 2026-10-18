"""Score query clusters against the 17 reference subtypes.

Three methods share one vectorised implementation (genes x subtypes matrices):

``jaccard``   |up_q ∩ B| / |up_q ∪ B|  — the method from the original report.
``signed``    J(up_q, B) − J(down_q, B). The reference only lists upregulated genes, so
              genes that are *down* in the query count as evidence against subtypes
              whose signature contains them.
``weighted``  Weighted Jaccard (Ruzicka: Σ min / Σ max) on within-cluster percentile
              ranks of the effect size, with reference genes further weighted by how
              consistently they are significant across cancer types (sig.freq).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signatures import DEFAULT_THRESHOLD, _check_clusters, _dedupe

METHODS = ("jaccard", "signed", "weighted")


@dataclass
class Reference:
    subtypes: list
    genes: pd.Index
    binary: np.ndarray  # subtypes x genes, 1 if gene is in the signature
    weights: np.ndarray  # subtypes x genes, rank x sig.freq weights (0 if absent)

    @classmethod
    def from_table(cls, ref_df, threshold=DEFAULT_THRESHOLD, use_sigfreq=True):
        ref = _dedupe(ref_df)
        genes = pd.Index(sorted(ref["Gene"].unique()))
        up = ref[ref["ES"] > threshold].copy()
        up["W"] = up.groupby("Cluster")["ES"].rank(pct=True)
        if use_sigfreq and "SigFreq" in up:
            up["W"] *= up["SigFreq"]
        W = up.pivot(index="Cluster", columns="Gene", values="W").reindex(columns=genes)
        subtypes = sorted(W.index)
        W = W.loc[subtypes]
        return cls(subtypes, genes, W.notna().to_numpy(float), W.fillna(0.0).to_numpy())


@dataclass
class Queries:
    clusters: list
    up_binary: np.ndarray  # clusters x genes
    down_binary: np.ndarray
    up_weights: np.ndarray

    @classmethod
    def from_table(cls, df, reference, clusters=None, threshold=DEFAULT_THRESHOLD):
        df = _dedupe(df[df["Gene"].isin(reference.genes)])
        if clusters is None:
            clusters = list(dict.fromkeys(df["Cluster"]))
        else:
            _check_clusters(df, clusters)
        es = df.pivot(index="Cluster", columns="Gene", values="ES").reindex(index=clusters, columns=reference.genes)
        es = es.fillna(0.0)
        up = es.where(es > threshold, 0.0)
        ranks = up.where(up > 0).rank(axis=1, pct=True).fillna(0.0)
        return cls(list(clusters), (up > 0).to_numpy(float), (es < -threshold).to_numpy(float), ranks.to_numpy())


def _jaccard_matrix(Q, R):
    """Q: n x g binary, R: k x g binary -> n x k Jaccard."""
    inter = Q @ R.T
    union = Q.sum(1, keepdims=True) + R.sum(1)[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def _ruzicka_matrix(Q, R):
    out = np.empty((Q.shape[0], R.shape[0]))
    for j, r in enumerate(R):
        mx = np.maximum(Q, r).sum(1)
        out[:, j] = np.where(mx > 0, np.minimum(Q, r).sum(1) / np.where(mx > 0, mx, 1), 0.0)
    return out


def score_matrix(queries, reference, method="jaccard"):
    """n_clusters x n_subtypes array of scores (higher = more similar)."""
    if method == "jaccard":
        return _jaccard_matrix(queries.up_binary, reference.binary)
    if method == "signed":
        return _jaccard_matrix(queries.up_binary, reference.binary) - _jaccard_matrix(queries.down_binary, reference.binary)
    if method == "weighted":
        return _ruzicka_matrix(queries.up_weights, reference.weights)
    raise ValueError(f"Unknown method '{method}'. Choose from {METHODS}.")


def score_clusters(query_df, ref_df, clusters=None, method="jaccard", threshold=DEFAULT_THRESHOLD):
    """Convenience wrapper: DataFrame (clusters x subtypes) of scores."""
    reference = Reference.from_table(ref_df, threshold=threshold)
    queries = Queries.from_table(query_df, reference, clusters=clusters, threshold=threshold)
    return pd.DataFrame(score_matrix(queries, reference, method), index=queries.clusters, columns=reference.subtypes)


def permutation_pvalues(queries, reference, method="jaccard", n_perm=1000, seed=0):
    """Empirical p-value that each cluster's best score beats random gene sets.

    For every permutation the cluster keeps the same number of up/down genes (and the
    same weights) but they are assigned to genes drawn at random from the reference
    gene universe. p = (1 + #{null best ≥ observed best}) / (1 + n_perm).
    """
    rng = np.random.default_rng(seed)
    observed = score_matrix(queries, reference, method).max(1)
    g = len(reference.genes)
    pvals, null_means = [], []
    for i in range(len(queries.clusters)):
        up_w = queries.up_weights[i][queries.up_binary[i] > 0]
        n_up, n_down = len(up_w), int(queries.down_binary[i].sum())
        Qu = np.zeros((n_perm, g)); Qd = np.zeros((n_perm, g)); Qw = np.zeros((n_perm, g))
        for p in range(n_perm):
            idx = rng.choice(g, size=n_up + n_down, replace=False)
            Qu[p, idx[:n_up]] = 1
            Qw[p, idx[:n_up]] = up_w
            Qd[p, idx[n_up:]] = 1
        null = score_matrix(Queries(list(range(n_perm)), Qu, Qd, Qw), reference, method).max(1)
        pvals.append((1 + (null >= observed[i]).sum()) / (1 + n_perm))
        null_means.append(null.mean())
    return np.array(pvals), np.array(null_means)
