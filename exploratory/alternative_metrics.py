#!/usr/bin/env python3
"""Re-run the approaches tried before settling on Jaccard similarity.

These are the methods from the original exploratory code (see ../archive/), now run on
the same inputs as label_clusters.py so they can be compared side by side:

* Manhattan distance on raw effect-size vectors
* Gower distance on effect sizes thresholded at 0.25
* NMF (3 components), nearest reference subtype in the factor space
* UMAP embedding onto a sphere (haversine output metric), nearest subtype on the sphere

For each CD8+ cluster in each TISCH dataset we record the nearest reference subtype
under every method and compare it with the Jaccard label.

Requires the optional extras:  pip install scikit-learn gower umap-learn
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cd8_subtyper import io  # noqa: E402
from cd8_subtyper.signatures import upregulated_genes  # noqa: E402
from cd8_subtyper.similarity import rank_subtypes  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42


def es_matrix(df, genes, clusters=None):
    """Cluster x gene effect-size matrix (missing = 0)."""
    df = df[df["Gene"].isin(genes)].drop_duplicates(["Cluster", "Gene"], keep="last")
    m = df.pivot(index="Cluster", columns="Gene", values="ES").reindex(columns=genes).fillna(0.0)
    return m.loc[clusters] if clusters is not None else m


def nearest(ref_X, q_X, ref_names, dist_fn):
    return [ref_names[int(np.argmin([dist_fn(q, r) for r in ref_X]))] for q in q_X]


def manhattan_labels(ref, q):
    return nearest(ref.values, q.values, list(ref.index), lambda a, b: np.abs(a - b).sum())


def gower_labels(ref, q):
    import gower

    allX = np.vstack([ref.values, q.values]).astype("f")
    allX = np.where(allX > 0.25, allX, 0).astype("f")
    D = gower.gower_matrix(allX)
    n = len(ref)
    return [ref.index[int(np.argmin(D[n + i, :n]))] for i in range(len(q))]


def nmf_labels(ref, q):
    from sklearn.decomposition import NMF

    allX = np.vstack([ref.values, q.values])
    allX = np.where(allX > 0.25, allX, 0)
    W = NMF(n_components=3, init="random", random_state=2, max_iter=1000).fit_transform(allX)
    n = len(ref)
    return nearest(W[:n], W[n:], list(ref.index), lambda a, b: np.linalg.norm(a - b))


def umap_sphere_labels(ref, q):
    import umap

    R = np.where(ref.values > 0.25, ref.values, 0)
    Q = np.where(q.values > 0.25, q.values, 0)
    mapper = umap.UMAP(output_metric="haversine", random_state=SEED, n_neighbors=min(15, len(R) - 1)).fit(R)

    def to_xyz(e):
        return np.c_[np.sin(e[:, 0]) * np.cos(e[:, 1]), np.sin(e[:, 0]) * np.sin(e[:, 1]), np.cos(e[:, 0])]

    return nearest(to_xyz(mapper.embedding_), to_xyz(mapper.transform(Q)), list(ref.index),
                   lambda a, b: np.linalg.norm(a - b))


METHODS = {
    "manhattan": manhattan_labels,
    "gower": gower_labels,
    "nmf": nmf_labels,
    "umap_sphere": umap_sphere_labels,
}


def main(outdir=ROOT / "results"):
    reference = io.load_reference()
    genes = sorted(set(reference["Gene"]))
    ref = es_matrix(reference, genes)
    ref_sets = upregulated_genes(reference)

    rows = []
    for name in io.available_datasets():
        table = io.load_tisch_clusters(io.find_tisch_file(name))
        clusters = io.list_cd8_clusters(table)
        q = es_matrix(table, genes, clusters)
        q_sets = upregulated_genes(table, genes=genes, clusters=clusters)
        labels = {m: fn(ref, q) for m, fn in METHODS.items()}
        for i, c in enumerate(clusters):
            row = {"dataset": name, "tisch_label": c, "jaccard": rank_subtypes(q_sets[c], ref_sets)[0][0]}
            row.update({m: labels[m][i] for m in METHODS})
            rows.append(row)

    df = pd.DataFrame(rows)
    outdir.mkdir(exist_ok=True)
    df.to_csv(outdir / "alternative_metrics_comparison.csv", index=False)

    short = df.copy()
    for col in ["jaccard", *METHODS]:
        short[col] = short[col].str.replace(r"CD8\.c\d+\((.*)\)", r"\1", regex=True)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(short.to_string(index=False))

    print("\nSummary (n = %d clusters)" % len(df))
    for m in ["jaccard", *METHODS]:
        agree = (df[m] == df["jaccard"]).mean()
        top = df[m].value_counts()
        print(f"  {m:12s} distinct subtypes used: {df[m].nunique():2d} | most common: "
              f"{top.index[0]} ({top.iloc[0]}/{len(df)}) | agrees with Jaccard: {agree:.0%}")


if __name__ == "__main__":
    main()
