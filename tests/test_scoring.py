import numpy as np
import pandas as pd
import pytest

from cd8_subtyper import io
from cd8_subtyper.scoring import METHODS, Queries, permutation_pvalues, score_clusters, score_matrix
from cd8_subtyper.signatures import upregulated_genes
from cd8_subtyper.similarity import rank_subtypes


def test_matrix_jaccard_matches_set_jaccard(ref_df, tisch):
    table = tisch("BRCA_GSE110686")
    clusters = io.list_cd8_clusters(table)
    S = score_clusters(table, ref_df, clusters)
    ref_sets = upregulated_genes(ref_df)
    q_sets = upregulated_genes(table, genes=set(ref_df["Gene"]), clusters=clusters)
    for c in clusters:
        for subtype, score in rank_subtypes(q_sets[c], ref_sets):
            assert S.loc[c, subtype] == pytest.approx(score)


def _query_from_signature(ref_df, subtype):
    sig = ref_df[ref_df["Cluster"] == subtype]
    return pd.DataFrame({"Cluster": "q", "Gene": sig["Gene"], "ES": sig["ES"]})


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("subtype", ["CD8.c01(Tn)", "CD8.c07(Temra)", "CD8.c12(terminal Tex)"])
def test_a_signature_matches_itself(ref_df, method, subtype):
    S = score_clusters(_query_from_signature(ref_df, subtype), ref_df, method=method)
    assert S.loc["q"].idxmax() == subtype
    if method != "weighted":  # weighted: reference side also carries sig.freq weights, so < 1
        assert S.loc["q", subtype] == pytest.approx(1.0)


def test_score_ranges(ref_df, reference, tisch):
    table = tisch("CRC_GSE136394")
    q = Queries.from_table(table, reference, clusters=io.list_cd8_clusters(table))
    jac, signed, weighted = (score_matrix(q, reference, m) for m in METHODS)
    assert ((jac >= 0) & (jac <= 1)).all()
    assert ((weighted >= 0) & (weighted <= 1)).all()
    assert (signed <= jac + 1e-12).all()  # the down-gene term can only subtract


def test_downregulated_genes_penalise_signed_score(ref_df, reference):
    tn = ref_df[ref_df["Cluster"] == "CD8.c01(Tn)"]
    up = pd.DataFrame({"Cluster": "q", "Gene": tn["Gene"], "ES": 1.0})
    down = pd.DataFrame({"Cluster": "q", "Gene": tn["Gene"].head(50), "ES": -1.0})
    no_down = score_clusters(up, ref_df, method="signed").loc["q", "CD8.c01(Tn)"]
    both = pd.concat([up.iloc[50:], down])
    with_down = score_clusters(both, ref_df, method="signed").loc["q", "CD8.c01(Tn)"]
    assert with_down < no_down


def test_unknown_method_raises(reference, tisch):
    table = tisch("CRC_GSE136394")
    q = Queries.from_table(table, reference, clusters=["CD8Tn"])
    with pytest.raises(ValueError):
        score_matrix(q, reference, "cosine")


def test_missing_cluster_raises(ref_df, tisch):
    with pytest.raises(KeyError):
        score_clusters(tisch("CRC_GSE136394"), ref_df, clusters=["CD8Tfoo"])


def test_permutation_test_separates_real_from_random(ref_df, reference):
    rng = np.random.default_rng(1)
    real = _query_from_signature(ref_df, "CD8.c07(Temra)").head(150)
    random_genes = rng.choice(reference.genes, 150, replace=False)
    rand = pd.DataFrame({"Cluster": "r", "Gene": random_genes, "ES": 1.0})
    q = Queries.from_table(pd.concat([real, rand]), reference)
    p, _ = permutation_pvalues(q, reference, n_perm=200, seed=0)
    assert p[0] < 0.01
    assert p[1] > 0.05
