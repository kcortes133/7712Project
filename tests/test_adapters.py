from math import log

import pandas as pd
import pytest

from cd8_subtyper.adapters import detect_format, from_seurat_markers, from_table, read_input
from cd8_subtyper.scoring import score_clusters

from conftest import MIXTURES


def test_seurat_v4_columns(tmp_path):
    df = pd.DataFrame({
        "p_val": [1e-10, 1e-8, 0.5], "avg_log2FC": [1.2, -0.8, 2.0], "pct.1": [0.9, 0.1, 0.5],
        "pct.2": [0.2, 0.6, 0.5], "p_val_adj": [1e-6, 1e-4, 1.0], "cluster": [0, 0, 1],
        "gene": ["PDCD1", "SELL", "TOX"],
    })
    path = tmp_path / "markers.csv"
    df.to_csv(path, index=False)
    assert detect_format(path) == "seurat"
    out = read_input(path)
    assert list(out.columns) == ["Cluster", "Gene", "ES"]
    assert len(out) == 2  # the non-significant TOX row is dropped
    assert out["Cluster"].tolist() == ["0", "0"]


def test_seurat_v3_natural_log_is_converted():
    df = pd.DataFrame({"avg_logFC": [log(4)], "p_val_adj": [0.0], "cluster": ["1"], "gene": ["GZMK"]})
    assert from_seurat_markers(df)["ES"].iloc[0] == pytest.approx(2.0)


def test_generic_table_and_uppercase():
    df = pd.DataFrame({"grp": ["a"], "sym": ["Pdcd1"], "lfc": [1.0]})
    out = from_table(df, cluster_col="grp", gene_col="sym", es_col="lfc", uppercase=True)
    assert out.iloc[0].to_dict() == {"Cluster": "a", "Gene": "PDCD1", "ES": 1.0}


def test_tisch_file_detected():
    from cd8_subtyper import io

    assert detect_format(io.find_tisch_file("CRC_GSE136394")) == "tisch"


# --- Scanpy / AnnData -------------------------------------------------------------

def test_anndata_cluster_labels(synthetic_adata, ref_df):
    pytest.importorskip("scanpy")
    from cd8_subtyper import io
    from cd8_subtyper.adapters import from_anndata

    de = from_anndata(synthetic_adata, "cluster")
    best = score_clusters(de, ref_df).idxmax(axis=1)
    # pure clusters get their exact subtype
    assert best["A"] == "CD8.c01(Tn)"
    assert best["C"] == "CD8.c07(Temra)"
    # a 70/30 mix of two exhausted subtypes is still called exhausted at the cluster level,
    # but not necessarily as either component -- the motivation for per-cell composition
    assert io.load_subtype_states()[best["B"]] == "exhausted"


def test_composition_recovers_mixtures(synthetic_adata, ref_df):
    pytest.importorskip("scanpy")
    from cd8_subtyper.cellscore import subtype_composition

    comp = subtype_composition(synthetic_adata, "cluster", ref_df)
    for cluster, mix in MIXTURES.items():
        for subtype, frac in mix.items():
            assert comp.loc[cluster, subtype] == pytest.approx(frac, abs=0.1)


def test_h5ad_end_to_end(synthetic_adata, tmp_path):
    pytest.importorskip("scanpy")
    import label_clusters

    path = tmp_path / "toy.h5ad"
    synthetic_adata.write_h5ad(path)
    label_clusters.main(["-i", str(path), "--groupby", "cluster", "--composition", "-o", str(tmp_path / "out")])
    labels = pd.read_csv(tmp_path / "out" / "cluster_labels.csv")
    assert labels.set_index("cluster").loc["A", "label_1"] == "CD8.c01(Tn)"
    assert (tmp_path / "out" / "subtype_composition.csv").exists()
