"""Regression test: the pipeline must reproduce Table 2 of the project report."""
import pytest

from cd8_subtyper import io
from cd8_subtyper.scoring import score_clusters

# (dataset, TISCH label, top-3 Jaccard labels) from Table 2 of docs/Project_Report.pdf
TABLE_2 = [
    ("BRCA_GSE110686", "CD8Tcm", ["ISG+ CD8+ T", "IL7R+ Tm", "GZMK+ Tem"]),
    ("BRCA_GSE110686", "CD8Tex", ["GZMK+ Tex", "GZMK+ Tem", "ZNF683+CXCR6+ Trm"]),
    ("BRCA_GSE110686", "CD8Tem", ["GZMK+ early Tem", "ZNF683+CXCR6- Tm", "GZMK+ Tem"]),
    ("BRCA_GSE110686", "CD8Teff", ["ZNF683+CXCR6+ Trm", "GZMK+ Tem", "GZMK+ Tex"]),
    ("BRCA_GSE114727", "CD8Tem", ["KIR+TXK+ NK-like", "KIR+EOMES+ NK-like", "ZNF683+CXCR6+ Trm"]),
    ("BRCA_GSE114727", "CD8Tcm", ["GZMK+ Tem", "GZMK+ early Tem", "ZNF683+CXCR6+ Trm"]),
    ("BRCA_GSE114727", "CD8Tex", ["GZMK+ Tex", "GZMK+ Tem", "ZNF683+CXCR6+ Trm"]),
    ("BRCA_GSE114727", "CD8Tn", ["Tn", "IL7R+ Tm", "GZMK+ early Tem"]),
    ("CRC_GSE136394", "CD8Tem", ["IL7R+ Tm", "Temra", "GZMK+ Tem"]),
    ("CRC_GSE136394", "CD8Tcm", ["terminal Tex", "ZNF683+CXCR6+ Trm", "GZMK+ Tex"]),
    ("CRC_GSE136394", "CD8Teff", ["Temra", "GZMK+ Tem", "KIR+EOMES+ NK-like"]),
    ("CRC_GSE136394", "CD8Tn", ["Tn", "TCF7+ Tex", "Tc17"]),
    ("NSCLC_GSE99254", "CD8Tex", ["GZMK+ Tex", "terminal Tex", "GZMK+ Tem"]),
    ("NSCLC_GSE99254", "CD8Tcm", ["ZNF683+CXCR6+ Trm", "GZMK+ Tem", "KIR+TXK+ NK-like"]),
    ("NSCLC_GSE99254", "CD8Teff", ["Temra", "KIR+EOMES+ NK-like", "KIR+TXK+ NK-like"]),
]


@pytest.fixture(scope="module")
def scores(ref_df):
    out = {}
    for name in {row[0] for row in TABLE_2}:
        table = io.load_tisch_clusters(io.find_tisch_file(name))
        out[name] = score_clusters(table, ref_df, io.list_cd8_clusters(table))
    return out


def short(subtype):
    return subtype.split("(", 1)[1][:-1]


@pytest.mark.parametrize("dataset, label, expected", TABLE_2)
def test_table2_top3(scores, dataset, label, expected):
    top3 = scores[dataset].loc[label].sort_values(ascending=False).index[:3]
    assert [short(s) for s in top3] == expected


def test_crc_central_memory_cluster_has_exhaustion_markers(ref_df):
    from cd8_subtyper.signatures import upregulated_genes

    table = io.load_tisch_clusters(io.find_tisch_file("CRC_GSE136394"))
    up = upregulated_genes(table, genes=set(ref_df["Gene"]), clusters=["CD8Tcm"])["CD8Tcm"]
    assert {"TOX", "CTLA4", "HAVCR2", "PDCD1"} <= up
