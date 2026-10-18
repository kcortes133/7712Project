import pandas as pd
import pytest

import label_clusters


@pytest.mark.parametrize("method", ["jaccard", "signed", "weighted"])
def test_cli_writes_outputs(tmp_path, method):
    label_clusters.main(["-d", "CRC_GSE136394", "-m", method, "-o", str(tmp_path), "--plots"])
    suffix = "" if method == "jaccard" else f"_{method}"
    labels = pd.read_csv(tmp_path / f"cluster_labels{suffix}.csv")
    assert set(labels["cluster"]) == {"CD8Tcm", "CD8Teff", "CD8Tem", "CD8Tn"}
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4
