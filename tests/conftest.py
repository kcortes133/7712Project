import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cd8_subtyper import io  # noqa: E402
from cd8_subtyper.cellscore import top_signature_genes  # noqa: E402
from cd8_subtyper.scoring import Reference  # noqa: E402

TN, GZMK_TEX, TERM_TEX, TEMRA = "CD8.c01(Tn)", "CD8.c11(GZMK+ Tex)", "CD8.c12(terminal Tex)", "CD8.c07(Temra)"

# Synthetic clusters and their true subtype mixtures (used by the AnnData tests).
MIXTURES = {
    "A": {TN: 1.0},
    "B": {GZMK_TEX: 0.7, TERM_TEX: 0.3},
    "C": {TEMRA: 1.0},
}


@pytest.fixture(scope="session")
def ref_df():
    return io.load_reference()


@pytest.fixture(scope="session")
def reference(ref_df):
    return Reference.from_table(ref_df)


@pytest.fixture(scope="session")
def tisch(request):
    def load(name):
        return io.load_tisch_clusters(io.find_tisch_file(name))
    return load


def make_synthetic_adata(ref_df, cells_per_cluster=150, seed=0):
    """Log-normalised-looking counts where each cell over-expresses its subtype's signature."""
    anndata = pytest.importorskip("anndata")
    rng = np.random.default_rng(seed)
    sigs = top_signature_genes(ref_df, n_genes=60)
    genes = sorted({g for s in sigs.values() for g in s} | {f"NOISE{i}" for i in range(300)})
    gidx = {g: i for i, g in enumerate(genes)}

    X, clusters, truth = [], [], []
    for cluster, mix in MIXTURES.items():
        subtypes = rng.choice(list(mix), size=cells_per_cluster, p=list(mix.values()))
        for st in subtypes:
            x = rng.gamma(1.0, 0.3, len(genes))  # background
            x[[gidx[g] for g in sigs[st]]] += rng.normal(2.0, 0.3, len(sigs[st])).clip(0)
            X.append(x)
            clusters.append(cluster)
            truth.append(st)
    obs = pd.DataFrame({"cluster": pd.Categorical(clusters), "truth": truth},
                       index=[f"cell{i}" for i in range(len(X))])
    return anndata.AnnData(X=np.array(X, dtype=np.float32), obs=obs, var=pd.DataFrame(index=genes))


@pytest.fixture(scope="session")
def synthetic_adata(ref_df):
    return make_synthetic_adata(ref_df)
