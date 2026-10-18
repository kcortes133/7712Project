"""Turn effect-size tables into sets of up- / down-regulated genes per cluster."""
DEFAULT_THRESHOLD = 0.15  # effect-size cut-off used by Zheng et al. for signature genes


def _dedupe(df):
    # TISCH can split one cell type across several numbered clusters, so the same
    # (label, gene) pair may appear more than once; keep the last value (as the
    # original analysis did).
    return df.drop_duplicates(subset=["Cluster", "Gene"], keep="last")


def _check_clusters(df, clusters):
    missing = [c for c in clusters if c not in set(df["Cluster"])]
    if missing:
        raise KeyError(f"Clusters not found: {missing}")


def upregulated_genes(df, genes=None, clusters=None, threshold=DEFAULT_THRESHOLD):
    """Return ``{cluster: set_of_genes}`` with effect size above ``threshold``.

    Parameters
    ----------
    df : DataFrame with columns Cluster, Gene, ES
    genes : optional iterable restricting the gene universe (the reference gene set)
    clusters : optional list of clusters to keep, in order
    """
    if genes is not None:
        df = df[df["Gene"].isin(set(genes))]
    df = _dedupe(df)
    up = df[df["ES"] > threshold]
    sets = {c: set(g["Gene"]) for c, g in up.groupby("Cluster", sort=False)}
    if clusters is not None:
        _check_clusters(df, clusters)
        sets = {c: sets.get(c, set()) for c in clusters}
    return sets


def downregulated_genes(df, genes=None, clusters=None, threshold=DEFAULT_THRESHOLD):
    """``{cluster: set_of_genes}`` with effect size below ``-threshold``."""
    flipped = df.assign(ES=-df["ES"])
    return upregulated_genes(flipped, genes=genes, clusters=clusters, threshold=threshold)

