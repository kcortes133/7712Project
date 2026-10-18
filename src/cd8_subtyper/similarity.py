"""Set-level helpers (simple, dependency-free versions of the scoring in scoring.py)."""


def jaccard(a, b):
    """|A ∩ B| / |A ∪ B| for two gene collections (0 if both are empty)."""
    a, b = set(a), set(b)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def rank_subtypes(query_genes, reference_sets):
    """Score one query gene set against every reference subtype, best match first.

    Returns a list of ``(subtype, score)`` tuples sorted by descending similarity.
    """
    scores = {name: jaccard(query_genes, genes) for name, genes in reference_sets.items()}
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def marker_overlap(query_genes, marker_sets):
    """For each curated marker category, which of its genes are upregulated in the cluster."""
    query_genes = set(query_genes)
    return {cat: sorted(query_genes & set(genes)) for cat, genes in marker_sets.items()}
