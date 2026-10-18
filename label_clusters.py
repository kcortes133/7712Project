#!/usr/bin/env python3
"""Label CD8+ (or CD4+) T cell clusters with the pan-cancer subtypes of Zheng et al. (2021).

Inputs: bundled TISCH datasets, or your own file — a Scanpy .h5ad, a Seurat
FindAllMarkers CSV, a TISCH table, or any cluster/gene/log2FC table.

Examples
--------
    python label_clusters.py                                   # all bundled datasets
    python label_clusters.py -d CRC_GSE136394 --plots
    python label_clusters.py -i pbmc.h5ad --groupby leiden --composition
    python label_clusters.py -i seurat_markers.csv -c 3 5 7
    python label_clusters.py -i de.csv --format table --cluster-col group --gene-col names --es-col logfc
    python label_clusters.py --method weighted                 # alternative scoring
    python label_clusters.py --lineage cd4                     # CD4 subtypes (needs the CD4 reference)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cd8_subtyper import io  # noqa: E402
from cd8_subtyper.adapters import FORMATS, read_input  # noqa: E402
from cd8_subtyper.scoring import METHODS, Queries, Reference, score_matrix  # noqa: E402
from cd8_subtyper.signatures import DEFAULT_THRESHOLD, upregulated_genes  # noqa: E402
from cd8_subtyper.similarity import marker_overlap  # noqa: E402

TOP_N = 3


def label_table(name, table, reference, markers, clusters, method, threshold, plot_dir, lineage="cd8"):
    """Score every requested cluster of one input table; returns (summary_rows, long_scores)."""
    if clusters is None:
        lineage_clusters = io.list_lineage_clusters(table, lineage)
        if not lineage_clusters and (io.list_lineage_clusters(table, "cd8") or io.list_lineage_clusters(table, "cd4")):
            return [], []  # TISCH-style table without clusters of this lineage
        clusters = lineage_clusters or list(dict.fromkeys(table["Cluster"]))  # non-TISCH input: all clusters
    clusters = [str(c) for c in clusters]
    queries = Queries.from_table(table, reference, clusters=clusters, threshold=threshold)
    S = pd.DataFrame(score_matrix(queries, reference, method), index=clusters, columns=reference.subtypes)
    up_sets = upregulated_genes(table, genes=reference.genes, clusters=clusters, threshold=threshold)

    summary, long_rows = [], []
    for cluster in clusters:
        ranking = S.loc[cluster].sort_values(ascending=False)
        row = {"dataset": name, "cluster": cluster, "lineage": lineage, "method": method,
               "n_upregulated": len(up_sets[cluster])}
        for i, (subtype, score) in enumerate(ranking.head(TOP_N).items(), start=1):
            row[f"label_{i}"] = subtype
            row[f"score_{i}"] = round(float(score), 4)
        for category, hits in marker_overlap(up_sets[cluster], markers).items():
            row[category] = ",".join(hits)
        summary.append(row)
        long_rows += [{"dataset": name, "cluster": cluster, "subtype": s, "score": v} for s, v in ranking.items()]
        if plot_dir is not None:
            from cd8_subtyper.plotting import plot_subtype_scores

            plot_subtype_scores(ranking.to_dict(), f"{name} – {cluster}", plot_dir / f"{name}_{cluster}.png",
                                xlabel=f"{method} score")
    return summary, long_rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("-d", "--dataset", nargs="+", help="bundled dataset id(s), e.g. BRCA_GSE110686 (default: all)")
    src.add_argument("-i", "--input", type=Path, help="your own file (.h5ad, Seurat markers CSV, TISCH TSV, table)")
    p.add_argument("--format", choices=FORMATS, default="auto", help="input format (default: detect)")
    p.add_argument("--groupby", help="obs column with cluster labels (required for .h5ad)")
    p.add_argument("--cluster-col", default="cluster", help="--format table: cluster column")
    p.add_argument("--gene-col", default="gene", help="--format table: gene column")
    p.add_argument("--es-col", default="log2FC", help="--format table: effect-size column")
    p.add_argument("--uppercase-genes", action="store_true", help="upper-case gene symbols (e.g. mouse data)")
    p.add_argument("-l", "--lineage", choices=io.LINEAGES, default="cd8",
                   help="reference to label against (default: %(default)s)")
    p.add_argument("-c", "--clusters", nargs="+",
                   help="clusters to score (default: the lineage's TISCH clusters, else all)")
    p.add_argument("-m", "--method", choices=METHODS, default="jaccard", help="scoring method (default: %(default)s)")
    p.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="effect-size cut-off for up/down-regulated (default: %(default)s)")
    p.add_argument("-o", "--outdir", type=Path, default=Path("results"), help="output folder (default: %(default)s)")
    p.add_argument("--plots", action="store_true", help="save a score bar chart per cluster")
    p.add_argument("--composition", action="store_true",
                   help=".h5ad only: assign each cell to a subtype and report per-cluster composition")
    p.add_argument("--n-genes", type=int, default=50, help="--composition: signature genes per subtype")
    args = p.parse_args(argv)

    try:
        ref_df = io.load_reference(lineage=args.lineage)
    except FileNotFoundError as e:
        p.error(str(e))
    reference = Reference.from_table(ref_df, threshold=args.threshold)
    markers = io.load_marker_sets()

    if args.input:
        name = args.input.name.split("_Cluster_")[0].rsplit(".", 1)[0]
        table_kwargs = dict(cluster_col=args.cluster_col, gene_col=args.gene_col, es_col=args.es_col)
        jobs = [(name, read_input(args.input, args.format, args.groupby, args.uppercase_genes, **table_kwargs))]
    else:
        names = args.dataset or io.available_datasets()
        jobs = [(n, io.load_tisch_clusters(io.find_tisch_file(n))) for n in names]

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = None
    if args.plots:
        plot_dir = args.outdir / "figures"
        plot_dir.mkdir(parents=True, exist_ok=True)

    summary, scores = [], []
    for name, table in jobs:
        s, sc = label_table(name, table, reference, markers, args.clusters, args.method, args.threshold, plot_dir,
                            args.lineage)
        if not s:
            continue
        summary += s
        scores += sc

    suffix = ("" if args.lineage == "cd8" else f"_{args.lineage}") + ("" if args.method == "jaccard" else f"_{args.method}")
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.outdir / f"cluster_labels{suffix}.csv", index=False)
    pd.DataFrame(scores).to_csv(args.outdir / f"all_subtype_scores{suffix}.csv", index=False)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df[["dataset", "cluster", "label_1", "score_1", "label_2", "label_3"]].to_string(index=False))
    print(f"\nWrote {args.outdir / f'cluster_labels{suffix}.csv'} and {args.outdir / f'all_subtype_scores{suffix}.csv'}")

    if args.composition:
        if not (args.input and str(args.input).endswith(".h5ad")):
            p.error("--composition needs an .h5ad input")
        import anndata

        from cd8_subtyper.cellscore import subtype_composition

        comp = subtype_composition(anndata.read_h5ad(args.input), args.groupby, ref_df, n_genes=args.n_genes,
                                   uppercase=args.uppercase_genes)
        comp.to_csv(args.outdir / "subtype_composition.csv")
        print("\nPer-cluster subtype composition (fraction of cells):")
        print(comp.round(2).to_string())


if __name__ == "__main__":
    main()
