#!/usr/bin/env python3
"""Evaluate labeling accuracy against TISCH annotations, with chance and permutation baselines.

For every scoring method (jaccard, signed, weighted) and every CD8+ cluster in the six
bundled datasets this records:

* hit@1 / hit@3   – is the top (or any top-3) subtype in the coarse state implied by
                    the TISCH label? (mapping: data/reference/subtype_states.csv)
* chance@1 / @3   – the same probability for randomly chosen subtypes
* p_value         – permutation test: does the best score beat random gene sets of
                    the same size? (1,000 permutations)

If results/alternative_metrics_comparison.csv exists (from
exploratory/alternative_metrics.py), the earlier methods are scored too (top-1 only).

    python scripts/evaluate.py [--lineage cd8|cd4] [--n-perm 1000]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cd8_subtyper import io  # noqa: E402
from cd8_subtyper.evaluation import (ATLAS_DATASETS, chance_hit_at_k, hit_at_k,  # noqa: E402
                                     is_evaluable, summarise)
from cd8_subtyper.scoring import METHODS, Queries, Reference, permutation_pvalues, score_matrix  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-l", "--lineage", choices=io.LINEAGES, default="cd8")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("-o", "--outdir", type=Path, default=ROOT / "results")
    args = ap.parse_args(argv)

    try:
        states = io.load_subtype_states(args.lineage)
        reference = Reference.from_table(io.load_reference(lineage=args.lineage))
    except FileNotFoundError as e:
        ap.error(str(e))
    tag = "" if args.lineage == "cd8" else f"_{args.lineage}"

    rows = []
    for name in io.available_datasets():
        table = io.load_tisch_clusters(io.find_tisch_file(name))
        clusters = [c for c in io.list_lineage_clusters(table, args.lineage) if is_evaluable(c, states)]
        if not clusters:
            continue
        queries = Queries.from_table(table, reference, clusters=clusters)
        for method in METHODS:
            S = score_matrix(queries, reference, method)
            pvals, null_means = permutation_pvalues(queries, reference, method, n_perm=args.n_perm)
            for i, cluster in enumerate(queries.clusters):
                order = S[i].argsort()[::-1]
                ranked = [reference.subtypes[j] for j in order]
                rows.append({
                    "method": method, "dataset": name, "tisch_label": cluster,
                    "atlas_dataset": name in ATLAS_DATASETS,
                    "top1": ranked[0], "top1_state": states[ranked[0]], "score": S[i, order[0]],
                    "null_mean_best": null_means[i], "p_value": pvals[i],
                    "hit@1": hit_at_k(ranked, cluster, states, 1), "hit@3": hit_at_k(ranked, cluster, states, 3),
                    "chance@1": chance_hit_at_k(cluster, states, 1), "chance@3": chance_hit_at_k(cluster, states, 3),
                })

    alt_csv = args.outdir / "alternative_metrics_comparison.csv"
    if alt_csv.exists() and args.lineage == "cd8":
        alt = pd.read_csv(alt_csv)
        for m in [c for c in alt.columns if c not in ("dataset", "tisch_label", "jaccard")]:
            for _, r in alt.iterrows():
                rows.append({
                    "method": m, "dataset": r.dataset, "tisch_label": r.tisch_label,
                    "atlas_dataset": r.dataset in ATLAS_DATASETS, "top1": r[m], "top1_state": states[r[m]],
                    "hit@1": hit_at_k([r[m]], r.tisch_label, states, 1), "hit@3": None,
                    "chance@1": chance_hit_at_k(r.tisch_label, states, 1), "chance@3": None,
                })

    per_cluster = pd.DataFrame(rows)
    args.outdir.mkdir(exist_ok=True)
    per_cluster.to_csv(args.outdir / f"evaluation_per_cluster{tag}.csv", index=False)

    overall = summarise(per_cluster)
    independent = summarise(per_cluster[~per_cluster["atlas_dataset"]])
    overall.to_csv(args.outdir / f"evaluation_summary{tag}.csv")
    independent.to_csv(args.outdir / f"evaluation_summary_independent{tag}.csv")

    pd.set_option("display.width", 200)
    print("All six datasets\n", overall.round(2).to_string(), sep="")
    print("\nFour independent datasets only\n", independent.round(2).to_string(), sep="")
    jac = per_cluster[per_cluster.method == "jaccard"]
    print("\nJaccard accuracy by TISCH label:")
    print(jac.groupby("tisch_label")["hit@1"].agg(["sum", "count"]).to_string())
    print("\nJaccard misses:")
    print(jac[~jac["hit@1"]][["dataset", "tisch_label", "top1", "score", "p_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
