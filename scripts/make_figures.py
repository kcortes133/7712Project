#!/usr/bin/env python3
"""Build the README figures from the bundled data.

    python scripts/make_figures.py          # writes results/figures/overview_heatmap.png,
                                            # evaluation_accuracy.png and, if the CD4 reference
                                            # is installed, overview_heatmap_cd4.png
Run scripts/evaluate.py first for the evaluation chart.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cd8_subtyper import io  # noqa: E402
from cd8_subtyper.evaluation import ATLAS_DATASETS  # noqa: E402
from cd8_subtyper.plotting import plot_evaluation, plot_overview_heatmap  # noqa: E402
from cd8_subtyper.scoring import score_clusters  # noqa: E402


def main():
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for lineage in io.LINEAGES:
        if not io.reference_available(lineage):
            continue
        ref_df = io.load_reference(lineage=lineage)
        rows = []
        for name in io.available_datasets():
            table = io.load_tisch_clusters(io.find_tisch_file(name))
            clusters = io.list_lineage_clusters(table, lineage)
            if not clusters:
                continue
            S = score_clusters(table, ref_df, clusters)
            rows += [{"dataset": name, "cluster": c, "subtype": s, "score": S.loc[c, s]}
                     for c in S.index for s in S.columns]
        path = out / ("overview_heatmap.png" if lineage == "cd8" else f"overview_heatmap_{lineage}.png")
        plot_overview_heatmap(pd.DataFrame(rows), io.load_subtype_states(lineage), path, ATLAS_DATASETS)
        print("wrote", path)

    summary_csv = ROOT / "results" / "evaluation_summary.csv"
    if summary_csv.exists():
        plot_evaluation(pd.read_csv(summary_csv, index_col=0), out / "evaluation_accuracy.png")
        print("wrote", out / "evaluation_accuracy.png")


if __name__ == "__main__":
    main()
