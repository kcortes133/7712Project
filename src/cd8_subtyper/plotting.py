"""Figures: per-cluster score bars, the overview heatmap and the evaluation chart."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
HIGHLIGHT = "#d62728"
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#d9d9d9"

STATE_ORDER = ["naive", "memory", "effector_memory", "effector", "Tfh", "Tfh_Th1", "Th1", "Th2", "Th17",
               "Treg", "exhausted", "other"]
STATE_TITLES = {"naive": "Naïve", "memory": "Memory", "effector_memory": "Effector memory",
                "effector": "Effector / NK-like", "exhausted": "Exhausted", "other": "Other",
                "Tfh": "Tfh", "Tfh_Th1": "Tfh/Th1", "Th1": "Th1", "Th2": "Th2", "Th17": "Th17", "Treg": "Treg"}
TISCH_ORDER = ["CD8Tn", "CD8Tcm", "CD8Tem", "CD8Teff", "CD8Tex",
               "CD4Tn", "CD4Tcm", "CD4Tem", "CD4Teff", "Tfh", "Th1", "Th2", "Th17", "Treg"]
TISCH_TITLES = {"CD8Tn": "Naïve (Tn)", "CD8Tcm": "Central memory (Tcm)", "CD8Tem": "Effector memory (Tem)",
                "CD8Teff": "Effector (Teff)", "CD8Tex": "Exhausted (Tex)",
                "CD4Tn": "Naïve (Tn)", "CD4Tcm": "Central memory (Tcm)", "CD4Tem": "Effector memory (Tem)",
                "CD4Teff": "Effector (Teff)", "Tfh": "Tfh", "Th1": "Th1", "Th2": "Th2", "Th17": "Th17",
                "Treg": "Treg"}


def short_name(subtype):
    return subtype.split("(", 1)[1][:-1] if "(" in subtype else subtype


def plot_subtype_scores(scores, title, out_path, xlabel="Jaccard similarity"):
    """Horizontal bars in reference order (c01 at top); the best match is highlighted."""
    names = sorted(scores)
    values = [scores[n] for n in names]
    best = max(scores, key=scores.get)
    colors = [HIGHLIGHT if n == best else "#1c5cab" for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, values, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CD8+ T cell subtype (Zheng et al. 2021)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overview_heatmap(scores, states, out_path, atlas_datasets=(), expected=None, reference_name="Zheng et al. 2021"):
    """Clusters (rows, grouped by TISCH label) x subtypes (columns, grouped by state).

    ``scores``: long DataFrame with dataset, cluster, subtype, score. Cells are scaled to
    each row's maximum; a dot marks the top label and boxes outline where a correct
    label would fall.
    """
    from .evaluation import TISCH_EXPECTED_STATES

    expected = expected or TISCH_EXPECTED_STATES
    order = lambda st: STATE_ORDER.index(st) if st in STATE_ORDER else len(STATE_ORDER)  # noqa: E731
    subtypes = sorted(states, key=lambda s: (order(states[s]), s))
    wide = scores.pivot_table(index=["dataset", "cluster"], columns="subtype", values="score")[subtypes]
    wide = wide.reset_index()
    wide["_o"] = wide["cluster"].map(lambda c: TISCH_ORDER.index(c) if c in TISCH_ORDER else len(TISCH_ORDER))
    wide = wide.sort_values(["_o", "dataset"]).drop(columns="_o").set_index(["dataset", "cluster"])
    M = wide.to_numpy()
    rel = M / M.max(axis=1, keepdims=True)

    n_rows, n_cols = M.shape
    fig, ax = plt.subplots(figsize=(13, 0.36 * n_rows + 3.2))
    im = ax.imshow(rel, cmap=BLUES, vmin=0, vmax=1, aspect="auto")

    # separators + expected-state boxes
    col_states = [states[s] for s in subtypes]
    row_labels = list(wide.index.get_level_values("cluster"))
    col_edges = [i for i in range(1, n_cols) if col_states[i] != col_states[i - 1]]
    row_edges = [i for i in range(1, n_rows) if row_labels[i] != row_labels[i - 1]]
    for x in col_edges:
        ax.axvline(x - 0.5, color="white", lw=3)
    for y in row_edges:
        ax.axhline(y - 0.5, color="white", lw=3)
    for label in dict.fromkeys(row_labels):
        rows = [i for i, r in enumerate(row_labels) if r == label]
        cols = [j for j, s in enumerate(col_states) if s in expected.get(label, ())]
        runs = []  # contiguous column runs
        for j in cols:
            if runs and j == runs[-1][-1] + 1:
                runs[-1].append(j)
            else:
                runs.append([j])
        for run in runs:
            ax.add_patch(Rectangle((run[0] - 0.5, rows[0] - 0.5), len(run), len(rows),
                                   fill=False, ec=HIGHLIGHT, lw=2, ls=(0, (4, 2)), zorder=3))

    # top-1 markers
    best = M.argmax(axis=1)
    ax.scatter(best, np.arange(n_rows), s=46, c="white", edgecolors=INK, linewidths=1.4, zorder=4)

    # axes labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([short_name(s) for s in subtypes], rotation=40, ha="right", fontsize=9, color=INK)
    datasets = list(wide.index.get_level_values("dataset"))
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([d + (" †" if d in atlas_datasets else "") for d in datasets], fontsize=8.5, color=INK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # column-group titles on top
    starts = [0] + col_edges
    ends = col_edges + [n_cols]
    for a, b in zip(starts, ends):
        ax.text((a + b - 1) / 2, -0.9, STATE_TITLES.get(col_states[a], col_states[a]), ha="center", va="bottom",
                fontsize=9.5, color=INK, fontweight="bold")
    # row-group titles on the left
    rstarts = [0] + row_edges
    rends = row_edges + [n_rows]
    for a, b in zip(rstarts, rends):
        ax.text(-0.15, (a + b - 1) / 2, TISCH_TITLES.get(row_labels[a], row_labels[a]), ha="right", va="center",
                fontsize=9.5, color=INK, fontweight="bold", transform=ax.get_yaxis_transform())
    ax.text(-0.15, -0.9, "TISCH annotation", ha="right", va="bottom", fontsize=9, color=MUTED,
            transform=ax.get_yaxis_transform())
    ax.text(n_cols / 2 - 0.5, -2.3, f"Reference subtype ({reference_name})", ha="center", va="bottom",
            fontsize=9, color=MUTED)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Jaccard similarity (scaled to row max)", color=MUTED, fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=MUTED, length=0)

    fig.text(0.01, 0.015,
             "○ top label    dashed box = subtypes consistent with the TISCH annotation    "
             "† dataset used to build the reference atlas", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.26, right=0.93, top=0.89, bottom=0.17)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_evaluation(summary, out_path):
    """Top-1 accuracy per method with the chance level as a reference line."""
    methods = list(summary.index)
    acc = summary["top1_accuracy"].to_numpy()
    chance = float(summary["chance_top1"].iloc[0])
    final = {"jaccard"}
    colors = ["#1c5cab" if m in final else "#9ec5f4" for m in methods]

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(methods))
    ax.barh(y, acc, color=colors, height=0.62)
    ax.axvline(chance, color=INK, lw=1.2, ls=(0, (3, 2)))
    ax.text(chance, len(methods) - 0.35, f" chance {chance:.0%}", fontsize=8.5, color=INK, va="bottom")
    for yi, a in zip(y, acc):
        ax.text(a + 0.01, yi, f"{a:.0%}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(methods, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Top-1 label in the expected coarse state", color=MUTED)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
