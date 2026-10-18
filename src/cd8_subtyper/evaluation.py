"""Evaluate subtype labels against TISCH's coarse annotations.

TISCH uses a handful of coarse labels; the reference has 17 (CD8) or 24 (CD4) fine
subtypes. Each fine subtype is assigned a coarse *state*
(``data/reference/subtype_states*.csv``) and a label counts as correct when its state
is one of those accepted for the TISCH label. The mapping is a judgement call, so it
lives in editable CSVs rather than in code.
"""
from math import comb

import pandas as pd

# TISCH label -> accepted coarse states
TISCH_EXPECTED_STATES = {
    # CD8
    "CD8Tn": {"naive"},
    "CD8Tcm": {"memory"},
    "CD8Tem": {"effector_memory"},
    "CD8Teff": {"effector"},
    "CD8Tex": {"exhausted"},
    # CD4
    "CD4Tn": {"naive"},
    "CD4Tcm": {"memory"},
    "CD4Tem": {"effector_memory"},
    "CD4Teff": {"effector", "effector_memory"},
    "Tfh": {"Tfh", "Tfh_Th1"},
    "Th1": {"Th1", "Tfh_Th1"},
    "Th2": {"Th2"},
    "Th17": {"Th17"},
    "Treg": {"Treg"},
}
# Backwards-compatible single-state view of the CD8 labels
TISCH_EXPECTED_STATE = {k: next(iter(v)) for k, v in TISCH_EXPECTED_STATES.items() if k.startswith("CD8")}

# Datasets that were part of the data Zheng et al. used to build the atlas.
ATLAS_DATASETS = {"BRCA_GSE114727", "CRC_GSE146771"}


def expected_states(tisch_label):
    return TISCH_EXPECTED_STATES.get(tisch_label, set())


def is_evaluable(tisch_label, states):
    """A label can only be scored if the reference has at least one subtype in an accepted state."""
    return bool(expected_states(tisch_label) & set(states.values()))


def hit_at_k(ranked_subtypes, tisch_label, states, k):
    """True if any of the top-k subtypes has a state accepted for ``tisch_label``."""
    ok = expected_states(tisch_label)
    return any(states.get(s) in ok for s in ranked_subtypes[:k])


def chance_hit_at_k(tisch_label, states, k):
    """Probability that k subtypes drawn at random include one with an accepted state."""
    ok = expected_states(tisch_label)
    n = len(states)
    m = sum(1 for s in states.values() if s in ok)
    return 1 - comb(n - m, k) / comb(n, k)


def summarise(per_cluster, group_col="method"):
    """Accuracy table from a per-cluster frame with hit@1 / hit@3 / chance columns."""
    def agg(g):
        out = {
            "n_clusters": len(g),
            "top1_accuracy": g["hit@1"].mean(),
            "chance_top1": g["chance@1"].mean(),
        }
        if g["hit@3"].notna().any():
            out["top3_accuracy"] = g["hit@3"].astype(float).mean()
            out["chance_top3"] = g["chance@3"].astype(float).mean()
        if "p_value" in g and g["p_value"].notna().any():
            out["significant_p<0.05"] = (g["p_value"] < 0.05).mean()
        return pd.Series(out)

    return pd.DataFrame({name: agg(g) for name, g in per_cluster.groupby(group_col, sort=False)}).T
