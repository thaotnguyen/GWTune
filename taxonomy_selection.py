import numpy as np


def choose_taxonomy_config(df):
    candidates = df[df["satisfies_constraint"]].copy()
    if candidates.empty:
        raise RuntimeError(
            "No configuration satisfies the all-models-per-community constraint even with fallback; widen the sweep."
        )

    best_ari = candidates["ari_mean"].max()
    candidates = candidates[np.isclose(candidates["ari_mean"], best_ari)]

    return candidates.sort_values(
        ["modularity", "edge_threshold_rel", "resolution"],
        ascending=[False, True, True],
    ).iloc[0]
