import argparse
import os
import json
import pickle as pkl
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations
from networkx.algorithms.community import louvain_communities, modularity
from sklearn.metrics import adjusted_rand_score
from taxonomy_selection import choose_taxonomy_config

parser = argparse.ArgumentParser()
parser.add_argument("--centroids", action="store_true", help="Use sae_centroids OT results instead of sae_medoids")
args = parser.parse_args()

OUTPUT_ROOT = "/home/ttn/Development/med-interp/GWTune/results"
TAG = "sae_centroids" if args.centroids else "sae_medoids"
EDGE_THRESHOLDS_REL = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
RESOLUTIONS = [0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]
SEEDS = list(range(16))
MAX_FALLBACK_MERGES = None  # auto-cap = K-1 per partition
N_REQUIRED_MODELS = 6

POINTS_PATH = os.path.join(OUTPUT_ROOT, f"kpartite_cluster_points_{TAG}.pkl")
SWEEP_PATH = os.path.join(OUTPUT_ROOT, f"taxonomy_sweep_{TAG}.json")
PARTITION_PATH = os.path.join(OUTPUT_ROOT, f"taxonomy_partition_{TAG}.pkl")

with open(os.path.join(OUTPUT_ROOT, f"centroid_ot_pairs_{TAG}.pkl"), "rb") as f:
    ot_pairs = pkl.load(f)
with open(os.path.join(OUTPUT_ROOT, f"centroid_index_{TAG}.pkl"), "rb") as f:
    medoid_index = pkl.load(f)

# fixed ordered node list ensures ARI uses a consistent index across seeds
NODES = []
NODE_TO_MODEL = {}
for name, info in medoid_index.items():
    for c in range(len(info["cluster_members"])):
        NODES.append((name, c))
        NODE_TO_MODEL[(name, c)] = name
NODE_INDEX = {node: i for i, node in enumerate(NODES)}
ALL_MODELS = sorted({m for m, _ in NODES})
assert len(ALL_MODELS) == N_REQUIRED_MODELS, f"expected {N_REQUIRED_MODELS} models, got {ALL_MODELS}"


# build the k-partite OT graph; thr is relative to the per-pair max, matching kpartite_cluster.py
def build_graph(edge_threshold_rel):
    G = nx.Graph()
    for node in NODES:
        G.add_node(node, partition=NODE_TO_MODEL[node])
    for key, T in ot_pairs.items():
        src, tgt = key.split("_vs_")
        thr = edge_threshold_rel * float(T.max())
        rows, cols = np.where(T > thr)
        for i, j in zip(rows, cols):
            G.add_edge((src, int(i)), (tgt, int(j)), weight=float(T[i, j]))
    return G


# seeded CPU Louvain. cuGraph currently returns identical partitions across seeds for
# these small graphs, which makes the ARI sweep uninformative.
def run_louvain(G, seed, resolution):
    return louvain_communities(G, weight="weight", seed=seed, resolution=resolution)


# repeatedly merge the most-deficient community into its best-connected neighbor until every
# community contains nodes from all 6 models, or until only one community remains
def post_merge_fallback(communities, G, max_merges):
    communities = [set(c) for c in communities]
    merges = 0
    while merges < max_merges and len(communities) > 1:
        # models present in each community
        present = [{NODE_TO_MODEL[n] for n in c} for c in communities]
        missing_counts = [N_REQUIRED_MODELS - len(p) for p in present]
        if max(missing_counts) == 0:
            break
        # pick most-deficient (tie: smallest community)
        worst = max(range(len(communities)), key=lambda i: (missing_counts[i], -len(communities[i])))
        # find best-connected target via summed inter-community edge weight
        weights_to = np.zeros(len(communities))
        for u in communities[worst]:
            for v, data in G[u].items():
                for j, c in enumerate(communities):
                    if j == worst:
                        continue
                    if v in c:
                        weights_to[j] += data["weight"]
                        break
        # if isolated, fall back to merging into the largest community
        if weights_to.sum() == 0:
            target = max(range(len(communities)), key=lambda i: -1 if i == worst else len(communities[i]))
        else:
            target = int(np.argmax(weights_to))
        communities[target] |= communities[worst]
        del communities[worst]
        merges += 1
    return communities, merges


# convert a list-of-sets partition into a per-node label vector aligned to NODES order
def labels_from_partition(communities):
    label_of = {}
    for cid, members in enumerate(communities):
        for node in members:
            label_of[node] = cid
    return np.asarray([label_of[node] for node in NODES], dtype=np.int64)


# sweep all (edge_threshold_rel, resolution) configurations
print(f"Sweeping {len(EDGE_THRESHOLDS_REL)} thresholds x {len(RESOLUTIONS)} resolutions x {len(SEEDS)} seeds")
sweep_rows = []
best_partitions = {}  # (thr, res) -> list of post-merge partitions
for thr in EDGE_THRESHOLDS_REL:
    G = build_graph(thr)
    n_edges = G.number_of_edges()
    print(f"thr={thr}: {G.number_of_nodes()} nodes, {n_edges} edges")
    if n_edges == 0:
        continue
    for res in RESOLUTIONS:
        partitions_after = []
        merges_list = []
        zero_merge_count = 0
        for seed in SEEDS:
            communities = run_louvain(G, seed, res)
            cap = MAX_FALLBACK_MERGES if MAX_FALLBACK_MERGES is not None else max(0, len(communities) - 1)
            merged, m = post_merge_fallback(communities, G, cap)
            partitions_after.append(merged)
            merges_list.append(m)
            if m == 0:
                zero_merge_count += 1

        # all post-merge partitions must satisfy the hard constraint (else config is dropped)
        all_ok = all(
            all(len({NODE_TO_MODEL[n] for n in c}) == N_REQUIRED_MODELS for c in p)
            for p in partitions_after
        )
        Ks = [len(p) for p in partitions_after]
        # ARI over all pairs of seed partitions
        label_vecs = [labels_from_partition(p) for p in partitions_after]
        if len(label_vecs) > 1:
            ari_scores = [adjusted_rand_score(a, b) for a, b in combinations(label_vecs, 2)]
            ari_mean = float(np.mean(ari_scores))
        else:
            ari_mean = 1.0
        # modularity from the most stable run (best ARI to others) — used as separation tiebreaker
        if len(label_vecs) > 1:
            agreements = []
            for i, a in enumerate(label_vecs):
                others = [adjusted_rand_score(a, b) for j, b in enumerate(label_vecs) if j != i]
                agreements.append(np.mean(others))
            best_idx = int(np.argmax(agreements))
        else:
            best_idx = 0
        best_part = partitions_after[best_idx]
        mod = float(modularity(G, best_part, weight="weight"))

        row = {
            "edge_threshold_rel": thr,
            "resolution": res,
            "K_min": int(min(Ks)),
            "K_max": int(max(Ks)),
            "K_modal": int(max(set(Ks), key=Ks.count)),
            "ari_mean": ari_mean,
            "merges_min": int(min(merges_list)),
            "merges_mean": float(np.mean(merges_list)),
            "zero_merge_seeds": zero_merge_count,
            "satisfies_constraint": bool(all_ok),
            "modularity": mod,
        }
        sweep_rows.append(row)
        best_partitions[(thr, res)] = best_part
        print(
            f"thr={thr} res={res}: K={row['K_modal']} ARI={ari_mean:.3f} "
            f"merges_min={row['merges_min']} mod={mod:.3f} ok={all_ok}"
        )

if not sweep_rows:
    raise RuntimeError("No valid sweep configurations produced any edges; widen the sweep grid.")

with open(SWEEP_PATH, "w") as f:
    json.dump(sweep_rows, f, indent=2)
print(f"Saved sweep table to {SWEEP_PATH}")

# ----- selection rule (lexicographic, all unsupervised) -----
df = pd.DataFrame(sweep_rows)
chosen = choose_taxonomy_config(df)
print(f"\nChosen config: thr={chosen['edge_threshold_rel']} res={chosen['resolution']} "
      f"K={chosen['K_modal']} ARI={chosen['ari_mean']:.3f} mod={chosen['modularity']:.3f} "
      f"merges_min={chosen['merges_min']}")

chosen_partition = best_partitions[(chosen["edge_threshold_rel"], chosen["resolution"])]

# verify the hard constraint one more time on the saved partition
for cid, members in enumerate(chosen_partition):
    models_seen = {NODE_TO_MODEL[n] for n in members}
    assert len(models_seen) == N_REQUIRED_MODELS, f"community {cid} missing models: {set(ALL_MODELS) - models_seen}"

with open(PARTITION_PATH, "wb") as f:
    pkl.dump({
        "partition": [list(c) for c in chosen_partition],
        "edge_threshold_rel": float(chosen["edge_threshold_rel"]),
        "resolution": float(chosen["resolution"]),
        "K": int(chosen["K_modal"]),
        "ari_mean": float(chosen["ari_mean"]),
        "modularity": float(chosen["modularity"]),
    }, f)
print(f"Saved chosen partition to {PARTITION_PATH}")

# emit the points DataFrame consumed by label_taxonomy.py (same schema as kpartite_cluster.py)
rows = []
for cid, members in enumerate(chosen_partition):
    models_seen = sorted({m for m, _ in members})
    print(f"community {cid}: {len(members)} medoids across {len(models_seen)} models")
    for (model_name, c) in members:
        info = medoid_index[model_name]
        texts = info["cluster_texts"][c]
        pmcids = info["cluster_pmcid_and_sentence_idx"][c]
        for s, p in zip(texts, pmcids):
            pmcid = p[0] if isinstance(p, (tuple, list)) else None
            tax = p[1] if isinstance(p, (tuple, list)) and len(p) > 1 else p
            rows.append({
                "x": 0.0,
                "y": 0.0,
                "sentence": s,
                "taxonomy_index": tax,
                "pmcid": pmcid,
                "model": info["model"],
                "cluster": str(cid),
            })

df_points = pd.DataFrame(rows)
df_points.to_pickle(POINTS_PATH)
print(f"Saved {len(df_points)} sentence rows to {POINTS_PATH}")
