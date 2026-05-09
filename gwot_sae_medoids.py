import argparse
import os
import pickle as pkl
import numpy as np
from scipy.spatial.distance import pdist, squareform

from src.align_representations import Representation, AlignRepresentations, OptimizationConfig

# (n_clusters, layer) per model, must match sae_cluster_medoids.py
MODELS = [
    ("deepseek-r1-distill-qwen-1.5b", 12, 20),
    ("deepseek-r1-distill-llama-8b", 10, 10),
    ("deepseek-r1-distill-qwen-14b", 20, 12),
    ("huatuogpt-o1-8b", 14, 16),
    ("gpt-oss-20b", 14, 16),
    ("qwq-32b", 36, 10),
]
parser = argparse.ArgumentParser()
parser.add_argument("--centroids", action="store_true", help="Use centroids output from sae_cluster_medoids.py --centroids")
args = parser.parse_args()
REPR_TYPE = "centroids" if args.centroids else "medoids"

OUTPUT_ROOT = "/home/ttn/Development/med-interp/GWTune/results"
METRIC = "cosine"
MDS_DIM = 64
MAX_ITER = 200
TAG = f"sae_{REPR_TYPE}"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

print(f"Loading SAE {REPR_TYPE}...")
representations = []
medoid_index = {}
for model, layer, n_clusters in MODELS:
    name = f"{model}_layer{layer}_clusters{n_clusters}"
    in_path = os.path.join(OUTPUT_ROOT, f"sae_{REPR_TYPE}_{model}_layer{layer}_clusters{n_clusters}.pkl")
    with open(in_path, "rb") as f:
        d = pkl.load(f)

    medoids = d["medoid_activations"].astype(np.float32)
    n = len(medoids)

    # cosine RDM between cluster medoids; same metric used in gwot.py for sentences
    rdm = squareform(pdist(medoids, metric=METRIC))

    # MDS_dim cannot exceed n-1; small n_clusters cases (10, 12) get a smaller embedding
    rep = Representation(
        name=name,
        sim_mat=rdm,
        get_embedding=True,
        MDS_dim=min(MDS_DIM, max(2, n - 1)),
    )
    rep.cluster_texts = d["cluster_texts"]
    rep.cluster_pmcid_and_sentence_idx = d["cluster_pmcid_and_sentence_idx"]
    rep.cluster_members = d["cluster_members"]
    rep.object_labels = [f"{name}_c{i}" for i in range(n)]

    representations.append(rep)
    medoid_index[name] = {
        "model": model,
        "layer": layer,
        "n_clusters": n_clusters,
        "cluster_texts": d["cluster_texts"],
        "cluster_pmcid_and_sentence_idx": d["cluster_pmcid_and_sentence_idx"],
        "cluster_members": d["cluster_members"],
    }
    print(f"{name}: {n} medoids")

# same optimizer config style as gwot_centroids.py / gwot.py
config = OptimizationConfig(
    eps_list=[1e-2, 10.0],
    eps_log=True,
    num_trial=64,
    sinkhorn_method="sinkhorn_log",
    to_types="torch",
    device="cuda",
    data_type="double",
    n_jobs=3,
    optuna_n_jobs=1,
    multi_gpu=[0, 1, 2],
    db_params={"drivername": "sqlite"},
    init_mat_plan="random",
    n_iter=3,
    max_iter=MAX_ITER,
    max_total_trials=MAX_ITER,
    sampler_name="tpe",
    pruner_name="hyperband",
    pruner_params={"n_startup_trials": 1, "n_warmup_steps": 2, "min_resource": 2, "reduction_factor": 3},
    # study-level early stop: quit after this many completed trials without improving best GW loss
    optuna_early_stopping_patience=50,
    optuna_early_stopping_min_trials=20,
)

align_representation = AlignRepresentations(
    config=config,
    representations_list=representations,
    histogram_matching=False,
    metric=METRIC,
    main_results_dir=OUTPUT_ROOT,
    data_name=f"gwot_{TAG}",
)

print("Computing pairwise GW on SAE medoids...")
align_representation.gw_alignment(
    compute_OT=True,
    save_dataframe=True,
    change_sampler_seed=False,
    sampler_seed=42,
    fix_random_init_seed=True,
    first_random_init_seed=42,
    parallel_method="multithread",
)

# OT[src,tgt] is a (C_src, C_tgt) coupling; one matrix per ordered pair
ot_pairs = {}
for pairwise in align_representation.pairwise_list:
    key = f"{pairwise.source.name}_vs_{pairwise.target.name}"
    ot = np.asarray(pairwise.OT)
    ot_pairs[key] = ot
    print(f"{key}: OT shape {ot.shape}")

with open(os.path.join(OUTPUT_ROOT, f"centroid_ot_pairs_{TAG}.pkl"), "wb") as f:
    pkl.dump(ot_pairs, f)
with open(os.path.join(OUTPUT_ROOT, f"centroid_index_{TAG}.pkl"), "wb") as f:
    pkl.dump(medoid_index, f)
print(f"Saved OT pairs and medoid index ({TAG}).")
