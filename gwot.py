import os
import json
import numpy as np
import pandas as pd
import pickle as pkl
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

from src.align_representations import Representation, AlignRepresentations, OptimizationConfig
from src.barycenter_projection import (
    cosine_distance_diagnostics,
    orthogonal_map,
    project_to_barycenter_space,
    save_projected_activation_pickle,
    validate_projection,
)

# models and their best layers, according to venhoff's method
MODELS = [
    ("deepseek-r1-distill-qwen-1.5b", 12),
    ("deepseek-r1-distill-llama-8b", 14),
    ("deepseek-r1-distill-qwen-14b", 14),
    ("gpt-oss-20b", 8),
    ("huatuogpt-o1-8b", 10),
    ("qwq-32b", 36),
]
ACTIVATIONS_ROOT = "/home/ttn/Development/med-interp/thinking-llms-interp/generate-responses/results/vars"
OUTPUT_ROOT = "/home/ttn/Development/med-interp/GWTune/results"
MDS_DIM = 64
N_SAMPLE = 4096
N_PCT_BINS = 10
SAMPLE_SEED = 42
METRIC = "cosine"
PROJECTION_BATCH_SIZE = 1024
DIAGNOSTIC_PAIRS = 50_000
DIAGNOSTIC_SEED = 42
BARYCENTER_USE_GPU = True
BARYCENTER_GPU_DEVICES = [0, 1, 2]
BARYCENTER_GPU_REG = None
BARYCENTER_GPU_REG_SCALE = 1e-2
BARYCENTER_GPU_NUMITERMAX = 64000
BARYCENTER_GPU_STOPTHR = 1e-8
BARYCENTER_GPU_PARALLEL = True

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def pairwise_dists(X, metric=METRIC):
    D2 = pdist(X, metric=METRIC)
    return squareform(D2)

# subsample activations evenly from pmcids and from sentence idx deciles
def stratified_subsample(pmcid_and_sentence_idx, n_sample, n_pct_bins, seed):
    df = pd.DataFrame(data=pmcid_and_sentence_idx, columns=["pmcid", "sentence_idx"])
    max_idx = df.groupby("pmcid")["sentence_idx"].transform("max")
    pct = np.where(max_idx > 0, df["sentence_idx"] / max_idx.replace(0, 1), 0.0)
    bins = np.minimum((pct * n_pct_bins).astype(int), n_pct_bins - 1)
    df["cell"] = list(zip(df["pmcid"], bins))

    rng = np.random.default_rng(seed)
    cells = list(df.groupby("cell").indices.items())
    rng.shuffle(cells)
    pools = [list(rng.permutation(idxs)) for _, idxs in cells]

    chosen = []
    while len(chosen) < n_sample and any(pools):
        for pool in pools:
            if not pool:
                continue
            chosen.append(pool.pop())
            if len(chosen) >= n_sample:
                break
    return np.array(sorted(chosen), dtype=np.int64)

print("Loading activations...")
representations = []
activation_records = []
for model, layer in MODELS:
    name = f"{model}_layer{layer}"
    activations_path = os.path.join(ACTIVATIONS_ROOT, f"activations_{model}_100000_{layer}.pkl")
    with open(activations_path, "rb") as f:
        activations, texts, pmcid_and_sentence_idx, mean_vector = pkl.load(f)

    sample_idx = stratified_subsample(pmcid_and_sentence_idx, N_SAMPLE, N_PCT_BINS, SAMPLE_SEED)
    sample_idx_check = stratified_subsample(pmcid_and_sentence_idx, N_SAMPLE, N_PCT_BINS, SAMPLE_SEED)
    if not np.array_equal(sample_idx, sample_idx_check):
        raise RuntimeError(f"Subsampling is not deterministic for {model}.")
    sampled_activations = activations[sample_idx]
    sampled_texts = [texts[i] for i in sample_idx]
    sampled_pmcid_and_sentence_idx = [pmcid_and_sentence_idx[i] for i in sample_idx]
    activation_records.append(
        {
            "model": model,
            "layer": layer,
            "name": name,
            "activations_path": activations_path,
            "sample_idx": sample_idx,
        }
    )

    rdm_path = os.path.join(OUTPUT_ROOT, f"rdm_{model}_layer{layer}_n{N_SAMPLE}_{METRIC}.npy")
    if os.path.exists(rdm_path):
        rdm = np.load(rdm_path)
        print(f"Loaded RDM for {model}, metric={METRIC}, subsampled to n={N_SAMPLE}.")
    else:
        print(f"Calculating RDM for {model}, metric={METRIC}, subsampled to n={N_SAMPLE}.")
        rdm = pairwise_dists(sampled_activations.astype(np.float32), metric=METRIC)
        np.save(rdm_path, rdm)
        print(f"Saved RDM for {model}, metric={METRIC}, subsampled to n={N_SAMPLE}.")
    if os.path.exists(os.path.join(OUTPUT_ROOT, f"representation_{model}_layer{layer}_n{N_SAMPLE}_{METRIC}.pkl")):
        with open(os.path.join(OUTPUT_ROOT, f"representation_{model}_layer{layer}_n{N_SAMPLE}_{METRIC}.pkl"), "rb") as f:
            representation = pkl.load(f)
        print(f"Loaded representation for {model}, metric={METRIC}, subsampled to n={N_SAMPLE}.")
    else:
        print(f"Building representation for {model}, metric={METRIC}, subsampled to n={N_SAMPLE}...")
        representation = Representation(
            name=name,
            sim_mat=rdm,
            get_embedding=True,
            MDS_dim=MDS_DIM,
        )
        with open(os.path.join(OUTPUT_ROOT, f"representation_{model}_layer{layer}_n{N_SAMPLE}_{METRIC}.pkl"), "wb") as f:
            pkl.dump(representation, f)
        print(f"Saved representation for {model}, metric={METRIC}, subsampled to n={N_SAMPLE}.")
    representation.object_labels = sampled_texts
    representation.pmcid_and_sentence_idx = sampled_pmcid_and_sentence_idx
    representations.append(representation)

config = OptimizationConfig(    
    eps_list = [1e-2, 10.0],
    eps_log = True,
    num_trial = 32,
    sinkhorn_method="sinkhorn_log", 
    to_types = "torch",
    device = "cuda",
    data_type = "double",
    n_jobs = 48,
    optuna_n_jobs = 3,
    multi_gpu = [0,1,2],
    db_params = {"drivername": "sqlite"},
    init_mat_plan = "random",
    n_iter = 3,
    max_iter = 200,
    sampler_name = "tpe",
    pruner_name = "hyperband",
    pruner_params = {"n_startup_trials": 1, 
                     "n_warmup_steps": 2, 
                     "min_resource": 2, 
                     "reduction_factor" : 3
                    },
)

align_representation = AlignRepresentations(
    config=config,
    representations_list=representations,   
    histogram_matching=False,
    metric=METRIC,
    main_results_dir=OUTPUT_ROOT,
    data_name=f"gwot_{METRIC}_n{N_SAMPLE}",
)

print("Computing pairwise GW...")
align_representation.gw_alignment(
    compute_OT=True,
    save_dataframe=True,
    change_sampler_seed=False,
    sampler_seed=42,
    fix_random_init_seed=True,
    first_random_init_seed=42,
    parallel_method="multithread",
)

# # choose the pivot with the lowest mean gw/wasserstein distance to others
# N = len(representations)
# name_to_idx = {r.name: i for i, r in enumerate(representations)}
# D = np.full((N, N), np.nan, dtype=np.float64)
# for pairwise in align_representation.pairwise_list:
#     i = name_to_idx[pairwise.source.name]
#     j = name_to_idx[pairwise.target.name]
#     D[i, j] = D[j, i] = float(pairwise.study.best_value)
# pivot = int(np.nanmean(D, axis=1).argmin())
# print(f"Pivot for barycenter alignment: {MODELS[pivot][0]}")

# ot_list_path = os.path.join(OUTPUT_ROOT, f"ot_list_{METRIC}_n{N_SAMPLE}.pkl")
# aligned_sampled_path = os.path.join(OUTPUT_ROOT, f"aligned_sampled_embeddings_{METRIC}_n{N_SAMPLE}_mds{MDS_DIM}.pkl")
# sampled_embeddings_before = [representation.embedding.copy() for representation in representations]

# if os.path.exists(ot_list_path) and os.path.exists(aligned_sampled_path):
#     with open(ot_list_path, "rb") as f:
#         ot_list = pkl.load(f)
#     with open(aligned_sampled_path, "rb") as f:
#         aligned_sampled_embeddings = pkl.load(f)
#     for representation, embedding in zip(representations, aligned_sampled_embeddings):
#         representation.embedding = embedding.copy()
#     print("Loaded OT list and aligned sampled embeddings...")
# else:
#     print("Aligning representations...")
#     ot_list = align_representation.barycenter_alignment(
#         pivot=pivot,
#         n_iter=16,
#         return_data=True,
#         use_gpu=BARYCENTER_USE_GPU,
#         gpu_devices=BARYCENTER_GPU_DEVICES,
#         gpu_reg=BARYCENTER_GPU_REG,
#         gpu_reg_scale=BARYCENTER_GPU_REG_SCALE,
#         gpu_numItermax=BARYCENTER_GPU_NUMITERMAX,
#         gpu_stopThr=BARYCENTER_GPU_STOPTHR,
#         gpu_parallel=BARYCENTER_GPU_PARALLEL,
#     )
#     aligned_sampled_embeddings = [representation.embedding.copy() for representation in representations]
#     with open(ot_list_path, "wb") as f:
#         pkl.dump(ot_list, f)
#     with open(aligned_sampled_path, "wb") as f:
#         pkl.dump(aligned_sampled_embeddings, f)
#     print(f"Saved OT list to {ot_list_path}.")
#     print(f"Saved aligned sampled embeddings to {aligned_sampled_path}.")

# print("Projecting all original vectors to barycenter space...")
# diagnostics = {}
# for record, sampled_before, aligned_sampled in zip(activation_records, sampled_embeddings_before, aligned_sampled_embeddings):
#     model = record["model"]
#     layer = record["layer"]
#     sample_idx = record["sample_idx"]
#     output_path = os.path.join(
#         OUTPUT_ROOT,
#         f"all_barycenter_{model}_layer{layer}_n{N_SAMPLE}_{METRIC}_mds{MDS_DIM}.pkl",
#     )

#     with open(record["activations_path"], "rb") as f:
#         activations, texts, pmcid_and_sentence_idx, mean_vector = pkl.load(f)

#     q = orthogonal_map(sampled_before, aligned_sampled)
#     projected = project_to_barycenter_space(
#         activations=activations.astype(np.float32),
#         landmark_activations=activations[sample_idx].astype(np.float32),
#         sample_idx=sample_idx,
#         landmark_mds_embedding=sampled_before,
#         aligned_landmark_embedding=aligned_sampled,
#         q=q,
#         batch_size=PROJECTION_BATCH_SIZE,
#     )
#     checks = validate_projection(q, projected, sample_idx, aligned_sampled)
#     preservation = cosine_distance_diagnostics(
#         original=activations.astype(np.float32),
#         projected=projected,
#         n_pairs=DIAGNOSTIC_PAIRS,
#         seed=DIAGNOSTIC_SEED,
#     )
#     save_projected_activation_pickle(output_path, projected, texts, pmcid_and_sentence_idx, model)
#     diagnostics[record["name"]] = {
#         "output_path": output_path,
#         **checks,
#         **preservation,
#     }
#     print(f"Saved full barycenter projection for {model} to {output_path}.")

# diagnostics_path = os.path.join(
#     OUTPUT_ROOT,
#     f"barycenter_projection_quality_{METRIC}_n{N_SAMPLE}_mds{MDS_DIM}.json",
# )
# with open(diagnostics_path, "w") as f:
#     json.dump(diagnostics, f, indent=2)
# print(f"Saved projection diagnostics to {diagnostics_path}.")
