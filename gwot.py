import os
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import pdist, squareform

from src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig

# Models and their best layers, according to Venhoff's method
MODELS = [
    ("deepseek-r1-distill-qwen-1.5b", 12),
    ("deepseek-r1-distill-llama-8b", 14),
    ("deepseek-r1-distill-qwen-14b", 14),
    ("gpt-oss-20b", 8),
    ("huatuogpt-o1-8b", 10),
    ("qwq-32b", 36),
]
ACTIVATIONS_ROOT = "/home/ttn/Development/med-interp/thinking-llms-interp/generate-responses/results/vars"
MDS_DIM = 128

n_representations = len(MODELS)

def pairwise_dists(X):
    D2 = euclidean_distances(X)
    D2 = 0.5 * (D2 + D2.T) # make distance matrix symmetric, otherwise you get rounding errors that cause a downstream error
    np.fill_diagonal(D2, 0.0)
    return np.sqrt(D2)

print("Loading activations...")
representations = []
for model, layer in MODELS:
    name = f"{model}_layer{layer}"
    activations_path = os.path.join(ACTIVATIONS_ROOT, f"activations_{model}_100000_{layer}.pkl")
    with open(activations_path, "rb") as f:
        activations, texts, pmcid_and_sentence_idx, mean_vector = pkl.load(f)
    if os.path.exists(os.path.join(ACTIVATIONS_ROOT, f"rdm_{model}_layer{layer}.npy")):
        sim_mat = np.load(os.path.join(ACTIVATIONS_ROOT, f"rdm_{model}_layer{layer}.npy"))
        print(f"Loaded RDM for {model}.")
    else:
        print(f"Calculating RDM for {model} (n={activations.shape[0]})...")
        sim_mat = pairwise_dists(activations.astype(np.float32))
        np.save(os.path.join(ACTIVATIONS_ROOT, f"rdm_{model}_layer{layer}.npy"), sim_mat)
        print(f"Saved RDM for {model}.")
    representation = Representation(
        name=name,
        sim_mat=sim_mat,
        embedding=None,
        get_embedding=False,
        MDS_dim=MDS_DIM,
    )
    representations.append(representation)

config = OptimizationConfig(    
    eps_list = [1e-2, 10.0],
    eps_log = True,
    num_trial = 100,
    sinkhorn_method="sinkhorn_log", 
    to_types = "torch",
    device = "cuda",
    data_type = "double",
    n_jobs = 3,
    multi_gpu = [0,1,2],
    init_mat_plan = "random",
    n_iter = 1,
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
    metric="cosine", 
    main_results_dir = "./results/",
)

visualize_config = VisualizationConfig(
    show_figure = False,
    fig_ext="pdf",
)

print("Aligning representations...")
ot_list = align_representation.barycenter_alignment(
        return_data = False,
        return_figure = False,
        visualization_config = visualize_config,
        save_dataframe=True,
        change_sampler_seed=False, 
        sampler_seed = 42, 
        fix_random_init_seed = True,
        first_random_init_seed = 42,
        parallel_method="multithread",
    )

align_representation.RSA_get_corr(metric = "pearson")

print(f"RSA correlation: {align_representation.RSA_corr}")

## Calculate the accuracy based on the OT plan. 
align_representation.calc_accuracy(top_k_list = [1, 5, 10], eval_type = "ot_plan")
align_representation.plot_accuracy(eval_type = "ot_plan", scatter = True)

top_k_accuracy = align_representation.top_k_accuracy
print(f"Top k accuracy: {top_k_accuracy}")
