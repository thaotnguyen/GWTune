import json
import os
import pickle as pkl
import numpy as np
from sklearn.cluster import HDBSCAN

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

os.makedirs(OUTPUT_ROOT, exist_ok=True)

with open(os.path.join(OUTPUT_ROOT, "hdbscan_params.json")) as f:
    params = json.load(f)

for model, layer in MODELS:
    name = f"{model}_layer{layer}"
    out_path = os.path.join(OUTPUT_ROOT, f"hdbscan_{model}_layer{layer}.pkl")
    if os.path.exists(out_path):
        print(f"Skipping {name}, exists.")
        continue
    cs = params[name]["min_cluster_size"]
    ms = params[name].get("min_samples")
    print(f"Loading {name}...")
    with open(os.path.join(ACTIVATIONS_ROOT, f"activations_{model}_100000_{layer}.pkl"), "rb") as f:
        activations, texts, pmcid_and_sentence_idx, _ = pkl.load(f)
    activations = activations.astype(np.float32)
    print(f"Running HDBSCAN (min_cluster_size={cs}, min_samples={ms})...")
    labels = HDBSCAN(min_cluster_size=cs, min_samples=ms, metric="cosine", n_jobs=-1).fit_predict(activations)
    n_noise = int((labels == -1).sum())
    cluster_ids = sorted(set(labels) - {-1})
    centroids = np.stack([activations[labels == c].mean(axis=0) for c in cluster_ids])
    cluster_members = [np.where(labels == c)[0].tolist() for c in cluster_ids]
    cluster_texts = [[texts[i] for i in m] for m in cluster_members]
    cluster_pmcid_and_sentence_idx = [[pmcid_and_sentence_idx[i] for i in m] for m in cluster_members]
    print(f"{name}: {len(cluster_ids)} clusters, noise {n_noise}/{len(labels)} ({n_noise/len(labels):.2%})")
    with open(out_path, "wb") as f:
        pkl.dump({
            "model": model,
            "layer": layer,
            "centroids": centroids,
            "cluster_members": cluster_members,
            "cluster_texts": cluster_texts,
            "cluster_pmcid_and_sentence_idx": cluster_pmcid_and_sentence_idx,
        }, f)
    print(f"Saved {out_path}")
