import argparse
import os
import sys
import pickle as pkl
import numpy as np
import torch

# allow importing thinking-llms-interp.utils.* (not a real package)
THINKING_LLMS_ROOT = "/home/ttn/Development/med-interp/thinking-llms-interp"
sys.path.insert(0, THINKING_LLMS_ROOT)

from utils.sae import SAE  # noqa: E402

# (n_clusters, layer) per model, confirmed by user from saved SAE evaluator scores
MODELS = [
    ("deepseek-r1-distill-qwen-1.5b", 12, 20),
    ("deepseek-r1-distill-llama-8b", 10, 10),
    ("deepseek-r1-distill-qwen-14b", 20, 12),
    ("huatuogpt-o1-8b", 14, 16),
    ("gpt-oss-20b", 14, 16),
    ("qwq-32b", 36, 10),
]
ACTIVATIONS_ROOT = "/home/ttn/Development/med-interp/thinking-llms-interp/generate-responses/results/vars"
SAE_ROOT = "/home/ttn/Development/med-interp/thinking-llms-interp/train-saes/results/vars/saes"
OUTPUT_ROOT = "/home/ttn/Development/med-interp/GWTune/results"
N_EXAMPLES = 100000
BATCH_SIZE = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--centroids", action="store_true", help="Use cluster centroids (means) instead of medoids")
args = parser.parse_args()
USE_CENTROIDS = args.centroids
REPR_TYPE = "centroids" if USE_CENTROIDS else "medoids"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# load an SAE checkpoint into the SAE module shape used at train time
def load_sae(model, layer, n_clusters):
    sae_path = os.path.join(SAE_ROOT, f"sae_{model}_layer{layer}_clusters{n_clusters}.pt")
    ckpt = torch.load(sae_path, weights_only=False, map_location="cpu")
    sae = SAE(ckpt["input_dim"], ckpt["num_latents"], k=ckpt.get("topk", 3))
    sae.encoder.weight.data = ckpt["encoder_weight"]
    sae.encoder.bias.data = ckpt["encoder_bias"]
    sae.W_dec.data = ckpt["decoder_weight"]
    sae.b_dec.data = ckpt["b_dec"]
    return sae, ckpt

# argmax over encoder features matches train_clustering.py's sae_topk pathway exactly
def assign_clusters(sae, normalized_activations):
    sae = sae.to(DEVICE).eval()
    X = torch.from_numpy(normalized_activations).float()
    labels = np.empty(len(X), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            batch = X[i:i+BATCH_SIZE].to(DEVICE, non_blocking=True)
            enc = sae.encoder(batch - sae.b_dec)
            labels[i:i+len(batch)] = enc.argmax(dim=1).cpu().numpy()
    return labels

# medoid = member whose (normalized) activation has smallest cosine distance to the cluster mean
def cluster_medoid(member_indices, normalized_activations):
    if len(member_indices) == 1:
        return int(member_indices[0])
    M = normalized_activations[member_indices]
    mean = M.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-12)
    # rows are already L2-normalized so dot product = cosine similarity
    sims = M @ mean
    best = int(np.argmax(sims))
    return int(member_indices[best])

print(f"Computing SAE clusters and {REPR_TYPE} per model...")
for model, layer, n_clusters in MODELS:
    out_path = os.path.join(OUTPUT_ROOT, f"sae_{REPR_TYPE}_{model}_layer{layer}_clusters{n_clusters}.pkl")
    if os.path.exists(out_path):
        print(f"[skip] {out_path}")
        continue

    # raw activations + per-model running mean used to train the SAE
    activations_path = os.path.join(ACTIVATIONS_ROOT, f"activations_{model}_{N_EXAMPLES}_{layer}.pkl")
    print(f"Loading activations: {activations_path}")
    with open(activations_path, "rb") as f:
        activations, texts, pmcid_and_sentence_idx, mean_vector = pkl.load(f)

    # activations are already centered+L2-normalized at save time (utils.py:583), so use as-is
    normalized = np.asarray(activations, dtype=np.float32)

    sae, ckpt = load_sae(model, layer, n_clusters)
    print(f"Loaded SAE {model} layer={layer} clusters={n_clusters}")

    # recompute cluster labels and verify they match the labels saved during training
    labels = assign_clusters(sae, normalized)
    saved_labels = np.asarray(ckpt["cluster_labels"]).astype(np.int64)
    if len(saved_labels) == len(labels):
        match = float((labels == saved_labels).mean())
        print(f"Cluster-label agreement with checkpoint: {match:.4f}")
        assert match > 0.999, f"Reproduced cluster labels disagree with checkpoint for {model}"
    else:
        print(f"[warn] checkpoint cluster_labels length {len(saved_labels)} != recomputed {len(labels)}; skipping check")

    # group sentences by cluster and pick a medoid per cluster
    cluster_members = [np.where(labels == c)[0].astype(np.int64) for c in range(n_clusters)]
    cluster_texts = [[texts[i] for i in members] for members in cluster_members]
    cluster_pmcid_and_sentence_idx = [[pmcid_and_sentence_idx[i] for i in members] for members in cluster_members]

    repr_activations = np.zeros((n_clusters, normalized.shape[1]), dtype=np.float32)
    repr_indices = []  # only meaningful for medoids; -1 for centroids and dead clusters
    for c in range(n_clusters):
        members = cluster_members[c]
        if len(members) == 0:
            # dead cluster: fall back to decoder row (already unit-norm)
            print(f"[warn] empty cluster {c} for {model}; using decoder row")
            repr_indices.append(-1)
            repr_activations[c] = sae.W_dec.data[c].cpu().numpy()
            repr_activations[c] /= np.linalg.norm(repr_activations[c]) + 1e-12
            continue
        if USE_CENTROIDS:
            mean = normalized[members].mean(axis=0)
            mean /= np.linalg.norm(mean) + 1e-12
            repr_activations[c] = mean
            repr_indices.append(-1)
        else:
            idx = cluster_medoid(members, normalized)
            repr_indices.append(idx)
            repr_activations[c] = normalized[idx]

    sizes = [len(m) for m in cluster_members]
    print(f"{model}: cluster sizes min={min(sizes)} max={max(sizes)} median={int(np.median(sizes))}")

    with open(out_path, "wb") as f:
        pkl.dump({
            "model": model,
            "layer": layer,
            "n_clusters": n_clusters,
            "repr_type": REPR_TYPE,
            "medoid_activations": repr_activations,  # keep key name for gwot_sae_medoids.py compatibility
            "medoid_sentence_indices": np.asarray(repr_indices, dtype=np.int64),
            "cluster_members": cluster_members,
            "cluster_texts": cluster_texts,
            "cluster_pmcid_and_sentence_idx": cluster_pmcid_and_sentence_idx,
            "mean_vector": mean_vector,
        }, f)
    print(f"Saved {out_path}")

    # release GPU memory before the next model
    del sae, normalized, activations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("Done.")
