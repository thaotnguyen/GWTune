"""Label every per-model GWOT-barycenter activation with the single
universal-taxonomy SAE trained by run_gwot.sh.

Reads:
  ../thinking-llms-interp/train-saes/results/vars/
    sae_topk_results_gwot_barycenter_layer0_gwot.json   (best k)
    saes/sae_gwot_barycenter_layer0_clusters{k}_gwot.pt  (best SAE)
  GWTune/results/all_barycenter_{model}_layer{L}_n4096_cosine_mds64.pkl
  ../thinking-llms-interp/generate-responses/results/vars/responses_{model}.json

Writes:
  ../white_box_data/{model}/all/results_gwot_sae.labeled.json
"""
import json
import os
import pickle as pkl
import re
import sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "thinking-llms-interp"))
from utils.sae import SAE  # noqa: E402

MODELS = [
    ("deepseek-r1-distill-qwen-1.5b", 12),
    ("deepseek-r1-distill-llama-8b", 14),
    ("deepseek-r1-distill-qwen-14b", 14),
    ("gpt-oss-20b", 8),
    ("huatuogpt-o1-8b", 10),
    ("qwq-32b", 36),
]
N_SAMPLE = 4096
METRIC = "cosine"
MDS_DIM = 64

GWOT_MODEL_ID = "gwot_barycenter"
GWOT_LAYER = 0
GWOT_GRANULARITY = "gwot"

TRAIN_SAES_VARS = os.path.join(REPO_ROOT, "thinking-llms-interp", "train-saes", "results", "vars")
RESULTS_DIR = os.path.join(HERE, "results")
RESPONSES_DIR = os.path.join(REPO_ROOT, "thinking-llms-interp", "generate-responses", "results", "vars")
WHITE_BOX_DIR = os.path.join(REPO_ROOT, "white_box_data")


def load_best_sae():
    """Return (sae_module, mean_vector, categories) for the best k."""
    results_json = os.path.join(
        TRAIN_SAES_VARS,
        f"sae_topk_results_{GWOT_MODEL_ID}_layer{GWOT_LAYER}_{GWOT_GRANULARITY}.json",
    )
    with open(results_json) as f:
        results = json.load(f)
    best_k = int(results["best_cluster"]["size"])
    print(f"Best k from eval: {best_k}  (avg_final_score={results['best_cluster'].get('avg_final_score')})")

    by_size = results["results_by_cluster_size"][str(best_k)]
    # The titles pipeline writes one rep per cluster size when REPETITIONS=1.
    rep = by_size["all_results"][0]
    categories = rep["categories"]   # list of [id, name, definition]

    sae_path = os.path.join(
        TRAIN_SAES_VARS, "saes",
        f"sae_{GWOT_MODEL_ID}_layer{GWOT_LAYER}_clusters{best_k}_{GWOT_GRANULARITY}.pt",
    )
    checkpoint = torch.load(sae_path, weights_only=False)
    sae = SAE(checkpoint["input_dim"], checkpoint["num_latents"], k=checkpoint.get("topk", 3))
    sae.encoder.weight.data = checkpoint["encoder_weight"]
    sae.encoder.bias.data = checkpoint["encoder_bias"]
    sae.W_dec.data = checkpoint["decoder_weight"]
    sae.b_dec.data = checkpoint["b_dec"]
    sae.eval()

    mean_vector = checkpoint.get("mean_vector")
    if mean_vector is None:
        raise RuntimeError("SAE checkpoint is missing mean_vector — cannot center activations.")
    if isinstance(mean_vector, np.ndarray):
        mean_vector = torch.from_numpy(mean_vector)
    mean_vector = mean_vector.to(torch.float32).flatten()
    print(f"Loaded SAE from {sae_path}; input_dim={checkpoint['input_dim']} k={checkpoint['num_latents']}")
    return sae, mean_vector, categories


@torch.no_grad()
def encode_to_cluster_ids(sae, mean_vector, activations, batch_size=4096, device="cuda"):
    """Replicate annotate_thinking.py L191-204 exactly, vectorized."""
    sae = sae.to(device)
    mean_vector = mean_vector.to(device)
    out = np.empty(activations.shape[0], dtype=np.int64)
    for start in range(0, activations.shape[0], batch_size):
        batch = torch.from_numpy(activations[start:start + batch_size]).to(device).to(torch.float32)
        batch = batch - mean_vector
        batch = batch / (batch.norm(dim=1, keepdim=True) + 1e-8)
        batch = batch - sae.b_dec
        latents = sae.encoder(batch)
        out[start:start + batch_size] = latents.argmax(dim=1).cpu().numpy()
    return out


def extract_diagnosis(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not matches:
        return ""
    last = matches[-1]
    if "<answer>" in last:
        last = last.split("<answer>")[-1]
    return last.strip()


def write_json_atomic(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def label_one_model(model: str, layer: int, sae, mean_vector, categories, taxonomy_blocks):
    bary_path = os.path.join(
        RESULTS_DIR,
        f"all_barycenter_{model}_layer{layer}_n{N_SAMPLE}_{METRIC}_mds{MDS_DIM}.pkl",
    )
    with open(bary_path, "rb") as f:
        activations, texts, pmcid_and_sentence_idx, _ = pkl.load(f)
    activations = np.asarray(activations, dtype=np.float32)
    print(f"[{model}] {activations.shape[0]} rows from {os.path.basename(bary_path)}")

    cluster_ids = encode_to_cluster_ids(
        sae, mean_vector, activations,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Build per-pmcid label lists.
    by_pmcid = defaultdict(list)   # pmcid -> [(sentence_idx, text, cluster_id), ...]
    for i, (pmcid, sidx) in enumerate(pmcid_and_sentence_idx):
        by_pmcid[str(pmcid)].append((int(sidx), texts[i], int(cluster_ids[i])))

    # Load responses for case-level metadata.
    responses_path = os.path.join(RESPONSES_DIR, f"responses_{model}.json")
    with open(responses_path) as f:
        responses = json.load(f)
    resp_by_pmcid = {str(r.get("pmcid")): r for r in responses}

    state_order = taxonomy_blocks["state_order"]
    state_to_idx = taxonomy_blocks["state_to_idx"]
    id_to_state = taxonomy_blocks["id_to_state"]   # cluster_id (int) -> state name
    id_to_universal = taxonomy_blocks["id_to_universal"]  # int -> "Ux"

    traces = []
    for sample_index, (pmcid, rows) in enumerate(by_pmcid.items()):
        rows.sort(key=lambda r: r[0])
        resp = resp_by_pmcid.get(pmcid, {})
        full_response = resp.get("full_response", "") or ""
        true_diagnosis = resp.get("gold_answer", "") or ""
        predicted_diagnosis = extract_diagnosis(full_response)

        label_json = {}
        sequence = []
        original_sequence_idx = []
        for j, (_sidx, text, cid) in enumerate(rows, start=1):
            state_name = id_to_state[cid]
            label_json[str(j)] = {"function": state_name, "sentence": text}
            sequence.append(state_to_idx[state_name])
            original_sequence_idx.append(str(cid))

        verified_correct = (
            bool(true_diagnosis)
            and (
                predicted_diagnosis.lower() == true_diagnosis.lower()
                or true_diagnosis.lower() in predicted_diagnosis.lower()
            )
        )

        traces.append({
            "pmcid": pmcid,
            "question_id": resp.get("question_id"),
            "sample_index": sample_index,
            "case_prompt": (resp.get("original_message") or {}).get("content", ""),
            "true_diagnosis": true_diagnosis,
            "predicted_diagnosis": predicted_diagnosis,
            "reasoning_trace": full_response,
            "verification_response": "",
            "verified_correct": verified_correct,
            "label_json": label_json,
            "sequence": sequence,
            "sequence_length": len(sequence),
            "original_sequence_idx": original_sequence_idx,
        })

    output = {
        "state_order": state_order,
        "state_to_idx": state_to_idx,
        "idx_to_universal": {str(cid): id_to_universal[cid] for cid in sorted(id_to_universal)},
        "universal_taxonomy": {id_to_universal[cid]: id_to_state[cid] for cid in sorted(id_to_state)},
        "traces": traces,
    }
    out_path = os.path.join(WHITE_BOX_DIR, model, "all", "results_gwot_sae.labeled.json")
    write_json_atomic(out_path, output)
    print(f"[{model}] wrote {out_path}  ({len(traces)} traces)")


def build_taxonomy_blocks(categories):
    """categories: list of [id, name, definition]. Returns shared taxonomy dicts."""
    # Sort by integer cluster id for stable ordering.
    sorted_cats = sorted(categories, key=lambda c: int(c[0]))
    id_to_state = {int(c[0]): c[1] for c in sorted_cats}
    state_order = [id_to_state[cid] for cid in sorted(id_to_state)]
    state_to_idx = {name: i for i, name in enumerate(state_order)}
    id_to_universal = {cid: f"U{cid + 1}" for cid in id_to_state}
    return {
        "state_order": state_order,
        "state_to_idx": state_to_idx,
        "id_to_state": id_to_state,
        "id_to_universal": id_to_universal,
    }


def main() -> None:
    sae, mean_vector, categories = load_best_sae()
    taxonomy_blocks = build_taxonomy_blocks(categories)
    print(f"Universal taxonomy ({len(taxonomy_blocks['state_order'])} buckets):")
    for cid in sorted(taxonomy_blocks["id_to_state"]):
        print(f"  {taxonomy_blocks['id_to_universal'][cid]}: {taxonomy_blocks['id_to_state'][cid]}")

    for model, layer in MODELS:
        label_one_model(model, layer, sae, mean_vector, categories, taxonomy_blocks)


if __name__ == "__main__":
    main()
