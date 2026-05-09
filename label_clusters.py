import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


N_SENTENCES = 50
T = 10
MODEL = "deepseek-v4-pro"
MAX_TOKENS = 16384
BASE_URL = "https://api.deepseek.com"
RETRY_BACKOFF_SECONDS = (2, 5, 10)
SEED = 42
RESULTS_ROOT = "/home/ttn/Development/med-interp/GWTune/results"
TAG = "sentences_n4096_cosine"
POINTS_PATH = os.path.join(RESULTS_ROOT, f"kpartite_cluster_points_{TAG}.pkl")
OUTPUT_PATH = os.path.join(RESULTS_ROOT, f"kpartite_cluster_labels_{TAG}.json")


LABEL_JSON_INSTRUCTIONS = (
    "Return JSON only. The JSON object must have exactly this shape: "
    '{"labels":[{"cluster":-1,"title":"Residual statements","description":"..."}]}. '
    "Do not include markdown fences, prose, or extra keys."
)


def cluster_sort_key(cluster):
    return int(cluster)


def sample_sentences(sentences, n_sentences, rng):
    sentences = list(sentences)
    if len(sentences) <= n_sentences:
        return sentences
    idx = np.sort(rng.choice(len(sentences), size=n_sentences, replace=False))
    return [sentences[i] for i in idx]


def build_round_payload(df, round_index, seed=SEED, n_sentences=N_SENTENCES):
    rng = np.random.default_rng(seed + round_index)
    clusters = sorted(df["cluster"].astype(str).unique(), key=cluster_sort_key)
    grouped = {
        int(cluster): df.loc[df["cluster"].astype(str) == cluster, "sentence"].dropna().astype(str).tolist()
        for cluster in clusters
    }
    counterexample_budget = max(1, n_sentences)
    items = []
    for cluster in sorted(grouped):
        other_clusters = [other for other in sorted(grouped) if other != cluster]
        per_other = max(1, counterexample_budget // max(1, len(other_clusters)))
        items.append(
            {
                "cluster": cluster,
                "examples": sample_sentences(grouped[cluster], n_sentences, rng),
                "counterexamples": [
                    {
                        "cluster": other,
                        "sentences": sample_sentences(grouped[other], per_other, rng),
                    }
                    for other in other_clusters
                ],
            }
        )
    return {"round": round_index, "clusters": items}


def build_round_messages(payload):
    return [
        {
            "role": "system",
            "content": (
                "You label clusters of sentences from LLM reasoning traces. "
                f"{LABEL_JSON_INSTRUCTIONS}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Every cluster contains sentences from LLM reasoning traces grouped by similar role or cognitive "
                "function. Label every cluster in one joint pass. Use examples to identify the represented reasoning "
                "process, and use counterexamples to keep categories contrastive and semantically orthogonal.\n\n"
                "For each cluster, produce one title and one description. Titles must be a crisp single concept, "
                "a simple noun or verb phrase, and must not contain slashes, parentheses, or compound labels. "
                "Descriptions must be 3-4 sentences, explain the reasoning process represented, explain what is "
                "included, explain what is not included using counterexamples, mention common words or phrases from "
                "the examples, and be precise enough to classify new examples reliably.\n\n"
                f"Round payload:\n{json.dumps(payload, ensure_ascii=False)}"
            ),
        },
    ]


def parse_tool_response(response):
    return parse_label_response(response)


def parse_label_response(response, expected_clusters=None):
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("DeepSeek response did not include JSON content.")
    try:
        args = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"DeepSeek response was not valid JSON: {exc}") from exc
    if not isinstance(args, dict) or not isinstance(args.get("labels"), list):
        raise ValueError("DeepSeek response must be a JSON object with a labels list.")

    labels = {}
    for item in args["labels"]:
        if not isinstance(item, dict):
            raise ValueError("Each label must be a JSON object.")
        cluster = item.get("cluster")
        title = item.get("title")
        description = item.get("description")
        if not isinstance(cluster, int):
            raise ValueError("Each label must include an integer cluster.")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Each label must include a non-empty title.")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Each label must include a non-empty description.")
        labels[str(item["cluster"])] = {
            "title": title,
            "description": description,
        }
    if expected_clusters is not None:
        expected = {str(cluster) for cluster in expected_clusters}
        actual = set(labels)
        missing = sorted(expected - actual, key=cluster_sort_key)
        extra = sorted(actual - expected, key=cluster_sort_key)
        if missing:
            raise ValueError(f"DeepSeek response missing labels for clusters: {missing}")
        if extra:
            raise ValueError(f"DeepSeek response included unexpected labels for clusters: {extra}")
    return labels


def response_metadata(response):
    metadata = {}
    choices = getattr(response, "choices", None) or []
    if choices:
        metadata["finish_reason"] = getattr(choices[0], "finish_reason", None)
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            metadata["content_preview"] = content[:500]
        else:
            metadata["content_preview"] = repr(content)
    metadata["response_id"] = getattr(response, "id", None)
    metadata["model"] = getattr(response, "model", None)
    usage = getattr(response, "usage", None)
    if usage is not None:
        metadata["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    return metadata


def format_deepseek_failure(context, attempts, error, response=None):
    parts = [
        f"DeepSeek labeling failed for {context}",
        f"attempts={attempts}",
        f"error={error}",
    ]
    if response is not None:
        for key, value in response_metadata(response).items():
            if value is not None:
                parts.append(f"{key}={value}")
    return "; ".join(parts)


def label_payload_with_retries(
    messages,
    expected_clusters,
    context,
    max_attempts=4,
    sleep_fn=time.sleep,
):
    last_response = None
    backoffs = list(RETRY_BACKOFF_SECONDS)
    for attempt in range(1, max_attempts + 1):
        try:
            last_response = call_deepseek(messages)
            return parse_label_response(last_response, expected_clusters=expected_clusters)
        except Exception as exc:
            if attempt >= max_attempts:
                raise ValueError(format_deepseek_failure(context, attempt, exc, last_response)) from exc
            sleep_for = backoffs[min(attempt - 1, len(backoffs) - 1)]
            sleep_fn(sleep_for)


def build_merge_payload(drafts):
    cluster_ids = sorted({cluster for draft in drafts for cluster in draft["labels"]}, key=cluster_sort_key)
    return {
        "clusters": [
            {
                "cluster": int(cluster),
                "drafts": [
                    {
                        "round": draft["round"],
                        "title": draft["labels"][cluster]["title"],
                        "description": draft["labels"][cluster]["description"],
                    }
                    for draft in drafts
                    if cluster in draft["labels"]
                ],
            }
            for cluster in cluster_ids
        ]
    }


def build_merge_messages(payload):
    return [
        {
            "role": "system",
            "content": (
                "You consolidate draft labels for clusters of LLM reasoning trace sentences. "
                f"{LABEL_JSON_INSTRUCTIONS}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Consolidate the draft labels for all clusters into one final contrastive label set. Keep titles "
                "as crisp single concepts with no slashes, parentheses, or compound labels. Make descriptions "
                "3-4 sentences and semantically orthogonal across clusters, preserving the evidence from drafts "
                "while resolving overlaps.\n\n"
                f"Draft labels:\n{json.dumps(payload, ensure_ascii=False)}"
            ),
        },
    ]


def deepseek_client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)


def call_deepseek(messages):
    client = deepseek_client()
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=MAX_TOKENS,
    )


def payload_cluster_ids(payload):
    key = "clusters"
    return [item["cluster"] for item in payload[key]]


def write_output(path, metadata, drafts, labels):
    output = {"metadata": metadata, "drafts": drafts, "labels": labels}
    write_json_atomic(path, output)


def write_json_atomic(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def checkpoint_path_for(output_path):
    return str(Path(output_path).with_suffix(".drafts.json"))


def load_checkpoint(path, metadata):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    if checkpoint.get("metadata") != metadata:
        return []
    drafts = checkpoint.get("drafts", [])
    if not isinstance(drafts, list):
        return []
    return sorted(drafts, key=lambda draft: draft["round"])


def write_checkpoint(path, metadata, drafts):
    write_json_atomic(path, {"metadata": metadata, "drafts": sorted(drafts, key=lambda draft: draft["round"])})


def run_labeling_pipeline(df, output_path, metadata, checkpoint_path=None):
    checkpoint_path = checkpoint_path or checkpoint_path_for(output_path)
    drafts = load_checkpoint(checkpoint_path, metadata)
    completed_rounds = {draft["round"] for draft in drafts}

    with ThreadPoolExecutor(max_workers=T) as pool:
        futures = {}
        for round_index in range(T):
            if round_index in completed_rounds:
                continue
            payload = build_round_payload(df, round_index=round_index)
            future = pool.submit(
                label_payload_with_retries,
                build_round_messages(payload),
                expected_clusters=payload_cluster_ids(payload),
                context=f"draft round {round_index + 1}/{T}",
            )
            futures[future] = (round_index, payload)

        for future in as_completed(futures):
            round_index, payload = futures[future]
            labels = future.result()
            drafts.append({"round": round_index, "payload": payload, "labels": labels})
            drafts = sorted(drafts, key=lambda draft: draft["round"])
            write_checkpoint(checkpoint_path, metadata, drafts)
            print(f"Completed draft round {round_index + 1}/{T}")

    drafts = sorted(drafts, key=lambda draft: draft["round"])
    write_checkpoint(checkpoint_path, metadata, drafts)

    merge_payload = build_merge_payload(drafts)
    final_labels = label_payload_with_retries(
        build_merge_messages(merge_payload),
        expected_clusters=payload_cluster_ids(merge_payload),
        context="final merge",
    )
    write_output(output_path, metadata, drafts, final_labels)
    return final_labels


def main():
    df = pd.read_pickle(POINTS_PATH)
    metadata = {
        "model": MODEL,
        "T": T,
        "N_SENTENCES": N_SENTENCES,
        "seed": SEED,
        "source_point_file": POINTS_PATH,
    }
    run_labeling_pipeline(
        df,
        OUTPUT_PATH,
        metadata,
    )
    print(f"Saved cluster labels to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
