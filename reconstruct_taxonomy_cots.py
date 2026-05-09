import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path


RESULTS_ROOT = Path("/home/ttn/Development/med-interp/GWTune/results")
RESPONSES_ROOT = Path("/home/ttn/Development/med-interp/thinking-llms-interp/generate-responses/results/vars")
DEFAULT_TAGS = ("sae_medoids", "sae_centroids")

_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>.*?</answer>", flags=re.IGNORECASE | re.DOTALL)
_FINAL_RESPONSE_RE = re.compile(r"(?im)^\s*##\s*Final\s+Response\s*$")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = data.get("labels", data)
    return {str(cluster): value for cluster, value in labels.items()}


def clean_text(text):
    text = _FINAL_RESPONSE_RE.sub("", text or "")
    text = text.replace("<think>", "").replace("</think>", "")
    text = _ANSWER_RE.sub("", text)
    return text.strip()


def extract_thinking_process(full_response):
    if not full_response:
        return ""
    candidates = [match.strip() for match in _THINK_BLOCK_RE.findall(full_response)]
    candidates = [value for value in candidates if value and value != "...your internal reasoning for the diagnosis..."]
    if candidates:
        return clean_text(max(candidates, key=len))
    return clean_text(full_response)


def regex_sentence_split(text):
    text = text.strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def make_sentence_splitter(use_spacy=False):
    if not use_spacy:
        return regex_sentence_split
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return regex_sentence_split

    def split(text):
        doc = nlp(text or "")
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    return split


def source_index(value, fallback):
    try:
        return int(value)
    except Exception:
        return int(fallback)


def pmcid_from_pair(value):
    if isinstance(value, (tuple, list)) and len(value) >= 1:
        return value[0]
    return None


def sentence_index_from_pair(value):
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return value[1]
    return value


def cluster_for_nodes(partition):
    mapping = {}
    for cid, members in enumerate(partition["partition"]):
        for node in members:
            model_key, cluster_index = node
            mapping[(model_key, int(cluster_index))] = str(cid)
    return mapping


def flatten_taxonomy_sentences(centroid_index, partition):
    node_to_cluster = cluster_for_nodes(partition)
    rows = []
    for model_key, info in centroid_index.items():
        model = info.get("model", model_key)
        for cluster_index, texts in enumerate(info["cluster_texts"]):
            taxonomy_cluster = node_to_cluster.get((model_key, int(cluster_index)))
            pairs = info["cluster_pmcid_and_sentence_idx"][cluster_index]
            members = info["cluster_members"][cluster_index]
            for offset, (text, pair) in enumerate(zip(texts, pairs)):
                member = members[offset] if offset < len(members) else offset
                rows.append(
                    {
                        "model_key": model_key,
                        "model": model,
                        "pmcid": pmcid_from_pair(pair),
                        "sentence_index": sentence_index_from_pair(pair),
                        "source_index": source_index(member, offset),
                        "text": str(text).strip(),
                        "taxonomy_cluster": taxonomy_cluster,
                    }
                )
    return sorted(rows, key=lambda row: (str(row["model"]), int(row["source_index"])))


def split_trace_rows(rows):
    traces = []
    current = []
    previous_model = None
    previous_pmcid = None
    previous_sentence_index = None

    for row in rows:
        sentence_index = row["sentence_index"]
        starts_new = (
            not current
            or row["model"] != previous_model
            or row["pmcid"] != previous_pmcid
            or (
                previous_sentence_index is not None
                and sentence_index is not None
                and int(sentence_index) <= int(previous_sentence_index)
            )
        )
        if starts_new and current:
            traces.append(current)
            current = []
        current.append(row)
        previous_model = row["model"]
        previous_pmcid = row["pmcid"]
        previous_sentence_index = sentence_index

    if current:
        traces.append(current)
    return traces


def is_ordered_subsequence(needle, haystack):
    if not needle:
        return False
    position = 0
    for item in haystack:
        if position < len(needle) and needle[position] == item:
            position += 1
        if position == len(needle):
            return True
    return False


def load_response_candidates(model, responses_root, sentence_splitter):
    path = responses_root / f"responses_{model}.json"
    if not path.exists():
        return [], path
    with open(path, "r", encoding="utf-8") as f:
        responses = json.load(f)

    candidates = []
    for response_index, item in enumerate(responses):
        thinking = extract_thinking_process(item.get("full_response", ""))
        sentences = tuple(sentence_splitter(thinking))
        candidates.append(
            {
                "response_index": response_index,
                "pmcid": str(item.get("pmcid")),
                "question_id": item.get("question_id"),
                "sample_index": item.get("sample_index"),
                "case_prompt": item.get("case_prompt") or item.get("question") or "",
                "true_diagnosis": item.get("true_diagnosis") or item.get("gold_answer") or "",
                "predicted_diagnosis": item.get("predicted_diagnosis") or item.get("extracted_answer") or "",
                "verification_response": item.get("verification_response") or "",
                "verified_correct": item.get("verified_correct")
                if item.get("verified_correct") is not None
                else item.get("is_correct"),
                "sentences": sentences,
            }
        )
    return candidates, path


def build_response_index(models, responses_root, sentence_splitter):
    by_model_pmcid = defaultdict(list)
    source_paths = {}
    for model in sorted(models):
        candidates, path = load_response_candidates(model, responses_root, sentence_splitter)
        source_paths[model] = str(path)
        for candidate in candidates:
            by_model_pmcid[(model, candidate["pmcid"])].append(candidate)
    return by_model_pmcid, source_paths


def match_response(trace_rows, response_index):
    model = trace_rows[0]["model"]
    pmcid = str(trace_rows[0]["pmcid"])
    trace_sentences = tuple(row["text"] for row in trace_rows)
    candidates = response_index.get((model, pmcid), [])
    for candidate in candidates:
        if candidate["sentences"] == trace_sentences:
            return candidate, "matched_exact_response"
    for candidate in candidates:
        if is_ordered_subsequence(trace_sentences, candidate["sentences"]):
            return candidate, "matched_subsequence_response"
    return None, "unmatched_response_id"


def build_taxonomy_header(labels):
    ordered_clusters = sorted(labels, key=lambda value: int(value))
    state_order = [labels[cluster].get("title", f"cluster {cluster}") for cluster in ordered_clusters]
    state_to_idx = {title: idx for idx, title in enumerate(state_order)}
    idx_to_universal = {cluster: f"U{int(cluster) + 1}" for cluster in ordered_clusters}
    universal_taxonomy = {f"U{int(cluster) + 1}": labels[cluster].get("title", f"cluster {cluster}") for cluster in ordered_clusters}
    return state_order, state_to_idx, idx_to_universal, universal_taxonomy


def materialize_trace(trace_rows, trace_instance, labels, state_to_idx, response_match):
    first = trace_rows[0]
    label_json = {}
    sequence = []
    original_sequence_idx = []
    for position, row in enumerate(trace_rows):
        label = labels.get(str(row["taxonomy_cluster"]), {})
        title = label.get("title", f"cluster {row['taxonomy_cluster']}")
        key = str(position + 1)
        label_json[key] = {"function": title, "sentence": row["text"]}
        sequence.append(int(state_to_idx[title]))
        original_sequence_idx.append(str(row["taxonomy_cluster"]))

    reasoning_trace = "\n".join(entry["sentence"] for entry in label_json.values())

    return {
        "pmcid": str(first["pmcid"]),
        "question_id": response_match.get("question_id") if response_match else None,
        "sample_index": response_match.get("sample_index") if response_match else None,
        "case_prompt": response_match.get("case_prompt", "") if response_match else "",
        "true_diagnosis": response_match.get("true_diagnosis", "") if response_match else "",
        "predicted_diagnosis": response_match.get("predicted_diagnosis", "") if response_match else "",
        "reasoning_trace": reasoning_trace,
        "verification_response": response_match.get("verification_response", "") if response_match else "",
        "verified_correct": response_match.get("verified_correct") if response_match else None,
        "label_json": label_json,
        "sequence": sequence,
        "sequence_length": len(sequence),
        "original_sequence_idx": original_sequence_idx,
        # Extra fields are intentionally omitted so this matches white_box_data results.labeled.json.
        # trace_instance/response_index/dedupe status are reflected only in generation logs.
    }


def nullable_int(value):
    if value is None:
        return -1
    try:
        return int(value)
    except Exception:
        return -1


def reconstruct_tag(
    tag,
    results_root=RESULTS_ROOT,
    responses_root=RESPONSES_ROOT,
    output_path=None,
    sentence_splitter=None,
    return_counts=False,
):
    results_root = Path(results_root)
    responses_root = Path(responses_root)
    output_path = Path(output_path) if output_path is not None else results_root / f"reconstructed_cots_{tag}.json"
    sentence_splitter = sentence_splitter or make_sentence_splitter()

    centroid_index_path = results_root / f"centroid_index_{tag}.pkl"
    partition_path = results_root / f"taxonomy_partition_{tag}.pkl"
    labels_path = results_root / f"kpartite_cluster_labels_{tag}.json"

    centroid_index = load_pickle(centroid_index_path)
    partition = load_pickle(partition_path)
    labels = load_labels(labels_path)
    state_order, state_to_idx, idx_to_universal, universal_taxonomy = build_taxonomy_header(labels)

    flat_rows = flatten_taxonomy_sentences(centroid_index, partition)
    trace_rows_list = split_trace_rows(flat_rows)
    response_index, _response_paths = build_response_index(
        {rows[0]["model"] for rows in trace_rows_list if rows},
        responses_root,
        sentence_splitter,
    )

    traces = []
    counts = Counter()
    seen_exact_ids = set()
    trace_instances = Counter()
    for trace_rows in trace_rows_list:
        model = trace_rows[0]["model"]
        pmcid = str(trace_rows[0]["pmcid"])
        trace_instance = trace_instances[(model, pmcid)]
        trace_instances[(model, pmcid)] += 1

        response_match, status = match_response(trace_rows, response_index)
        counts[status] += 1
        if response_match and response_match.get("question_id") is not None:
            exact_id = (model, pmcid, str(response_match.get("question_id")))
            if exact_id in seen_exact_ids:
                counts["dropped_duplicate_exact_id"] += 1
                continue
            seen_exact_ids.add(exact_id)

        traces.append(materialize_trace(trace_rows, trace_instance, labels, state_to_idx, response_match))

    missing_labels = sum(1 for trace in traces for cluster in trace["original_sequence_idx"] if cluster not in labels)
    counts["missing_taxonomy_labels"] = missing_labels
    counts["input_trace_count"] = len(trace_rows_list)
    counts["output_trace_count"] = len(traces)

    output = {
        "state_order": state_order,
        "state_to_idx": state_to_idx,
        "idx_to_universal": idx_to_universal,
        "universal_taxonomy": universal_taxonomy,
        "traces": sorted(
            traces,
            key=lambda trace: (str(trace["pmcid"]), str(trace["question_id"]), nullable_int(trace["sample_index"])),
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(output_path)
    if return_counts:
        return output, dict(sorted(counts.items()))
    return output


def main():
    parser = argparse.ArgumentParser(description="Reconstruct ordered CoTs with SAE taxonomy labels.")
    parser.add_argument("--tag", action="append", choices=DEFAULT_TAGS, help="Tag to reconstruct. Defaults to both.")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--responses-root", type=Path, default=RESPONSES_ROOT)
    parser.add_argument(
        "--spacy-matcher",
        action="store_true",
        help="Use spaCy sentence splitting for response-ID matching. Slower, but closer to activation extraction.",
    )
    args = parser.parse_args()

    tags = args.tag or list(DEFAULT_TAGS)
    sentence_splitter = make_sentence_splitter(use_spacy=args.spacy_matcher)
    for tag in tags:
        output, counts = reconstruct_tag(
            tag,
            results_root=args.results_root,
            responses_root=args.responses_root,
            sentence_splitter=sentence_splitter,
            return_counts=True,
        )
        path = args.results_root / f"reconstructed_cots_{tag}.json"
        print(
            f"Saved {path} with {counts.get('output_trace_count', 0)} traces "
            f"({counts.get('dropped_duplicate_exact_id', 0)} duplicates dropped)."
        )


if __name__ == "__main__":
    main()
