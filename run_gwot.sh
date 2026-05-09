#!/usr/bin/env bash
# Train + title + evaluate + visualize a single SAE on the concatenated
# GWOT-barycenter activations from all six models. The output JSON's
# best_cluster gives the universal taxonomy used by
# label_barycenter_activations.py.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SAES_DIR="$(realpath "$HERE/../thinking-llms-interp/train-saes")"

# Step 1 prereq: build_barycenter_train_saes_input.py must have been run.
N_EXAMPLES_FILE="$HERE/results/barycenter_concat_n_examples.txt"
if [ ! -f "$N_EXAMPLES_FILE" ]; then
    echo "Missing $N_EXAMPLES_FILE -- run: python $HERE/build_barycenter_train_saes_input.py" >&2
    exit 1
fi
N_EXAMPLES="$(cat "$N_EXAMPLES_FILE")"

MODEL="gwot_barycenter"
LAYER=0
CLUSTERS="3 4 5 6 7 8 9 10"
GRANULARITY="gwot"
CLUSTERING_METHODS="sae_topk"
REPETITIONS=1

echo "=== gwot_barycenter SAE pipeline (n_examples=$N_EXAMPLES) ==="

cd "$TRAIN_SAES_DIR"

# Train SAEs across the cluster sweep.
python train_clustering.py \
    --model "$MODEL" --layer "$LAYER" \
    --clusters $CLUSTERS \
    --n_examples "$N_EXAMPLES" \
    --clustering_methods $CLUSTERING_METHODS \
    --granularity "$GRANULARITY"

# Title generation: submit + wait/process.
python generate_titles_trained_clustering.py \
    --model "$MODEL" --layer "$LAYER" \
    --clusters $CLUSTERS \
    --n_examples "$N_EXAMPLES" \
    --clustering_methods $CLUSTERING_METHODS \
    --repetitions "$REPETITIONS" \
    --command submit \
    --evaluator_model gpt-5-mini \
    --max_workers 30 \
    --granularity "$GRANULARITY"

python generate_titles_trained_clustering.py \
    --model "$MODEL" --layer "$LAYER" \
    --clusters $CLUSTERS \
    --n_examples "$N_EXAMPLES" \
    --clustering_methods $CLUSTERING_METHODS \
    --repetitions "$REPETITIONS" \
    --command process --wait-batch-completion \
    --granularity "$GRANULARITY"

# Evaluation: submit + wait/process.
python evaluate_trained_clustering.py \
    --model "$MODEL" --layer "$LAYER" \
    --clusters $CLUSTERS \
    --n_examples "$N_EXAMPLES" \
    --clustering_methods $CLUSTERING_METHODS \
    --repetitions "$REPETITIONS" \
    --command submit \
    --accuracy_target_cluster_percentage 0.2 \
    --max_workers 10 \
    --granularity "$GRANULARITY"

python evaluate_trained_clustering.py \
    --model "$MODEL" --layer "$LAYER" \
    --clusters $CLUSTERS \
    --n_examples "$N_EXAMPLES" \
    --clustering_methods $CLUSTERING_METHODS \
    --repetitions "$REPETITIONS" \
    --command process --wait-batch-completion \
    --granularity "$GRANULARITY"

# Visualizations.
python visualize_results.py \
    --model "$MODEL" --layer "$LAYER" \
    --clusters $CLUSTERS \
    --clustering_methods $CLUSTERING_METHODS \
    --granularity "$GRANULARITY"

python visualize_comparison.py \
    --model "$MODEL" --layer "$LAYER" \
    --granularity "$GRANULARITY"

python visualize_clusters.py \
    --model "$MODEL" \
    --granularity "$GRANULARITY"

echo "=== Done. Best k written to sae_topk_results_${MODEL}_layer${LAYER}_${GRANULARITY}.json ==="
