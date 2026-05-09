#!/usr/bin/env bash
set -euo pipefail

# end-to-end pipeline: SAE-cluster medoids -> GWOT -> k-partite taxonomy -> labels
cd "$(dirname "$0")"

# python sae_cluster_medoids.py
python gwot_sae_medoids.py
python kpartite_taxonomy.py
python label_taxonomy.py
