import argparse
import os

import pandas as pd

# reuse the existing labeler logic verbatim; only the input/output paths change
import label_clusters as _lc

RESULTS_ROOT = "/home/ttn/Development/med-interp/GWTune/results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--centroids", action="store_true", help="Use sae_centroids points instead of sae_medoids")
    args = parser.parse_args()

    tag = "sae_centroids" if args.centroids else "sae_medoids"
    points_path = os.path.join(RESULTS_ROOT, f"kpartite_cluster_points_{tag}.pkl")
    output_path = os.path.join(RESULTS_ROOT, f"kpartite_cluster_labels_{tag}.json")

    df = pd.read_pickle(points_path)
    metadata = {
        "model": _lc.MODEL,
        "T": _lc.T,
        "N_SENTENCES": _lc.N_SENTENCES,
        "seed": _lc.SEED,
        "source_point_file": points_path,
        "tag": tag,
    }
    _lc.run_labeling_pipeline(df, output_path, metadata)
    print(f"Saved cluster labels to {output_path}")


if __name__ == "__main__":
    main()
