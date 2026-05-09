import pickle as pkl

import numpy as np
from scipy.stats import pearsonr, spearmanr


def save_projected_activation_pickle(output_path, activations, texts, pmcid_and_sentence_idx, model):
    with open(output_path, "wb") as f:
        pkl.dump((activations, texts, pmcid_and_sentence_idx, model), f)


def normalize_rows(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def orthogonal_map(source, target):
    if source.shape != target.shape:
        raise ValueError(f"source and target must have the same shape, got {source.shape} and {target.shape}.")

    u, _, vt = np.linalg.svd(source.T @ target, full_matrices=False)
    return u @ vt


def landmark_projector(landmark_embedding):
    landmark_embedding = np.asarray(landmark_embedding, dtype=np.float64)
    ref = landmark_embedding[0]
    landmark_norm_sq = np.sum(landmark_embedding * landmark_embedding, axis=1)
    a = 2.0 * (ref - landmark_embedding[1:])
    pinv_t = np.linalg.pinv(a, rcond=1e-10).T
    rhs_offset = -landmark_norm_sq[1:] + landmark_norm_sq[0]
    return pinv_t, rhs_offset


def project_from_landmark_distances(distances, pinv_t, rhs_offset):
    distances = np.asarray(distances, dtype=np.float64)
    dist_sq = distances * distances
    rhs = dist_sq[:, 1:] - dist_sq[:, [0]] + rhs_offset
    return rhs @ pinv_t


def project_to_barycenter_space(
    activations,
    landmark_activations,
    sample_idx,
    landmark_mds_embedding,
    aligned_landmark_embedding,
    q,
    batch_size=1024,
):
    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    pinv_t, rhs_offset = landmark_projector(landmark_mds_embedding)
    landmark_activations = normalize_rows(landmark_activations)
    projected_mds = np.empty((activations.shape[0], landmark_mds_embedding.shape[1]), dtype=np.float64)

    for start in range(0, activations.shape[0], batch_size):
        end = min(start + batch_size, activations.shape[0])
        batch = normalize_rows(activations[start:end])
        distances = 1.0 - batch @ landmark_activations.T
        distances = np.clip(distances, 0.0, 2.0)
        projected_mds[start:end] = project_from_landmark_distances(distances, pinv_t, rhs_offset)

    projected_mds[sample_idx] = landmark_mds_embedding
    projected = projected_mds @ q
    projected[sample_idx] = aligned_landmark_embedding
    return projected


def validate_projection(q, projected, sample_idx, aligned_landmark_embedding, atol=1e-6):
    orthogonality_error = float(np.max(np.abs(q.T @ q - np.eye(q.shape[1]))))
    landmark_error = float(np.max(np.abs(projected[sample_idx] - aligned_landmark_embedding)))

    if orthogonality_error > atol:
        raise ValueError(f"Orthogonal map check failed: max |Q.T @ Q - I| = {orthogonality_error}.")
    if landmark_error > atol:
        raise ValueError(f"Landmark projection check failed: max landmark error = {landmark_error}.")

    return {
        "orthogonality_max_abs_error": orthogonality_error,
        "landmark_max_abs_error": landmark_error,
    }


def cosine_distance_diagnostics(original, projected, n_pairs=50_000, seed=42, idx=None):
    rng = np.random.default_rng(seed)
    if idx is None:
        i = rng.integers(0, original.shape[0], size=n_pairs)
        j = rng.integers(0, original.shape[0], size=n_pairs)
        keep = i != j
        i = i[keep]
        j = j[keep]
    else:
        idx = np.asarray(idx, dtype=np.int64)
        i, j = idx[:, 0], idx[:, 1]

    original_dist = 1.0 - np.sum(normalize_rows(original[i]) * normalize_rows(original[j]), axis=1)
    projected_dist = 1.0 - np.sum(normalize_rows(projected[i]) * normalize_rows(projected[j]), axis=1)
    abs_error = np.abs(original_dist - projected_dist)

    return {
        "n_pairs": int(len(original_dist)),
        "pearson": float(pearsonr(original_dist, projected_dist)[0]),
        "spearman": float(spearmanr(original_dist, projected_dist)[0]),
        "mae": float(np.mean(abs_error)),
        "p95_abs_error": float(np.percentile(abs_error, 95)),
    }
