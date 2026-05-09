import cupy as cp
import json
import os
import pickle as pkl
import numpy as np
import optuna
from cuml.cluster.hdbscan import HDBSCAN
from tqdm import tqdm

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
CS_LOW = 5
CS_HIGH = 1000
MS_LOW = 1
N_TRIALS = 64

os.makedirs(OUTPUT_ROOT, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

params = {}
for model, layer in MODELS:
    name = f"{model}_layer{layer}"
    print(f"\nSearching {name}...")
    with open(os.path.join(ACTIVATIONS_ROOT, f"activations_{model}_100000_{layer}.pkl"), "rb") as f:
        activations, _, _, _ = pkl.load(f)
    activations = cp.asarray(activations.astype(np.float32))

    def objective(trial):
        cs = trial.suggest_int("min_cluster_size", CS_LOW, CS_HIGH, log=True)
        ms = trial.suggest_int("min_samples", MS_LOW, cs, log=True)
        model = HDBSCAN(min_cluster_size=cs, min_samples=ms, metric="l2")
        labels = model.fit_predict(activations)
        n_clusters = len(set(labels.tolist()) - {-1})
        noise_frac = float((labels == -1).mean())
        persistence = np.asarray(cp.asnumpy(model.cluster_persistence_)) if n_clusters > 0 else np.zeros(0)
        total_persistence = float(persistence.sum())
        mean_persistence = float(persistence.mean()) if persistence.size else 0.0
        trial.set_user_attr("n_clusters", n_clusters)
        trial.set_user_attr("noise_frac", noise_frac)
        trial.set_user_attr("total_persistence", total_persistence)
        trial.set_user_attr("mean_persistence", mean_persistence)
        return total_persistence * (1.0 - noise_frac)

    db_path = os.path.join(OUTPUT_ROOT, f"hdbscan_optuna_{model}_layer{layer}.db")
    study = optuna.create_study(
        study_name=name,
        direction="maximize",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )
    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, N_TRIALS - n_complete)

    with tqdm(total=remaining) as pbar:
        for _ in range(remaining):
            trial = study.ask()
            value = objective(trial)
            study.tell(trial, value)

            best = study.best_trial
            best_n_clusters = best.user_attrs.get("n_clusters", "?")
            best_noise = best.user_attrs.get("noise_frac", None)
            best_noise_str = "?" if best_noise is None else f"{best_noise:.3f}"
            best_persist = best.user_attrs.get("total_persistence", None)
            best_persist_str = "?" if best_persist is None else f"{best_persist:.3f}"
            pbar.set_description(
                f"Best trial: {best.number}. Best value: {study.best_value:.6g} "
                f"(n_clusters={best_n_clusters}, noise={best_noise_str}, persist={best_persist_str})"
            )

            pbar.update(1)

    best = study.best_trial
    best_cs = best.params["min_cluster_size"]
    best_ms = best.params["min_samples"]
    n_clusters = best.user_attrs["n_clusters"]
    noise_frac = best.user_attrs["noise_frac"]
    total_persistence = best.user_attrs["total_persistence"]
    print(f"{name}: best min_cluster_size={best_cs}, min_samples={best_ms}, n_clusters={n_clusters}, noise={noise_frac:.3f}, persist={total_persistence:.3f}, score={best.value:.4f}")
    params[name] = {
        "min_cluster_size": best_cs,
        "min_samples": best_ms,
        "n_clusters": n_clusters,
        "noise_frac": noise_frac,
        "total_persistence": total_persistence,
    }

out_path = os.path.join(OUTPUT_ROOT, "hdbscan_params.json")
with open(out_path, "w") as f:
    json.dump(params, f, indent=2)
print(f"\nSaved {out_path}")
