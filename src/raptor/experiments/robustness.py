"""Unified five-method directional robustness with exact new-run perturbations.

This entrypoint preserves method-native coordinates and finite fitting budgets.
It does not import historical checkpoints or claim verified historical pairing.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import signal
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from raptor.core import DEFAULT_DATASETS, model_tag
from raptor.experiments.accuracy_multiseed import (
    METHOD_HPARAMETERS, _reject_checkpoint, array_sha256, atomic_json, atomic_savez,
    canonical_sha256, environment_identity, file_lock, file_sha256, source_version,
)
from raptor.experiments.accuracy_preparation import inspect_embedding

METHODS = ("singlelr", "xrfm", "gcs", "fixed_reg", "caa")
DEFAULT_MODELS = ("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-32B-Instruct", "google/gemma-7b-it")
COORDINATES = {"singlelr": "raw_classifier_weights_without_intercept", "gcs": "raw_sample_mean_without_intercept",
               "fixed_reg": "raw_classifier_weights_without_intercept", "caa": "raw_mean_difference",
               "xrfm": "standardized_space_AGOP_eigenvector"}
PROTOCOL = "native_directions_drop20_v1"
_STOP = False


def perturbation_indices(y, *, seed, layer, run, drop_fraction=.2, test_size=.2, val_size=.2):
    """Replay the frozen historical RNG recipe; persist these actual new indices."""
    pool, test = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)
    rng = np.random.RandomState(seed + layer * 1000 + run)
    retained = rng.choice(pool, size=len(pool)-int(round(drop_fraction*len(pool))), replace=False)
    split_seed = int(rng.randint(0, 1_000_000))
    train, val = train_test_split(retained, test_size=val_size, stratify=y[retained], random_state=split_seed)
    return {"retained_pool_idx": retained, "train_sub_idx": train, "val_idx": val,
            "test_idx": test, "final_refit_idx": np.r_[train, val]}, split_seed


def direction_statistics(directions, *, required_k=20):
    values = np.asarray(directions, dtype=np.float64)
    if values.ndim != 2 or len(values) != required_k:
        raise ValueError("complete fixed K directions are required")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("nonfinite or zero direction; never silently reduce the denominator")
    unit = values / norms[:, None]
    i, j = np.triu_indices(required_k, 1)
    similarities = np.abs(np.sum(unit[i] * unit[j], axis=1))
    return {"mean_absolute_cosine": float(similarities.mean()), "variance_ddof0": float(similarities.var(ddof=0)),
            "n_pairs": int(len(similarities)), "n_directions": required_k,
            "pair_indices": np.c_[i, j], "pair_absolute_cosine": similarities}


def _payload_digest(arrays, metadata):
    return canonical_sha256({"arrays": {key: array_sha256(value) for key, value in sorted(arrays.items())},
                             "metadata": {key: value for key, value in metadata.items() if key != "payload_sha256"}})


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save(path, arrays, metadata):
    metadata = _jsonable(dict(metadata))
    metadata["payload_sha256"] = _payload_digest(arrays, metadata)
    atomic_savez(path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))


def _load(path, identity):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files if key != "metadata_json"}
            metadata = json.loads(source["metadata_json"].item())
        if metadata.get("identity") != identity or metadata.get("payload_sha256") != _payload_digest(arrays, metadata):
            return None
        if metadata.get("status") not in {"candidate_frozen", "complete", "nonconverged", "error", "degenerate", "nonfinite"}:
            return None
        return arrays, metadata
    except (OSError, ValueError, KeyError, EOFError):
        return None


def _task_identity(task):
    return {key: task[key] for key in ("protocol", "model", "dataset", "layer", "method", "seed", "runs",
                                      "drop_fraction", "test_size", "val_size", "hyperparameters",
                                      "embedding", "code_version", "environment_sha256")}


def _run_native(task, X, y, indices, *, native_seed, candidate_path, identity):
    method = task["method"]
    hp = task["hyperparameters"]
    if method == "fixed_reg" and hp.get("fixed_c") != 1.0:
        raise ValueError("Fixed-reg requires its native fixed C1")
    if method in {"singlelr", "fixed_reg"}:
        from raptor.experiments.run_raptor import fit_raptor_layer, refit_raptor_layer
        restored = _load(candidate_path, identity)
        if restored is None:
            if candidate_path.exists():
                _reject_checkpoint(candidate_path)
            candidate = fit_raptor_layer(X[indices["train_sub_idx"]], y[indices["train_sub_idx"]],
                X[indices["val_idx"]], y[indices["val_idx"]], seed=native_seed,
                fixed_c=1.0 if method == "fixed_reg" else None, max_iter=hp["max_iter"], refit=False)
            candidate_arrays = {key: value for key, value in candidate.items() if isinstance(value, np.ndarray)}
            candidate_decision = {key: value for key, value in candidate.items() if not isinstance(value, np.ndarray)}
            _save(candidate_path, candidate_arrays, {"identity": identity, "status": "candidate_frozen", "decision": candidate_decision})
        else:
            candidate_arrays, candidate_meta = restored
            if candidate_meta["status"] != "candidate_frozen":
                raise ValueError("invalid native candidate checkpoint phase")
            candidate = {**candidate_meta["decision"], **candidate_arrays}
        final = refit_raptor_layer(candidate, X[indices["train_sub_idx"]], y[indices["train_sub_idx"]],
                                  X[indices["val_idx"]], y[indices["val_idx"]])
        return np.asarray(final["W"]), {
            "converged": not bool(final["max_iter_hits_tune"] or final["max_iter_hits_fit"]),
            "max_iter_hits_tune": int(final["max_iter_hits_tune"]), "max_iter_hits_fit": int(final["max_iter_hits_fit"]),
            "final_n_iter": int(final["n_iter"]), "best_C": float(final["best_C"]),
            "candidate_val_acc_pre_refit": float(final["val_acc_pre_refit"]), "B": float(final["B"]),
            "candidate_fit_partition": "train_sub", "scaler_fit_partition": "train_sub",
            "final_fit_partition": "train_sub+validation", "test_scored": False,
        }
    if method == "caa":
        # Audited raw-mean routine. Midpoint classification is not used here.
        from raptor.probes.dim_midpoint import fit_midpoint
        fitted = fit_midpoint(X, y, indices["retained_pool_idx"])
        return fitted["mean_difference"], {"converged": True, "optimization": "none", "test_scored": False,
                                          "fit_counts": fitted["fit_counts"], "direction_fit_partition": "retained_pool_idx"}
    if method == "xrfm":
        from raptor.probes.xrfm import run_xrfm
        retained, test = indices["retained_pool_idx"], indices["test_idx"]
        _, validation, selected, direction = run_xrfm(X[retained], y[retained], X[test], y[test],
            val_size=task["val_size"], random_state=native_seed, rfm_iters=hp["rfm_iters"],
            n_components=hp["n_components"], summary_only=False, return_details=True, return_concept=True)
        if direction is None:
            raise ValueError("xRFM did not return its standardized AGOP direction")
        return np.asarray(direction), {"converged": None, "convergence_status": "fixed_native_RFM_budget",
            "selected_hyperparameters": selected, "direction_fit_partition": "retained_pool_with_native_internal_split",
            "validation_score_native": float(validation), "test_scored": True,
            "test_score_used_for_selection": False, "direction_coordinate": COORDINATES[method]}
    if method == "gcs":
        from raptor.experiments.run_gcs import _run_gcs_all_layers
        diagnostics = []
        result = _run_gcs_all_layers([X], y, indices["retained_pool_idx"], indices["test_idx"], seed=native_seed,
            gcs_n_iter=hp["n_iter"], gcs_train_frac=hp["train_frac"], gcs_val_frac=hp["internal_val_frac"],
            gcs_max_iter=hp["max_iter_lr"], gcs_early_loops=hp["early_loops"], gcs_val_thresh=hp["val_thresh"],
            gcs_standardize=hp["standardize"], gcs_bootstrap=hp["bootstrap"], gcs_stratified=hp["stratified"],
            gcs_sample_n=hp["sample_n"], diagnostics=diagnostics)
        hits = sum(item["iteration_limit_hit"] for item in diagnostics)
        concept = np.asarray(result["concept"][0])
        return concept[:-1], {"converged": hits == 0, "optimization_attempts": diagnostics,
            "iteration_limit_hits": hits, "B": float(concept[-1]),
            "direction_fit_partition": "native_subsamples_of_retained_pool", "test_scored": True,
            "test_score_used_for_selection": False}
    raise ValueError("unknown robustness method")


def run_layer(task, *, max_new_units=None):
    """Checkpoint every genuine perturbation, retaining terminal finite-budget outcomes."""
    version = source_version()
    if version["source_sha256"] != task["code_version"]["source_sha256"]:
        raise ValueError("scientific source changed after robustness preparation")
    if environment_identity()["sha256"] != task["environment_sha256"]:
        raise ValueError("environment changed after robustness preparation")
    embedding = task["embedding"]
    if file_sha256(Path(embedding["path"])) != embedding["sha256"]:
        raise ValueError("embedding differs from robustness manifest")
    output = Path(task["output_dir"])
    with file_lock(output / ".lock"):
        with np.load(embedding["path"], allow_pickle=False) as data:
            X = np.vstack([data[f"X_pos_{task['layer']}"], data[f"X_neg_{task['layer']}"]])
        y = np.r_[np.ones(embedding["n_pos"], dtype=int), np.zeros(embedding["n_neg"], dtype=int)]
        computed = restored = 0
        for run in range(task["runs"]):
            indices, split_seed = perturbation_indices(y, seed=task["seed"], layer=task["layer"], run=run,
                drop_fraction=task["drop_fraction"], test_size=task["test_size"], val_size=task["val_size"])
            identity = {**_task_identity(task), "run": run, "native_seed": task["seed"]+run,
                        "val_split_seed": split_seed,
                        "ordered_index_sha256": {key: array_sha256(value) for key, value in indices.items()}}
            path = output / f"run_{run:02d}.npz"
            record = _load(path, identity)
            if record is not None:
                # Index payload must also match the freshly reconstructed recipe.
                if all(key in record[0] and np.array_equal(record[0][key], value) for key, value in indices.items()):
                    restored += 1
                    print(f"[robustness][restored] {task['method']} layer={task['layer']} run={run} status={record[1]['status']}", flush=True)
                    continue
            if _STOP or (max_new_units is not None and computed >= max_new_units):
                break
            if path.exists():
                _reject_checkpoint(path)
            arrays = dict(indices)
            metadata = {"identity": identity, "coordinate_space": COORDINATES[task["method"]],
                        "intercept_in_direction": False, "observed_code_version": version}
            started = time.perf_counter()
            try:
                direction, details = _run_native(task, X, y, indices, native_seed=task["seed"]+run,
                    candidate_path=output / f"candidate_{run:02d}.npz", identity=identity)
                direction = np.asarray(direction).reshape(-1)
                if len(direction) != embedding["dim"]:
                    raise ValueError("direction dimension includes an intercept or is otherwise invalid")
                arrays["direction"] = direction
                norm = float(np.linalg.norm(direction.astype(np.float64)))
                status = "nonfinite" if not np.isfinite(norm) or not np.isfinite(direction).all() else "degenerate" if norm == 0 else "nonconverged" if details.get("converged") is False else "complete"
                metadata.update(details=details, status=status, valid_direction=status in {"complete", "nonconverged"})
            except Exception as error:
                metadata.update(status="error", valid_direction=False,
                                error={"type": type(error).__name__, "message": str(error)})
            metadata["elapsed_seconds"] = time.perf_counter()-started
            _save(path, arrays, metadata)
            if _load(path, identity) is None:
                raise RuntimeError("new robustness unit failed its own integrity verification")
            computed += 1
            print(f"[robustness][computed] {task['method']} layer={task['layer']} run={run} status={metadata['status']}", flush=True)
        return {"computed": computed, "restored": restored, "stop_requested": _STOP}


def prepare(args):
    models = args.models.split(",")
    datasets = args.datasets.split(",")
    methods = args.methods.split(",")
    if (len(set(models)) != len(models) or len(set(datasets)) != len(datasets)
            or len(set(methods)) != len(methods) or set(methods)-set(METHODS)):
        raise ValueError("invalid duplicate/unknown grid entries")
    if args.runs < 2:
        raise ValueError("at least two perturbations are required")
    code = source_version()
    environment = environment_identity()
    output = args.output.resolve()
    tasks = []
    overrides = json.loads(args.config_json.read_text()) if args.config_json else {}
    if (args.runs != 20 or overrides.get("method_hyperparameters")) and not overrides.get("synthetic_smoke"):
        raise ValueError("reduced fitting budgets or K are permitted only for explicitly labelled synthetic_smoke")
    for model in models:
        for dataset in datasets:
            embedding = inspect_embedding(args.emb_dir / f"{model_tag(model)}_{dataset}_embeddings.npz", model, dataset)
            for layer in range(embedding["layers"]):
                for method in methods:
                    hp = copy.deepcopy(METHOD_HPARAMETERS[method]) if method != "caa" else {"direction": "raw_mean_difference"}
                    hp.update(overrides.get("method_hyperparameters", {}).get(method, {}))
                    tasks.append({"protocol": PROTOCOL, "model": model, "dataset": dataset, "layer": layer,
                        "method": method, "seed": args.seed, "runs": args.runs, "drop_fraction": .2,
                        "test_size": .2, "val_size": .2, "hyperparameters": hp,
                        "embedding": embedding, "code_version": code, "environment_sha256": environment["sha256"],
                        "output_dir": str(output / model_tag(model) / dataset / method / f"layer_{layer}")})
    manifest = {"schema": "public_robustness_v1", "protocol": PROTOCOL, "models": models,
        "datasets": datasets, "methods": methods, "runs": args.runs, "seed": args.seed,
        "synthetic_smoke": bool(overrides.get("synthetic_smoke", False)), "code_version": code,
        "environment": environment, "output_root": str(output), "tasks": tasks,
        "n_layers": sum(task["method"] == methods[0] for task in tasks), "n_direction_units": len(tasks)*args.runs,
        "coordinates": {method: COORDINATES[method] for method in methods},
        "historical_pairing": "not verified; these are new-run persisted perturbations"}
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"manifest": str(output / "manifest.json"), "settings": len(models)*len(datasets),
                      "layers": manifest["n_layers"], "tasks": len(tasks), "direction_units": manifest["n_direction_units"]}))
    return manifest


def validate_manifest(manifest):
    if manifest.get("schema") != "public_robustness_v1" or manifest.get("protocol") != PROTOCOL:
        raise ValueError("unknown robustness schema/protocol")
    models, datasets, methods = manifest["models"], manifest["datasets"], manifest["methods"]
    if any(not values or len(values) != len(set(values)) for values in (models, datasets, methods)) or set(methods)-set(METHODS):
        raise ValueError("invalid robustness manifest grid")
    keys, targets, embeddings = set(), set(), {}
    for task in manifest["tasks"]:
        key = (task["model"], task["dataset"], task["layer"], task["method"])
        if key in keys or task["output_dir"] in targets:
            raise ValueError("duplicate robustness task or output")
        keys.add(key); targets.add(task["output_dir"])
        if (task["model"] not in models or task["dataset"] not in datasets or task["method"] not in methods
                or task["code_version"] != manifest["code_version"] or task["environment_sha256"] != manifest["environment"]["sha256"]
                or task["seed"] != manifest["seed"] or task["runs"] != manifest["runs"]
                or task["drop_fraction"] != .2 or task["test_size"] != .2 or task["val_size"] != .2
                or task["protocol"] != PROTOCOL):
            raise ValueError("task identity differs from robustness manifest")
        setting = key[:2]
        if setting in embeddings and embeddings[setting] != task["embedding"]:
            raise ValueError("methods/layers reference different embeddings")
        embeddings[setting] = task["embedding"]
    expected = {(model, dataset, layer, method) for model in models for dataset in datasets
                for layer in range(embeddings.get((model, dataset), {}).get("layers", 0)) for method in methods}
    if (keys != expected or len(embeddings) != len(models)*len(datasets)
            or manifest["n_layers"] != sum(e["layers"] for e in embeddings.values())
            or manifest["n_direction_units"] != len(keys)*manifest["runs"]):
        raise ValueError("robustness manifest does not cover the complete fixed layer grid")
    return len(keys)


def aggregate(manifest, *, output):
    validate_manifest(manifest)
    layers, missing = [], []
    for task in manifest["tasks"]:
        embedding = task["embedding"]
        y = np.r_[np.ones(embedding["n_pos"], dtype=int), np.zeros(embedding["n_neg"], dtype=int)]
        records = []
        for run in range(task["runs"]):
            indices, split_seed = perturbation_indices(y, seed=task["seed"], layer=task["layer"], run=run,
                drop_fraction=task["drop_fraction"], test_size=task["test_size"], val_size=task["val_size"])
            identity = {**_task_identity(task), "run": run, "native_seed": task["seed"]+run,
                        "val_split_seed": split_seed, "ordered_index_sha256": {key: array_sha256(value) for key, value in indices.items()}}
            record = _load(Path(task["output_dir"]) / f"run_{run:02d}.npz", identity)
            if record is None or not record[1].get("valid_direction"):
                missing.append({"model": task["model"], "dataset": task["dataset"], "layer": task["layer"],
                                "method": task["method"], "run": run,
                                "reason": record[1]["status"] if record else "missing_or_incompatible"})
            else:
                if any(key not in record[0] or not np.array_equal(record[0][key], value) for key, value in indices.items()):
                    raise ValueError("robustness payload indices differ from declared perturbation")
                records.append(record)
        if len(records) == task["runs"]:
            stats = direction_statistics([record[0]["direction"] for record in records], required_k=task["runs"])
            row = {"model": task["model"], "dataset": task["dataset"], "layer": task["layer"], "method": task["method"],
                   "mean_absolute_cosine": stats["mean_absolute_cosine"], "n_pairs": stats["n_pairs"],
                   "n_directions": task["runs"], "nonconverged_directions": sum(record[1]["status"] == "nonconverged" for record in records)}
            _save(Path(task["output_dir"]) / "pairs.npz",
                  {"pair_indices": stats["pair_indices"], "pair_absolute_cosine": stats["pair_absolute_cosine"]},
                  {"identity": _task_identity(task), "status": "complete", "summary": row})
            layers.append(row)
    settings = []
    for model in manifest["models"]:
        for dataset in manifest["datasets"]:
            for method in manifest["methods"]:
                expected = [task for task in manifest["tasks"] if (task["model"], task["dataset"], task["method"]) == (model, dataset, method)]
                rows = [row for row in layers if (row["model"], row["dataset"], row["method"]) == (model, dataset, method)]
                complete = len(rows) == len(expected) and len(expected) > 0
                settings.append({"model": model, "dataset": dataset, "method": method,
                    "status": "complete" if complete else "incomplete", "expected_layers": len(expected), "completed_layers": len(rows),
                    "Mean": float(np.mean([row["mean_absolute_cosine"] for row in rows])) if complete else None,
                    "Max": float(np.max([row["mean_absolute_cosine"] for row in rows])) if complete else None,
                    "nonconverged_directions": sum(row["nonconverged_directions"] for row in rows)})
    complete = not missing and all(row["status"] == "complete" for row in settings)
    medians = {method: {stat: float(np.median([row[stat] for row in settings if row["method"] == method]))
                       for stat in ("Mean", "Max")}
               for method in manifest["methods"]} if complete else None
    summary = {"status": "complete_with_nonconvergence" if complete and any(row["nonconverged_directions"] for row in settings) else "complete" if complete else "incomplete",
        "schema": "public_robustness_summary_v1", "protocol": manifest["protocol"], "seed": manifest["seed"],
        "runs": manifest["runs"], "settings": settings, "layers": layers, "missing": missing, "median_across_settings": medians,
        "coordinates": manifest["coordinates"], "historical_pairing": manifest["historical_pairing"],
        "source_manifest_sha256": canonical_sha256(manifest)}
    atomic_json(output, summary)
    return summary


def _stop(_sig, _frame):
    global _STOP
    _STOP = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Inspect embeddings and build a new complete manifest.")
    prep.add_argument("--emb-dir", type=Path, default=Path("embeddings_all"))
    prep.add_argument("--output", type=Path, default=Path("exp_results/robustness_release"))
    prep.add_argument("--models", default=",".join(DEFAULT_MODELS))
    prep.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    prep.add_argument("--methods", default=",".join(METHODS))
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--runs", type=int, default=20)
    prep.add_argument("--config-json", type=Path, help="Explicit configuration overrides, recorded in every task.")
    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--task-index", type=int, help="One layer-method task, zero based; omit for the entire manifest.")
    run.add_argument("--dry-run", action="store_true")
    agg = commands.add_parser("aggregate")
    agg.add_argument("--manifest", type=Path, required=True)
    agg.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return
    manifest = json.loads(args.manifest.read_text())
    if args.command == "aggregate":
        summary = aggregate(manifest, output=args.output or args.manifest.parent / "summary.json")
        if summary["status"] == "incomplete":
            raise SystemExit(2)
        return
    validate_manifest(manifest)
    targets = [task["output_dir"] for task in manifest["tasks"]]
    if len(targets) != len(set(targets)):
        raise ValueError("duplicate robustness output directories")
    if args.dry_run:
        for task in manifest["tasks"]:
            if not Path(task["embedding"]["path"]).is_file():
                raise FileNotFoundError(task["embedding"]["path"])
        print(json.dumps({"tasks": len(targets), "layers": manifest["n_layers"], "direction_units": manifest["n_direction_units"]}))
        return
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGUSR1, _stop)
    tasks = manifest["tasks"] if args.task_index is None else [manifest["tasks"][args.task_index]]
    for task in tasks:
        result = run_layer(task)
        if result["stop_requested"]:
            raise SystemExit(75)


if __name__ == "__main__":
    main()
