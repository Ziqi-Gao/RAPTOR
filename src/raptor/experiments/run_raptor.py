import argparse
import json
import os
import tempfile
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from raptor.core import (
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    build_xy,
    ensure_root,
    load_or_create_splits,
    maybe_load_embeddings,
    mkdir,
    model_tag,
    normalize_dataset,
    now,
    parse_list,
)
from raptor.probes.tuning import (
    PRE_REFIT_VALIDATION_SOURCE,
    VALIDATION_PROTOCOL_VERSION,
    tune_raptor_c,
)


def fit_raptor_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    *,
    fixed_c: Optional[float] = None,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    penalty: str = "l2",
    seed: int = 42,
    selection_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    refit: bool = True,
) -> Dict[str, Any]:
    """Freeze a train-only validation decision, then refit on train+validation.

    Test data is optional and is used only after selection and refit. The scaler
    is fit once on train_sub and reused for both the candidate and final model.
    ``refit=False`` returns the candidate so callers can freeze all layer choices
    before calling ``refit_raptor_layer``. ``selection_callback`` may atomically
    persist the candidate decision before refit begins. Both standardised and original-coordinate parameters are
    returned; candidate_* values always describe the pre-refit probe.
    """
    if (X_test is None) != (y_test is None):
        raise ValueError("X_test and y_test must be supplied together")
    if not len(X_val) or not len(y_val):
        raise ValueError("pre-refit validation requires a nonempty validation partition")
    scaler = StandardScaler().fit(X_train)
    X_tr = scaler.transform(X_train)
    X_validation = scaler.transform(X_val)
    C_star, best_val, stats = tune_raptor_c(
        X_tr, y_train, X_validation, y_val, return_stats=True,
        fixed_c=fixed_c, seed=seed, solver=solver, penalty=penalty,
    )
    # Keep the historical candidate-grid max_iter=1000. ``max_iter`` controls
    # the final refit as it did before the public phase API was introduced.
    scale = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
    candidate_W_std = stats["candidate_coef"].ravel().copy()
    candidate_B_std = float(stats["candidate_intercept"][0])
    candidate_W = candidate_W_std / scale
    candidate_B = candidate_B_std - float(np.dot(candidate_W, scaler.mean_))
    decision = {
        "best_C": float(C_star),
        "val_acc_pre_refit": float(best_val),
        "val_acc": float(best_val),
        "validation_source": PRE_REFIT_VALIDATION_SOURCE,
        "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
        "c_selection": stats["c_selection"],
        "candidate_fit_partition": "train_sub",
        "scaler_fit_partition": "train_sub",
        "final_fit_partition": "train_sub",
        "planned_refit_partition": "train_sub+validation",
        "fit_phase": "candidate_only",
        "max_iter": int(max_iter),
        "solver": solver,
        "penalty": penalty,
        "method_seed": int(seed),
        "n_candidate_train": int(len(y_train)),
        "n_validation": int(len(y_val)),
        "n_final_train": int(len(y_train)),
        "scaler_mean": scaler.mean_.copy(),
        "scaler_scale": scale.copy(),
        "candidate_W_std": candidate_W_std,
        "candidate_B_std": candidate_B_std,
        "candidate_W": candidate_W,
        "candidate_B": candidate_B,
        "max_iter_hits_tune": int(stats["max_iter_hits"]),
    }
    if selection_callback is not None:
        selection_callback({key: value.copy() if isinstance(value, np.ndarray) else value
                            for key, value in decision.items()})

    if not refit:
        return {
            **decision, "acc_test": None,
            "W_std": candidate_W_std.copy(), "B_std": candidate_B_std,
            "W": candidate_W.astype(np.float32), "B": np.float32(candidate_B),
            "concept": np.append(candidate_W, candidate_B).astype(np.float32),
        }
    return refit_raptor_layer(
        decision, X_train, y_train, X_val, y_val, X_test, y_test,
    )


def refit_raptor_layer(
    candidate: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Refit a frozen C with the saved train-only scaler; never rescore validation."""
    if (X_test is None) != (y_test is None):
        raise ValueError("X_test and y_test must be supplied together")
    if candidate["validation_source"] != PRE_REFIT_VALIDATION_SOURCE:
        raise ValueError("Refit requires a frozen train-only validation candidate")
    scale = np.asarray(candidate["scaler_scale"])
    mean = np.asarray(candidate["scaler_mean"])
    # Reconstruct the fitted transformer without fitting it again. In particular,
    # retain sklearn's float32 transform rounding used by existing embeddings.
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.n_features_in_ = mean, scale, len(mean)
    max_iter = int(candidate["max_iter"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = LogisticRegression(
            solver=candidate["solver"], penalty=candidate["penalty"],
            C=float(candidate["best_C"]), max_iter=max_iter,
            random_state=int(candidate["method_seed"]),
        )
        clf.fit(np.vstack([scaler.transform(X_train), scaler.transform(X_val)]),
                np.concatenate([y_train, y_val]))
    n_iter = int(np.max(clf.n_iter_))
    # Preserve the legacy final-vector folding precision.
    W_std = clf.coef_.ravel().astype(np.float32)
    B_std = float(clf.intercept_[0])
    W = W_std / scale
    B = B_std - float(np.dot(W, mean))
    test_acc = None
    if X_test is not None:
        test_acc = float(accuracy_score(y_test, clf.predict(scaler.transform(X_test))))
    return {
        **candidate,
        "fit_phase": "refit_complete",
        "final_fit_partition": "train_sub+validation",
        "n_final_train": int(len(y_train) + len(y_val)),
        "acc_test": test_acc,
        "W_std": W_std,
        "B_std": B_std,
        "W": W.astype(np.float32),
        "B": np.float32(B),
        "concept": np.append(W.astype(np.float32), np.float32(B)),
        "n_iter": n_iter,
        "max_iter_hits_fit": int(n_iter >= max_iter),
        "max_iter_hits_total": int(candidate["max_iter_hits_tune"]) + int(n_iter >= max_iter),
    }


def _run_singlelr_all_layers(
    X_layers: List[np.ndarray],
    y: np.ndarray,
    train_sub_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    max_iter: int,
    solver: str,
    penalty: str,
    seed: int,
    fixed_c: Optional[float] = None,
    selection_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    layer_selection_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, np.ndarray]:
    # All train-only decisions are made before any layer is refit or tested.
    candidates = [
        fit_raptor_layer(
            X[train_sub_idx], y[train_sub_idx], X[val_idx], y[val_idx],
            max_iter=max_iter, solver=solver, penalty=penalty, seed=seed,
            fixed_c=fixed_c, selection_callback=selection_callback, refit=False,
        )
        for X in X_layers
    ]
    selected_layer = int(np.argmax([item["val_acc_pre_refit"] for item in candidates]))
    if layer_selection_callback is not None:
        layer_selection_callback({
            "selected_layer": selected_layer,
            "selected_validation_accuracy": candidates[selected_layer]["val_acc_pre_refit"],
            "candidate_layers": list(range(len(candidates))),
            "val_acc_pre_refit": [item["val_acc_pre_refit"] for item in candidates],
            "best_C": [item["best_C"] for item in candidates],
            "tie_rule": "lowest layer index",
            "validation_source": PRE_REFIT_VALIDATION_SOURCE,
            "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
            "stage": "layer_selection_frozen_before_refit",
        })
    results = [
        refit_raptor_layer(
            candidate, X[train_sub_idx], y[train_sub_idx], X[val_idx], y[val_idx],
            X[test_idx], y[test_idx],
        )
        for X, candidate in zip(X_layers, candidates)
    ]
    output = {key: np.asarray([result[key] for result in results]) for key in results[0]}
    for key in ("validation_source", "validation_protocol_version"):
        output[key] = np.asarray(results[0][key])
    output["selected_layer"] = np.asarray(selected_layer)
    output["selected_validation_accuracy"] = np.asarray(candidates[selected_layer]["val_acc_pre_refit"])
    return output


def _save_layer_selection(path: str, decision: Dict[str, Any]) -> None:
    """Atomically save the full layer decision before legacy-grid refits."""
    fd, tmp_path = tempfile.mkstemp(prefix=".selection.", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(decision, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def run_singlelr_experiments(
    models,
    datasets,
    emb_dir,
    results_dir,
    *,
    test_size,
    val_size,
    seed,
    max_iter,
):
    for model_id in models:
        mtag = model_tag(model_id)
        for dataset in datasets:
            emb_path = os.path.join(emb_dir, f"{mtag}_{dataset}_embeddings.npz")
            if not os.path.isfile(emb_path):
                print(f"[{now()}] singlelr skip missing embeddings: {emb_path}")
                continue
            out_dir = os.path.join(results_dir, mtag, dataset)
            mkdir(out_dir)

            X_pos, X_neg = maybe_load_embeddings(emb_path)
            if not X_pos or not X_neg:
                print(f"[{now()}] singlelr empty embeddings: {emb_path}")
                continue
            X_layers, y = build_xy(X_pos, X_neg)
            if len(np.unique(y)) < 2:
                print(f"[{now()}] singlelr only one class: {emb_path}")
                continue

            split_path = os.path.join(out_dir, "splits.npz")
            _, train_sub_idx, val_idx, test_idx = load_or_create_splits(
                split_path, y, test_size, val_size, seed
            )

            res = _run_singlelr_all_layers(
                X_layers,
                y,
                train_sub_idx,
                val_idx,
                test_idx,
                max_iter=max_iter,
                solver="lbfgs",
                penalty="l2",
                seed=seed,
                layer_selection_callback=lambda decision: _save_layer_selection(
                    os.path.join(out_dir, "singlelr_layer_selection.json"), decision
                ),
            )
            np.savez_compressed(os.path.join(out_dir, "singlelr_results.npz"), **res)
            print(f"[{now()}] singlelr done: model={model_id} dataset={dataset}")


def main() -> None:
    ensure_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", type=str, default="./embeddings_all")
    ap.add_argument("--results_dir", type=str, default="./exp_results")
    ap.add_argument("--models", type=str, default="all")
    ap.add_argument("--datasets", type=str, default="all")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--singlelr_max_iter", type=int, default=1000)
    args = ap.parse_args()

    models = parse_list(args.models, DEFAULT_MODELS)
    datasets = parse_list(args.datasets, DEFAULT_DATASETS, normalizer=normalize_dataset)
    run_singlelr_experiments(
        models,
        datasets,
        args.emb_dir,
        args.results_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        max_iter=args.singlelr_max_iter,
    )


if __name__ == "__main__":
    main()
