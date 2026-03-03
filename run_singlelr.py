#!/usr/bin/env python3
import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from experiment_utils import (
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
from tune_C_on_embeddings import _tune_single


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
) -> Dict[str, np.ndarray]:
    L = len(X_layers)
    d = X_layers[0].shape[1]
    acc_test = np.zeros(L, dtype=float)
    best_C = np.zeros(L, dtype=float)
    val_acc = np.zeros(L, dtype=float)
    max_iter_hits_tune = np.zeros(L, dtype=int)
    max_iter_hits_fit = np.zeros(L, dtype=int)
    n_iter = np.zeros(L, dtype=int)
    W = np.zeros((L, d), dtype=np.float32)
    B = np.zeros((L,), dtype=np.float32)

    do_tune = len(val_idx) > 0 and len(np.unique(y[train_sub_idx])) > 1

    for l in range(L):
        X_tr_raw = X_layers[l][train_sub_idx]
        y_tr = y[train_sub_idx]
        X_val_raw = X_layers[l][val_idx] if len(val_idx) else None
        y_val = y[val_idx] if len(val_idx) else None
        X_te_raw = X_layers[l][test_idx]
        y_te = y[test_idx]

        # Standardize using train_sub only, then reuse the same scaler.
        scaler = StandardScaler().fit(X_tr_raw)
        X_tr = scaler.transform(X_tr_raw)
        X_val = scaler.transform(X_val_raw) if X_val_raw is not None else None
        X_te = scaler.transform(X_te_raw)

        if do_tune and X_val is not None and y_val is not None and len(y_val):
            C_star, best_val, stats = _tune_single(
                X_tr,
                y_tr,
                X_val,
                y_val,
                [],
                [],
                False,
                0,
                return_stats=True,
            )
            best_C[l] = float(C_star)
            val_acc[l] = float(best_val)
            max_iter_hits_tune[l] = int(stats.get("max_iter_hits", 0))
        else:
            best_C[l] = 1.0
            val_acc[l] = -1.0
            max_iter_hits_tune[l] = 0

        if X_val is not None and y_val is not None and len(y_val):
            X_full = np.vstack([X_tr, X_val])
            y_full = np.concatenate([y_tr, y_val])
        else:
            X_full = X_tr
            y_full = y_tr

        # Final fit uses the tuned C on the full train (train_sub + val).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf = LogisticRegression(
                solver=solver,
                penalty=penalty,
                C=float(best_C[l]),
                max_iter=max_iter,
                random_state=seed,
            )
            clf.fit(X_full, y_full)

        nit = clf.n_iter_
        if isinstance(nit, (list, np.ndarray)):
            nit = max(nit)
        n_iter[l] = int(nit)
        max_iter_hits_fit[l] = 1 if int(nit) >= int(max_iter) else 0

        acc_test[l] = float(accuracy_score(y_te, clf.predict(X_te)))

        # Convert (w_std, b_std) back to original embedding space.
        w_std = clf.coef_.ravel().astype(np.float32)
        b_std = float(clf.intercept_[0])
        scale = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
        w_orig = w_std / scale
        b_orig = b_std - float(np.dot(w_orig, scaler.mean_))

        W[l, :] = w_orig
        B[l] = b_orig

    concept = np.hstack([W, B.reshape(-1, 1)])
    return {
        "acc_test": acc_test,
        "best_C": best_C,
        "val_acc": val_acc,
        "max_iter_hits_tune": max_iter_hits_tune,
        "max_iter_hits_fit": max_iter_hits_fit,
        "max_iter_hits_total": max_iter_hits_tune + max_iter_hits_fit,
        "n_iter": n_iter,
        "W": W,
        "B": B,
        "concept": concept,
    }


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
