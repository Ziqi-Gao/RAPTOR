#!/usr/bin/env python3
import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    save_json,
)
from simulate_xrfm import run_xrfm


def _run_xrfm_all_layers(
    X_layers,
    y,
    train_idx,
    test_idx,
    *,
    val_size: float,
    seed: int,
    rfm_iters: Optional[int],
    n_components: int,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    L = len(X_layers)
    d = X_layers[0].shape[1]
    acc_test = np.full(L, np.nan, dtype=float)
    val_acc = np.full(L, np.nan, dtype=float)
    concept = np.full((L, d), np.nan, dtype=float)
    hparams_list: List[Dict[str, Any]] = []

    for l in range(L):
        X_tr = X_layers[l][train_idx]
        y_tr = y[train_idx]
        X_te = X_layers[l][test_idx]
        y_te = y[test_idx]
        try:
            out = run_xrfm(
                X_tr,
                y_tr,
                X_te,
                y_te,
                val_size=val_size,
                random_state=seed,
                rfm_iters=rfm_iters,
                n_components=n_components,
                summary_only=False,
                return_details=True,
                return_concept=True,
            )
            test_metric, val_metric, best_h, concept_vec = out
            acc_test[l] = float(test_metric)
            val_acc[l] = float(val_metric)
            if concept_vec is not None:
                concept[l, :] = np.asarray(concept_vec, dtype=float).reshape(-1)[:d]
            hparams_list.append(best_h if isinstance(best_h, dict) else {"hparams": best_h})
        except Exception as exc:
            hparams_list.append({"error": str(exc)})
            continue

    return {
        "acc_test": acc_test,
        "val_acc": val_acc,
        "concept": concept,
    }, hparams_list


def run_xrfm_experiments(
    models,
    datasets,
    emb_dir,
    results_dir,
    *,
    test_size,
    val_size,
    seed,
    rfm_iters,
    n_components,
):
    for model_id in models:
        mtag = model_tag(model_id)
        for dataset in datasets:
            emb_path = os.path.join(emb_dir, f"{mtag}_{dataset}_embeddings.npz")
            if not os.path.isfile(emb_path):
                print(f"[{now()}] xrfm skip missing embeddings: {emb_path}")
                continue
            out_dir = os.path.join(results_dir, mtag, dataset)
            mkdir(out_dir)

            X_pos, X_neg = maybe_load_embeddings(emb_path)
            if not X_pos or not X_neg:
                print(f"[{now()}] xrfm empty embeddings: {emb_path}")
                continue
            X_layers, y = build_xy(X_pos, X_neg)
            if len(np.unique(y)) < 2:
                print(f"[{now()}] xrfm only one class: {emb_path}")
                continue

            split_path = os.path.join(out_dir, "splits.npz")
            train_idx, _, _, test_idx = load_or_create_splits(
                split_path, y, test_size, val_size, seed
            )

            res, hparams = _run_xrfm_all_layers(
                X_layers,
                y,
                train_idx,
                test_idx,
                val_size=val_size,
                seed=seed,
                rfm_iters=rfm_iters,
                n_components=n_components,
            )
            np.savez_compressed(os.path.join(out_dir, "rfm_results.npz"), **res)
            save_json(os.path.join(out_dir, "rfm_hparams.json"), hparams)
            print(f"[{now()}] xrfm done: model={model_id} dataset={dataset}")


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
    ap.add_argument("--rfm_iters", type=int, default=10)
    ap.add_argument("--rfm_n_components", type=int, default=1)
    args = ap.parse_args()

    models = parse_list(args.models, DEFAULT_MODELS)
    datasets = parse_list(args.datasets, DEFAULT_DATASETS, normalizer=normalize_dataset)
    run_xrfm_experiments(
        models,
        datasets,
        args.emb_dir,
        args.results_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        rfm_iters=args.rfm_iters,
        n_components=args.rfm_n_components,
    )


if __name__ == "__main__":
    main()
