#!/usr/bin/env python3
import argparse
import os
from typing import Any, Dict

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
)
from probe_sampler import ProbingConfig, run_probing


def _sample_ring(
    mu: np.ndarray,
    sigma: np.ndarray,
    num_samples: int,
    band: int,
    rng: np.random.Generator,
) -> np.ndarray:
    lower = mu - band * sigma
    upper = mu + band * sigma
    inner_low = mu - (band - 1) * sigma
    inner_high = mu + (band - 1) * sigma

    samples = np.empty((num_samples, mu.shape[0]), dtype=float)
    filled = 0
    nz_mask = sigma > 1e-12

    if not np.any(nz_mask):
        samples[:] = mu
        return samples

    while filled < num_samples:
        need = num_samples - filled
        cand = rng.uniform(low=lower, high=upper, size=(need, mu.shape[0]))
        in_inner = np.all(
            (cand[:, nz_mask] >= inner_low[nz_mask]) &
            (cand[:, nz_mask] <= inner_high[nz_mask]),
            axis=1,
        )
        good = cand[~in_inner]
        k = min(len(good), need)
        if k > 0:
            samples[filled: filled + k] = good[:k]
            filled += k

    if filled < num_samples:
        samples[filled:] = mu
    return samples


def _run_gcs_all_layers(
    X_layers,
    y,
    train_idx,
    test_idx,
    *,
    seed: int,
    gcs_n_iter: int,
    gcs_train_frac: float,
    gcs_val_frac: float,
    gcs_max_iter: int,
    gcs_early_loops: int,
    gcs_val_thresh: float,
    gcs_standardize: bool,
    gcs_bootstrap: bool,
    gcs_stratified: bool,
    gcs_sample_n: int,
) -> Dict[str, Any]:
    X_layers_train = [X[train_idx] for X in X_layers]
    y_train = y[train_idx]
    X_layers_test = [X[test_idx] for X in X_layers]
    y_test = y[test_idx]

    probe_cfg = ProbingConfig(
        n_iter=gcs_n_iter,
        train_frac=gcs_train_frac,
        val_frac=gcs_val_frac,
        max_iter_lr=gcs_max_iter,
        early_loops=gcs_early_loops,
        val_thresh=gcs_val_thresh,
        standardize=gcs_standardize,
        bootstrap=gcs_bootstrap,
        stratified=gcs_stratified,
        random_state=seed,
    )

    _, _, _, observed_layers, _ = run_probing(
        X_layers_train,
        y_train,
        probe_cfg,
        X_layers_holdout=X_layers_test,
        y_holdout=y_test,
    )

    L = len(X_layers)
    rng = np.random.default_rng(seed)
    acc_test = np.full(L, np.nan, dtype=float)
    concept = np.full((L, X_layers[0].shape[1] + 1), np.nan, dtype=float)
    for l in range(L):
        obs = observed_layers[l]
        mu = obs.mean(axis=0)
        sigma = obs.std(axis=0)
        samples = _sample_ring(mu, sigma, gcs_sample_n, 1, rng)
        w = samples[:, :-1]
        b = samples[:, -1]
        logits = X_layers_test[l].dot(w.T) + b
        preds = (logits >= 0).astype(int)
        acc_test[l] = float((preds == y_test[:, None]).mean(axis=0).mean())

        concept_vec = samples.mean(axis=0)
        concept[l, : min(len(concept_vec), concept.shape[1])] = concept_vec[: concept.shape[1]]

    return {"acc_test": acc_test, "concept": concept}


def run_gcs_experiments(
    models,
    datasets,
    emb_dir,
    results_dir,
    *,
    test_size,
    val_size,
    seed,
    gcs_n_iter,
    gcs_train_frac,
    gcs_val_frac,
    gcs_max_iter,
    gcs_early_loops,
    gcs_val_thresh,
    gcs_standardize,
    gcs_bootstrap,
    gcs_stratified,
    gcs_sample_n,
):
    for model_id in models:
        mtag = model_tag(model_id)
        for dataset in datasets:
            emb_path = os.path.join(emb_dir, f"{mtag}_{dataset}_embeddings.npz")
            if not os.path.isfile(emb_path):
                print(f"[{now()}] gcs skip missing embeddings: {emb_path}")
                continue
            out_dir = os.path.join(results_dir, mtag, dataset)
            mkdir(out_dir)

            X_pos, X_neg = maybe_load_embeddings(emb_path)
            if not X_pos or not X_neg:
                print(f"[{now()}] gcs empty embeddings: {emb_path}")
                continue
            X_layers, y = build_xy(X_pos, X_neg)
            if len(np.unique(y)) < 2:
                print(f"[{now()}] gcs only one class: {emb_path}")
                continue

            split_path = os.path.join(out_dir, "splits.npz")
            train_idx, _, _, test_idx = load_or_create_splits(
                split_path, y, test_size, val_size, seed
            )

            res = _run_gcs_all_layers(
                X_layers,
                y,
                train_idx,
                test_idx,
                seed=seed,
                gcs_n_iter=gcs_n_iter,
                gcs_train_frac=gcs_train_frac,
                gcs_val_frac=gcs_val_frac,
                gcs_max_iter=gcs_max_iter,
                gcs_early_loops=gcs_early_loops,
                gcs_val_thresh=gcs_val_thresh,
                gcs_standardize=gcs_standardize,
                gcs_bootstrap=gcs_bootstrap,
                gcs_stratified=gcs_stratified,
                gcs_sample_n=gcs_sample_n,
            )
            np.savez_compressed(os.path.join(out_dir, "gcs_results.npz"), **res)
            print(f"[{now()}] gcs done: model={model_id} dataset={dataset}")


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
    ap.add_argument("--gcs_n_iter", type=int, default=1000)
    ap.add_argument("--gcs_train_frac", type=float, default=0.1)
    ap.add_argument("--gcs_val_frac", type=float, default=0.3)
    ap.add_argument("--gcs_max_iter", type=int, default=100)
    ap.add_argument("--gcs_early_loops", type=int, default=10)
    ap.add_argument("--gcs_val_thresh", type=float, default=0.90)
    ap.add_argument("--gcs_sample_n", type=int, default=1000)
    ap.add_argument("--gcs_no_standardize", action="store_true")
    ap.add_argument("--gcs_bootstrap", action="store_true")
    ap.add_argument("--gcs_no_stratified", action="store_true")
    args = ap.parse_args()

    models = parse_list(args.models, DEFAULT_MODELS)
    datasets = parse_list(args.datasets, DEFAULT_DATASETS, normalizer=normalize_dataset)
    run_gcs_experiments(
        models,
        datasets,
        args.emb_dir,
        args.results_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        gcs_n_iter=args.gcs_n_iter,
        gcs_train_frac=args.gcs_train_frac,
        gcs_val_frac=args.gcs_val_frac,
        gcs_max_iter=args.gcs_max_iter,
        gcs_early_loops=args.gcs_early_loops,
        gcs_val_thresh=args.gcs_val_thresh,
        gcs_standardize=not args.gcs_no_standardize,
        gcs_bootstrap=args.gcs_bootstrap,
        gcs_stratified=not args.gcs_no_stratified,
        gcs_sample_n=args.gcs_sample_n,
    )


if __name__ == "__main__":
    main()
