#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm


@dataclass
class ProbingConfig:
    n_iter: int = 300  # Number of sampled base models.
    train_frac: float = 0.70  # Fraction of the pool used for each sampled subset.
    val_frac: float = 0.30  # Validation split ratio inside each sampled subset.
    max_iter_lr: int = 100  # Max iterations for each LR fit.
    early_loops: int = 10  # Retry loops for early stopping.
    val_thresh: float = 0.90  # Validation threshold for early stopping.
    standardize: bool = True  # Standardize features using subset-train statistics only.
    bootstrap: bool = False  # True: sample with replacement; False: without replacement.
    stratified: bool = True  # Preserve class balance while sampling.
    random_state: int | None = None  # Random seed; None means non-deterministic.


def run_probing(
    X_layers, y_pool, cfg: ProbingConfig,
    *, X_layers_holdout, y_holdout
):
    """
    Train LR probes repeatedly on sampled subsets from a training pool.

    Returns:
      - W: (n_iter, L, d), probe weights mapped back to raw feature space.
      - B: (n_iter, L), probe intercepts mapped back to raw feature space.
      - A: (n_iter, L), hold-out accuracy for each sampled probe.
      - observed_layers: list[(n_iter, d+1)], per-layer [w_raw|b_raw].
      - val_splits: List[np.ndarray], per-iteration OOB indices in pool coordinates.

    If standardization is enabled, parameters are transformed back from z-score space:
      w_raw = w / sigma
      b_raw = b - <mu/sigma, w>
    """
    if X_layers_holdout is None or y_holdout is None:
        raise ValueError("必须提供 X_layers_holdout 与 y_holdout，A 需要在最终 test 上评估。")

    rng = np.random.default_rng(cfg.random_state)

    L = len(X_layers)
    n_pool = len(y_pool)
    d = X_layers[0].shape[1]
    n_train = max(1, int(round(n_pool * cfg.train_frac)))

    W = np.zeros((cfg.n_iter, L, d), dtype=np.float32)
    B = np.zeros((cfg.n_iter, L),     dtype=np.float32)
    A = np.zeros((cfg.n_iter, L),     dtype=np.float32)  # hold-out accuracy
    val_splits: list[np.ndarray] = []                    # Per-iteration OOB indices.

    # Pre-compute class-wise indices for stratified sampling.
    classes, counts = np.unique(y_pool, return_counts=True)
    cls_to_idx = {c: np.flatnonzero(y_pool == c) for c in classes}

    for it in tqdm(range(cfg.n_iter), desc="Sampling & Eval (probe_sampler)"):
        # 1) Sample one in-bag subset from the pool.
        if cfg.stratified:
            take_list = []
            for c, cnt in zip(classes, counts):
                k = max(1, int(round(cfg.train_frac * cnt)))
                src = cls_to_idx[c]
                if cfg.bootstrap:
                    choose = rng.choice(src, size=k, replace=True)
                else:
                    k = min(k, len(src))
                    choose = rng.choice(src, size=k, replace=False)
                take_list.append(choose)
            idx_sub = np.concatenate(take_list, axis=0)
        else:
            replace = True if cfg.bootstrap else False
            idx_sub = rng.choice(n_pool, size=n_train, replace=replace)

        # 2) Compute in-bag/OOB split in pool coordinates.
        inbag_unique = np.unique(idx_sub)
        all_idx = np.arange(n_pool, dtype=int)
        oob_idx_pool = np.setdiff1d(all_idx, inbag_unique, assume_unique=True)
        val_splits.append(oob_idx_pool.copy())

        # 3) Internal train/val split for early stopping only.
        idx_rel = np.arange(len(idx_sub))
        idx_tr_rel, idx_val_rel = train_test_split(
            idx_rel, test_size=cfg.val_frac, shuffle=True
        )
        tr_idx_pool  = idx_sub[idx_tr_rel]
        val_idx_pool = idx_sub[idx_val_rel]

        for l in range(L):
            X_tr_raw  = X_layers[l][tr_idx_pool]
            y_tr      = y_pool[tr_idx_pool]
            X_val_raw = X_layers[l][val_idx_pool]
            y_val     = y_pool[val_idx_pool]
            X_hold_raw= X_layers_holdout[l]

            # 4) Optional standardization on subset-train, then fold back later.
            if cfg.standardize:
                scaler = StandardScaler().fit(X_tr_raw)
                X_tr   = scaler.transform(X_tr_raw)
                X_val  = scaler.transform(X_val_raw)
                X_hold = scaler.transform(X_hold_raw)
                mu = scaler.mean_
                sigma = scaler.scale_
                sigma = np.where((sigma == 0) | ~np.isfinite(sigma), 1.0, sigma)
            else:
                X_tr, X_val, X_hold = X_tr_raw, X_val_raw, X_hold_raw
                mu = None
                sigma = None

            best_w_raw = None
            best_b_raw = None
            best_hold_acc = 0.0

            # 5) Early-stop loop.
            for _ in range(cfg.early_loops):
                clf = LogisticRegression(
                    penalty="l2", solver="lbfgs", fit_intercept=True, max_iter=cfg.max_iter_lr
                )
                clf.fit(X_tr, y_tr)

                val_acc  = accuracy_score(y_val,  clf.predict(X_val))
                hold_acc = accuracy_score(y_holdout, clf.predict(X_hold))

                # Convert parameters back to raw feature space.
                w = clf.coef_.ravel()
                b = float(clf.intercept_[0])
                if cfg.standardize:
                    w_raw = (w / sigma).astype(np.float32)
                    b_raw = float(b - float(np.dot(mu / sigma, w)))
                else:
                    w_raw = w.astype(np.float32)
                    b_raw = b

                best_w_raw   = w_raw
                best_b_raw   = b_raw
                best_hold_acc= float(hold_acc)

                nit = clf.n_iter_
                nit = max(nit) if isinstance(nit, (list, np.ndarray)) else nit
                did_conv = (nit < cfg.max_iter_lr)
                if did_conv or val_acc > cfg.val_thresh:
                    break

            # 6) Save the selected parameters and hold-out metric.
            W[it, l, :] = best_w_raw
            B[it, l]    = best_b_raw
            A[it, l]    = best_hold_acc

    observed_layers = build_observed_layers(W, B)
    print(f"[probe_sampler] n_iter={cfg.n_iter}, train_frac={cfg.train_frac}, ")
    return W, B, A, observed_layers, val_splits


def build_observed_layers(W: np.ndarray, B: np.ndarray):
    """Pack W and B into per-layer [w|b] matrices."""
    n_iter, L, d = W.shape
    obs = []
    for l in range(L):
        w_mat = W[:, l, :]               # (n_iter, d)
        b_vec = B[:, l].reshape(-1, 1)   # (n_iter, 1)
        obs.append(np.hstack([w_mat, b_vec]))  # (n_iter, d+1)
    return obs


def save_probing_npz(save_dir: str, base_name: str, W: np.ndarray, B: np.ndarray, A: np.ndarray):
    """Save probe weights, intercepts, and hold-out accuracy."""
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, f"{base_name}_w.npz"),   W=W)
    np.savez(os.path.join(save_dir, f"{base_name}_b.npz"),   B=B)
    np.savez(os.path.join(save_dir, f"{base_name}_acc.npz"), Acc=A)


def save_val_splits(save_dir: str, base_name: str, val_splits: list[np.ndarray]):
    """Save per-iteration OOB indices as an object array."""
    os.makedirs(save_dir, exist_ok=True)
    obj = np.array(val_splits, dtype=object)
    np.savez(os.path.join(save_dir, f"{base_name}_val_splits.npz"), val_splits=obj)
