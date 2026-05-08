from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


@dataclass
class ProbingConfig:
    n_iter: int = 300
    train_frac: float = 0.70
    val_frac: float = 0.30
    max_iter_lr: int = 100
    early_loops: int = 10
    val_thresh: float = 0.90
    standardize: bool = True
    bootstrap: bool = False
    stratified: bool = True
    random_state: int | None = None


def run_probing(X_layers, y_pool, cfg: ProbingConfig, *, X_layers_holdout, y_holdout):
    if X_layers_holdout is None or y_holdout is None:
        raise ValueError("X_layers_holdout and y_holdout are required.")

    rng = np.random.default_rng(cfg.random_state)
    L = len(X_layers)
    n_pool = len(y_pool)
    d = X_layers[0].shape[1]
    n_train = max(1, int(round(n_pool * cfg.train_frac)))

    W = np.zeros((cfg.n_iter, L, d), dtype=np.float32)
    B = np.zeros((cfg.n_iter, L), dtype=np.float32)
    A = np.zeros((cfg.n_iter, L), dtype=np.float32)
    val_splits: list[np.ndarray] = []

    classes, counts = np.unique(y_pool, return_counts=True)
    cls_to_idx = {c: np.flatnonzero(y_pool == c) for c in classes}

    for it in tqdm(range(cfg.n_iter), desc="Sampling & Eval (GCS)"):
        if cfg.stratified:
            take_list = []
            for c, count in zip(classes, counts):
                k = max(1, int(round(cfg.train_frac * count)))
                src = cls_to_idx[c]
                if cfg.bootstrap:
                    chosen = rng.choice(src, size=k, replace=True)
                else:
                    chosen = rng.choice(src, size=min(k, len(src)), replace=False)
                take_list.append(chosen)
            idx_sub = np.concatenate(take_list, axis=0)
        else:
            idx_sub = rng.choice(n_pool, size=n_train, replace=cfg.bootstrap)

        inbag_unique = np.unique(idx_sub)
        oob_idx_pool = np.setdiff1d(np.arange(n_pool, dtype=int), inbag_unique, assume_unique=True)
        val_splits.append(oob_idx_pool.copy())

        idx_rel = np.arange(len(idx_sub))
        idx_tr_rel, idx_val_rel = train_test_split(
            idx_rel,
            test_size=cfg.val_frac,
            shuffle=True,
        )
        tr_idx_pool = idx_sub[idx_tr_rel]
        val_idx_pool = idx_sub[idx_val_rel]

        for layer in range(L):
            X_tr_raw = X_layers[layer][tr_idx_pool]
            y_tr = y_pool[tr_idx_pool]
            X_val_raw = X_layers[layer][val_idx_pool]
            y_val = y_pool[val_idx_pool]
            X_hold_raw = X_layers_holdout[layer]

            if cfg.standardize:
                scaler = StandardScaler().fit(X_tr_raw)
                X_tr = scaler.transform(X_tr_raw)
                X_val = scaler.transform(X_val_raw)
                X_hold = scaler.transform(X_hold_raw)
                mu = scaler.mean_
                sigma = np.where(
                    (scaler.scale_ == 0) | ~np.isfinite(scaler.scale_),
                    1.0,
                    scaler.scale_,
                )
            else:
                X_tr, X_val, X_hold = X_tr_raw, X_val_raw, X_hold_raw
                mu = None
                sigma = None

            best_w_raw = None
            best_b_raw = None
            best_hold_acc = 0.0

            for _ in range(cfg.early_loops):
                clf = LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    fit_intercept=True,
                    max_iter=cfg.max_iter_lr,
                )
                clf.fit(X_tr, y_tr)
                val_acc = accuracy_score(y_val, clf.predict(X_val))
                hold_acc = accuracy_score(y_holdout, clf.predict(X_hold))

                w = clf.coef_.ravel()
                b = float(clf.intercept_[0])
                if cfg.standardize:
                    w_raw = (w / sigma).astype(np.float32)
                    b_raw = float(b - float(np.dot(mu / sigma, w)))
                else:
                    w_raw = w.astype(np.float32)
                    b_raw = b

                best_w_raw = w_raw
                best_b_raw = b_raw
                best_hold_acc = float(hold_acc)

                n_iter = clf.n_iter_
                n_iter = max(n_iter) if isinstance(n_iter, (list, np.ndarray)) else n_iter
                if int(n_iter) < cfg.max_iter_lr or val_acc > cfg.val_thresh:
                    break

            W[it, layer, :] = best_w_raw
            B[it, layer] = best_b_raw
            A[it, layer] = best_hold_acc

    return W, B, A, build_observed_layers(W, B), val_splits


def build_observed_layers(W: np.ndarray, B: np.ndarray):
    n_iter, L, _ = W.shape
    observed = []
    for layer in range(L):
        observed.append(np.hstack([W[:, layer, :], B[:, layer].reshape(n_iter, 1)]))
    return observed


def save_probing_npz(save_dir: str, base_name: str, W: np.ndarray, B: np.ndarray, A: np.ndarray):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, f"{base_name}_w.npz"), W=W)
    np.savez(os.path.join(save_dir, f"{base_name}_b.npz"), B=B)
    np.savez(os.path.join(save_dir, f"{base_name}_acc.npz"), Acc=A)


def save_val_splits(save_dir: str, base_name: str, val_splits: list[np.ndarray]):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, f"{base_name}_val_splits.npz"),
        val_splits=np.array(val_splits, dtype=object),
    )
