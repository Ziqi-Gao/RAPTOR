#!/usr/bin/env python3
from __future__ import annotations
from typing import Optional, Tuple, List, Union
import argparse, time, json, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

def _fit_with_convergence(clf: LogisticRegression, X, y):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X, y)
        converged = True
        for wi in w:
            if issubclass(wi.category, ConvergenceWarning):
                converged = False; break
    n_iter = getattr(clf, "n_iter_", None)
    n_iter_disp = int(np.max(n_iter)) if isinstance(n_iter, (list, tuple, np.ndarray)) else int(n_iter) if n_iter is not None else -1
    return converged, n_iter_disp

def _sample_stratified_without_replacement(y: np.ndarray, frac: float, rng: np.random.RandomState) -> np.ndarray:
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    target = max(1, int(round(frac * n)))
    per_class = {c: max(1, int(round(target * (counts[i]/n)))) for i, c in enumerate(classes)}
    while sum(per_class.values()) > target:
        cmax = max(per_class, key=lambda c: per_class[c])
        if per_class[cmax] > 1: per_class[cmax] -= 1
        else: break
    while sum(per_class.values()) < target:
        cmin = min(per_class, key=lambda c: per_class[c])
        per_class[cmin] += 1
    idx_list = []
    for c in classes:
        idx_c = np.where(y == c)[0]; k = min(per_class[c], len(idx_c))
        idx_list.append(rng.choice(idx_c, size=k, replace=False))
    bag = np.concatenate(idx_list, axis=0); rng.shuffle(bag); return bag

def _colwise_oob_mean_impute(Z: np.ndarray, mask_valid: np.ndarray) -> np.ndarray:
    Z = Z.copy()
    for j in range(Z.shape[1]):
        col = Z[:, j]; mv = mask_valid[:, j]
        mu = np.mean(col[mv]) if np.any(mv) else 0.0
        col[~mv] = mu; Z[:, j] = col
    return Z

def _standardize_by_oob(Z_tr: np.ndarray, Z_te: Optional[np.ndarray]):
    mu = np.nanmean(Z_tr, axis=0)
    sigma = np.nanstd(Z_tr, axis=0)
    sigma[(sigma == 0) | ~np.isfinite(sigma)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    Z_tr_s = (Z_tr - mu[None, :]) / sigma[None, :]
    Z_te_s = None if Z_te is None else (Z_te - mu[None, :]) / sigma[None, :]
    return Z_tr_s, Z_te_s

def run_ensemble_bagging(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    *, alpha: float=0.6, M: int=20, agg: str="mean_std",
    standardize_logits: bool=True,
    solver: str="lbfgs", penalty: str="l2", C: float=1.0, max_iter: int=1000,
    random_state: int=0, n_jobs: Optional[int]=None,
    summary_only: bool = True,                  # NEW: 仅输出汇总
    return_details: bool = True,                # NEW: 返回 (acc, nonconv)
    save_path: Optional[str]=None
) -> Union[float, Tuple[float, int]]:
    t0 = time.perf_counter()
    rng = np.random.RandomState(random_state)
    n_train = X_train.shape[0]

    base_models: List[LogisticRegression] = []
    inbag_list: List[np.ndarray] = []
    conv_flags: List[bool] = []

    for _ in range(M):
        bag_idx = _sample_stratified_without_replacement(y_train, alpha, rng)
        Xb, yb = X_train[bag_idx], y_train[bag_idx]
        tries = 0
        while len(np.unique(yb)) < 2 and tries < 5:
            bag_idx = _sample_stratified_without_replacement(y_train, alpha, rng)
            Xb, yb = X_train[bag_idx], y_train[bag_idx]; tries += 1
        if len(np.unique(yb)) < 2:
            Xb, yb = X_train, y_train; bag_idx = np.arange(n_train)
        clf = LogisticRegression(solver=solver, penalty=penalty, C=C, max_iter=max_iter, n_jobs=n_jobs)
        conv, _ = _fit_with_convergence(clf, Xb, yb)
        conv_flags.append(conv); base_models.append(clf); inbag_list.append(bag_idx)

    n_not_conv = int(np.sum(~np.array(conv_flags, dtype=bool)))

    # OOB logits
    Z_tr = np.full((n_train, M), np.nan, dtype=float)
    mask_valid = np.zeros_like(Z_tr, dtype=bool)
    for m, (clf, bag_idx) in enumerate(zip(base_models, inbag_list)):
        oob = np.setdiff1d(np.arange(n_train), bag_idx, assume_unique=False)
        Z_tr[oob, m] = clf.decision_function(X_train[oob])
        mask_valid[oob, m] = True
    Z_te = np.column_stack([clf.decision_function(X_test) for clf in base_models])

    Z_tr_imp = _colwise_oob_mean_impute(Z_tr, mask_valid)
    Z_te_used = Z_te.copy()
    if standardize_logits:
        Z_tr_used, Z_te_used = _standardize_by_oob(Z_tr_imp, Z_te_used)
    else:
        Z_tr_used = Z_tr_imp

    if agg == "matrix":
        Meta_X_tr, Meta_X_te = Z_tr_used, Z_te_used
    elif agg == "mean_raw":
        s_tr = Z_tr_imp.mean(axis=1, keepdims=True)
        mu = s_tr.mean(axis=0); sigma = s_tr.std(axis=0); sigma[sigma==0]=1.0
        Meta_X_tr = (s_tr-mu)/sigma
        s_te = Z_te.mean(axis=1, keepdims=True)
        Meta_X_te = (s_te-mu)/sigma
    elif agg == "mean_std":
        s_tr = Z_tr_used.mean(axis=1, keepdims=True)
        s_te = Z_te_used.mean(axis=1, keepdims=True)
        mu = s_tr.mean(axis=0); sigma = s_tr.std(axis=0); sigma[sigma==0]=1.0
        Meta_X_tr = (s_tr-mu)/sigma
        Meta_X_te = (s_te-mu)/sigma
    else:
        raise ValueError("agg must be one of {'matrix','mean_raw','mean_std'}")

    meta = LogisticRegression(solver=solver, penalty=penalty, C=C, max_iter=max_iter, n_jobs=n_jobs)
    meta.fit(Meta_X_tr, y_train)
    y_pred = meta.predict(Meta_X_te)
    acc = float(accuracy_score(y_test, y_pred))

    if summary_only:
        print(f"[ensemble][summary] acc={acc:.4f} | non_converged={n_not_conv}/{M} | time={time.perf_counter()-t0:.3f}s")

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"test_acc": acc, "n_not_converged": n_not_conv}, f, ensure_ascii=False, indent=2)

    return (acc, n_not_conv) if return_details else acc

# ---- optional CLI (仅打印汇总) ----
def _gen_toy(n: int, d: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    u = rng.normal(size=d); u /= max(1e-12, np.linalg.norm(u))
    mu = 1.0
    n0 = n // 2; n1 = n - n0
    X0 = rng.normal(size=(n0, d)) - mu*u
    X1 = rng.normal(size=(n1, d)) + mu*u
    X = np.vstack([X0, X1]); y = np.array([0]*n0 + [1]*n1, dtype=int)
    return X, y

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--d", type=int, default=20)
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--agg", type=str, default="mean_std", choices=["matrix","mean_raw","mean_std"])
    args = ap.parse_args()

    X, y = _gen_toy(args.n, args.d, args.random_state)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, stratify=y, random_state=args.random_state)
    run_ensemble_bagging(Xtr, ytr, Xte, yte,
                         alpha=args.alpha, M=args.M, agg=args.agg,
                         random_state=args.random_state,
                         summary_only=True, return_details=False)
