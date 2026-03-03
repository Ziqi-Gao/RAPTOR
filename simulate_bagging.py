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
    """Fit LR and detect ConvergenceWarning; return (converged: bool, n_iter: int)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X, y)
        converged = True
        for wi in w:
            if issubclass(wi.category, ConvergenceWarning):
                converged = False
                break
    n_iter = getattr(clf, "n_iter_", None)
    if isinstance(n_iter, (list, tuple, np.ndarray)):
        n_iter_disp = int(np.max(n_iter))
    else:
        n_iter_disp = int(n_iter) if n_iter is not None else -1
    return converged, n_iter_disp

def _sample_stratified_without_replacement(y: np.ndarray, frac: float, rng: np.random.RandomState) -> np.ndarray:
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    target = max(1, int(round(frac * n)))
    per_class = {c: max(1, int(round(target * (counts[i]/n)))) for i,c in enumerate(classes)}
    while sum(per_class.values()) > target:
        cmax = max(per_class, key=lambda c: per_class[c])
        if per_class[cmax] > 1: per_class[cmax] -= 1
        else: break
    while sum(per_class.values()) < target:
        cmin = min(per_class, key=lambda c: per_class[c])
        per_class[cmin] += 1
    idx_list: List[np.ndarray] = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        k = min(per_class[c], len(idx_c))
        idx_list.append(rng.choice(idx_c, size=k, replace=False))
    bag = np.concatenate(idx_list, axis=0); rng.shuffle(bag)
    return bag

def run_bagging_simple(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    *, alpha: float=0.6, M: int=20,
    solver: str="lbfgs", penalty: str="l2", C: float=1.0, max_iter: int=1000,
    random_state: int=0, n_jobs: Optional[int]=None,
    summary_only: bool = True,                   # NEW: 仅输出汇总
    return_details: bool = True,                 # NEW: 返回 (acc, nonconv)
    save_path: Optional[str]=None
) -> Union[float, Tuple[float, int]]:
    t0 = time.perf_counter()
    rng = np.random.RandomState(random_state)

    n_not_conv = 0
    n_iter_list: List[int] = []

    p_sum = np.zeros(len(X_test), dtype=float)
    for m in range(M):
        bag_idx = _sample_stratified_without_replacement(y_train, alpha, rng)
        Xb, yb = X_train[bag_idx], y_train[bag_idx]
        tries = 0
        while len(np.unique(yb)) < 2 and tries < 5:
            bag_idx = _sample_stratified_without_replacement(y_train, alpha, rng)
            Xb, yb = X_train[bag_idx], y_train[bag_idx]
            tries += 1
        if len(np.unique(yb)) < 2:
            Xb, yb = X_train, y_train
        clf = LogisticRegression(solver=solver, penalty=penalty, C=C, max_iter=max_iter, n_jobs=n_jobs)
        ok, n_it = _fit_with_convergence(clf, Xb, yb)
        if not ok: n_not_conv += 1
        n_iter_list.append(n_it)
        p_sum += clf.predict_proba(X_test)[:, 1]

    p_avg = p_sum / M
    y_pred = (p_avg >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, y_pred))

    if summary_only:
        print(f"[bagging][summary] acc={acc:.4f} | non_converged={n_not_conv}/{M} | time={time.perf_counter()-t0:.3f}s")

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "test_acc": acc,
                "n_not_converged": n_not_conv,
                "per_model_n_iter": n_iter_list
            }, f, ensure_ascii=False, indent=2)

    return (acc, n_not_conv) if return_details else acc

# ---- optional CLI (仍只打印汇总) ----
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
    args = ap.parse_args()

    X, y = _gen_toy(args.n, args.d, args.random_state)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, stratify=y, random_state=args.random_state)
    run_bagging_simple(X_tr, y_tr, X_te, y_te,
                       alpha=args.alpha, M=args.M, random_state=args.random_state,
                       summary_only=True, return_details=False)
