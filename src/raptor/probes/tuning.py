from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


RAPTOR_C_GRID = np.logspace(np.log10(1e-4), np.log10(100.0), num=100, dtype=float)


def tune_raptor_c(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    max_iter: int = 1000,
    return_stats: bool = False,
) -> Tuple[float, float] | Tuple[float, float, Dict[str, Any]]:
    """Tune the inverse ridge strength C for RAPTOR logistic probes."""
    best_C = 1.0
    best_val = -1.0
    max_iter_hits = 0
    start = time.perf_counter()

    clf = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        max_iter=max_iter,
        warm_start=True,
    )
    for C in RAPTOR_C_GRID:
        clf.set_params(C=float(C))
        clf.fit(X_train, y_train)
        n_iter = clf.n_iter_
        if isinstance(n_iter, (list, np.ndarray)):
            n_iter = max(n_iter)
        if int(n_iter) >= int(max_iter):
            max_iter_hits += 1
        val_acc = float(accuracy_score(y_val, clf.predict(X_val)))
        if val_acc > best_val:
            best_val = val_acc
            best_C = float(C)

    if return_stats:
        return best_C, best_val, {
            "tune_time": time.perf_counter() - start,
            "max_iter_hits": max_iter_hits,
        }
    return best_C, best_val


def _tune_single(
    Xtr,
    ytr,
    Xval,
    yval,
    C_grid=None,
    refine_mults=None,
    auto_refine=False,
    refine_rounds=0,
    *,
    return_stats: bool = False,
):
    """Compatibility wrapper for older experiment modules."""
    del C_grid, refine_mults, auto_refine, refine_rounds
    return tune_raptor_c(Xtr, ytr, Xval, yval, return_stats=return_stats)
