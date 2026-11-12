from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


RAPTOR_C_GRID = np.logspace(np.log10(1e-4), np.log10(100.0), num=100, dtype=float)
VALIDATION_PROTOCOL_VERSION = "pre_refit_validation_v1"
PRE_REFIT_VALIDATION_SOURCE = "pre_refit_train_only"


def tune_raptor_c(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    max_iter: int = 1000,
    return_stats: bool = False,
    fixed_c: float | None = None,
    seed: int | None = None,
    solver: str = "lbfgs",
    penalty: str = "l2",
) -> Tuple[float, float] | Tuple[float, float, Dict[str, Any]]:
    """Score train-only candidates and retain the selected pre-refit model.

    ``fixed_c`` evaluates one predeclared C without a validation search. The
    validation score still comes from that train-only candidate. The optional
    stats retain copies of its parameters before any caller performs a refit.
    """
    if fixed_c is not None and (not np.isfinite(fixed_c) or fixed_c <= 0):
        raise ValueError("fixed_c must be finite and positive")
    best_C = 1.0
    best_val = -1.0
    max_iter_hits = 0
    start = time.perf_counter()
    best_coef = None
    best_intercept = None

    clf = LogisticRegression(
        solver=solver,
        penalty=penalty,
        max_iter=max_iter,
        warm_start=True,
        random_state=seed,
    )
    for C in ([float(fixed_c)] if fixed_c is not None else RAPTOR_C_GRID):
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
            best_coef = clf.coef_.copy()
            best_intercept = clf.intercept_.copy()

    if return_stats:
        return best_C, best_val, {
            "tune_time": time.perf_counter() - start,
            "max_iter_hits": max_iter_hits,
            "candidate_coef": best_coef,
            "candidate_intercept": best_intercept,
            "validation_source": PRE_REFIT_VALIDATION_SOURCE,
            "protocol_version": VALIDATION_PROTOCOL_VERSION,
            "c_selection": "fixed_predeclared" if fixed_c is not None else "outer_validation_grid",
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
    fixed_c: float | None = None,
    seed: int | None = None,
):
    """Compatibility wrapper for older experiment modules."""
    del C_grid, refine_mults, auto_refine, refine_rounds
    return tune_raptor_c(
        Xtr, ytr, Xval, yval, return_stats=return_stats, fixed_c=fixed_c, seed=seed
    )
