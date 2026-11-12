"""CAA-style raw mean difference with fixed-objective scalar logistic fitting.

This is a common-GCAV comparator, not a full reproduction of original CAA.
Only theta and c are optimized; the high-dimensional direction is fixed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit

METHOD = "dim_gcav"
DISPLAY_NAME = "Mean-difference + GCAV"
DESCRIPTION = "CAA-style mean-difference direction with scalar logistic calibration and GCAV."
EPSILON = 1e-6


def objective(parameters: np.ndarray, q: np.ndarray, y: np.ndarray):
    """Exact mean logistic loss + epsilon/2 theta^2, with free intercept."""
    theta, c = parameters
    logits = theta * q + c
    residual = expit(logits) - y
    curvature = expit(logits) * expit(-logits)
    value = float(np.mean(np.logaddexp(0.0, (1.0 - 2.0 * y) * logits)) + EPSILON / 2.0 * theta * theta)
    gradient = np.array([np.mean(residual * q) + EPSILON * theta, np.mean(residual)], dtype=np.float64)
    hessian = np.array([[np.mean(curvature * q * q) + EPSILON, np.mean(curvature * q)],
                        [np.mean(curvature * q), np.mean(curvature)]], dtype=np.float64)
    return value, gradient, hessian


def fit_scalar(q: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    q, y = np.asarray(q, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if q.ndim != 1 or q.shape != y.shape or not np.isfinite(q).all() or set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("finite one-dimensional projections and both binary classes required")
    rate = float(np.mean(y))
    parameters = np.array([0.0, np.log(rate) - np.log1p(-rate)], dtype=np.float64)
    history = []
    for iteration in range(200):
        value, gradient, hessian = objective(parameters, q, y)
        gradient_norm = float(np.max(np.abs(gradient)))
        history.append({"objective": value, "gradient_inf": gradient_norm})
        if gradient_norm <= 1e-10:
            break
        if not np.isfinite(hessian).all():
            raise ValueError("nonfinite scalar Hessian")
        step = np.linalg.solve(hessian, gradient)
        descent = float(gradient @ step)
        if not np.isfinite(step).all() or descent <= 0:
            raise ValueError("scalar Newton step is not a finite descent direction")
        fraction = 1.0
        while fraction >= 2.0 ** -40:
            candidate = parameters - fraction * step
            candidate_value = objective(candidate, q, y)[0]
            # Near the optimum, objective changes can round below FP64 epsilon.
            if candidate_value <= value - 1e-4 * fraction * descent or (
                    descent < 1e-16 and candidate_value <= value + 2 * np.finfo(float).eps):
                parameters = candidate
                break
            fraction *= 0.5
        else:
            raise ValueError(f"scalar objective line search failed, gradient={gradient_norm}")
    else:
        raise ValueError("scalar objective did not converge in 200 iterations")
    value, gradient, _ = objective(parameters, q, y)
    if not np.isfinite(parameters).all() or np.max(np.abs(gradient)) > 1e-10:
        raise ValueError("scalar objective did not meet the frozen convergence criterion")
    return {"theta": float(parameters[0]), "c": float(parameters[1]), "epsilon": EPSILON,
            "objective": value, "gradient_inf": float(np.max(np.abs(gradient))),
            "iterations": iteration, "optimizer": "fp64_newton_armijo", "converged": True,
            "gradient_tolerance": 1e-10, "history": history}


def _chunks(indices: np.ndarray, chunk_rows: int = 512):
    for offset in range(0, len(indices), chunk_rows):
        yield indices[offset:offset + chunk_rows]


def project(hidden: np.ndarray, direction: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(hidden[chunk], dtype=np.float64) @ direction for chunk in _chunks(indices)])


def fit_mean_difference(hidden: np.ndarray, labels: np.ndarray, fit_indices: np.ndarray) -> dict[str, Any]:
    """Fit raw-coordinate means and two scalars using exactly fit_indices."""
    labels = np.asarray(labels)
    indices = np.asarray(fit_indices, dtype=np.int64)
    if hidden.ndim != 2 or labels.shape != (len(hidden),) or indices.ndim != 1 or not len(indices):
        raise ValueError("hidden/labels/fit indices have incompatible shape")
    if len(np.unique(indices)) != len(indices) or np.min(indices) < 0 or np.max(indices) >= len(hidden):
        raise ValueError("fit indices must be unique valid rows")
    means = []
    counts = []
    for label in (0, 1):
        selected = indices[labels[indices] == label]
        if not len(selected):
            raise ValueError("both binary classes are required in the fit partition")
        total = np.zeros(hidden.shape[1], dtype=np.float64)
        for chunk in _chunks(selected):
            block = np.asarray(hidden[chunk], dtype=np.float64)
            if not np.isfinite(block).all():
                raise ValueError("nonfinite raw activation in fit partition")
            total += np.sum(block, axis=0, dtype=np.float64)
        means.append(total / len(selected))
        counts.append(int(len(selected)))
    difference = means[1] - means[0]
    norm = float(np.linalg.norm(difference))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("zero/nonfinite raw class mean difference; investigate inputs")
    direction = difference / norm
    z = project(hidden, direction, indices)
    mu = float(np.mean(z, dtype=np.float64))
    scale = float(np.std(z, dtype=np.float64, ddof=0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("degenerate projection variance; investigate inputs")
    fit = fit_scalar((z - mu) / scale, labels[indices])
    a = fit["theta"] / scale
    b = fit["c"] - a * mu
    if not np.isfinite([a, b]).all() or a <= 0:
        raise ValueError("nonpositive/nonfinite fitted slope; investigate labels/coordinates/convergence")
    equivalent_error = float(np.max(np.abs(fit["theta"] * ((z - mu) / scale) + fit["c"] - (a * z + b))))
    return {"mean_negative": means[0], "mean_positive": means[1], "mean_difference": difference,
            "direction": direction, "direction_norm": norm, "mu_z": mu, "s_z": scale,
            "projection_ddof": 0, "a": float(a), "b": float(b), "W": a * direction,
            "fit_counts": {"negative": counts[0], "positive": counts[1], "total": len(indices)},
            "equivalence_max_abs_error": equivalent_error, **fit}
