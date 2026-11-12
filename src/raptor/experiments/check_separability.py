"""Finite-budget linear separability diagnostic on all rows of an embedding cache.

The fit/check kernels are retained from the research implementation. This
in-sample diagnostic does not report held-out accuracy or consume the main
benchmark splits. Historical paper table provenance is not available.
"""
import argparse
import csv
import glob
import hashlib
import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

@dataclass
class SVMResult:
    separable: bool
    min_success_c: Optional[float]
    min_train_error: float
    had_convergence_warning: bool
    had_exception: bool
    margin_min: Optional[float]
    w_norm: Optional[float]

@dataclass
class PerceptronResult:
    zero_error: bool
    epoch_reached: Optional[int]

def _read_meta_string(arr: np.ndarray) -> Optional[str]:
    if arr.size < 1:
        return None
    if arr.dtype.kind in {"U", "S", "O"}:
        val = arr.ravel()[0]
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="ignore")
        return str(val)
    return None

def _parse_layers(keys: Iterable[str]) -> List[int]:
    layers = []
    for k in keys:
        if k.startswith("X_pos_"):
            try:
                layers.append(int(k.split("_")[-1]))
            except ValueError:
                continue
    layers = sorted(set(layers))
    return layers

def build_embedding_index(emb_dir: str) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}
    for path in sorted(glob.glob(os.path.join(emb_dir, "*_embeddings.npz"))):
        try:
            with np.load(path) as data:
                model = _read_meta_string(data["meta_model"]) if "meta_model" in data else None
                dataset = _read_meta_string(data["meta_dataset"]) if "meta_dataset" in data else None
        except Exception:
            logging.exception("Failed to read embedding metadata from %s", path)
            continue

        if not dataset:
            base = os.path.basename(path)
            dataset = base.replace("_embeddings.npz", "")
        key = dataset
        if key in index:
            if model:
                key = f"{model}::{dataset}"
            else:
                key = f"{dataset}::{os.path.basename(path)}"
        index[key] = {
            "path": path,
            "dataset": dataset,
            "model": model or "UNKNOWN_MODEL",
        }
    return index

def _to_pm_one(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    uniq = set(np.unique(y).tolist())
    if uniq.issubset({0, 1}):
        return 2 * y.astype(int) - 1
    if uniq.issubset({-1, 1}):
        return y.astype(int)
    if uniq.issubset({-1, 0, 1}):
        y2 = y.astype(int).copy()
        y2[y2 == 0] = -1
        return y2
    raise ValueError(f"Unsupported label set: {sorted(uniq)} (expected 0/1 or -1/1)")

def _fit_linear_svm(
    X_std: np.ndarray,
    y_pm: np.ndarray,
    c_value: float,
    random_state: int,
    max_iter_start: int = 20000,
    max_iter_cap: int = 200000,
) -> Tuple[Optional[LinearSVC], bool, bool]:
    """
    Train LinearSVC with retries if convergence warnings appear.

    Returns: (model_or_none, had_convergence_warning, had_exception)
    """
    max_iter = max_iter_start
    had_warning = False
    while True:
        try:
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always", ConvergenceWarning)
                model = LinearSVC(
                    C=c_value,
                    fit_intercept=True,
                    max_iter=max_iter,
                    random_state=random_state,
                )
                model.fit(X_std, y_pm)
                if any(issubclass(w.category, ConvergenceWarning) for w in wlist):
                    had_warning = True
                    if max_iter < max_iter_cap:
                        max_iter = min(max_iter * 5, max_iter_cap)
                        logging.warning(
                            "ConvergenceWarning for C=%s; retrying with max_iter=%d",
                            c_value,
                            max_iter,
                        )
                        continue
            return model, had_warning, False
        except Exception as exc:
            logging.exception("LinearSVC failed for C=%s with error: %s", c_value, exc)
            return None, had_warning, True

def svm_separability_check(
    X_std: np.ndarray,
    y_pm: np.ndarray,
    c_values: Iterable[float],
    random_state: int,
) -> SVMResult:
    min_error = 1.0
    min_success_c = None
    margin_min = None
    w_norm = None
    had_warning = False
    had_exception = False

    for c_value in c_values:
        model, warn, exc = _fit_linear_svm(X_std, y_pm, c_value, random_state)
        had_warning = had_warning or warn
        had_exception = had_exception or exc
        if model is None:
            continue
        preds = model.predict(X_std)
        err = float(np.mean(preds != y_pm))
        min_error = min(min_error, err)
        if err == 0.0 and min_success_c is None:
            min_success_c = float(c_value)
            w = model.coef_.ravel()
            b = float(model.intercept_.ravel()[0]) if model.fit_intercept else 0.0
            scores = X_std @ w + b
            margin_min = float(np.min(y_pm * scores))
            w_norm = float(np.linalg.norm(w))
            break

    return SVMResult(
        separable=min_success_c is not None,
        min_success_c=min_success_c,
        min_train_error=min_error,
        had_convergence_warning=had_warning,
        had_exception=had_exception,
        margin_min=margin_min,
        w_norm=w_norm,
    )

def perceptron_check(
    X_std: np.ndarray,
    y_pm: np.ndarray,
    max_epochs: int,
    random_state: int,
) -> PerceptronResult:
    clf = Perceptron(
        max_iter=1,
        tol=None,
        fit_intercept=True,
        shuffle=True,
        random_state=random_state,
        warm_start=True,
    )
    zero_epoch = None
    for epoch in range(1, max_epochs + 1):
        clf.fit(X_std, y_pm)
        preds = clf.predict(X_std)
        err = float(np.mean(preds != y_pm))
        if err == 0.0:
            zero_epoch = epoch
            break
    return PerceptronResult(zero_error=zero_epoch is not None, epoch_reached=zero_epoch)

def iter_embedding_layers(
    path: str,
    only_layer: Optional[int] = None,
) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    with np.load(path) as data:
        keys = list(data.keys())
        layers = _parse_layers(keys)
        if only_layer is not None:
            layers = [l for l in layers if l == only_layer]
        for layer in layers:
            pos_key = f"X_pos_{layer}"
            neg_key = f"X_neg_{layer}"
            if pos_key not in data or neg_key not in data:
                logging.warning("Missing %s/%s in %s", pos_key, neg_key, path)
                continue
            X_pos = np.asarray(data[pos_key], dtype=float)
            X_neg = np.asarray(data[neg_key], dtype=float)
            X = np.vstack([X_pos, X_neg])
            y = np.concatenate([np.ones(len(X_pos), dtype=int), -np.ones(len(X_neg), dtype=int)])
            yield layer, X, y

def run_embedding_dataset(
    name: str,
    info: Dict[str, str],
    c_values: Iterable[float],
    random_state: int,
    only_layer: Optional[int] = None,
) -> List[dict]:
    rows = []
    path = info["path"]
    dataset = info["dataset"]
    model = info["model"]
    logging.info("Loading embedding dataset: %s (%s)", name, path)

    for layer, X, y in iter_embedding_layers(path, only_layer=only_layer):
        try:
            if X.ndim != 2:
                raise ValueError(f"X must be 2D, got shape {X.shape}")
            y_pm = _to_pm_one(y)
            if X.shape[0] != y_pm.shape[0]:
                raise ValueError(f"Mismatched X/y sizes: {X.shape[0]} vs {y_pm.shape[0]}")
            n, d = X.shape
            scaler = StandardScaler()
            X_std = scaler.fit_transform(X)
            svm_res = svm_separability_check(X_std, y_pm, c_values, random_state)
            perc_res = perceptron_check(X_std, y_pm, max_epochs=2000, random_state=random_state)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "layer": layer,
                    "n": n,
                    "d": d,
                    "svm_separable": svm_res.separable,
                    "svm_min_success_c": svm_res.min_success_c,
                    "svm_min_train_error": svm_res.min_train_error,
                    "svm_convergence_warning": svm_res.had_convergence_warning,
                    "svm_exception": svm_res.had_exception,
                    "svm_margin_min": svm_res.margin_min,
                    "svm_w_norm": svm_res.w_norm,
                    "perceptron_zero_error": perc_res.zero_error,
                    "perceptron_epoch": perc_res.epoch_reached,
                }
            )
        except Exception as exc:
            logging.exception("Layer %s failed for dataset %s: %s", layer, name, exc)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "layer": layer,
                    "n": "",
                    "d": "",
                    "svm_separable": False,
                    "svm_min_success_c": "",
                    "svm_min_train_error": "",
                    "svm_convergence_warning": False,
                    "svm_exception": True,
                    "svm_margin_min": "",
                    "svm_w_norm": "",
                    "perceptron_zero_error": False,
                    "perceptron_epoch": "",
                }
            )
    if not rows and only_layer is not None:
        logging.error("Layer %s not found in %s", only_layer, path)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "layer": only_layer,
                "n": "",
                "d": "",
                "svm_separable": False,
                "svm_min_success_c": "",
                "svm_min_train_error": "",
                "svm_convergence_warning": False,
                "svm_exception": True,
                "svm_margin_min": "",
                "svm_w_norm": "",
                "perceptron_zero_error": False,
                "perceptron_epoch": "",
            }
        )
    return rows

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings', nargs='+', required=True, type=Path,
                        help='Explicit embedding NPZ files; no ambiguous dataset discovery.')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--layer', type=int)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    rows = []
    inputs = []
    for path in args.embeddings:
        with np.load(path, allow_pickle=False) as data:
            model = _read_meta_string(data['meta_model']) if 'meta_model' in data else 'unrecorded'
            dataset = _read_meta_string(data['meta_dataset']) if 'meta_dataset' in data else path.stem
        info = dict(path=str(path), model=model, dataset=dataset)
        rows.extend(run_embedding_dataset(path.name, info, [1e4, 1e6, 1e8], args.seed, args.layer))
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8*1024*1024), b''): h.update(chunk)
        inputs.append(dict(path=str(path), sha256=h.hexdigest()))
    if not rows: raise ValueError('no layers evaluated')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'separability_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metadata = dict(protocol='all_cache_rows_in_sample_diagnostic_v1', seed=args.seed,
                    inputs=inputs, layers=len(rows), svm_separable=sum(r['svm_separable'] for r in rows),
                    perceptron_zero_error=sum(r['perceptron_zero_error'] for r in rows),
                    failed_layers=sum(r['svm_exception'] for r in rows),
                    C_grid=[1e4,1e6,1e8], perceptron_epochs=2000,
                    warning_policy='SVM native retries from 20000 to 200000 iterations; warnings retained',
                    historical_paper_reproduced=False)
    (args.out/'summary.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return int(metadata['failed_layers'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
