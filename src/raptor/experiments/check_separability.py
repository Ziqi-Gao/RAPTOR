#!/usr/bin/env python3
"""Check whether binary datasets are linearly separable on the training set.

This script can read embedding NPZ files from ``embeddings_all`` or RFM hidden
state pickle files from ``RFM/hidden_states`` and treats each layer as an
independent experiment.
"""

import argparse
import csv
import glob
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from raptor.core import REPO_ROOT


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


def _load_pickle_cpu(path: str):
    try:
        import io
        import pickle
        import torch
    except Exception as exc:
        raise RuntimeError(f"Failed to import torch/pickle for {path}: {exc}") from exc

    orig = torch.storage._load_from_bytes

    def _load_from_bytes(b):
        return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)

    torch.storage._load_from_bytes = _load_from_bytes
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    finally:
        torch.storage._load_from_bytes = orig


def _parse_rfm_filename(path: str) -> Optional[Dict[str, str]]:
    base = os.path.basename(path)
    if not base.endswith(".pth"):
        return None
    name = base[:-4]
    if "_prompt_" not in name:
        return None
    prefix, prompt = name.rsplit("_prompt_", 1)
    if prefix.startswith("halubench_"):
        rest = prefix[len("halubench_") :]
        for split in ("train", "test"):
            token = f"_{split}_"
            if token in rest:
                source, model = rest.split(token, 1)
                return {
                    "task": "halubench",
                    "source_ds": source,
                    "split": split,
                    "model": model,
                    "prompt": prompt,
                    "dataset": f"halubench_{source}_{split}",
                }
    if prefix.startswith("toxic_chat_"):
        rest = prefix[len("toxic_chat_") :]
        for split in ("train", "test"):
            token = f"{split}_"
            if rest.startswith(token):
                model = rest[len(token) :]
                return {
                    "task": "toxic_chat",
                    "source_ds": "",
                    "split": split,
                    "model": model,
                    "prompt": prompt,
                    "dataset": f"toxic_chat_{split}",
                }
    return None


def build_rfm_hidden_index(hidden_dir: str) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}
    for path in sorted(glob.glob(os.path.join(hidden_dir, "*.pth"))):
        info = _parse_rfm_filename(path)
        if not info:
            logging.warning("Skip unrecognized hidden state file: %s", path)
            continue
        key = info["dataset"]
        if key in index:
            key = f'{info["model"]}::{key}'
        info["path"] = path
        index[key] = info
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


def _pretty_print_table(rows: List[List[str]]) -> None:
    if not rows:
        return
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for i, r in enumerate(rows):
        line = "  ".join(val.ljust(widths[idx]) for idx, val in enumerate(r))
        if i == 1:
            sep = "  ".join("-" * w for w in widths)
            print(sep)
        print(line)


def _safe_name(text: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in text)


def _get_halubench_split_indices(
    n_total: int,
    source_ds: str,
    test_ratio: float,
    split_seed: int,
    results_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    import pickle

    os.makedirs(results_dir, exist_ok=True)
    out_name = os.path.join(
        results_dir,
        f"{source_ds}_train_test_split_seed_{split_seed}_test_{test_ratio:.2f}.pkl",
    )
    try:
        with open(out_name, "rb") as f:
            split = pickle.load(f)
            return split["train_indices"], split["test_indices"]
    except Exception:
        pass

    rng = np.random.RandomState(split_seed)
    indices = rng.permutation(n_total)
    n_test = max(1, int(n_total * float(test_ratio)))
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    with open(out_name, "wb") as f:
        pickle.dump({"train_indices": train_indices, "test_indices": test_indices}, f)
    return train_indices, test_indices


def _load_rfm_labels(
    task: str,
    split: str,
    source_ds: str,
    test_ratio: float,
    split_seed: int,
    rfm_root: str,
) -> np.ndarray:
    from datasets import load_dataset

    if task == "toxic_chat":
        ds = load_dataset("lmsys/toxic-chat", "toxicchat1123")
        subset = ds["train"] if split == "train" else ds["test"]
        return np.array([x["toxicity"] for x in subset], dtype=int)
    if task == "halubench":
        ds = load_dataset("PatronusAI/HaluBench")["test"]
        ds = ds.filter(lambda x: x["source_ds"] == source_ds)
        labels = np.array([int(x["label"] == "FAIL") for x in ds], dtype=int)
        results_dir = os.path.join(rfm_root, "results", "halubench_results", source_ds)
        train_idx, test_idx = _get_halubench_split_indices(
            len(labels),
            source_ds,
            test_ratio,
            split_seed,
            results_dir,
        )
        idx = train_idx if split == "train" else test_idx
        return labels[idx]
    raise ValueError(f"Unsupported task for RFM labels: {task}")


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


def iter_rfm_layers(
    path: str,
    labels: np.ndarray,
    only_layer: Optional[int] = None,
) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    hidden_states = _load_pickle_cpu(path)
    layers = sorted(k for k in hidden_states.keys() if isinstance(k, int))
    if only_layer is not None:
        layers = [l for l in layers if l == only_layer]
    for layer in layers:
        X = hidden_states[layer]
        if hasattr(X, "detach"):
            X = X.detach().cpu().numpy()
        else:
            X = np.asarray(X, dtype=float)
        if X.shape[0] != labels.shape[0]:
            raise ValueError(f"Label length {labels.shape[0]} != X rows {X.shape[0]}")
        yield layer, X, labels


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


def run_rfm_dataset(
    name: str,
    info: Dict[str, str],
    c_values: Iterable[float],
    random_state: int,
    only_layer: Optional[int],
    test_ratio: float,
    split_seed: int,
    rfm_root: str,
) -> List[dict]:
    rows = []
    path = info["path"]
    dataset = info["dataset"]
    model = info["model"]
    task = info["task"]
    source_ds = info["source_ds"]
    split = info["split"]
    logging.info("Loading RFM hidden states: %s (%s)", name, path)

    try:
        labels = _load_rfm_labels(task, split, source_ds, test_ratio, split_seed, rfm_root)
    except Exception as exc:
        logging.exception("Failed to load labels for %s: %s", name, exc)
        return [
            {
                "dataset": dataset,
                "model": model,
                "layer": only_layer if only_layer is not None else "",
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
        ]

    for layer, X, y in iter_rfm_layers(path, labels, only_layer=only_layer):
        try:
            if X.ndim != 2:
                raise ValueError(f"X must be 2D, got shape {X.shape}")
            y_pm = _to_pm_one(y)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Check linear separability on training sets.")
    parser.add_argument(
        "--datasets",
        required=True,
        type=str,
        help="Comma-separated dataset names, or 'all' for all datasets in the selected source.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./results",
        help="Output directory for CSV.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="emb",
        choices=["emb", "rfm"],
        help="Dataset source: emb (embedding NPZ files) or rfm (RFM hidden states).",
    )
    parser.add_argument(
        "--emb-dir",
        type=str,
        default="./embeddings_all",
        help="Embedding directory containing *_embeddings.npz files.",
    )
    parser.add_argument(
        "--rfm-hidden-dir",
        type=str,
        default="./RFM/hidden_states",
        help="RFM hidden state directory (only for --source rfm).",
    )
    parser.add_argument(
        "--rfm-test-ratio",
        type=float,
        default=0.2,
        help="Test ratio used for halubench train/test split (rfm only).",
    )
    parser.add_argument(
        "--rfm-split-seed",
        type=int,
        default=0,
        help="Split seed used for halubench train/test split (rfm only).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Optional single layer id to run.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    emb_dir = args.emb_dir
    rfm_root = str(REPO_ROOT / "RFM")
    if args.source == "emb":
        emb_index = build_embedding_index(emb_dir)
        if not emb_index:
            logging.error("No embedding files found in %s", emb_dir)
            return 1
        data_index = emb_index
    else:
        rfm_index = build_rfm_hidden_index(args.rfm_hidden_dir)
        if not rfm_index:
            logging.error("No RFM hidden states found in %s", args.rfm_hidden_dir)
            return 1
        data_index = rfm_index

    datasets_raw = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not datasets_raw:
        logging.error("No datasets provided.")
        return 1

    if len(datasets_raw) == 1 and datasets_raw[0].lower() == "all":
        datasets = sorted(data_index.keys())
    else:
        datasets = datasets_raw

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "separability_summary.csv")
    if args.layer is not None:
        if len(datasets) == 1 and datasets[0] in data_index:
            info = data_index[datasets[0]]
            model_tag = _safe_name(info.get("model", "unknown").replace("/", "-"))
            dataset_tag = _safe_name(info.get("dataset", datasets[0]))
            out_csv = os.path.join(
                out_dir,
                f"separability_{model_tag}_{dataset_tag}_layer{args.layer}.csv",
            )
        elif len(datasets) > 1:
            out_csv = os.path.join(out_dir, f"separability_layer{args.layer}.csv")

    c_values = [1e4, 1e6, 1e8]
    random_state = 42

    rows: List[dict] = []
    for name in datasets:
        if name not in data_index:
            logging.error("Dataset %s not found in source %s", name, args.source)
            rows.append(
                {
                    "dataset": name,
                    "model": "",
                    "layer": "",
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
            continue
        if args.source == "emb":
            rows.extend(
                run_embedding_dataset(
                    name,
                    data_index[name],
                    c_values,
                    random_state,
                    only_layer=args.layer,
                )
            )
        else:
            rows.extend(
                run_rfm_dataset(
                    name,
                    data_index[name],
                    c_values,
                    random_state,
                    only_layer=args.layer,
                    test_ratio=args.rfm_test_ratio,
                    split_seed=args.rfm_split_seed,
                    rfm_root=rfm_root,
                )
            )

    fieldnames = [
        "dataset",
        "model",
        "layer",
        "n",
        "d",
        "svm_separable",
        "svm_min_success_c",
        "svm_min_train_error",
        "svm_convergence_warning",
        "svm_exception",
        "svm_margin_min",
        "svm_w_norm",
        "perceptron_zero_error",
        "perceptron_epoch",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    table = [
        [
            "dataset",
            "model",
            "layer",
            "n",
            "d",
            "svm_sep",
            "min_C",
            "min_err",
            "warn",
            "perc_zero",
            "perc_epoch",
        ],
        [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
    ]
    for r in rows:
        table.append(
            [
                str(r["dataset"]),
                str(r["model"]),
                str(r["layer"]),
                str(r["n"]),
                str(r["d"]),
                str(r["svm_separable"]),
                str(r["svm_min_success_c"]),
                str(r["svm_min_train_error"]),
                str(r["svm_convergence_warning"]),
                str(r["perceptron_zero_error"]),
                str(r["perceptron_epoch"]),
            ]
        )
    _pretty_print_table(table)
    logging.info("Saved CSV summary to %s", out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
