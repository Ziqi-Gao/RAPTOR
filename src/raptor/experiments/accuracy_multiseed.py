"""Restartable, seed-isolated Table 1 rebuttal experiment tasks."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from raptor.core import DATASET_PATHS, REPO_ROOT, split_indices
from raptor.experiments.run_gcs import _sample_ring
from raptor.experiments.run_raptor import _run_singlelr_all_layers
from raptor.probes.gcs_sampler import ProbingConfig, run_probing
from raptor.probes.tuning import PRE_REFIT_VALIDATION_SOURCE, VALIDATION_PROTOCOL_VERSION


SCHEMA_VERSION = 1
DEFAULT_OUTER_SEEDS = (42, 43, 44, 45, 46)
ROW_PROTOCOL = "rowwise"
GROUPED_PROTOCOL = "grouped_split_sensitivity"
METHODS = ("singlelr", "xrfm", "gcs", "fixed_reg", "dim_midpoint")
GROUPED_METHODS = ("singlelr", "xrfm", "gcs")
METHOD_SEED_OFFSETS = {"singlelr": 0, "xrfm": 100_000, "gcs": 200_000, "fixed_reg": 0, "dim_midpoint": 0}
RESULT_FILENAMES = {
    "singlelr": "singlelr_results.npz",
    "xrfm": "rfm_results.npz",
    "gcs": "gcs_results.npz",
    "fixed_reg": "fixed_reg_results.npz",
    "dim_midpoint": "dim_midpoint_results.npz",
}
# Native budgets. Smaller synthetic runs must explicitly override these values.
METHOD_HPARAMETERS = {
    "singlelr": {"max_iter": 1000, "solver": "lbfgs", "penalty": "l2", "C_grid": "logspace(-4,2,100)"},
    "fixed_reg": {"max_iter": 1000, "solver": "lbfgs", "penalty": "l2", "fixed_c": 1.0},
    "xrfm": {"rfm_iters": 10, "n_components": 1, "tuning_metric": "accuracy", "standardize": True},
    "gcs": {"n_iter": 1000, "train_frac": 0.1, "internal_val_frac": 0.3, "max_iter_lr": 100,
            "early_loops": 10, "val_thresh": 0.90, "sample_n": 1000, "standardize": True,
            "bootstrap": False, "stratified": True},
    "dim_midpoint": {"readout": "raw_euclidean_midpoint_ge_zero_v1", "coordinate_space": "raw"},
}


def protocol_methods(manifest, protocol):
    declared = manifest.get("protocol", {}).get("methods_by_protocol", {})
    methods = declared.get(protocol, manifest.get("grid", {}).get("methods", METHODS))
    if protocol == GROUPED_PROTOCOL:
        methods = declared.get(protocol, [m for m in methods if m in GROUPED_METHODS])
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError("invalid or empty declared method set")
    if protocol == GROUPED_PROTOCOL and set(methods) - set(GROUPED_METHODS):
        raise ValueError("the verified grouped sensitivity supports only RAPTOR/xRFM/GCS")
    return list(methods)


GROUP_COLUMNS = {"counterfact": "subject", "cities": "city"}

_STOP_REQUESTED = False


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one outer seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"outer seeds must be unique: {seeds}")
    return seeds


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_savez(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=str(path.parent),
    )
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **payload)
        with open(tmp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def source_version(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    # Hash the installed scientific package using location-independent names.
    # Wheels and source checkouts of the same package have the same identity.
    package_root = Path(__file__).resolve().parents[1]
    if repo_root != REPO_ROOT:
        package_root = repo_root / "src" / "raptor"
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = "raptor/" + path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    commit = os.environ.get("RAPTOR_GIT_COMMIT", "").strip()
    if not commit:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_root),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
    return {
        "git_commit": commit,
        "source_sha256": digest.hexdigest(),
    }


def environment_identity() -> dict[str, Any]:
    packages = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "unknown")
        direct_url = distribution.read_text("direct_url.json")
        packages.append(
            {
                "name": name,
                "version": distribution.version,
                "direct_url": json.loads(direct_url) if direct_url else None,
            }
        )
    payload = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "runtime_library_environment": {key: os.environ.get(key) for key in (
            "LD_LIBRARY_PATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "packages": sorted(
            packages,
            key=lambda item: (
                item["name"].lower(),
                item["version"],
                json.dumps(item["direct_url"], sort_keys=True),
            ),
        ),
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def seed_directory(
    results_root: Path,
    protocol: str,
    model_tag: str,
    dataset: str,
    seed: int,
) -> Path:
    return results_root / protocol / model_tag / dataset / f"seed_{seed}"


def _load_group_ids(
    dataset: str,
    *,
    n_pos: int,
    n_neg: int,
    data_path: Optional[Path] = None,
) -> np.ndarray:
    if dataset not in GROUP_COLUMNS:
        raise ValueError(f"grouped split is not defined for dataset={dataset}")
    path = data_path or Path(DATASET_PATHS[dataset])
    group_column = GROUP_COLUMNS[dataset]
    positive: list[str] = []
    negative: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if group_column not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing grouping column {group_column}")
        for row in reader:
            try:
                label = int(row["label"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid label in {path}") from exc
            group = row.get(group_column)
            if group is None or group == "":
                raise ValueError(f"empty {group_column} in {path}")
            if label == 1 and len(positive) < n_pos:
                positive.append(group)
            elif label == 0 and len(negative) < n_neg:
                negative.append(group)
            if len(positive) == n_pos and len(negative) == n_neg:
                break
    if len(positive) != n_pos or len(negative) != n_neg:
        raise ValueError(
            f"group alignment mismatch for {dataset}: "
            f"groups=({len(positive)},{len(negative)}) "
            f"embeddings=({n_pos},{n_neg})"
        )
    return np.asarray(positive + negative, dtype=str)


def grouped_split_indices(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(y) != len(groups):
        raise ValueError("labels and groups must have the same length")
    all_idx = np.arange(len(y))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_rel, test_rel = next(outer.split(all_idx, y, groups))
    train_idx = all_idx[train_rel]
    test_idx = all_idx[test_rel]
    if val_size <= 0:
        return train_idx, np.array([], dtype=int), test_idx
    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=seed + 1,
    )
    train_sub_rel, val_rel = next(
        inner.split(train_idx, y[train_idx], groups[train_idx])
    )
    train_sub_idx = train_idx[train_sub_rel]
    val_idx = train_idx[val_rel]
    assert_zero_group_overlap(groups, train_sub_idx, val_idx, test_idx)
    return train_sub_idx, val_idx, test_idx


def assert_zero_group_overlap(
    groups: np.ndarray,
    train_sub_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    partitions = {
        "train_sub": set(groups[train_sub_idx].tolist()),
        "validation": set(groups[val_idx].tolist()),
        "test": set(groups[test_idx].tolist()),
    }
    pairs = (
        ("train_sub", "validation"),
        ("train_sub", "test"),
        ("validation", "test"),
    )
    for left, right in pairs:
        overlap = partitions[left].intersection(partitions[right])
        if overlap:
            raise ValueError(
                f"group leakage between {left} and {right}: {sorted(overlap)[:5]}"
            )


def _validate_partition(
    name: str,
    indices: np.ndarray,
    y: np.ndarray,
    *,
    require_two_classes: bool = True,
) -> None:
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{name} indices must be a 1D integer array")
    if len(indices) == 0:
        raise ValueError(f"{name} partition cannot be empty")
    if int(indices.min()) < 0 or int(indices.max()) >= len(y):
        raise ValueError(f"{name} contains an out-of-range index")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} contains duplicate indices")
    if require_two_classes and len(np.unique(y[indices])) != 2:
        raise ValueError(f"{name} must contain both classes")


def validate_split(
    y: np.ndarray,
    train_sub_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
) -> None:
    _validate_partition("train_sub", train_sub_idx, y)
    _validate_partition("validation", val_idx, y)
    _validate_partition("test", test_idx, y)
    if np.intersect1d(train_sub_idx, val_idx).size:
        raise ValueError("train_sub and validation indices overlap")
    if np.intersect1d(train_sub_idx, test_idx).size:
        raise ValueError("train_sub and test indices overlap")
    if np.intersect1d(val_idx, test_idx).size:
        raise ValueError("validation and test indices overlap")
    union = np.concatenate([train_sub_idx, val_idx, test_idx])
    if not np.array_equal(np.sort(union), np.arange(len(y))):
        raise ValueError("split partitions do not cover every row exactly once")
    if groups is not None:
        assert_zero_group_overlap(groups, train_sub_idx, val_idx, test_idx)


def create_split_artifact(
    path: Path,
    *,
    y: np.ndarray,
    protocol: str,
    seed: int,
    test_size: float,
    val_size: float,
    model: str,
    dataset: str,
    embedding: Mapping[str, Any],
    groups: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    if protocol == ROW_PROTOCOL:
        train_sub_idx, val_idx, test_idx = split_indices(
            y,
            test_size,
            val_size,
            seed,
        )
    elif protocol == GROUPED_PROTOCOL:
        if groups is None:
            raise ValueError("group IDs are required for grouped sensitivity")
        train_sub_idx, val_idx, test_idx = grouped_split_indices(
            y,
            groups,
            test_size,
            val_size,
            seed,
        )
    else:
        raise ValueError(f"unknown split protocol: {protocol}")
    validate_split(
        y,
        train_sub_idx,
        val_idx,
        test_idx,
        groups=groups if protocol == GROUPED_PROTOCOL else None,
    )
    train_idx = np.concatenate([train_sub_idx, val_idx])
    split_sha = array_sha256(train_sub_idx, val_idx, test_idx)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "protocol": protocol,
        "group_column": GROUP_COLUMNS.get(dataset) if groups is not None else None,
        "model": model,
        "dataset": dataset,
        "split_seed": int(seed),
        "test_size": float(test_size),
        "val_size_within_non_test": float(val_size),
        "n_rows": int(len(y)),
        "n_train_sub": int(len(train_sub_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "label_sha256": array_sha256(y),
        "group_sha256": array_sha256(groups) if groups is not None else None,
        "split_sha256": split_sha,
        "embedding_sha256": embedding["sha256"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_savez(
        path,
        train_idx=train_idx,
        train_sub_idx=train_sub_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return metadata


def ensure_split_artifact(
    path: Path,
    *,
    y: np.ndarray,
    protocol: str,
    seed: int,
    test_size: float,
    val_size: float,
    model: str,
    dataset: str,
    embedding: Mapping[str, Any],
    groups: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    expected = {
        "protocol": protocol,
        "model": model,
        "dataset": dataset,
        "split_seed": int(seed),
        "test_size": float(test_size),
        "val_size_within_non_test": float(val_size),
        "n_rows": int(len(y)),
        "label_sha256": array_sha256(y),
        "group_sha256": array_sha256(groups) if groups is not None else None,
        "embedding_sha256": embedding["sha256"],
    }
    with file_lock(path.with_suffix(f"{path.suffix}.lock")):
        if path.exists():
            try:
                _, _, _, metadata = load_split_artifact(
                    path,
                    y=y,
                    expected=expected,
                    groups=groups,
                )
                return metadata
            except ValueError:
                _reject_checkpoint(path)
        return create_split_artifact(
            path,
            y=y,
            protocol=protocol,
            seed=seed,
            test_size=test_size,
            val_size=val_size,
            model=model,
            dataset=dataset,
            embedding=embedding,
            groups=groups,
        )


def load_split_artifact(
    path: Path,
    *,
    y: np.ndarray,
    expected: Optional[Mapping[str, Any]] = None,
    groups: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as data:
            train_sub_idx = np.asarray(data["train_sub_idx"], dtype=int)
            val_idx = np.asarray(data["val_idx"], dtype=int)
            test_idx = np.asarray(data["test_idx"], dtype=int)
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid split artifact: {path}") from exc
    validate_split(
        y,
        train_sub_idx,
        val_idx,
        test_idx,
        groups=groups if metadata.get("protocol") == GROUPED_PROTOCOL else None,
    )
    if metadata.get("status") != "complete":
        raise ValueError(f"incomplete split artifact: {path}")
    if metadata.get("split_sha256") != array_sha256(
        train_sub_idx,
        val_idx,
        test_idx,
    ):
        raise ValueError(f"split checksum mismatch: {path}")
    for key, value in (expected or {}).items():
        if metadata.get(key) != value:
            raise ValueError(
                f"split metadata mismatch for {key}: "
                f"expected={value!r} actual={metadata.get(key)!r}"
            )
    return train_sub_idx, val_idx, test_idx, metadata


def select_validation_layer(
    layer_indices: Iterable[int],
    validation_accuracy: Iterable[float],
    test_accuracy: Iterable[float],
) -> dict[str, float | int]:
    layers = np.asarray(list(layer_indices), dtype=int)
    val = np.asarray(list(validation_accuracy), dtype=float)
    test = np.asarray(list(test_accuracy), dtype=float)
    if len(set(layers.tolist())) != len(layers):
        raise ValueError("layer indices must be unique")
    if not (len(layers) == len(val) == len(test)) or len(layers) == 0:
        raise ValueError(
            "layer, validation, and test vectors must be non-empty and aligned"
        )
    if not np.all(np.isfinite(val)) or not np.all(np.isfinite(test)):
        raise ValueError(
            "layer selection requires finite validation and test accuracies"
        )
    best_val = float(np.max(val))
    tied_layers = layers[val == best_val]
    selected_layer = int(np.min(tied_layers))
    selected_position = int(np.flatnonzero(layers == selected_layer)[0])
    return {
        "selected_layer": selected_layer,
        "selected_validation_accuracy": best_val,
        "best_layer_test_accuracy": float(test[selected_position]),
        "average_layer_test_accuracy": float(np.mean(test)),
    }


def _as_fraction(value: float) -> float:
    value = float(value)
    if value > 1.0:
        value /= 100.0
    return value


def _run_singlelr_layer(
    X: np.ndarray,
    y: np.ndarray,
    train_sub_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    seed: int,
    hyperparameters: Mapping[str, Any],
    selection_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    result = _run_singlelr_all_layers(
        [X],
        y,
        train_sub_idx,
        val_idx,
        test_idx,
        max_iter=int(hyperparameters["max_iter"]),
        solver="lbfgs",
        penalty="l2",
        seed=seed,
        fixed_c=hyperparameters.get("fixed_c"),
        selection_callback=selection_callback,
    )
    return {
        "validation_accuracy": float(result["val_acc"][0]),
        "test_accuracy": float(result["acc_test"][0]),
        "concept": np.asarray(result["concept"][0], dtype=np.float32),
        "candidate_parameters": {
            key: result[key][0] for key in (
                "candidate_W", "candidate_B", "candidate_W_std", "candidate_B_std",
                "scaler_mean", "scaler_scale", "W_std", "B_std",
            )
        },
        "method_details": {
            "best_C": float(result["best_C"][0]),
            "n_iter": int(result["n_iter"][0]),
            "max_iter_hits_tune": int(result["max_iter_hits_tune"][0]),
            "max_iter_hits_fit": int(result["max_iter_hits_fit"][0]),
            "refit_rule": "tune on train_sub/outer-validation; refit on train_sub+validation",
            "validation_source": PRE_REFIT_VALIDATION_SOURCE,
            "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
            "c_selection": str(result["c_selection"][0]),
            "candidate_fit_partition": "train_sub",
            "scaler_fit_partition": "train_sub",
            "final_fit_partition": "train_sub+validation",
            "n_candidate_train": len(train_sub_idx),
            "n_final_train": len(train_sub_idx) + len(val_idx),
        },
    }


def _run_xrfm_layer(
    X: np.ndarray,
    y: np.ndarray,
    train_sub_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    seed: int,
    hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    from raptor.probes.xrfm import run_xrfm
    test_metric, val_metric, best_h, concept = run_xrfm(
        X[train_sub_idx],
        y[train_sub_idx],
        X[test_idx],
        y[test_idx],
        X_val=X[val_idx],
        y_val=y[val_idx],
        val_size=0.0,
        random_state=seed,
        rfm_iters=int(hyperparameters["rfm_iters"]),
        n_components=int(hyperparameters["n_components"]),
        summary_only=False,
        return_details=True,
        return_concept=True,
    )
    concept_array = (
        np.asarray(concept, dtype=np.float32).reshape(-1)
        if concept is not None
        else np.array([], dtype=np.float32)
    )
    return {
        "validation_accuracy": _as_fraction(val_metric),
        "test_accuracy": _as_fraction(test_metric),
        "concept": concept_array,
        "method_details": {
            "selected_hyperparameters": best_h,
            "refit_rule": "no refit; tune on train_sub/outer-validation and evaluate selected model on test",
            "validation_source": PRE_REFIT_VALIDATION_SOURCE,
            "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
            "final_fit_partition": "train_sub",
            "n_final_train": len(train_sub_idx),
        },
    }


def _evaluate_gcs_samples(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    samples: np.ndarray,
) -> float:
    weights = samples[:, :-1]
    intercept = samples[:, -1]
    predictions = (X[indices].dot(weights.T) + intercept >= 0).astype(int)
    return float((predictions == y[indices, None]).mean(axis=0).mean())


def _run_gcs_layer(
    X: np.ndarray,
    y: np.ndarray,
    train_sub_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    seed: int,
    hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    config = ProbingConfig(
        n_iter=int(hyperparameters["n_iter"]),
        train_frac=float(hyperparameters["train_frac"]),
        val_frac=float(hyperparameters["internal_val_frac"]),
        max_iter_lr=int(hyperparameters["max_iter_lr"]),
        early_loops=int(hyperparameters["early_loops"]),
        val_thresh=float(hyperparameters["val_thresh"]),
        standardize=bool(hyperparameters["standardize"]),
        bootstrap=bool(hyperparameters["bootstrap"]),
        stratified=bool(hyperparameters["stratified"]),
        random_state=seed,
    )
    _, _, _, observed_layers, _ = run_probing(
        [X[train_sub_idx]],
        y[train_sub_idx],
        config,
        X_layers_holdout=[X[val_idx]],
        y_holdout=y[val_idx],
    )
    observed = observed_layers[0]
    rng = np.random.default_rng(seed)
    samples = _sample_ring(
        observed.mean(axis=0),
        observed.std(axis=0),
        int(hyperparameters["sample_n"]),
        1,
        rng,
    )
    return {
        "validation_accuracy": _evaluate_gcs_samples(
            X,
            y,
            val_idx,
            samples,
        ),
        "test_accuracy": _evaluate_gcs_samples(
            X,
            y,
            test_idx,
            samples,
        ),
        "concept": np.asarray(samples.mean(axis=0), dtype=np.float32),
        "method_details": {
            "observed_vectors": int(len(observed)),
            "sampled_vectors": int(len(samples)),
            "refit_rule": "no refit; construct GCS on train_sub and evaluate identical sampled probes on outer-validation/test",
            "validation_source": PRE_REFIT_VALIDATION_SOURCE,
            "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
            "final_fit_partition": "subsamples_of_train_sub",
            "outer_training_pool_size": len(train_sub_idx),
        },
    }


def _run_midpoint_layer(X, y, train_sub_idx, val_idx, test_idx, *, seed, hyperparameters, selection_callback=None):
    """Own train-only validation, then native raw means refit on the union."""
    from raptor.probes.dim_midpoint import fit_midpoint, accuracy, READOUT_VERSION
    if hyperparameters.get("readout", READOUT_VERSION) != READOUT_VERSION:
        raise ValueError("CAA accuracy must retain its raw midpoint readout")
    if hyperparameters.get("coordinate_space", "raw") != "raw":
        raise ValueError("CAA midpoint accuracy requires raw embeddings")
    candidate = fit_midpoint(X, y, train_sub_idx)
    validation = accuracy(X, y, val_idx, candidate)
    if selection_callback is not None:
        selection_callback({"candidate_W": candidate["W"], "candidate_B": candidate["b"],
                            "val_acc_pre_refit": validation, "validation_source": PRE_REFIT_VALIDATION_SOURCE})
    final = fit_midpoint(X, y, np.r_[train_sub_idx, val_idx])
    test = accuracy(X, y, test_idx, final)
    return {
        "validation_accuracy": validation, "test_accuracy": test,
        "concept": np.r_[final["W"], final["b"]],
        "candidate_parameters": {"candidate_W": candidate["W"], "candidate_B": candidate["b"],
                                 "mean_negative": final["mean_negative"], "mean_positive": final["mean_positive"]},
        "method_details": {"refit_rule": "train-only raw means; refit raw means on train_sub+validation",
            "readout": READOUT_VERSION, "coordinate_space": "raw", "no_logistic_fit": True,
            "validation_source": PRE_REFIT_VALIDATION_SOURCE,
            "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
            "candidate_fit_partition": "train_sub", "final_fit_partition": "train_sub+validation",
            "n_candidate_train": len(train_sub_idx), "n_final_train": len(train_sub_idx)+len(val_idx),
            "candidate_degenerate": candidate["degenerate"], "final_degenerate": final["degenerate"]},
    }


def _run_fixed_reg_layer(*args, hyperparameters, **kwargs):
    if hyperparameters.get("fixed_c") != 1.0:
        raise ValueError("Fixed-reg retains its predeclared sklearn C=1")
    return _run_singlelr_layer(*args, hyperparameters=hyperparameters, **kwargs)


METHOD_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "singlelr": _run_singlelr_layer,
    "xrfm": _run_xrfm_layer,
    "gcs": _run_gcs_layer,
    "fixed_reg": _run_fixed_reg_layer,
    "dim_midpoint": _run_midpoint_layer,
}


def _checkpoint_identity(
    task: Mapping[str, Any],
    *,
    layer: int,
    split_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    method = str(task["method"])
    return {
        "schema_version": SCHEMA_VERSION,
        **({"scope": task.get("scope", task["embedding"].get("scope"))}
           if task.get("scope", task["embedding"].get("scope")) is not None else {}),
        "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
        "protocol": task["protocol"],
        "model": task["model"],
        "model_tag": task["model_tag"],
        "dataset": task["dataset"],
        "outer_seed": int(task["seed"]),
        "split_seed": int(task["seed"]),
        "split_sha256": split_metadata["split_sha256"],
        "method": method,
        "method_seed": int(task["seed"]) + METHOD_SEED_OFFSETS[method],
        "layer": int(layer),
        "hyperparameters": task["hyperparameters"],
        "embedding_sha256": task["embedding"]["sha256"],
        "embedding_n_pos": int(task["embedding"]["n_pos"]),
        "embedding_n_neg": int(task["embedding"]["n_neg"]),
        "embedding_layers": int(task["embedding"]["layers"]),
        "embedding_dim": int(task["embedding"]["dim"]),
        "code_version": task["code_version"],
        "environment_sha256": task["environment_sha256"],
    }


def checkpoint_path(seed_dir: Path, method: str, layer: int) -> Path:
    return seed_dir / "checkpoints" / method / f"layer_{layer}.npz"


def checkpoint_payload_sha256(arrays, metadata):
    return canonical_sha256({
        "arrays": {key: array_sha256(value) for key, value in sorted(arrays.items())},
        "metadata": {key: value for key, value in metadata.items() if key != "payload_sha256"},
    })


def validate_checkpoint(
    path: Path,
    expected_identity: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            validation_accuracy = float(np.asarray(data["validation_accuracy"]).item())
            test_accuracy = float(np.asarray(data["test_accuracy"]).item())
            concept = np.asarray(data["concept"])
            if metadata.get("payload_sha256") is not None:
                arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
                if metadata["payload_sha256"] != checkpoint_payload_sha256(arrays, metadata):
                    return None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if metadata.get("status") != "complete":
        return None
    if metadata.get("identity") != dict(expected_identity):
        return None
    if not (
        np.isfinite(validation_accuracy)
        and np.isfinite(test_accuracy)
        and 0.0 <= validation_accuracy <= 1.0
        and 0.0 <= test_accuracy <= 1.0
    ):
        return None
    return {
        "metadata": metadata,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "concept": concept,
    }


def _reject_checkpoint(path: Path) -> None:
    if not path.exists():
        return
    rejected = path.with_name(f"{path.name}.rejected.{int(time.time())}.{os.getpid()}")
    os.replace(path, rejected)
    _fsync_directory(path.parent)


def _embedding_stat_matches(task: Mapping[str, Any]) -> Path:
    from raptor.experiments.accuracy_preparation import verify_prepared_embedding
    verify_prepared_embedding(task["embedding"], model=task["model"], dataset=task["dataset"],
                              synthetic=task.get("scope", task["embedding"].get("scope")) == "synthetic")
    path = Path(task["embedding"]["path"])
    stat = path.stat()
    if int(stat.st_size) != int(task["embedding"]["size"]):
        raise ValueError(f"embedding size changed after manifest creation: {path}")
    if int(stat.st_mtime_ns) != int(task["embedding"]["mtime_ns"]):
        raise ValueError(f"embedding mtime changed after manifest creation: {path}")
    return path


def _signal_handler(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(
        f"[accuracy] received signal={signum}; "
        "will stop after the current atomic layer checkpoint",
        flush=True,
    )


def install_signal_handlers() -> None:
    signal.signal(signal.SIGUSR1, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def execute_task(
    task: Mapping[str, Any],
    *,
    method_runners: Optional[Mapping[str, Callable[..., dict[str, Any]]]] = None,
) -> dict[str, int]:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    method = str(task["method"])
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    current_version = source_version()
    if current_version["source_sha256"] != task["code_version"]["source_sha256"]:
        raise ValueError(
            "source tree changed after task-manifest creation: "
            f"expected={task['code_version']} actual={current_version}"
        )
    current_environment = environment_identity()
    if current_environment["sha256"] != task["environment_sha256"]:
        raise ValueError(
            "environment changed after task-manifest creation: "
            f"expected={task['environment_sha256']} "
            f"actual={current_environment['sha256']}"
        )
    emb_path = _embedding_stat_matches(task)
    seed_dir = seed_directory(
        Path(task["results_root"]),
        str(task["protocol"]),
        str(task["model_tag"]),
        str(task["dataset"]),
        int(task["seed"]),
    )
    split_path = seed_dir / "splits.npz"

    with np.load(emb_path, allow_pickle=False) as embeddings:
        n_pos = int(task["embedding"]["n_pos"])
        n_neg = int(task["embedding"]["n_neg"])
        y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
        groups = None
        if task["protocol"] == GROUPED_PROTOCOL:
            groups = _load_group_ids(
                str(task["dataset"]),
                n_pos=n_pos,
                n_neg=n_neg,
                data_path=Path(task["group_data_path"]) if task.get("group_data_path") else None,
            )
        train_sub_idx, val_idx, test_idx, split_metadata = load_split_artifact(
            split_path,
            y=y,
            groups=groups,
            expected={
                "protocol": task["protocol"],
                "model": task["model"],
                "dataset": task["dataset"],
                "split_seed": int(task["seed"]),
                "embedding_sha256": task["embedding"]["sha256"],
            },
        )
        runners = dict(method_runners or METHOD_RUNNERS)
        runner = runners[method]
        computed = 0
        restored = 0
        for layer in [int(value) for value in task["layers"]]:
            identity = _checkpoint_identity(
                task,
                layer=layer,
                split_metadata=split_metadata,
            )
            out_path = checkpoint_path(seed_dir, method, layer)
            restored_payload = validate_checkpoint(out_path, identity)
            if restored_payload is not None:
                restored += 1
                print(
                    f"[accuracy][restored] method={method} layer={layer} "
                    f"path={out_path}",
                    flush=True,
                )
                if _STOP_REQUESTED:
                    break
                continue
            if out_path.exists():
                _reject_checkpoint(out_path)
                print(
                    f"[accuracy][rejected] incompatible checkpoint layer={layer}",
                    flush=True,
                )
            pos = np.asarray(embeddings[f"X_pos_{layer}"])
            neg = np.asarray(embeddings[f"X_neg_{layer}"])
            X = np.vstack([pos, neg])
            if X.ndim != 2 or len(X) != len(y):
                raise ValueError(
                    f"embedding shape mismatch at layer={layer}: {X.shape}"
                )
            started = time.perf_counter()
            def save_candidate_decision(decision: dict[str, Any]) -> None:
                atomic_savez(
                    out_path.with_suffix(".selection.npz"),
                    **{key: np.asarray(value) for key, value in decision.items()},
                    metadata_json=np.asarray(json.dumps({
                        "status": "selection_frozen_before_refit", "identity": identity,
                        "validation_source": PRE_REFIT_VALIDATION_SOURCE,
                        "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
                    }, sort_keys=True)),
                )

            # The real RAPTOR runner persists its train-only candidate before
            # the final refit. Other methods keep their existing no-refit flow.
            runner_options = (
                {"selection_callback": save_candidate_decision}
                if runner in (_run_singlelr_layer, _run_fixed_reg_layer, _run_midpoint_layer) else {}
            )
            result = runner(
                X,
                y,
                train_sub_idx,
                val_idx,
                test_idx,
                seed=int(task["seed"]) + METHOD_SEED_OFFSETS[method],
                hyperparameters=task["hyperparameters"],
                **runner_options,
            )
            validation_accuracy = float(result["validation_accuracy"])
            test_accuracy = float(result["test_accuracy"])
            if not (
                np.isfinite(validation_accuracy)
                and np.isfinite(test_accuracy)
                and 0.0 <= validation_accuracy <= 1.0
                and 0.0 <= test_accuracy <= 1.0
            ):
                raise ValueError(
                    f"invalid accuracy at layer={layer}: "
                    f"validation={validation_accuracy} test={test_accuracy}"
                )
            metadata = {
                "status": "complete",
                "identity": identity,
                "method_details": result.get("method_details", {}),
                "observed_code_version": current_version,
                "elapsed_seconds": time.perf_counter() - started,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "slurm": {
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                    "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                    "restart_count": os.environ.get("SLURM_RESTART_COUNT"),
                },
            }
            arrays = {
                "validation_accuracy": np.asarray(validation_accuracy),
                "validation_accuracy_pre_refit": np.asarray(validation_accuracy),
                "validation_source": np.asarray(PRE_REFIT_VALIDATION_SOURCE),
                "validation_protocol_version": np.asarray(VALIDATION_PROTOCOL_VERSION),
                "test_accuracy": np.asarray(test_accuracy),
                "concept": np.asarray(result.get("concept", np.array([], dtype=np.float32))),
                **{key: np.asarray(value) for key, value in result.get("candidate_parameters", {}).items()},
            }
            metadata["payload_sha256"] = checkpoint_payload_sha256(arrays, metadata)
            atomic_savez(out_path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
            if validate_checkpoint(out_path, identity) is None:
                raise RuntimeError(f"checkpoint verification failed: {out_path}")
            computed += 1
            print(
                f"[accuracy][computed] method={method} layer={layer} "
                f"validation={validation_accuracy:.6f} "
                f"test={test_accuracy:.6f} path={out_path}",
                flush=True,
            )
            if _STOP_REQUESTED:
                break
    return {
        "computed": computed,
        "restored": restored,
        "stop_requested": int(_STOP_REQUESTED),
    }


def _load_jsonl_record(path: Path, index: int) -> dict[str, Any]:
    if index < 0:
        raise ValueError("task index must be non-negative")
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            if index != 0:
                raise IndexError(f"task index {index} is outside {path}")
            return payload
        if isinstance(payload, list) and index < len(payload):
            return payload[index]
        raise IndexError(f"task index {index} is outside {path}")
    with path.open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise IndexError(f"task index {index} is outside {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()
    install_signal_handlers()
    task = _load_jsonl_record(args.task_file, args.task_index)
    outcome = execute_task(task)
    print(f"[accuracy] task outcome={json.dumps(outcome, sort_keys=True)}")
    if outcome["stop_requested"]:
        raise SystemExit(75)


if __name__ == "__main__":
    main()
