"""Validate embeddings, create shared splits, and build accuracy task arrays."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.lib import format as npy_format

from raptor.core import DEFAULT_DATASETS, DEFAULT_MODELS, model_tag
from raptor.embedding_cache import EXPECTED_LAYERS
from raptor.experiments.accuracy_multiseed import (
    DEFAULT_OUTER_SEEDS,
    GROUPED_PROTOCOL,
    METHODS,
    GROUPED_METHODS,
    METHOD_HPARAMETERS,
    ROW_PROTOCOL,
    atomic_json,
    ensure_split_artifact,
    environment_identity,
    file_sha256,
    parse_seeds,
    seed_directory,
    source_version,
    _load_group_ids,
)



def _npy_shape(npz: np.lib.npyio.NpzFile, key: str) -> tuple[tuple[int, ...], np.dtype]:
    member = f"{key}.npy"
    try:
        handle = npz.zip.open(member)
    except KeyError as exc:
        raise ValueError(f"missing array {key}") from exc
    with handle:
        version = npy_format.read_magic(handle)
        shape, _fortran, dtype = npy_format._read_array_header(handle, version)
    return tuple(int(value) for value in shape), np.dtype(dtype)


def validate_layer_set(model: str, layers: Iterable[int], *, synthetic: bool = False) -> None:
    """Require the physical model inventory, independent of a cache's own claims.

    Historical caches may lack metadata, but a known model still has a fixed
    physical layer count. Custom tiny inputs require explicit synthetic scope;
    that scope never shortens a known benchmark model.
    """
    layers = list(layers)
    expected = EXPECTED_LAYERS.get(model)
    if expected is not None:
        if layers != list(range(expected)):
            raise ValueError(f"{model}: expected all {expected} physical layers")
    elif not synthetic:
        raise ValueError(f"unknown model {model!r}: explicit synthetic scope is required")
    elif not layers or layers != list(range(len(layers))):
        raise ValueError("synthetic cache requires consecutive matching layers from zero")


def validate_embedding_metadata(data, *, model: str, dataset: str, layers: int, dim: int) -> None:
    """Check existing scalar metadata without loading large layer arrays."""
    for key, expected in (("meta_model", model), ("meta_dataset", dataset),
                          ("meta_layers", layers), ("meta_dim", dim)):
        if key not in data.files:
            continue  # Older benchmark inputs are bound by the explicit model.
        value = np.asarray(data[key])
        if value.size != 1 or value.reshape(-1)[0].item() != expected:
            raise ValueError(f"{key} differs from the requested model/dataset or actual cache layout")
    if "meta_identity_json" in data.files:
        identity = json.loads(np.asarray(data["meta_identity_json"]).item())
        if identity.get("model") != model or identity.get("dataset") != dataset:
            raise ValueError("extraction model/dataset identity differs")


def validate_embedding_record(embedding, *, model: str, dataset: str,
                              synthetic: bool = False) -> None:
    """Reject shortened formal inventories even in an old prepared manifest."""
    for key, expected in (("model", model), ("dataset", dataset)):
        if key in embedding and embedding[key] != expected:
            raise ValueError(f"prepared embedding {key} identity differs")
    count = embedding["layers"]
    if type(count) is not int or count < 1:
        raise ValueError("prepared embedding layer count must be a positive integer")
    validate_layer_set(model, range(count), synthetic=synthetic)


def verify_prepared_embedding(embedding, *, model: str, dataset: str,
                              synthetic: bool = False) -> None:
    """Check headers/metadata against a prepared record, without another hash."""
    validate_embedding_record(embedding, model=model, dataset=dataset, synthetic=synthetic)
    observed = inspect_embedding(Path(embedding["path"]), model, dataset,
                                 synthetic=synthetic, hash_content=False)
    for key in ("layers", "n_pos", "n_neg", "dim"):
        if observed[key] != embedding[key]:
            raise ValueError(f"prepared embedding {key} differs from actual cache")


def inspect_embedding(path: Path, model: str, dataset: str, *, synthetic: bool = False,
                      hash_content: bool = True) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing embedding: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            pos_layers = sorted(
                int(key.rsplit("_", 1)[1])
                for key in data.files
                if key.startswith("X_pos_")
            )
            neg_layers = sorted(
                int(key.rsplit("_", 1)[1])
                for key in data.files
                if key.startswith("X_neg_")
            )
            if not pos_layers or pos_layers != list(range(len(pos_layers))):
                raise ValueError(f"non-contiguous positive layers in {path}")
            if pos_layers != neg_layers:
                raise ValueError(f"positive/negative layer mismatch in {path}")
            validate_layer_set(model, pos_layers, synthetic=synthetic)
            n_pos = n_neg = dim = None
            dtype_names: set[str] = set()
            for layer in pos_layers:
                pos_shape, pos_dtype = _npy_shape(data, f"X_pos_{layer}")
                neg_shape, neg_dtype = _npy_shape(data, f"X_neg_{layer}")
                if len(pos_shape) != 2 or len(neg_shape) != 2:
                    raise ValueError(f"embedding layer {layer} is not 2D in {path}")
                if pos_shape[1] != neg_shape[1]:
                    raise ValueError(f"embedding dimension mismatch at layer {layer}")
                if n_pos is None:
                    n_pos, n_neg, dim = pos_shape[0], neg_shape[0], pos_shape[1]
                if (pos_shape[0], neg_shape[0], pos_shape[1]) != (
                    n_pos,
                    n_neg,
                    dim,
                ):
                    raise ValueError(f"inconsistent embedding shape at layer {layer}")
                dtype_names.update([str(pos_dtype), str(neg_dtype)])
            if n_pos is None or n_neg is None or dim is None:
                raise ValueError(f"empty embedding: {path}")
            if min(n_pos, n_neg) < 2:
                raise ValueError(f"embedding has insufficient class rows: {path}")
            if dim < 1 or not all(np.issubdtype(np.dtype(name), np.floating) for name in dtype_names):
                raise ValueError("embedding requires nonempty floating-point feature arrays")
            validate_embedding_metadata(data, model=model, dataset=dataset,
                                        layers=len(pos_layers), dim=dim)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid embedding {path}: {exc}") from exc
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "model": model,
        "dataset": dataset,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        **({"sha256": file_sha256(path)} if hash_content else {}),
        "scope": "synthetic" if synthetic else "benchmark",
        "layers": len(pos_layers),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "dim": int(dim),
        "dtypes": sorted(dtype_names),
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return count


def _chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _parse_list(value: str, defaults: list[str]) -> list[str]:
    if value.strip().lower() == "all":
        return list(defaults)
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("list argument cannot be empty")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", type=Path, default=Path("embeddings_all"))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("exp_results/accuracy_multiseed"),
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("logs/accuracy_tasks"),
    )
    parser.add_argument("--models", default="all")
    parser.add_argument("--synthetic-canary", action="store_true",
                        help="Explicitly label custom tiny inputs; never shortens a benchmark model.")
    parser.add_argument("--datasets", default="all")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_OUTER_SEEDS),
    )
    parser.add_argument("--skip-models", default="")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--protocols", default=ROW_PROTOCOL+","+GROUPED_PROTOCOL)
    parser.add_argument("--group-data-dir", type=Path, default=None,
                        help="Directory containing counterfact.csv and cities.csv for grouped IDs.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument(
        "--layers-per-task",
        type=int,
        default=None,
        help="Override every method-specific layer shard size.",
    )
    parser.add_argument("--singlelr-layers-per-task", type=int, default=4)
    parser.add_argument("--xrfm-layers-per-task", type=int, default=1)
    parser.add_argument("--gcs-layers-per-task", type=int, default=2)
    parser.add_argument("--max-array-size", type=int, default=10000)
    args = parser.parse_args()

    method_chunk_sizes = {
        "singlelr": args.singlelr_layers_per_task,
        "xrfm": args.xrfm_layers_per_task,
        "gcs": args.gcs_layers_per_task,
        "fixed_reg": args.singlelr_layers_per_task,
        "dim_midpoint": args.singlelr_layers_per_task,
    }
    if args.layers_per_task is not None:
        method_chunk_sizes = {method: args.layers_per_task for method in METHODS}
    if any(size < 1 for size in method_chunk_sizes.values()):
        raise ValueError("layers-per-task values must be positive")
    if args.max_array_size < 1:
        raise ValueError("--max-array-size must be positive")
    methods = _parse_list(args.methods, list(METHODS))
    if len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError("unknown or duplicate methods")
    requested_protocols = _parse_list(args.protocols, [ROW_PROTOCOL, GROUPED_PROTOCOL])
    if set(requested_protocols) - {ROW_PROTOCOL, GROUPED_PROTOCOL}:
        raise ValueError("unknown protocols")
    models = _parse_list(args.models, DEFAULT_MODELS)
    datasets = _parse_list(args.datasets, DEFAULT_DATASETS)
    seeds = parse_seeds(args.seeds)
    skip_models = {item.strip() for item in args.skip_models.split(",") if item.strip()}
    unknown_models = set(models).difference(DEFAULT_MODELS)
    unknown_datasets = set(datasets).difference(DEFAULT_DATASETS)
    unknown_skips = skip_models.difference(models)
    if unknown_models and not args.synthetic_canary:
        raise ValueError(f"unknown models: {sorted(unknown_models)}")
    if unknown_datasets and not args.synthetic_canary:
        raise ValueError(f"unknown datasets: {sorted(unknown_datasets)}")
    if unknown_skips:
        raise ValueError(f"--skip-models not selected: {sorted(unknown_skips)}")

    active_models = [model for model in models if model not in skip_models]
    code_version = source_version()
    environment = environment_identity()
    embedding_map: dict[tuple[str, str], dict[str, Any]] = {}
    missing: list[dict[str, str]] = []
    for model in active_models:
        tag = model_tag(model)
        for dataset in datasets:
            path = args.emb_dir / f"{tag}_{dataset}_embeddings.npz"
            if not path.is_file():
                missing.append({"model": model, "dataset": dataset, "path": str(path)})
                continue
            embedding_map[(model, dataset)] = inspect_embedding(
                path,
                model,
                dataset,
                synthetic=args.synthetic_canary,
            )
    if missing:
        details = "\n".join(
            f"  {entry['model']} / {entry['dataset']}: {entry['path']}"
            for entry in missing
        )
        raise FileNotFoundError(
            "cannot prepare accuracy arrays until every active embedding is "
            f"present and valid:\n{details}"
        )

    protocols = {
        ROW_PROTOCOL: list(datasets),
        GROUPED_PROTOCOL: [
            dataset for dataset in datasets if dataset in {"counterfact", "cities"}
        ],
    }
    protocols = {key: value for key, value in protocols.items() if key in requested_protocols}
    methods_by_protocol = {p: [m for m in methods if p != GROUPED_PROTOCOL or m in GROUPED_METHODS] for p in protocols}
    if any(not values for values in methods_by_protocol.values()):
        raise ValueError("no requested supported methods for protocol")
    tasks_by_class: dict[str, list[dict[str, Any]]] = {}
    split_records: list[dict[str, Any]] = []
    checkpoint_targets: set[str] = set()

    for protocol, protocol_datasets in protocols.items():
        for model in active_models:
            tag = model_tag(model)
            for dataset in protocol_datasets:
                embedding = embedding_map[(model, dataset)]
                y = np.concatenate(
                    [
                        np.ones(embedding["n_pos"], dtype=int),
                        np.zeros(embedding["n_neg"], dtype=int),
                    ]
                )
                groups = None
                if protocol == GROUPED_PROTOCOL:
                    groups = _load_group_ids(
                        dataset,
                        n_pos=embedding["n_pos"],
                        n_neg=embedding["n_neg"],
                        data_path=args.group_data_dir / f"{dataset}.csv" if args.group_data_dir else None,
                    )
                for seed in seeds:
                    out_dir = seed_directory(
                        args.results_root,
                        protocol,
                        tag,
                        dataset,
                        seed,
                    )
                    split_path = out_dir / "splits.npz"
                    split_metadata = ensure_split_artifact(
                        split_path,
                        y=y,
                        protocol=protocol,
                        seed=seed,
                        test_size=args.test_size,
                        val_size=args.val_size,
                        model=model,
                        dataset=dataset,
                        embedding=embedding,
                        groups=groups,
                    )
                    split_records.append(
                        {
                            "path": str(split_path),
                            "model": model,
                            "dataset": dataset,
                            "protocol": protocol,
                            "seed": seed,
                            "split_sha256": split_metadata["split_sha256"],
                            "group_data_path": str((args.group_data_dir / f"{dataset}.csv").resolve()) if args.group_data_dir and groups is not None else None,
                            "partition_sizes": {
                                "train_sub": split_metadata["n_train_sub"],
                                "validation": split_metadata["n_validation"],
                                "test": split_metadata["n_test"],
                            },
                        }
                    )
                    run_meta_path = out_dir / "run_meta.json"
                    prior_meta: dict[str, Any] = {}
                    if run_meta_path.is_file():
                        try:
                            with run_meta_path.open(
                                "r",
                                encoding="utf-8",
                            ) as handle:
                                prior_meta = json.load(handle)
                        except (OSError, json.JSONDecodeError):
                            prior_meta = {}
                    preserve_status = (
                        prior_meta.get("status", "planned")
                        if prior_meta.get("code_version") == code_version
                        and prior_meta.get("split", {}).get("split_sha256")
                        == split_metadata["split_sha256"]
                        else "planned"
                    )
                    new_run_meta = {
                        "scope": "synthetic" if args.synthetic_canary else "benchmark",
                        "schema_version": 1,
                        "status": preserve_status,
                        "experiment": "Table 1 five-outer-seed rebuttal",
                        "sensitivity_label": (
                            "grouped-split sensitivity"
                            if protocol == GROUPED_PROTOCOL
                            else None
                        ),
                        "protocol": protocol,
                        "model": model,
                        "model_tag": tag,
                        "dataset": dataset,
                        "outer_seed": seed,
                        "split": split_metadata,
                        "embedding": embedding,
                        "methods": methods_by_protocol[protocol],
                        "layers": list(range(embedding["layers"])),
                        "code_version": code_version,
                        "environment_sha256": environment["sha256"],
                    }
                    for key in ("method_completion", "aggregated_at"):
                        if key in prior_meta and preserve_status != "planned":
                            new_run_meta[key] = prior_meta[key]
                    atomic_json(run_meta_path, new_run_meta)
                    scale = "large" if embedding["dim"] >= 7000 else "standard"
                    for method in methods_by_protocol[protocol]:
                        task_class = f"{method}_{scale}"
                        for layers in _chunks(
                            list(range(embedding["layers"])),
                            method_chunk_sizes[method],
                        ):
                            task = {
                                "scope": "synthetic" if args.synthetic_canary else "benchmark",
                                "schema_version": 1,
                                "protocol": protocol,
                                "model": model,
                                "model_tag": tag,
                                "dataset": dataset,
                                "seed": seed,
                                "method": method,
                                "layers": layers,
                                "hyperparameters": METHOD_HPARAMETERS[method],
                                "group_data_path": str((args.group_data_dir / f"{dataset}.csv").resolve()) if args.group_data_dir and groups is not None else None,
                                "embedding": embedding,
                                "results_root": str(args.results_root.resolve()),
                                "code_version": code_version,
                                "environment_sha256": environment["sha256"],
                            }
                            tasks_by_class.setdefault(task_class, []).append(task)
                            for layer in layers:
                                target = (
                                    out_dir
                                    / "checkpoints"
                                    / method
                                    / f"layer_{layer}.npz"
                                )
                                target_key = str(target.resolve())
                                if target_key in checkpoint_targets:
                                    raise ValueError(
                                        f"duplicate output target: {target_key}"
                                    )
                                checkpoint_targets.add(target_key)

    args.task_dir.mkdir(parents=True, exist_ok=True)
    task_files: dict[str, dict[str, Any]] = {}
    for task_class in sorted(tasks_by_class):
        method, scale = task_class.rsplit("_", 1)
        for part, rows in enumerate(
            _chunks(tasks_by_class[task_class], args.max_array_size)
        ):
            batch_name = f"{task_class}_part{part:03d}"
            path = args.task_dir / f"{batch_name}.jsonl"
            count = _write_jsonl(path, rows)
            task_files[batch_name] = {
                "path": str(path.resolve()),
                "count": count,
                "method": method,
                "scale": scale,
                "resource_class": task_class,
                "sha256": file_sha256(path),
            }

    blocked = [
        {
            "model": model,
            "datasets": list(datasets),
            "reason": "explicitly excluded: model access unavailable",
        }
        for model in sorted(skip_models)
    ]
    manifest = {
        "scope": "synthetic" if args.synthetic_canary else "benchmark",
        "schema_version": 1,
        "status": "validated",
        "experiment": "Table 1 five-outer-seed rebuttal",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable] + sys.argv,
        "environment": environment,
        "code_version": code_version,
        "protocol": {
            "outer_seeds": seeds,
            "protocols": requested_protocols,
            "methods_by_protocol": methods_by_protocol,
            "test_size": args.test_size,
            "val_size_within_non_test": args.val_size,
            "tie_rule": "lowest layer index among exact validation-accuracy ties",
            "primary": ROW_PROTOCOL,
            "secondary": "grouped-split sensitivity",
            "group_columns": {"counterfact": "subject", "cities": "city"},
            "layers_per_task": method_chunk_sizes,
            "max_array_size": args.max_array_size,
        },
        "grid": {
            "requested_models": models,
            "active_models": active_models,
            "datasets": datasets,
            "methods": methods,
            "blocked": blocked,
        },
        "embeddings": list(embedding_map.values()),
        "splits": split_records,
        "task_files": task_files,
        "total_tasks": sum(entry["count"] for entry in task_files.values()),
        "total_layer_checkpoints": len(checkpoint_targets),
        "output_root": str(args.results_root.resolve()),
    }
    atomic_json(args.task_dir / "manifest.json", manifest)
    print(json.dumps(manifest["task_files"], indent=2))
    print(
        f"validated embeddings={len(embedding_map)} "
        f"splits={len(split_records)} "
        f"tasks={manifest['total_tasks']} "
        f"layer_checkpoints={manifest['total_layer_checkpoints']}"
    )


if __name__ == "__main__":
    main()
