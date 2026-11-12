"""Aggregate outer-seed accuracy checkpoints without consulting test for selection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from scipy.stats import t as student_t

from raptor.probes.tuning import PRE_REFIT_VALIDATION_SOURCE, VALIDATION_PROTOCOL_VERSION

from raptor.experiments.accuracy_multiseed import (
    GROUPED_PROTOCOL,
    protocol_methods,
    checkpoint_payload_sha256,
    METHODS,
    RESULT_FILENAMES,
    ROW_PROTOCOL,
    _checkpoint_identity,
    _load_group_ids,
    array_sha256,
    atomic_json,
    atomic_savez,
    checkpoint_path,
    load_split_artifact,
    seed_directory,
    select_validation_layer,
)


def sample_summary(
    values_by_seed: Mapping[int, float], seeds: list[int]
) -> dict[str, Any]:
    ordered = [
        {"seed": int(seed), "value": float(values_by_seed[seed])}
        for seed in seeds
        if seed in values_by_seed
    ]
    values = np.asarray([entry["value"] for entry in ordered], dtype=float)
    complete = len(ordered) == len(seeds)
    return {
        "requested_seeds": [int(seed) for seed in seeds],
        "seed_values": ordered,
        "completed_seeds": [entry["seed"] for entry in ordered],
        "n_completed": len(ordered),
        "status": "complete" if complete else "incomplete",
        "mean": float(np.mean(values)) if len(values) else None,
        "sample_std_ddof1": (
            float(np.std(values, ddof=1)) if len(values) >= 2 else None
        ),
    }


def paired_summary(
    left_by_seed: Mapping[int, float],
    right_by_seed: Mapping[int, float],
    seeds: list[int],
    *,
    comparison: str,
) -> dict[str, Any]:
    common = [seed for seed in seeds if seed in left_by_seed and seed in right_by_seed]
    differences = {
        int(seed): float(left_by_seed[seed] - right_by_seed[seed]) for seed in common
    }
    summary = sample_summary(differences, seeds)
    n = summary["n_completed"]
    mean = summary["mean"]
    std = summary["sample_std_ddof1"]
    if n >= 2 and mean is not None and std is not None:
        critical = float(student_t.ppf(0.975, df=n - 1))
        half_width = critical * float(std) / math.sqrt(n)
        interval: Optional[dict[str, Any]] = {
            "level": 0.95,
            "method": "two-sided Student t interval over paired outer-seed differences",
            "degrees_of_freedom": n - 1,
            "lower": float(mean - half_width),
            "upper": float(mean + half_width),
        }
    else:
        interval = None
    summary.update(
        {
            "comparison": comparison,
            "difference_direction": "RAPTOR minus baseline",
            "mean_paired_difference": mean,
            "confidence_interval_95": interval,
        }
    )
    return summary


def _read_layer_checkpoint(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            val_acc = float(np.asarray(data["validation_accuracy"]).item())
            test_acc = float(np.asarray(data["test_accuracy"]).item())
            concept = np.asarray(data["concept"])
            if metadata.get("payload_sha256") is not None:
                arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
                if metadata["payload_sha256"] != checkpoint_payload_sha256(arrays, metadata):
                    raise ValueError("accuracy checkpoint payload checksum mismatch")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid accuracy checkpoint: {path}") from exc
    if metadata.get("status") != "complete":
        raise ValueError(f"incomplete accuracy checkpoint: {path}")
    if not (
        np.isfinite(val_acc)
        and np.isfinite(test_acc)
        and 0.0 <= val_acc <= 1.0
        and 0.0 <= test_acc <= 1.0
    ):
        raise ValueError(f"invalid accuracy values in {path}")
    return {
        "metadata": metadata,
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc,
        "concept": concept,
    }


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["status"]
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _frozen_tasks(manifest: Mapping[str, Any]) -> dict[tuple, Mapping[str, Any]]:
    """Read scientific identities from the frozen task files, never from CPs."""
    if not manifest.get("task_files") or not manifest.get("environment", {}).get("sha256"):
        raise ValueError("aggregation requires frozen task_files and environment identity")
    embeddings = {(e["model"], e["dataset"]): e for e in manifest["embeddings"]}
    tasks: dict[tuple, Mapping[str, Any]] = {}
    for entry in manifest["task_files"].values():
        path = Path(entry["path"])
        raw = path.read_bytes()
        if entry.get("sha256") and hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"frozen task file checksum mismatch: {path}")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        if len(rows) != int(entry["count"]):
            raise ValueError(f"frozen task file count mismatch: {path}")
        for task in rows:
            model, dataset, seed = task["model"], task["dataset"], int(task["seed"])
            # A private aggregation may restrict a larger frozen grid. Paths
            # remain the original task paths; only the output root is redirected.
            if (model not in manifest["grid"]["active_models"]
                    or dataset not in manifest["grid"]["datasets"]
                    or seed not in manifest["protocol"]["outer_seeds"]):
                continue
            method = task["method"]
            if (task.get("schema_version") != manifest.get("schema_version", 1)
                    or method not in protocol_methods(manifest, task["protocol"]) or entry.get("method", method) != method
                    or task["protocol"] not in (ROW_PROTOCOL, GROUPED_PROTOCOL)
                    or task["code_version"] != manifest["code_version"]
                    or task["environment_sha256"] != manifest["environment"]["sha256"]
                    or task["embedding"] != embeddings[(model, dataset)]):
                raise ValueError(f"frozen task differs from manifest: {path}")
            layers = task["layers"]
            if (len(set(layers)) != len(layers) or not layers
                    or any(not isinstance(layer, int) or not 0 <= layer < task["embedding"]["layers"]
                           for layer in layers)):
                raise ValueError(f"invalid frozen task layers: {path}")
            for layer in layers:
                key = (task["protocol"], model, dataset, seed, method, layer)
                if key in tasks:
                    raise ValueError(f"duplicate frozen task unit: {key}")
                tasks[key] = task
    return tasks


def _persisted_split(manifest, protocol, model, dataset, seed, embedding):
    records = [r for r in manifest.get("splits", [])
               if (r["protocol"], r["model"], r["dataset"], int(r["seed"]))
               == (protocol, model, dataset, seed)]
    if len(records) != 1:
        raise ValueError("exactly one frozen split record is required")
    record = records[0]
    y = np.r_[np.ones(embedding["n_pos"], dtype=int), np.zeros(embedding["n_neg"], dtype=int)]
    groups = (_load_group_ids(dataset, n_pos=embedding["n_pos"], n_neg=embedding["n_neg"],
                              data_path=Path(record["group_data_path"]) if record.get("group_data_path") else None)
              if protocol == GROUPED_PROTOCOL else None)
    _, _, _, metadata = load_split_artifact(
        Path(record["path"]), y=y, groups=groups,
        expected={"protocol": protocol, "model": model, "dataset": dataset,
                  "split_seed": seed, "embedding_sha256": embedding["sha256"],
                  "split_sha256": record["split_sha256"],
                  "label_sha256": array_sha256(y),
                  "group_sha256": array_sha256(groups) if groups is not None else None,
                  "test_size": manifest["protocol"]["test_size"],
                  "val_size_within_non_test": manifest["protocol"]["val_size_within_non_test"]},
    )
    return metadata


def _verify_task_identity(identity, task, layer, split, manifest):
    expected = _checkpoint_identity(task, layer=layer, split_metadata=split)
    actual = dict(identity)
    # Earlier public artifacts predate this optional field. Preserve their
    # identity rather than relabelling their validation provenance.
    declared = {value for value in (
        task.get("validation_protocol_version"),
        manifest.get("validation_protocol_version"),
        manifest["protocol"].get("validation_protocol_version"),
    ) if value is not None}
    if len(declared) > 1:
        raise ValueError("conflicting frozen validation protocol versions")
    if declared and declared != {VALIDATION_PROTOCOL_VERSION}:
        raise ValueError("unknown frozen validation protocol version")
    if declared:
        expected["validation_protocol_version"] = next(iter(declared))
    else:
        expected.pop("validation_protocol_version", None)
        version = actual.pop("validation_protocol_version", None)
        if version not in (None, VALIDATION_PROTOCOL_VERSION):
            raise ValueError("unknown checkpoint validation protocol version")
    if actual != expected:
        keys = set(actual) | set(expected)
        mismatches = {key: (expected.get(key), actual.get(key))
                      for key in keys if expected.get(key) != actual.get(key)}
        raise ValueError(f"identity mismatch: {mismatches}")


def aggregate_accuracy(
    manifest: Mapping[str, Any],
    *,
    summary_json: Path,
    summary_csv: Path,
) -> dict[str, Any]:
    seeds = [int(seed) for seed in manifest["protocol"]["outer_seeds"]]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("outer seeds must be nonempty and unique")
    models = list(manifest["grid"]["active_models"])
    datasets = list(manifest["grid"]["datasets"])
    results_root = Path(manifest["output_root"])
    embedding_lookup = {
        (entry["model"], entry["dataset"]): entry for entry in manifest["embeddings"]
    }
    protocols = {
        ROW_PROTOCOL: datasets,
        GROUPED_PROTOCOL: [
            dataset for dataset in datasets if dataset in {"counterfact", "cities"}
        ],
    }
    requested_protocols = manifest["protocol"].get("protocols", list(protocols))
    protocols = {key: value for key, value in protocols.items() if key in requested_protocols}
    seed_metrics: dict[tuple[str, str, str, str, str], dict[int, float]] = {}
    cells: list[dict[str, Any]] = []
    layer_results: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    frozen_tasks = _frozen_tasks(manifest)

    for protocol, protocol_datasets in protocols.items():
        for model in models:
            model_entries = [
                entry for entry in manifest["embeddings"] if entry["model"] == model
            ]
            if not model_entries:
                raise ValueError(f"manifest has no embedding for active model={model}")
            tag = model_entries[0]["path"].rsplit("/", 1)[-1]
            suffixes = [f"_{dataset}_embeddings.npz" for dataset in protocol_datasets]
            model_tag = None
            for suffix in suffixes:
                if tag.endswith(suffix):
                    model_tag = tag[: -len(suffix)]
                    break
            if model_tag is None:
                model_tag = model.replace("/", "-")
            for dataset in protocol_datasets:
                embedding = embedding_lookup[(model, dataset)]
                expected_layers = list(range(int(embedding["layers"])))
                for seed in seeds:
                    out_dir = seed_directory(
                        results_root,
                        protocol,
                        model_tag,
                        dataset,
                        seed,
                    )
                    method_complete: dict[str, bool] = {}
                    method_split_sha: dict[str, str] = {}
                    try:
                        split = _persisted_split(manifest, protocol, model,
                                                 dataset, seed, embedding)
                        split_error = None
                    except (OSError, ValueError, KeyError) as exc:
                        split, split_error = None, f"invalid persisted split: {exc}"
                    methods = protocol_methods(manifest, protocol)
                    for method in methods:
                        layer_payloads = []
                        for layer in expected_layers:
                            path = checkpoint_path(out_dir, method, layer)
                            if not path.is_file():
                                missing.append(
                                    {
                                        "protocol": protocol,
                                        "model": model,
                                        "dataset": dataset,
                                        "seed": seed,
                                        "method": method,
                                        "layer": layer,
                                        "reason": "missing checkpoint",
                                    }
                                )
                                continue
                            try:
                                if split_error:
                                    raise ValueError(split_error)
                                task = frozen_tasks.get((protocol, model, dataset, seed, method, layer))
                                if task is None or task["model_tag"] != model_tag:
                                    raise ValueError("missing or incompatible frozen task unit")
                                payload = _read_layer_checkpoint(path)
                                identity = payload["metadata"]["identity"]
                                _verify_task_identity(identity, task, layer, split, manifest)
                                layer_payloads.append(payload)
                            except (ValueError, KeyError) as exc:
                                missing.append(
                                    {
                                        "protocol": protocol,
                                        "model": model,
                                        "dataset": dataset,
                                        "seed": seed,
                                        "method": method,
                                        "layer": layer,
                                        "reason": str(exc),
                                    }
                                )
                        method_complete[method] = len(layer_payloads) == len(
                            expected_layers
                        )
                        if not method_complete[method]:
                            continue
                        split_hashes = {
                            item["metadata"]["identity"]["split_sha256"]
                            for item in layer_payloads
                        }
                        if len(split_hashes) != 1:
                            raise ValueError(
                                f"layers use different splits for "
                                f"{protocol}/{model}/{dataset}/seed_{seed}/{method}"
                            )
                        method_split_sha[method] = next(iter(split_hashes))
                        layer_indices = [
                            int(item["metadata"]["identity"]["layer"])
                            for item in layer_payloads
                        ]
                        order = np.argsort(layer_indices)
                        layer_indices_array = np.asarray(layer_indices, dtype=int)[
                            order
                        ]
                        val_acc = np.asarray(
                            [item["validation_accuracy"] for item in layer_payloads],
                            dtype=float,
                        )[order]
                        test_acc = np.asarray(
                            [item["test_accuracy"] for item in layer_payloads],
                            dtype=float,
                        )[order]
                        selection = select_validation_layer(
                            layer_indices_array,
                            val_acc,
                            test_acc,
                        )
                        layer_results.append({"protocol": protocol, "model": model, "dataset": dataset,
                            "method": method, "seed": seed, "layer_indices": layer_indices_array.tolist(),
                            "validation_accuracy_pre_refit": val_acc.tolist(), "test_accuracy": test_acc.tolist(),
                            "split_sha256": method_split_sha[method]})
                        concepts = np.empty(len(layer_payloads), dtype=object)
                        for position, source_index in enumerate(order):
                            concepts[position] = layer_payloads[int(source_index)][
                                "concept"
                            ]
                        method_metadata = {
                            "schema_version": 1,
                            "status": "complete",
                            "protocol": protocol,
                            "sensitivity_label": (
                                "grouped-split sensitivity"
                                if protocol == GROUPED_PROTOCOL
                                else None
                            ),
                            "model": model,
                            "model_tag": model_tag,
                            "dataset": dataset,
                            "outer_seed": seed,
                            "method": method,
                            "split_sha256": layer_payloads[0]["metadata"]["identity"][
                                "split_sha256"
                            ],
                            "embedding_sha256": embedding["sha256"],
                            "code_version": manifest["code_version"],
                            "layer_selection_source": "outer validation accuracy only",
                            "tie_rule": manifest["protocol"]["tie_rule"],
                            **selection,
                        }
                        validation_is_pre_refit = all(
                            item["metadata"].get("method_details", {}).get("validation_source")
                            == PRE_REFIT_VALIDATION_SOURCE
                            for item in layer_payloads
                        )
                        method_metadata["validation_source"] = (
                            PRE_REFIT_VALIDATION_SOURCE if validation_is_pre_refit
                            else "legacy_source_audit_required"
                        )
                        # Concepts can have method-specific dimensions, so save
                        # them per layer as individually named numeric arrays.
                        save_payload: dict[str, Any] = {
                            "layer_indices": layer_indices_array,
                            "val_acc": val_acc,
                            "acc_test": test_acc,
                            "selected_layer": np.asarray(
                                selection["selected_layer"], dtype=int
                            ),
                            "selected_validation_accuracy": np.asarray(
                                selection["selected_validation_accuracy"]
                            ),
                            "best_layer_test_accuracy": np.asarray(
                                selection["best_layer_test_accuracy"]
                            ),
                            "average_layer_test_accuracy": np.asarray(
                                selection["average_layer_test_accuracy"]
                            ),
                            "metadata_json": np.asarray(
                                json.dumps(method_metadata, sort_keys=True)
                            ),
                        }
                        if validation_is_pre_refit:
                            save_payload.update(
                                val_acc_pre_refit=val_acc.copy(),
                                validation_source=np.asarray(PRE_REFIT_VALIDATION_SOURCE),
                                validation_protocol_version=np.asarray(VALIDATION_PROTOCOL_VERSION),
                            )
                        for layer, concept in zip(layer_indices_array, concepts):
                            save_payload[f"concept_{int(layer)}"] = concept
                        atomic_savez(
                            out_dir / RESULT_FILENAMES[method],
                            **save_payload,
                        )
                        for metric in (
                            "best_layer_test_accuracy",
                            "average_layer_test_accuracy",
                        ):
                            key = (protocol, model, dataset, method, metric)
                            seed_metrics.setdefault(key, {})[seed] = float(
                                selection[metric]
                            )
                    if (
                        len(method_split_sha) > 1
                        and len(set(method_split_sha.values())) != 1
                    ):
                        raise ValueError(
                            f"methods do not share a split for "
                            f"{protocol}/{model}/{dataset}/seed_{seed}: "
                            f"{method_split_sha}"
                        )
                    run_meta_path = out_dir / "run_meta.json"
                    if run_meta_path.is_file():
                        with run_meta_path.open("r", encoding="utf-8") as handle:
                            run_meta = json.load(handle)
                    else:
                        run_meta = {}
                    run_meta["status"] = (
                        "complete"
                        if all(method_complete.values())
                        and len(method_complete) == len(methods)
                        else "incomplete"
                    )
                    run_meta["method_completion"] = method_complete
                    run_meta["aggregated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    atomic_json(run_meta_path, run_meta)

    # Include zero-completion cells explicitly instead of omitting failed methods.
    for protocol, protocol_datasets in protocols.items():
        for model in models:
            for dataset in protocol_datasets:
                for method in protocol_methods(manifest, protocol):
                    for metric in ("best_layer_test_accuracy", "average_layer_test_accuracy"):
                        seed_metrics.setdefault((protocol, model, dataset, method, metric), {})
    csv_rows: list[dict[str, Any]] = []
    for key in sorted(seed_metrics):
        protocol, model, dataset, method, metric = key
        summary = sample_summary(seed_metrics[key], seeds)
        cell = {
            "protocol": protocol,
            "sensitivity_label": (
                "grouped-split sensitivity" if protocol == GROUPED_PROTOCOL else None
            ),
            "model": model,
            "dataset": dataset,
            "method": method,
            "metric": metric,
            **summary,
        }
        cells.append(cell)
        csv_rows.append(
            {
                "protocol": protocol,
                "model": model,
                "dataset": dataset,
                "method": method,
                "metric": metric,
                "status": summary["status"],
                "requested_seeds": json.dumps(summary["requested_seeds"]),
                "completed_seeds": json.dumps(summary["completed_seeds"]),
                "seed_values": json.dumps(summary["seed_values"]),
                "n_completed": summary["n_completed"],
                "mean": summary["mean"],
                "sample_std_ddof1": summary["sample_std_ddof1"],
            }
        )

    paired: list[dict[str, Any]] = []
    for protocol, protocol_datasets in protocols.items():
        for model in models:
            for dataset in protocol_datasets:
                for metric in (
                    "best_layer_test_accuracy",
                    "average_layer_test_accuracy",
                ):
                    raptor = seed_metrics.get(
                        (protocol, model, dataset, "singlelr", metric),
                        {},
                    )
                    for baseline in [method for method in protocol_methods(manifest, protocol) if method != "singlelr"]:
                        baseline_values = seed_metrics.get(
                            (protocol, model, dataset, baseline, metric),
                            {},
                        )
                        paired.append(
                            {
                                "protocol": protocol,
                                "sensitivity_label": (
                                    "grouped-split sensitivity"
                                    if protocol == GROUPED_PROTOCOL
                                    else None
                                ),
                                "model": model,
                                "dataset": dataset,
                                "metric": metric,
                                "baseline": baseline,
                                **paired_summary(
                                    raptor,
                                    baseline_values,
                                    seeds,
                                    comparison=f"singlelr_minus_{baseline}",
                                ),
                            }
                        )

    expected_cell_count = sum(len(models) * len(sets) * len(protocol_methods(manifest, protocol)) * 2
                              for protocol, sets in protocols.items())
    complete_cells = sum(cell["status"] == "complete" for cell in cells)
    status = (
        "complete"
        if complete_cells == expected_cell_count and not missing
        else "incomplete"
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "experiment": manifest["experiment"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requested_outer_seeds": seeds,
        "code_version": manifest["code_version"],
        "environment": manifest.get("environment"),
        "blocked": manifest["grid"].get("blocked", []),
        "coverage": {
            "expected_summary_cells": expected_cell_count,
            "complete_summary_cells": complete_cells,
            "missing_or_invalid_layer_checkpoints": len(missing),
        },
        "cells": cells,
        "layer_results": layer_results,
        "paired_comparisons": paired,
        "missing": missing,
    }
    atomic_json(summary_json, payload)
    _atomic_csv(summary_csv, csv_rows)
    return payload
