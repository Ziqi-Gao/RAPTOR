"""A consecutive cache prefix must never redefine a benchmark model's layers."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from raptor.core import model_tag
from raptor.embedding_cache import EXPECTED_LAYERS
from raptor.experiments import accuracy_preparation as preparation
from raptor.experiments import accuracy_multiseed as accuracy
from raptor.experiments import timing
from raptor.experiments.accuracy_aggregation import _frozen_tasks

MODEL = "Qwen/Qwen2.5-3B-Instruct"
ROOT = Path(__file__).resolve().parents[1]


def cache(tmp_path, *, model=MODEL, layers=36, metadata=True, **overrides):
    path = tmp_path / f"{model_tag(model)}_STSA_embeddings.npz"
    arrays = {f"X_{kind}_{layer}": np.ones((20, 4), dtype=np.float32)
              for layer in range(layers) for kind in ("pos", "neg")}
    if metadata:
        arrays.update(meta_model=np.asarray(model), meta_dataset=np.asarray("STSA"),
                      meta_layers=np.asarray(layers), meta_dim=np.asarray(4))
    arrays.update(overrides)
    np.savez(path, **arrays)
    return path


@pytest.mark.parametrize("model,expected", list(EXPECTED_LAYERS.items()))
def test_known_model_prefix_rejected_even_with_synthetic_scope(model, expected):
    assert expected > 2
    for synthetic in (False, True):
        with pytest.raises(ValueError, match=f"all {expected} physical layers"):
            preparation.validate_layer_set(model, range(2), synthetic=synthetic)


def test_review_reproduction_declared_36_but_two_actual_layers(tmp_path):
    path = cache(tmp_path, layers=2, meta_layers=np.asarray(36))
    with pytest.raises(ValueError, match="all 36 physical layers"):
        preparation.inspect_embedding(path, MODEL, "STSA")
    with pytest.raises(ValueError, match="all 36 physical layers"):
        timing.embedding_headers(path, model=MODEL, dataset="STSA")


@pytest.mark.parametrize("field,value", [
    ("meta_model", "google/gemma-7b-it"), ("meta_dataset", "cities"),
    ("meta_layers", 35), ("meta_dim", 5), ("meta_dim", [4, 4]),
])
def test_existing_metadata_cannot_disagree_with_actual_layout(tmp_path, field, value):
    path = cache(tmp_path, **{field: np.asarray(value)})
    with pytest.raises(ValueError, match=field):
        preparation.inspect_embedding(path, MODEL, "STSA")
    with pytest.raises(ValueError, match=field):
        timing.embedding_headers(path, model=MODEL, dataset="STSA")


def test_complete_legacy_cache_bound_to_known_model_without_metadata(tmp_path):
    path = cache(tmp_path, metadata=False)
    observed = preparation.inspect_embedding(path, MODEL, "STSA")
    assert observed["layers"] == 36 and observed["scope"] == "benchmark"
    # Header inventory and metadata checks never need another full content hash.
    assert "sha256" not in preparation.inspect_embedding(path, MODEL, "STSA", hash_content=False)


def test_custom_tiny_input_requires_explicit_synthetic_label(tmp_path):
    path = cache(tmp_path, model="synthetic", layers=2)
    with pytest.raises(ValueError, match="synthetic scope"):
        preparation.inspect_embedding(path, "synthetic", "STSA")
    observed = preparation.inspect_embedding(path, "synthetic", "STSA", synthetic=True)
    assert observed["scope"] == "synthetic" and observed["layers"] == 2


def test_accuracy_prepare_rejects_before_writing_tasks_or_splits(tmp_path, monkeypatch):
    cache(tmp_path, layers=2, meta_layers=np.asarray(36))
    monkeypatch.setattr(sys, "argv", ["prepare", "--emb-dir", str(tmp_path),
        "--models", MODEL, "--datasets", "STSA", "--protocols", "rowwise",
        "--task-dir", str(tmp_path / "tasks"), "--results-root", str(tmp_path / "results")])
    with pytest.raises(ValueError, match="all 36 physical layers"):
        preparation.main()
    assert not (tmp_path / "tasks").exists() and not (tmp_path / "results").exists()


def test_legacy_short_manifest_rejected_by_aggregation_and_task_execution(tmp_path):
    path = cache(tmp_path, layers=2)
    stat = path.stat()
    embedding = {"model": MODEL, "dataset": "STSA", "path": str(path), "layers": 2,
        "dim": 4, "n_pos": 20, "n_neg": 20, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    manifest = {"task_files": {"not_read": {}}, "environment": {"sha256": "x"}, "embeddings": [embedding]}
    with pytest.raises(ValueError, match="all 36 physical layers"):
        _frozen_tasks(manifest)
    with pytest.raises(ValueError, match="all 36 physical layers"):
        accuracy._embedding_stat_matches({"model": MODEL, "dataset": "STSA", "embedding": embedding})


def test_timing_config_cannot_override_known_model_physical_count(tmp_path, monkeypatch):
    cache(tmp_path, layers=2)
    config = {"models": [MODEL], "datasets": ["STSA"], "layer_counts": {MODEL: 2}}
    with pytest.raises(ValueError, match="all 36 physical layers"):
        timing.prepare_manifest(config, tmp_path, tmp_path / "run", scope="synthetic")


def test_explicit_synthetic_accuracy_cli_prepares_runs_and_aggregates(tmp_path):
    cache(tmp_path, model="synthetic", layers=2)
    tasks, results = tmp_path / "tasks", tmp_path / "results"
    def cli(script, *args):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                       cwd=ROOT, check=True, capture_output=True, text=True)
    cli("prepare_accuracy_multiseed.py", "--synthetic-canary", "--models", "synthetic",
        "--datasets", "STSA", "--methods", "dim_midpoint", "--protocols", "rowwise",
        "--emb-dir", tmp_path, "--task-dir", tasks, "--results-root", results)
    manifest = tasks / "manifest.json"
    assert json.loads(manifest.read_text())["scope"] == "synthetic"
    cli("run_accuracy.py", "--manifest", manifest, "--dry-run")
    cli("run_accuracy.py", "--manifest", manifest)
    cli("aggregate_accuracy_multiseed.py", "--manifest", manifest,
        "--out-json", tmp_path / "summary.json", "--out-csv", tmp_path / "summary.csv")
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["scope"] == "synthetic" and summary["status"] == "complete"
    assert summary["layer_results"][0]["layer_indices"] == [0, 1]


def _public_cli(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True)


def test_robustness_prepare_cli_rejects_known_model_prefix(tmp_path):
    cache(tmp_path, layers=2, meta_layers=np.asarray(36))
    output = tmp_path / "robustness"
    result = _public_cli("run_robustness.py", "prepare", "--emb-dir", tmp_path,
                         "--models", MODEL, "--datasets", "STSA", "--methods", "caa",
                         "--output", output)
    assert result.returncode != 0 and "all 36 physical layers" in result.stderr
    assert not output.exists()


def test_robustness_cli_rejects_legacy_short_manifest_before_work(tmp_path):
    cache(tmp_path)
    output = tmp_path / "robustness"
    prepared = _public_cli("run_robustness.py", "prepare", "--emb-dir", tmp_path,
                           "--models", MODEL, "--datasets", "STSA", "--methods", "caa",
                           "--output", output)
    assert prepared.returncode == 0, prepared.stderr
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Reconstruct an older manifest that mistook the observed two-layer prefix
    # for this model's entire inventory, keeping all its counts consistent.
    path = cache(tmp_path, layers=2)
    manifest["tasks"] = [task for task in manifest["tasks"] if task["layer"] < 2]
    for task in manifest["tasks"]:
        task["embedding"].update(layers=2, sha256=accuracy.file_sha256(path),
                                 size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
        task["embedding"].pop("scope", None)
    manifest.update(n_layers=2, n_direction_units=2 * manifest["runs"])
    manifest_path.write_text(json.dumps(manifest))
    for command in (("run", "--dry-run"), ("run",), ("aggregate",)):
        result = _public_cli("run_robustness.py", *command, "--manifest", manifest_path)
        assert result.returncode != 0 and "all 36 physical layers" in result.stderr
    assert sorted(p.name for p in output.iterdir()) == ["manifest.json"]


def test_timing_cli_rejects_shortened_prepared_inventory_before_work(tmp_path):
    cache(tmp_path)
    output = tmp_path / "timing"
    # A one-setting scope keeps this regression tiny, but cannot relax the
    # declared physical inventory of a known benchmark model.
    manifest = timing.prepare_manifest({"models": [MODEL], "datasets": ["STSA"]},
                                       tmp_path, output, scope="synthetic")
    cache(tmp_path, layers=2)
    manifest["settings"][0]["embedding"]["layers"] = [0, 1]
    manifest["task_count"] = 2 * len(manifest["seeds"])
    manifest["method_unit_count"] = manifest["task_count"] * len(timing.METHODS)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    for command in (("validate",), ("aggregate",), ("run", "--task-index", "0")):
        result = _public_cli("run_timing.py", *command, "--manifest", manifest_path)
        assert result.returncode != 0 and "all 36 physical layers" in result.stderr
    assert not (output / "records").exists() and not (output / "aggregate.json").exists()
