"""Protocol/aggregation fixtures do not substitute for the real five-method canary."""
import copy
from pathlib import Path

import numpy as np
import pytest

from raptor.experiments import timing
from raptor.experiments import accuracy_multiseed as accuracy


def inputs(tmp_path, layers=2):
    emb = tmp_path / "embeddings"
    emb.mkdir()
    rng = np.random.default_rng(10)
    arrays = {}
    for layer in range(layers):
        arrays[f"X_pos_{layer}"] = rng.normal(size=(50, 6)).astype(np.float32)
        arrays[f"X_neg_{layer}"] = rng.normal(size=(50, 6)).astype(np.float32)
    accuracy.atomic_savez(emb / "synthetic_tiny_embeddings.npz", **arrays)
    config = {"models": ["synthetic"], "datasets": ["tiny"], "seeds": list(timing.SEEDS),
              "layer_counts": {"synthetic": layers}}
    return emb, config


def prepared(tmp_path):
    emb, config = inputs(tmp_path)
    root = tmp_path / "run"
    return root, timing.prepare_manifest(config, emb, root, scope="synthetic")


def fixture_record(manifest, setting, split, layer, method, seconds):
    comparison = {"cpu_model": "explicit statistical fixture", "backend": "cpu", "threads": 1,
                  "interop_threads": 1, "affinity_cpu_count": 1}
    env = {"comparison": comparison, "comparison_sha256": timing.digest(comparison)}
    record = {"status": "complete", "identity": timing.unit_identity(manifest, setting, split, layer, method),
              "wall_seconds": seconds, "process_cpu_seconds": seconds,
              "environment": env, "environment_after": copy.deepcopy(env),
              "result": {"validation_accuracy": .5, "test_accuracy": .7},
              "trace": {"six_trial_complete": True} if method == "xrfm" else {}}
    record["payload_sha256"] = timing.digest(record)
    return record


def populate_statistical_fixtures(root, manifest):
    for setting, split, layer in timing.tasks(manifest):
        for method in timing.METHODS:
            # seed42 [1,3], seed43 [2,4], ... ; per-seed medians2..6; mean4.
            seconds = split["seed"] - 41 + layer * 2
            path = timing.checkpoint_path(root, setting, split, layer, method)
            timing.write_json(path, fixture_record(manifest, setting, split, layer, method, seconds))


def test_native_budgets():
    assert timing.HPARAMS == accuracy.METHOD_HPARAMETERS
    assert timing.HPARAMS["singlelr"]["C_grid"] == "logspace(-4,2,100)"
    assert timing.HPARAMS["xrfm"]["rfm_iters"] == 10
    assert timing.HPARAMS["gcs"]["n_iter"] == 1000
    assert timing.HPARAMS["gcs"]["sample_n"] == 1000
    assert timing.HPARAMS["fixed_reg"]["fixed_c"] == 1
    assert "logistic" not in timing.HPARAMS["dim_midpoint"]


def test_clock_boundary_is_explicit():
    includes = " ".join(timing.BOUNDARY["includes"])
    excludes = " ".join(timing.BOUNDARY["excludes"])
    assert all(word in includes for word in ("preprocessing", "candidate", "refit", "validation", "test"))
    assert all(word in excludes for word in ("hashing", "warmup", "checkpoint", "cross-layer"))


def test_preparation_has_no_history_or_head_dependency(tmp_path):
    root, manifest = prepared(tmp_path)
    timing.validate_manifest(manifest)
    assert manifest["task_count"] == 10 and manifest["method_unit_count"] == 50
    assert manifest["seeds"] == [42, 43, 44, 45, 46]
    assert all(not Path(s["embedding"]["path"]).is_absolute() for s in manifest["settings"])
    with pytest.raises(ValueError, match="already exists"):
        emb = root.parent / "embeddings"
        timing.prepare_manifest({"models": ["synthetic"], "datasets": ["tiny"]}, emb, root, scope="synthetic")


def test_seed_paths_and_shared_split_identity(tmp_path):
    root, manifest = prepared(tmp_path)
    paths, splits = set(), []
    for setting, split, layer in timing.tasks(manifest):
        data, metadata = timing.load_layer(root, setting, split, layer)
        identities = [timing.unit_identity(manifest, setting, split, layer, method) for method in timing.METHODS]
        assert len({i["split_sha256"] for i in identities}) == 1
        assert len({i["embedding_sha256"] for i in identities}) == 1
        assert metadata["partition_counts"] == [64, 16, 20]
        for method in timing.METHODS:
            paths.add(timing.checkpoint_path(root, setting, split, layer, method))
        if layer == 0:
            splits.append(tuple(data[2]))
    assert len(paths) == 50 and len(set(splits)) == 5


def test_changed_input_or_split_rejected(tmp_path):
    root, manifest = prepared(tmp_path)
    setting, split, layer = next(timing.tasks(manifest))
    split_path = root / split["path"]
    split_path.write_bytes(split_path.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="split file changed"):
        timing.load_layer(root, setting, split, layer)


def test_changed_source_or_budget_rejected(tmp_path):
    _, manifest = prepared(tmp_path)
    changed = copy.deepcopy(manifest)
    changed["source"]["sha256"] = "wrong-source"
    with pytest.raises(ValueError, match="source bytes"):
        timing.validate_manifest(changed)
    changed = copy.deepcopy(manifest)
    changed["hyperparameters"]["gcs"]["n_iter"] = 2
    with pytest.raises(ValueError, match="budgets"):
        timing.validate_manifest(changed)


def test_strict_hierarchical_aggregation_and_ties(tmp_path):
    root, manifest = prepared(tmp_path)
    populate_statistical_fixtures(root, manifest)
    result = timing.aggregate(root / "manifest.json")
    assert len(result["seed_rows"]) == 25
    assert len(result["setting_points"]) == 5
    for point in result["setting_points"]:
        assert point["seed_median_seconds"] == [2, 3, 4, 5, 6]
        assert point["mean_seed_median_seconds"] == 4
        assert point["sample_sd_seed_median_seconds"] == pytest.approx(np.sqrt(2.5))
    assert set(result["median_setting_seconds"].values()) == {4}
    assert all(row["selected_layer_from_fresh_validation"] == 0 for row in result["seed_rows"])
    timing.plot(root / "aggregate.json", root / "plot")
    assert all((root / ("plot." + suffix)).stat().st_size > 0 for suffix in ("png", "svg", "pdf"))


def test_missing_layer_seed_fails_loudly(tmp_path):
    root, manifest = prepared(tmp_path)
    populate_statistical_fixtures(root, manifest)
    setting, split, layer = list(timing.tasks(manifest))[-1]
    timing.checkpoint_path(root, setting, split, layer, "gcs").unlink()
    with pytest.raises(ValueError, match="incomplete or incompatible grid"):
        timing.aggregate(root / "manifest.json")


def test_checksum_and_mixed_run_rejected(tmp_path):
    root, manifest = prepared(tmp_path)
    populate_statistical_fixtures(root, manifest)
    setting, split, layer = next(timing.tasks(manifest))
    path = timing.checkpoint_path(root, setting, split, layer, "fixed_reg")
    record = timing.read_json(path)
    record["identity"]["run_id"] = "another-run"
    record.pop("payload_sha256")
    record["payload_sha256"] = timing.digest(record)
    timing.write_json(path, record)
    with pytest.raises(ValueError, match="incompatible"):
        timing.aggregate(root / "manifest.json")


def test_mixed_hardware_rejected(tmp_path):
    root, manifest = prepared(tmp_path)
    populate_statistical_fixtures(root, manifest)
    setting, split, layer = next(timing.tasks(manifest))
    path = timing.checkpoint_path(root, setting, split, layer, "dim_midpoint")
    record = timing.read_json(path)
    record.pop("payload_sha256")
    for key in ("environment", "environment_after"):
        record[key]["comparison"]["cpu_model"] = "other CPU class"
        record[key]["comparison_sha256"] = timing.digest(record[key]["comparison"])
    record["payload_sha256"] = timing.digest(record)
    timing.write_json(path, record)
    with pytest.raises(ValueError, match="mixed resource"):
        timing.aggregate(root / "manifest.json")


def test_partial_checksum_and_failed_xrfm_ledger_rejected(tmp_path):
    root, manifest = prepared(tmp_path)
    setting, split, layer = next(timing.tasks(manifest))
    identity = timing.unit_identity(manifest, setting, split, layer, "xrfm")
    path = timing.checkpoint_path(root, setting, split, layer, "xrfm")
    timing.write_json(path, {"status": "complete"})
    with pytest.raises(KeyError):
        timing.load_record(path, identity)
    record = fixture_record(manifest, setting, split, layer, "xrfm", 1.0)
    record["wall_seconds"] = 2.0
    timing.write_json(path, record)
    with pytest.raises(ValueError, match="corrupt"):
        timing.load_record(path, identity)
    record = fixture_record(manifest, setting, split, layer, "xrfm", 1.0)
    record.pop("payload_sha256")
    record["trace"]["six_trial_complete"] = False
    record["payload_sha256"] = timing.digest(record)
    timing.write_json(path, record)
    with pytest.raises(ValueError, match="six-trial"):
        timing.load_record(path, identity)



def test_lazy_midpoint_module_preloaded_outside_native_clock():
    import sys
    modules = timing.preload_native_modules()
    assert modules == ["raptor.probes.dim_midpoint"]
    assert all(name in sys.modules for name in modules)
