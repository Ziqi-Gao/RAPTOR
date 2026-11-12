"""Offline protocol checks using genuine native fits and small synthetic inputs."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from raptor.experiments import accuracy_multiseed as accuracy
from raptor.experiments import robustness
from raptor.experiments.accuracy_preparation import inspect_embedding
from raptor.experiments.run_raptor import fit_raptor_layer
from raptor.probes.gcs_sampler import ProbingConfig, run_probing


def synthetic():
    rng = np.random.default_rng(104)
    y = np.r_[np.ones(80, dtype=int), np.zeros(80, dtype=int)]
    X = rng.normal(size=(160, 4)).astype(np.float32)
    X[:, 0] += y * .8
    return X, y


def test_five_native_accuracy_method_registry_and_refit_rules():
    assert set(accuracy.METHODS) == {"singlelr", "xrfm", "gcs", "fixed_reg", "dim_midpoint"}
    assert accuracy.protocol_methods({"grid": {"methods": list(accuracy.METHODS)}}, accuracy.GROUPED_PROTOCOL) == list(accuracy.GROUPED_METHODS)
    X, y = synthetic()
    train, val, test = accuracy.split_indices(y, .2, .2, 42)
    hp = accuracy.METHOD_HPARAMETERS["fixed_reg"]
    observed = accuracy.METHOD_RUNNERS["fixed_reg"](X, y, train, val, test, seed=42, hyperparameters=hp)
    native = fit_raptor_layer(X[train], y[train], X[val], y[val], X[test], y[test], fixed_c=1, seed=42)
    assert observed["validation_accuracy"] == native["val_acc_pre_refit"]
    assert observed["test_accuracy"] == native["acc_test"]
    np.testing.assert_array_equal(observed["concept"], native["concept"])
    assert observed["method_details"]["final_fit_partition"] == "train_sub+validation"
    assert observed["method_details"]["c_selection"] == "fixed_predeclared"
    with pytest.raises(ValueError, match="predeclared"):
        accuracy.METHOD_RUNNERS["fixed_reg"](X, y, train, val, test, seed=42, hyperparameters={"fixed_c": .1})


def test_midpoint_native_readout_and_candidate_before_refit():
    X, y = synthetic()
    train, val, test = accuracy.split_indices(y, .2, .2, 42)
    frozen = []
    result = accuracy.METHOD_RUNNERS["dim_midpoint"](X, y, train, val, test, seed=42,
        hyperparameters=accuracy.METHOD_HPARAMETERS["dim_midpoint"], selection_callback=frozen.append)
    assert len(frozen) == 1
    assert frozen[0]["val_acc_pre_refit"] == result["validation_accuracy"]
    fitted = np.r_[train, val]
    pos = X[fitted][y[fitted] == 1].astype(np.float64).mean(axis=0)
    neg = X[fitted][y[fitted] == 0].astype(np.float64).mean(axis=0)
    manual = ((X[test].astype(np.float64)-.5*(pos+neg)) @ (pos-neg) >= 0)
    assert result["test_accuracy"] == np.mean(manual == y[test])
    assert result["concept"].dtype == np.float64
    assert result["method_details"]["no_logistic_fit"] is True


def test_gcs_diagnostics_preserve_original_fits():
    X, y = synthetic()
    config = ProbingConfig(n_iter=3, train_frac=.5, early_loops=2, random_state=17)
    original = run_probing([X], y, config, X_layers_holdout=[X[:20]], y_holdout=y[:20])
    ledger = []
    observed = run_probing([X], y, config, X_layers_holdout=[X[:20]], y_holdout=y[:20], diagnostics=ledger)
    for i in range(3):
        np.testing.assert_array_equal(original[i], observed[i])
    assert len(ledger) >= 3
    assert all(row["iteration_limit_hit"] == (row["n_iter"] >= row["max_iter"]) for row in ledger)


def test_robustness_exact_rng_recipe_and_persisted_partitions():
    _, y = synthetic()
    indices, split_seed = robustness.perturbation_indices(y, seed=42, layer=3, run=8)
    from sklearn.model_selection import train_test_split
    pool, test = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=42)
    rng = np.random.RandomState(42+3000+8)
    retained = rng.choice(pool, size=len(pool)-int(round(.2*len(pool))), replace=False)
    expected_split = int(rng.randint(0, 1_000_000))
    train, val = train_test_split(retained, test_size=.2, stratify=y[retained], random_state=expected_split)
    assert expected_split == split_seed
    for key, expected in (("retained_pool_idx", retained), ("test_idx", test), ("train_sub_idx", train), ("val_idx", val)):
        np.testing.assert_array_equal(indices[key], expected)
    assert not set(indices["retained_pool_idx"]) & set(test)
    assert set(indices["final_refit_idx"]) == set(retained)


def test_robustness_uses_full_pair_denominator_and_abs_sign():
    vectors = np.array([[1., 0.], [-1., 0.], [0., 1.]])
    stats = robustness.direction_statistics(vectors, required_k=3)
    assert stats["n_pairs"] == 3
    assert stats["mean_absolute_cosine"] == pytest.approx(1/3)
    with pytest.raises(ValueError, match="fixed K"):
        robustness.direction_statistics(vectors[:2], required_k=3)
    with pytest.raises(ValueError, match="zero direction"):
        robustness.direction_statistics(np.array([[1.,0.],[0.,0.],[0.,1.]]), required_k=3)


def tiny_task(tmp_path, *, method="fixed_reg", max_iter=1):
    X, y = synthetic()
    emb = tmp_path / "tiny_STSA_embeddings.npz"
    np.savez(emb, X_pos_0=X[y == 1], X_neg_0=X[y == 0])
    embedding = inspect_embedding(emb, "tiny", "STSA", synthetic=True)
    return {"protocol": robustness.PROTOCOL, "model": "tiny", "dataset": "STSA", "layer": 0,
        "method": method, "seed": 42, "runs": 2, "drop_fraction": .2, "test_size": .2, "val_size": .2,
        "hyperparameters": {**accuracy.METHOD_HPARAMETERS[method], "max_iter": max_iter} if method != "caa" else {"direction": "raw_mean_difference"},
        "embedding": embedding, "code_version": robustness.source_version(),
        "environment_sha256": "fixture-environment", "output_dir": str(tmp_path / "units")}


def manifest_for(task):
    return {"schema": "public_robustness_v1", "protocol": robustness.PROTOCOL, "models": ["tiny"],
        "datasets": ["STSA"], "methods": [task["method"]], "runs": task["runs"], "seed": 42,
        "code_version": task["code_version"], "environment": {"sha256": "fixture-environment"},
        "tasks": [task], "n_layers": 1, "n_direction_units": task["runs"],
        "coordinates": {task["method"]: robustness.COORDINATES[task["method"]]},
        "historical_pairing": "not verified", "synthetic_smoke": True}


def test_real_finite_budget_checkpoint_resume_and_complete_aggregation(tmp_path):
    task = tiny_task(tmp_path)
    with mock.patch.object(robustness, "environment_identity", return_value={"sha256": "fixture-environment"}):
        computed = robustness.run_layer(task)
        assert computed["computed"] == 2
        original = {path.name: path.read_bytes() for path in Path(task["output_dir"]).glob("*.npz")}
        with mock.patch.object(robustness, "_run_native", side_effect=AssertionError("resume must not fit again")):
            restored = robustness.run_layer(task)
        assert restored["restored"] == 2
        assert original == {path.name: path.read_bytes() for path in Path(task["output_dir"]).glob("*.npz")}
    # max_iter1 final fits are deliberately retained as real finite-budget outputs.
    summary = robustness.aggregate(manifest_for(task), output=tmp_path / "summary.json")
    assert summary["status"] == "complete_with_nonconvergence"
    assert summary["settings"][0]["nonconverged_directions"] == 2
    assert summary["layers"][0]["n_pairs"] == 1
    with np.load(Path(task["output_dir"])/"run_00.npz", allow_pickle=False) as data:
        assert data["direction"].shape == (4,)  # b is archived separately, excluded from direction
        meta = json.loads(data["metadata_json"].item())
        assert meta["details"]["max_iter_hits_fit"] == 1
        assert meta["intercept_in_direction"] is False
        assert data["test_idx"].ndim == 1
    (Path(task["output_dir"])/"run_01.npz").unlink()
    incomplete = robustness.aggregate(manifest_for(task), output=tmp_path / "incomplete.json")
    assert incomplete["status"] == "incomplete"
    assert incomplete["settings"][0]["Mean"] is None


def test_robustness_resume_retries_only_technical_error_with_same_indices(tmp_path):
    task = tiny_task(tmp_path, method="caa")
    task["runs"] = 20
    native = robustness._run_native
    attempts = []

    def fail_once(*args, **kwargs):
        attempts.append({key: value.copy() for key, value in args[3].items()})
        if len(attempts) == 1:
            raise OSError("injected transient storage failure")
        return native(*args, **kwargs)

    with mock.patch.object(robustness, "environment_identity", return_value={"sha256": "fixture-environment"}):
        with mock.patch.object(robustness, "_run_native", side_effect=fail_once):
            first = robustness.run_layer(task)
        assert first["computed"] == 20
        assert first["errors"] == first["invalid_directions"] == 1
        path = Path(task["output_dir"]) / "run_00.npz"
        error_bytes = path.read_bytes()
        successful = {p.name: p.read_bytes() for p in path.parent.glob("run_*.npz") if p != path}
        incomplete = robustness.aggregate(manifest_for(task), output=tmp_path / "incomplete.json")
        assert incomplete["status"] == "incomplete"
        with mock.patch.object(robustness, "_run_native", wraps=native) as retry:
            resumed = robustness.run_layer(task)
        assert resumed["computed"] == 1
        assert resumed["restored"] == 19
        assert resumed["errors"] == 0
        assert resumed["retried_errors"] == 1
        assert retry.call_count == 1
        for key, expected in attempts[0].items():
            np.testing.assert_array_equal(retry.call_args.args[3][key], expected)
        assert retry.call_args.kwargs["native_seed"] == 42
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["metadata_json"].item())
            assert meta["status"] == "complete"
            assert meta["identity"]["run"] == 0
        history = list(path.parent.glob("run_00.npz.error.*"))
        assert len(history) == 1 and history[0].read_bytes() == error_bytes
        assert successful == {p.name: p.read_bytes() for p in path.parent.glob("run_*.npz") if p != path}
        complete = robustness.aggregate(manifest_for(task), output=tmp_path / "complete.json")
        assert complete["status"] == "complete"
        assert complete["layers"][0]["n_pairs"] == 190
        with mock.patch.object(robustness, "_run_native", side_effect=AssertionError("completed units must restore")):
            assert robustness.run_layer(task)["restored"] == 20


def test_robustness_cli_reports_execution_error_and_plain_resume_recovers(tmp_path, monkeypatch, capsys):
    task = tiny_task(tmp_path, method="caa")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_for(task)))
    monkeypatch.setattr(sys, "argv", ["run_robustness.py", "run", "--manifest", str(manifest_path)])
    native = robustness._run_native

    def unavailable(*args, **kwargs):
        if kwargs["native_seed"] == 42:
            raise OSError("storage unavailable")
        return native(*args, **kwargs)

    with mock.patch.object(robustness, "environment_identity", return_value={"sha256": "fixture-environment"}):
        for _ in range(2):
            with mock.patch.object(robustness, "_run_native", side_effect=unavailable):
                with pytest.raises(SystemExit) as stopped:
                    robustness.main()
            assert stopped.value.code == 2
        assert "resume" in capsys.readouterr().err.lower()
        with mock.patch.object(robustness, "_run_native", wraps=native) as retry:
            robustness.main()
        assert retry.call_count == 1
    assert len(list(Path(task["output_dir"]).glob("run_00.npz.error.*"))) == 2
    assert robustness.aggregate(manifest_for(task), output=tmp_path / "summary.json")["status"] == "complete"


@pytest.mark.parametrize("status", ["degenerate", "nonfinite"])
def test_robustness_scientific_invalid_directions_restore_without_retry(tmp_path, status):
    task = tiny_task(tmp_path, method="caa")
    # Native CAA returns a zero direction or a genuine finite direction whose
    # float64 norm overflows. Neither is an execution exception or a retry cue.
    value = 0.0 if status == "degenerate" else 1e154
    embedding = Path(task["embedding"]["path"])
    np.savez(embedding, X_pos_0=np.full((80, 4), value), X_neg_0=np.full((80, 4), -value))
    task["embedding"] = inspect_embedding(embedding, "tiny", "STSA", synthetic=True)
    with mock.patch.object(robustness, "environment_identity", return_value={"sha256": "fixture-environment"}):
        first = robustness.run_layer(task)
        assert first["errors"] == 0 and first["invalid_directions"] == 2
        original = {path.name: path.read_bytes() for path in Path(task["output_dir"]).glob("*.npz")}
        with mock.patch.object(robustness, "_run_native", side_effect=AssertionError("scientific outcome must restore")):
            resumed = robustness.run_layer(task)
        assert resumed["computed"] == resumed["retried_errors"] == resumed["errors"] == 0
        assert resumed["restored"] == resumed["invalid_directions"] == 2
        assert original == {path.name: path.read_bytes() for path in Path(task["output_dir"]).glob("*.npz")}
    with np.load(Path(task["output_dir"]) / "run_00.npz", allow_pickle=False) as data:
        assert json.loads(data["metadata_json"].item())["status"] == status
    summary = robustness.aggregate(manifest_for(task), output=tmp_path / "incomplete.json")
    assert summary["status"] == "incomplete"
    assert summary["settings"][0]["Mean"] is None


def test_robustness_refit_error_retry_reuses_frozen_candidate(tmp_path):
    from raptor.experiments import run_raptor
    task = tiny_task(tmp_path)
    native_refit = run_raptor.refit_raptor_layer
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected refit execution failure")
        return native_refit(*args, **kwargs)

    with mock.patch.object(robustness, "environment_identity", return_value={"sha256": "fixture-environment"}):
        with mock.patch.object(run_raptor, "refit_raptor_layer", side_effect=fail_once):
            assert robustness.run_layer(task)["errors"] == 1
        candidate = Path(task["output_dir"]) / "candidate_00.npz"
        original = candidate.read_bytes()
        with mock.patch.object(run_raptor, "fit_raptor_layer", side_effect=AssertionError("candidate must stay frozen")):
            resumed = robustness.run_layer(task)
        assert resumed["computed"] == resumed["restored"] == resumed["retried_errors"] == 1
        assert resumed["errors"] == resumed["invalid_directions"] == 0
        assert candidate.read_bytes() == original
    summary = robustness.aggregate(manifest_for(task), output=tmp_path / "summary.json")
    assert summary["status"] == "complete_with_nonconvergence"
    assert summary["settings"][0]["nonconverged_directions"] == 2


@pytest.mark.parametrize("phase", ["run", "candidate"])
def test_robustness_resume_rejects_truncated_zip_and_preserves_other_outcomes(tmp_path, phase):
    from raptor.experiments import run_raptor

    task = tiny_task(tmp_path)
    native = robustness._run_native
    native_refit = run_raptor.refit_raptor_layer
    refit_attempts = 0

    def fail_first_refit(*args, **kwargs):
        nonlocal refit_attempts
        refit_attempts += 1
        if phase == "candidate" and refit_attempts == 1:
            raise OSError("injected refit failure before candidate corruption")
        return native_refit(*args, **kwargs)

    with mock.patch.object(robustness, "environment_identity", return_value={"sha256": "fixture-environment"}):
        with mock.patch.object(robustness, "_run_native", wraps=native) as first_native:
            with mock.patch.object(run_raptor, "refit_raptor_layer", side_effect=fail_first_refit):
                first = robustness.run_layer(task)
        assert first["computed"] == 2
        assert first["errors"] == int(phase == "candidate")
        expected_call = first_native.call_args_list[0]
        output = Path(task["output_dir"])
        damaged = output / ("run_00.npz" if phase == "run" else "candidate_00.npz")
        damaged_bytes = damaged.read_bytes()[:-10]
        damaged.write_bytes(damaged_bytes)
        # All other scientific outcomes, including the finite nonconverged
        # run_01 direction, must survive this storage repair byte for byte.
        stable = {path.name: path.read_bytes() for path in output.glob("*.npz")
                  if path not in {damaged, output / "run_00.npz"}}
        with mock.patch.object(robustness, "_run_native", wraps=native) as retry:
            resumed = robustness.run_layer(task)
        assert resumed["computed"] == resumed["restored"] == 1
        assert resumed["errors"] == resumed["invalid_directions"] == 0
        assert retry.call_count == 1
        assert retry.call_args.kwargs["identity"] == expected_call.kwargs["identity"]
        assert retry.call_args.kwargs["native_seed"] == expected_call.kwargs["native_seed"] == 42
        for key, expected in expected_call.args[3].items():
            np.testing.assert_array_equal(retry.call_args.args[3][key], expected)
        rejected = list(output.glob(damaged.name + ".rejected.*"))
        assert len(rejected) == 1 and rejected[0].read_bytes() == damaged_bytes
        assert all((output / name).read_bytes() == contents for name, contents in stable.items())
    summary = robustness.aggregate(manifest_for(task), output=tmp_path / "summary.json")
    assert summary["status"] == "complete_with_nonconvergence"
    assert summary["settings"][0]["nonconverged_directions"] == 2


def test_robustness_rejects_layer_grid_pruning_and_source_mixing(tmp_path):
    task = tiny_task(tmp_path)
    manifest = manifest_for(task)
    robustness.validate_manifest(manifest)
    bad = copy.deepcopy(manifest)
    bad["tasks"][0]["embedding"]["layers"] = 2
    with pytest.raises(ValueError, match="complete fixed layer"):
        robustness.validate_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["tasks"][0]["environment_sha256"] = "foreign"
    with pytest.raises(ValueError, match="identity"):
        robustness.validate_manifest(bad)


def test_robustness_coordinates_preserve_native_xrfm_identity():
    assert robustness.COORDINATES["xrfm"] == "standardized_space_AGOP_eigenvector"
    assert robustness.COORDINATES["caa"] == "raw_mean_difference"
    assert "without_intercept" in robustness.COORDINATES["fixed_reg"]
