"""Native offline accuracy outputs exercise the public steering preparation path."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from raptor.experiments.accuracy_multiseed import execute_task
from raptor.steering.controls import file_hash, load_bundle
from raptor.steering.prepare_probes import export_checkpoints, fit_caa


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def native_probes(tmp_path_factory):
    root = tmp_path_factory.mktemp("native_steering_export")
    embeddings = root / "embeddings"
    embeddings.mkdir()
    model = "synthetic-steering-export"
    rng = np.random.default_rng(28)
    arrays = {}
    for layer in range(4):
        arrays[f"X_pos_{layer}"] = (rng.normal(size=(40, 3)) + .5).astype(np.float32)
        arrays[f"X_neg_{layer}"] = (rng.normal(size=(40, 3)) - .5).astype(np.float32)
    embedding = embeddings / f"{model}_STSA_embeddings.npz"
    np.savez(embedding, **arrays)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/prepare_accuracy_multiseed.py"),
        "--emb-dir", str(embeddings), "--models", model, "--datasets", "STSA",
        "--synthetic-canary", "--methods", "singlelr", "--protocols", "rowwise",
        "--seeds", "42", "--results-root", str(root / "results"),
        "--task-dir", str(root / "tasks"),
    ], check=True, capture_output=True, text=True)
    manifest = json.loads((root / "tasks/manifest.json").read_text())
    task_file = Path(next(iter(manifest["task_files"].values()))["path"])
    task = json.loads(task_file.read_text().splitlines()[0])
    assert execute_task(task)["computed"] == 4  # Actual native 100-C fits/refits.
    seed = root / "results/rowwise" / model / "STSA/seed_42"
    checkpoints = seed / "checkpoints/singlelr"
    with np.load(checkpoints / "layer_0.npz", allow_pickle=False) as data:
        identity = json.loads(str(data["metadata_json"].item()))["identity"]
    binding = {key: identity[key] for key in
               ("model", "dataset", "method", "split_sha256", "embedding_sha256")}
    binding.update(activation_site="transformer_block_output", coordinate_evidence="synthetic fixture")
    return dict(checkpoints=checkpoints, binding=binding, embedding=embedding, split=seed / "splits.npz")


def rewrite_checkpoint(path, *, identity=None, status=None, concept=None):
    """Explicit negative-fixture mutations of an actual native checkpoint."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["identity"].update(identity or {})
    if status is not None:
        metadata["status"] = status
    if concept is not None:
        arrays["concept"] = concept
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez(path, **arrays)


def test_documented_shell_glob_exports_native_finals(native_probes, tmp_path):
    # Execute the actual documented command with Bash glob expansion, unedited.
    (tmp_path / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    (tmp_path / "RESULTS/checkpoints").mkdir(parents=True)
    (tmp_path / "RESULTS/checkpoints/singlelr").symlink_to(native_probes["checkpoints"], target_is_directory=True)
    (tmp_path / "MY_BINDING.json").write_text(json.dumps(native_probes["binding"]))
    docs = (ROOT / "docs/appendix_and_steering.md").read_text()
    command = "python scripts/prepare_steering_probes.py export" + docs.split(
        "python scripts/prepare_steering_probes.py export", 1)[1].split("\npython scripts/steer.py", 1)[0]
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", "-c", command], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "excluded 4 selection snapshot(s)" in result.stdout
    arrays, metadata = load_bundle(tmp_path / "outputs/probes/raptor.npz")
    assert arrays["W"].shape == (4, 3)
    assert metadata["source_identity"]["embedding_layers"] == 4
    assert len(metadata["sources"]) == 4
    assert not any("selection" in item["path"] for item in metadata["sources"])


def test_export_rejects_completed_prefix(native_probes, tmp_path):
    finals = [native_probes["checkpoints"] / f"layer_{layer}.npz" for layer in range(3)]
    with pytest.raises(ValueError, match="complete physical layer set.*missing=\\[3\\]"):
        export_checkpoints(finals, native_probes["binding"], tmp_path / "prefix.npz")
    assert not (tmp_path / "prefix.npz").exists()


@pytest.mark.parametrize("declared_layers", [4, 36])
def test_known_model_rejects_prefix_even_if_identity_claims_short_model(native_probes, tmp_path, declared_layers):
    binding = dict(native_probes["binding"], model="Qwen/Qwen2.5-3B-Instruct")
    paths = []
    for layer in range(4):
        path = tmp_path / f"layer_{layer}.npz"
        shutil.copyfile(native_probes["checkpoints"] / path.name, path)
        rewrite_checkpoint(path, identity=dict(model=binding["model"], embedding_layers=declared_layers))
        paths.append(path)
    with pytest.raises(ValueError, match="36 physical layers|complete physical layer set"):
        export_checkpoints(paths, binding, tmp_path / "prefix.npz")


@pytest.mark.parametrize("failure", ["incomplete", "corrupt", "mixed", "dimension", "filename"])
def test_glob_does_not_hide_bad_final(native_probes, tmp_path, failure):
    directory = tmp_path / "checkpoints"
    shutil.copytree(native_probes["checkpoints"], directory)
    final = directory / "layer_2.npz"
    if failure == "incomplete":
        rewrite_checkpoint(final, status="selection_frozen_before_refit")
    elif failure == "corrupt":
        final.write_bytes(b"not an NPZ checkpoint")
    elif failure == "mixed":
        rewrite_checkpoint(final, identity=dict(outer_seed=43))
    elif failure == "dimension":
        rewrite_checkpoint(final, concept=np.ones(3))
    else:
        rewrite_checkpoint(final, identity=dict(layer=3))
    with pytest.raises(ValueError):
        export_checkpoints(sorted(directory.glob("layer_*.npz")), native_probes["binding"], tmp_path / "bad.npz")
    assert not (tmp_path / "bad.npz").exists()


def test_caa_rejects_known_model_prefix_before_fitting(native_probes, tmp_path):
    binding = dict(native_probes["binding"], model="Qwen/Qwen2.5-3B-Instruct", scope="synthetic")
    with pytest.raises(ValueError, match="36 physical layers"):
        fit_caa(native_probes["embedding"], native_probes["split"], binding, tmp_path / "bad.npz")


@pytest.mark.parametrize("failure", ["negative_layers", "metadata"])
def test_caa_checks_cache_inventory_and_metadata(native_probes, tmp_path, failure):
    with np.load(native_probes["embedding"], allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if failure == "negative_layers":
        arrays.pop("X_neg_3")
    else:
        arrays["meta_dim"] = np.asarray(99)
    embedding = tmp_path / "bad_embedding.npz"
    np.savez(embedding, **arrays)
    binding = dict(native_probes["binding"], scope="synthetic", embedding_sha256=file_hash(embedding))
    with pytest.raises(ValueError, match="positive/negative layer mismatch|meta_dim"):
        fit_caa(embedding, native_probes["split"], binding, tmp_path / "bad.npz")


def test_caa_complete_explicit_synthetic_cache(native_probes, tmp_path):
    binding = dict(native_probes["binding"], scope="synthetic",
                   reliability=dict(enabled=False, threshold=0, comparison=">="))
    output = tmp_path / "caa.npz"
    fit_caa(native_probes["embedding"], native_probes["split"], binding, output)
    arrays, metadata = load_bundle(output)
    assert arrays["W"].shape == (4, 3)
    assert metadata["active_layers"] == [1, 2]
