"""Portable five-method computational-cost reproduction, independent of old runs.

The public workflow prepares its own raw-input and split manifest.  Native
methods are reused without reducing their search, fit, or sampling budgets.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from raptor.core import model_tag
from raptor.experiments import accuracy_multiseed as accuracy
from raptor.experiments.accuracy_preparation import validate_layer_set, validate_embedding_metadata

SCHEMA = "raptor.unified_timing.v1"
METHODS = ("singlelr", "xrfm", "gcs", "fixed_reg", "dim_midpoint")
SEEDS = (42, 43, 44, 45, 46)
# These are the actual accepted native timing budgets, not the small CI budgets.
HPARAMS = json.loads(json.dumps(accuracy.METHOD_HPARAMETERS))

BOUNDARY = {
    "clock": "time.perf_counter_ns; synchronized CPU backend; process_time_ns also saved",
    "includes": ["partition slicing", "native train-only preprocessing",
                 "complete native candidate training/search", "outer validation",
                 "prescribed refit or no-refit", "direction/readout construction",
                 "validation/test scoring", "passive xRFM trial observer overhead",
                 "native return-value and metadata assembly"],
    "excludes": ["imports/downloads", "raw embedding loading/hashing", "split preparation",
                 "common synthetic warmup", "resource collection", "checkpoint/export",
                 "cross-layer selection", "LLM extraction", "scheduler queue"],
}
_STOP = False


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return accuracy.file_sha256(Path(path))


def read_json(path):
    return json.loads(Path(path).read_text())


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    accuracy.atomic_json(Path(path), jsonable(value))


def relative(path, base):
    return os.path.relpath(Path(path).resolve(), Path(base).resolve())


def source_identity():
    root = Path(__file__).resolve().parents[1]
    files = {str(p.relative_to(root)): file_sha(p) for p in sorted(root.rglob("*.py"))}
    return {"algorithm": "sha256_release_python_sources", "files": files, "sha256": digest(files)}


def release_version():
    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
        return {"commit_if_available": result.stdout.strip(), "source_identity_authoritative": True}
    except (OSError, subprocess.CalledProcessError):
        return {"commit_if_available": None, "source_identity_authoritative": True}


def dependency_identity():
    import xrfm
    root = Path(xrfm.__file__).resolve().parent
    files = {str(p.relative_to(root)): file_sha(p) for p in sorted(root.rglob("*.py"))}
    return {"distribution": importlib.metadata.version("xrfm"), "python_source_sha256": digest(files)}


def embedding_headers(path, *, model, dataset, synthetic=False):
    headers = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                raise ValueError("embedding cache must contain only NumPy arrays")
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            headers[name[:-4]] = {"shape": list(shape), "dtype": str(dtype)}
    positive = sorted(int(k[6:]) for k in headers if k.startswith("X_pos_"))
    negative = sorted(int(k[6:]) for k in headers if k.startswith("X_neg_"))
    if positive != negative or positive != list(range(len(positive))) or not positive:
        raise ValueError("embedding cache requires complete consecutive matching positive/negative layers")
    validate_layer_set(model, positive, synthetic=synthetic)
    first = headers["X_pos_0"]["shape"]
    npos, width = first if len(first) == 2 else (0, 0)
    neg = headers["X_neg_0"]["shape"]
    nneg = neg[0] if len(neg) == 2 else 0
    if not npos or not nneg or not width:
        raise ValueError("embedding cache has empty or malformed dimensions")
    for layer in positive:
        for prefix, count in (("X_pos_", npos), ("X_neg_", nneg)):
            if headers[prefix + str(layer)] != {"shape": [count, width], "dtype": "float32"}:
                raise ValueError("all native layers require consistent raw float32 shapes")
    with np.load(path, allow_pickle=False) as archive:
        validate_embedding_metadata(archive, model=model, dataset=dataset, layers=len(positive), dim=width)
    return {"layers": positive, "n_positive": npos, "n_negative": nneg, "width": width}


def prepare_manifest(config, embeddings, output, *, scope="full", run_id=None):
    """Prepare new splits using only raw embeddings/config; no historical decisions."""
    output = Path(output).resolve()
    if (output / "manifest.json").exists():
        raise ValueError("manifest already exists: choose a fresh output directory")
    models, datasets = config["models"], config["datasets"]
    seeds = config.get("seeds", list(SEEDS))
    if len(models) != len(set(models)) or len(datasets) != len(set(datasets)):
        raise ValueError("duplicate model/dataset identities")
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int for s in seeds):
        raise ValueError("outer seeds must be explicit unique integers")
    if scope == "full" and (len(models) != 6 or len(datasets) != 6 or seeds != list(SEEDS)):
        raise ValueError("formal timing requires six models, six datasets and seeds42-46")
    resources = config.get("resources", {"backend": "cpu", "threads": 1, "affinity_cpus": 1})
    if resources != {"backend": "cpu", "threads": 1, "affinity_cpus": 1}:
        raise ValueError("accepted CPU1 protocol requires cpu backend, one thread and one affinity CPU")
    settings = []
    for model in models:
        for dataset in datasets:
            path = Path(embeddings) / f"{model_tag(model)}_{dataset}_embeddings.npz"
            before = path.stat()
            headers = embedding_headers(path, model=model, dataset=dataset, synthetic=scope == "synthetic")
            expected = config.get("layer_counts", {}).get(model)
            if expected is not None and len(headers["layers"]) != expected:
                raise ValueError(f"{model}/{dataset}: expected all {expected} physical layers")
            sha = file_sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("embedding changed during preparation")
            emb = {"path": relative(path, output), "sha256": sha, "size": after.st_size,
                   "mtime_ns": after.st_mtime_ns, **headers}
            y = np.r_[np.ones(headers["n_positive"], dtype=np.int64),
                      np.zeros(headers["n_negative"], dtype=np.int64)]
            splits = []
            for seed in seeds:
                split_path = output / "splits" / model_tag(model) / dataset / f"seed_{seed}.npz"
                meta = accuracy.create_split_artifact(split_path, y=y, protocol="rowwise", seed=seed,
                    test_size=.2, val_size=.2, model=model, dataset=dataset, embedding=emb)
                splits.append({"seed": seed, "path": relative(split_path, output),
                               "sha256": file_sha(split_path), "identity_sha256": meta["split_sha256"]})
            settings.append({"model": model, "dataset": dataset, "scope": scope,
                             "embedding": emb, "splits": splits})
    manifest = {"schema": SCHEMA, "scope": scope, "run_id": run_id or uuid.uuid4().hex,
                "source": source_identity(), "release_version": release_version(),
                "dependency": dependency_identity(), "methods": list(METHODS), "seeds": seeds,
                "hyperparameters": HPARAMS, "boundary": BOUNDARY, "resources": resources,
                "settings": settings, "setting_count": len(settings),
                "task_count": sum(len(s["embedding"]["layers"]) * len(seeds) for s in settings),
                "selection": "fresh pre-refit outer-validation; lowest physical layer on exact tie",
                "aggregation": "median complete physical layers per seed; equal mean seeds per setting; median settings"}
    manifest["method_unit_count"] = manifest["task_count"] * len(METHODS)
    write_json(output / "manifest.json", manifest)
    return manifest


def validate_manifest(manifest, *, current_source=True):
    if manifest.get("schema") != SCHEMA or manifest.get("methods") != list(METHODS):
        raise ValueError("wrong timing schema or method coverage")
    if manifest.get("hyperparameters") != HPARAMS or manifest.get("boundary") != BOUNDARY:
        raise ValueError("native budgets or clock boundary changed")
    if current_source and manifest.get("source") != source_identity():
        raise ValueError("release source bytes differ from prepared manifest; prepare a new run")
    if manifest.get("dependency") != dependency_identity():
        raise ValueError("installed xRFM dependency identity differs")
    expected = sum(len(s["embedding"]["layers"]) * len(manifest["seeds"]) for s in manifest["settings"])
    if expected != manifest["task_count"] or expected * len(METHODS) != manifest["method_unit_count"]:
        raise ValueError("manifest counts do not match the complete grid")
    keys = [(s["model"], s["dataset"]) for s in manifest["settings"]]
    if len(keys) != len(set(keys)) or len(keys) != manifest["setting_count"]:
        raise ValueError("duplicate or missing setting identity")
    if manifest["scope"] == "full" and (len(keys) != 36 or manifest["seeds"] != list(SEEDS)):
        raise ValueError("full scope requires36settings and5seeds")
    paths = set()
    for setting in manifest["settings"]:
        layers = setting["embedding"]["layers"]
        validate_layer_set(setting["model"], layers, synthetic=manifest["scope"] == "synthetic")
        if setting.get("scope", "full") != manifest["scope"]:
            raise ValueError("setting scope differs from timing manifest")
        if [s["seed"] for s in setting["splits"]] != manifest["seeds"]:
            raise ValueError("missing or reordered seed split")
        for split in setting["splits"]:
            if split["path"] in paths:
                raise ValueError("split path collision")
            paths.add(split["path"])


def tasks(manifest):
    for setting in manifest["settings"]:
        for split in setting["splits"]:
            for layer in setting["embedding"]["layers"]:
                yield setting, split, layer


def checkpoint_path(root, setting, split, layer, method):
    return Path(root) / "records" / model_tag(setting["model"]) / setting["dataset"] / f"seed_{split['seed']}" / f"layer_{layer:03d}_{method}.json"


def unit_identity(manifest, setting, split, layer, method):
    return {"schema": SCHEMA, "manifest_sha256": digest(manifest), "run_id": manifest["run_id"],
            "model": setting["model"], "dataset": setting["dataset"], "seed": split["seed"],
            "method_seed": split["seed"] + accuracy.METHOD_SEED_OFFSETS[method],
            "layer": layer, "method": method, "split_sha256": split["identity_sha256"],
            "embedding_sha256": setting["embedding"]["sha256"], "source_sha256": manifest["source"]["sha256"],
            "hyperparameters": manifest["hyperparameters"][method]}


def load_record(path, identity):
    """Malformed or incompatible records never count as complete."""
    record = read_json(path)
    checksum = record.pop("payload_sha256")
    if digest(record) != checksum or record.get("identity") != identity or record.get("status") != "complete":
        raise ValueError("corrupt, incompatible or incomplete native timing record")
    if not np.isfinite(record["wall_seconds"]) or record["wall_seconds"] <= 0:
        raise ValueError("invalid native wall clock")
    for environment in (record["environment"], record["environment_after"]):
        if digest(environment["comparison"]) != environment["comparison_sha256"]:
            raise ValueError("resource comparison metadata checksum differs")
        if any(environment["comparison"].get(k) != v for k, v in
               (("backend", "cpu"), ("threads", 1), ("interop_threads", 1), ("affinity_cpu_count", 1))):
            raise ValueError("record is outside the CPU1 comparison protocol")
    if record["environment"]["comparison_sha256"] != record["environment_after"]["comparison_sha256"]:
        raise ValueError("resources changed within the recorded method")
    if identity["method"] == "xrfm" and not record.get("trace", {}).get("six_trial_complete"):
        raise ValueError("native six-trial search not complete")
    for key in ("validation_accuracy", "test_accuracy"):
        if not 0 <= record["result"][key] <= 1:
            raise ValueError("invalid native score")
    return record


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def runtime_environment():
    import torch
    import scipy
    import sklearn
    import threadpoolctl
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "xrfm", "threadpoolctl"):
        versions[package] = importlib.metadata.version(package)
    pools = jsonable(threadpool_info())
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    hardware = {"cpu_model": cpu_model(), "architecture": platform.machine(), "backend": "cpu",
                "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                "affinity_cpu_count": len(affinity) if affinity is not None else None,
                "versions": versions, "python": platform.python_version(),
                "runtime_versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                    "scikit-learn": sklearn.__version__, "torch": str(torch.__version__),
                    "threadpoolctl": threadpoolctl.__version__},
                "torch_cuda_build": torch.version.cuda,
                "precision": "rawfloat32; native sklearn precision; CAA means/readoutfloat64",
                "native_pools": [{k: p.get(k) for k in ("internal_api", "num_threads", "version", "architecture")}
                                 for p in pools]}
    if hardware["threads"] != 1 or hardware["interop_threads"] != 1 or hardware["affinity_cpu_count"] != 1:
        raise ValueError("CPU1 thread/affinity settings are not effective")
    if any(p["num_threads"] != 1 for p in pools):
        raise ValueError("a native thread pool escaped the CPU1 limit")
    if torch.cuda.is_available():
        raise ValueError("CPU timing requires CUDA_VISIBLE_DEVICES='' before starting Python")
    return {"comparison": hardware, "comparison_sha256": digest(hardware), "affinity_cpu_ids": affinity,
            "native_pools_full": pools, "platform": platform.platform(), "host": platform.node(),
            "cuda_build": torch.version.cuda, "cuda_available": False, "gpu_models": [],
            "clock": vars(time.get_clock_info("perf_counter")),
            "environment_threads": {k: os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}


@contextlib.contextmanager
def cpu1():
    import torch
    previous = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    if previous is not None:
        os.sched_setaffinity(0, {min(previous)})
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    try:
        with threadpool_limits(limits=1):
            yield
    finally:
        if previous is not None:
            os.sched_setaffinity(0, previous)


def preload_native_modules():
    """Load lazy native CAA code before starting any method interval."""
    names = ["raptor.probes.dim_midpoint"]
    for name in names:
        importlib.import_module(name)
    return names


def warmup():
    """Common kernels only, outside every method clock; never fake a method run."""
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(9357)
    X = StandardScaler().fit_transform(rng.normal(size=(64, 6)).astype(np.float32))
    LogisticRegression(max_iter=20).fit(X, np.arange(64) % 2)
    torch.linalg.eigh(torch.from_numpy(X.T @ X))


def load_layer(root, setting, split, layer, *, verify_embedding=True):
    path = (Path(root) / setting["embedding"]["path"]).resolve()
    emb = setting["embedding"]
    start = time.perf_counter_ns()
    if verify_embedding and file_sha(path) != emb["sha256"]:
        raise ValueError("embedding content identity changed")
    if embedding_headers(path, model=setting["model"], dataset=setting["dataset"],
                         synthetic=setting.get("scope") == "synthetic") != {k: emb[k] for k in ("layers", "n_positive", "n_negative", "width")}:
        raise ValueError("embedding dimensions changed")
    with np.load(path, allow_pickle=False) as archive:
        X = np.vstack([archive[f"X_pos_{layer}"], archive[f"X_neg_{layer}"]])
    if not np.isfinite(X).all():
        raise ValueError("nonfinite raw embeddings")
    y = np.r_[np.ones(emb["n_positive"], dtype=np.int64), np.zeros(emb["n_negative"], dtype=np.int64)]
    split_path = Path(root) / split["path"]
    if file_sha(split_path) != split["sha256"]:
        raise ValueError("split file changed")
    train, val, test, _ = accuracy.load_split_artifact(split_path, y=y, expected={
        "protocol": "rowwise", "split_seed": split["seed"], "model": setting["model"],
        "dataset": setting["dataset"], "embedding_sha256": emb["sha256"]})
    if accuracy.array_sha256(train, val, test) != split["identity_sha256"]:
        raise ValueError("ordered split identity changed")
    if any(set(np.unique(y[p])) != {0, 1} for p in (train, val, test)):
        raise ValueError("both classes required in every outer partition")
    return (X, y, train, val, test), {"loading_and_validation_seconds": (time.perf_counter_ns() - start) / 1e9,
        "raw_layer_sha256": accuracy.array_sha256(X), "label_sha256": accuracy.array_sha256(y),
        "partition_counts": [len(train), len(val), len(test)]}


def _stop(signum, frame):
    global _STOP
    _STOP = True


def run_manifest(path, *, task_index=None, all_tasks=False):
    """Atomic unit resume. Stop signals finish/flush the active native method."""
    from raptor.experiments.timing_observer import XRFMLedger
    root = Path(path).resolve().parent
    manifest = read_json(path)
    validate_manifest(manifest)
    indexed = list(enumerate(tasks(manifest)))
    if task_index is not None:
        if not 0 <= task_index < len(indexed):
            raise ValueError("task index outside manifest")
        indexed = [indexed[task_index]]
    elif not all_tasks:
        raise ValueError("choose --task-index or --all explicitly")
    previous_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGUSR1", signal.SIGTERM)):
        previous_handlers[sig] = signal.signal(sig, _stop)
    global _STOP
    _STOP = False
    counts = {"computed": 0, "restored": 0, "rejected": 0}
    try:
        with cpu1():
            preload_started = time.perf_counter_ns()
            preloaded = preload_native_modules()
            preload_seconds = (time.perf_counter_ns() - preload_started) / 1e9
            started = time.perf_counter_ns()
            warmup()
            warm_seconds = (time.perf_counter_ns() - started) / 1e9
            environment = runtime_environment()
            # One content hash per embedding per worker invocation, outside method clocks.
            verified = set()
            for index, (setting, split, layer) in indexed:
                if _STOP:
                    break
                key = setting["embedding"]["sha256"]
                data, input_meta = load_layer(root, setting, split, layer, verify_embedding=key not in verified)
                verified.add(key)
                offset = (split["seed"] - 42 + layer) % len(METHODS)
                order = METHODS[offset:] + METHODS[:offset]
                for method in order:
                    if _STOP:
                        break
                    identity = unit_identity(manifest, setting, split, layer, method)
                    checkpoint = checkpoint_path(root, setting, split, layer, method)
                    # Serialize concurrent workers targeting one unit; each file is atomic.
                    with accuracy.file_lock(checkpoint.with_suffix(".lock")):
                        if checkpoint.exists():
                            try:
                                record = load_record(checkpoint, identity)
                                if record["environment"]["comparison_sha256"] != environment["comparison_sha256"]:
                                    raise ValueError("resume hardware/software comparison class differs")
                                counts["restored"] += 1
                                print(f"restored task={index} layer={layer} method={method}", flush=True)
                                continue
                            except (OSError, ValueError, KeyError, TypeError) as exc:
                                rejected = checkpoint.with_name(checkpoint.stem + ".rejected_" + uuid.uuid4().hex + ".json")
                                checkpoint.replace(rejected)
                                counts["rejected"] += 1
                                print(f"rejected task={index} method={method}: {exc}", flush=True)
                        observer = XRFMLedger(len(data[3])) if method == "xrfm" else contextlib.nullcontext()
                        try:
                            with observer:
                                wall_start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
                                result = accuracy.METHOD_RUNNERS[method](*data,
                                    seed=identity["method_seed"], hyperparameters=manifest["hyperparameters"][method])
                                wall_end, cpu_end = time.perf_counter_ns(), time.process_time_ns()
                            trace = observer.snapshot() if method == "xrfm" else {}
                            if method == "xrfm" and not trace["six_trial_complete"]:
                                raise ValueError("native xRFM did not finish all six candidate trials")
                            concept = np.asarray(result["concept"])
                            if not concept.size or not np.isfinite(concept).all():
                                raise ValueError("native method returned missing/nonfinite concept")
                            scores = [float(result[k]) for k in ("validation_accuracy", "test_accuracy")]
                            if not all(np.isfinite(s) and 0 <= s <= 1 for s in scores):
                                raise ValueError("nonfinite or invalid native accuracy")
                            after = runtime_environment()
                            if after["comparison_sha256"] != environment["comparison_sha256"]:
                                raise ValueError("resources changed inside native method")
                            record = {"status": "complete", "identity": identity,
                                "wall_seconds": (wall_end - wall_start) / 1e9,
                                "process_cpu_seconds": (cpu_end - cpu_start) / 1e9,
                                "environment": environment, "environment_after": after, "trace": trace,
                                "outside_clock": {"warmup_seconds_per_worker": warm_seconds,
                                    "native_modules_preloaded": preloaded,
                                    "native_module_preload_seconds": preload_seconds, **input_meta},
                                "result": {"validation_accuracy": scores[0], "test_accuracy": scores[1],
                                    "concept_sha256": accuracy.array_sha256(concept),
                                    "concept_norm": float(np.linalg.norm(concept.astype(np.float64))),
                                    "method_details": jsonable(result["method_details"])}}
                            record["payload_sha256"] = digest(record)
                            write_json(checkpoint, record)
                            load_record(checkpoint, identity)
                            counts["computed"] += 1
                            print(f"computed task={index} layer={layer} method={method} seconds={record['wall_seconds']:.6g}", flush=True)
                        except BaseException as exc:
                            write_json(checkpoint.with_suffix(".failed.json"), {"status": "failed", "identity": identity,
                                "error": repr(exc), "not_an_accepted_timing": True})
                            raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return {**counts, "interrupted_after_atomic_unit": _STOP}


def aggregate(path):
    """Strict full denominator. Never merge another run or historical timing."""
    root = Path(path).resolve().parent
    manifest = read_json(path)
    validate_manifest(manifest)
    rows, groups, comparison = [], {}, set()
    for setting, split, layer in tasks(manifest):
        for method in METHODS:
            checkpoint = checkpoint_path(root, setting, split, layer, method)
            try:
                record = load_record(checkpoint, unit_identity(manifest, setting, split, layer, method))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"incomplete or incompatible grid: {checkpoint}: {exc}") from exc
            comparison.add(record["environment"]["comparison_sha256"])
            key = (setting["model"], setting["dataset"], method, split["seed"])
            groups.setdefault(key, []).append((layer, record))
    if len(comparison) != 1:
        raise ValueError("mixed resource/backend/environment comparison classes")
    for key, values in groups.items():
        layers = [v[0] for v in values]
        selected = min(values, key=lambda item: (-item[1]["result"]["validation_accuracy"], item[0]))
        rows.append({"model": key[0], "dataset": key[1], "method": key[2], "seed": key[3],
                     "layers": layers, "layer_count": len(layers),
                     "median_layer_seconds": float(np.median([v[1]["wall_seconds"] for v in values])),
                     "selected_layer_from_fresh_validation": selected[0],
                     "selected_validation_accuracy": selected[1]["result"]["validation_accuracy"]})
    setting_points = []
    for setting in manifest["settings"]:
        for method in METHODS:
            seed_rows = [next(r for r in rows if (r["model"], r["dataset"], r["method"], r["seed"]) ==
                         (setting["model"], setting["dataset"], method, seed)) for seed in manifest["seeds"]]
            values = [r["median_layer_seconds"] for r in seed_rows]
            setting_points.append({"model": setting["model"], "dataset": setting["dataset"], "method": method,
                "seeds": manifest["seeds"], "seed_median_seconds": values, "n_seeds": len(values),
                "mean_seed_median_seconds": float(np.mean(values)),
                "sample_sd_seed_median_seconds": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    summary = {"schema": SCHEMA, "scope": manifest["scope"], "run_id": manifest["run_id"],
        "manifest_sha256": digest(manifest), "source_sha256": manifest["source"]["sha256"],
        "aggregation": manifest["aggregation"], "resource_comparison_sha256": next(iter(comparison)),
        "method_unit_count": manifest["method_unit_count"], "seed_rows": rows, "setting_points": setting_points,
        "median_setting_seconds": {method: float(np.median([p["mean_seed_median_seconds"] for p in setting_points
                                                               if p["method"] == method])) for method in METHODS}}
    write_json(root / "aggregate.json", summary)
    return summary


def plot(aggregate_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = read_json(aggregate_path)
    if summary.get("schema") != SCHEMA or set(summary["median_setting_seconds"]) != set(METHODS):
        raise ValueError("plot requires the strict five-method aggregate")
    labels = {"singlelr": "RAPTOR", "xrfm": "xRFM", "gcs": "GCS", "fixed_reg": "Fixed-reg", "dim_midpoint": "CAA"}
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, method in enumerate(METHODS):
        points = [p["mean_seed_median_seconds"] for p in summary["setting_points"] if p["method"] == method]
        offsets = np.linspace(-.15, .15, len(points))
        ax.scatter(np.full(len(points), i) + offsets, points, s=15, alpha=.6)
        ax.scatter([i], [summary["median_setting_seconds"][method]], marker="D", s=65, c="black", zorder=4)
    ax.set_xticks(range(len(METHODS)), [labels[m] for m in METHODS])
    ax.set_yscale("log")
    ax.set_ylabel("Seconds per native layer pipeline")
    ax.set_title("Synthetic pipeline canary" if summary["scope"] == "synthetic" else "36 settings; five seeds; complete layers")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix("." + extension), dpi=200)
    plt.close(fig)


def synthetic_canary(output):
    """Run all five actual methods at FULL native budgets on tiny embeddings."""
    root = Path(output).resolve()
    emb = root / "inputs"
    emb.mkdir(parents=True, exist_ok=True)
    path = emb / "synthetic_tiny_embeddings.npz"
    rng = np.random.default_rng(9271)
    # Enough rows avoid single-class GCS tiny-subsample failures without changing its budget.
    arrays = {}
    for layer in range(2):
        arrays[f"X_pos_{layer}"] = (rng.normal(size=(400, 6)) + .2).astype(np.float32)
        arrays[f"X_neg_{layer}"] = (rng.normal(size=(400, 6)) - .2).astype(np.float32)
    accuracy.atomic_savez(path, **arrays)
    config = {"models": ["synthetic"], "datasets": ["tiny"], "seeds": list(SEEDS),
              "layer_counts": {"synthetic": 2}}
    manifest = prepare_manifest(config, emb, root / "run", scope="synthetic")
    first = run_manifest(root / "run/manifest.json", all_tasks=True)
    resumed = run_manifest(root / "run/manifest.json", all_tasks=True)
    if first["computed"] != 50 or resumed["restored"] != 50 or resumed["computed"]:
        raise ValueError("five-method real canary did not write/resume all50native units")
    summary = aggregate(root / "run/manifest.json")
    plot(root / "run/aggregate.json", root / "run/timing")
    report = {"only_synthetic": True, "full_native_budgets": True, "first": first, "resumed": resumed,
              "methods": list(METHODS), "seeds": list(SEEDS), "layers": [0, 1],
              "method_units": manifest["method_unit_count"], "resource_comparison_sha256": summary["resource_comparison_sha256"]}
    write_json(root / "CANARY.json", report)
    return report


def plan_grid(config, embeddings):
    """Read array headers only; inspect the full grid without hashing or training."""
    rows, paths, count = [], set(), 0
    for model in config["models"]:
        for dataset in config["datasets"]:
            path = Path(embeddings) / f"{model_tag(model)}_{dataset}_embeddings.npz"
            header = embedding_headers(path, model=model, dataset=dataset)
            expected = config["layer_counts"][model]
            if header["layers"] != list(range(expected)):
                raise ValueError(f"incomplete physical layers: {model}/{dataset}")
            for seed in config["seeds"]:
                for layer in header["layers"]:
                    for method in METHODS:
                        target = f"{model_tag(model)}/{dataset}/seed_{seed}/layer_{layer:03d}_{method}.json"
                        if target in paths:
                            raise ValueError("output collision")
                        paths.add(target)
            count += len(header["layers"]) * len(config["seeds"])
            rows.append({"model": model, "dataset": dataset, **header})
    if len(rows) != 36 or config["seeds"] != list(SEEDS) or count != 8040 or len(paths) != 40200:
        raise ValueError("formal grid differs from36settings/5seeds/8040layer-seed tasks/40200method units")
    return {"scope": "full", "status": "header_dry_run_no_training", "setting_count": len(rows),
            "task_count": count, "method_unit_count": len(paths), "seeds": config["seeds"], "settings": rows,
            "content_hashing": "performed by prepare; headers only in this dry run"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="prepare new input/split/run identities without training")
    prep.add_argument("--config", type=Path, default=Path("configs/timing.yaml"))
    prep.add_argument("--embeddings", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--seeds", help="explicit comma-separated outer seeds; formal scope requires42-46")
    plan = sub.add_parser("plan", help="header-only full-grid validation without training")
    plan.add_argument("--config", type=Path, default=Path("configs/timing.yaml"))
    plan.add_argument("--embeddings", type=Path, required=True)
    plan.add_argument("--output", type=Path)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--task-index", type=int)
    run.add_argument("--all", action="store_true", dest="all_tasks")
    agg = sub.add_parser("aggregate")
    agg.add_argument("--manifest", type=Path, required=True)
    plotting = sub.add_parser("plot")
    plotting.add_argument("--aggregate", type=Path, required=True)
    plotting.add_argument("--output", type=Path, required=True)
    canary = sub.add_parser("canary", help="actual five-method tiny synthetic run, full native budgets")
    canary.add_argument("--output", type=Path, required=True)
    dry = sub.add_parser("validate", help="verify full input/split/task coverage without training")
    dry.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        import yaml
        result = plan_grid(yaml.safe_load(args.config.read_text()), args.embeddings)
        if args.output:
            write_json(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k != "settings"}))
    elif args.command == "prepare":
        import yaml
        config = yaml.safe_load(args.config.read_text())
        if args.seeds:
            config["seeds"] = accuracy.parse_seeds(args.seeds)
        result = prepare_manifest(config, args.embeddings, args.output)
        print(json.dumps({k: result[k] for k in ("run_id", "setting_count", "task_count", "method_unit_count")}))
    elif args.command == "run":
        print(json.dumps(run_manifest(args.manifest, task_index=args.task_index, all_tasks=args.all_tasks)))
    elif args.command == "aggregate":
        result = aggregate(args.manifest)
        print(json.dumps(result["median_setting_seconds"]))
    elif args.command == "plot":
        plot(args.aggregate, args.output)
    elif args.command == "canary":
        print(json.dumps(synthetic_canary(args.output)))
    elif args.command == "validate":
        manifest = read_json(args.manifest)
        validate_manifest(manifest)
        root = args.manifest.resolve().parent
        for setting in manifest["settings"]:
            # Check every seed and layer header; content-hash full cache once.
            for i, split in enumerate(setting["splits"]):
                load_layer(root, setting, split, setting["embedding"]["layers"][0], verify_embedding=i == 0)
        print(json.dumps({"status": "valid_no_training", "settings": manifest["setting_count"],
                          "tasks": manifest["task_count"], "method_units": manifest["method_unit_count"]}))


if __name__ == "__main__":
    main()
