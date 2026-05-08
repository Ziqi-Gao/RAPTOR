#!/usr/bin/env python3
import argparse
import json
import math
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_KEYS = {
    "RALoP": "singlelr_sec",
    "xRFM": "xrfm_sec",
    "GCS": "gcs_sec",
}

MODEL_SIZES = {
    "Qwen-Qwen2.5-3B-Instruct": 3.0,
    "Qwen-Qwen2.5-7B-Instruct": 7.0,
    "Qwen-Qwen2.5-32B-Instruct": 32.0,
    "google-gemma-7b-it": 7.0,
    "meta-llama-Meta-Llama-3.1-8B-Instruct": 8.0,
    "meta-llama-Meta-Llama-3.1-70B-Instruct": 70.0,
    "meta-llama-Llama-3.3-70B-Instruct": 70.0,
}


def _load_meta(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_size(model_tag: str) -> float:
    if model_tag in MODEL_SIZES:
        return MODEL_SIZES[model_tag]
    m = re.search(r"(\\d+(?:\\.\\d+)?)B", model_tag)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return float("nan")


def _collect_per_layer_times(results_dir: str):
    # per_model_method_dataset_times[model][method] = [per_layer_time for each dataset]
    per_model_method_dataset_times = {}

    for model_tag in os.listdir(results_dir):
        model_dir = os.path.join(results_dir, model_tag)
        if not os.path.isdir(model_dir):
            continue
        for dataset in os.listdir(model_dir):
            layers_dir = os.path.join(model_dir, dataset, "layers")
            if not os.path.isdir(layers_dir):
                continue
            totals = {label: 0.0 for label in METHOD_KEYS}
            counts = {label: 0 for label in METHOD_KEYS}
            for layer_name in os.listdir(layers_dir):
                if not layer_name.startswith("layer_"):
                    continue
                meta_path = os.path.join(layers_dir, layer_name, "task_meta.json")
                if not os.path.isfile(meta_path):
                    continue
                meta = _load_meta(meta_path)
                timings = meta.get("timings_sec", {})
                for label, key in METHOD_KEYS.items():
                    if key not in timings:
                        continue
                    totals[label] += float(timings[key])
                    counts[label] += 1
            for label in METHOD_KEYS:
                if counts[label] == 0:
                    continue
                per_layer = totals[label] / counts[label]
                per_model_method_dataset_times.setdefault(model_tag, {}).setdefault(label, []).append(per_layer)

    return per_model_method_dataset_times


def _aggregate(per_model_method_dataset_times):
    # per_model_method_stats[model][method] = (mean, std)
    per_model_method_stats = {}
    for model_tag, method_map in per_model_method_dataset_times.items():
        for method, vals in method_map.items():
            arr = np.array(vals, dtype=float)
            if arr.size == 0:
                continue
            mean = float(arr.mean())
            std = float(arr.std()) if arr.size > 1 else 0.0
            per_model_method_stats.setdefault(model_tag, {})[method] = (mean, std)
    return per_model_method_stats


def _make_jitter(models_by_size, base_jitter=0.06):
    jitter = {}
    for size, models in models_by_size.items():
        n = len(models)
        if n == 1:
            jitter[models[0]] = 0.0
            continue
        offsets = np.linspace(-base_jitter, base_jitter, n)
        for m, off in zip(sorted(models), offsets):
            jitter[m] = float(off)
    return jitter


def _plot(stats, out_path: str, use_log: bool):
    if not stats:
        return False

    models = sorted(stats.keys(), key=lambda m: (_infer_size(m), m))
    sizes = {m: _infer_size(m) for m in models}
    models_by_size = {}
    for m, s in sizes.items():
        if math.isnan(s):
            continue
        models_by_size.setdefault(s, []).append(m)
    jitter = _make_jitter(models_by_size)

    method_offsets = {"RALoP": -0.25, "xRFM": 0.0, "GCS": 0.25}
    colors = {"RALoP": "#7b2cbf", "xRFM": "#d62728", "GCS": "#1f77b4"}

    size_levels = sorted({s for s in sizes.values() if not math.isnan(s)})
    size_to_x = {s: i for i, s in enumerate(size_levels)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in ["RALoP", "xRFM", "GCS"]:
        xs, ys, yerr = [], [], []
        for m in models:
            if method not in stats.get(m, {}):
                continue
            size = sizes[m]
            if math.isnan(size):
                continue
            mean, std = stats[m][method]
            base_x = size_to_x.get(size)
            if base_x is None:
                continue
            xs.append(base_x + method_offsets[method] + jitter.get(m, 0.0))
            ys.append(mean)
            yerr.append(std)
        if not xs:
            continue
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt="o",
            markersize=5,
            capsize=3,
            linewidth=1.2,
            color=colors[method],
            label=method,
        )

    ax.set_xlabel("Model size (B)")
    ax.set_ylabel("Seconds per layer")
    ax.set_title("Per layer training time")
    ax.grid(True, alpha=0.25)
    if use_log:
        ax.set_yscale("log")
    xticks = list(range(len(size_levels)))
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [str(int(x)) if float(x).is_integer() else str(x) for x in size_levels]
    )
    ax.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="./exp_results")
    ap.add_argument("--out_path", type=str, default="./plot/time_summary/per_layer_time_vs_scale.png")
    ap.add_argument("--log_scale", action="store_true", default=True)
    args = ap.parse_args()

    per_model_method_dataset_times = _collect_per_layer_times(args.results_dir)
    stats = _aggregate(per_model_method_dataset_times)
    ok = _plot(stats, args.out_path, args.log_scale)
    if ok:
        print(args.out_path)
    else:
        print("No data to plot.")


if __name__ == "__main__":
    main()
