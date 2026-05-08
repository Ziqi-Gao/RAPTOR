#!/usr/bin/env python3
import argparse
import json
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_KEYS = {
    "RALoP": "singlelr_sec",
    "xRFM": "xrfm_sec",
    "GCS": "gcs_sec",
}

MODEL_LABELS = {
    "meta-llama-Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama-Meta-Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "meta-llama-Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "Qwen-Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "Qwen-Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen-Qwen2.5-32B-Instruct": "Qwen2.5-32B",
    "google-gemma-7b-it": "Gemma-7B",
}

MODEL_ORDER = [
    "Qwen-Qwen2.5-3B-Instruct",
    "Qwen-Qwen2.5-7B-Instruct",
    "Qwen-Qwen2.5-32B-Instruct",
    "google-gemma-7b-it",
    "meta-llama-Meta-Llama-3.1-8B-Instruct",
    "meta-llama-Meta-Llama-3.1-70B-Instruct",
    "meta-llama-Llama-3.3-70B-Instruct",
]


def _load_meta(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_totals(results_dir: str):
    totals = {}

    for model_tag in os.listdir(results_dir):
        model_dir = os.path.join(results_dir, model_tag)
        if not os.path.isdir(model_dir):
            continue
        for dataset in os.listdir(model_dir):
            layers_dir = os.path.join(model_dir, dataset, "layers")
            if not os.path.isdir(layers_dir):
                continue
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
                    totals.setdefault(model_tag, {}).setdefault(label, 0.0)
                    totals[model_tag][label] += float(timings[key])

    return totals


def _ordered_models(totals):
    known = [m for m in MODEL_ORDER if m in totals]
    extra = sorted([m for m in totals.keys() if m not in MODEL_ORDER])
    return known + extra


def _label_model(tag: str) -> str:
    return MODEL_LABELS.get(tag, tag)


def _to_hours(seconds: float) -> float:
    return seconds / 3600.0


def _plot(totals, out_path: str):
    models = _ordered_models(totals)
    if not models:
        return False

    labels = [_label_model(m) for m in models]
    x = np.arange(len(models))
    width = 0.25

    colors = {
        "RALoP": "#7b2cbf",
        "xRFM": "#d62728",
        "GCS": "#1f77b4",
    }

    fig_w = max(10, 1.2 * len(models))
    fig_h = 6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for i, method in enumerate(["RALoP", "xRFM", "GCS"]):
        vals = []
        for m in models:
            sec = totals.get(m, {}).get(method)
            if sec is None or sec <= 0:
                vals.append(np.nan)
            else:
                vals.append(_to_hours(sec))
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width=width, label=method, color=colors[method])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Total wall-clock hours (log scale)")
    ax.set_xlabel("Model")
    ax.set_title("Total probing time per model (aggregated across datasets and layers)")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="./exp_results")
    ap.add_argument("--out_path", type=str, default="./plot/time_summary/total_time_per_model.png")
    args = ap.parse_args()

    totals = _collect_totals(args.results_dir)
    ok = _plot(totals, args.out_path)
    if ok:
        print(args.out_path)
    else:
        print("No data to plot.")


if __name__ == "__main__":
    main()
