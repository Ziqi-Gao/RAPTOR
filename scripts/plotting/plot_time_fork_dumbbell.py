#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]

METHOD_KEYS = {
    "RAPTOR": "singlelr_sec",
    "xRFM": "xrfm_sec",
    "GCS": "gcs_sec",
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

DATASET_ORDER = ["STSA", "sarcasm", "hatexplain", "counterfact", "cities", "common"]
DATASET_LABELS = {
    "STSA": "STSA",
    "sarcasm": "Sarcasm",
    "hatexplain": "HateXplain",
    "counterfact": "CounterFact",
    "cities": "Cities",
    "common": "Common",
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

METHOD_MARKERS = {
    "RAPTOR": "o",
    "xRFM": "^",
    "GCS": "s",
}

METHOD_COLORS = {
    "RAPTOR": "#7b2cbf",
    "xRFM": "#d62728",
    "GCS": "#1f77b4",
}


def _load_meta(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_medians(results_dir: str):
    # values[dataset][model][method] -> list of per-layer seconds
    values = {}

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
                for method, key in METHOD_KEYS.items():
                    val = timings.get(key)
                    if val is None:
                        continue
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        continue
                    if val <= 0:
                        continue
                    values.setdefault(dataset, {}).setdefault(model_tag, {}).setdefault(method, []).append(val)

    # median per (dataset, model, method)
    medians = {}
    for dataset, model_map in values.items():
        for model_tag, method_map in model_map.items():
            for method, vals in method_map.items():
                if not vals:
                    continue
                med = float(np.median(np.asarray(vals, dtype=float)))
                medians.setdefault(dataset, {}).setdefault(model_tag, {})[method] = med

    return medians


def _label_model(model_tag: str) -> str:
    return MODEL_LABELS.get(model_tag, model_tag)


def _plot(medians, out_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True, sharey=True)
    axes = axes.ravel()

    model_order = list(reversed(MODEL_ORDER))

    all_vals = []
    for dataset in DATASET_ORDER:
        for model_tag in model_order:
            vals = medians.get(dataset, {}).get(model_tag, {})
            for method in METHOD_KEYS:
                v = vals.get(method)
                if v is not None and v > 0:
                    all_vals.append(v)
    if all_vals:
        xmin = min(all_vals) * 0.8
        xmax = max(all_vals) * 1.2
    else:
        xmin, xmax = 1.0, 1000.0

    for idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[idx]
        ax.set_title(DATASET_LABELS.get(dataset, dataset), pad=6, fontsize=14)
        for row, model_tag in enumerate(model_order):
            y = row
            vals = medians.get(dataset, {}).get(model_tag, {})
            r = vals.get("RAPTOR")
            x = vals.get("xRFM")
            g = vals.get("GCS")

            if r is not None:
                ax.scatter(
                    r,
                    y,
                    marker=METHOD_MARKERS["RAPTOR"],
                    color=METHOD_COLORS["RAPTOR"],
                    s=60,
                    zorder=3,
                    edgecolor="white",
                    linewidth=0.6,
                )
            if x is not None:
                ax.scatter(
                    x,
                    y,
                    marker=METHOD_MARKERS["xRFM"],
                    color=METHOD_COLORS["xRFM"],
                    s=60,
                    zorder=3,
                    edgecolor="white",
                    linewidth=0.6,
                )
            if g is not None:
                ax.scatter(
                    g,
                    y,
                    marker=METHOD_MARKERS["GCS"],
                    color=METHOD_COLORS["GCS"],
                    s=60,
                    zorder=3,
                    edgecolor="white",
                    linewidth=0.6,
                )

            if r is not None and x is not None:
                ax.plot([r, x], [y, y], color="#888888", linewidth=1.6, linestyle="-")
            if r is not None and g is not None:
                ax.plot([r, g], [y, y], color="#888888", linewidth=1.6, linestyle="-")

        ax.set_yticks(range(len(model_order)))
        col = idx % 3
        if col == 0:
            ax.set_yticklabels([_label_model(m) for m in model_order], fontsize=13)
        else:
            ax.tick_params(labelleft=False)

        ax.grid(True, axis="x", alpha=0.25)
        ax.grid(False, axis="y")
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.vlines([10, 100, 1000], ymin=-0.5, ymax=len(model_order) - 0.5, colors="#444444", alpha=0.15, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=13)

    for ax in axes[len(DATASET_ORDER):]:
        ax.axis("off")

    fig.supxlabel("Median seconds per layer (log scale)", y=0.02, fontsize=15)
    fig.supylabel("Model", x=0.035, fontsize=15)

    legend_items = [
        Line2D([0], [0], marker=METHOD_MARKERS["RAPTOR"], color="none",
               markerfacecolor=METHOD_COLORS["RAPTOR"], markersize=7, label="RAPTOR"),
        Line2D([0], [0], marker=METHOD_MARKERS["xRFM"], color="none",
               markerfacecolor=METHOD_COLORS["xRFM"], markersize=7, label="xRFM"),
        Line2D([0], [0], marker=METHOD_MARKERS["GCS"], color="none",
               markerfacecolor=METHOD_COLORS["GCS"], markersize=7, label="GCS"),
    ]
    fig.legend(handles=legend_items, loc="upper center", frameon=False, ncol=5, bbox_to_anchor=(0.5, 0.992), fontsize=13)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout(rect=[0.05, 0.055, 0.95, 0.94])
    plt.subplots_adjust(wspace=0.12, hspace=0.22, top=0.925, bottom=0.08, left=0.15)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="./exp_results")
    ap.add_argument("--out_path", type=str, default="./plot/time_summary/per_layer_time_fork_dumbbell.png")
    args = ap.parse_args()

    medians = _collect_medians(args.results_dir)
    if not medians:
        print("No data to plot.")
        return
    _plot(medians, args.out_path)
    print(args.out_path)


if __name__ == "__main__":
    main()
