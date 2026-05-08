#!/usr/bin/env python3
import argparse
import json
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

METHOD_COLORS = {
    "RALoP": "#7b2cbf",
    "xRFM": "#d62728",
    "GCS": "#1f77b4",
}


def _load_meta(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_samples(results_dir: str):
    samples = {k: [] for k in METHOD_KEYS}
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
                    val = timings.get(key)
                    if val is None:
                        continue
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        continue
                    if val <= 0:
                        continue
                    samples[label].append(val)
    return samples


def _plot(samples, out_path: str):
    order = ["RALoP", "xRFM", "GCS"]
    data = [samples.get(label, []) for label in order]
    if not any(len(v) for v in data):
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    vp = ax.violinplot(
        data,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    for body, label in zip(vp["bodies"], order):
        body.set_facecolor(METHOD_COLORS[label])
        body.set_edgecolor("#333333")
        body.set_alpha(0.6)
    for key in ("cmins", "cmaxes", "cbars", "cmedians"):
        if key in vp:
            vp[key].set_color("#222222")
            vp[key].set_linewidth(1.2)

    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels([f"{label} (n={len(samples.get(label, []))})" for label in order])
    ax.set_xlabel("Method")
    ax.set_ylabel("Seconds per layer (log scale)")
    ax.set_title("Per layer training time")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="./exp_results")
    ap.add_argument("--out_path", type=str, default="./plot/time_summary/per_layer_time_violin.png")
    args = ap.parse_args()

    samples = _collect_samples(args.results_dir)
    ok = _plot(samples, args.out_path)
    if ok:
        print(args.out_path)
    else:
        print("No data to plot.")


if __name__ == "__main__":
    main()
