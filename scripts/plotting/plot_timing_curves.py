#!/usr/bin/env python3
import argparse
import json
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_KEYS = ("singlelr_sec", "xrfm_sec", "gcs_sec")


def _load_meta(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect(results_dir: str):
    layer_pat = re.compile(r"layer_(\d+)$")
    out = {}
    all_times = {k: [] for k in METHOD_KEYS}

    for model_tag in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model_tag)
        if not os.path.isdir(model_dir):
            continue
        for dataset in sorted(os.listdir(model_dir)):
            layers_dir = os.path.join(model_dir, dataset, "layers")
            if not os.path.isdir(layers_dir):
                continue
            key = (model_tag, dataset)
            out.setdefault(key, {})
            for layer_name in os.listdir(layers_dir):
                m = layer_pat.match(layer_name)
                if not m:
                    continue
                layer_idx = int(m.group(1))
                meta_path = os.path.join(layers_dir, layer_name, "task_meta.json")
                if not os.path.isfile(meta_path):
                    continue
                meta = _load_meta(meta_path)
                timings = meta.get("timings_sec", {})
                if not all(k in timings for k in METHOD_KEYS):
                    continue
                out[key][layer_idx] = {k: float(timings[k]) for k in METHOD_KEYS}
                for k in METHOD_KEYS:
                    all_times[k].append(float(timings[k]))

    return out, all_times


def _plot_group(model_tag: str, dataset: str, layer_map, out_dir: str) -> str:
    layers = sorted(layer_map.keys())
    if not layers:
        return ""
    y_single = [layer_map[l]["singlelr_sec"] for l in layers]
    y_xrfm = [layer_map[l]["xrfm_sec"] for l in layers]
    y_gcs = [layer_map[l]["gcs_sec"] for l in layers]

    plt.figure(figsize=(10, 4.5))
    plt.plot(layers, y_single, color="#7b2cbf", linewidth=1.5, label="SingleLR")
    plt.plot(layers, y_xrfm, color="#d62728", linewidth=1.5, label="xRFM")
    plt.plot(layers, y_gcs, color="#1f77b4", linewidth=1.5, label="GCS")
    plt.xlabel("Layer")
    plt.ylabel("Seconds")
    plt.title(f"{model_tag} | {dataset} | n={len(layers)}")
    all_vals = y_single + y_xrfm + y_gcs
    if all_vals:
        ymin = min(all_vals)
        ymax = max(all_vals)
        if ymin == ymax:
            pad = 0.01
        else:
            pad = (ymax - ymin) * 0.08
        plt.ylim(ymin - pad, ymax + pad)
    plt.grid(True, alpha=0.25)
    plt.legend()

    os.makedirs(out_dir, exist_ok=True)
    fname = f"{model_tag}__{dataset}__time.png"
    out_path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _summarize(all_times):
    summary = {}
    for k, vals in all_times.items():
        arr = np.array(vals, dtype=float)
        if arr.size == 0:
            summary[k] = None
            continue
        summary[k] = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="./exp_results")
    ap.add_argument("--out_dir", type=str, default="./plot/timing_curves")
    args = ap.parse_args()

    groups, all_times = _collect(args.results_dir)
    total = 0
    for (model_tag, dataset), layer_map in sorted(groups.items()):
        out_path = _plot_group(model_tag, dataset, layer_map, args.out_dir)
        if out_path:
            total += 1
    summary = _summarize(all_times)
    print(f"Saved {total} plots to {args.out_dir}")
    print("Summary (seconds):")
    for k in METHOD_KEYS:
        stats = summary.get(k)
        if not stats:
            print(f"  {k}: no data")
            continue
        print(
            f"  {k}: n={stats['count']} mean={stats['mean']:.3f} "
            f"median={stats['median']:.3f} p90={stats['p90']:.3f}"
        )


if __name__ == "__main__":
    main()
