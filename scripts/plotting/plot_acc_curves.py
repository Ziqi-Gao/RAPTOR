#!/usr/bin/env python3
import argparse
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_acc(path: str) -> float:
    data = np.load(path, allow_pickle=False)
    if "acc_test" not in data:
        raise KeyError(f"missing acc_test in {path}")
    val = data["acc_test"]
    return float(np.asarray(val).reshape(-1)[0])


def _load_acc_vector(path: str):
    data = np.load(path, allow_pickle=False)
    if "acc_test" not in data:
        raise KeyError(f"missing acc_test in {path}")
    vec = np.asarray(data["acc_test"], dtype=float).reshape(-1)
    return {i: float(vec[i]) for i in range(len(vec))}


def _collect_results(results_dir: str):
    layer_pat = re.compile(r"layer_(\d+)$")
    methods = ("singlelr", "xrfm", "gcs")
    req = {
        "singlelr": "singlelr_results.npz",
        "xrfm": "rfm_results.npz",
        "gcs": "gcs_results.npz",
    }
    out = {}
    for model_tag in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model_tag)
        if not os.path.isdir(model_dir):
            continue
        for dataset in sorted(os.listdir(model_dir)):
            ds_dir = os.path.join(model_dir, dataset)
            layers_dir = os.path.join(ds_dir, "layers")
            if not os.path.isdir(layers_dir):
                continue
            key = (model_tag, dataset)
            out.setdefault(key, {})
            for layer_name in os.listdir(layers_dir):
                m = layer_pat.match(layer_name)
                if not m:
                    continue
                layer_idx = int(m.group(1))
                layer_dir = os.path.join(layers_dir, layer_name)
                if not os.path.isdir(layer_dir):
                    continue
                paths = {k: os.path.join(layer_dir, req[k]) for k in methods}
                if not all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in paths.values()):
                    continue
                out[key][layer_idx] = {
                    "singlelr": _load_acc(paths["singlelr"]),
                    "xrfm": _load_acc(paths["xrfm"]),
                    "gcs": _load_acc(paths["gcs"]),
                }
            mlp_path = os.path.join(ds_dir, "mlp_results.npz")
            if os.path.isfile(mlp_path) and os.path.getsize(mlp_path) > 0:
                try:
                    mlp_map = _load_acc_vector(mlp_path)
                    for l, v in mlp_map.items():
                        if l in out[key]:
                            out[key][l]["mlp"] = v
                except Exception:
                    pass
    return out


def _plot_group(model_tag: str, dataset: str, layer_map, out_dir: str) -> str:
    layers = sorted(layer_map.keys())
    if not layers:
        return ""

    def _normalize(vals):
        if len(vals) == 0:
            return vals, False
        finite = [v for v in vals if np.isfinite(v)]
        if not finite:
            return vals, False
        if max(finite) > 1.0:
            return [(v / 100.0) if np.isfinite(v) else np.nan for v in vals], True
        return vals, False

    y_single_raw = [layer_map[l]["singlelr"] for l in layers]
    y_xrfm_raw = [layer_map[l]["xrfm"] for l in layers]
    y_gcs_raw = [layer_map[l]["gcs"] for l in layers]
    y_mlp_raw = [layer_map[l].get("mlp", np.nan) for l in layers]

    y_single, _ = _normalize(y_single_raw)
    y_xrfm, xrfm_scaled = _normalize(y_xrfm_raw)
    y_gcs, _ = _normalize(y_gcs_raw)
    y_mlp, mlp_scaled = _normalize(y_mlp_raw)

    plt.figure(figsize=(10, 4.5))
    plt.plot(layers, y_single, color="#7b2cbf", linewidth=1.5, label="RAPTOR")
    plt.plot(layers, y_xrfm, color="#d62728", linewidth=1.5, label="xRFM")
    plt.plot(layers, y_gcs, color="#1f77b4", linewidth=1.5, label="GCS")
    if any(np.isfinite(v) for v in y_mlp):
        plt.plot(layers, y_mlp, color="#2ca02c", linewidth=1.5, label="MLP")

    all_vals = [v for v in (y_single + y_xrfm + y_gcs + y_mlp) if np.isfinite(v)]
    if all_vals:
        ymin = float(min(all_vals))
        ymax = float(max(all_vals))
        if ymin == ymax:
            pad = 0.01
        else:
            pad = (ymax - ymin) * 0.08
        plt.ylim(ymin - pad, ymax + pad)
    plt.xlabel("Layer")
    plt.ylabel("Test Accuracy")
    scale_note_parts = []
    if xrfm_scaled:
        scale_note_parts.append("xRFM/100")
    if mlp_scaled:
        scale_note_parts.append("MLP/100")
    scale_note = f" ({', '.join(scale_note_parts)})" if scale_note_parts else ""
    plt.title(f"{model_tag} | {dataset} | n={len(layers)}{scale_note}")
    plt.grid(True, alpha=0.25)
    plt.legend()

    os.makedirs(out_dir, exist_ok=True)
    fname = f"{model_tag}__{dataset}__acc.png"
    out_path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="./exp_results")
    ap.add_argument("--out_dir", type=str, default="./plot/acc_curves")
    args = ap.parse_args()

    groups = _collect_results(args.results_dir)
    total = 0
    for (model_tag, dataset), layer_map in sorted(groups.items()):
        out_path = _plot_group(model_tag, dataset, layer_map, args.out_dir)
        if out_path:
            total += 1
    print(f"Saved {total} plots to {args.out_dir}")


if __name__ == "__main__":
    main()
