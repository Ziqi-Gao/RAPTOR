#!/usr/bin/env python3
import argparse
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]

MODEL_ORDER = [
    "Qwen-Qwen2.5-3B-Instruct",
    "Qwen-Qwen2.5-7B-Instruct",
    "Qwen-Qwen2.5-32B-Instruct",
    "google-gemma-7b-it",
    "meta-llama-Meta-Llama-3.1-8B-Instruct",
    "meta-llama-Meta-Llama-3.1-70B-Instruct",
    "meta-llama-Llama-3.3-70B-Instruct",
]

def _load_acc(path: str) -> float:
    data = np.load(path, allow_pickle=False)
    if "acc_test" not in data:
        raise KeyError(f"missing acc_test in {path}")
    val = float(np.asarray(data["acc_test"]).reshape(-1)[0])
    if val > 1.0:
        val = val / 100.0
    return val


def _collect(results_dir: str):
    data = {}
    models = set()
    datasets = set()

    for model_tag in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model_tag)
        if not os.path.isdir(model_dir):
            continue
        models.add(model_tag)
        for dataset in sorted(os.listdir(model_dir)):
            layers_dir = os.path.join(model_dir, dataset, "layers")
            if not os.path.isdir(layers_dir):
                continue
            datasets.add(dataset)
            for layer_name in os.listdir(layers_dir):
                if not layer_name.startswith("layer_"):
                    continue
                layer_dir = os.path.join(layers_dir, layer_name)
                if not os.path.isdir(layer_dir):
                    continue
                single_path = os.path.join(layer_dir, "singlelr_results.npz")
                rfm_path = os.path.join(layer_dir, "rfm_results.npz")
                gcs_path = os.path.join(layer_dir, "gcs_results.npz")
                if not (os.path.isfile(single_path) and os.path.getsize(single_path) > 0):
                    continue
                entry = data.setdefault(model_tag, {}).setdefault(dataset, {})
                entry.setdefault("singlelr", []).append(_load_acc(single_path))
                if os.path.isfile(rfm_path) and os.path.getsize(rfm_path) > 0:
                    entry.setdefault("rfm", []).append(_load_acc(rfm_path))
                if os.path.isfile(gcs_path) and os.path.getsize(gcs_path) > 0:
                    entry.setdefault("gcs", []).append(_load_acc(gcs_path))

    return sorted(models), sorted(datasets), data


def _aggregate(data, agg: str):
    out = {}
    for model_tag, ds_map in data.items():
        for dataset, method_map in ds_map.items():
            entry = out.setdefault(model_tag, {}).setdefault(dataset, {})
            for method, vals in method_map.items():
                if not vals:
                    continue
                if agg == "best":
                    entry[method] = float(max(vals))
                else:
                    entry[method] = float(sum(vals) / len(vals))
    return out


def _short_model_name(model_tag: str) -> str:
    mapping = {
        "meta-llama-Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
        "meta-llama-Meta-Llama-3.1-70B-Instruct": "Llama-3.1-70B",
        "meta-llama-Llama-3.3-70B-Instruct": "Llama-3.3-70B",
        "Qwen-Qwen2.5-3B-Instruct": "Qwen2.5-3B",
        "Qwen-Qwen2.5-7B-Instruct": "Qwen2.5-7B",
        "Qwen-Qwen2.5-32B-Instruct": "Qwen2.5-32B",
        "google-gemma-7b-it": "Gemma-7B",
    }
    return mapping.get(model_tag, model_tag)


def _order_models(models):
    ordered = [m for m in MODEL_ORDER if m in models]
    ordered.extend([m for m in models if m not in ordered])
    return ordered


def _build_matrix(models, datasets, data, left_key: str, right_key: str):
    rows = len(models)
    cols = len(datasets)
    mat = np.full((rows, cols), np.nan, dtype=float)
    for i, model_tag in enumerate(models):
        for j, dataset in enumerate(datasets):
            vals = data.get(model_tag, {}).get(dataset, {})
            if left_key not in vals or right_key not in vals:
                continue
            base = float(vals[right_key])
            if base == 0.0:
                continue
            delta = float(vals[left_key]) - base
            mat[i, j] = (delta / base) * 100.0
    return mat


def _filter_rows(mat, row_labels):
    keep = [i for i in range(mat.shape[0]) if np.isfinite(mat[i]).any()]
    if not keep:
        return mat[:0, :], []
    return mat[keep, :], [row_labels[i] for i in keep]


def _filter_common(mats, row_labels, col_labels):
    if not mats:
        return mats, [], []
    rows = mats[0].shape[0]
    cols = mats[0].shape[1]
    row_keep = []
    for i in range(rows):
        if any(np.isfinite(mat[i]).any() for mat in mats):
            row_keep.append(i)
    col_keep = []
    for j in range(cols):
        if any(np.isfinite(mat[:, j]).any() for mat in mats):
            col_keep.append(j)
    if not row_keep or not col_keep:
        return [m[:0, :0] for m in mats], [], []
    filtered = [m[np.ix_(row_keep, col_keep)] for m in mats]
    return filtered, [row_labels[i] for i in row_keep], [col_labels[j] for j in col_keep]


def _make_cmap_norm(mats):
    finite = np.concatenate([m[np.isfinite(m)] for m in mats if m.size])
    if finite.size == 0:
        return None, None
    vmax = float(np.max(np.abs(finite)))
    if vmax == 0.0:
        vmax = 1e-6
    neg_color = "#2a9d8f"  # teal
    pos_color = "#7b2cbf"  # purple
    cmap = LinearSegmentedColormap.from_list("teal_white_purple", [neg_color, "#ffffff", pos_color])
    cmap.set_bad(color="#f0f0f0")
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    return cmap, norm


def _plot_heatmap(mat, models, datasets, title, out_path):
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return False
    vmax = float(np.max(np.abs(finite)))
    if vmax == 0.0:
        vmax = 1e-6

    neg_color = "#2a9d8f"  # teal
    pos_color = "#7b2cbf"  # purple
    cmap = LinearSegmentedColormap.from_list("teal_white_purple", [neg_color, "#ffffff", pos_color])
    cmap.set_bad(color="#f0f0f0")
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig_w = max(8, 0.45 * len(datasets))
    fig_h = max(6, 0.4 * len(models))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Model", fontsize=14)
    ax.set_xlabel("Dataset", fontsize=14)
    ax.set_yticks(np.arange(len(models)))
    short_models = [_short_model_name(m) for m in models]
    ax.set_yticklabels(short_models, fontsize=13)
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=13)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy Difference (%)", fontsize=13)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def _plot_heatmap_grid(mats, titles, models, datasets, out_path, tight=False):
    if not mats or not models or not datasets:
        return False
    cmap, norm = _make_cmap_norm(mats)
    if cmap is None:
        return False

    fig_w = max(10, 0.6 * len(datasets))
    fig_h = max(8, 0.55 * len(models))
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.06], wspace=0.15, hspace=0.2)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1], sharex=fig.axes[0], sharey=fig.axes[0]),
        fig.add_subplot(gs[1, 0], sharex=fig.axes[0], sharey=fig.axes[0]),
        fig.add_subplot(gs[1, 1], sharex=fig.axes[0], sharey=fig.axes[0]),
    ]
    cax = fig.add_subplot(gs[:, 2])
    ims = []
    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ims.append(im)
        ax.set_title(title, fontsize=14)

    short_models = [_short_model_name(m) for m in models]
    for idx, ax in enumerate(axes):
        row = idx // 2
        col = idx % 2
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_yticks(np.arange(len(models)))
        if row == 1:
            ax.set_xticklabels(datasets, fontsize=13)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_yticklabels(short_models, fontsize=13)
        else:
            ax.tick_params(labelleft=False)

    fig.supxlabel("Dataset", y=0.05, fontsize=14)
    fig.supylabel("Model", x=0.005, fontsize=14)

    cbar = fig.colorbar(ims[0], cax=cax)
    cbar.set_label("Accuracy Difference (%)", fontsize=13)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.subplots_adjust(left=0.16, bottom=0.16, right=0.93, top=0.92)
    if tight:
        plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    else:
        plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def _plot_heatmap_row(mats, titles, models, datasets, out_path, tight=False):
    if not mats or not models or not datasets:
        return False
    cmap, norm = _make_cmap_norm(mats)
    if cmap is None:
        return False

    fig_w = max(14, 0.9 * len(datasets) * 4)
    fig_h = max(4.2, 0.55 * len(models))
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.06], wspace=0.15, hspace=0.2)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1], sharey=fig.axes[0]),
        fig.add_subplot(gs[0, 2], sharey=fig.axes[0]),
        fig.add_subplot(gs[0, 3], sharey=fig.axes[0]),
    ]
    cax = fig.add_subplot(gs[0, 4])

    ims = []
    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ims.append(im)
        ax.set_title(title, fontsize=14)

    short_models = [_short_model_name(m) for m in models]
    for idx, ax in enumerate(axes):
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(datasets, fontsize=13)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        if idx == 0:
            ax.set_yticklabels(short_models, fontsize=13)
        else:
            ax.tick_params(labelleft=False)

    fig.supxlabel("Dataset", y=0.06, fontsize=14)
    fig.supylabel("Model", x=0.01, fontsize=14)

    cbar = fig.colorbar(ims[0], cax=cax)
    cbar.set_label("Accuracy Difference (%)", fontsize=13)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.subplots_adjust(left=0.09, bottom=0.22, right=0.94, top=0.88)
    if tight:
        plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    else:
        plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="./exp_results")
    ap.add_argument("--out_dir", type=str, default="./plot/acc_diff_heatmaps")
    args = ap.parse_args()

    models, datasets, raw = _collect(args.results_dir)
    if not models or not datasets:
        print("No data found.")
        return
    models = _order_models(models)

    avg_data = _aggregate(raw, "avg")
    best_data = _aggregate(raw, "best")

    mat_avg_s_g = _build_matrix(models, datasets, avg_data, "singlelr", "gcs")
    mat_avg_s_r = _build_matrix(models, datasets, avg_data, "singlelr", "rfm")
    mat_best_s_g = _build_matrix(models, datasets, best_data, "singlelr", "gcs")
    mat_best_s_r = _build_matrix(models, datasets, best_data, "singlelr", "rfm")

    (mat_avg_s_g, mat_avg_s_r, mat_best_s_g, mat_best_s_r), models_common, datasets_common = _filter_common(
        [mat_avg_s_g, mat_avg_s_r, mat_best_s_g, mat_best_s_r],
        models,
        datasets,
    )

    out1 = os.path.join(args.out_dir, "singlelr_minus_gcs_avg.png")
    out2 = os.path.join(args.out_dir, "singlelr_minus_rfm_avg.png")
    out3 = os.path.join(args.out_dir, "singlelr_minus_gcs_best.png")
    out4 = os.path.join(args.out_dir, "singlelr_minus_rfm_best.png")
    out_grid = os.path.join(args.out_dir, "singlelr_diff_2x2.png")
    out_grid_pdf = os.path.join(args.out_dir, "singlelr_diff_2x2.pdf")
    out_row_pdf = os.path.join(args.out_dir, "singlelr_diff_1x4.pdf")

    saved = []
    if _plot_heatmap(mat_avg_s_g, models_common, datasets_common, "RAPTOR - GCS (avg acc %)", out1):
        saved.append(out1)
    if _plot_heatmap(mat_avg_s_r, models_common, datasets_common, "RAPTOR - RFM (avg acc %)", out2):
        saved.append(out2)
    if _plot_heatmap(mat_best_s_g, models_common, datasets_common, "RAPTOR - GCS (best acc %)", out3):
        saved.append(out3)
    if _plot_heatmap(mat_best_s_r, models_common, datasets_common, "RAPTOR - RFM (best acc %)", out4):
        saved.append(out4)
    if _plot_heatmap_grid(
        [mat_avg_s_g, mat_avg_s_r, mat_best_s_g, mat_best_s_r],
        [
            "RAPTOR - GCS (avg acc %)",
            "RAPTOR - RFM (avg acc %)",
            "RAPTOR - GCS (best acc %)",
            "RAPTOR - RFM (best acc %)",
        ],
        models_common,
        datasets_common,
        out_grid,
    ):
        saved.append(out_grid)
    if _plot_heatmap_grid(
        [mat_avg_s_g, mat_avg_s_r, mat_best_s_g, mat_best_s_r],
        [
            "RAPTOR - GCS (avg acc %)",
            "RAPTOR - RFM (avg acc %)",
            "RAPTOR - GCS (best acc %)",
            "RAPTOR - RFM (best acc %)",
        ],
        models_common,
        datasets_common,
        out_grid_pdf,
        tight=True,
    ):
        saved.append(out_grid_pdf)
    if _plot_heatmap_row(
        [mat_avg_s_g, mat_avg_s_r, mat_best_s_g, mat_best_s_r],
        [
            "RAPTOR - GCS (avg acc %)",
            "RAPTOR - RFM (avg acc %)",
            "RAPTOR - GCS (best acc %)",
            "RAPTOR - RFM (best acc %)",
        ],
        models_common,
        datasets_common,
        out_row_pdf,
        tight=True,
    ):
        saved.append(out_row_pdf)

    if saved:
        print("Saved:")
        for p in saved:
            print(p)
    else:
        print("No plots saved (missing data).")


if __name__ == "__main__":
    main()
