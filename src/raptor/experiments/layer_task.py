import argparse
import os
import time

import numpy as np

from raptor.core import (
    build_xy,
    ensure_root,
    load_or_create_splits,
    maybe_load_embeddings,
    mkdir,
    model_tag,
    normalize_dataset,
    now,
    save_json,
)
from raptor.experiments.run_gcs import _run_gcs_all_layers
from raptor.experiments.run_raptor import _run_singlelr_all_layers
from raptor.experiments.run_xrfm import _run_xrfm_all_layers


def main() -> None:
    ensure_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--emb_dir", type=str, default="./embeddings_all")
    ap.add_argument("--results_dir", type=str, default="./exp_results")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", type=str, default="singlelr,xrfm,gcs")
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--singlelr_max_iter", type=int, default=1000)
    ap.add_argument("--rfm_iters", type=int, default=10)
    ap.add_argument("--rfm_n_components", type=int, default=1)

    ap.add_argument("--gcs_n_iter", type=int, default=1000)
    ap.add_argument("--gcs_train_frac", type=float, default=0.1)
    ap.add_argument("--gcs_val_frac", type=float, default=0.3)
    ap.add_argument("--gcs_max_iter", type=int, default=100)
    ap.add_argument("--gcs_early_loops", type=int, default=10)
    ap.add_argument("--gcs_val_thresh", type=float, default=0.90)
    ap.add_argument("--gcs_sample_n", type=int, default=1000)
    ap.add_argument("--gcs_no_standardize", action="store_true")
    ap.add_argument("--gcs_bootstrap", action="store_true")
    ap.add_argument("--gcs_no_stratified", action="store_true")
    args = ap.parse_args()

    dataset = normalize_dataset(args.dataset)
    mtag = model_tag(args.model)
    emb_path = os.path.join(args.emb_dir, f"{mtag}_{dataset}_embeddings.npz")
    if not os.path.isfile(emb_path):
        raise FileNotFoundError(f"missing embeddings: {emb_path}")

    X_pos, X_neg = maybe_load_embeddings(emb_path)
    if not X_pos or not X_neg:
        raise ValueError(f"empty embeddings: {emb_path}")
    if args.layer < 0 or args.layer >= len(X_pos):
        raise ValueError(f"layer {args.layer} out of range (0..{len(X_pos)-1})")

    X_layers, y = build_xy([X_pos[args.layer]], [X_neg[args.layer]])
    if len(np.unique(y)) < 2:
        raise ValueError("only one class in labels")

    out_dir = os.path.join(args.results_dir, mtag, dataset, "layers", f"layer_{args.layer}")
    mkdir(out_dir)

    split_path = os.path.join(args.results_dir, mtag, dataset, "splits.npz")
    split_lock = f"{split_path}.lock"
    while True:
        try:
            os.mkdir(split_lock)
            break
        except FileExistsError:
            time.sleep(2)
    try:
        train_idx, train_sub_idx, val_idx, test_idx = load_or_create_splits(
            split_path, y, args.test_size, args.val_size, args.seed
        )
    finally:
        try:
            os.rmdir(split_lock)
        except OSError:
            pass

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]

    def _save_npz_atomic(path, **data):
        tmp = f"{path}.tmp.{os.getpid()}.npz"
        np.savez_compressed(tmp, **data)
        os.replace(tmp, path)

    def _ready(path):
        return os.path.isfile(path) and os.path.getsize(path) > 0
    timings = {}

    if "singlelr" in methods:
        single_path = os.path.join(out_dir, "singlelr_results.npz")
        if args.force or not _ready(single_path):
            t0 = time.perf_counter()
            res = _run_singlelr_all_layers(
                X_layers,
                y,
                train_sub_idx,
                val_idx,
                test_idx,
                max_iter=args.singlelr_max_iter,
                solver="lbfgs",
                penalty="l2",
                seed=args.seed,
            )
            _save_npz_atomic(single_path, **res)
            timings["singlelr_sec"] = time.perf_counter() - t0
            print(f"[time] singlelr_sec={timings['singlelr_sec']:.3f}")

    if "xrfm" in methods or "rfm" in methods:
        rfm_path = os.path.join(out_dir, "rfm_results.npz")
        if args.force or not _ready(rfm_path):
            t0 = time.perf_counter()
            res, hparams = _run_xrfm_all_layers(
                X_layers,
                y,
                train_idx,
                test_idx,
                val_size=args.val_size,
                seed=args.seed,
                rfm_iters=args.rfm_iters,
                n_components=args.rfm_n_components,
            )
            _save_npz_atomic(rfm_path, **res)
            save_json(os.path.join(out_dir, "rfm_hparams.json"), hparams)
            timings["xrfm_sec"] = time.perf_counter() - t0
            print(f"[time] xrfm_sec={timings['xrfm_sec']:.3f}")

    if "gcs" in methods:
        gcs_path = os.path.join(out_dir, "gcs_results.npz")
        if args.force or not _ready(gcs_path):
            t0 = time.perf_counter()
            res = _run_gcs_all_layers(
                X_layers,
                y,
                train_idx,
                test_idx,
                seed=args.seed,
                gcs_n_iter=args.gcs_n_iter,
                gcs_train_frac=args.gcs_train_frac,
                gcs_val_frac=args.gcs_val_frac,
                gcs_max_iter=args.gcs_max_iter,
                gcs_early_loops=args.gcs_early_loops,
                gcs_val_thresh=args.gcs_val_thresh,
                gcs_standardize=not args.gcs_no_standardize,
                gcs_bootstrap=args.gcs_bootstrap,
                gcs_stratified=not args.gcs_no_stratified,
                gcs_sample_n=args.gcs_sample_n,
            )
            _save_npz_atomic(gcs_path, **res)
            timings["gcs_sec"] = time.perf_counter() - t0
            print(f"[time] gcs_sec={timings['gcs_sec']:.3f}")

    save_json(
        os.path.join(out_dir, "task_meta.json"),
        {
            "model": args.model,
            "dataset": dataset,
            "layer": args.layer,
            "embeddings": emb_path,
            "splits": split_path,
            "methods": methods,
            "timings_sec": timings,
            "timestamp": now(),
        },
    )


if __name__ == "__main__":
    main()
