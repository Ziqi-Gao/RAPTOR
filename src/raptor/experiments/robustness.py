"""
Robustness test: drop a fraction of training samples and compute concept vectors
with SingleLR, RFM (xRFM), and GCS. Repeats N runs and reports cosine similarity stats.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from sklearn.model_selection import train_test_split

from raptor.core import (
    build_xy,
    ensure_root,
    maybe_load_embeddings,
    model_tag,
    normalize_dataset,
)
from raptor.experiments.run_gcs import _run_gcs_all_layers
from raptor.experiments.run_raptor import _run_singlelr_all_layers
from raptor.experiments.run_xrfm import _run_xrfm_all_layers


METHOD_LABELS = {
    "singlelr": "SingleLR",
    "xrfm": "RFM",
    "gcs": "GCS",
}


def _parse_methods(methods_arg: str) -> List[str]:
    methods = []
    for item in methods_arg.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key in ("single", "singlelr"):
            key = "singlelr"
        elif key in ("rfm", "xrfm"):
            key = "xrfm"
        elif key == "gcs":
            key = "gcs"
        else:
            raise ValueError(f"unknown method: {item}")
        if key not in methods:
            methods.append(key)
    return methods


def _resolve_layers(layer_arg: str, num_layers: int) -> List[int]:
    if layer_arg is None:
        return list(range(num_layers))
    if isinstance(layer_arg, str):
        la = layer_arg.strip().lower()
        if la in {"all", "*"}:
            return list(range(num_layers))
        if la in {"mid", "middle"}:
            return [num_layers // 2]
        try:
            lay = int(la)
        except ValueError as exc:
            raise ValueError(f"--layer expects int/'mid'/'all', got {layer_arg}") from exc
    else:
        lay = int(layer_arg)
    if lay < 0:
        lay = num_layers + lay
    if lay < 0 or lay >= num_layers:
        raise ValueError(f"layer {lay} out of range (0..{num_layers - 1})")
    return [lay]


def drop_indices(
    base_idx: np.ndarray, frac: float, rng: np.random.RandomState
) -> np.ndarray:
    n = int(len(base_idx))
    if n == 0:
        return base_idx.copy()
    if frac <= 0:
        return base_idx.copy()
    k = int(round(frac * n))
    if k <= 0:
        return base_idx.copy()
    if k >= n:
        return np.array([], dtype=base_idx.dtype)
    keep_n = n - k
    keep = rng.choice(base_idx, size=keep_n, replace=False)
    return np.array(keep, dtype=base_idx.dtype)


def _split_train_val(
    base_idx: np.ndarray, y: np.ndarray, val_size: float, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    if val_size <= 0 or len(base_idx) == 0:
        return base_idx.copy(), np.array([], dtype=base_idx.dtype)
    stratify = y[base_idx]
    if len(np.unique(stratify)) < 2:
        stratify = None
    train_sub_idx, val_idx = train_test_split(
        base_idx,
        test_size=val_size,
        stratify=stratify,
        random_state=rng.randint(0, 1_000_000),
    )
    return (
        np.array(train_sub_idx, dtype=base_idx.dtype),
        np.array(val_idx, dtype=base_idx.dtype),
    )


def _cosine_pair_stats(vectors: List[np.ndarray]) -> Tuple[float, float, int]:
    vecs = []
    for v in vectors:
        v = np.asarray(v, dtype=float).ravel()
        norm = np.linalg.norm(v)
        if norm == 0:
            continue
        vecs.append(v / norm)
    n = len(vecs)
    if n < 2:
        return float("nan"), float("nan"), 0
    sims: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(float(np.dot(vecs[i], vecs[j])))
    sims_arr = np.array(sims, dtype=float)
    return float(sims_arr.mean()), float(sims_arr.var(ddof=0)), int(len(sims_arr))


def _run_singlelr(
    X_layer: np.ndarray,
    y: np.ndarray,
    train_sub_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    max_iter: int,
    seed: int,
) -> Tuple[np.ndarray, float, int, float]:
    t0 = time.perf_counter()
    res = _run_singlelr_all_layers(
        [X_layer],
        y,
        train_sub_idx,
        val_idx,
        test_idx,
        max_iter=max_iter,
        solver="lbfgs",
        penalty="l2",
        seed=seed,
    )
    elapsed = time.perf_counter() - t0
    best_c = float(res.get("best_C", [float("nan")])[0])
    n_iter = int(res.get("n_iter", [0])[0])
    return res["concept"][0], best_c, n_iter, elapsed


def _run_rfm(
    X_layer: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    val_size: float,
    seed: int,
    rfm_iters: int,
    n_components: int,
) -> np.ndarray:
    res, _ = _run_xrfm_all_layers(
        [X_layer],
        y,
        train_idx,
        test_idx,
        val_size=val_size,
        seed=seed,
        rfm_iters=rfm_iters,
        n_components=n_components,
    )
    return res["concept"][0]


def _run_gcs(
    X_layer: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    seed: int,
    gcs_n_iter: int,
    gcs_train_frac: float,
    gcs_val_frac: float,
    gcs_max_iter: int,
    gcs_early_loops: int,
    gcs_val_thresh: float,
    gcs_standardize: bool,
    gcs_bootstrap: bool,
    gcs_stratified: bool,
    gcs_sample_n: int,
) -> np.ndarray:
    res = _run_gcs_all_layers(
        [X_layer],
        y,
        train_idx,
        test_idx,
        seed=seed,
        gcs_n_iter=gcs_n_iter,
        gcs_train_frac=gcs_train_frac,
        gcs_val_frac=gcs_val_frac,
        gcs_max_iter=gcs_max_iter,
        gcs_early_loops=gcs_early_loops,
        gcs_val_thresh=gcs_val_thresh,
        gcs_standardize=gcs_standardize,
        gcs_bootstrap=gcs_bootstrap,
        gcs_stratified=gcs_stratified,
        gcs_sample_n=gcs_sample_n,
    )
    return res["concept"][0]


def main() -> None:
    ensure_root()
    ap = argparse.ArgumentParser(description="Occlusion robustness on concept vectors.")
    ap.add_argument("--emb_path", type=str, default=None, help="Path to *_embeddings.npz")
    ap.add_argument("--emb_dir", type=str, default="embeddings_all")
    ap.add_argument("--model", type=str, default="google/gemma-7b-it")
    ap.add_argument("--dataset", type=str, default="STSA")
    ap.add_argument("--layer", type=str, default="all")
    ap.add_argument("--methods", type=str, default="singlelr,xrfm,gcs")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--occlude_frac", type=float, default=0.1)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None, help="Output JSON path (default: exp_results/...)")

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

    methods = _parse_methods(args.methods)

    if args.emb_path:
        emb_path = Path(args.emb_path)
    else:
        dataset = normalize_dataset(args.dataset)
        mtag = model_tag(args.model)
        emb_path = Path(args.emb_dir) / f"{mtag}_{dataset}_embeddings.npz"

    if not emb_path.exists():
        raise FileNotFoundError(f"missing embeddings: {emb_path}")

    if args.out:
        out_path = Path(args.out)
    else:
        stem = emb_path.stem.replace("_embeddings", "")
        out_path = Path("exp_results") / f"occlusion_robustness_{stem}.json"

    X_pos, X_neg = maybe_load_embeddings(str(emb_path))
    if not X_pos or not X_neg:
        raise ValueError(f"empty embeddings: {emb_path}")
    X_layers, y = build_xy(X_pos, X_neg)

    layers_to_run = _resolve_layers(args.layer, len(X_layers))

    all_idx = np.arange(len(y))
    train_pool_idx, test_idx = train_test_split(
        all_idx,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )

    all_layers: List[Dict] = []
    for layer in layers_to_run:
        X_layer = X_layers[layer]
        vecs_by_method: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
        kept_counts: Dict[str, List[int]] = {m: [] for m in methods}
        drop_counts: Dict[str, List[int]] = {m: [] for m in methods}

        for r in range(args.runs):
            rng = np.random.RandomState(args.seed + layer * 1000 + r)
            seed = args.seed + r
            train_idx_run = drop_indices(train_pool_idx, args.occlude_frac, rng)
            train_sub_idx_run, val_idx_run = _split_train_val(
                train_idx_run, y, args.val_size, rng
            )
            drop_count = int(len(train_pool_idx) - len(train_idx_run))

            if "singlelr" in methods:
                vec, best_c, n_iter, elapsed = _run_singlelr(
                    X_layer,
                    y,
                    train_sub_idx_run,
                    val_idx_run,
                    test_idx,
                    max_iter=args.singlelr_max_iter,
                    seed=seed,
                )
                vecs_by_method["singlelr"].append(vec)
                print(
                    "[singlelr] "
                    f"layer={layer} run={r} time_sec={elapsed:.3f} "
                    f"n_iter={n_iter} best_C={best_c:.6g}"
                )
                kept_counts["singlelr"].append(int(len(train_sub_idx_run)))
                drop_counts["singlelr"].append(drop_count)

            if "xrfm" in methods:
                vecs_by_method["xrfm"].append(
                    _run_rfm(
                        X_layer,
                        y,
                        train_idx_run,
                        test_idx,
                        val_size=args.val_size,
                        seed=seed,
                        rfm_iters=args.rfm_iters,
                        n_components=args.rfm_n_components,
                    )
                )
                kept_counts["xrfm"].append(int(len(train_idx_run)))
                drop_counts["xrfm"].append(drop_count)

            if "gcs" in methods:
                vecs_by_method["gcs"].append(
                    _run_gcs(
                        X_layer,
                        y,
                        train_idx_run,
                        test_idx,
                        seed=seed,
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
                )
                kept_counts["gcs"].append(int(len(train_idx_run)))
                drop_counts["gcs"].append(drop_count)

        method_entries: Dict[str, Dict] = {}
        for method in methods:
            vecs = vecs_by_method[method]
            mean_cos, var_cos, num_pairs = _cosine_pair_stats(vecs)
            method_entries[METHOD_LABELS[method]] = {
                "num_vectors": len(vecs),
                "vector_dim": int(len(vecs[0])) if vecs else 0,
                "vectors": [v.tolist() for v in vecs],
                "train_kept": kept_counts[method],
                "train_dropped": drop_counts[method],
                "cosine_mean": mean_cos,
                "cosine_var": var_cos,
                "num_pairs": num_pairs,
            }

        all_layers.append(
            {
                "layer": layer,
                "methods": method_entries,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "emb_path": str(emb_path),
        "runs": args.runs,
        "occlude_frac": args.occlude_frac,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "methods": [METHOD_LABELS[m] for m in methods],
        "layers": all_layers,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
