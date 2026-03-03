#!/usr/bin/env python3
"""
Tune logistic regularization C on real embedding datasets and compare
singleLR / bagging / ensemble / xRFM accuracy.

Prerequisite: generate *_embeddings.npz files with save_embeddings.py.
"""
import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from simulate_bagging import run_bagging_simple
from simulate_ensemble import run_ensemble_bagging
from simulate_xrfm import run_xrfm

# Dense geometric grid for single LR tuning.
SINGLE_FINE_GRID = np.logspace(np.log10(1e-4), np.log10(100.0), num=100, dtype=float)


def _parse_floats(val: str) -> List[float]:
    return [float(x) for x in val.split(",") if x.strip()]


def _load_embeddings(npz_path: str, layer: int) -> Tuple[np.ndarray, np.ndarray, Dict, List[int]]:
    data = np.load(npz_path, allow_pickle=False)
    layer_keys = [k for k in data.keys() if k.startswith("X_pos_")]
    if not layer_keys:
        raise ValueError("NPZ 中未找到 X_pos_* 键，确认是否由 save_embeddings.py 生成")
    avail_layers = sorted(int(k.split("_")[-1]) for k in layer_keys)
    L = max(avail_layers) + 1
    lay = layer
    if lay < 0:
        lay = L + lay if abs(lay) < L else avail_layers[-1]
    if lay not in avail_layers:
        raise ValueError(f"指定的层 {layer} 不在可用层 {avail_layers} 中")

    X_pos = data[f"X_pos_{lay}"]
    X_neg = data[f"X_neg_{lay}"]
    y_pos = np.ones(len(X_pos), dtype=int)
    y_neg = np.zeros(len(X_neg), dtype=int)
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])

    meta = {
        "layer": lay,
        "model": str(data["meta_model"][0]) if "meta_model" in data else "",
        "dataset": str(data["meta_dataset"][0]) if "meta_dataset" in data else "",
        "dim": int(data["meta_dim"][0]) if "meta_dim" in data else int(X.shape[1]),
    }
    return X, y, meta, avail_layers


def _tune_single(
    Xtr,
    ytr,
    Xval,
    yval,
    C_grid,
    refine_mults,
    auto_refine,
    refine_rounds,
    *,
    return_stats: bool = False,
):
    # Ignore external grid/refine; sweep a dense geometric grid.
    best_C, best_val = None, -1.0
    max_iter_hits = 0
    t0 = time.perf_counter()

    # Warm-start across ascending C values to speed up scans.
    grid = np.array(SINGLE_FINE_GRID, dtype=float)
    grid.sort()
    clf = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        max_iter=1000,
        warm_start=True,
    )
    for Cval in grid:
        clf.set_params(C=float(Cval))
        clf.fit(Xtr, ytr)
        nit = clf.n_iter_
        if isinstance(nit, (list, np.ndarray)):
            nit = max(nit)
        if int(nit) >= int(clf.max_iter):
            max_iter_hits += 1
        val_acc = float(accuracy_score(yval, clf.predict(Xval)))
        if val_acc > best_val:
            best_val, best_C = val_acc, Cval

    tune_time = time.perf_counter() - t0
    best_C = best_C if best_C is not None else 1.0
    if return_stats:
        return best_C, best_val, {"tune_time": tune_time, "max_iter_hits": max_iter_hits}
    return best_C, best_val


def _tune_bagging(Xtr, ytr, Xval, yval, C_grid, refine_mults, auto_refine, refine_rounds, alpha, M, seed):
    best_C, best_val = None, -1.0
    for Cval in C_grid:
        val_ret = run_bagging_simple(
            Xtr, ytr, Xval, yval, alpha=alpha, M=M,
            solver="lbfgs", penalty="l2", C=Cval, max_iter=1000,
            random_state=seed, n_jobs=None,
            summary_only=False, return_details=True, save_path=None
        )
        val_acc = val_ret[0] if isinstance(val_ret, tuple) else float(val_ret)
        if val_acc > best_val:
            best_val, best_C = val_acc, Cval
    if auto_refine and best_C is not None and refine_rounds > 0:
        tried = set(float(c) for c in C_grid)
        for _ in range(refine_rounds):
            improved = False
            for m in refine_mults:
                C2 = float(best_C) * float(m)
                if C2 <= 0:
                    continue
                if any(abs(C2 - c) / c < 1e-12 for c in tried if c > 0):
                    continue
                val_ret = run_bagging_simple(
                    Xtr, ytr, Xval, yval, alpha=alpha, M=M,
                    solver="lbfgs", penalty="l2", C=C2, max_iter=1000,
                    random_state=seed, n_jobs=None,
                    summary_only=False, return_details=True, save_path=None
                )
                val_acc = val_ret[0] if isinstance(val_ret, tuple) else float(val_ret)
                tried.add(C2)
                if val_acc > best_val:
                    best_val, best_C = val_acc, C2
                    improved = True
            if not improved:
                break
    return best_C if best_C is not None else 1.0, best_val


def _tune_ensemble(Xtr, ytr, Xval, yval, C_grid, refine_mults, auto_refine, refine_rounds, alpha, M, agg, seed):
    best_C, best_val = None, -1.0
    for Cval in C_grid:
        val_ret = run_ensemble_bagging(
            Xtr, ytr, Xval, yval, alpha=alpha, M=M, agg=agg,
            solver="lbfgs", penalty="l2", C=Cval, max_iter=1000,
            random_state=seed, n_jobs=None,
            summary_only=False, return_details=True, save_path=None
        )
        val_acc = val_ret[0] if isinstance(val_ret, tuple) else float(val_ret)
        if val_acc > best_val:
            best_val, best_C = val_acc, Cval
    if auto_refine and best_C is not None and refine_rounds > 0:
        tried = set(float(c) for c in C_grid)
        for _ in range(refine_rounds):
            improved = False
            for m in refine_mults:
                C2 = float(best_C) * float(m)
                if C2 <= 0:
                    continue
                if any(abs(C2 - c) / c < 1e-12 for c in tried if c > 0):
                    continue
                val_ret = run_ensemble_bagging(
                    Xtr, ytr, Xval, yval, alpha=alpha, M=M, agg=agg,
                    solver="lbfgs", penalty="l2", C=C2, max_iter=1000,
                    random_state=seed, n_jobs=None,
                    summary_only=False, return_details=True, save_path=None
                )
                val_acc = val_ret[0] if isinstance(val_ret, tuple) else float(val_ret)
                tried.add(C2)
                if val_acc > best_val:
                    best_val, best_C = val_acc, C2
                    improved = True
            if not improved:
                break
    return best_C if best_C is not None else 1.0, best_val


def main():
    ap = argparse.ArgumentParser(description="在真实数据集 embedding 上调 C，比较 singleLR / bagging / ensemble 准确率")
    ap.add_argument("--emb_path", type=str, required=True, help="save_embeddings.py 生成的 *_embeddings.npz 路径")
    ap.add_argument("--layer", type=int, default=-1, help="使用哪一层的 embedding（0-based，-1 表示最后一层）")
    ap.add_argument("--methods", type=str, default="single,xrfm,ensemble", help="逗号分隔，可选 single,xrfm,bagging,ensemble")
    ap.add_argument("--test_size", type=float, default=0.2, help="测试集占比")
    ap.add_argument("--val_size", type=float, default=0.2, help="训练集内用于挑选 C 的验证集占比")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（切分 & bagging/ensemble 采样）")
    ap.add_argument("--alpha", type=float, default=0.6, help="bagging/ensemble 采样占比")
    ap.add_argument("--M", type=int, default=20, help="bagging/ensemble 基模型数")
    ap.add_argument("--agg", type=str, default="mean_std", choices=["matrix", "mean_raw", "mean_std"], help="ensemble 元分类器输入方式")
    ap.add_argument("--tune_C", dest="tune_C", action="store_true", default=True, help="是否在验证集调 C，再全量重训")
    ap.add_argument("--no_tune_C", dest="tune_C", action="store_false", help="跳过调参，直接用 C=1.0")
    ap.add_argument("--C_grid", type=str, default="0.01,0.1,1.0,10.0,100.0", help="粗扫 C 网格，逗号分隔")
    ap.add_argument("--auto_refine_C", action="store_true", help="对粗扫最优 C 再做倍率细扫")
    ap.add_argument("--C_refine_multipliers", type=str, default="0.1,0.3,0.5,1.0,2.0,3.0", help="细扫倍率列表")
    ap.add_argument("--refine_rounds", type=int, default=1, help="细扫重复轮数（最佳 C 每轮再乘给定倍率），0 表示不重复")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    C_grid = _parse_floats(args.C_grid)
    refine_mults = _parse_floats(args.C_refine_multipliers)

    X, y, meta, avail_layers = _load_embeddings(args.emb_path, args.layer)
    print(f"[info] load embeddings: {args.emb_path}")
    print(f"[info] dataset={meta.get('dataset','')} model={meta.get('model','')} dim={meta.get('dim')} layer={meta.get('layer')} (available {avail_layers})")
    print(f"[info] total samples={len(y)} | pos={int(y.sum())} neg={int((y==0).sum())}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    do_tune = bool(args.tune_C and len(np.unique(y_tr)) > 1 and args.val_size > 0)
    if do_tune:
        X_tr_tr, X_val, y_tr_tr, y_val = train_test_split(
            X_tr, y_tr, test_size=args.val_size, stratify=y_tr, random_state=args.seed + 1
        )
    else:
        X_tr_tr, y_tr_tr = X_tr, y_tr
        X_val, y_val = X_te, y_te  # Placeholder.

    results = []

    if "single" in methods:
        if do_tune:
            best_C, best_val = _tune_single(X_tr_tr, y_tr_tr, X_val, y_val, C_grid, refine_mults, args.auto_refine_C, args.refine_rounds)
        else:
            best_C, best_val = 1.0, -1.0
        acc = run_single_lr(
            X_tr, y_tr, X_te, y_te, solver="lbfgs", penalty="l2",
            C=float(best_C), max_iter=1000, n_jobs=None, summary_only=True, save_path=None
        )
        results.append({"method": "single", "acc": acc, "best_C": best_C, "val_acc": best_val, "extra": ""})

    if "xrfm" in methods:
        # xRFM uses its own internal search; C is a compatibility placeholder.
        acc, best_val, best_h = run_xrfm(
            X_tr, y_tr, X_te, y_te,
            C=1.0, val_size=args.val_size,
            random_state=args.seed,
            summary_only=True, save_path=None,
            use_default_params=False,
            return_details=True,
        )
        results.append({"method": "xrfm", "acc": acc, "best_C": "paper_search", "val_acc": best_val, "extra": f"hparams={best_h}"})

    if "bagging" in methods:
        # Keep C fixed at 1.0 for this baseline.
        best_C, best_val = 1.0, -1.0
        bag_ret = run_bagging_simple(
            X_tr, y_tr, X_te, y_te, alpha=args.alpha, M=args.M,
            solver="lbfgs", penalty="l2", C=float(best_C), max_iter=1000,
            random_state=args.seed, n_jobs=None,
            summary_only=True, return_details=True, save_path=None
        )
        if isinstance(bag_ret, tuple):
            acc, n_nc = bag_ret
        else:
            acc, n_nc = float(bag_ret), ""
        results.append({"method": "bagging", "acc": acc, "best_C": best_C, "val_acc": best_val, "extra": f"non_converged={n_nc}"})

    if "ensemble" in methods:
        # Keep C fixed at 1.0 for this baseline.
        best_C, best_val = 1.0, -1.0
        ens_ret = run_ensemble_bagging(
            X_tr, y_tr, X_te, y_te, alpha=args.alpha, M=args.M, agg=args.agg,
            solver="lbfgs", penalty="l2", C=float(best_C), max_iter=1000,
            random_state=args.seed, n_jobs=None,
            summary_only=True, return_details=True, save_path=None
        )
        if isinstance(ens_ret, tuple):
            acc, n_nc = ens_ret
        else:
            acc, n_nc = float(ens_ret), ""
        results.append({"method": "ensemble", "acc": acc, "best_C": best_C, "val_acc": best_val, "extra": f"non_converged={n_nc}"})

    print("\n===== accuracy (test) =====")
    for r in results:
        val_txt = f"val_acc={r['val_acc']:.4f}" if r["val_acc"] >= 0 else "val_acc=n/a"
        extra = f" | {r['extra']}" if r["extra"] else ""
        print(f"{r['method']:>9} | acc={r['acc']:.4f} | best_C={r['best_C']} | {val_txt}{extra}")


if __name__ == "__main__":
    main()
