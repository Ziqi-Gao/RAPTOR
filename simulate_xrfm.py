#!/usr/bin/env python3
"""
simulate_xrfm.py — 使用作者 neural_controllers 仓库中的 RFM probe 实现。

接口仍与 simulate_singleLR.run_single_lr 保持一致，输入 (X_train, y_train, X_test, y_test)，
返回测试集指标，必要时返回验证集指标与最优超参。
"""
from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---- 依赖：neural_controllers 仓库 & xrfm ----
REPO_PROBE_PATH = Path(__file__).resolve().parent / "neural_controllers_repo"
if REPO_PROBE_PATH.exists():
    sys.path.insert(0, str(REPO_PROBE_PATH))

try:
    import torch
    import direction_utils as nc_direction_utils
    from xrfm import RFM
except ImportError as e:  # pragma: no cover - 环境缺失时抛出
    raise ImportError(
        "需要安装 xrfm，并确保 neural_controllers_repo 位于当前目录（或已在 PYTHONPATH 中）。"
    ) from e


def _to_label_tensor(y: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.as_tensor(y, dtype=torch.float32, device=device)
    if t.ndim == 1:
        t = t.view(-1, 1)
    return t


def _normalize_metric_name(name: str) -> str:
    aliases = {
        "accuracy": "acc",
        "acc": "acc",
        "auc": "auc",
        "f1": "f1",
        "mse": "mse",
        "top_agop_vectors_ols_auc": "top_agop_vectors_ols_auc",
    }
    return aliases.get(name.lower(), "acc")


def _train_rfm_like_repo(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    *,
    hyperparams: Dict[str, Any],
    search_space: Dict[str, Any],
    tuning_metric: str,
    device_str: str,
) -> Tuple[RFM, float, Dict[str, Any], torch.Tensor]:
    """
    按作者 neural_controllers 的实现搜索 RFM，基于预测指标选择最优。
    返回 (最佳模型, 验证评分, 最优超参, 概念向量)。
    """
    if not search_space:
        search_space = {
            "regs": [1e-3],
            "bws": [1, 10, 100],
            "center_grads": [True, False],
        }

    maximize_metric = tuning_metric in {"f1", "auc", "acc", "top_agop_vectors_ols_auc"}
    best_score = float("-inf") if maximize_metric else float("inf")
    best_model: Optional[RFM] = None
    best_params: Dict[str, Any] = {}
    best_concept_vector: Optional[torch.Tensor] = None
    caught_errors = []

    # xRFM 内部对 accuracy 使用关键字 "accuracy" 而非 "acc"
    rfm_metric = "accuracy" if tuning_metric == "acc" else tuning_metric

    for reg in search_space["regs"]:
        for bw in search_space["bws"]:
            for center_grads in search_space["center_grads"]:
                try:
                    rfm_params = {
                        "model": {
                            "kernel": "l2_high_dim",
                            "bandwidth": bw,
                            "tuning_metric": rfm_metric,
                        },
                        "fit": {
                            "reg": reg,
                            "iters": int(max(1, hyperparams["rfm_iters"])),
                            "center_grads": center_grads,
                            "early_stop_rfm": True,
                            "get_agop_best_model": True,
                            "top_k": int(max(1, hyperparams["n_components"])),
                        },
                    }
                    model = RFM(**rfm_params["model"], device=device_str)
                    model.fit((train_X, train_y), (val_X, val_y), **rfm_params["fit"])

                    if tuning_metric == "top_agop_vectors_ols_auc":
                        top_k = int(max(1, hyperparams["n_components"]))
                        targets = val_y

                        _, U = torch.lobpcg(model.agop_best_model, k=top_k)
                        top_eigenvectors = U[:, :top_k]
                        projections = val_X @ top_eigenvectors
                        projections = projections.reshape(-1, top_k)

                        XtX = projections.T @ projections
                        Xty = projections.T @ targets
                        betas = torch.linalg.pinv(XtX) @ Xty
                        preds = torch.sigmoid(projections @ betas).reshape(targets.shape)
                        val_score = nc_direction_utils.roc_auc_score(
                            targets.detach().cpu().numpy(), preds.detach().cpu().numpy()
                        )
                    else:
                        pred_proba = model.predict(val_X)
                        metrics = nc_direction_utils.compute_prediction_metrics(
                            pred_proba, val_y
                        )
                        val_score = metrics.get(tuning_metric, metrics.get("acc"))

                    if val_score is None:
                        continue

                    is_better = (val_score > best_score) if maximize_metric else (val_score < best_score)
                    if is_better or best_model is None:
                        best_score = float(val_score)
                        best_model = deepcopy(model)
                        best_params = {
                            "bandwidth": bw,
                            "reg": reg,
                            "center_grads": center_grads,
                            "iters": int(max(1, hyperparams["rfm_iters"])),
                            "n_components": int(max(1, hyperparams["n_components"])),
                            "tuning_metric": tuning_metric,
                        }
                except Exception:
                    import traceback
                    caught_errors.append(traceback.format_exc())
                    continue

    if best_model is None:
        return None, None, {"errors": caught_errors}, None

    try:
        _, U = torch.lobpcg(best_model.agop_best_model, k=1)
        best_concept_vector = U[:, 0]
    except Exception:
        best_concept_vector = None

    return best_model, best_score, best_params, best_concept_vector


def run_xrfm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    solver: str = "lbfgs",        # 占位，为兼容 singleLR 接口
    max_iter: int = 1000,         # 占位，默认会被 rfm_iters 覆盖/截断
    C: float = 1.0,               # 占位，为兼容 singleLR 接口
    penalty: str = "l2",          # 占位，为兼容 singleLR 接口
    n_jobs: Optional[int] = None, # 占位，为兼容 singleLR 接口
    summary_only: bool = True,
    save_path: Optional[str] = None,
    val_size: float = 0.2,
    rfm_iters: Optional[int] = None,
    n_trees: int = 1,             # 保留占位，当前实现未用
    n_tree_iters: int = 0,        # 保留占位，当前实现未用
    split_method: str = "top_vector_agop_on_subset",  # 保留占位，当前实现未用
    max_leaf_size: int = 60_000,  # 保留占位，当前实现未用
    tuning_metric: str = "accuracy",
    random_state: Optional[int] = None,
    n_threads: Optional[int] = None,   # 保留占位，当前实现未用
    use_default_params: bool = False,  # 兼容旧接口；当前忽略，始终按作者搜索空间
    hparam_trials: int = 8,       # 兼容旧接口；当前忽略，使用固定搜索空间
    standardize: bool = True,
    return_details: bool = False,
    split_subset_size: int = 1500,  # 保留占位，当前实现未用
    n_components: int = 1,
    return_concept: bool = False,
    concept_save_path: Optional[str] = None,
) -> Union[float, Tuple[Any, ...]]:
    """
    用 neural_controllers 的 RFM probe 训练一个探针，返回测试集准确率（或指定指标）。

    - solver/penalty/C/n_jobs 等参数仅为接口兼容，不参与 RFM 训练。
    - tuning_metric 支持 acc/auc/f1/top_agop_vectors_ols_auc/mse，默认兼容传入 accuracy->acc。
    - return_details=True 时返回 (test_metric, val_metric, best_hparams)。
    """
    del solver, penalty, n_jobs, C, n_trees, n_tree_iters, split_method, max_leaf_size, n_threads, use_default_params, hparam_trials, split_subset_size  # 占位变量避免未使用告警

    t0 = time.perf_counter()
    metric_key = _normalize_metric_name(tuning_metric)

    rng_seed = 0 if random_state is None else int(random_state)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if val_size is None or val_size <= 0 or len(X_train) < 5:
        X_tr, X_val, y_tr, y_val = X_train, X_test, y_train, y_test
    else:
        strat = y_train if len(np.unique(y_train)) > 1 else None
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            stratify=strat,
            random_state=rng_seed,
        )

    if standardize:
        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_val = scaler.transform(X_val)
        X_test_proc = scaler.transform(X_test)
    else:
        X_test_proc = X_test

    X_tr_t = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    X_te_t = torch.as_tensor(X_test_proc, dtype=torch.float32, device=device)
    y_tr_t = _to_label_tensor(y_tr, device)
    y_val_t = _to_label_tensor(y_val, device)
    y_te_t = _to_label_tensor(y_test, device)

    hyperparams = {
        "rfm_iters": int(rfm_iters if rfm_iters is not None else min(max_iter, 10)),
        "n_components": int(max(1, n_components)),
    }
    search_space = {}  # 保留兼容参数，当前搜索空间由 _train_rfm_like_repo 内部定义

    def _search_on_device(device_label: str):
        return _train_rfm_like_repo(
            X_tr_t.to(device_label),
            y_tr_t.to(device_label),
            X_val_t.to(device_label),
            y_val_t.to(device_label),
            hyperparams=hyperparams,
            search_space=search_space,
            tuning_metric=metric_key,
            device_str=device_label,
        )

    best_model, best_val_score, best_h, concept_vec = _search_on_device(device_str)

    # 若 GPU 搜索失败，自动尝试 CPU；输出首条异常摘要，便于排查
    if best_model is None and device_str == "cuda":
        err_summary = best_h.get("errors", []) if isinstance(best_h, dict) else []
        if err_summary:
            print(
                "[xrfm][warn] GPU 搜索失败，切换 CPU 重试。示例异常: "
                f"{err_summary[0].splitlines()[-1]}",
                file=sys.stderr,
            )
        else:
            print("[xrfm][warn] GPU 搜索失败，切换 CPU 重试。", file=sys.stderr)
        best_model, best_val_score, best_h, concept_vec = _search_on_device("cpu")

    if best_model is None:
        err_summary = best_h.get("errors", []) if isinstance(best_h, dict) else []
        msg = "RFM 搜索失败，未找到可用模型。"
        if err_summary:
            msg += f" 示例异常：{err_summary[0].splitlines()[-1]}"
        raise RuntimeError(msg)

    X_te_for_pred = X_te_t.to(best_model.device) if hasattr(best_model, "device") else X_te_t
    test_preds = best_model.predict(X_te_for_pred)
    test_metrics = nc_direction_utils.compute_prediction_metrics(test_preds, y_te_t.to(best_model.device))
    test_metric = test_metrics.get(metric_key, test_metrics.get("acc"))

    if test_metric is None:
        raise RuntimeError("测试指标计算失败，未找到对应的 metric。")

    if concept_vec is not None:
        if concept_save_path:
            np.save(concept_save_path, concept_vec.detach().cpu().numpy())
        concept_vec_np = concept_vec.detach().cpu().numpy()
    else:
        concept_vec_np = None

    if summary_only:
        elapsed = time.perf_counter() - t0
        print(
            f"[xrfm][summary] metric={metric_key} | test={test_metric:.4f} | "
            f"val={best_val_score:.4f} | hparams={best_h} | time={elapsed:.3f}s"
        )

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(
                {"test_metric": float(test_metric), "val_metric": float(best_val_score), "hparams": best_h},
                f,
                ensure_ascii=False,
                indent=2,
            )

    if return_details:
        if return_concept:
            return float(test_metric), float(best_val_score), best_h, concept_vec_np
        return float(test_metric), float(best_val_score), best_h
    if return_concept:
        return float(test_metric), concept_vec_np
    return float(test_metric)


__all__ = ["run_xrfm"]
