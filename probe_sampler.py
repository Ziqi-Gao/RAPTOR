# # probe_sampler.py
# from __future__ import annotations
# from dataclasses import dataclass
# import os
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from tqdm import tqdm

# @dataclass
# class ProbingConfig:
#     n_iter: int = 1000           # 迭代（采样）次数
#     train_frac: float = 0.10     # 每次从 pool 里抽取的训练比例
#     max_iter_lr: int = 100       # LR 最大迭代
#     early_loops: int = 10        # 早停重试次数
#     val_thresh: float = 0.90     # 提前停止阈值（在子集上的 val，而不是 hold-out）
#     standardize: bool = True     # 是否标准化
#     random_state: int | None = None  # 随机种子(None=不固定)

# def run_probing(
#     X_layers, y_pool, cfg: ProbingConfig,
#     *,
#     X_layers_holdout, y_holdout
# ):
#     """
#     在给定的训练池 (X_layers, y_pool) 上做 cfg.n_iter 次 probing。
#     返回:
#       - W: (n_iter, L, d)
#       - B: (n_iter, L)
#       - A: (n_iter, L)  每次模型在最终 hold-out 上的 acc
#       - observed_layers: list[(n_iter, d+1)]  每层的 [w|b]
#       - val_splits: List[np.ndarray]  长度 n_iter；每个是该次 LR 的 “valid/OOF 索引”（相对于 pool）
#     """
#     if X_layers_holdout is None or y_holdout is None:
#         raise ValueError("必须提供 X_layers_holdout 与 y_holdout，A 需要在最终 test 上评估。")

#     rng = np.random.default_rng(cfg.random_state)

#     L = len(X_layers)
#     n_pool = len(y_pool)
#     d = X_layers[0].shape[1]
#     n_train = max(1, int(round(n_pool * cfg.train_frac)))

#     W = np.zeros((cfg.n_iter, L, d), dtype=np.float32)
#     B = np.zeros((cfg.n_iter, L),     dtype=np.float32)
#     A = np.zeros((cfg.n_iter, L),     dtype=np.float32)  # hold-out acc
#     val_splits: list[np.ndarray] = []                   # 每次的 valid 索引（对所有层一致）

#     for it in tqdm(range(cfg.n_iter), desc="Sampling & Eval (probe_sampler)"):
#         # 1) 从训练池抽取一个子训练集（有放回）
#         idx_sub = rng.choice(n_pool, size=n_train, replace=True)

#         # 2) 在索引空间切分 train/valid（valid 仅用于早停/记录 OOF；对所有层共用）
#         idx_rel = np.arange(len(idx_sub))
#         idx_tr_rel, idx_val_rel = train_test_split(idx_rel, test_size=0.3, shuffle=True)
#         tr_idx_pool = idx_sub[idx_tr_rel]   # 相对于 pool 的全局索引
#         val_idx_pool = idx_sub[idx_val_rel] # 相对于 pool 的全局索引
#         val_splits.append(val_idx_pool.copy())  # 保存本次的 valid/OOF 索引

#         for l in range(L):
#             X_tr = X_layers[l][tr_idx_pool]
#             y_tr = y_pool[tr_idx_pool]

#             # 构造“验证集”仅用于早停；最终 A 用 hold-out
#             X_val = X_layers[l][val_idx_pool]
#             y_val = y_pool[val_idx_pool]

#             # 3) 标准化（仅用本次的训练子集拟合），并用同一 scaler 变换 hold-out/val
#             if cfg.standardize:
#                 scaler = StandardScaler().fit(X_tr)
#                 X_tr   = scaler.transform(X_tr)
#                 X_val  = scaler.transform(X_val)
#                 X_hold = scaler.transform(X_layers_holdout[l])
#             else:
#                 X_hold = X_layers_holdout[l]

#             best_coef = None
#             best_intercept = None
#             best_hold_acc = 0.0

#             # 4) 早停循环：val 收敛或超过阈值即停止
#             for _ in range(cfg.early_loops):
#                 clf = LogisticRegression(
#                     penalty="l2", solver="lbfgs", fit_intercept=True, max_iter=cfg.max_iter_lr
#                 )
#                 clf.fit(X_tr, y_tr)

#                 nit = clf.n_iter_
#                 if isinstance(nit, (list, np.ndarray)):
#                     nit = max(nit)
#                 did_conv = (nit < cfg.max_iter_lr)

#                 val_acc  = accuracy_score(y_val,  clf.predict(X_val))
#                 hold_acc = accuracy_score(y_holdout, clf.predict(X_hold))

#                 # 记录当前（即使没到 early stop 也保留最新一次）
#                 best_coef      = clf.coef_.ravel().astype(np.float32)
#                 best_intercept = float(clf.intercept_[0])
#                 best_hold_acc  = float(hold_acc)

#                 if did_conv or val_acc > cfg.val_thresh:
#                     break

#             # 5) 写入结果（W/B 为本次最终模型；A 为该模型在 hold-out 上的 acc）
#             W[it, l, :] = best_coef
#             B[it, l]    = best_intercept
#             A[it, l]    = best_hold_acc

#     observed_layers = build_observed_layers(W, B)
#     return W, B, A, observed_layers, val_splits

# def build_observed_layers(W: np.ndarray, B: np.ndarray):
#     """将 W,B 组合成 list[(n_iter, d+1)] 的 [w|b]"""
#     n_iter, L, d = W.shape
#     obs = []
#     for l in range(L):
#         w_mat = W[:, l, :]               # (n_iter, d)
#         b_vec = B[:, l].reshape(-1, 1)   # (n_iter, 1)
#         obs.append(np.hstack([w_mat, b_vec]))  # (n_iter, d+1)
#     return obs

# def save_probing_npz(save_dir: str, base_name: str, W: np.ndarray, B: np.ndarray, A: np.ndarray):
#     """保存 w/b/acc（A 为 hold-out acc）"""
#     os.makedirs(save_dir, exist_ok=True)
#     np.savez(os.path.join(save_dir, f"{base_name}_w.npz"),   W=W)
#     np.savez(os.path.join(save_dir, f"{base_name}_b.npz"),   B=B)
#     np.savez(os.path.join(save_dir, f"{base_name}_acc.npz"), Acc=A)

# def save_val_splits(save_dir: str, base_name: str, val_splits: list[np.ndarray]):
#     """
#     保存每次 LR 的 valid/OOF 索引（相对于 pool 的全局索引）。
#     采用 dtype=object 的数组存储，便于每次长度不同。
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     obj = np.array(val_splits, dtype=object)
#     np.savez(os.path.join(save_dir, f"{base_name}_val_splits.npz"), val_splits=obj)
# probe_sampler.py
from __future__ import annotations
from dataclasses import dataclass
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

@dataclass
class ProbingConfig:
    n_iter: int = 300           # 迭代（采样）次数 = 基模型总数
    train_frac: float = 0.70      # 每次从 pool 里抽取的训练比例（子样本占比）
    val_frac: float = 0.30        # 子样本内部的验证比例（用于早停）
    max_iter_lr: int = 100        # LR 最大迭代
    early_loops: int = 10         # 早停重试次数
    val_thresh: float = 0.90      # 提前停止阈值（在子集上的 val）
    standardize: bool = True      # 是否对特征做 z-score（仅用本次训练子集拟合）
    bootstrap: bool = False       # True=有放回抽样；False=无放回子采样
    stratified: bool = True       # 是否分层抽样（建议 True）
    random_state: int | None = None  # 随机种子(None=不固定)

def run_probing(
    X_layers, y_pool, cfg: ProbingConfig,
    *, X_layers_holdout, y_holdout
):
    """
    在训练池 (X_layers, y_pool) 上做 cfg.n_iter 次 probing，得到每次的 LR 基模型。
    返回:
      - W: (n_iter, L, d)        # 已“折回原空间”的权重 w_raw
      - B: (n_iter, L)           # 已“折回原空间”的截距 b_raw
      - A: (n_iter, L)           # 该次模型在最终 hold-out 上的 acc
      - observed_layers: list[(n_iter, d+1)]   # 每层拼接的 [w_raw|b_raw]
      - val_splits: List[np.ndarray]           # ★ 每次的 “OOB 索引”（相对于 pool）；供 bagging 作为严格 OOF
    说明：
      * 若 standardize=True，则我们把每次的 (w, b) 从 z-score 空间折回到原空间：
            w_raw = w / sigma
            b_raw = b - <mu/sigma, w>
        因为 StandardScaler 定义 z=(x-u)/s，所以 (x·w_raw + b_raw) 与 ((x-u)/s·w + b) 等价。:contentReference[oaicite:1]{index=1}
    """
    if X_layers_holdout is None or y_holdout is None:
        raise ValueError("必须提供 X_layers_holdout 与 y_holdout，A 需要在最终 test 上评估。")

    rng = np.random.default_rng(cfg.random_state)

    L = len(X_layers)
    n_pool = len(y_pool)
    d = X_layers[0].shape[1]
    n_train = max(1, int(round(n_pool * cfg.train_frac)))

    W = np.zeros((cfg.n_iter, L, d), dtype=np.float32)
    B = np.zeros((cfg.n_iter, L),     dtype=np.float32)
    A = np.zeros((cfg.n_iter, L),     dtype=np.float32)  # hold-out acc
    val_splits: list[np.ndarray] = []                    # 此处存 "OOB 索引"

    # 预备分层索引
    classes, counts = np.unique(y_pool, return_counts=True)
    cls_to_idx = {c: np.flatnonzero(y_pool == c) for c in classes}

    for it in tqdm(range(cfg.n_iter), desc="Sampling & Eval (probe_sampler)"):
        # 1) 从训练池分层抽取一个子训练“候选集”（有/无放回），规模约 n_train
        if cfg.stratified:
            take_list = []
            for c, cnt in zip(classes, counts):
                k = max(1, int(round(cfg.train_frac * cnt)))
                src = cls_to_idx[c]
                if cfg.bootstrap:
                    choose = rng.choice(src, size=k, replace=True)
                else:
                    k = min(k, len(src))
                    choose = rng.choice(src, size=k, replace=False)
                take_list.append(choose)
            idx_sub = np.concatenate(take_list, axis=0)
        else:
            replace = True if cfg.bootstrap else False
            idx_sub = rng.choice(n_pool, size=n_train, replace=replace)

        # 2) 计算本次的 in-bag / OOB（相对于 pool 的全局索引）
        inbag_unique = np.unique(idx_sub)
        all_idx = np.arange(n_pool, dtype=int)
        oob_idx_pool = np.setdiff1d(all_idx, inbag_unique, assume_unique=True)
        val_splits.append(oob_idx_pool.copy())   # ★ 记录 OOB 供 bagging 使用（严格 OOF）

        # 3) 子集内部再划分 train/val（仅用于早停/诊断，不参与 bagging）
        idx_rel = np.arange(len(idx_sub))
        idx_tr_rel, idx_val_rel = train_test_split(
            idx_rel, test_size=cfg.val_frac, shuffle=True
        )
        tr_idx_pool  = idx_sub[idx_tr_rel]   # 相对于 pool
        val_idx_pool = idx_sub[idx_val_rel]  # 相对于 pool

        for l in range(L):
            X_tr_raw  = X_layers[l][tr_idx_pool]
            y_tr      = y_pool[tr_idx_pool]
            X_val_raw = X_layers[l][val_idx_pool]
            y_val     = y_pool[val_idx_pool]
            X_hold_raw= X_layers_holdout[l]

            # 4) 可选：标准化（仅用“本次训练子集”拟合），随后把系数折回原空间保存
            if cfg.standardize:
                scaler = StandardScaler().fit(X_tr_raw)
                X_tr   = scaler.transform(X_tr_raw)
                X_val  = scaler.transform(X_val_raw)
                X_hold = scaler.transform(X_hold_raw)
                mu = scaler.mean_
                sigma = scaler.scale_
                sigma = np.where((sigma == 0) | ~np.isfinite(sigma), 1.0, sigma)
            else:
                X_tr, X_val, X_hold = X_tr_raw, X_val_raw, X_hold_raw
                mu = None
                sigma = None

            best_w_raw = None
            best_b_raw = None
            best_hold_acc = 0.0

            # 5) 早停循环
            for _ in range(cfg.early_loops):
                clf = LogisticRegression(
                    penalty="l2", solver="lbfgs", fit_intercept=True, max_iter=cfg.max_iter_lr
                )
                clf.fit(X_tr, y_tr)

                # 评估
                val_acc  = accuracy_score(y_val,  clf.predict(X_val))
                hold_acc = accuracy_score(y_holdout, clf.predict(X_hold))

                # —— 折回到“原特征空间”的等价 (w_raw, b_raw) —— #
                w = clf.coef_.ravel()           # z-score 空间
                b = float(clf.intercept_[0])
                if cfg.standardize:
                    w_raw = (w / sigma).astype(np.float32)
                    b_raw = float(b - float(np.dot(mu / sigma, w)))
                else:
                    w_raw = w.astype(np.float32)
                    b_raw = b

                best_w_raw   = w_raw
                best_b_raw   = b_raw
                best_hold_acc= float(hold_acc)

                # 满足提前停止
                nit = clf.n_iter_
                nit = max(nit) if isinstance(nit, (list, np.ndarray)) else nit
                did_conv = (nit < cfg.max_iter_lr)
                if did_conv or val_acc > cfg.val_thresh:
                    break

            # 6) 写入“原空间”参数与 hold-out acc
            W[it, l, :] = best_w_raw
            B[it, l]    = best_b_raw
            A[it, l]    = best_hold_acc

    observed_layers = build_observed_layers(W, B)
    print(f"[probe_sampler] n_iter={cfg.n_iter}, train_frac={cfg.train_frac}, ")
    return W, B, A, observed_layers, val_splits

def build_observed_layers(W: np.ndarray, B: np.ndarray):
    """将 W,B 组合成 list[(n_iter, d+1)] 的 [w|b]（此处的 W,B 已为“原特征空间”）"""
    n_iter, L, d = W.shape
    obs = []
    for l in range(L):
        w_mat = W[:, l, :]               # (n_iter, d)
        b_vec = B[:, l].reshape(-1, 1)   # (n_iter, 1)
        obs.append(np.hstack([w_mat, b_vec]))  # (n_iter, d+1)
    return obs

def save_probing_npz(save_dir: str, base_name: str, W: np.ndarray, B: np.ndarray, A: np.ndarray):
    """保存 w/b/acc（A 为 hold-out acc）"""
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, f"{base_name}_w.npz"),   W=W)
    np.savez(os.path.join(save_dir, f"{base_name}_b.npz"),   B=B)
    np.savez(os.path.join(save_dir, f"{base_name}_acc.npz"), Acc=A)

def save_val_splits(save_dir: str, base_name: str, val_splits: list[np.ndarray]):
    """
    保存每次 LR 的 OOB 索引（相对于 pool 的全局索引）。
    采用 dtype=object 的数组存储，便于每次长度不同。
    （函数名沿用 val_splits 以兼容 main.py，但内容是 OOB）
    """
    os.makedirs(save_dir, exist_ok=True)
    obj = np.array(val_splits, dtype=object)
    np.savez(os.path.join(save_dir, f"{base_name}_val_splits.npz"), val_splits=obj)
