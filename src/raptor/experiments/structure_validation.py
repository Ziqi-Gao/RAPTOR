#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate plug-in accuracy structure on embedding data in embeddings_all/*.npz.

Expected NPZ format (per save_embeddings.py):
  - X_pos_0, X_pos_1, ... X_pos_{L-1}
  - X_neg_0, X_neg_1, ... X_neg_{L-1}

Example:
  python3 acc_structure_validation.py \
    --emb_npz embeddings_all/Qwen-Qwen2.5-7B-Instruct_STSA_embeddings.npz \
    --layer 10 \
    --out exp_results/acc_struct/qwen7b_stsa_layer10
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPClassifier
from scipy.stats import spearmanr, pearsonr
from scipy.special import expit
from scipy.stats import norm as normal_dist


def load_embedding_layer(npz_path: str, layer: int) -> Tuple[np.ndarray, np.ndarray]:
    pos_key = f"X_pos_{layer}"
    neg_key = f"X_neg_{layer}"
    with np.load(npz_path, allow_pickle=False) as data:
        if pos_key not in data or neg_key not in data:
            keys = sorted(list(data.keys()))
            raise ValueError(
                f"Missing {pos_key} or {neg_key} in {npz_path}. Available keys: {keys}"
            )
        X_pos = data[pos_key]
        X_neg = data[neg_key]
    if X_pos.ndim != 2 or X_neg.ndim != 2:
        raise ValueError("X_pos/X_neg must be 2D arrays.")
    if X_pos.shape[1] != X_neg.shape[1]:
        raise ValueError("X_pos/X_neg feature dims do not match.")
    X = np.vstack([X_pos, X_neg]).astype(np.float64)
    y = np.concatenate(
        [np.ones(len(X_pos), dtype=int), np.zeros(len(X_neg), dtype=int)],
        axis=0,
    )
    return X, y


def default_out_dir(npz_path: str, layer: int) -> str:
    base = os.path.splitext(os.path.basename(npz_path))[0]
    return os.path.join("exp_results", "acc_structure_validation", base, f"layer_{layer}")


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def plugin_acc_pred(
    U: np.ndarray,
    p_cal: np.ndarray,
    a: float,
    b: float,
    sigma: float
) -> float:
    """
    Plug-in version of your Acc structure:
      y|U ~ Bernoulli(g(U))
      S|U ~ N(a U + b, sigma^2)
      predict y_hat = 1{S>=0}
    Then:
      Acc = E[ g(U) * P(S>=0|U) + (1-g(U))*P(S<0|U) ].
    Replace expectation by empirical average over U samples.
    """
    if sigma <= 1e-12:
        # almost deterministic: S approx aU
        ps_ge0 = (a * U + b >= 0).astype(float)
    else:
        ps_ge0 = normal_dist.cdf((a * U + b) / sigma)  # Phi((aU+b)/sigma)

    acc = np.mean(p_cal * ps_ge0 + (1 - p_cal) * (1 - ps_ge0))
    return float(acc)


def calibrate_probs(
    U_oof: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    eps: float = 1e-6,
) -> np.ndarray:
    p_cal = np.zeros_like(U_oof, dtype=np.float64)
    for tr_idx, te_idx in splits:
        u_tr = U_oof[tr_idx]
        y_tr = y[tr_idx]
        u_te = U_oof[te_idx]
        if np.unique(y_tr).size < 2:
            p_cal[te_idx] = float(np.mean(y_tr))
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(u_tr, y_tr)
        p_cal[te_idx] = iso.predict(u_te)
    return np.clip(p_cal, eps, 1 - eps)


def parse_model_dataset(emb_npz: str) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(emb_npz))[0]
    suffix = "_embeddings"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    if "_" in base:
        model, dataset = base.rsplit("_", 1)
    else:
        model, dataset = "", ""
    return model, dataset


def parse_float_list(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        return None
    return [float(x) for x in items]


def stratified_split_indices(
    y: np.ndarray,
    eval_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=eval_frac, random_state=seed)
    pool_idx, eval_idx = next(splitter.split(np.zeros(len(y)), y))
    return pool_idx, eval_idx


def stratified_subsample_indices(
    y: np.ndarray,
    n_sub: int,
    seed: int,
) -> np.ndarray:
    if n_sub >= len(y):
        return np.arange(len(y))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_sub, random_state=seed)
    sub_idx, _ = next(splitter.split(np.zeros(len(y)), y))
    return sub_idx


def scale_train_eval(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    standardize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if not standardize:
        return X_train, X_eval
    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(X_train), scaler.transform(X_eval)


def train_oracle(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    args: argparse.Namespace,
    Cs_oracle: List[float],
):
    if args.oracle == "lr_cv":
        oracle = LogisticRegressionCV(
            Cs=Cs_oracle,
            cv=3,
            penalty="l2",
            solver=args.solver,
            max_iter=args.oracle_max_iter,
            fit_intercept=args.fit_intercept,
            n_jobs=None,
        )
    elif args.oracle == "mlp":
        oracle = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=1e-4,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=args.seed,
        )
    else:
        raise ValueError(f"Unknown oracle: {args.oracle}")
    oracle.fit(X_tr, y_tr)
    return oracle


def run_crossfit(
    X: np.ndarray,
    y: np.ndarray,
    Cs_probe: List[float],
    Cs_oracle: List[float],
    args: argparse.Namespace,
) -> Tuple[
    np.ndarray,
    Dict[float, np.ndarray],
    np.ndarray,
    Dict[float, np.ndarray],
    List[Tuple[np.ndarray, np.ndarray]],
]:
    n = X.shape[0]
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    splits = list(skf.split(X, y))

    U_oof = np.zeros(n, dtype=np.float64)
    S_oof: Dict[float, np.ndarray] = {C: np.zeros(n, dtype=np.float64) for C in Cs_probe}
    y_oof = np.zeros(n, dtype=int)
    yhat_oof: Dict[float, np.ndarray] = {C: np.zeros(n, dtype=int) for C in Cs_probe}

    for tr_idx, te_idx in splits:
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        y_oof[te_idx] = y_te

        if args.standardize:
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        oracle = train_oracle(X_tr, y_tr, args, Cs_oracle)
        p_te = oracle.predict_proba(X_te)[:, 1]
        U_oof[te_idx] = logit(p_te)

        for C in Cs_probe:
            clf = LogisticRegression(
                C=C,
                penalty="l2",
                solver=args.solver,
                max_iter=args.max_iter,
                fit_intercept=args.fit_intercept,
            )
            clf.fit(X_tr, y_tr)
            S = clf.decision_function(X_te).astype(np.float64)
            S_oof[C][te_idx] = S
            yhat_oof[C][te_idx] = (S >= 0).astype(int)

    return U_oof, S_oof, y_oof, yhat_oof, splits


def fit_su_regression(U: np.ndarray, S: np.ndarray) -> Tuple[float, float, float, float]:
    reg = LinearRegression(fit_intercept=True)
    reg.fit(U.reshape(-1, 1), S)
    a_hat = float(reg.coef_[0])
    b_hat = float(reg.intercept_)
    S_pred = reg.predict(U.reshape(-1, 1))
    resid = S - S_pred
    sigma_hat = float(np.std(resid, ddof=1))

    ss_res = float(np.sum((S - S_pred) ** 2))
    ss_tot = float(np.sum((S - np.mean(S)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return a_hat, b_hat, sigma_hat, float(r2)


def reduce_whiten_random_project(
    X_sub: np.ndarray,
    X_eval: np.ndarray,
    p_prime: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_sub_std = scaler.fit_transform(X_sub)
    X_eval_std = scaler.transform(X_eval)

    n_sub, p_orig = X_sub_std.shape
    n_comp = min(n_sub - 1, p_orig)
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
    X_sub_pca = pca.fit_transform(X_sub_std)
    X_eval_pca = pca.transform(X_eval_std)

    ev = pca.explained_variance_
    denom = np.sqrt(ev + 1e-12)
    X_sub_w = X_sub_pca / denom
    X_eval_w = X_eval_pca / denom

    rng = np.random.default_rng(seed)
    R = rng.normal(size=(n_comp, p_prime)) / np.sqrt(float(p_prime))
    X_sub_prime = X_sub_w @ R
    X_eval_prime = X_eval_w @ R
    return X_sub_prime, X_eval_prime


def eval_oracle_fold_ensemble(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    X_eval: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    Cs_oracle: List[float],
) -> np.ndarray:
    p_eval_folds = []
    for tr_idx, _ in splits:
        X_tr = X_sub[tr_idx]
        y_tr = y_sub[tr_idx]
        oracle_fold = train_oracle(X_tr, y_tr, args, Cs_oracle)
        p_eval_folds.append(oracle_fold.predict_proba(X_eval)[:, 1])
    return np.mean(np.vstack(p_eval_folds), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npz", required=True, help="Embedding NPZ path (contains X_pos_i/X_neg_i)")
    ap.add_argument("--layer", required=True, type=int, help="Layer index to use")
    ap.add_argument("--out", default=None, help="Output directory (default: exp_results/acc_structure_validation/...)")

    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--exp", type=str, default="A", choices=["A", "B"],
                    help="Experiment mode: A (vary n) or B (fixed delta)")

    ap.add_argument("--train_fracs", default=None,
                    help="comma-separated training fractions for Experiment A (optional)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--eval_frac", type=float, default=0.2)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--subsample_seed", type=int, default=0)
    ap.add_argument("--deltas", default="0.5,1,2",
                    help="comma-separated deltas for Experiment B")
    ap.add_argument("--p_list", default="512,1024,2048,3072",
                    help="comma-separated p' list for Experiment B")
    ap.add_argument("--proj_seed", type=int, default=0)
    ap.add_argument("--dr_method", type=str, default="pca_whiten_rp",
                    help="Dimensionality reduction method for Experiment B")
    ap.add_argument("--C_fixed", type=float, default=0.001,
                    help="Fixed probe C for Experiment B")

    # Probe grid: sklearn uses C = 1/lambda
    ap.add_argument("--Cs", type=str, default="0.00001,0.0001,0.001,0.01,0.1,0.3,1,3,10,30,100",
                    help="comma-separated list of C values for probe LR")
    # Oracle
    ap.add_argument("--oracle", type=str, default="lr_cv",
                    choices=["lr_cv", "mlp"], help="oracle model to generate U-hat")
    ap.add_argument("--oracle_Cs", type=str, default="0.01,0.03,0.1,0.3,1,3,10,30,100",
                    help="Cs grid for LogisticRegressionCV oracle")
    ap.add_argument("--oracle_max_iter", type=int, default=5000)

    # Common LR settings
    ap.add_argument("--solver", type=str, default="lbfgs")
    ap.add_argument("--max_iter", type=int, default=5000)

    ap.add_argument("--sigma_floor", type=float, default=0.2,
                    help="Floor for sigma used in plug-in predictor")

    ap.add_argument("--fit_intercept", action="store_true",
                    help="Fit intercept in LR; score uses Xw + b")
    ap.add_argument("--standardize", action="store_true",
                    help="Standardize X within each fold using training split only")
    ap.add_argument("--no_standardize", dest="standardize", action="store_false",
                    help="Disable standardization (use raw embeddings)")
    ap.set_defaults(standardize=True)

    ap.add_argument("--calibrate_oracle", dest="calibrate_oracle", action="store_true",
                    help="Calibrate oracle logits with isotonic regression (default)")
    ap.add_argument("--no_calibrate_oracle", dest="calibrate_oracle", action="store_false",
                    help="Disable calibration and use sigmoid(U)")
    ap.set_defaults(calibrate_oracle=True)

    args = ap.parse_args()

    out_dir = args.out if args.out else default_out_dir(args.emb_npz, args.layer)
    os.makedirs(out_dir, exist_ok=True)

    X, y = load_embedding_layer(args.emb_npz, args.layer)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"Shape mismatch: X {X.shape}, y {y.shape}")

    n, p = X.shape

    Cs_probe = [float(s) for s in args.Cs.split(",")]
    Cs_oracle = [float(s) for s in args.oracle_Cs.split(",")]

    train_fracs = parse_float_list(args.train_fracs)
    model_name, dataset_name = parse_model_dataset(args.emb_npz)

    if args.exp == "B":
        deltas = parse_float_list(args.deltas) or []
        p_list = [int(x) for x in (args.p_list.split(",") if args.p_list else []) if x.strip()]
        if args.dr_method != "pca_whiten_rp":
            raise ValueError(f"Unsupported dr_method: {args.dr_method}")

        pool_idx, eval_idx = stratified_split_indices(y, args.eval_frac, args.eval_seed)
        X_pool, y_pool = X[pool_idx], y[pool_idx]
        X_eval, y_eval = X[eval_idx], y[eval_idx]

        rows_b: List[Dict[str, float]] = []
        prob_eps = 1e-6

        for d_idx, delta in enumerate(deltas):
            for p_idx, p_prime in enumerate(p_list):
                if delta <= 0:
                    raise ValueError(f"Invalid delta: {delta}")
                n_prime = int(np.floor(float(p_prime) / float(delta)))
                if n_prime > len(y_pool) or n_prime < 200:
                    print(f"[warn] skip delta={delta} p'={p_prime} (n'={n_prime})")
                    continue
                combo_idx = d_idx * max(1, len(p_list)) + p_idx
                for r in range(args.repeats):
                    seed = args.subsample_seed + r + 1000 * combo_idx
                    sub_rel_idx = stratified_subsample_indices(y_pool, n_prime, seed)
                    X_sub = X_pool[sub_rel_idx]
                    y_sub = y_pool[sub_rel_idx]

                    n_sub = X_sub.shape[0]
                    skf = StratifiedKFold(
                        n_splits=args.kfold,
                        shuffle=True,
                        random_state=args.seed,
                    )
                    splits = list(skf.split(X_sub, y_sub))
                    U_oof = np.zeros(n_sub, dtype=np.float64)
                    S_oof = np.zeros(n_sub, dtype=np.float64)
                    p_eval_folds = []
                    s_eval_folds = []

                    for fold_id, (tr_idx, te_idx) in enumerate(splits):
                        X_tr_raw = X_sub[tr_idx]
                        y_tr = y_sub[tr_idx]
                        X_te_raw = X_sub[te_idx]

                        scaler = StandardScaler(with_mean=True, with_std=True)
                        X_tr_std = scaler.fit_transform(X_tr_raw)
                        X_te_std = scaler.transform(X_te_raw)
                        X_eval_std = scaler.transform(X_eval)

                        n_comp = min(X_tr_std.shape[0] - 1, X_tr_std.shape[1])
                        pca = PCA(
                            n_components=n_comp,
                            svd_solver="randomized",
                            random_state=args.proj_seed + r + fold_id,
                        )
                        X_tr_pca = pca.fit_transform(X_tr_std)
                        X_te_pca = pca.transform(X_te_std)
                        X_eval_pca = pca.transform(X_eval_std)

                        ev = pca.explained_variance_
                        denom = np.sqrt(ev + 1e-12)
                        X_tr_w = X_tr_pca / denom
                        X_te_w = X_te_pca / denom
                        X_eval_w = X_eval_pca / denom

                        rng = np.random.default_rng(args.proj_seed + r + fold_id)
                        R = rng.normal(size=(n_comp, p_prime)) / np.sqrt(float(p_prime))
                        X_tr_prime = X_tr_w @ R
                        X_te_prime = X_te_w @ R
                        X_eval_prime = X_eval_w @ R

                        oracle = train_oracle(X_tr_prime, y_tr, args, Cs_oracle)
                        p_te = oracle.predict_proba(X_te_prime)[:, 1]
                        U_oof[te_idx] = logit(p_te)
                        p_eval_folds.append(oracle.predict_proba(X_eval_prime)[:, 1])

                        clf_fold = LogisticRegression(
                            C=args.C_fixed,
                            penalty="l2",
                            solver="liblinear",
                            max_iter=args.max_iter,
                            fit_intercept=args.fit_intercept,
                        )
                        clf_fold.fit(X_tr_prime, y_tr)
                        S_oof[te_idx] = clf_fold.decision_function(X_te_prime).astype(np.float64)
                        s_eval_folds.append(
                            clf_fold.decision_function(X_eval_prime).astype(np.float64)
                        )

                    a_hat, b_hat, sigma_hat, r2 = fit_su_regression(U_oof, S_oof)

                    p_eval = np.mean(np.vstack(p_eval_folds), axis=0)
                    U_eval = logit(p_eval)
                    S_eval = np.mean(np.vstack(s_eval_folds), axis=0)
                    yhat_eval = (S_eval >= 0).astype(int)
                    true_acc_eval = float(accuracy_score(y_eval, yhat_eval))

                    if args.calibrate_oracle:
                        if np.unique(y_sub).size < 2:
                            p_cal_eval = np.full_like(
                                U_eval, float(np.mean(y_sub)), dtype=np.float64
                            )
                        else:
                            iso = IsotonicRegression(out_of_bounds="clip")
                            iso.fit(U_oof, y_sub)
                            p_cal_eval = iso.predict(U_eval)
                        p_cal_eval = np.clip(p_cal_eval, prob_eps, 1 - prob_eps)
                    else:
                        p_cal_eval = np.clip(expit(U_eval), prob_eps, 1 - prob_eps)

                    sigma_used = max(sigma_hat, float(args.sigma_floor))
                    pred_acc_eval = plugin_acc_pred(
                        U_eval, p_cal_eval, a_hat, b_hat, sigma_used
                    )

                    rows_b.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "layer": int(args.layer),
                        "delta": float(delta),
                        "p_prime": int(p_prime),
                        "n_prime": int(n_prime),
                        "true_acc_eval": float(true_acc_eval),
                        "pred_acc_eval": float(pred_acc_eval),
                        "a": float(a_hat),
                        "b": float(b_hat),
                        "sigma": float(sigma_hat),
                        "R2": float(r2),
                    })

        csv_path = os.path.join(out_dir, "acc_structure_validation_expB.csv")
        if rows_b:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows_b[0].keys()))
                writer.writeheader()
                writer.writerows(rows_b)

        summary_b: Dict[str, object] = {
            "mode": "expB",
            "emb_npz": args.emb_npz,
            "layer": int(args.layer),
            "p_orig": int(p),
            "eval_frac": float(args.eval_frac),
            "eval_seed": int(args.eval_seed),
            "subsample_seed": int(args.subsample_seed),
            "proj_seed": int(args.proj_seed),
            "dr_method": args.dr_method,
            "repeats": int(args.repeats),
            "oracle": args.oracle,
            "sigma_floor": float(args.sigma_floor),
            "calibrate_oracle": bool(args.calibrate_oracle),
            "deltas": deltas,
            "p_list": p_list,
            "C_fixed": float(args.C_fixed),
            "csv_path": csv_path,
        }

        collapse = {}
        for delta in deltas:
            by_p = {}
            for row in rows_b:
                if row["delta"] != float(delta):
                    continue
                by_p.setdefault(row["p_prime"], []).append(row)
            p_means = []
            for p_prime, vals in by_p.items():
                true_mean = float(np.mean([v["true_acc_eval"] for v in vals]))
                pred_mean = float(np.mean([v["pred_acc_eval"] for v in vals]))
                p_means.append((p_prime, true_mean, pred_mean))
            if p_means:
                true_vals = [x[1] for x in p_means]
                pred_vals = [x[2] for x in p_means]
                collapse[str(delta)] = {
                    "p_means": p_means,
                    "true_acc_mean_over_p": float(np.mean(true_vals)),
                    "true_acc_std_over_p": float(np.std(true_vals, ddof=0)),
                    "pred_acc_mean_over_p": float(np.mean(pred_vals)),
                    "pred_acc_std_over_p": float(np.std(pred_vals, ddof=0)),
                    "self_avg_true_std": float(np.std(true_vals, ddof=0)),
                    "self_avg_pred_std": float(np.std(pred_vals, ddof=0)),
                }
            else:
                collapse[str(delta)] = {
                    "p_means": [],
                    "true_acc_mean_over_p": None,
                    "true_acc_std_over_p": None,
                    "pred_acc_mean_over_p": None,
                    "pred_acc_std_over_p": None,
                    "self_avg_true_std": None,
                    "self_avg_pred_std": None,
                }

        summary_b["collapse"] = collapse
        with open(os.path.join(out_dir, "acc_structure_validation_expB.json"), "w", encoding="utf-8") as f:
            json.dump(summary_b, f, indent=2)
        print(f"[done] Experiment B results saved to: {csv_path}")
        return

    if train_fracs:
        pool_idx, eval_idx = stratified_split_indices(y, args.eval_frac, args.eval_seed)
        X_pool, y_pool = X[pool_idx], y[pool_idx]
        X_eval, y_eval = X[eval_idx], y[eval_idx]

        rows: List[Dict[str, float]] = []
        prob_eps = 1e-6

        for frac in train_fracs:
            if frac <= 0 or frac > 1:
                raise ValueError(f"Invalid train_frac: {frac}")
            n_sub = int(np.floor(frac * len(y_pool)))
            if n_sub < 2:
                raise ValueError(f"train_frac too small: {frac} gives n={n_sub}")
            for r in range(args.repeats):
                sub_rel_idx = stratified_subsample_indices(
                    y_pool, n_sub, args.subsample_seed + r
                )
                X_sub = X_pool[sub_rel_idx]
                y_sub = y_pool[sub_rel_idx]

                U_oof, S_oof, _, _, splits = run_crossfit(
                    X_sub, y_sub, Cs_probe, Cs_oracle, args
                )
                if args.calibrate_oracle:
                    p_cal_oof = calibrate_probs(U_oof, y_sub, splits, eps=prob_eps)
                else:
                    p_cal_oof = np.clip(expit(U_oof), prob_eps, 1 - prob_eps)

                metrics_by_C: Dict[float, Tuple[float, float, float, float]] = {}
                for C in Cs_probe:
                    metrics_by_C[C] = fit_su_regression(U_oof, S_oof[C])

                p_eval_folds = []
                for tr_idx, _ in splits:
                    X_tr = X_sub[tr_idx]
                    y_tr = y_sub[tr_idx]
                    if args.standardize:
                        scaler = StandardScaler(with_mean=True, with_std=True)
                        X_tr = scaler.fit_transform(X_tr)
                        X_eval_fold = scaler.transform(X_eval)
                    else:
                        X_eval_fold = X_eval
                    oracle_fold = train_oracle(X_tr, y_tr, args, Cs_oracle)
                    p_eval_folds.append(oracle_fold.predict_proba(X_eval_fold)[:, 1])

                p_eval = np.mean(np.vstack(p_eval_folds), axis=0)
                U_eval = logit(p_eval)

                if args.calibrate_oracle:
                    if np.unique(y_sub).size < 2:
                        p_cal_eval = np.full_like(
                            U_eval, float(np.mean(y_sub)), dtype=np.float64
                        )
                    else:
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit(U_oof, y_sub)
                        p_cal_eval = iso.predict(U_eval)
                    p_cal_eval = np.clip(p_cal_eval, prob_eps, 1 - prob_eps)
                else:
                    p_cal_eval = np.clip(expit(U_eval), prob_eps, 1 - prob_eps)

                for C in Cs_probe:
                    clf = LogisticRegression(
                        C=C,
                        penalty="l2",
                        solver=args.solver,
                        max_iter=args.max_iter,
                        fit_intercept=args.fit_intercept,
                    )
                    X_sub_full, X_eval_full = scale_train_eval(
                        X_sub, X_eval, args.standardize
                    )
                    clf.fit(X_sub_full, y_sub)
                    S_eval = clf.decision_function(X_eval_full).astype(np.float64)
                    yhat_eval = (S_eval >= 0).astype(int)
                    true_acc_eval = float(accuracy_score(y_eval, yhat_eval))

                    a_hat, b_hat, sigma_hat, r2 = metrics_by_C[C]
                    sigma_used = max(sigma_hat, float(args.sigma_floor))
                    pred_acc_eval = plugin_acc_pred(
                        U_eval, p_cal_eval, a_hat, b_hat, sigma_used
                    )

                    rows.append({
                        "frac": float(frac),
                        "repeat": int(r),
                        "n": int(n_sub),
                        "p": int(p),
                        "delta": float(p / n_sub),
                        "C": float(C),
                        "true_acc_eval": true_acc_eval,
                        "pred_acc_eval": float(pred_acc_eval),
                        "a_hat": float(a_hat),
                        "b_hat": float(b_hat),
                        "sigma_hat": float(sigma_hat),
                        "sigma_used": float(sigma_used),
                        "r2_SU": float(r2),
                    })

        csv_path = os.path.join(out_dir, "acc_structure_validation_expA.csv")
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        mean_rows: List[Dict[str, float]] = []
        corr_by_C: Dict[str, Dict[str, object]] = {}
        for C in Cs_probe:
            by_frac: Dict[float, List[Dict[str, float]]] = {}
            for row in rows:
                if row["C"] != float(C):
                    continue
                by_frac.setdefault(row["frac"], []).append(row)
            fracs_sorted = sorted(by_frac.keys())
            true_means = []
            pred_means = []
            for frac in fracs_sorted:
                vals = by_frac[frac]
                true_mean = float(np.mean([v["true_acc_eval"] for v in vals]))
                pred_mean = float(np.mean([v["pred_acc_eval"] for v in vals]))
                mean_rows.append({
                    "frac": float(frac),
                    "C": float(C),
                    "n": int(np.floor(frac * len(y_pool))),
                    "p": int(p),
                    "delta": float(p / max(1, int(np.floor(frac * len(y_pool))))),
                    "true_acc_mean": true_mean,
                    "pred_acc_mean": pred_mean,
                })
                true_means.append(true_mean)
                pred_means.append(pred_mean)

            if len(true_means) >= 2:
                sp = spearmanr(true_means, pred_means).correlation
                pr = pearsonr(true_means, pred_means)[0]
            else:
                sp = None
                pr = None

            corr_by_C[str(C)] = {
                "spearman": float(sp) if sp is not None else None,
                "pearson": float(pr) if pr is not None else None,
                "n_fracs": len(true_means),
            }

        summary = {
            "mode": "expA",
            "emb_npz": args.emb_npz,
            "layer": int(args.layer),
            "p": int(p),
            "eval_frac": float(args.eval_frac),
            "eval_seed": int(args.eval_seed),
            "subsample_seed": int(args.subsample_seed),
            "repeats": int(args.repeats),
            "kfold": int(args.kfold),
            "standardize": bool(args.standardize),
            "fit_intercept": bool(args.fit_intercept),
            "oracle": args.oracle,
            "sigma_floor": float(args.sigma_floor),
            "calibrate_oracle": bool(args.calibrate_oracle),
            "train_fracs": train_fracs,
            "Cs_probe": Cs_probe,
            "Cs_oracle": Cs_oracle,
            "csv_path": csv_path,
            "mean_rows": mean_rows,
            "corr_by_C": corr_by_C,
        }

        with open(os.path.join(out_dir, "acc_structure_validation_expA.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[done] Experiment A results saved to: {csv_path}")
        return

    U_oof, S_oof, y_oof, yhat_oof, splits = run_crossfit(X, y, Cs_probe, Cs_oracle, args)

    # Now we have full out-of-fold U and S for every sample.
    prob_eps = 1e-6
    if args.calibrate_oracle:
        p_cal_oof = calibrate_probs(U_oof, y, splits, eps=prob_eps)
        cal_method = "isotonic"
    else:
        p_cal_oof = np.clip(expit(U_oof), prob_eps, 1 - prob_eps)
        cal_method = "sigmoid"

    p_cal_brier = float(brier_score_loss(y, p_cal_oof))
    p_cal_log_loss = float(log_loss(y, p_cal_oof, labels=[0, 1]))

    results: List[Dict[str, float]] = []

    U = U_oof

    for C in Cs_probe:
        S = S_oof[C]

        a_hat, b_hat, sigma_hat, r2 = fit_su_regression(U, S)

        # True oof accuracy
        true_acc = float(accuracy_score(y_oof, yhat_oof[C]))

        # Predicted accuracy by plug-in structure (empirical average over U)
        sigma_used = max(sigma_hat, float(args.sigma_floor))
        pred_acc = plugin_acc_pred(U, p_cal_oof, a_hat, b_hat, sigma_used)

        results.append({
            "C": C,
            "true_acc": true_acc,
            "pred_acc": pred_acc,
            "a_hat": a_hat,
            "b_hat": b_hat,
            "sigma_hat": sigma_hat,
            "r2_SU": float(r2),
        })

    # Trend metrics
    true = np.array([r["true_acc"] for r in results])
    pred = np.array([r["pred_acc"] for r in results])

    sp = spearmanr(true, pred).correlation
    pr = pearsonr(true, pred)[0] if len(true) > 1 else 0.0

    summary = {
        "emb_npz": args.emb_npz,
        "layer": int(args.layer),
        "n": int(n),
        "p": int(p),
        "kfold": int(args.kfold),
        "standardize": bool(args.standardize),
        "fit_intercept": bool(args.fit_intercept),
        "oracle": args.oracle,
        "sigma_floor": float(args.sigma_floor),
        "calibrate_oracle": bool(args.calibrate_oracle),
        "calibration_method": cal_method,
        "p_cal_brier": p_cal_brier,
        "p_cal_log_loss": p_cal_log_loss,
        "Cs_probe": Cs_probe,
        "Cs_oracle": Cs_oracle,
        "spearman_true_vs_pred": float(sp) if sp is not None else None,
        "pearson_true_vs_pred": float(pr),
        "results": results,
    }

    with open(os.path.join(out_dir, "acc_structure_validation.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print quick view
    print("Spearman(true_acc, pred_acc) =", summary["spearman_true_vs_pred"])
    print("Pearson (true_acc, pred_acc) =", summary["pearson_true_vs_pred"])
    print("\nTop by true_acc:")
    for r in sorted(results, key=lambda x: x["true_acc"], reverse=True)[:5]:
        print(
            "  C={C:>7g}  true={true_acc:.4f}  pred={pred_acc:.4f}  a={a_hat:.4f}  "
            "b={b_hat:.4f}  sigma={sigma_hat:.4f}  R2={r2_SU:.3f}".format(**r)
        )

    print("\nTop by pred_acc:")
    for r in sorted(results, key=lambda x: x["pred_acc"], reverse=True)[:5]:
        print(
            "  C={C:>7g}  true={true_acc:.4f}  pred={pred_acc:.4f}  a={a_hat:.4f}  "
            "b={b_hat:.4f}  sigma={sigma_hat:.4f}  R2={r2_SU:.3f}".format(**r)
        )


if __name__ == "__main__":
    main()
