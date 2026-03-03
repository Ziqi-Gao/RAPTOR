#!/usr/bin/env python3
import argparse

from experiment_utils import (
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    ensure_root,
    normalize_dataset,
    parse_list,
)
from run_embeddings import run_embeddings
from run_gcs import run_gcs_experiments
from run_singlelr import run_singlelr_experiments
from run_xrfm import run_xrfm_experiments


def main() -> None:
    ensure_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", type=str, default="./embeddings_all")
    ap.add_argument("--results_dir", type=str, default="./exp_results")
    ap.add_argument("--models", type=str, default="all")
    ap.add_argument("--datasets", type=str, default="all")
    ap.add_argument("--model_path", type=str, default=".")
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--quant", type=int, default=32)
    ap.add_argument("--noise", type=str, default="non-noise")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_embeddings", action="store_true")
    ap.add_argument("--skip_embeddings", action="store_true")
    ap.add_argument("--methods", type=str, default="xrfm,singlelr,gcs")

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

    models = parse_list(args.models, DEFAULT_MODELS)
    datasets = parse_list(args.datasets, DEFAULT_DATASETS, normalizer=normalize_dataset)
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]

    if not args.skip_embeddings:
        run_embeddings(
            models,
            datasets,
            args.emb_dir,
            model_path=args.model_path,
            cuda=args.cuda,
            quant=args.quant,
            noise=args.noise,
            force=args.force_embeddings,
        )

    if "singlelr" in methods:
        run_singlelr_experiments(
            models,
            datasets,
            args.emb_dir,
            args.results_dir,
            test_size=args.test_size,
            val_size=args.val_size,
            seed=args.seed,
            max_iter=args.singlelr_max_iter,
        )

    if "xrfm" in methods or "rfm" in methods:
        run_xrfm_experiments(
            models,
            datasets,
            args.emb_dir,
            args.results_dir,
            test_size=args.test_size,
            val_size=args.val_size,
            seed=args.seed,
            rfm_iters=args.rfm_iters,
            n_components=args.rfm_n_components,
        )

    if "gcs" in methods:
        run_gcs_experiments(
            models,
            datasets,
            args.emb_dir,
            args.results_dir,
            test_size=args.test_size,
            val_size=args.val_size,
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


if __name__ == "__main__":
    main()
