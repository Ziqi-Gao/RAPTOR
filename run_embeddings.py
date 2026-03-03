#!/usr/bin/env python3
import argparse
import os

from dataset import DataProcessing
from save_embeddings import collect_embeddings

from experiment_utils import (
    DATASET_PATHS,
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    MODEL_LOAD_OVERRIDES,
    MODEL_QUANT_OVERRIDES,
    ensure_root,
    mkdir,
    model_tag,
    normalize_dataset,
    parse_list,
    save_embeddings_npz,
    save_json,
    now,
)


def run_embeddings(
    models,
    datasets,
    emb_dir,
    *,
    model_path,
    cuda,
    quant,
    noise,
    force,
):
    mkdir(emb_dir)
    err_dir = os.path.join(emb_dir, "errors")
    mkdir(err_dir)

    for model_id in models:
        mtag = model_tag(model_id)
        load_id = model_id
        override = MODEL_LOAD_OVERRIDES.get(model_id)
        if override and os.path.isdir(override):
            load_id = override
        quant_for_model = MODEL_QUANT_OVERRIDES.get(model_id, quant)

        for dataset in datasets:
            data_path = DATASET_PATHS.get(dataset, "")
            emb_path = os.path.join(emb_dir, f"{mtag}_{dataset}_embeddings.npz")
            if not force and os.path.isfile(emb_path):
                continue

            print(f"[{now()}] embeddings start: model={model_id} dataset={dataset}")
            try:
                dp = DataProcessing(data_path=data_path, data_name=dataset, noise=noise)
                pos_q, neg_q, _, _ = dp.dispacher()
                X_pos, X_neg, _, _ = collect_embeddings(
                    model_id=load_id,
                    cache_dir=model_path,
                    quant=quant_for_model,
                    cuda=cuda,
                    pos_q=pos_q,
                    neg_q=neg_q,
                )
                save_embeddings_npz(emb_path, model_id, dataset, X_pos, X_neg)
                print(f"[{now()}] saved embeddings: {emb_path}")
            except Exception as exc:
                err_path = os.path.join(err_dir, f"{mtag}_{dataset}.json")
                save_json(err_path, {"error": str(exc)})
                print(f"[{now()}] embeddings failed: {model_id} {dataset} {exc}")


def main() -> None:
    ensure_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", type=str, default="./embeddings_all")
    ap.add_argument("--models", type=str, default="all")
    ap.add_argument("--datasets", type=str, default="all")
    ap.add_argument("--model_path", type=str, default=".")
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--quant", type=int, default=32)
    ap.add_argument("--noise", type=str, default="non-noise")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    models = parse_list(args.models, DEFAULT_MODELS)
    datasets = parse_list(args.datasets, DEFAULT_DATASETS, normalizer=normalize_dataset)
    run_embeddings(
        models,
        datasets,
        args.emb_dir,
        model_path=args.model_path,
        cuda=args.cuda,
        quant=args.quant,
        noise=args.noise,
        force=args.force,
    )


if __name__ == "__main__":
    main()
