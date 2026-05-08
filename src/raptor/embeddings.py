from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from raptor.core import save_embeddings_npz
from raptor.data import DataProcessing
from raptor.model_utils import LLM


def resolve_hf_snapshot_path(model_path: str) -> str:
    """Resolve a Hugging Face cache repo root to its snapshot directory."""
    if not os.path.isdir(model_path):
        return model_path
    if os.path.isfile(os.path.join(model_path, "config.json")):
        return model_path

    snapshots_dir = os.path.join(model_path, "snapshots")
    refs_dir = os.path.join(model_path, "refs")
    if os.path.isdir(snapshots_dir):
        main_ref = os.path.join(refs_dir, "main")
        if os.path.isfile(main_ref):
            with open(main_ref, "r", encoding="utf-8") as f:
                commit = f.read().strip()
            candidate = os.path.join(snapshots_dir, commit)
            if os.path.isdir(candidate):
                return candidate
        snapshots = [
            d for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ]
        snapshots.sort()
        if snapshots:
            return os.path.join(snapshots_dir, snapshots[-1])
    return model_path


def build_layer_count(model_id: str, model) -> int:
    layer_map = {
        "google/gemma-2b-it": 18,
        "google/gemma-7b-it": 28,
        "meta-llama/Llama-2-7b-chat-hf": 32,
        "meta-llama/Llama-2-13b-chat-hf": 40,
        "meta-llama/Llama-2-70b-chat-hf": 80,
        "Qwen/Qwen1.5-0.5B-Chat": 24,
        "Qwen/Qwen1.5-1.8B-Chat": 24,
        "Qwen/Qwen1.5-4B-Chat": 40,
        "Qwen/Qwen1.5-7B-Chat": 32,
        "Qwen/Qwen1.5-14B-Chat": 40,
        "Qwen/Qwen1.5-72B-Chat": 80,
    }
    if model_id in layer_map:
        return layer_map[model_id]
    cfg = getattr(model, "config", None)
    n_layers = getattr(cfg, "num_hidden_layers", None)
    if n_layers is None:
        raise KeyError(f"Unable to determine number of layers for {model_id}.")
    return int(n_layers)


def collect_embeddings(
    model_id: str,
    cache_dir: str,
    quant: int,
    cuda: int,
    pos_q: List[str],
    neg_q: List[str],
):
    quant_cfg = None
    if quant == 8:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == 4:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model_source = resolve_hf_snapshot_path(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_source, cache_dir=cache_dir)
    if quant == 32:
        model = AutoModelForCausalLM.from_pretrained(model_source, cache_dir=cache_dir)
    elif quant in (4, 8):
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            quantization_config=quant_cfg,
            cache_dir=cache_dir,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )

    layer_count = build_layer_count(model_id, model)
    llm = LLM(cuda_id=cuda, layer_num=layer_count, quant=quant)
    pos_layers = [[] for _ in range(layer_count)]
    neg_layers = [[] for _ in range(layer_count)]

    for samples, storage, desc in (
        (pos_q, pos_layers, "positive embeddings"),
        (neg_q, neg_layers, "negative embeddings"),
    ):
        for text in tqdm(samples, desc=desc):
            with torch.no_grad():
                hidden = llm.get_hidden_states(model, tokenizer, text)
            for layer in range(layer_count):
                storage[layer].append(hidden[layer, -1, :].cpu().numpy())

    X_pos = [np.vstack(pos_layers[layer]) for layer in range(layer_count)]
    X_neg = [np.vstack(neg_layers[layer]) for layer in range(layer_count)]
    dim = int(X_pos[0].shape[1]) if X_pos else 0
    return X_pos, X_neg, layer_count, dim


def main() -> None:
    parser = argparse.ArgumentParser(description="Save per-layer last-token LLM embeddings.")
    parser.add_argument("--savepath", type=str, default="./embeddings_all")
    parser.add_argument("--model_path", type=str, default=".")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--quant", type=int, default=32, choices=[4, 8, 16, 32])
    parser.add_argument("--noise", type=str, default="non-noise", choices=["noise", "non-noise"])
    args = parser.parse_args()

    data = DataProcessing(args.datapath, args.dataset, args.noise)
    pos_q, neg_q, _, _ = data.dispatcher()
    X_pos, X_neg, _, _ = collect_embeddings(
        model_id=args.model,
        cache_dir=args.model_path,
        quant=args.quant,
        cuda=args.cuda,
        pos_q=pos_q,
        neg_q=neg_q,
    )

    os.makedirs(args.savepath, exist_ok=True)
    model_tag = args.model.replace("/", "-")
    out_npz = os.path.join(args.savepath, f"{model_tag}_{args.dataset}_embeddings.npz")
    save_embeddings_npz(out_npz, args.model, args.dataset, X_pos, X_neg)
    print(f"Saved embeddings to: {out_npz}")


if __name__ == "__main__":
    main()
