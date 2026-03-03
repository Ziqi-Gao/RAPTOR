#!/usr/bin/env python3
"""
Save initial LLM embeddings (hidden states) only.

This script loads a tokenizer + model, runs the model to collect per-layer
hidden states for each sample (last-token vectors, consistent with main1.py),
and saves them to a compressed NPZ file. It does NOT train or save any
bagging or LR weights — only the raw initial embeddings.

Output layout (single .npz):
  - X_pos_0, X_pos_1, ... X_pos_{L-1}  (each: [N_pos, d])
  - X_neg_0, X_neg_1, ... X_neg_{L-1}  (each: [N_neg, d])
  - y_pos, y_neg                         (labels used: 1 for pos, 0 for neg)
  - meta_model, meta_dataset, meta_layers, meta_dim (strings/ints for metadata)

Example:
  python save_embeddings.py \
    --model google/gemma-2b-it \
    --datapath ./dataset/stsa.binary.train --dataset STSA \
    --cuda 0 --quant 32 --savepath ./lab_rs/emb/
"""

import argparse
import os
import pickle
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from dataset import DataProcessing
from util import LLM


def _resolve_hf_snapshot_path(model_path: str) -> str:
    """If given a local HF cache repo (models--org--name), resolve to its snapshot folder.

    Accepts:
      - A direct model folder containing config.json → return as-is
      - A HF cache folder like models--org--name with subdirs blobs/refs/snapshots →
        resolve refs/main (or the only snapshot) to models--.../snapshots/<commit>
      - Anything else → return as-is (caller may still succeed or raise)
    """
    if not os.path.isdir(model_path):
        return model_path

    # If it already looks like a proper model dir (has config.json), keep it
    if os.path.isfile(os.path.join(model_path, "config.json")):
        return model_path

    snapshots_dir = os.path.join(model_path, "snapshots")
    refs_dir = os.path.join(model_path, "refs")
    if os.path.isdir(snapshots_dir):
        # Prefer refs/main if present
        main_ref = os.path.join(refs_dir, "main")
        if os.path.isfile(main_ref):
            with open(main_ref, "r") as f:
                commit = f.read().strip()
            candidate = os.path.join(snapshots_dir, commit)
            if os.path.isdir(candidate):
                return candidate
        # Else, pick the first subdir under snapshots
        subs = [d for d in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, d))]
        subs.sort()  # deterministic
        if subs:
            return os.path.join(snapshots_dir, subs[-1])

    return model_path


def _build_layer_count(model_id: str, model) -> int:
    """Determine number of transformer layers.

    Prefers a curated map for common Chat/IT models; falls back to
    model.config.num_hidden_layers.
    """
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
    val = layer_map.get(model_id, None)
    if val is not None:
        return val
    cfg = getattr(model, "config", None)
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None:
        raise KeyError(
            f"Unable to determine number of layers for {model_id}; please extend layer_map or ensure config.num_hidden_layers is set."
        )
    return int(n)


def collect_embeddings(
    model_id: str,
    cache_dir: str,
    quant: int,
    cuda: int,
    pos_q: List[str],
    neg_q: List[str],
):
    """Load model/tokenizer and collect last-token hidden states per layer.

    Returns (X_pos_layers, X_neg_layers, tot_layer, dim)
    where X_pos_layers/X_neg_layers are lists of shape (n_samples, d) arrays.
    """
    quant_cfg = None
    if quant == 8:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == 4:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Allow passing a HF cache repo root; resolve to snapshot dir if needed
    model_id_resolved = _resolve_hf_snapshot_path(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id_resolved, cache_dir=cache_dir)
    if quant == 32:
        model = AutoModelForCausalLM.from_pretrained(model_id_resolved, cache_dir=cache_dir)
    elif quant in (4, 8):
        model = AutoModelForCausalLM.from_pretrained(
            model_id_resolved, quantization_config=quant_cfg, cache_dir=cache_dir
        )
    else:  # 16-bit
        model = AutoModelForCausalLM.from_pretrained(
            model_id_resolved, torch_dtype=torch.bfloat16, cache_dir=cache_dir
        )

    tot_layer = _build_layer_count(model_id_resolved, model)
    Model = LLM(cuda_id=cuda, layer_num=tot_layer, quant=quant)

    rp_pos = [[] for _ in range(tot_layer)]
    rp_neg = [[] for _ in range(tot_layer)]

    for samples, storage in [(pos_q, rp_pos), (neg_q, rp_neg)]:
        for q in tqdm(samples, desc="Collecting hidden states"):
            with torch.no_grad():
                hs = Model.get_hidden_states(model, tokenizer, q)  # (layers, seq, dim)
            for l in range(tot_layer):
                storage[l].append(hs[l, -1, :].cpu().numpy())

    X_pos = [np.vstack(rp_pos[l]) for l in range(tot_layer)]
    X_neg = [np.vstack(rp_neg[l]) for l in range(tot_layer)]
    dim = int(X_pos[0].shape[1]) if len(X_pos) > 0 else 0
    return X_pos, X_neg, tot_layer, dim


def main():
    parser = argparse.ArgumentParser(description="Save initial LLM embeddings (per layer, last-token)")
    parser.add_argument('--savepath',   type=str, default='./lab_rs/emb/', help='目录用于保存 .npz 文件')
    parser.add_argument('--model_path', type=str, default='./',            help='模型缓存目录')
    parser.add_argument('--model',      type=str, default='google/gemma-2b-it', help='模型名称/路径')
    parser.add_argument('--dataset',    type=str, default='STSA',          help='数据集名称（非 concept 模式）')
    parser.add_argument('--datapath',   type=str, default='./dataset/stsa.binary.train', help='数据文件路径（非 concept）')
    parser.add_argument('--cuda',       type=int, default=0,               help='CUDA 设备号')
    parser.add_argument('--quant',      type=int, default=32,              help='量化位数：4,8,16,32')
    parser.add_argument('--noise',      type=str, default='non-noise',     help='是否加噪：noise/non-noise')
    parser.add_argument('--concept',    type=str, default='',              help='概念名（如 Bird），读取 dataset/raw/{concept}.pkl')

    args = parser.parse_args()

    # Prepare input samples.
    if args.concept:
        one_pkl = os.path.join('dataset', 'raw', f"{args.concept}.pkl")
        if not os.path.isfile(one_pkl):
            raise FileNotFoundError(f"{one_pkl} 不存在，请检查概念名称")
        with open(one_pkl, 'rb') as f:
            d = pickle.load(f)
        key = list(d.keys())[0]
        pos_q = d[key].get('positive', [])
        neg_q = d[key].get('negative', [])
        dataset_tag = args.concept
        print(f"✅ 加载 {args.concept}: 正例 {len(pos_q)} 条, 负例 {len(neg_q)} 条")
    else:
        DP = DataProcessing(data_path=args.datapath, data_name=args.dataset, noise=args.noise)
        pos_q, neg_q, _, _ = DP.dispacher()
        dataset_tag = args.dataset

    # Collect raw hidden-state embeddings.
    X_pos, X_neg, L, d = collect_embeddings(
        model_id=args.model,
        cache_dir=args.model_path,
        quant=args.quant,
        cuda=args.cuda,
        pos_q=pos_q,
        neg_q=neg_q,
    )

    # Prepare output file path.
    os.makedirs(args.savepath, exist_ok=True)
    model_tag = args.model.replace('/', '-')
    base_name = f"{model_tag}_{dataset_tag}"
    out_npz = os.path.join(args.savepath, f"{base_name}_embeddings.npz")

    # Build compressed payload.
    save_dict = {}
    for l in range(L):
        save_dict[f"X_pos_{l}"] = X_pos[l]
        save_dict[f"X_neg_{l}"] = X_neg[l]
    # Binary labels follow the legacy convention: pos=1, neg=0.
    save_dict["y_pos"] = np.ones(len(X_pos[0]), dtype=int) if L > 0 and len(X_pos[0]) else np.array([], dtype=int)
    save_dict["y_neg"] = np.zeros(len(X_neg[0]), dtype=int) if L > 0 and len(X_neg[0]) else np.array([], dtype=int)
    # Metadata fields.
    save_dict["meta_model"] = np.array([args.model])
    save_dict["meta_dataset"] = np.array([dataset_tag])
    save_dict["meta_layers"] = np.array([L])
    save_dict["meta_dim"] = np.array([d])

    np.savez_compressed(out_npz, **save_dict)
    print(f">>> Saved embeddings to: {out_npz}")


if __name__ == "__main__":
    main()
