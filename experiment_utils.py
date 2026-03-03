#!/usr/bin/env python3
import json
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODELS = [
    "google/gemma-7b-it",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]

DEFAULT_DATASETS = [
    "STSA",
    "sarcasm",
    "hatexplain",
    "counterfact",
    "cities",
    "common",
]

ALIAS_DATASETS = {
    "stsa": "STSA",
    "sarcasm": "sarcasm",
    "hatexplain": "hatexplain",
    "counterfact": "counterfact",
    "cities": "cities",
    "common": "common",
}

DATASET_PATHS = {
    "STSA": os.path.join(ROOT_DIR, "dataset", "stsa.binary.train"),
    "sarcasm": os.path.join(ROOT_DIR, "dataset", "sarcasm.json"),
    "hatexplain": os.path.join(ROOT_DIR, "data", "hatexplain"),
    "counterfact": os.path.join(ROOT_DIR, "dataset", "counterfact.csv"),
    "cities": os.path.join(ROOT_DIR, "dataset", "cities.csv"),
    "common": os.path.join(ROOT_DIR, "dataset", "common_claim.csv"),
}

MODEL_LOAD_OVERRIDES = {
    "meta-llama/Llama-3.3-70B-Instruct": os.path.join(
        ROOT_DIR, "RFM", "hf_cache", "llama-3.3-70b-4bit"
    ),
}

MODEL_QUANT_OVERRIDES = {
    "meta-llama/Meta-Llama-3.1-70B-Instruct": 4,
    "Qwen/Qwen2.5-32B-Instruct": 4,
}


def ensure_root() -> None:
    os.chdir(ROOT_DIR)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def model_tag(model_id: str) -> str:
    return model_id.replace("/", "-")


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_dataset(name: str) -> str:
    key = name.strip().lower()
    return ALIAS_DATASETS.get(key, name.strip())


def parse_list(
    arg: str,
    default_list: Iterable[str],
    *,
    normalizer: Optional[Callable[[str], str]] = None,
) -> List[str]:
    if arg.strip().lower() == "all":
        return list(default_list)
    items = [x.strip() for x in arg.split(",") if x.strip()]
    if normalizer:
        items = [normalizer(x) for x in items]
    return items


def save_embeddings_npz(
    out_path: str,
    model_id: str,
    dataset_tag: str,
    X_pos_layers: List[np.ndarray],
    X_neg_layers: List[np.ndarray],
) -> None:
    L = len(X_pos_layers)
    d = int(X_pos_layers[0].shape[1]) if L > 0 else 0
    save_dict: Dict[str, Any] = {}
    for l in range(L):
        save_dict[f"X_pos_{l}"] = X_pos_layers[l]
        save_dict[f"X_neg_{l}"] = X_neg_layers[l]
    save_dict["y_pos"] = np.ones(len(X_pos_layers[0]), dtype=int) if L > 0 else np.array([], dtype=int)
    save_dict["y_neg"] = np.zeros(len(X_neg_layers[0]), dtype=int) if L > 0 else np.array([], dtype=int)
    save_dict["meta_model"] = np.array([model_id])
    save_dict["meta_dataset"] = np.array([dataset_tag])
    save_dict["meta_layers"] = np.array([L])
    save_dict["meta_dim"] = np.array([d])
    np.savez_compressed(out_path, **save_dict)


def maybe_load_embeddings(npz_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    data = np.load(npz_path, allow_pickle=False)
    pos_keys = sorted(
        (k for k in data.keys() if k.startswith("X_pos_")),
        key=lambda k: int(k.split("_")[-1]),
    )
    neg_keys = sorted(
        (k for k in data.keys() if k.startswith("X_neg_")),
        key=lambda k: int(k.split("_")[-1]),
    )
    X_pos = [data[k] for k in pos_keys]
    X_neg = [data[k] for k in neg_keys]
    return X_pos, X_neg


def build_xy(
    X_pos_layers: List[np.ndarray], X_neg_layers: List[np.ndarray]
) -> Tuple[List[np.ndarray], np.ndarray]:
    X_layers = [
        np.vstack([X_pos_layers[l], X_neg_layers[l]]) for l in range(len(X_pos_layers))
    ]
    y = np.concatenate(
        [
            np.ones(len(X_pos_layers[0]), dtype=int),
            np.zeros(len(X_neg_layers[0]), dtype=int),
        ],
        axis=0,
    )
    return X_layers, y


def split_indices(
    y: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, stratify=y, random_state=seed
    )
    if val_size <= 0:
        return train_idx, np.array([], dtype=int), test_idx
    train_sub_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size,
        stratify=y[train_idx],
        random_state=seed + 1,
    )
    return train_sub_idx, val_idx, test_idx


def load_or_create_splits(
    split_path: str,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if os.path.isfile(split_path):
        data = np.load(split_path, allow_pickle=False)
        train_idx = data["train_idx"] if "train_idx" in data else None
        train_sub_idx = data["train_sub_idx"] if "train_sub_idx" in data else None
        val_idx = data["val_idx"] if "val_idx" in data else None
        test_idx = data["test_idx"] if "test_idx" in data else None
        if train_sub_idx is None and train_idx is not None:
            train_sub_idx = train_idx
        if val_idx is None:
            val_idx = np.array([], dtype=int)
        if test_idx is None:
            raise ValueError(f"split file missing test_idx: {split_path}")
        if train_idx is None:
            train_idx = (
                np.concatenate([train_sub_idx, val_idx]) if len(val_idx) else train_sub_idx
            )
        return train_idx, train_sub_idx, val_idx, test_idx

    train_sub_idx, val_idx, test_idx = split_indices(y, test_size, val_size, seed)
    train_idx = np.concatenate([train_sub_idx, val_idx]) if len(val_idx) else train_sub_idx
    np.savez(
        split_path,
        train_idx=train_idx,
        train_sub_idx=train_sub_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    return train_idx, train_sub_idx, val_idx, test_idx
