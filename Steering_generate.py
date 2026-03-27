#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Activation steering generation script (STSA focus), refactored to follow the
`GCS/script/steering_emotion.py` template:

    1. load evaluation prompts
    2. sweep steering strengths per concept vector
    3. dump generations to CSV for downstream analysis

ADAPTIVE STEERING ALGORITHM (Steering from Positive to Negative):
This script implements sequential layer-by-layer steering adapted from adversarial attack algorithms,
to steer FROM positive sentiment TOWARDS negative sentiment.

Sequential Steering Algorithm:
    Require: LLM with L layers, classifier P_m, thresholds P_target = 0.01% (negative), P_1 = 90%, instruction x
    1: for l = 1 to L do
    2:     if TestAcc(P_m) > P_1 then                    # Only use high-accuracy layers
    3:         e ← Embedding of x at the l-th layer after steering previous layers
    4:         if P_m(e) > P_target then                 # Steer if still too positive (not negative enough)
    5:             ε ← (sigmoid^(-1)(P_target) - b - w^T·e)   # Adaptive magnitude (negative to steer to negative)
    6:             e ← e + ε·v                            # Apply steering perturbation
    7:     end if
    8: end for

Key Implementation Details:
    - Sequential layer-by-layer steering (hooks are registered in sorted order)
    - Line 2: Only steer layers with test accuracy > P_1 (default 90%)
    - Line 3: Embeddings obtained after previous layers have been steered (via hooks)
    - Line 4: Steer when P_m(e) > P_target (i.e., still too positive, need to steer towards negative)
    - Line 5: Adaptive epsilon (negative value) decreases perturbation to reach target negative probability
    - Line 6: v is the normalized concept vector; negative epsilon steers away from positive sentiment
    
See register_steering_hooks_sequential() for the main implementation.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Project imports (match steering_emotion.py style)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "MS-Thesis"))

from dataset import DataProcessing  # type: ignore  # local module


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PromptSpec:
    label: str
    prompt: str
    source: str
    use_chat_template: Optional[bool] = None
    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass
class SteeringConfig:
    savepath: Path
    model_id: str             # canonical HF repo id (for vector filenames)
    model_source: str         # path or repo id used to load weights
    dataset_tag: str
    cache_dir: Path
    output_csv: Path

    quant_bits: int = 16
    cuda_device: int = 0
    layers: Optional[Sequence[int]] = None
    vector_kind: str = "bagging"

    # GCS vector loading parameters (for vector_kind="gcs")
    gcs_pkl_path: Optional[Path] = None
    gcs_concept: Optional[str] = None

    # Adaptive steering parameters
    target_prob: float = 0.01  # Target probability for negative steering (0.0001 = 0.01% positive, 99.99% negative)
    accuracy_threshold: float = 0.9  # Minimum test accuracy to use a layer (0.9 = 90%)
    bias: float = 0.0
    normalize_l2: bool = True
    
    # Position-based steering parameters
    use_position_coefficients: bool = False  # Enable position-dependent steering strength
    position_decay_rate: float = 0.1  # Decay rate for exponential position weighting (higher = faster decay)

    num_samples: int = 1
    manners: tuple[str, ...] = ("steer_to_negative",)  # Prompt asks for positive, steering moves to negative
    max_tokens: int = 30
    top_p: float = 0.95
    temperature: float = 0.75
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    chat_template_default: bool = True

    seed_base: int = 1


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def set_seed_local(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_model_tag(model_or_dir: str) -> str:
    """Normalize model identifiers (HF repo id or local path) into filename prefix."""
    s = model_or_dir.rstrip("/").strip()
    if not s:
        return ""

    if "models--" in s and "snapshots" in s:
        # Hugging Face cache path: keep portion between models-- and snapshots
        s = s.split("models--", 1)[1]
        s = s.split("snapshots", 1)[0].strip("/-")
    elif s.startswith("models--"):
        s = s[len("models--") :]

    return s.replace("--", "-").replace("/", "-")


def load_model_and_tokenizer(model_source: str, cache_dir: Path, quant_bits: int, cuda: int):
    qcfg = None
    if quant_bits == 8:
        qcfg = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_source, cache_dir=str(cache_dir), use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if quant_bits == 32:
        model = AutoModelForCausalLM.from_pretrained(
            model_source, cache_dir=str(cache_dir), trust_remote_code=True
        )
    elif quant_bits == 8:
        model = AutoModelForCausalLM.from_pretrained(
            model_source, cache_dir=str(cache_dir), quantization_config=qcfg, trust_remote_code=True
        )
    else:  # 16-bit
        model = AutoModelForCausalLM.from_pretrained(
            model_source, cache_dir=str(cache_dir), torch_dtype=torch.bfloat16, trust_remote_code=True
        )

    device = torch.device(f"cuda:{cuda}" if cuda >= 0 and torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer, device


def is_qwen3_model(model) -> bool:
    """Check if the model is a Qwen3 model based on config."""
    model_type = getattr(model.config, "model_type", "").lower()
    return model_type == "qwen3"


def update_generation_config(model, temperature=None, top_p=None, top_k=None, repetition_penalty=None, do_sample=None, verbose=False, **kwargs):
    """Update model's generation_config with the provided parameters.
    
    Some models (like Qwen3) require parameters to be set via generation_config
    rather than passed directly to generate(). This function handles that.
    
    Args:
        model: The model to update
        temperature: Temperature for sampling
        top_p: Top-p (nucleus) sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
        do_sample: Whether to use sampling
        verbose: If True, print the updated config values
        **kwargs: Additional generation config parameters
    """
    if getattr(model, "generation_config", None) is not None:
        updated = []
        if temperature is not None:
            model.generation_config.temperature = temperature
            updated.append(f"temperature={temperature}")
        if top_p is not None:
            model.generation_config.top_p = top_p
            updated.append(f"top_p={top_p}")
        if top_k is not None:
            model.generation_config.top_k = top_k
            updated.append(f"top_k={top_k}")
        if repetition_penalty is not None:
            # Some models might not support repetition_penalty in generation_config
            if hasattr(model.generation_config, 'repetition_penalty'):
                model.generation_config.repetition_penalty = repetition_penalty
                updated.append(f"repetition_penalty={repetition_penalty}")
        if do_sample is not None:
            model.generation_config.do_sample = do_sample
            updated.append(f"do_sample={do_sample}")
        # Update any other kwargs that are valid generation_config attributes
        for key, value in kwargs.items():
            if hasattr(model.generation_config, key):
                setattr(model.generation_config, key, value)
                updated.append(f"{key}={value}")
        
        if verbose and updated:
            print(f"[INFO] Updated generation_config: {', '.join(updated)}")


def infer_num_layers(model) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    if hasattr(model.config, "n_layer"):
        return int(model.config.n_layer)
    return 32


def get_blocks(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "transformer") and hasattr(model.model.transformer, "h"):
        return model.model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Unable to locate transformer blocks.")


def load_test_accuracies(
    savepath: Path,
    model_id: str,
    dataset_tag: str,
    tot_layer: int,
    *,
    vector_kind: str,
) -> list[Optional[float]]:
    """
    Load test accuracies for each layer from the saved Gaussian results.
    
    Accuracy is stored separately in .npz files (not in the .npy sample files).
    Returns None for layers where no accuracy data exists.
    
    Supports:
      - "bagging": gauss_bagging_results.npz
      - "stacking": gauss_kfold_results.npz
      - "singlelr": exp_results/{model_tag}/{dataset}/singlelr_results.npz (with C tuning)
      - "baseline": {model}_{dataset}_baseline_results.npz (C=1.0, no tuning)
      - "gcs": no test accuracies (returns None for all layers)
    """
    base = f"{to_model_tag(model_id)}_{dataset_tag}"

    # GCS pkl vectors have no saved test accuracies
    if vector_kind == "gcs":
        print(f"[GCS] No per-layer test accuracies; set --accuracy-threshold 0 to use all layers")
        return [None] * tot_layer

    # Baseline: load holdout accuracies from baseline_results.npz
    if vector_kind == "baseline":
        results_file = savepath / f"{base}_baseline_results.npz"
        if not results_file.is_file():
            print(f"[WARNING] Baseline results not found: {results_file}")
            return [None] * tot_layer
        data = np.load(results_file)
        if "_acc_holdout" in data:
            acc_vec = data["_acc_holdout"]
            return [float(acc_vec[i]) if i < len(acc_vec) else None for i in range(tot_layer)]
        accuracies: list[Optional[float]] = []
        for layer in range(tot_layer):
            key = f"layer{layer}_acc_holdout"
            accuracies.append(float(data[key]) if key in data else None)
        return accuracies

    # Handle singlelr (tuned C) separately - different file structure
    if vector_kind == "singlelr":
        # Try exp_results path first (run_singlelr.py output)
        results_file = savepath / to_model_tag(model_id) / dataset_tag / "singlelr_results.npz"
        if not results_file.is_file():
            # Fall back to savepath directly
            results_file = savepath / f"{base}_singlelr_results.npz"
        
        if not results_file.is_file():
            print(f"[WARNING] singlelr results file not found: {results_file}")
            return [None] * tot_layer
        
        data = np.load(results_file)
        if "acc_test" in data:
            acc_vector = data["acc_test"]
            return [float(acc_vector[i]) if i < len(acc_vector) else None for i in range(tot_layer)]
        print(f"[WARNING] 'acc_test' not found in {results_file}")
        return [None] * tot_layer
    
    # Map vector_kind to corresponding results file
    results_file_map = {
        "bagging": "gauss_bagging_results.npz",
        "stacking": "gauss_kfold_results.npz",  # kfold is the new name for stacking
    }
    
    if vector_kind not in results_file_map:
        raise ValueError(f"Unsupported vector_kind '{vector_kind}' (choose from {list(results_file_map)} or 'singlelr').")

    results_file = savepath / f"{base}_{results_file_map[vector_kind]}"
    
    if not results_file.is_file():
        print(f"[WARNING] Test accuracy file not found: {results_file}")
        return [None] * tot_layer

    # Load the .npz file containing results
    data = np.load(results_file)
    
    # Try to get the summary vector first (most efficient)
    if "layer_acc_holdout_vector" in data:
        acc_vector = data["layer_acc_holdout_vector"]
        if len(acc_vector) >= tot_layer:
            return [float(acc_vector[i]) for i in range(tot_layer)]
        else:
            print(f"[WARNING] layer_acc_holdout_vector has {len(acc_vector)} elements, expected {tot_layer}")
    
    # Fall back to individual layer keys
    accuracies: list[Optional[float]] = []
    for layer in range(tot_layer):
        # Try to get holdout accuracy first (preferred), fall back to pool accuracy
        acc_key_holdout = f"layer{layer}_acc_holdout"
        acc_key_pool = f"layer{layer}_acc"
        
        if acc_key_holdout in data:
            # Use scalar value or mean if it's an array
            acc_val = data[acc_key_holdout]
            mean_accuracy = float(acc_val.mean() if acc_val.shape else acc_val)
            accuracies.append(mean_accuracy)
        elif acc_key_pool in data:
            # Fall back to pool accuracy
            acc_val = data[acc_key_pool]
            mean_accuracy = float(acc_val.mean() if acc_val.shape else acc_val)
            accuracies.append(mean_accuracy)
        else:
            accuracies.append(None)
    
    return accuracies


def load_concept_vectors(
    savepath: Path,
    model_id: str,
    dataset_tag: str,
    tot_layer: int,
    device,
    *,
    vector_kind: str,
    normalize_l2: bool = True,
    gcs_pkl_path: Optional[Path] = None,
    gcs_concept: Optional[str] = None,
) -> list[Optional[torch.Tensor]]:
    """
    Load concept vectors for steering.
    
    Supports:
      - "bagging": samp/{model}_{dataset}_layer{i}_bagging_samp.npy
      - "stacking": samp/{model}_{dataset}_layer{i}_stacking_samp.npy  
      - "singlelr": exp_results/{model_tag}/{dataset}/singlelr_results.npz (with C tuning)
      - "baseline": {model}_{dataset}_baseline_results.npz (C=1.0, GCS-comparable)
      - "gcs": GCS pkl files ({concept: {layer: coef_vector_or_list}})
    """
    base = f"{to_model_tag(model_id)}_{dataset_tag}"

    # ----- GCS vectors (pkl format) -----
    if vector_kind == "gcs":
        import pickle
        if gcs_pkl_path is None:
            raise ValueError("vector_kind='gcs' requires --gcs-pkl path.")
        pkl_path = Path(gcs_pkl_path)
        if not pkl_path.is_file():
            raise FileNotFoundError(f"GCS pkl not found: {pkl_path}")
        with open(pkl_path, "rb") as f:
            gcs_data = pickle.load(f)
        available_concepts = list(gcs_data.keys())
        if gcs_concept is not None:
            if gcs_concept not in gcs_data:
                raise KeyError(f"Concept '{gcs_concept}' not in pkl. Available: {available_concepts}")
            concept_data = gcs_data[gcs_concept]
        else:
            concept_key = available_concepts[0]
            print(f"[GCS] Using first concept: '{concept_key}'")
            concept_data = gcs_data[concept_key]
        vectors: list[Optional[torch.Tensor]] = []
        for layer in range(tot_layer):
            if layer not in concept_data or concept_data[layer] is None:
                vectors.append(None)
                continue
            vec_data = concept_data[layer]
            if isinstance(vec_data, list):
                arrs = [np.asarray(v, dtype=np.float32) for v in vec_data if v is not None]
                if not arrs:
                    vectors.append(None)
                    continue
                mean_arr = np.mean(np.stack(arrs, axis=0), axis=0)
            elif isinstance(vec_data, np.ndarray):
                mean_arr = vec_data.astype(np.float32)
                if mean_arr.ndim == 2:
                    mean_arr = mean_arr.mean(axis=0)
            else:
                vectors.append(None)
                continue
            vec = torch.tensor(mean_arr, device=device)
            if normalize_l2:
                vec_norm = torch.norm(vec, p=2)
                if vec_norm > 1e-8:
                    vec = vec / vec_norm
            vectors.append(vec)
        print(f"[GCS] Loaded {sum(1 for v in vectors if v is not None)}/{tot_layer} vectors")
        return vectors

    # ----- baseline (single LR, C=1.0, no tuning) -----
    if vector_kind == "baseline":
        results_file = savepath / f"{base}_baseline_results.npz"
        if not results_file.is_file():
            raise FileNotFoundError(
                f"Baseline results not found: {results_file}\n"
                f"Run: python generate_gcs_baseline_vectors.py --models {model_id} --datasets {dataset_tag}"
            )
        data = np.load(results_file)
        vectors: list[Optional[torch.Tensor]] = []
        for layer in range(tot_layer):
            key = f"layer{layer}_w"
            if key not in data:
                vectors.append(None)
                continue
            wb = data[key]  # (1, d+1), last col is bias
            w = wb[0, :-1].astype(np.float32)
            vec = torch.tensor(w, device=device)
            if normalize_l2:
                vec_norm = torch.norm(vec, p=2)
                if vec_norm > 1e-8:
                    vec = vec / vec_norm
            vectors.append(vec)
        print(f"[Baseline] Loaded {sum(1 for v in vectors if v is not None)}/{tot_layer} vectors from {results_file}")
        return vectors

    # ----- singlelr (tuned C) -----
    if vector_kind == "singlelr":
        # Try exp_results path first (run_singlelr.py output)
        results_file = savepath / to_model_tag(model_id) / dataset_tag / "singlelr_results.npz"
        if not results_file.is_file():
            # Fall back to savepath directly
            results_file = savepath / f"{base}_singlelr_results.npz"
        
        if not results_file.is_file():
            print(f"[WARNING] singlelr results file not found: {results_file}")
            return [None] * tot_layer
        
        data = np.load(results_file)
        
        # singlelr_results.npz contains: W (L, d), B (L,), concept (L, d+1)
        if "concept" in data:
            concept = data["concept"]  # (L, d+1)
        elif "W" in data and "B" in data:
            W = data["W"]  # (L, d)
            B = data["B"]  # (L,)
            concept = np.hstack([W, B.reshape(-1, 1)])  # (L, d+1)
        else:
            print(f"[WARNING] 'concept' or 'W'+'B' not found in {results_file}")
            return [None] * tot_layer
        
        vectors: list[Optional[torch.Tensor]] = []
        for layer in range(tot_layer):
            if layer >= len(concept):
                vectors.append(None)
                continue
            # concept[layer] is (d+1,) where last element is bias
            vec = torch.tensor(concept[layer, :-1].astype(np.float32), device=device)
            
            # Normalize to L2 norm = 1
            if normalize_l2:
                vec_norm = torch.norm(vec, p=2)
                if vec_norm > 1e-8:
                    vec = vec / vec_norm
            
            vectors.append(vec)
        return vectors
    
    # Original logic for bagging/stacking
    suffix_map = {
        "bagging": "bagging_samp.npy",
        "stacking": "stacking_samp.npy",
    }
    if vector_kind not in suffix_map:
        raise ValueError(f"Unsupported vector_kind '{vector_kind}' (choose from {list(suffix_map)}, 'singlelr', 'baseline', 'gcs').")

    samp_dir = savepath / "samp"
    suffix = suffix_map[vector_kind]

    vectors: list[Optional[torch.Tensor]] = []
    for layer in range(tot_layer):
        file_path = samp_dir / f"{base}_layer{layer}_{suffix}"
        if not file_path.is_file():
            vectors.append(None)
            continue
        arr = np.load(file_path)  # (N, d+1) or (1, d+1)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"{file_path} should have shape (N, d+1); got {arr.shape}")
        mean_row = arr.mean(axis=0)
        vec = torch.tensor(mean_row[:-1].astype(np.float32), device=device)
        
        # Normalize to L2 norm = 1
        if normalize_l2:
            vec_norm = torch.norm(vec, p=2)
            if vec_norm > 1e-8:
                vec = vec / vec_norm
        
        vectors.append(vec)
    return vectors


def collapse_to_single_line(text: str) -> str:
    """Condense whitespace so the string fits on a single line for logging/export."""
    return " ".join(text.strip().split())


def compute_adaptive_epsilon(
    embedding: torch.Tensor,
    concept_vector: torch.Tensor,
    bias: float,
    target_prob: float,
) -> torch.Tensor:
    """
    Compute adaptive perturbation magnitude to steer towards target probability.
    
    For negative steering (steering away from positive/hate):
    - target_prob should be LOW (e.g., 0.0001) to make output negative/non-hate
    - Steers from positive (high P_m) towards negative (low P_m)
    - When current P_m is high (positive/hate), epsilon will be negative to steer away
    
    For positive steering (steering towards positive/hate):
    - target_prob should be HIGH (e.g., 0.99) to make output positive/hate
    - Steers from negative (low P_m) towards positive (high P_m)
    - When current P_m is low (negative/non-hate), epsilon will be positive to steer towards
    
    Formula: ε = I(should_steer) · (sigmoid^(-1)(P_target) - b - w^T·e)
    
    Args:
        embedding: Current embedding (batch_size, hidden_dim)
        concept_vector: Normalized concept vector (hidden_dim,) - this is w/||w||
        bias: Bias term for probability calculation
        target_prob: Target probability (e.g., 0.0001 for negative steering, 0.99 for positive steering)
    
    Returns:
        epsilon: Adaptive perturbation magnitude (negative to steer away, positive to steer towards)
    """
    # Convert target probability to logit: sigmoid^(-1)(P_target)
    logit_target = torch.log(torch.tensor(target_prob / (1 - target_prob), device=embedding.device))
    
    # Since concept_vector is already L2 normalized, ||w|| = 1
    # Compute w^T·e (dot product)
    w_dot_e = torch.sum(embedding * concept_vector, dim=-1, keepdim=True)
    
    # Compute current logit: b + w^T·e
    current_logit = bias + w_dot_e
    current_prob = torch.sigmoid(current_logit)
    
    # Compute epsilon: (logit_target - current_logit)
    # This moves the embedding towards the target probability
    # For target_prob=0.0001 (negative): logit_target ≈ -9.21
    # If current_logit is large positive (P_m ≈ 1.0), epsilon will be large negative
    # Negative epsilon steers away from positive sentiment towards negative
    epsilon = logit_target - current_logit
    
    # Only apply if we haven't reached the target yet
    # For negative steering (target_prob < 0.5): steer when current_prob > target_prob (still too positive/non-hate)
    # For positive steering (target_prob > 0.5): steer when current_prob < target_prob (still too negative/non-hate)
    should_steer = (current_prob < target_prob if target_prob > 0.5 
                   else current_prob > target_prob).float()
    epsilon = epsilon * should_steer
    
    return epsilon


def register_steering_hooks_sequential(
    model,
    vectors: Sequence[Optional[torch.Tensor]],
    layer_ids: Sequence[int],
    *,
    target_prob: float = 0.0001,
    bias: float = 0.0,
    test_accuracies: Optional[Sequence[Optional[float]]] = None,
    accuracy_threshold: float = 0.9,
    use_position_coefficients: bool = False,
    position_decay_rate: float = 0.1,
    verbose: bool = True,
):
    """
    Register adaptive steering hooks following sequential layer-by-layer steering algorithm.
    
    Adaptive Steering Algorithm:
        - Epsilon computed dynamically per layer to reach target_prob
        - Algorithm:
            for l = 1 to L do:
                if TestAcc(P_m) > P_1 then:
                    e ← Embedding at l-th layer after steering previous layers
                    if should_steer(e, P_target) then:
                        ε ← (sigmoid^(-1)(P_target) - b - w^T·e)  # Adaptive magnitude
                        e ← e + ε·v                        # Apply perturbation
                        - For target_prob < 0.5: negative epsilon steers away from positive/hate
                        - For target_prob > 0.5: positive epsilon steers towards positive/hate
    
    Position-based Steering (optional):
        - When use_position_coefficients=True, applies position-dependent weights
        - Uses exponential decay: weight(pos) = exp(-position_decay_rate * pos)
        - Earlier tokens receive stronger steering, later tokens receive weaker steering
    
    Args:
        target_prob: Target probability for adaptive steering (0.0001 = steer away from hate, 0.99 = steer towards hate)
        bias: Bias term for probability calculation (default 0.0)
        test_accuracies: Test accuracies for each layer (for P_1 check)
        accuracy_threshold: Threshold P_1 for test accuracy (default 0.9)
        use_position_coefficients: Enable position-dependent steering strength (default False)
        position_decay_rate: Decay rate for exponential weighting (default 0.1)
        verbose: If True, print epsilon values for first token only (default True)
    """
    blocks = get_blocks(model)
    handles = []
    
    # Sort layers to ensure sequential processing
    sorted_layers = sorted(layer_ids)
    
    # Flag to print only for first forward pass (first token generation)
    first_pass = {"done": False}
    
    print(f"\n[Adaptive Steering] Sequential layer-by-layer steering:")
    print(f"  - Target layers: {sorted_layers}")
    print(f"  - P_target (target_prob): {target_prob}")
    print(f"  - P_1 (accuracy_threshold): {accuracy_threshold}")
    print(f"  - Position-based steering: {use_position_coefficients}")
    if use_position_coefficients:
        print(f"  - Position decay rate: {position_decay_rate}")
    
    # Track generation step for position-based steering
    # This counter increments each time we generate a new token
    generation_step = {"count": 0}

    for layer in sorted_layers:
        if not (0 <= layer < len(blocks)):
            raise ValueError(f"Layer {layer} out of range; model has {len(blocks)} layers.")
        vec = vectors[layer]
        if vec is None:
            raise FileNotFoundError(f"Missing concept vector for layer {layer}.")
        
        # Check if TestAcc(P_m) > P_1
        if test_accuracies is not None and test_accuracies[layer] is not None:
            if test_accuracies[layer] <= accuracy_threshold:
                print(f"  [Layer {layer}] SKIP: TestAcc={test_accuracies[layer]:.4f} <= P_1={accuracy_threshold}")
                continue
            else:
                print(f"  [Layer {layer}] TestAcc={test_accuracies[layer]:.4f} > P_1 ✓")

        def hook_fn(_module, _inputs, output, *, layer_idx=layer, base_vec=vec, last_layer=sorted_layers[-1], all_layers=sorted_layers):
            if isinstance(output, tuple):
                hidden, others = output[0], output[1:]
            else:
                hidden, others = output, None

            # Only steer the last token (the one being generated)
            hidden = hidden.clone()
            last = hidden[:, -1, :]
            steer_vec = base_vec.to(device=last.device, dtype=last.dtype)
            if steer_vec.shape[-1] != last.shape[-1]:
                raise ValueError(
                    f"Layer {layer_idx}: vector dim {steer_vec.shape[-1]} != hidden dim {last.shape[-1]}"
                )

            # Compute current probability P_m(e) for last token only
            w_dot_e = torch.sum(last * steer_vec, dim=-1, keepdim=True)
            current_logit = bias + w_dot_e
            current_prob = torch.sigmoid(current_logit)
            
            # Steer if current probability hasn't reached target yet
            # For target_prob > 0.5 (steer towards positive/hate): steer when current_prob < target_prob (still too low)
            # For target_prob < 0.5 (steer towards negative/non-hate): steer when current_prob > target_prob (still too high)
            should_steer = (current_prob.item() < target_prob if target_prob > 0.5 
                           else current_prob.item() > target_prob)
            
            # Initialize epsilon for DEBUG output
            epsilon = None
            
            if should_steer:
                # Compute adaptive epsilon to reach target_prob
                epsilon = compute_adaptive_epsilon(
                    embedding=last,
                    concept_vector=steer_vec,
                    bias=bias,
                    target_prob=target_prob,
                )
                
                # Apply position-based decay if enabled
                # Decay based on how many tokens have been generated so far
                if use_position_coefficients:
                    # Compute exponential decay with more moderate scaling
                    exp_weight = 1.5*torch.exp(torch.tensor(-position_decay_rate * generation_step["count"], 
                                                           device=epsilon.device, dtype=epsilon.dtype))
                    # Apply saturation: use max(exp_weight, 0.5) to allow some decay but maintain minimum steering
                    position_weight = torch.max(exp_weight, torch.tensor(1, device=epsilon.device, dtype=epsilon.dtype))
                    epsilon = epsilon * position_weight
                    
                    # Print with position weight info
                    if verbose and layer_idx == last_layer:
                        epsilon_val = epsilon.item() if epsilon.numel() == 1 else epsilon.mean().item()
                        print(f"  [Step {generation_step['count']}][Layer {layer_idx}] P_m={current_prob.item():.4f}, "
                              f"epsilon={epsilon_val:.4f}, weight={position_weight.item():.4f}, STEERING")
                else:
                    # Print epsilon values only for last layer (to avoid spam)
                    if verbose and layer_idx == last_layer:
                        epsilon_val = epsilon.item() if epsilon.numel() == 1 else epsilon.mean().item()
                        print(f"  [Step {generation_step['count']}][Layer {layer_idx}] P_m={current_prob.item():.4f}, "
                              f"epsilon={epsilon_val:.4f}, STEERING")
                
                # Apply perturbation: epsilon already has correct sign
                new_last = last + epsilon * steer_vec
                
                # DEBUG: Print steering effect for last layer
                if verbose and layer_idx == last_layer:
                    new_w_dot_e = torch.sum(new_last * steer_vec, dim=-1, keepdim=True)
                    new_logit = bias + new_w_dot_e
                    new_prob = torch.sigmoid(new_logit)
                    logit_change = (new_logit - current_logit).item()
                    print(f"    -> After steering: P_m={new_prob.item():.4f}, logit={new_logit.item():.2f} "
                          f"(change={logit_change:.2f}), w_dot_e={new_w_dot_e.item():.2f}")
            else:
                # Already at target probability, no steering needed
                if verbose and layer_idx == last_layer:
                    print(f"  [Step {generation_step['count']}][Layer {layer_idx}] P_m={current_prob.item():.4f}, "
                          f"SKIP (already at target)")
                new_last = last
            
            hidden[:, -1, :] = new_last
            
            # DEBUG: Print info for all layers in Step 0 to verify steering is applied
            if verbose and generation_step["count"] == 0:  # Only for first token to avoid spam
                final_w_dot_e = torch.sum(new_last * steer_vec, dim=-1, keepdim=True)
                final_logit = bias + final_w_dot_e
                final_prob = torch.sigmoid(final_logit)
                # Get epsilon value (already computed if should_steer is True)
                if epsilon is not None:
                    epsilon_val = epsilon.item() if epsilon.numel() == 1 else epsilon.mean().item()
                else:
                    epsilon_val = 0.0
                print(f"  [DEBUG Step 0][Layer {layer_idx}] Final P_m={final_prob.item():.4f}, "
                      f"epsilon={epsilon_val:.2f}, steered={should_steer}")
            
            # Increment generation step counter after last layer processes
            # This tracks how many tokens have been generated
            if layer_idx == last_layer:
                generation_step["count"] += 1

            return (hidden, *others) if others is not None else hidden

        handles.append(blocks[layer].register_forward_hook(hook_fn))

    return handles


def register_steering_hooks(
    model,
    vectors: Sequence[Optional[torch.Tensor]],
    layer_ids: Iterable[int],
    *,
    target_prob: float = 0.99,
    bias: float = 0.0,
    test_accuracies: Optional[Sequence[Optional[float]]] = None,
    accuracy_threshold: float = 0.9,
    use_position_coefficients: bool = False,
    position_decay_rate: float = 0.1,
    verbose: bool = True,
):
    """
    Register adaptive steering hooks (wrapper for register_steering_hooks_sequential).
    
    Args:
        target_prob: Target probability for adaptive steering (0.99 = positive, 0.01 = negative)
        bias: Bias term for probability calculation (default 0.0)
        test_accuracies: Test accuracies for each layer (for P_1 check)
        accuracy_threshold: Threshold P_1 for test accuracy (default 0.9)
        use_position_coefficients: Enable position-dependent steering strength (default False)
        position_decay_rate: Decay rate for exponential weighting (default 0.1)
        verbose: If True, print epsilon values for first token only (default True)
    """
    return register_steering_hooks_sequential(
        model=model,
        vectors=vectors,
        layer_ids=list(layer_ids),
        target_prob=target_prob,
        bias=bias,
        test_accuracies=test_accuracies,
        accuracy_threshold=accuracy_threshold,
        use_position_coefficients=use_position_coefficients,
        position_decay_rate=position_decay_rate,
        verbose=verbose,
    )


def encode_chat(tokenizer, prompt: str, device):
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt")
    return enc["input_ids"].to(device)


def detect_style_change(
    model, tokenizer, device, original_text: str, steered_text: str, manner: str
) -> str:
    """
    Use the LLM to determine if there's a style change between original and steered text.
    
    Dataset-specific style change detection prompts:
    - STSA: sentiment change (positive ↔ negative)
    - sarcasm: tone change (sarcastic ↔ sincere)
    - hatexplain: toxicity change (hate ↔ non-hate)
    - counterfact: factual accuracy change (false ↔ true claim)
    
    Returns:
        "yes" if style change is detected, "no" otherwise
    """
    # =========================================================================
    # STSA (Sentiment)
    # =========================================================================
    if manner == "steer_to_negative" or manner == "neutral_to_negative":
        evaluation_prompt = (
            f"Compare the following two texts and determine if the sentiment changed from neutral/positive to negative.\n\n"
            f"Text 1: {original_text}\n\n"
            f"Text 2: {steered_text}\n\n"
            f"Answer with only 'yes' or 'no': Did the sentiment become more negative?"
        )
    elif manner == "steer_to_positive" or manner == "neutral_to_positive":
        evaluation_prompt = (
            f"Compare the following two texts and determine if the sentiment changed from neutral/negative to positive.\n\n"
            f"Text 1: {original_text}\n\n"
            f"Text 2: {steered_text}\n\n"
            f"Answer with only 'yes' or 'no': Did the sentiment become more positive?"
        )
    
    # =========================================================================
    # Sarcasm
    # =========================================================================
    elif manner == "neutral_to_sarcastic":
        evaluation_prompt = (
            f"Compare the following two texts and determine if the tone changed from sincere/neutral to sarcastic or ironic.\n\n"
            f"Text 1: {original_text}\n\n"
            f"Text 2: {steered_text}\n\n"
            f"Answer with only 'yes' or 'no': Did the text become sarcastic or ironic?"
        )
    elif manner == "neutral_to_sincere":
        evaluation_prompt = (
            f"Compare the following two texts and determine if the tone changed from sarcastic/ironic to more sincere and straightforward.\n\n"
            f"Text 1: {original_text}\n\n"
            f"Text 2: {steered_text}\n\n"
            f"Answer with only 'yes' or 'no': Did the text become more sincere and less sarcastic?"
        )
    
    # =========================================================================
    # HateXplain (Hate Speech)
    # =========================================================================
    elif manner == "steer_to_hate" or manner == "neutral_to_hate":
        evaluation_prompt = (
            f"Compare the following two texts and determine if the content changed from neutral/respectful to hateful, offensive, or harmful.\n\n"
            f"Text 1: {original_text}\n\n"
            f"Text 2: {steered_text}\n\n"
            f"Answer with only 'yes' or 'no': Did the text become hateful or offensive?"
        )
    elif manner == "steer_to_non_hate" or manner == "neutral_to_nonhate":
        evaluation_prompt = (
            f"Compare the following two texts and determine if the content changed from hateful/offensive to more neutral and respectful.\n\n"
            f"Text 1: {original_text}\n\n"
            f"Text 2: {steered_text}\n\n"
            f"Answer with only 'yes' or 'no': Did the text become less hateful and more respectful?"
        )
    
    # =========================================================================
    # Counterfact (Factual Accuracy / Truthfulness)
    # Evaluate if steering achieved the target: lie→truth or truth→lie
    # =========================================================================
    elif manner == "lie_to_truth":
        # Original was prompted to be FALSE, steered should be TRUE
        # Check: Is the steered output factually TRUE?
        evaluation_prompt = (
            f"Evaluate the following two statements about factual accuracy.\n\n"
            f"Statement 1 (generated without steering): {original_text}\n\n"
            f"Statement 2 (generated with steering towards truth): {steered_text}\n\n"
            f"The goal was to steer from false claims towards truthful statements.\n"
            f"Answer with only 'yes' or 'no': Is Statement 2 more factually accurate or truthful than Statement 1?"
        )
    elif manner == "truth_to_lie":
        # Original was prompted to be TRUE, steered should be FALSE
        # Check: Is the steered output factually FALSE?
        evaluation_prompt = (
            f"Evaluate the following two statements about factual accuracy.\n\n"
            f"Statement 1 (generated without steering): {original_text}\n\n"
            f"Statement 2 (generated with steering towards false claims): {steered_text}\n\n"
            f"The goal was to steer from truthful statements towards false claims.\n"
            f"Answer with only 'yes' or 'no': Is Statement 2 less factually accurate or more false than Statement 1?"
        )
    
    # =========================================================================
    # Default fallback
    # =========================================================================
    else:
        evaluation_prompt = (
            f"Compare the following two texts and determine if there is a noticeable style change "
            f"(particularly in terms of being more {manner}).\n\n"
            f"Text 1: {original_text}\n\n"
            f"Text 2: {steered_text}\n\n"
            f"Answer with only 'yes' or 'no': Is there a style change?"
        )
    
    # Use chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": evaluation_prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt")
    else:
        enc = tokenizer(evaluation_prompt, return_tensors="pt")
    
    input_ids = enc["input_ids"].to(device)
    
    # Generate with deterministic settings for consistency
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=10,  # Only need "yes" or "no"
            do_sample=False,  # Deterministic
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    
    # Parse response to extract yes/no
    if "yes" in response:
        return "yes"
    elif "no" in response:
        return "no"
    else:
        # If unclear, check first word
        first_word = response.split()[0] if response.split() else ""
        return "yes" if first_word == "yes" else "no"


# ---------------------------------------------------------------------------
# Prompt preparation (Multi-dataset support)
# ---------------------------------------------------------------------------

# Dataset-specific prompt templates
# concept_direction: which direction the concept vector (w) points to
#   - For logistic regression: sigmoid(w·x + b) → 1 means positive class
# steering_configs: mapping from manner to target_prob
#   - Higher target_prob (e.g., 0.9999) steers TOWARDS concept_direction
#   - Lower target_prob (e.g., 0.0001) steers AWAY FROM concept_direction
DATASET_TEMPLATES = {
    "STSA": {
        # Concept vector: w → positive sentiment (label=1=positive, label=0=negative)
        "concept_direction": "positive",
        "instruction": "Briefly describe your thoughts on a movie in one sentence (15 words max), keeping a balanced and neutral tone.",
        "steering_configs": {
            "neutral_to_negative": {
                "target_prob": 0.0001,
                "description": "Prompt: neutral → Steer AWAY from positive → towards negative",
            },
            "neutral_to_positive": {
                "target_prob": 0.9999,
                "description": "Prompt: neutral → Steer TOWARDS positive",
            },
        },
        "default_manner": "neutral_to_negative",
    },
    "sarcasm": {
        # Concept vector: w → sarcastic (label=1=sarcastic, label=0=sincere)
        "concept_direction": "sarcastic",
        # Use the headline from dataset and ask for a neutral comment, then steer to sarcastic/sincere
        "instruction": "Read this headline and write a brief, neutral one-sentence comment (15 words max):\n\"{text}\"",
        "steering_configs": {
            "neutral_to_sarcastic": {
                "target_prob": 0.9999,
                "description": "Prompt: reaction to headline → Steer TOWARDS sarcastic tone",
            },
            "neutral_to_sincere": {
                "target_prob": 0.0001,
                "description": "Prompt: reaction to headline → Steer AWAY from sarcastic → towards sincere",
            },
        },
        "default_manner": "neutral_to_sarcastic",
    },
    "hatexplain": {
        # Concept vector: w → hate (label=0=hate as positive class in code, label=1,2=non-hate)
        "concept_direction": "hate",
        "instruction": "Write a brief, neutral comment on a social topic in 15 words or less.",
        "steering_configs": {
            "neutral_to_hate": {
                "target_prob": 0.9999,
                "description": "Prompt: neutral → Steer TOWARDS hate",
            },
            "neutral_to_nonhate": {
                "target_prob": 0.0001,
                "description": "Prompt: neutral → Steer AWAY from hate → towards non-hate",
            },
        },
        "default_manner": "neutral_to_hate",
    },
    "counterfact": {
        # Concept vector: w → fact/truth (label=1=true, label=0=false)
        "concept_direction": "fact/truth",
        "instruction": None,  # Uses manner-specific instructions below
        "steering_configs": {
            "lie_to_truth": {
                "target_prob": 0.9999,
                "description": "Prompt: lie → Steer TOWARDS truth/fact",
                "instruction": "Make a false claim about a famous person or place by changing a key fact. Only output the false statement in 15 words or less.",
            },
            "truth_to_lie": {
                "target_prob": 0.0001,
                "description": "Prompt: truth → Steer AWAY from truth → towards lie",
                "instruction": "State a well-known true fact about a famous person or place. Only output the true statement in 15 words or less.",
            },
        },
        "default_manner": "lie_to_truth",
    },
}

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task.\n"
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "{instruction}\n\n"
    "### Response:"
)


def load_stsa_samples(
    datapath: Path,
    pos_limit: Optional[int],
    neg_limit: Optional[int],
    *,
    seed: int,
) -> tuple[list[str], list[str]]:
    dp = DataProcessing(data_path=str(datapath), data_name="STSA", noise="non-noise")
    pos_samples, neg_samples, _prompt, _cot = dp.dispacher()

    rng = random.Random(seed)

    def sample(items: Sequence[str], limit: Optional[int]) -> list[str]:
        if limit is None or limit <= 0 or len(items) <= limit:
            return list(items)
        return rng.sample(list(items), limit)

    pos_subset = sample(pos_samples, pos_limit)
    neg_subset = sample(neg_samples, neg_limit)
    return pos_subset, neg_subset


def load_sarcasm_samples(
    datapath: Path,
    pos_limit: Optional[int],
    neg_limit: Optional[int],
    *,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Load sarcasm dataset samples for steering evaluation.
    
    Expected format: JSONL with 'headline' and 'is_sarcastic' fields.
    """
    import json
    
    pos_samples = []
    neg_samples = []
    
    # Read JSONL file
    with open(datapath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            headline = data.get('headline', '').strip()
            if not headline:
                continue
            
            is_sarcastic = data.get('is_sarcastic', 0)
            if is_sarcastic == 1:
                pos_samples.append(headline)
            else:
                neg_samples.append(headline)
    
    rng = random.Random(seed)
    
    def sample(items: Sequence[str], limit: Optional[int]) -> list[str]:
        if limit is None or limit <= 0 or len(items) <= limit:
            return list(items)
        return rng.sample(list(items), limit)
    
    pos_subset = sample(pos_samples, pos_limit)
    neg_subset = sample(neg_samples, neg_limit)
    return pos_subset, neg_subset


def load_hatexplain_samples(
    datapath: Optional[Path],
    pos_limit: Optional[int],
    neg_limit: Optional[int],
    *,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Load hatexplain dataset samples for steering evaluation.
    
    Strategy: label=0 (hate) as positive samples, label=1 (normal) + label=2 (offensive) as negative samples.
    If datapath is None or empty, loads from Hugging Face.
    """
    from collections import Counter
    import os
    from datasets import load_dataset, load_from_disk
    
    # Set cache location
    os.environ['HF_DATASETS_CACHE'] = '/projects/p32759/.cache/hf/datasets'
    
    # Load dataset
    if datapath and datapath.exists() and datapath.is_dir():
        print(f"Loading hatexplain dataset from local path: {datapath}")
        dataset = load_from_disk(str(datapath))
    else:
        # Load from Hugging Face (will use cache if already downloaded)
        print(f"Loading hatexplain dataset from Hugging Face (cache: /projects/p32759/.cache/hf/datasets)")
        dataset = load_dataset("hatexplain", trust_remote_code=True, 
                              cache_dir="/projects/p32759/.cache/hf/datasets")
    
    hate_texts = []      # label=0 (hate) - positive samples
    non_hate_texts = []  # label=1 (normal) + label=2 (offensive) - negative samples
    
    # Process train split
    for sample in dataset.get('train', []):
        post_tokens = sample.get('post_tokens', [])
        if not post_tokens:
            continue
            
        # Merge tokens into text
        text = ' '.join(post_tokens)
        
        # Get annotator labels, use majority voting
        annotators = sample.get('annotators', {})
        if annotators and 'label' in annotators:
            labels = annotators['label']  # This is a list, e.g., [0, 2, 2]
            if labels:
                label_counts = Counter(labels)
                most_common_label = label_counts.most_common(1)[0][0]
                
                # Classify: 0=hate (positive), 1 or 2=non-hate (negative)
                if most_common_label == 0:
                    hate_texts.append(text)
                else:  # label=1 or 2
                    non_hate_texts.append(text)
    
    rng = random.Random(seed)
    
    def sample(items: Sequence[str], limit: Optional[int]) -> list[str]:
        if limit is None or limit <= 0 or len(items) <= limit:
            return list(items)
        return rng.sample(list(items), limit)
    
    pos_subset = sample(hate_texts, pos_limit)
    neg_subset = sample(non_hate_texts, neg_limit)
    return pos_subset, neg_subset


def load_counterfact_samples(
    datapath: Path,
    pos_limit: Optional[int],
    neg_limit: Optional[int],
    *,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Load counterfact dataset samples for steering evaluation."""
    import pandas as pd
    
    df = pd.read_csv(datapath)
    
    # label=1 is true fact, label=0 is counterfact (false)
    pos_samples = df[df['label'] == 1]['statement'].tolist()
    neg_samples = df[df['label'] == 0]['statement'].tolist()
    
    rng = random.Random(seed)
    
    def sample(items: Sequence[str], limit: Optional[int]) -> list[str]:
        if limit is None or limit <= 0 or len(items) <= limit:
            return list(items)
        return rng.sample(list(items), limit)
    
    pos_subset = sample(pos_samples, pos_limit)
    neg_subset = sample(neg_samples, neg_limit)
    
    return pos_subset, neg_subset


def load_dataset_samples(
    dataset_tag: str,
    datapath: Optional[Path],
    pos_limit: Optional[int],
    neg_limit: Optional[int],
    *,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Load samples from different datasets based on dataset_tag."""
    if dataset_tag.upper() == "STSA":
        if datapath is None:
            raise ValueError("STSA dataset requires datapath to be specified")
        return load_stsa_samples(datapath, pos_limit, neg_limit, seed=seed)
    elif dataset_tag.lower() == "sarcasm":
        if datapath is None:
            raise ValueError("sarcasm dataset requires datapath to be specified")
        return load_sarcasm_samples(datapath, pos_limit, neg_limit, seed=seed)
    elif dataset_tag.lower() == "hatexplain":
        return load_hatexplain_samples(datapath, pos_limit, neg_limit, seed=seed)
    elif dataset_tag.lower() == "counterfact":
        if datapath is None:
            raise ValueError("counterfact dataset requires datapath to be specified")
        return load_counterfact_samples(datapath, pos_limit, neg_limit, seed=seed)
    else:
        raise ValueError(f"Unsupported dataset_tag: {dataset_tag}. Supported: STSA, sarcasm, hatexplain, counterfact")


def get_dataset_template(dataset_tag: str) -> dict:
    """Get dataset-specific template configuration."""
    if dataset_tag.upper() == "STSA":
        return DATASET_TEMPLATES["STSA"]
    elif dataset_tag.lower() == "sarcasm":
        return DATASET_TEMPLATES["sarcasm"]
    elif dataset_tag.lower() == "hatexplain":
        return DATASET_TEMPLATES["hatexplain"]
    elif dataset_tag.lower() == "counterfact":
        return DATASET_TEMPLATES["counterfact"]
    else:
        raise ValueError(f"Unsupported dataset_tag: {dataset_tag}. Supported: STSA, sarcasm, hatexplain, counterfact")


def get_target_prob_for_manner(dataset_tag: str, manner: str) -> float:
    """Get the target_prob for a specific manner in a dataset.
    
    Args:
        dataset_tag: The dataset identifier
        manner: The steering manner (e.g., 'neutral_to_negative')
    
    Returns:
        The target probability for the specified manner
    """
    template = get_dataset_template(dataset_tag)
    steering_configs = template.get("steering_configs", {})
    if manner in steering_configs:
        return steering_configs[manner].get("target_prob", 0.5)
    # Fallback: return 0.5 if manner not found
    print(f"Warning: manner '{manner}' not found in {dataset_tag}, using target_prob=0.5")
    return 0.5


def get_default_target_prob(dataset_tag: str) -> float:
    """Get the default target_prob for a dataset (using default manner)."""
    template = get_dataset_template(dataset_tag)
    default_manner = template.get("default_manner", "")
    steering_configs = template.get("steering_configs", {})
    if default_manner and default_manner in steering_configs:
        return steering_configs[default_manner].get("target_prob", 0.5)
    # Fallback for backward compatibility
    return template.get("target_prob", 0.5)


def get_default_manners(dataset_tag: str) -> list[str]:
    """Get the default manners for a dataset (returns the default manner)."""
    template = get_dataset_template(dataset_tag)
    default_manner = template.get("default_manner", "")
    if default_manner:
        return [default_manner]
    # Fallback for backward compatibility
    return template.get("manners", ["neutral"])


def get_all_manners(dataset_tag: str) -> list[str]:
    """Get all available manners for a dataset."""
    template = get_dataset_template(dataset_tag)
    steering_configs = template.get("steering_configs", {})
    return list(steering_configs.keys())


def print_dataset_config_summary(dataset_tag: str, selected_manners: Optional[Sequence[str]] = None) -> None:
    """Print a summary of the dataset configuration for debugging/logging."""
    template = get_dataset_template(dataset_tag)
    steering_configs = template.get("steering_configs", {})
    default_manner = template.get("default_manner", "")
    
    print(f"\n[Dataset Config] {dataset_tag}")
    print(f"  Concept direction: w → {template.get('concept_direction', 'unknown')}")
    print(f"  Available steering configurations:")
    
    for manner, config in steering_configs.items():
        is_default = " (default)" if manner == default_manner else ""
        is_selected = " ✓" if selected_manners and manner in selected_manners else ""
        target_prob = config.get("target_prob", "N/A")
        description = config.get("description", "")
        direction = "TOWARDS" if target_prob > 0.5 else "AWAY FROM"
        print(f"    • {manner}{is_default}{is_selected}")
        print(f"      target_prob={target_prob} ({direction} concept direction)")
        print(f"      {description}")


def build_dataset_prompt_specs(
    sentences: Sequence[str],
    manners: Sequence[str],
    dataset_tag: str,
    pos_sentences: Optional[Sequence[str]] = None,
    neg_sentences: Optional[Sequence[str]] = None,
) -> list[PromptSpec]:
    """Build prompt specifications for different datasets.
    
    Note: For neutral prompts that don't use {text}, we still iterate over sentences
    to generate multiple independent samples for evaluation. Each sample uses the same
    prompt but will produce different outputs due to sampling randomness.
    
    Supports manner-specific instructions: if a steering_config has an "instruction" field,
    it will be used instead of the main template instruction.
    
    Supports manner-specific sample selection: if a steering_config has a "use_samples" field,
    it will select from pos_sentences ("pos") or neg_sentences ("neg") instead of sentences.
    """
    specs: list[PromptSpec] = []
    template = get_dataset_template(dataset_tag)
    steering_configs = template.get("steering_configs", {})
    
    # Use dataset-specific manners if not provided
    if not manners:
        manners = template.get("manners", ["negative", "positive"])

    for manner in manners:
        # Check for manner-specific instruction first, fall back to main instruction
        if manner in steering_configs and "instruction" in steering_configs[manner]:
            instruction_template = steering_configs[manner]["instruction"]
        else:
            instruction_template = template["instruction"]
        
        # Skip if no instruction template available
        if instruction_template is None:
            raise ValueError(f"No instruction template found for manner '{manner}' in dataset '{dataset_tag}'")
        
        # Check for manner-specific sample selection (e.g., counterfact uses different labels)
        use_samples = steering_configs.get(manner, {}).get("use_samples", None)
        if use_samples == "pos" and pos_sentences is not None:
            manner_sentences = pos_sentences
            print(f"  [Manner '{manner}'] Using pos_sentences (label=1, true): {len(manner_sentences)} samples")
        elif use_samples == "neg" and neg_sentences is not None:
            manner_sentences = neg_sentences
            print(f"  [Manner '{manner}'] Using neg_sentences (label=0, false): {len(manner_sentences)} samples")
        else:
            manner_sentences = sentences
        
        for text in manner_sentences:
            # Check if instruction uses {text} placeholder
            if "{text}" in instruction_template:
                instruction = instruction_template.format(text=text)
            elif "{manner}" in instruction_template:
                instruction = instruction_template.format(manner=manner)
            else:
                # Fixed instruction (no placeholders) - use as is
                instruction = instruction_template
            
            prompt = PROMPT_TEMPLATE.format(instruction=instruction)
            specs.append(
                PromptSpec(
                    label=manner,
                    prompt=prompt,
                    source=text,  # Keep original text for reference/style change detection
                    use_chat_template=None,
                    meta={"manner": manner, "dataset": dataset_tag},
                )
            )

    return specs


def build_gcs_prompt_specs(
    sentences: Sequence[str],
    manners: Sequence[str],
) -> list[PromptSpec]:
    """Legacy function for backward compatibility with STSA."""
    return build_dataset_prompt_specs(sentences, manners, "STSA")

def prepare_encoded_prompts(
    tokenizer, device, prompt_specs: Sequence[PromptSpec], chat_template_default: bool
) -> list[dict[str, object]]:
    encoded = []
    for idx, spec in enumerate(prompt_specs):
        use_chat = spec.use_chat_template
        if use_chat is None:
            use_chat = chat_template_default
        if use_chat and hasattr(tokenizer, "apply_chat_template"):
            ids = encode_chat(tokenizer, spec.prompt, device)
        else:
            toks = tokenizer(spec.prompt, return_tensors="pt")
            ids = toks["input_ids"].to(device)

        encoded.append(
            {
                "index": idx,
                "label": spec.label,
                "prompt": spec.prompt,
                "source": spec.source,
                "ids": ids,
                "meta": dict(spec.meta),
            }
        )
    return encoded


# ---------------------------------------------------------------------------
# Generation core (CSV writer)
# ---------------------------------------------------------------------------
def generate_bagging_steering_csv(
    *,
    config: SteeringConfig,
    prompt_specs: Sequence[PromptSpec],
) -> list[list[object]]:
    if not prompt_specs:
        raise ValueError("No prompt specifications provided.")

    # Adaptive steering: epsilon computed dynamically per layer
    print("[INFO] Using adaptive steering: epsilon computed dynamically per layer")

    model, tokenizer, device = load_model_and_tokenizer(
        config.model_source, config.cache_dir, config.quant_bits, config.cuda_device
    )
    tot_layer = infer_num_layers(model)
    vectors = load_concept_vectors(
        config.savepath,
        config.model_id,
        config.dataset_tag,
        tot_layer,
        device,
        vector_kind=config.vector_kind,
        normalize_l2=config.normalize_l2,
        gcs_pkl_path=config.gcs_pkl_path,
        gcs_concept=config.gcs_concept,
    )
    
    # Load test accuracies (only use layers where TestAcc > P_1)
    test_accuracies = load_test_accuracies(
        config.savepath,
        config.model_id,
        config.dataset_tag,
        tot_layer,
        vector_kind=config.vector_kind,
    )

    if config.layers is None:
        # Filter layers based on vector availability and test accuracy
        target_layers = []
        vectors_loaded = sum(1 for v in vectors if v is not None)
        print(f"[INFO] Loaded {vectors_loaded}/{tot_layer} concept vectors")
        if test_accuracies:
            acc_available = sum(1 for a in test_accuracies if a is not None)
            print(f"[INFO] Loaded {acc_available}/{tot_layer} test accuracies")
        
        for layer in range(1, tot_layer - 1):
            if vectors[layer] is None:
                print(f"[INFO] Skipping layer {layer}: concept vector not found")
                continue
            
            # Only use layers where TestAcc(P_m) > P_1 (default 90%)
            if test_accuracies is not None and config.accuracy_threshold > 0:
                if test_accuracies[layer] is None or test_accuracies[layer] < config.accuracy_threshold:
                    print(f"[INFO] Skipping layer {layer}: test_acc={test_accuracies[layer]} < {config.accuracy_threshold}")
                    continue
            
            target_layers.append(layer)
            print(f"[INFO] Using layer {layer}: test_acc={test_accuracies[layer] if test_accuracies and test_accuracies[layer] is not None else 'N/A'}")
    else:
        target_layers = []
        for layer in config.layers:
            if not (0 <= layer < tot_layer):
                raise ValueError(f"Layer {layer} out of range for model with {tot_layer} layers.")
            if vectors[layer] is None:
                raise FileNotFoundError(f"Missing concept vector for layer {layer}.")
            
            # Check test accuracy constraint
            if test_accuracies is not None and config.accuracy_threshold > 0:
                if test_accuracies[layer] is None or test_accuracies[layer] < config.accuracy_threshold:
                    print(f"[WARNING] Layer {layer} has test_acc={test_accuracies[layer]} < {config.accuracy_threshold}")
            
            target_layers.append(layer)
    
    if not target_layers:
        raise RuntimeError("No usable layers found for steering. Check concept vectors and test accuracies.")

    encoded_prompts = prepare_encoded_prompts(
        tokenizer, device, prompt_specs, config.chat_template_default
    )

    # Update generation_config with parameters (required for some models like Qwen3)
    # Qwen3 and some other models require parameters to be set via generation_config
    update_generation_config(
        model,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        do_sample=True,
        verbose=True,  # Print what was updated
    )

    os.makedirs(config.output_csv.parent, exist_ok=True)

    header = [
        "model_id",
        "dataset_tag",
        "layer",
        "manner",
        "sample_id",
        "source_text",
        "original_generation",
        "steered_generation",
        "style_change_detected",
        "target_prob",
        "accuracy_threshold",
    ]
    rows: list[list[object]] = [header]

    # Build generation kwargs
    # For Qwen3 models, do NOT pass temperature/top_p/top_k directly (they cause warnings)
    # These parameters are set via generation_config instead
    is_qwen3 = is_qwen3_model(model)
    
    base_kwargs = dict(
        max_new_tokens=config.max_tokens,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    
    # For Qwen3 models, skip these parameters in generate() call
    # They are already set via generation_config above
    if not is_qwen3:
        # For other models, include these parameters directly
        base_kwargs.update({
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
            "no_repeat_ngram_size": config.no_repeat_ngram_size,
        })
    else:
        # For Qwen3, only include no_repeat_ngram_size if supported
        # (temperature, top_p, repetition_penalty are in generation_config)
        base_kwargs["no_repeat_ngram_size"] = config.no_repeat_ngram_size
    
    if is_qwen3:
        print(f"[INFO] Detected Qwen3 model - using generation_config for temperature/top_p parameters")

    layer_display = ",".join(str(layer) for layer in target_layers)

    for prompt_cfg in encoded_prompts:
        for sample_idx in range(config.num_samples):
            seed = config.seed_base + sample_idx + prompt_cfg["index"] * 100000
            set_seed_local(seed)

            base_out = model.generate(input_ids=prompt_cfg["ids"], **base_kwargs)
            base_tokens = base_out[0]
            base_new = base_tokens[prompt_cfg["ids"].shape[1] :]
            base_text = tokenizer.decode(base_new, skip_special_tokens=True).strip()

            set_seed_local(seed)
            
            # Register adaptive steering hooks
            handles = register_steering_hooks(
                model,
                vectors,
                layer_ids=target_layers,
                target_prob=config.target_prob,
                bias=config.bias,
                test_accuracies=test_accuracies,
                accuracy_threshold=config.accuracy_threshold,
                use_position_coefficients=config.use_position_coefficients,
                position_decay_rate=config.position_decay_rate,
            )
            try:
                steered_out = model.generate(input_ids=prompt_cfg["ids"], **base_kwargs)
            finally:
                for h in handles:
                    h.remove()

            steered_tokens = steered_out[0]
            steered_new = steered_tokens[prompt_cfg["ids"].shape[1] :]
            steered_text = tokenizer.decode(steered_new, skip_special_tokens=True).strip()

            # Detect style change using the LLM
            style_change = detect_style_change(
                model, tokenizer, device, base_text, steered_text, prompt_cfg['label']
            )

            # Print progress
            print(f"\n{'='*80}")
            print(
                f"Parameters: model={config.model_id}, dataset={config.dataset_tag}, "
                f"layers={layer_display}, manner={prompt_cfg['label']}, sample={sample_idx}"
            )
            print(f"Original generation: {base_text}")
            print(f"Steered generation: {steered_text}")
            print(f"Style change detected: {style_change}")
            print(f"{'='*80}")

            rows.append(
                [
                    config.model_id,
                    config.dataset_tag,
                    layer_display,
                    prompt_cfg["label"],
                    sample_idx,
                    prompt_cfg["source"],
                    base_text,
                    steered_text,
                    style_change,
                    config.target_prob,
                    config.accuracy_threshold,
                ]
            )

    with open(config.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return rows


def generate_unified_steering_csv(
    *,
    config: SteeringConfig,
    prompt_specs: Sequence[PromptSpec],
    all_manners: Sequence[str],
) -> list[list[object]]:
    """
    Generate steering results with unified experiment design:
    - One original generation per input
    - Multiple steered generations (one per manner/direction)
    
    This is more efficient for datasets that start from neutral (STSA, sarcasm, hatexplain)
    where the same neutral prompt is used for both steering directions.
    
    Args:
        config: Steering configuration
        prompt_specs: Prompt specifications (should be for a single base prompt, not per-manner)
        all_manners: All steering directions to evaluate (e.g., ["neutral_to_positive", "neutral_to_negative"])
    """
    if not prompt_specs:
        raise ValueError("No prompt specifications provided.")
    if not all_manners:
        raise ValueError("No manners specified for unified steering.")

    print(f"[INFO] Using UNIFIED steering mode: 1 original + {len(all_manners)} steered per input")
    print(f"[INFO] Manners: {all_manners}")

    model, tokenizer, device = load_model_and_tokenizer(
        config.model_source, config.cache_dir, config.quant_bits, config.cuda_device
    )
    tot_layer = infer_num_layers(model)
    vectors = load_concept_vectors(
        config.savepath,
        config.model_id,
        config.dataset_tag,
        tot_layer,
        device,
        vector_kind=config.vector_kind,
        normalize_l2=config.normalize_l2,
        gcs_pkl_path=config.gcs_pkl_path,
        gcs_concept=config.gcs_concept,
    )
    
    test_accuracies = load_test_accuracies(
        config.savepath,
        config.model_id,
        config.dataset_tag,
        tot_layer,
        vector_kind=config.vector_kind,
    )

    # Determine target layers
    if config.layers is None:
        target_layers = []
        vectors_loaded = sum(1 for v in vectors if v is not None)
        print(f"[INFO] Loaded {vectors_loaded}/{tot_layer} concept vectors")
        if test_accuracies:
            acc_available = sum(1 for a in test_accuracies if a is not None)
            print(f"[INFO] Loaded {acc_available}/{tot_layer} test accuracies")
        
        for layer in range(1, tot_layer - 1):
            if vectors[layer] is None:
                continue
            if test_accuracies is not None and config.accuracy_threshold > 0:
                if test_accuracies[layer] is None or test_accuracies[layer] < config.accuracy_threshold:
                    continue
            target_layers.append(layer)
            print(f"[INFO] Using layer {layer}: test_acc={test_accuracies[layer] if test_accuracies and test_accuracies[layer] is not None else 'N/A'}")
    else:
        target_layers = list(config.layers)
    
    if not target_layers:
        raise RuntimeError("No usable layers found for steering.")

    # Get unique prompts (deduplicate by source text since we're doing unified)
    seen_sources = set()
    unique_prompts = []
    for spec in prompt_specs:
        if spec.source not in seen_sources:
            seen_sources.add(spec.source)
            unique_prompts.append(spec)
    
    print(f"[INFO] Processing {len(unique_prompts)} unique prompts × {config.num_samples} samples × {len(all_manners)} manners")

    encoded_prompts = prepare_encoded_prompts(
        tokenizer, device, unique_prompts, config.chat_template_default
    )

    update_generation_config(
        model,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        do_sample=True,
        verbose=True,
    )

    os.makedirs(config.output_csv.parent, exist_ok=True)

    # Build dynamic header based on manners
    header = [
        "model_id",
        "dataset_tag",
        "layer",
        "sample_id",
        "source_text",
        "original_generation",
    ]
    # Add columns for each manner
    for manner in all_manners:
        header.append(f"steered_{manner}")
        header.append(f"style_change_{manner}")
    header.append("accuracy_threshold")
    
    rows: list[list[object]] = [header]

    # Build generation kwargs
    is_qwen3 = is_qwen3_model(model)
    base_kwargs = dict(
        max_new_tokens=config.max_tokens,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    if not is_qwen3:
        base_kwargs.update({
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
            "no_repeat_ngram_size": config.no_repeat_ngram_size,
        })
    else:
        base_kwargs["no_repeat_ngram_size"] = config.no_repeat_ngram_size
        print(f"[INFO] Detected Qwen3 model - using generation_config for temperature/top_p parameters")

    layer_display = ",".join(str(layer) for layer in target_layers)
    
    # Get target_prob for each manner
    manner_target_probs = {}
    for manner in all_manners:
        manner_target_probs[manner] = get_target_prob_for_manner(config.dataset_tag, manner)
        print(f"[INFO] Manner '{manner}': target_prob = {manner_target_probs[manner]}")

    for prompt_cfg in encoded_prompts:
        for sample_idx in range(config.num_samples):
            seed = config.seed_base + sample_idx + prompt_cfg["index"] * 100000
            set_seed_local(seed)

            # Generate original (once per input)
            base_out = model.generate(input_ids=prompt_cfg["ids"], **base_kwargs)
            base_tokens = base_out[0]
            base_new = base_tokens[prompt_cfg["ids"].shape[1]:]
            base_text = tokenizer.decode(base_new, skip_special_tokens=True).strip()

            # Generate steered outputs for each manner
            steered_results = {}
            for manner in all_manners:
                set_seed_local(seed)  # Same seed for fair comparison
                
                target_prob = manner_target_probs[manner]
                
                handles = register_steering_hooks(
                    model,
                    vectors,
                    layer_ids=target_layers,
                    target_prob=target_prob,
                    bias=config.bias,
                    test_accuracies=test_accuracies,
                    accuracy_threshold=config.accuracy_threshold,
                    use_position_coefficients=config.use_position_coefficients,
                    position_decay_rate=config.position_decay_rate,
                    verbose=(manner == all_manners[0]),  # Only verbose for first manner
                )
                try:
                    steered_out = model.generate(input_ids=prompt_cfg["ids"], **base_kwargs)
                finally:
                    for h in handles:
                        h.remove()

                steered_tokens = steered_out[0]
                steered_new = steered_tokens[prompt_cfg["ids"].shape[1]:]
                steered_text = tokenizer.decode(steered_new, skip_special_tokens=True).strip()

                # Detect style change
                style_change = detect_style_change(
                    model, tokenizer, device, base_text, steered_text, manner
                )
                
                steered_results[manner] = {
                    "text": steered_text,
                    "style_change": style_change,
                }

            # Print progress
            print(f"\n{'='*80}")
            print(f"Parameters: model={config.model_id}, dataset={config.dataset_tag}, layers={layer_display}, sample={sample_idx}")
            print(f"Original: {base_text}")
            for manner in all_manners:
                print(f"Steered ({manner}): {steered_results[manner]['text']}")
                print(f"  Style change: {steered_results[manner]['style_change']}")
            print(f"{'='*80}")

            # Build row
            row = [
                config.model_id,
                config.dataset_tag,
                layer_display,
                sample_idx,
                prompt_cfg["source"],
                base_text,
            ]
            for manner in all_manners:
                row.append(steered_results[manner]["text"])
                row.append(steered_results[manner]["style_change"])
            row.append(config.accuracy_threshold)
            
            rows.append(row)

    with open(config.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run activation steering generation for STSA.")
    parser.add_argument("--savepath", type=Path, default=Path("./lab_rs"))
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-7B-Chat")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("./.cache/hf"))
    parser.add_argument("--dataset-tag", type=str, default="STSA", choices=["STSA", "sarcasm", "hatexplain", "counterfact"])
    parser.add_argument("--datapath", type=Path, default=None, help="Path to dataset file. For hatexplain, can be empty to load from Hugging Face.")
    parser.add_argument("--positives", type=int, default=32, help="Max positive samples to evaluate (<=0 for all).")
    parser.add_argument("--negatives", type=int, default=32, help="Max negative samples to evaluate (<=0 for all).")
    parser.add_argument("--vector-kind", choices=["bagging", "stacking", "singlelr", "baseline", "gcs"], default="bagging")
    parser.add_argument("--gcs-pkl", type=Path, default=None,
                       help="Path to GCS concept vector pkl file (for --vector-kind=gcs)")
    parser.add_argument("--gcs-concept", type=str, default=None,
                       help="Concept key inside GCS pkl (for --vector-kind=gcs)")
    parser.add_argument("--layers", type=str, default="", help="Comma-separated layer ids; empty = all middle layers.")
    parser.add_argument("--manners", type=str, default="", help="Comma-separated emotional manners to evaluate. If empty, uses dataset-specific defaults.")
    parser.add_argument("--num-samples", type=int, default=3, help="Generations per prompt.")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--quant", type=int, default=16, choices=[8, 16, 32])
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("./results/steering_generation.csv"))
    
    # Adaptive steering parameters
    parser.add_argument("--target-prob", type=float, default=None, help="Target probability for adaptive steering. If not specified, uses dataset-specific default. (Low values steer AWAY from concept direction, high values steer TOWARDS)")
    parser.add_argument("--accuracy-threshold", type=float, default=0.9, help="Minimum test accuracy to use a layer (default 90%%)")
    parser.add_argument("--bias", type=float, default=0.0, help="Bias term for probability calculation in adaptive steering")
    parser.add_argument("--no-normalize-l2", action="store_true", help="Disable L2 normalization of concept vectors")
    
    # Position-based steering parameters
    parser.add_argument("--use-position-coefficients", action="store_true", help="Enable position-dependent steering strength (earlier tokens get stronger steering)")
    parser.add_argument("--position-decay-rate", type=float, default=0.1, help="Decay rate for exponential position weighting (default 0.1, higher = faster decay)")
    
    # Unified experiment mode
    parser.add_argument("--unified-manners", action="store_true", 
                       help="Enable unified experiment mode: generate one original + multiple steered outputs per input. "
                            "Best for datasets with neutral starting point (STSA, sarcasm, hatexplain).")
    
    return parser.parse_args()


def parse_layers_arg(layers_arg: str) -> Optional[list[int]]:
    if not layers_arg.strip():
        return None
    result = []
    for chunk in layers_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            start_i, end_i = int(start), int(end)
            result.extend(range(start_i, end_i + 1))
        else:
            result.append(int(chunk))
    return sorted(set(result))


def parse_manners_arg(manners_arg: str, dataset_tag: str = "STSA") -> list[str]:
    tokens = [tok.strip() for tok in manners_arg.split(",") if tok.strip()]
    if not tokens:
        # Use dataset-specific default manners (now defaults to only "negative")
        template = get_dataset_template(dataset_tag)
        return template.get("manners", ["negative"])
    return tokens


def main() -> None:
    args = parse_args()
    
    # Set default datapath based on dataset_tag if not explicitly provided
    if args.datapath is None:
        if args.dataset_tag.upper() == "STSA":
            args.datapath = Path("./dataset/stsa.binary.train")
        elif args.dataset_tag.lower() == "sarcasm":
            args.datapath = Path("./dataset/sarcasm.json")
        elif args.dataset_tag.lower() == "hatexplain":
            args.datapath = None  # Will load from Hugging Face
        elif args.dataset_tag.lower() == "counterfact":
            args.datapath = Path("./dataset/counterfact.csv")

    os.environ.setdefault("HF_HOME", str(args.cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(args.cache_dir))

    # Determine if using unified mode
    use_unified = args.unified_manners
    
    # Get manners: for unified mode, use all manners; otherwise use args or default
    if use_unified:
        # Unified mode: use all available manners for the dataset
        manners = tuple(get_all_manners(args.dataset_tag))
        print(f"\n[UNIFIED MODE] Will generate 1 original + {len(manners)} steered per input")
    elif args.manners:
        manners = tuple(parse_manners_arg(args.manners, args.dataset_tag))
    else:
        manners = tuple(get_default_manners(args.dataset_tag))
    
    # Get target_prob: if user specified, use that; otherwise get from manner config
    # (In unified mode, target_prob is determined per-manner during generation)
    if args.target_prob is not None:
        target_prob = args.target_prob
        target_prob_source = "(from args)"
    elif len(manners) == 1:
        target_prob = get_target_prob_for_manner(args.dataset_tag, manners[0])
        target_prob_source = f"(from manner '{manners[0]}')"
    else:
        target_prob = get_default_target_prob(args.dataset_tag)
        target_prob_source = "(dataset default, per-manner in unified mode)"
    
    # Print dataset configuration summary
    print_dataset_config_summary(args.dataset_tag, manners)
    print(f"\n  Selected configuration:")
    print(f"    Mode: {'UNIFIED' if use_unified else 'SEPARATE'}")
    print(f"    Manners: {manners} {'(all for unified)' if use_unified else ('(from args)' if args.manners else '(dataset default)')}")
    if not use_unified:
        print(f"    Target prob: {target_prob} {target_prob_source}")

    pos_limit = args.positives if args.positives > 0 else None
    neg_limit = args.negatives if args.negatives > 0 else None
    
    # Load samples using the new dataset routing system
    pos_samples, neg_samples = load_dataset_samples(
        args.dataset_tag,
        args.datapath,
        pos_limit=pos_limit,
        neg_limit=neg_limit,
        seed=args.seed,
    )

    evaluation_sentences = list(neg_samples) + list(pos_samples)
    if not evaluation_sentences:
        raise ValueError("No evaluation sentences available for prompt construction.")

    # For unified mode, build prompts with just one manner (they share the same neutral prompt)
    # For separate mode, build prompts for each manner
    if use_unified:
        # Use only the first manner for prompt building (all manners share same neutral prompt)
        prompts = build_dataset_prompt_specs(
            evaluation_sentences,
            [manners[0]],  # Just use first manner for base prompts
            args.dataset_tag,
            pos_sentences=pos_samples,
            neg_sentences=neg_samples,
        )
    else:
        prompts = build_dataset_prompt_specs(
            evaluation_sentences,
            manners,
            args.dataset_tag,
            pos_sentences=pos_samples,
            neg_sentences=neg_samples,
        )

    config = SteeringConfig(
        savepath=args.savepath,
        model_id=args.model,
        model_source=str(args.model_path) if args.model_path else args.model,
        dataset_tag=args.dataset_tag,
        cache_dir=args.cache_dir,
        output_csv=args.output,
        quant_bits=args.quant,
        cuda_device=args.cuda,
        layers=parse_layers_arg(args.layers),
        vector_kind=args.vector_kind,
        gcs_pkl_path=getattr(args, 'gcs_pkl', None),
        gcs_concept=getattr(args, 'gcs_concept', None),
        target_prob=target_prob,
        accuracy_threshold=args.accuracy_threshold,
        bias=args.bias,
        normalize_l2=not args.no_normalize_l2,
        use_position_coefficients=args.use_position_coefficients,
        position_decay_rate=args.position_decay_rate,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        chat_template_default=not args.no_chat_template,
        seed_base=args.seed,
        manners=manners,
    )

    # Use appropriate generation function
    if use_unified:
        generate_unified_steering_csv(config=config, prompt_specs=prompts, all_manners=manners)
    else:
        generate_bagging_steering_csv(config=config, prompt_specs=prompts)


if __name__ == "__main__":
    main()
