"""Portable probe-coordinate measurements and resumable text generation.

The approved GCAV kernels implement the intervention. Observers record its
actual pre/post hidden states without changing the intervention. No judge runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def load_bundle(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {key: np.asarray(data[key]) for key in ("W", "B", "val_acc_pre_refit")}
    if metadata.get("schema") != "raptor.steering.own_probe.v1":
        raise ValueError("unsupported probe bundle schema; prepare your own bundle")
    if metadata.get("validation_source") != "pre_refit_train_only":
        raise ValueError("reliability must use pre-refit train-only outer validation")
    for field in ("model", "dataset", "split_sha256", "embedding_sha256", "activation_site", "method", "source_identity"):
        if not metadata.get(field): raise ValueError("missing probe identity " + field)
    w, b, val = arrays["W"], arrays["B"], arrays["val_acc_pre_refit"]
    if w.ndim != 2 or b.shape != (len(w),) or val.shape != b.shape:
        raise ValueError("probe shape mismatch")
    if not all(np.isfinite(a).all() for a in arrays.values()) or np.any((val < 0) | (val > 1)):
        raise ValueError("nonfinite probe or invalid validation fraction")
    norms = np.linalg.norm(w, axis=1, keepdims=True)
    if np.any(norms <= 0): raise ValueError("zero probe direction")
    arrays["directions"] = w / norms
    return arrays, metadata


def checkpoint_valid(path, identity):
    try:
        value = json.loads(Path(path).read_text())
        return (value.get("identity") == identity and value.get("status") == "complete"
                and value.get("payload_sha256") == digest(value["payload"]))
    except (OSError, ValueError, KeyError):
        return False


def summarize_trajectories(records, unit="first_step"):
    if unit not in {"first_step", "all_steps"}: raise ValueError("unknown denominator")
    rows = [r for record in records for r in record["trajectory"] if unit == "all_steps" or r["step"] == 0]
    if not rows: raise ValueError("no measured layer-prompt/token units")
    alpha = np.asarray([abs(r["alpha_actual"]) for r in rows])
    return dict(unit="layer-prompt" if unit == "first_step" else "layer-token",
                N=len(rows), success_rate=float(np.mean([r["success"] for r in rows])),
                intervention_rate=float(np.mean([r["intervened"] for r in rows])),
                alpha_median=float(np.median(alpha)), alpha_p90=float(np.quantile(alpha, .9)),
                alpha_max=float(np.max(alpha)),
                success_definition="measured post-hook probe probability meets target; no tolerance")


def install_observed_hooks(model, arrays, active, *, target, mode, activation_site):
    """Return handles and real measurements; native kernels are unchanged."""
    import torch
    from .gcav import get_blocks, register_steering_hooks_sequential
    vectors = [torch.as_tensor(v) for v in arrays["directions"]]
    weights = [torch.as_tensor(v) for v in arrays["W"]]
    biases = arrays["B"].tolist() if mode == "raw_calibrated" else [0.] * len(weights)
    blocks = get_blocks(model)
    modules = {layer: blocks[layer] if activation_site == "transformer_block_output"
               else blocks[layer].post_attention_layernorm for layer in active}
    before, steps, trace, handles = {}, {layer: 0 for layer in active}, [], []
    def last(output): return (output[0] if isinstance(output, tuple) else output)[:, -1, :]
    for layer, module in modules.items():
        def observe_before(_module, _inputs, output, layer=layer):
            before[layer] = last(output).detach().clone()
        handles.append(module.register_forward_hook(observe_before))
    try:
        handles.extend(register_steering_hooks_sequential(
            model, vectors, active, target_prob=target, accuracy_threshold=0,
            reliability_enabled=False, validation_accuracies=arrays["val_acc_pre_refit"].tolist(),
            probe_weights=weights if mode == "raw_calibrated" else None,
            probe_biases=biases if mode == "raw_calibrated" else None,
            activation_site=activation_site, verbose=False))
        for layer, module in modules.items():
            def observe_after(_module, _inputs, output, layer=layer):
                pre, post = before[layer].double(), last(output).detach().double()
                v = vectors[layer].to(post.device).double()
                weight = weights[layer].to(post.device).double() if mode == "raw_calibrated" else v
                p_before = torch.sigmoid(pre @ weight + biases[layer]).item()
                p_after = torch.sigmoid(post @ weight + biases[layer]).item()
                alpha = (((post-pre) @ v) / (v @ v)).item()
                trace.append(dict(layer=layer, step=steps[layer], probability_before=p_before,
                                  probability_after=p_after, alpha_actual=alpha,
                                  measurement_dtype="float64_from_actual_pre_post_hidden_states",
                                  intervened=bool(torch.any(post != pre).item()),
                                  success=p_after >= target if target > .5 else p_after <= target))
                steps[layer] += 1
            handles.append(module.register_forward_hook(observe_after))
    except Exception:
        for handle in handles: handle.remove()
        raise
    return handles, trace


def run(config, bundle_path, prompt_path, output, *, plan_only=False):
    arrays, meta = load_bundle(bundle_path)
    for key in ("model", "dataset", "activation_site"):
        if config[key] != meta[key]: raise ValueError("config/probe mismatch " + key)
    if config["mode"] not in {"legacy_unit", "raw_calibrated"}: raise ValueError("unknown calibration mode")
    from .gcav import select_active_layers
    policy = config["reliability"]
    active = select_active_layers(list(arrays["directions"]), arrays["val_acc_pre_refit"],
                                  layers=config.get("layers"), accuracy_threshold=policy["threshold"],
                                  reliability_enabled=policy["enabled"], reliability_comparison=policy["comparison"])
    if not active: raise ValueError("no active layers; no generation attempted")
    if meta["method"] == "dim_gcav" and sorted(active) != sorted(meta.get("final_refit_layers", [])):
        raise ValueError("CAA generation mask differs from the mask frozen before final refit")
    prompts = [json.loads(line) for line in Path(prompt_path).read_text().splitlines() if line.strip()]
    ids = [row["prompt_id"] for row in prompts]
    if len(set(ids)) != len(ids) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("prompt IDs must be unique nonnegative integers")
    if any(not isinstance(row["text"], str) or not row["text"] for row in prompts): raise ValueError("empty prompt")
    seeds = config["generation_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds): raise ValueError("explicit unique seeds required")
    source = {name: file_hash(Path(__file__).with_name(name)) for name in ("controls.py", "gcav.py")}
    identity = dict(protocol="measured_gcav_v1", config=config, probe_sha256=file_hash(bundle_path),
                    prompts_sha256=file_hash(prompt_path), source_sha256=source, active_layers=active)
    output = Path(output)
    selection = dict(identity=identity, probe_metadata=meta, prompt_count=len(prompts),
                     record_count=len(prompts)*len(seeds), summary_unit="first-step layer-prompt")
    if plan_only: return selection
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != selection:
        raise ValueError("incompatible resume identity; use a new output directory")
    atomic_json(manifest, selection)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]
    tokenizer = model = None
    records = []
    for prompt in prompts:
        for base_seed in seeds:
            # Explicit recorded full seed; Original and methods use the same seed.
            generation_seed = base_seed + int(prompt["prompt_id"]) * 100000
            key = dict(prompt_id=prompt["prompt_id"], generation_seed=generation_seed)
            unit_identity = dict(run_sha256=digest(identity), **key)
            path = output/"units"/(digest(key)+".json")
            if checkpoint_valid(path, unit_identity):
                records.append(json.loads(path.read_text())["payload"]); continue
            if model is None:
                tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
                model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config["revision"],
                                                           torch_dtype=dtype).to(config["device"]).eval()
                if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
            text = tokenizer.apply_chat_template([dict(role="user", content=prompt["text"])],
                       tokenize=False, add_generation_prompt=True) if config["chat_template"] else prompt["text"]
            inputs = tokenizer(text, return_tensors="pt").to(config["device"])
            def generate():
                random.seed(generation_seed); np.random.seed(generation_seed); torch.manual_seed(generation_seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(generation_seed)
                with torch.inference_mode():
                    result = model.generate(**inputs, **config["decoding"], pad_token_id=tokenizer.pad_token_id)
                tokens = result[0, inputs.input_ids.shape[1]:].tolist()
                return tokens, tokenizer.decode(tokens, skip_special_tokens=True)
            original_tokens, original = generate()
            handles, trajectory = install_observed_hooks(model, arrays, active, target=config["target_probability"],
                            mode=config["mode"], activation_site=config["activation_site"])
            try: tokens, steered = generate()
            finally:
                for handle in handles: handle.remove()
            payload = dict(**key, model_id=config["model"], task=config["dataset"], direction=config["direction"],
                           method=meta["method"], prompt_text=prompt["text"], original_output=original,
                           steered_output=steered, token_ids=tokens, original_token_ids=original_tokens,
                           trajectory=trajectory)
            atomic_json(path, dict(identity=unit_identity, status="complete", payload=payload, payload_sha256=digest(payload)))
            records.append(payload)
    summary = summarize_trajectories(records)
    atomic_json(output/"control_summary.json", summary)
    atomic_json(output/"generations.json", records)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(json.loads(args.config.read_text()), args.probe, args.prompts, args.out,
                         plan_only=args.plan_only), indent=2))


if __name__ == "__main__": main()
