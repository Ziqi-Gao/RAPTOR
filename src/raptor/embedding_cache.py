"""Validate completed extraction caches before treating an NPZ as reusable."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

SOURCE_FILES = ("core.py", "data.py", "model_utils.py", "embeddings.py", "embedding_checkpoint.py",
                "embedding_cache.py", "experiments/run_embeddings.py")
EXPECTED_LAYERS = {"google/gemma-7b-it": 28, "meta-llama/Meta-Llama-3.1-8B-Instruct": 32,
                   "meta-llama/Meta-Llama-3.1-70B-Instruct": 80, "Qwen/Qwen2.5-3B-Instruct": 36,
                   "Qwen/Qwen2.5-7B-Instruct": 28, "Qwen/Qwen2.5-32B-Instruct": 64}


def extraction_source_identity():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update(name.encode()+b"\0"+root.joinpath(name).read_bytes())
    return digest.hexdigest()


def extraction_identity(*, model, dataset, model_source, quant, noise, input_sha256):
    return {"schema": "raptor.embedding_cache.v1", "model": model, "dataset": dataset,
            "model_source": str(model_source), "quant": int(quant), "noise": noise,
            "ordered_input_sha256": input_sha256, "source_sha256": extraction_source_identity(),
            "tap": "model.layers.{layer}.post_attention_layernorm.output[0][-1,:]",
            "coordinate_space": "raw", "row_order": "positive_then_negative", "dtype": "float32"}


def validate_completed_embedding(path, *, model, dataset, n_pos, n_neg, identity):
    """Read every layer with CRC/shape/finite and extraction-identity checks.

    Old schema-only caches remain usable as explicitly prepared benchmark inputs,
    but cannot silently certify completion of a newly requested extraction.
    """
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as source:
            layers = sorted(int(key[6:]) for key in source.files if key.startswith("X_pos_"))
            negative = sorted(int(key[6:]) for key in source.files if key.startswith("X_neg_"))
            if not layers or layers != list(range(len(layers))) or negative != layers:
                raise ValueError("missing or nonconsecutive positive/negative layers")
            if model in EXPECTED_LAYERS and len(layers) != EXPECTED_LAYERS[model]:
                raise ValueError("cache lacks the full model layer set")
            def scalar(key):
                value = np.asarray(source[key])
                if value.size != 1:
                    raise ValueError("malformed scalar cache metadata")
                return value.reshape(-1)[0].item()
            if scalar("meta_model") != model or scalar("meta_dataset") != dataset or scalar("meta_layers") != len(layers):
                raise ValueError("model/dataset/layer cache metadata differs")
            dim = int(scalar("meta_dim"))
            if dim <= 0 or min(n_pos, n_neg) <= 0:
                raise ValueError("empty cache dimensions")
            saved = json.loads(scalar("meta_identity_json"))
            if saved != identity:
                raise ValueError("extraction identity differs (inputs, quantization, noise, model or source)")
            for prefix, count, label in (("X_pos_", n_pos, 1), ("X_neg_", n_neg, 0)):
                labels = source["y_pos" if label else "y_neg"]
                if labels.shape != (count,) or not np.all(labels == label):
                    raise ValueError("cache class labels/counts differ")
                for layer in layers:
                    value = source[prefix+str(layer)]
                    if value.shape != (count, dim) or value.dtype != np.dtype("float32") or not np.isfinite(value).all():
                        raise ValueError("cache layer shape/dtype/finiteness differs")
        return {"status": "verified_complete", "layers": len(layers), "dim": dim, "n_pos": n_pos, "n_neg": n_neg}
    except Exception as error:
        raise ValueError(f"unverified/incompatible existing embedding cache {path}: {error}; choose a fresh --emb_dir or explicitly --force") from error
