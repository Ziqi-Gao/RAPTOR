import re
import json
# Optional deps; not required for local embedding extraction
try:  # pragma: no cover - optional
    import sagemaker  # type: ignore
    from sagemaker.huggingface import (  # type: ignore
        HuggingFaceModel,
        get_huggingface_llm_image_uri,
    )
except Exception:  # pragma: no cover - optional
    sagemaker = None  # type: ignore
    HuggingFaceModel = None  # type: ignore
    get_huggingface_llm_image_uri = None  # type: ignore

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
try:  # pragma: no cover - optional
    import baukit  # type: ignore
    from baukit import TraceDict  # type: ignore
except Exception:  # pragma: no cover - optional
    baukit = None  # type: ignore
    TraceDict = None  # type: ignore

import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

"""
    Adding noise to the questions.
    Input: 
        question - the question needs to add noise.
    Return:
        noised_question - the question after adding noise.
    should have only aaa/bbb
"""
import numpy as np
def add_noise(question):
    # Adding three letters after the period.
    noise = np.random.choice(a=2) + 97
    noise = [chr(noise)] * 3
    noised_question = ''.join(noise) + question
    return noised_question

class LLM(torch.nn.Module):
    
    """
        Given a total layer, we construct a layer name list.
    """
    def __init__(self, cuda_id, layer_num, quant):
        self.layer_num = layer_num
        self.cuda_id = cuda_id
        self.layer_names = []
        self.quant = quant
        for i in range(self.layer_num):
            self.layer_names.append(f'model.layers.{i}.post_attention_layernorm')

    def get_hidden_states(self, model, tok, prefix, device="cuda:1"):
        """Return a tensor of shape (L, seq_len, dim) with per-layer hidden states.

        Preferred: use baukit.TraceDict to hook 'post_attention_layernorm' as in
        earlier experiments. If baukit is unavailable, fall back to
        model(..., output_hidden_states=True), stacking layer outputs (skip
        embedding layer 0) to keep shape compatibility.
        """
        if self.quant == 32:
            device = f"cuda:{self.cuda_id}"
            model = model.to(device)
        else:
            # Use the model's parameter device for quantized loads.
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = f"cuda:{self.cuda_id}"
        # Prepare inputs on the appropriate device
        inp = {k: torch.tensor(v)[None].to(device) for k, v in tok(prefix).items()}

        # Path A: use TraceDict if available
        if TraceDict is not None:
            with TraceDict(model, self.layer_names) as tr:
                _ = model(**inp)["logits"]
            return torch.stack([tr[ln].output[0] for ln in self.layer_names])

        # Path B: fallback to output_hidden_states
        out = model(**inp, output_hidden_states=True)
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "Model did not return hidden_states; ensure the model supports output_hidden_states."
            )
        # hidden_states: tuple length L+1 (embeddings + each layer). Skip 0th.
        layers = list(hidden_states)[1:1 + self.layer_num]
        # Ensure we have expected number of layers
        if len(layers) != self.layer_num:
            # Best-effort: trim/pad to self.layer_num
            layers = layers[: self.layer_num]
        return torch.stack([t[0] for t in layers])
    
        
