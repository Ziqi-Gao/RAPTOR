from __future__ import annotations

import numpy as np
import torch

try:  # pragma: no cover - optional tracing backend
    from baukit import TraceDict  # type: ignore
except Exception:  # pragma: no cover
    TraceDict = None  # type: ignore


def add_noise(question: str) -> str:
    noise_char = chr(int(np.random.choice(a=2)) + 97)
    return noise_char * 3 + question


class LLM(torch.nn.Module):
    """Collect per-layer hidden states from causal LMs."""

    def __init__(self, cuda_id: int, layer_num: int, quant: int):
        super().__init__()
        self.layer_num = layer_num
        self.cuda_id = cuda_id
        self.quant = quant
        self.layer_names = [
            f"model.layers.{i}.post_attention_layernorm" for i in range(layer_num)
        ]

    def get_hidden_states(self, model, tokenizer, text: str):
        if self.quant == 32:
            device = f"cuda:{self.cuda_id}" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
        else:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = f"cuda:{self.cuda_id}" if torch.cuda.is_available() else "cpu"

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if TraceDict is not None:
            with TraceDict(model, self.layer_names) as traces:
                _ = model(**inputs)["logits"]
            return torch.stack([traces[name].output[0] for name in self.layer_names])

        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states.")
        layers = list(hidden_states)[1 : 1 + self.layer_num]
        if len(layers) != self.layer_num:
            layers = layers[: self.layer_num]
        return torch.stack([tensor[0] for tensor in layers])
