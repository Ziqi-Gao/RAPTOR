from __future__ import annotations

import numpy as np
import torch



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
                device = model.get_input_embeddings().weight.device
            except (AttributeError, StopIteration):
                try:
                    device = next(model.parameters()).device
                except StopIteration:
                    device = (
                        f"cuda:{self.cuda_id}"
                        if torch.cuda.is_available()
                        else "cpu"
                    )
            if getattr(device, "type", None) == "meta":
                device = f"cuda:{self.cuda_id}" if torch.cuda.is_available() else "cpu"

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Native PyTorch hooks collect the identical module output as TraceDict.
        # A missing tracing package must never switch to block output_hidden_states.
        modules = dict(model.named_modules())
        missing = [name for name in self.layer_names if name not in modules]
        if missing:
            raise ValueError(f"Model lacks required post_attention_layernorm taps: {missing}")
        captured = {}
        handles = []
        def capture(name):
            def hook(module, inputs, output):
                captured[name] = output[0].detach().cpu()
            return hook
        try:
            for name in self.layer_names:
                handles.append(modules[name].register_forward_hook(capture(name)))
            model(**inputs)
            return [captured[name] for name in self.layer_names]
        finally:
            for handle in handles:
                handle.remove()
