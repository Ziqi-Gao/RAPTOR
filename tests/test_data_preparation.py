import json
import numpy as np
from raptor.data import DataProcessing
from raptor.prepare_data import inspect_dataset


def test_counterfact_group_order_and_truncation(tmp_path):
    import pandas as pd
    pd.DataFrame({"statement": ["n0", "p0", "p1", "n1"], "label": [0, 1, 1, 0], "subject": ["a", "b", "a", "b"]}).to_csv(tmp_path / "counterfact.csv", index=False)
    pos, neg = DataProcessing(str(tmp_path / "counterfact.csv"), "counterfact").counterfact()
    assert (pos, neg) == (["p0", "p1"], ["n0", "n1"])
    row = inspect_dataset(tmp_path, "counterfact")
    assert row["group_count"] == 2 and row["positive_count"] == row["negative_count"] == 2


def test_hatexplain_original_train_order_majority_and_ties(tmp_path):
    folder = tmp_path / "hatexplain"; folder.mkdir()
    records = {"a": {"post_tokens": ["first"], "annotators": [{"label": 0}, {"label": 0}, {"label": 1}]}, "b": {"post_tokens": ["second"], "annotators": [{"label": 1}, {"label": 1}, {"label": 2}]}, "tie": {"post_tokens": ["omit"], "annotators": [{"label": 0}, {"label": 1}, {"label": 2}]}}
    (folder / "dataset.json").write_text(json.dumps(records))
    (folder / "post_id_divisions.json").write_text(json.dumps({"train": ["b", "tie", "a"]}))
    assert DataProcessing(str(folder), "hatexplain").hatexplain() == (["first"], ["second"])
    assert inspect_dataset(tmp_path, "hatexplain")["positive_count"] == 1


def test_post_attention_tap_matches_traced_reference():
    import pytest
    torch = pytest.importorskip("torch")
    from raptor.model_utils import LLM
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.post_attention_layernorm = torch.nn.LayerNorm(3)
        def forward(self, x):
            return self.post_attention_layernorm(x) + 7
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.model = torch.nn.Module(); self.model.layers = torch.nn.ModuleList([Block(), Block()])
        def forward(self, input_ids, **kwargs):
            x = input_ids.float()
            for block in self.model.layers: x = block(x)
            return {"logits": x}
    model = Model(); inputs = torch.tensor([[[1, 2, 4], [2, 5, 1]]])
    values = LLM(0, 2, 16).get_hidden_states(model, lambda *a, **k: {"input_ids": inputs}, "example")
    expected = model.model.layers[0].post_attention_layernorm(inputs.float())[0]
    torch.testing.assert_close(values[0], expected)
    assert not torch.allclose(values[0], expected + 7)
    assert not any(block.post_attention_layernorm._forward_hooks for block in model.model.layers)
