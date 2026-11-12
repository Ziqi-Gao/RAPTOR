"""Completed extraction files require proven compatible identity and full arrays."""
import copy
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from raptor.core import save_embeddings_npz
from raptor.embedding_cache import extraction_identity, validate_completed_embedding
from raptor.embedding_checkpoint import hash_embedding_inputs
from raptor.experiments.run_embeddings import run_embeddings


def fixture(tmp_path):
    positive, negative = ["p0", "p1"], ["n0", "n1", "n2"]
    identity = extraction_identity(model="tiny", dataset="STSA", model_source="tiny", quant=32,
                                   noise="non-noise", input_sha256=hash_embedding_inputs(positive, negative))
    pos = [np.ones((2, 3), dtype=np.float32), np.full((2, 3), 2, dtype=np.float32)]
    neg = [np.zeros((3, 3), dtype=np.float32), np.full((3, 3), -1, dtype=np.float32)]
    path = tmp_path / "tiny_STSA_embeddings.npz"
    save_embeddings_npz(path, "tiny", "STSA", pos, neg, embedding_identity=identity)
    return path, identity, positive, negative


def verify(path, identity):
    return validate_completed_embedding(path, model="tiny", dataset="STSA", n_pos=2, n_neg=3, identity=identity)


def mutate(path, change):
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key].copy() for key in data.files}
    change(values)
    np.savez(path, **values)


def test_complete_embedding_cache_verified(tmp_path):
    path, identity, _, _ = fixture(tmp_path)
    assert verify(path, identity) == {"status": "verified_complete", "layers": 2, "dim": 3, "n_pos": 2, "n_neg": 3}


@pytest.mark.parametrize("failure", ["nonfinite", "layer", "shape", "model", "identity", "labels", "dtype"])
def test_embedding_cache_rejects_corrupt_or_incompatible_arrays(tmp_path, failure):
    path, identity, _, _ = fixture(tmp_path)
    def change(values):
        if failure == "nonfinite": values["X_pos_1"][0, 0] = np.nan
        elif failure == "layer": del values["X_neg_1"]
        elif failure == "shape": values["X_neg_1"] = np.zeros((3, 4), dtype=np.float32)
        elif failure == "model": values["meta_model"] = np.array(["foreign"])
        elif failure == "identity": values["meta_identity_json"] = np.asarray(json.dumps({**identity, "quant": 4}))
        elif failure == "labels": values["y_pos"][0] = 0
        elif failure == "dtype": values["X_pos_0"] = values["X_pos_0"].astype(np.float64)
    mutate(path, change)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="unverified/incompatible"):
        verify(path, identity)
    assert path.read_bytes() == before


def test_embedding_cache_existence_does_not_certify_completion(tmp_path):
    path, identity, _, _ = fixture(tmp_path)
    path.write_bytes(b"incomplete zip payload")
    with pytest.raises(ValueError, match="unverified/incompatible"):
        verify(path, identity)
    assert path.read_bytes() == b"incomplete zip payload"


def test_new_extraction_rejects_unidentified_historical_cache(tmp_path):
    path, identity, _, _ = fixture(tmp_path)
    mutate(path, lambda values: values.pop("meta_identity_json"))
    with pytest.raises(ValueError, match="meta_identity_json"):
        verify(path, identity)


def test_verified_cache_skips_real_model_collection(tmp_path):
    path, identity, pos, neg = fixture(tmp_path)
    processing = mock.Mock()
    processing.dispacher.return_value = (pos, neg, None, None)
    # No LLM or model loader is imported on a verified complete-cache skip.
    with mock.patch("raptor.experiments.run_embeddings.DataProcessing", return_value=processing):
        run_embeddings(["tiny"], ["STSA"], str(tmp_path), model_path="unused", cuda=0, quant=32,
                       noise="non-noise", force=False)
    assert path.is_file()


def test_unverified_existing_cache_is_preserved_and_fails_loudly(tmp_path):
    path, identity, pos, neg = fixture(tmp_path)
    before = b"partial existing result"; path.write_bytes(before)
    processing = mock.Mock(); processing.dispacher.return_value = (pos, neg, None, None)
    with mock.patch("raptor.experiments.run_embeddings.DataProcessing", return_value=processing):
        with pytest.raises(RuntimeError, match="Embedding extraction failed"):
            run_embeddings(["tiny"], ["STSA"], str(tmp_path), model_path="unused", cuda=0, quant=32,
                           noise="non-noise", force=False)
    assert path.read_bytes() == before
