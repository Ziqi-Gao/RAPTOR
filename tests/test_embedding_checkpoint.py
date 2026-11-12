from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from raptor import embeddings
from raptor.embedding_checkpoint import EmbeddingCheckpoint


class EmbeddingCheckpointTest(unittest.TestCase):
    def test_interrupted_collection_resumes_at_next_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir) / "checkpoint"
            identity = {
                "model": "tiny-model",
                "dataset": "tiny-dataset",
                "input_sha256": "abc",
            }
            first = EmbeddingCheckpoint(
                checkpoint_dir,
                identity=identity,
                n_pos=2,
                n_neg=1,
                layers=2,
                dim=3,
            )
            sample_0 = np.arange(6, dtype=np.float32).reshape(2, 3)
            first.append(0, sample_0)
            first.close()

            resumed = EmbeddingCheckpoint(
                checkpoint_dir,
                identity=identity,
                n_pos=2,
                n_neg=1,
                layers=2,
                dim=3,
            )
            self.assertEqual(resumed.completed, 1)
            with (checkpoint_dir / "progress.json").open(
                "r",
                encoding="utf-8",
            ) as handle:
                self.assertEqual(json.load(handle)["completed"], 1)

            sample_1 = sample_0 + 10
            sample_2 = sample_0 + 20
            resumed.append(1, sample_1)
            resumed.append(2, sample_2)
            positive, negative = resumed.layer_arrays()
            np.testing.assert_array_equal(positive[0][0], sample_0[0])
            np.testing.assert_array_equal(positive[1][1], sample_1[1])
            np.testing.assert_array_equal(negative[0][0], sample_2[0])

            resumed.remove()
            self.assertFalse(checkpoint_dir.exists())

    def test_identity_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir) / "checkpoint"
            checkpoint = EmbeddingCheckpoint(
                checkpoint_dir,
                identity={"model": "first"},
                n_pos=1,
                n_neg=1,
                layers=1,
                dim=2,
            )
            checkpoint.close()
            with self.assertRaisesRegex(
                ValueError,
                "checkpoint identity mismatch",
            ):
                EmbeddingCheckpoint(
                    checkpoint_dir,
                    identity={"model": "second"},
                    n_pos=1,
                    n_neg=1,
                    layers=1,
                    dim=2,
                )

    def test_collection_skips_checkpointed_samples_after_interruption(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir) / "checkpoint"
            model = SimpleNamespace(
                config=SimpleNamespace(
                    num_hidden_layers=2,
                    hidden_size=3,
                )
            )

            def hidden(offset: float):
                return [
                    torch.tensor([[offset, offset + 1, offset + 2]]),
                    torch.tensor([[offset + 3, offset + 4, offset + 5]]),
                ]

            first_llm = mock.Mock()
            first_llm.get_hidden_states.side_effect = [
                hidden(0),
                RuntimeError("simulated interruption"),
            ]
            common_patches = (
                mock.patch.object(
                    embeddings.AutoTokenizer,
                    "from_pretrained",
                    return_value=mock.Mock(),
                ),
                mock.patch.object(
                    embeddings.AutoModelForCausalLM,
                    "from_pretrained",
                    return_value=model,
                ),
            )
            with (
                common_patches[0],
                common_patches[1],
                mock.patch.object(embeddings, "LLM", return_value=first_llm),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "simulated interruption",
                ):
                    embeddings.collect_embeddings(
                        model_id="tiny-model",
                        cache_dir=tmp_dir,
                        quant=32,
                        cuda=0,
                        pos_q=["positive-0", "positive-1"],
                        neg_q=["negative-0"],
                        checkpoint_dir=str(checkpoint_dir),
                        checkpoint_identity={"dataset": "tiny"},
                    )

            resumed_llm = mock.Mock()
            resumed_llm.get_hidden_states.side_effect = [
                hidden(10),
                hidden(20),
            ]
            with (
                mock.patch.object(
                    embeddings.AutoTokenizer,
                    "from_pretrained",
                    return_value=mock.Mock(),
                ),
                mock.patch.object(
                    embeddings.AutoModelForCausalLM,
                    "from_pretrained",
                    return_value=model,
                ),
                mock.patch.object(
                    embeddings,
                    "LLM",
                    return_value=resumed_llm,
                ),
            ):
                positive, negative, _, _, checkpoint = (
                    embeddings.collect_embeddings(
                        model_id="tiny-model",
                        cache_dir=tmp_dir,
                        quant=32,
                        cuda=0,
                        pos_q=["positive-0", "positive-1"],
                        neg_q=["negative-0"],
                        checkpoint_dir=str(checkpoint_dir),
                        checkpoint_identity={"dataset": "tiny"},
                    )
                )

            self.assertEqual(resumed_llm.get_hidden_states.call_count, 2)
            np.testing.assert_array_equal(
                positive[0][0],
                np.array([0, 1, 2], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                positive[1][1],
                np.array([13, 14, 15], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                negative[0][0],
                np.array([20, 21, 22], dtype=np.float32),
            )
            checkpoint.remove()


if __name__ == "__main__":
    unittest.main()
