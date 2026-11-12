from __future__ import annotations

import csv
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from raptor.experiments import accuracy_multiseed as accuracy


class AccuracyMultiseedProtocolTest(unittest.TestCase):
    def test_source_version_accepts_submit_host_commit_override(self) -> None:
        with patch.dict(
            "os.environ",
            {"RAPTOR_GIT_COMMIT": "submitted-commit"},
        ):
            version = accuracy.source_version()
        self.assertEqual(version["git_commit"], "submitted-commit")
        self.assertEqual(len(version["source_sha256"]), 64)

    def test_seed_paths_are_isolated(self) -> None:
        root = Path("/tmp/results")
        first = accuracy.seed_directory(
            root,
            accuracy.ROW_PROTOCOL,
            "model",
            "dataset",
            42,
        )
        second = accuracy.seed_directory(
            root,
            accuracy.ROW_PROTOCOL,
            "model",
            "dataset",
            43,
        )
        self.assertNotEqual(first, second)
        self.assertEqual(first.name, "seed_42")
        self.assertEqual(second.name, "seed_43")

    def test_row_split_is_reproducible_and_keeps_existing_proportions(self) -> None:
        y = np.tile([0, 1], 500)
        first = accuracy.split_indices(y, 0.2, 0.2, 42)
        repeated = accuracy.split_indices(y, 0.2, 0.2, 42)
        changed = accuracy.split_indices(y, 0.2, 0.2, 43)
        for left, right in zip(first, repeated):
            np.testing.assert_array_equal(left, right)
        self.assertFalse(np.array_equal(first[0], changed[0]))
        self.assertEqual([len(part) for part in first], [640, 160, 200])

    def test_validation_selection_never_uses_test_and_ties_choose_lowest(self) -> None:
        selected = accuracy.select_validation_layer(
            [4, 2, 7],
            [0.8, 0.8, 0.7],
            [0.99, 0.55, 1.0],
        )
        self.assertEqual(selected["selected_layer"], 2)
        self.assertEqual(selected["best_layer_test_accuracy"], 0.55)
        self.assertAlmostEqual(
            selected["average_layer_test_accuracy"],
            (0.99 + 0.55 + 1.0) / 3,
        )

    def test_grouped_split_has_zero_entity_overlap(self) -> None:
        groups = np.repeat(np.asarray([f"g{i}" for i in range(100)]), 2)
        y = np.tile([1, 0], 100)
        train, val, test = accuracy.grouped_split_indices(
            y,
            groups,
            0.2,
            0.2,
            42,
        )
        accuracy.validate_split(y, train, val, test, groups=groups)
        self.assertFalse(
            set(groups[train]).intersection(groups[val])
            or set(groups[train]).intersection(groups[test])
            or set(groups[val]).intersection(groups[test])
        )

    def test_group_ids_preserve_positive_then_negative_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "counterfact.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["statement", "label", "subject"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"statement": "p0", "label": 1, "subject": "s0"},
                        {"statement": "n0", "label": 0, "subject": "s0"},
                        {"statement": "p1", "label": 1, "subject": "s1"},
                        {"statement": "n1", "label": 0, "subject": "s1"},
                    ]
                )
            groups = accuracy._load_group_ids(
                "counterfact",
                n_pos=2,
                n_neg=2,
                data_path=path,
            )
            np.testing.assert_array_equal(groups, ["s0", "s1", "s0", "s1"])

    def test_interrupted_task_restores_completed_layer_and_recomputes_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            emb_path = tmp / "tiny_embeddings.npz"
            rng = np.random.default_rng(9)
            accuracy.atomic_savez(
                emb_path,
                X_pos_0=rng.normal(size=(20, 4)),
                X_neg_0=rng.normal(size=(20, 4)),
                X_pos_1=rng.normal(size=(20, 4)),
                X_neg_1=rng.normal(size=(20, 4)),
            )
            stat = emb_path.stat()
            embedding = {
                "scope": "synthetic",
                "path": str(emb_path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": accuracy.file_sha256(emb_path),
                "layers": 2,
                "n_pos": 20,
                "n_neg": 20,
                "dim": 4,
            }
            results = tmp / "results"
            split_path = (
                results
                / accuracy.ROW_PROTOCOL
                / "tiny"
                / "tiny"
                / "seed_42"
                / "splits.npz"
            )
            y = np.tile([1, 0], 20)
            # Embeddings are positive-then-negative, unlike np.tile.
            y = np.concatenate([np.ones(20, dtype=int), np.zeros(20, dtype=int)])
            accuracy.create_split_artifact(
                split_path,
                y=y,
                protocol=accuracy.ROW_PROTOCOL,
                seed=42,
                test_size=0.2,
                val_size=0.2,
                model="tiny",
                dataset="tiny",
                embedding=embedding,
            )
            task = {
                "protocol": accuracy.ROW_PROTOCOL,
                "model": "tiny",
                "model_tag": "tiny",
                "dataset": "tiny",
                "seed": 42,
                "method": "singlelr",
                "layers": [0, 1],
                "hyperparameters": {"max_iter": 5},
                "embedding": embedding,
                "results_root": str(results),
                "code_version": accuracy.source_version(),
                "environment_sha256": accuracy.environment_identity()["sha256"],
            }
            calls = {"count": 0}

            def interrupted(*_args, **_kwargs):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise RuntimeError("simulated interruption")
                return {
                    "validation_accuracy": 0.6,
                    "test_accuracy": 0.7,
                    "concept": np.ones(5),
                }

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                accuracy.execute_task(
                    task,
                    method_runners={"singlelr": interrupted},
                )

            resumed_calls = {"count": 0}

            def resumed(*_args, **_kwargs):
                resumed_calls["count"] += 1
                return {
                    "validation_accuracy": 0.65,
                    "test_accuracy": 0.75,
                    "concept": np.ones(5),
                }

            outcome = accuracy.execute_task(
                task,
                method_runners={"singlelr": resumed},
            )
            self.assertEqual(outcome["restored"], 1)
            self.assertEqual(outcome["computed"], 1)
            self.assertEqual(resumed_calls["count"], 1)

            final = accuracy.execute_task(
                task,
                method_runners={"singlelr": resumed},
            )
            self.assertEqual(final["restored"], 2)
            self.assertEqual(final["computed"], 0)

    def test_advance_signal_stops_after_atomic_layer_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            emb_path = tmp / "signal_embeddings.npz"
            rng = np.random.default_rng(10)
            accuracy.atomic_savez(
                emb_path,
                X_pos_0=rng.normal(size=(20, 3)),
                X_neg_0=rng.normal(size=(20, 3)),
                X_pos_1=rng.normal(size=(20, 3)),
                X_neg_1=rng.normal(size=(20, 3)),
            )
            stat = emb_path.stat()
            embedding = {
                "scope": "synthetic",
                "path": str(emb_path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": accuracy.file_sha256(emb_path),
                "layers": 2,
                "n_pos": 20,
                "n_neg": 20,
                "dim": 3,
            }
            results = tmp / "results"
            split_path = (
                results
                / accuracy.ROW_PROTOCOL
                / "tiny"
                / "tiny"
                / "seed_42"
                / "splits.npz"
            )
            y = np.concatenate([np.ones(20, dtype=int), np.zeros(20, dtype=int)])
            accuracy.create_split_artifact(
                split_path,
                y=y,
                protocol=accuracy.ROW_PROTOCOL,
                seed=42,
                test_size=0.2,
                val_size=0.2,
                model="tiny",
                dataset="tiny",
                embedding=embedding,
            )
            task = {
                "protocol": accuracy.ROW_PROTOCOL,
                "model": "tiny",
                "model_tag": "tiny",
                "dataset": "tiny",
                "seed": 42,
                "method": "singlelr",
                "layers": [0, 1],
                "hyperparameters": {"max_iter": 5},
                "embedding": embedding,
                "results_root": str(results),
                "code_version": accuracy.source_version(),
                "environment_sha256": accuracy.environment_identity()["sha256"],
            }

            def request_signal(*_args, **_kwargs):
                accuracy._signal_handler(signal.SIGUSR1, None)
                return {
                    "validation_accuracy": 0.6,
                    "test_accuracy": 0.7,
                    "concept": np.ones(4),
                }

            stopped = accuracy.execute_task(
                task,
                method_runners={"singlelr": request_signal},
            )
            self.assertEqual(stopped["computed"], 1)
            self.assertEqual(stopped["stop_requested"], 1)

            def finish(*_args, **_kwargs):
                return {
                    "validation_accuracy": 0.65,
                    "test_accuracy": 0.75,
                    "concept": np.ones(4),
                }

            resumed = accuracy.execute_task(
                task,
                method_runners={"singlelr": finish},
            )
            self.assertEqual(resumed["restored"], 1)
            self.assertEqual(resumed["computed"], 1)


if __name__ == "__main__":
    unittest.main()
