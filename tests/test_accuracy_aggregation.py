from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from raptor.experiments.accuracy_aggregation import aggregate_accuracy, paired_summary, sample_summary
from raptor.experiments.accuracy_multiseed import (
    GROUPED_PROTOCOL, METHODS, ROW_PROTOCOL, _checkpoint_identity, atomic_savez, checkpoint_path,
    create_split_artifact, seed_directory,
)


class AccuracyAggregationTest(unittest.TestCase):
    def fixture(self, tmp: Path, seeds=(42, 43, 44, 45, 46)):
        embedding = {"model": "model", "dataset": "STSA",
                     "path": str(tmp / "model_STSA_embeddings.npz"),
                     "sha256": "emb-sha", "layers": 2, "n_pos": 20, "n_neg": 20, "dim": 4}
        manifest = {"schema_version": 1, "experiment": "test",
                    "protocol": {"outer_seeds": list(seeds), "tie_rule": "lowest layer index",
                                 "test_size": .2, "val_size_within_non_test": .2},
                    "grid": {"active_models": ["model"], "datasets": ["STSA"], "methods": ["singlelr", "xrfm", "gcs"], "blocked": []},
                    "output_root": str(tmp / "results"), "embeddings": [embedding],
                    "code_version": {"git_commit": "x", "source_sha256": "y"},
                    "environment": {"sha256": "frozen-env"}, "splits": []}
        tasks, paths = [], {}
        for seed in seeds:
            directory = seed_directory(Path(manifest["output_root"]), ROW_PROTOCOL, "model", "STSA", seed)
            split_path = directory / "splits.npz"
            split = create_split_artifact(
                split_path, y=np.r_[np.ones(20, dtype=int), np.zeros(20, dtype=int)],
                protocol=ROW_PROTOCOL, seed=seed, test_size=.2, val_size=.2,
                model="model", dataset="STSA", embedding=embedding,
            )
            manifest["splits"].append({"path": str(split_path), "model": "model", "dataset": "STSA",
                                       "protocol": ROW_PROTOCOL, "seed": seed, "split_sha256": split["split_sha256"]})
            for position, method in enumerate(("singlelr", "xrfm", "gcs")):
                task = {"schema_version": 1, "protocol": ROW_PROTOCOL, "model": "model", "model_tag": "model",
                        "dataset": "STSA", "seed": seed, "method": method, "layers": [0, 1],
                        "hyperparameters": {"frozen_budget": 17 + position}, "embedding": embedding,
                        "results_root": manifest["output_root"], "code_version": manifest["code_version"],
                        "environment_sha256": "frozen-env"}
                tasks.append(task)
                for layer in [0, 1]:
                    metadata = {"status": "complete", "identity": _checkpoint_identity(task, layer=layer, split_metadata=split),
                                "method_details": {"validation_source": "pre_refit_train_only"}}
                    path = checkpoint_path(directory, method, layer)
                    atomic_savez(path, validation_accuracy=np.asarray(.9 if layer == 0 else .8),
                                 test_accuracy=np.asarray(.6 + position * .05 if layer == 0 else .99),
                                 concept=np.ones(4 if method == "xrfm" else 5),
                                 metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
                    paths[(method, seed, layer)] = path
        task_path = tmp / "frozen.jsonl"
        task_path.write_text("".join(json.dumps(task) + "\n" for task in tasks))
        manifest["task_files"] = {"fixture": {"path": str(task_path), "count": len(tasks),
                                              "sha256": hashlib.sha256(task_path.read_bytes()).hexdigest()}}
        return manifest, paths

    def aggregate(self, tmp, manifest):
        return aggregate_accuracy(manifest, summary_json=tmp / "summary.json", summary_csv=tmp / "summary.csv")

    def mutate(self, path, change):
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
        metadata = json.loads(str(values["metadata_json"]))
        change(metadata)
        values["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        atomic_savez(path, **values)

    def test_sample_std_uses_ddof_one_and_missing_is_incomplete(self):
        complete = sample_summary({42: .1, 43: .3}, [42, 43])
        self.assertEqual(complete["status"], "complete")
        self.assertAlmostEqual(complete["sample_std_ddof1"], np.sqrt(.02))
        self.assertEqual(sample_summary({42: .1}, [42, 43])["status"], "incomplete")

    def test_paired_interval_uses_outer_seed_differences(self):
        left = {42: .8, 43: .9, 44: .7, 45: .85, 46: .75}
        right = {seed: value - .1 for seed, value in left.items()}
        result = paired_summary(left, right, [42, 43, 44, 45, 46], comparison="singlelr_minus_xrfm")
        self.assertAlmostEqual(result["mean_paired_difference"], .1)
        self.assertAlmostEqual(result["confidence_interval_95"]["lower"], .1)
        self.assertAlmostEqual(result["confidence_interval_95"]["upper"], .1)

    def test_full_aggregation_selects_on_validation_not_test(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td); manifest, paths = self.fixture(tmp)
            result = self.aggregate(tmp, manifest)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["coverage"]["complete_summary_cells"], 6)
            with np.load(paths[("singlelr", 42, 0)].parents[2] / "singlelr_results.npz", allow_pickle=False) as data:
                self.assertEqual(data["validation_source"].item(), "pre_refit_train_only")
                np.testing.assert_array_equal(data["val_acc_pre_refit"], data["val_acc"])
                self.assertEqual(int(data["selected_layer"]), 0)
                self.assertAlmostEqual(float(data["best_layer_test_accuracy"]), .6)
                self.assertAlmostEqual(float(data["average_layer_test_accuracy"]), (.6 + .99) / 2)
            paths[("gcs", 46, 1)].unlink()
            incomplete = self.aggregate(tmp, manifest)
            self.assertEqual(incomplete["status"], "incomplete")
            self.assertGreater(incomplete["coverage"]["missing_or_invalid_layer_checkpoints"], 0)

    def test_no_persistent_split_cannot_be_complete_even_with_plausible_cp_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td); manifest, _ = self.fixture(tmp)
            for record in manifest["splits"]: Path(record["path"]).unlink()
            result = self.aggregate(tmp, manifest)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["coverage"]["complete_summary_cells"], 0)
            self.assertTrue(all("persisted split" in row["reason"] for row in result["missing"]))

    def test_session4_wrong_seed_hparams_without_splits_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td); manifest, paths = self.fixture(tmp)
            for path in paths.values():
                self.mutate(path, lambda m: m["identity"].update(split_seed=999, method_seed=999, hyperparameters={"not_frozen": 999}))
            for record in manifest["splits"]: Path(record["path"]).unlink()
            result = self.aggregate(tmp, manifest)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["coverage"]["complete_summary_cells"], 0)

    def test_every_frozen_identity_field_is_checked(self):
        changes = {"schema_version": 999, "model_tag": "other", "split_seed": 999, "method_seed": 999,
                   "hyperparameters": {"not_frozen": 999}, "split_sha256": "self_claimed_other_split",
                   "embedding_n_pos": 999, "embedding_n_neg": 999, "embedding_layers": 999,
                   "embedding_dim": 999, "environment_sha256": "other-env", "code_version": {"git_commit": "other"},
                   "outer_seed": 999, "protocol": "other", "layer": 999, "unexpected_identity_field": True}
        for field, value in changes.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                tmp = Path(td); manifest, paths = self.fixture(tmp, seeds=[42])
                self.mutate(paths[("singlelr", 42, 0)], lambda m: m["identity"].update({field: value}))
                result = self.aggregate(tmp, manifest)
                self.assertEqual(result["status"], "incomplete")
                self.assertTrue(any("identity mismatch" in r["reason"] for r in result["missing"]))
                self.assertFalse((paths[("singlelr", 42, 0)].parents[2] / "singlelr_results.npz").exists())

    def test_missing_identity_and_shared_but_wrong_split_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td); manifest, paths = self.fixture(tmp, seeds=[42])
            for path in paths.values(): self.mutate(path, lambda m: m["identity"].update(split_sha256="same_false_hash"))
            self.assertEqual(self.aggregate(tmp, manifest)["coverage"]["complete_summary_cells"], 0)
            self.mutate(paths[("singlelr", 42, 0)], lambda m: m.pop("identity"))
            self.assertEqual(self.aggregate(tmp, manifest)["status"], "incomplete")

    def test_missing_frozen_tasks_or_task_file_tampering_fails_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td); manifest, _ = self.fixture(tmp, seeds=[42])
            absent = copy.deepcopy(manifest); absent.pop("task_files")
            with self.assertRaisesRegex(ValueError, "frozen task_files"): self.aggregate(tmp, absent)
            entry = next(iter(manifest["task_files"].values())); Path(entry["path"]).write_text("[]\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"): self.aggregate(tmp, manifest)
            entry.pop("sha256")
            with self.assertRaisesRegex(ValueError, "count mismatch"): self.aggregate(tmp, manifest)

    def test_duplicate_frozen_unit_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td); manifest, _ = self.fixture(tmp, seeds=[42])
            entry = next(iter(manifest["task_files"].values())); path = Path(entry["path"]); rows = path.read_text().splitlines()
            path.write_text("\n".join(rows + rows[:1]) + "\n"); entry.pop("sha256"); entry["count"] += 1
            with self.assertRaisesRegex(ValueError, "duplicate frozen task unit"): self.aggregate(tmp, manifest)

    def test_frozen_split_record_and_real_array_digest_both_required(self):
        for mode in ["no-record", "wrong-declared-hash", "tamper-real-array"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                tmp = Path(td); manifest, _ = self.fixture(tmp, seeds=[42])
                if mode == "no-record": manifest["splits"] = []
                elif mode == "wrong-declared-hash": manifest["splits"][0]["split_sha256"] = "wrong"
                else:
                    p = Path(manifest["splits"][0]["path"])
                    with np.load(p, allow_pickle=False) as z: d = {k: z[k] for k in z.files}
                    d["train_sub_idx"] = d["train_sub_idx"][::-1]; atomic_savez(p, **d)
                result = self.aggregate(tmp, manifest)
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["coverage"]["complete_summary_cells"], 0)

    def test_legacy_missing_version_keeps_true_identity_and_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td); manifest, paths = self.fixture(tmp, seeds=[42])
            def legacy(m):
                m["identity"].pop("validation_protocol_version"); m["method_details"] = {}
            for path in paths.values(): self.mutate(path, legacy)
            original = {p: p.read_bytes() for p in paths.values()}
            self.assertEqual(self.aggregate(tmp, manifest)["status"], "complete")
            self.assertTrue(all(p.read_bytes() == b for p, b in original.items()))
            with np.load(paths[("singlelr", 42, 0)].parents[2] / "singlelr_results.npz", allow_pickle=False) as z:
                self.assertNotIn("val_acc_pre_refit", z.files)
                self.assertEqual(json.loads(str(z["metadata_json"]))["validation_source"], "legacy_source_audit_required")

    def test_declared_version_is_required_and_unknown_implicit_version_rejected(self):
        for declared in [True, False, "unknown-explicit"]:
            with self.subTest(declared=declared), tempfile.TemporaryDirectory() as td:
                tmp = Path(td); manifest, paths = self.fixture(tmp, seeds=[42])
                if declared == "unknown-explicit":
                    manifest["protocol"]["validation_protocol_version"] = "unknown"
                    self.mutate(paths[("singlelr", 42, 0)], lambda m: m["identity"].update(validation_protocol_version="unknown"))
                elif declared:
                    manifest["protocol"]["validation_protocol_version"] = "pre_refit_validation_v1"
                    self.mutate(paths[("singlelr", 42, 0)], lambda m: m["identity"].pop("validation_protocol_version"))
                else: self.mutate(paths[("singlelr", 42, 0)], lambda m: m["identity"].update(validation_protocol_version="unknown"))
                self.assertEqual(self.aggregate(tmp, manifest)["status"], "incomplete")

    def test_grouped_split_is_checked_against_actual_subjects_and_kept_secondary(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td); manifest, _ = self.fixture(tmp, seeds=[42])
            embedding = manifest["embeddings"][0]
            embedding.update(dataset="counterfact", path=str(tmp / "model_counterfact_embeddings.npz"))
            manifest["grid"]["datasets"] = ["counterfact"]
            entry = next(iter(manifest["task_files"].values())); task_path = Path(entry["path"])
            tasks = [json.loads(line) for line in task_path.read_text().splitlines()]
            groups = np.asarray(["s" + str(i) for i in range(20)] * 2)
            directory = seed_directory(Path(manifest["output_root"]), GROUPED_PROTOCOL, "model", "counterfact", 42)
            split_path = directory / "splits.npz"
            split = create_split_artifact(
                split_path, y=np.r_[np.ones(20, dtype=int), np.zeros(20, dtype=int)],
                protocol=GROUPED_PROTOCOL, seed=42, test_size=.2, val_size=.2,
                model="model", dataset="counterfact", embedding=embedding, groups=groups,
            )
            manifest["splits"] = [{"path": str(split_path), "model": "model", "dataset": "counterfact",
                                    "protocol": GROUPED_PROTOCOL, "seed": 42, "split_sha256": split["split_sha256"]}]
            for task in tasks:
                task.update(protocol=GROUPED_PROTOCOL, dataset="counterfact", embedding=embedding)
                for layer in [0, 1]:
                    metadata = {"status": "complete", "identity": _checkpoint_identity(task, layer=layer, split_metadata=split)}
                    atomic_savez(checkpoint_path(directory, task["method"], layer), validation_accuracy=np.asarray(.9),
                                 test_accuracy=np.asarray(.6), concept=np.ones(5), metadata_json=np.asarray(json.dumps(metadata)))
            task_path.write_text("".join(json.dumps(t) + "\n" for t in tasks))
            entry["sha256"] = hashlib.sha256(task_path.read_bytes()).hexdigest()
            with patch("raptor.experiments.accuracy_aggregation._load_group_ids", return_value=groups):
                result = self.aggregate(tmp, manifest)
            self.assertEqual(result["status"], "incomplete")  # Primary rowwise remains absent.
            self.assertEqual(result["coverage"]["complete_summary_cells"], 6)
            self.assertTrue(all(cell["protocol"] == GROUPED_PROTOCOL for cell in result["cells"] if cell["status"] == "complete"))
            with patch("raptor.experiments.accuracy_aggregation._load_group_ids", return_value=np.repeat("same_subject", 40)):
                self.assertEqual(self.aggregate(tmp, manifest)["coverage"]["complete_summary_cells"], 0)

    def test_private_output_redirect_reads_frozen_original_split_without_touching_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            import shutil
            tmp = Path(td); manifest, paths = self.fixture(tmp, seeds=[42])
            protected = {p: p.read_bytes() for p in paths.values()}
            protected.update({Path(r["path"]): Path(r["path"]).read_bytes() for r in manifest["splits"]})
            private = copy.deepcopy(manifest); private["output_root"] = str(tmp / "private")
            for (method, seed, layer), source in paths.items():
                target = checkpoint_path(seed_directory(Path(private["output_root"]), ROW_PROTOCOL, "model", "STSA", seed), method, layer)
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
            self.assertEqual(self.aggregate(tmp, private)["status"], "complete")
            self.assertTrue(all(p.read_bytes() == b for p, b in protected.items()))
            self.assertFalse((Path(manifest["output_root"]) / "rowwise/model/STSA/seed_42/singlelr_results.npz").exists())


if __name__ == "__main__": unittest.main()
