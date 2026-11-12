"""Regression tests through the classification and generation call sites."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from raptor.experiments import run_raptor
from raptor.probes import tuning
from raptor.steering import generate


class ValidationProtocolTest(unittest.TestCase):
    def test_fixed_candidate_is_scored_before_refit_and_test_is_optional(self):
        X_train = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        y_train = np.asarray([0, 0, 1, 1])
        X_val = np.asarray([[1.5], [-1.5]])
        y_val = np.asarray([0, 1])
        fits = []
        original_fit = StandardScaler.fit

        def record_scaler(scaler, X, *args, **kwargs):
            fits.append(np.asarray(X).copy())
            return original_fit(scaler, X, *args, **kwargs)

        with patch.object(StandardScaler, "fit", record_scaler):
            result = run_raptor.fit_raptor_layer(
                X_train, y_train, X_val, y_val, fixed_c=1.0, seed=42
            )
        self.assertEqual(float(result["val_acc_pre_refit"]), 0.0)
        self.assertEqual(float(result["val_acc"]), 0.0)
        self.assertEqual(float(result["best_C"]), 1.0)
        self.assertIsNone(result["acc_test"])
        np.testing.assert_array_equal(fits, [X_train])
        np.testing.assert_allclose(
            X_val @ result["candidate_W"] + result["candidate_B"],
            ((X_val - result["scaler_mean"]) / result["scaler_scale"])
            @ result["candidate_W_std"] + result["candidate_B_std"],
        )
        self.assertEqual(result["validation_source"], "pre_refit_train_only")
        self.assertEqual(result["final_fit_partition"], "train_sub+validation")

    def test_classification_c_and_validation_do_not_depend_on_test_labels(self):
        X = np.asarray([[-3.0], [-2.0], [2.0], [3.0], [-1.0], [1.0], [-4.0], [4.0]])
        y = np.asarray([0, 0, 1, 1, 0, 1, 0, 1])
        args = ([X], y, np.arange(4), np.asarray([4, 5]), np.asarray([6, 7]))
        first = run_raptor._run_singlelr_all_layers(
            *args, max_iter=100, solver="lbfgs", penalty="l2", seed=42
        )
        changed_y = y.copy()
        changed_y[6:] = 1 - changed_y[6:]
        second = run_raptor._run_singlelr_all_layers(
            [X], changed_y, *args[2:], max_iter=100, solver="lbfgs", penalty="l2", seed=42
        )
        for key in ["best_C", "val_acc_pre_refit", "candidate_W", "candidate_B", "W", "B"]:
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first["acc_test"], 1 - second["acc_test"])

    def test_layer_decision_is_frozen_before_any_refit(self):
        X = np.asarray([[-3.0], [-2.0], [2.0], [3.0], [-1.0], [1.0], [-4.0], [4.0]])
        y = np.asarray([0, 0, 1, 1, 0, 1, 0, 1])
        events = []
        old_fit = run_raptor.LogisticRegression.fit

        def record_fit(model, X, y, *args, **kwargs):
            events.append(("fit", len(y)))
            return old_fit(model, X, y, *args, **kwargs)

        def frozen(decision):
            self.assertEqual(decision["selected_layer"], 0)
            self.assertNotIn(("fit", 6), events)
            events.append(("frozen", 0))

        with patch.object(run_raptor.LogisticRegression, "fit", record_fit), \
             patch.object(tuning, "RAPTOR_C_GRID", [0.1, 1.0]):
            result = run_raptor._run_singlelr_all_layers(
                [X, X], y, np.arange(4), np.asarray([4, 5]), np.asarray([6, 7]),
                max_iter=100, solver="lbfgs", penalty="l2", seed=42,
                layer_selection_callback=frozen,
            )
        self.assertEqual(events, [("fit", 4)] * 4 + [("frozen", 0), ("fit", 6), ("fit", 6)])
        self.assertEqual(result["validation_source"].item(), "pre_refit_train_only")

    def test_explicit_candidate_phase_does_not_refit_or_evaluate_test(self):
        X = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.asarray([0, 0, 1, 1])
        candidate = run_raptor.fit_raptor_layer(X, y, X, 1 - y, fixed_c=1.0, refit=False)
        self.assertEqual(candidate["fit_phase"], "candidate_only")
        self.assertEqual(candidate["n_final_train"], 4)
        final = run_raptor.refit_raptor_layer(candidate, X, y, X, 1 - y)
        self.assertEqual(final["fit_phase"], "refit_complete")
        self.assertEqual(final["n_final_train"], 8)
        self.assertEqual(final["val_acc_pre_refit"], candidate["val_acc_pre_refit"])
        self.assertEqual(final["val_acc_pre_refit"], 0.0)
        self.assertIsNone(final["acc_test"])

    def test_both_generation_entrypoints_use_multiple_pre_refit_validation_layers(self):
        for unified in [False, True]:
            with self.subTest(unified=unified), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                probe_dir = root / "model" / "STSA"
                probe_dir.mkdir(parents=True)
                np.savez(
                    probe_dir / "singlelr_results.npz",
                    val_acc_pre_refit=[0.0, 0.95, 0.96, 0.9, 0.0],
                    acc_test=[1.0, 0.1, 0.2, 0.99, 1.0],
                    validation_source="pre_refit_train_only",
                )
                config = generate.SteeringConfig(
                    savepath=root, model_id="model", model_source="model", dataset_tag="STSA",
                    cache_dir=root, output_csv=root / "out.csv", vector_kind="singlelr",
                )
                model = SimpleNamespace(
                    config=SimpleNamespace(num_hidden_layers=5, model_type="test"),
                    generate=lambda **kwargs: torch.tensor([[1, 2]]),
                )
                tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=2, decode=lambda *a, **k: "text")
                encoded = [{"index": 0, "label": "neutral_to_positive", "source": "input", "ids": torch.tensor([[1]])}]
                captured = []

                def register(*args, **kwargs):
                    captured.append(kwargs)
                    return []

                with patch.object(generate, "load_model_and_tokenizer", return_value=(model, tokenizer, "cpu")), \
                     patch.object(generate, "load_concept_vectors", return_value=[torch.ones(2)] * 5), \
                     patch.object(generate, "prepare_encoded_prompts", return_value=encoded), \
                     patch.object(generate, "detect_style_change", return_value="unused"), \
                     patch.object(generate, "register_steering_hooks", side_effect=register):
                    kwargs = {"config": config, "prompt_specs": [generate.PromptSpec("neutral", "prompt", "input")]}
                    if unified:
                        generate.generate_unified_steering_csv(**kwargs, all_manners=["neutral_to_positive"])
                    else:
                        generate.generate_bagging_steering_csv(**kwargs)
                self.assertEqual(captured[0]["layer_ids"], [1, 2])
                self.assertIn("validation_accuracies", captured[0])
                self.assertNotIn("test_accuracies", captured[0])

    def test_steering_rejects_test_only_reliability(self):
        model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace()] * 3))
        with self.assertRaisesRegex(ValueError, "validation"):
            generate.register_steering_hooks(
                model, [torch.ones(2)] * 3, [1], test_accuracies=[1.0] * 3
            )

    def test_validation_loader_rejects_legacy_scores_without_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            np.savez(root / "singlelr_results.npz", val_acc=[0.95] * 4, acc_test=[1.0] * 4)
            with self.assertRaisesRegex(ValueError, "audited pre-refit"):
                generate.load_validation_accuracies(root, "model", "STSA", 4, vector_kind="singlelr")

    def test_explicit_layers_keep_multiple_layers_and_strict_validation_threshold(self):
        active = generate.select_active_layers(
            [torch.ones(2)] * 5, [0.0, 0.95, 0.96, 0.9, 0.0],
            layers=[1, 2, 3], accuracy_threshold=0.9,
        )
        self.assertEqual(active, [1, 2])


if __name__ == "__main__":
    unittest.main()
