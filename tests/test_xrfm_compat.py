from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from raptor.probes.xrfm import _CompatibleRFM, run_xrfm


class XrfmCompatibilityTest(unittest.TestCase):
    def test_missing_validation_never_falls_back_to_test(self) -> None:
        X = np.arange(40, dtype=float).reshape(10, 4)
        y = np.tile([0, 1], 5)
        for n_train, val_size in ((10, 0), (10, None), (4, 0.2)):
            with self.subTest(n_train=n_train, val_size=val_size):
                with mock.patch("raptor.probes.xrfm._train_rfm_like_repo") as train:
                    with self.assertRaisesRegex(ValueError, "test data cannot"):
                        run_xrfm(X[:n_train], y[:n_train], X, y, val_size=val_size)
                    train.assert_not_called()

    def test_light_kernel_cuda_branch_uses_runtime_class_tuple(self) -> None:
        model = _CompatibleRFM(
            kernel="l2_high_dim",
            bandwidth=1,
            tuning_metric="accuracy",
            device="cpu",
        )
        model.device = torch.device("cuda")
        target = "xrfm.rfm_src.recursive_feature_machine.memory_scaling_factor"
        with mock.patch(target, return_value=1.0):
            batch_size = model._compute_optimal_M_batch(
                n=32,
                c=1,
                d=64,
            )
        self.assertEqual(batch_size, 32)

    def test_explicit_outer_validation_disables_internal_split(self) -> None:
        captured = {}

        class FakeModel:
            device = torch.device("cpu")

            def predict(self, X):
                return torch.zeros((len(X), 1), dtype=torch.float32)

        def fake_train(train_X, train_y, val_X, val_y, **kwargs):
            captured["n_train"] = len(train_X)
            captured["n_val"] = len(val_X)
            captured["validation_source"] = kwargs["hyperparams"]["validation_source"]
            return (
                FakeModel(),
                75.0,
                {"bandwidth": 1},
                torch.ones(train_X.shape[1]),
            )

        X_train = np.arange(40, dtype=float).reshape(10, 4)
        y_train = np.tile([0, 1], 5)
        X_val = np.arange(16, dtype=float).reshape(4, 4)
        y_val = np.tile([0, 1], 2)
        X_test = np.arange(12, dtype=float).reshape(3, 4)
        y_test = np.zeros(3, dtype=int)
        with (
            mock.patch(
                "raptor.probes.xrfm._train_rfm_like_repo",
                side_effect=fake_train,
            ),
            mock.patch(
                "raptor.probes.xrfm.torch.cuda.is_available", return_value=False
            ),
        ):
            test_metric, val_metric, hparams = run_xrfm(
                X_train,
                y_train,
                X_test,
                y_test,
                X_val=X_val,
                y_val=y_val,
                random_state=42,
                return_details=True,
                summary_only=False,
            )
        self.assertEqual(captured["n_train"], 10)
        self.assertEqual(captured["n_val"], 4)
        self.assertEqual(
            captured["validation_source"],
            "explicit_outer_validation",
        )
        self.assertEqual(test_metric, 100.0)
        self.assertEqual(val_metric, 75.0)
        self.assertEqual(
            hparams["validation_source"],
            "explicit_outer_validation",
        )


if __name__ == "__main__":
    unittest.main()
