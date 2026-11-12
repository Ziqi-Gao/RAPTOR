"""Passive six-trial ledger retained from the native fair-timing producer.

The observer forwards every native call and includes its overhead in the clock.
CPU backend only; no search failures are hidden in a successful timing result.
"""
from __future__ import annotations
import sys
import weakref
import numpy as np
import torch
from raptor.probes import xrfm as xrfm_probe

def _scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value if isinstance(value, (str, int, float, bool, type(None))) else None


class XRFMLedger:
    """Forward native calls unchanged; expose silently caught trial failures.

    The entry-only frame hook disables itself before the first native search
    instruction. No profiler runs inside any RFM fit or prediction. This follows
    the already reviewed controlled xRFM timing observer.
    """
    def __init__(self, validation_rows):
        self.validation_rows = validation_rows
        self.trials, self.instances, self.search_calls, self.errors = [], {}, [], []

    def __enter__(self):
        if sys.getprofile() is not None:
            raise RuntimeError("timing refuses an active profiler")
        owner, cls = self, xrfm_probe._CompatibleRFM
        self.originals = (cls.__init__, cls.fit, cls.predict, xrfm_probe._train_rfm_like_repo,
                          cls._compute_validation_metrics)
        old_init, old_fit, old_predict, old_search, old_metrics = self.originals

        def init(model, *args, **kwargs):
            row = {"constructor": dict(kwargs), "fit_calls": 0, "fit_completed": False,
                   "outer_validation_calls": 0, "outer_validation_finite": True,
                   "inner_predict_calls": 0, "iteration_validation_calls": [], "exceptions": []}
            owner.trials.append(row)
            owner.instances[id(model)] = (weakref.ref(model), row)
            try:
                return old_init(model, *args, **kwargs)
            except BaseException as exc:
                row["exceptions"].append("init: " + repr(exc))
                raise

        def row_for(model):
            entry = owner.instances.get(id(model))
            return entry[1] if entry and entry[0]() is model else None

        def fit(model, *args, **kwargs):
            row = row_for(model)
            if row is None:
                raise RuntimeError("unregistered native xRFM fit")
            row.update({"fit_calls": row["fit_calls"] + 1, "fit_kwargs": dict(kwargs),
                        "native_compute_profiler_disabled": sys.getprofile() is None,
                        "train_shape": list(args[0][0].shape),
                        "validation_shape": list(args[1][0].shape),
                        "train_device": str(args[0][0].device),
                        "validation_device": str(args[1][0].device), "in_fit": True})
            try:
                result = old_fit(model, *args, **kwargs)
                row.update({"fit_completed": True, "actual_device": str(model.device),
                            "native_best_iter_if_exposed": _scalar(getattr(model, "best_iter", None))})
                return result
            except BaseException as exc:
                row["exceptions"].append("fit: " + repr(exc))
                raise
            finally:
                row["in_fit"] = False

        def predict(model, *args, **kwargs):
            row = row_for(model)
            try:
                result = old_predict(model, *args, **kwargs)
            except BaseException as exc:
                if row is not None:
                    row["exceptions"].append("predict: " + repr(exc))
                raise
            if row is not None:
                if row.get("in_fit"):
                    row["inner_predict_calls"] += 1
                elif row["fit_completed"]:
                    row["outer_validation_calls"] += 1
                    row["outer_validation_rows"] = int(args[0].shape[0])
                    finite = bool(torch.isfinite(result).all().item()) if torch.is_tensor(result) else bool(np.isfinite(result).all())
                    row["outer_validation_finite"] &= finite
            return result

        def metrics(model, *args, **kwargs):
            row = row_for(model)
            if row is not None:
                row["iteration_validation_calls"].append({
                    "iteration_num": _scalar(kwargs.get("iteration_num")),
                    "is_final": bool(kwargs.get("is_final", False)),
                    "native_compute_profiler_disabled": sys.getprofile() is None})
            return old_metrics(model, *args, **kwargs)

        def search(*args, **kwargs):
            if sys.getprofile() is not None:
                raise RuntimeError("refuse to replace an active profiler")
            captured = []
            entry = {"entry_captures": 0, "caught_errors": None, "frame_cleared": False,
                     "entry_only_profile_hook": True}
            owner.search_calls.append(entry)

            def capture(frame, event, arg):
                if event == "call" and frame.f_code is old_search.__code__:
                    captured.append(frame)
                    entry["entry_captures"] += 1
                    sys.setprofile(None)
            sys.setprofile(capture)
            try:
                return old_search(*args, **kwargs)
            finally:
                sys.setprofile(None)
                if len(captured) == 1:
                    frame = captured.pop()
                    try:
                        entry["caught_errors"] = list(frame.f_locals["caught_errors"])
                        frame.clear()
                        entry["frame_cleared"] = not bool(frame.f_locals)
                    except BaseException as exc:
                        owner.errors.append("frame snapshot/clear: " + repr(exc))
                    del frame
                else:
                    owner.errors.append("search entry not captured exactly once")

        cls.__init__, cls.fit, cls.predict = init, fit, predict
        cls._compute_validation_metrics = metrics
        xrfm_probe._train_rfm_like_repo = search
        return self

    def __exit__(self, typ, exc, tb):
        cls = xrfm_probe._CompatibleRFM
        cls.__init__, cls.fit, cls.predict, xrfm_probe._train_rfm_like_repo, cls._compute_validation_metrics = self.originals
        self.instances.clear()

    def snapshot(self):
        expected = [[.001, bw, center] for bw in (1, 10, 100) for center in (True, False)]
        keys = [[row.get("fit_kwargs", {}).get("reg"), row["constructor"].get("bandwidth"),
                 row.get("fit_kwargs", {}).get("center_grads")] for row in self.trials]
        valid = keys == expected and len(self.search_calls) == 1 and not self.errors
        for search in self.search_calls:
            valid = valid and search["entry_captures"] == 1 and search["caught_errors"] == [] and search["frame_cleared"]
        for row in self.trials:
            valid = valid and row["fit_calls"] == 1 and row["fit_completed"] and not row["exceptions"]
            valid = valid and row["native_compute_profiler_disabled"]
            valid = valid and bool(row["iteration_validation_calls"])
            valid = valid and all(call["native_compute_profiler_disabled"] for call in row["iteration_validation_calls"])
            valid = valid and row["train_device"] == row["validation_device"] == row["actual_device"] == "cpu"
            valid = valid and row["constructor"].get("device") == "cpu"
            valid = valid and row["constructor"].get("kernel") == "l2_high_dim"
            valid = valid and row["constructor"].get("tuning_metric") == "accuracy"
            valid = valid and row["outer_validation_calls"] == 1 and row["outer_validation_finite"]
            valid = valid and row.get("outer_validation_rows") == self.validation_rows
            valid = valid and row["fit_kwargs"] == {"reg": .001, "iters": 10,
                "center_grads": row["fit_kwargs"]["center_grads"], "early_stop_rfm": True,
                "get_agop_best_model": True, "top_k": 1}
        return {"six_trial_complete": bool(valid), "expected_trial_keys": expected,
                "observed_trial_keys": keys, "trials": self.trials,
                "search_calls": self.search_calls, "observer_errors": self.errors,
                "instrumentation": "native calls unchanged; entry hook disabled during computation; overhead included"}

