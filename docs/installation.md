# Installation and offline checks

## Core RAPTOR

Use Python 3.9–3.12; Python 3.11 is the CI configuration. The core installation
runs RAPTOR, Fixed-reg, GCS, midpoint, data validation and aggregation on existing
embeddings. The [README Quick Start](../README.md#quick-start) needs only this core.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python examples/minimal_probe.py
```

Run these commands from the cloned repository. On Windows, activate the environment
with `.venv\Scripts\Activate.ps1` in PowerShell instead. The example creates only
small synthetic arrays, calls the real RAPTOR implementation and prints its
selected C, held-out accuracy and raw direction shape. It downloads no model and
writes no benchmark outputs. Installation itself requires access to package
indexes or locally available wheels.

## Experiment dependencies

Choose extras for the workflow you intend to run:

| Extra | Adds | Used by |
|---|---|---|
| `xrfm` | PyTorch 2.8.0 and xRFM 0.4.3 | xRFM accuracy/robustness; five-method timing |
| `extract` | Transformers, Datasets, Accelerate, SentencePiece | Embedding extraction and model generation |
| `quant` | bitsandbytes | Quantized model loading |
| `plots` | Matplotlib 3.9.4 | Figures from new summaries |
| `judge` | Optional API client | Explicitly requested judging workflows |
| `test` | pytest and build | Protocol tests and package checks |

For the CPU five-method workflow, install the CPU PyTorch wheel first:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[xrfm,plots,test]'
```

Add `extract` when loading language models. For all extraction, plotting and
protocol-test dependencies:

```bash
python -m pip install -e '.[xrfm,extract,plots,test]'
# Only when the chosen model needs quantized loading:
python -m pip install -e '.[quant]'
```

GPU extraction requires a compatible PyTorch/CUDA driver installation and the
model-specific precision settings in [data.md](data.md). CPU timing is the
validated public route; a full GPU timing workflow has not been validated.
Gated models require access approval and authentication outside the repository.
Do not commit credentials or model caches.

## Offline protocol checks

Install `xrfm,extract,plots,test` before running both checks below; the full test
suite includes model-interface tests that need `extract`. The canary runs all five
actual methods at their native per-layer budgets on a synthetic cache. It computes 50 records,
restores all 50, strictly aggregates and exports PNG/PDF/SVG. Use a fresh output
path each time. No language model or judge is called.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/run_timing.py canary --output outputs/timing_synthetic_canary
python -m pytest -q
```

The canary alone needs only `xrfm,plots,test`. Synthetic success validates
execution and checkpoint behavior, not complete benchmark reproduction or a speed
ranking.

## Versions and validation boundaries

The numerical implementation pins NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.6.1,
and pandas 2.2.3 in [pyproject.toml](../pyproject.toml). The prior local numerical
validation used Python 3.9.21; [requirements-tested.txt](../requirements-tested.txt)
records that environment. The [validation record](validation.md) distinguishes
isolated core installation, installed-wheel checks, current-source tests and CI.

The original README's xRFM Git pin declared version 0.1.0, while the inspected
production/test implementation used 0.4.3. They are not asserted to be equivalent;
[assets.md](assets.md) retains the dependency investigation and attribution.
Historical timing resource/clock limitations remain in [timing.md](timing.md).
