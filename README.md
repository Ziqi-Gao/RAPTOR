# RAPTOR: Ridge-Adaptive Logistic Probes

This repository accompanies the paper:

**RAPTOR: Ridge-Adaptive Logistic Probes**  
Ziqi Gao, Yaotian Zhu, Qingcheng Zeng, Xu Zhao, Ziqing Wang, Feng Ruan, Kaize Ding  
arXiv: https://arxiv.org/abs/2602.00158

## TL;DR

RAPTOR is a ridge-regularized logistic probe with validation-tuned regularization strength.  
It is designed for probe-then-steer workflows where concept vectors should be:

- accurate,
- directionally stable under small data perturbations,
- cheap to train at scale.

This repo provides the experiment pipeline centered on `run_experiments.py`.

## What This Code Reproduces

The current main pipeline reproduces the probe benchmark part of the paper:

- layer-wise probe training and evaluation,
- comparisons against xRFM and GCS,
- shared train/val/test splits per `(model, dataset)`.

Primary entrypoint:

- `run_experiments.py`

Core methods:

- RAPTOR: `run_singlelr.py`
- xRFM baseline: `run_xrfm.py`
- GCS baseline: `run_gcs.py`

Embedding extraction:

- `run_embeddings.py` and `save_embeddings.py`

## Main Findings (from paper)

From Table 1 in arXiv v2 (`2602.00158v2`):

- Across 42 `(model, dataset)` settings, RAPTOR improves **average layer accuracy** over GCS in **42/42** settings.
- RAPTOR improves **best-layer accuracy** over GCS in **41/42** settings (1 tie).
- Grid-wide mean accuracy (computed from Table 1):
  - Best-layer: RAPTOR **87.41** vs GCS **85.45** (+1.96)
  - Mean-over-layers: RAPTOR **82.08** vs GCS **79.08** (+3.00)
- Versus xRFM on the same table:
  - Best-layer: RAPTOR **87.41** vs xRFM **87.12** (+0.29)
  - Mean-over-layers: RAPTOR **82.08** vs xRFM **82.05** (+0.03)

The paper also reports competitive directional robustness and significantly lower training cost.

## Environment Setup

### 1) Create conda environment

```bash
conda env create -f environment.yml
conda activate kav311
```

If `environment.yml` contains a machine-specific `prefix`, remove that line before creating the env on a new machine.

### 2) Install xRFM (required for xRFM baseline)

```bash
pip install git+https://github.com/dmbeaglehole/xRFM.git@773fae8
```

## Data

Default datasets used by `run_experiments.py` are defined in `experiment_utils.py`:

- `STSA` -> `dataset/stsa.binary.train`
- `sarcasm` -> `dataset/sarcasm.json`
- `hatexplain` -> `data/hatexplain`
- `counterfact` -> `dataset/counterfact.csv`
- `cities` -> `dataset/cities.csv`
- `common` -> `dataset/common_claim.csv`

Default model list is also in `experiment_utils.py` (`DEFAULT_MODELS`).

## Quick Start

### Full benchmark (all default models and datasets)

```bash
python run_experiments.py \
  --model_path . \
  --cuda 0 \
  --quant 32 \
  --noise non-noise \
  --methods xrfm,singlelr,gcs
```

### Single model + dataset

```bash
python run_experiments.py \
  --models meta-llama/Meta-Llama-3.1-8B-Instruct \
  --datasets STSA \
  --methods xrfm,singlelr,gcs \
  --model_path . \
  --cuda 0
```

### Re-run methods without recomputing embeddings

```bash
python run_experiments.py \
  --skip_embeddings \
  --methods xrfm,singlelr,gcs
```

### RAPTOR only (fastest baseline check)

```bash
python run_experiments.py \
  --methods singlelr
```

## Outputs

### Embeddings

`embeddings_all/{model_tag}_{dataset}_embeddings.npz`

Contains per-layer positive/negative embeddings:

- `X_pos_0 ... X_pos_{L-1}`
- `X_neg_0 ... X_neg_{L-1}`

### Results

`exp_results/{model_tag}/{dataset}/`

- `singlelr_results.npz` (RAPTOR)
- `rfm_results.npz` and `rfm_hparams.json` (xRFM)
- `gcs_results.npz` (GCS)
- `splits.npz` (shared split indices)

## Repository Layout (core)

- `run_experiments.py` - end-to-end pipeline
- `run_embeddings.py` - embedding generation
- `run_singlelr.py` - RAPTOR probe training
- `run_xrfm.py` - xRFM baseline wrapper
- `run_gcs.py` - GCS baseline wrapper
- `run_layer_task.py` - per-layer execution helper
- `experiment_utils.py` - shared configs and utilities
- `simulate_xrfm.py` - xRFM adapter
- `probe_sampler.py` - GCS probing sampler

## Citation

```bibtex
@misc{gao2026raptorridgeadaptivelogisticprobes,
  title={RAPTOR: Ridge-Adaptive Logistic Probes},
  author={Ziqi Gao and Yaotian Zhu and Qingcheng Zeng and Xu Zhao and Ziqing Wang and Feng Ruan and Kaize Ding},
  year={2026},
  eprint={2602.00158},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2602.00158}
}
```

## Acknowledgments

- xRFM baseline: https://github.com/dmbeaglehole/xRFM
- neural controller utilities included in `neural_controllers_repo`
