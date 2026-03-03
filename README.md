# RAPTOR: RIDGE-ADAPTIVE LOGISTIC PROBES

This repository contains the minimal code and data needed to reproduce the `run_experiments.py` pipeline.

## Reproduce

1. Create the environment:

```bash
conda env create -f environment.yml
conda activate cdenv
```

2. Run experiments:

```bash
python run_experiments.py
```

Outputs are written to:
- `./embeddings_all`
- `./exp_results`

## Included scope

This repo intentionally keeps only the files required for the `run_experiments.py` chain:
- embedding generation
- SingleLR / xRFM / GCS runs
- required default datasets and xRFM dependency code
