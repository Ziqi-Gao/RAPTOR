# RAPTOR: Ridge-Adaptive Logistic Probes

This repository contains the code for:

**RAPTOR: Ridge-Adaptive Logistic Probes**  
Ziqi Gao, Yaotian Zhu, Qingcheng Zeng, Xu Zhao, Ziqing Wang, Feng Ruan, Kaize Ding  
arXiv: <https://arxiv.org/abs/2602.00158>

RAPTOR trains ridge-regularized logistic probes on frozen LLM hidden states.
The validation split selects the regularization strength, the final probe is
refit on train plus validation data, and the learned weights are folded back to
the original hidden-state coordinate system for concept-vector use.

## Repository Layout

```text
src/raptor/
  core.py                         Shared model/dataset config, splits, I/O helpers
  data.py                         Six RAPTOR benchmark dataset loaders
  embeddings.py                   Hidden-state extraction
  model_utils.py                  LLM hidden-state tracing helper
  probes/
    tuning.py                     RAPTOR C-grid tuning
    gcs_sampler.py                GCS probe sampler
    xrfm.py                       xRFM adapter
  experiments/
    benchmark.py                  End-to-end benchmark entrypoint
    run_raptor.py                 RAPTOR layer-wise probes
    run_xrfm.py                   xRFM layer-wise baseline
    run_gcs.py                    GCS layer-wise baseline
    check_separability.py         Linear separability diagnostics
    layer_task.py                 Single-layer task helper for cluster arrays
    robustness.py                 Occlusion robustness runs
    structure_validation.py       Accuracy-structure validation
  steering/
    generate.py                   Activation steering generation
    evaluate.py                   LLM-judge steering evaluation

scripts/                          Thin command-line wrappers
scripts/plotting/                 Paper figure helpers
configs/default.yaml              Default paper grid
dataset/                          Benchmark data files and HateXplain loader
```

The computational logic from the original experiment scripts is preserved; this
cleanup reorganizes imports, removes unrelated code, and removes machine-local
paths and generated artifacts from version control.

## Setup

```bash
conda env create -f environment.yml
conda activate raptor
pip install -e .
```

The xRFM baseline is installed from the commit used by the experiments:

```bash
pip install git+https://github.com/dmbeaglehole/xRFM.git@773fae8
```

For gated Hugging Face models, authenticate outside the repository, for example
with `huggingface-cli login`. Do not commit tokens or local model caches.

## Data And Embeddings

Default datasets are defined in `src/raptor/core.py`:

- `STSA`
- `sarcasm`
- `hatexplain`
- `counterfact`
- `cities`
- `common`

Embeddings are saved as:

```text
embeddings_all/{model_tag}_{dataset}_embeddings.npz
```

Each file contains `X_pos_0 ... X_pos_{L-1}` and
`X_neg_0 ... X_neg_{L-1}`.

Generate embeddings for one setting:

```bash
python scripts/run_embeddings.py \
  --models meta-llama/Meta-Llama-3.1-8B-Instruct \
  --datasets STSA \
  --model_path . \
  --cuda 0 \
  --quant 32
```

## Probe Benchmark

Run the full default grid:

```bash
python scripts/run_experiments.py \
  --model_path . \
  --cuda 0 \
  --quant 32 \
  --methods xrfm,singlelr,gcs
```

Run methods from existing embeddings:

```bash
python scripts/run_experiments.py \
  --skip_embeddings \
  --methods xrfm,singlelr,gcs
```

Run RAPTOR only:

```bash
python scripts/run_raptor.py \
  --models meta-llama/Meta-Llama-3.1-8B-Instruct \
  --datasets STSA
```

Results are saved under:

```text
exp_results/{model_tag}/{dataset}/
  splits.npz
  singlelr_results.npz
  rfm_results.npz
  rfm_hparams.json
  gcs_results.npz
```

## Robustness, Timing, And Steering

Occlusion robustness:

```bash
python scripts/run_robustness.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset STSA \
  --layer all \
  --methods singlelr,xrfm,gcs
```

Single-layer cluster task:

```bash
python scripts/run_layer_task.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset STSA \
  --layer 10 \
  --methods singlelr,xrfm,gcs
```

Accuracy-structure validation:

```bash
python scripts/validate_accuracy_structure.py \
  --emb_npz embeddings_all/meta-llama-Meta-Llama-3.1-8B-Instruct_STSA_embeddings.npz \
  --layer 10 \
  --out exp_results/acc_structure/llama8b_stsa_layer10
```

Linear separability diagnostics:

```bash
python scripts/check_separability.py \
  --emb-dir embeddings_all \
  --datasets STSA,sarcasm \
  --out exp_results/separability
```

Activation steering:

```bash
python scripts/steer.py \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset STSA \
  --vector-kind singlelr \
  --savepath exp_results
```

LLM-judge steering evaluation:

```bash
python scripts/evaluate_steering.py \
  --in_csv outputs/steering_results.csv \
  --out_csv outputs/steering_evaluation.csv \
  --concept_desc joyful
```

Set the OpenAI API key in your shell environment before running the judge.

## Adding New Experiments

Add reusable logic under `src/raptor/experiments/` and expose it through a thin
wrapper in `scripts/`. Keep generated outputs in `exp_results/`,
`embeddings_all/`, `plots/`, or `logs/`; these paths are ignored by git.

When adding a new probe method, prefer this pattern:

1. Put reusable training code in `src/raptor/probes/{method}.py`.
2. Put grid orchestration in `src/raptor/experiments/run_{method}.py`.
3. Save results into `exp_results/{model_tag}/{dataset}/`.
4. Reuse `load_or_create_splits` from `src/raptor/core.py`.

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

## Notes

- This repository should not contain generated embeddings, model weights,
  experiment logs, API keys, Hugging Face tokens, or local absolute paths.
- The `origin` remote from the old thesis repository is not used for publishing
  this cleanup. The intended GitHub target is `git@github.com:Ziqi-Gao/RAPTOR.git`.
