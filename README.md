# RAPTOR: Ridge-Adaptive Logistic Probes

Ziqi Gao, Yaotian Zhu, Qingcheng Zeng, Xu Zhao, Ziqing Wang, Feng Ruan, Kaize Ding

[Paper: camera-ready arXiv v4](https://arxiv.org/abs/2602.00158v4) ·
[Paper-to-code map](docs/paper_map.csv) · [Protocols and provenance](docs/provenance.md)

RAPTOR fits a validation-tuned ridge-logistic probe to frozen layer
representations and folds its weights back to the raw activation coordinates.
This repository provides the paper's accuracy, directional robustness,
computational cost, steering, separability, and structure-diagnostic workflows.
The public paper marks this version as accepted at NeurIPS 2026. No proceedings
DOI or separate release certification is asserted.

The primary accuracy scope is **six models × six datasets × five methods × five
outer split seeds 42–46**. Llama 3.3-70B is deferred. CAA accuracy means raw
mean difference with Euclidean midpoint readout; its robustness and calibrated
steering definitions are different. Best layers are selected using each method's
pre-refit outer-validation accuracy, with lowest-layer exact tie breaking.

## Installation

Python 3.9.21 and the exact versions in [requirements-tested.txt](requirements-tested.txt)
were used for local synthetic validation. Python 3.11 is the CI configuration.
The package pins NumPy 1.26.4, SciPy 1.13.1, sklearn 1.6.1, and pandas 2.2.3 to preserve
the inspected numerical implementation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[xrfm,extract,plots,test]'
```

Core-only installation is `python -m pip install -e .`; it supports RAPTOR,
Fixed-reg, GCS, midpoint, data validation, and aggregation on existing embeddings.
Extras are `xrfm` (PyTorch 2.8.0/xRFM 0.4.3), `extract` (model loading/generation),
`quant` (bitsandbytes), `plots`, `judge` (optional API client), and `test`.
For CPU-only PyTorch, install its CPU wheel before the extras:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

GPU extraction requires a compatible PyTorch/CUDA driver installation; quantized
models also need `python -m pip install -e '.[quant]'`. CPU timing is the validated
public route. No GPU timing validation or full model extraction is claimed.
The old README's xRFM Git pin declared 0.1.0, while actual production/test code used
0.4.3; the release records this discrepancy rather than presenting them as identical.

## Data and model preparation

See [data.md](docs/data.md) for all six exact source schemas, ordering, truncation,
class mapping, grouped keys, historical file hashes, cache format and access limits.
Raw dataset text and weights are not redistributed. After obtaining the documented
inputs, validate them without downloading anything:

```bash
python scripts/prepare_data.py --data-root dataset --output outputs/data_identity.json
python scripts/run_embeddings.py --models Qwen/Qwen2.5-3B-Instruct --datasets STSA \
  --model_path models --emb_dir embeddings_all --quant 32
```

Embeddings use the final-token output of `post_attention_layernorm` at every
transformer layer, stored as FP32, positive rows then negative rows. See the
model-specific precision overrides in the data guide. Existing compatible
embeddings can be reused for every probe split seed.

## Quick offline check

This runs all **five actual methods**, full native per-layer budgets, on a small
synthetic embedding cache. It saves 50 records, resumes them without fitting again,
aggregates and plots. It downloads no models and calls no judge.

```bash
CUDA_VISIBLE_DEVICES='' python scripts/run_timing.py canary \
  --output outputs/timing_synthetic_canary
python -m pytest -q
```

Synthetic success establishes executable behavior and checkpoint compatibility;
it does not reproduce the complete benchmark or establish a speed ranking.

## Accuracy and grouped sensitivity

```bash
python scripts/prepare_accuracy_multiseed.py --emb-dir embeddings_all \
  --seeds 42,43,44,45,46 --results-root exp_results/accuracy_release \
  --task-dir exp_results/accuracy_release/tasks
python scripts/run_accuracy.py --manifest exp_results/accuracy_release/tasks/manifest.json --dry-run
# Expensive: run only when ready for the complete prepared grid.
python scripts/run_accuracy.py --manifest exp_results/accuracy_release/tasks/manifest.json
python scripts/aggregate_accuracy_multiseed.py \
  --manifest exp_results/accuracy_release/tasks/manifest.json \
  --out-json exp_results/accuracy_release/summary.json \
  --out-csv exp_results/accuracy_release/summary.csv
python scripts/plotting/plot_accuracy_multiseed.py \
  --summary exp_results/accuracy_release/summary.json --view all --out-dir plots/accuracy_release
```

Preparation defaults to primary rowwise five-method accuracy plus separately
labelled CounterFact/Cities grouped sensitivity for the three original methods.
Grouped runs require the aligned source CSVs. Both protocols preserve ~64/16/20
train/validation/test intent; grouped row counts can vary. [Accuracy details](docs/accuracy.md)
explain shared splits, native refit differences, fixed C=1 normalization, full-layer
means, sample SD, paired 95% Student-t intervals, and incomplete-grid rejection.

## Directional robustness

```bash
python scripts/run_robustness.py prepare --emb-dir embeddings_all --output exp_results/robustness_release
python scripts/run_robustness.py run --manifest exp_results/robustness_release/manifest.json --dry-run
python scripts/run_robustness.py run --manifest exp_results/robustness_release/manifest.json
python scripts/run_robustness.py aggregate --manifest exp_results/robustness_release/manifest.json
python scripts/plotting/plot_robustness.py --summary exp_results/robustness_release/summary.json \
  --output plots/robustness_release.pdf
```

The [robustness protocol](docs/robustness.md) covers 18 settings/768 layers per method,
K=20 perturbations and 190 pairs per layer. It preserves raw RAPTOR/GCS/Fixed-reg/CAA
directions and historical standardized-space xRFM AGOP directions. Intercepts
are excluded. Finite iteration-limit outputs remain in the denominator, including
the reported 16 Fixed-reg directions at Gemma/STSA's first layer.

## Five-method computational cost

One public entrypoint runs all five native methods from fresh manifests and
cached inputs; it does not import old validation decisions or paper timings.

```bash
python scripts/run_timing.py plan --config configs/timing.yaml --embeddings embeddings_all \
  --output outputs/timing_plan.json
python scripts/run_timing.py prepare --config configs/timing.yaml --embeddings embeddings_all \
  --seeds 42,43,44,45,46 --output outputs/timing_new_run
CUDA_VISIBLE_DEVICES='' python scripts/run_timing.py run --manifest outputs/timing_new_run/manifest.json --all
python scripts/run_timing.py aggregate --manifest outputs/timing_new_run/manifest.json
python scripts/plot_timing.py --aggregate outputs/timing_new_run/aggregate.json --output plots/timing_new_run
```

This full grid has 40,200 method observations. [Timing details](docs/timing.md)
state the exact clocks, native budgets, CPU1 protocol, hardware/thread metadata,
checkpoint boundaries, and median of layers → mean of five seeds → median of 36 settings summary.
The published five-method figure mixes historical sources; the new strict
aggregator rejects that reference data as a new run. The author's confirmation
that hardware/workflow were unchanged is preserved separately from unavailable
historical clock/thread/seed/allocation/producer evidence.

## Steering and appendix experiments

[The paper map](docs/paper_map.csv) lists every body/appendix experiment with its
command, input, output, producer and validation boundary. [Steering](docs/appendix_and_steering.md)
keeps original control statistics separate from the published four-metric 1–10 judge
and later optional calibrated/six-metric work. [Appendix diagnostics](docs/appendix_and_steering.md)
cover linear separability and the 12 setting structure experiment. These commands
provide new executable workflows; old missing alpha/probability trajectories and
old judge-score provenance cannot be recovered from generated text.

## Reported results and verification

```bash
python scripts/recompute_reported.py --input reported_results --output outputs/reported_recomputed.json
```

[reported_results](reported_results/README.md) contains small numeric reference
CSVs with sanitized source identities, separate from new run outputs. The command
checks 900 seed rows, shared splits, 420 summaries including descriptive dataset means,
and 288 paired comparisons using the common statistics implementation. It runs no
training. [Validation](docs/validation.md) distinguishes tests run, workflows
provided but not fully run, and unresolved historical provenance.

This release did **not** run a full benchmark, extract large-model activations,
generate text, call a paid judge, or submit/change Slurm jobs. It does not claim
that the reorganized code has newly reproduced every paper number. The public
v4 was fully available for audit; the owner's local ZIP was not accessible here.

## License and citation

The owner explicitly deferred the author-code license choice. [LICENSE](LICENSE)
is a pending-status notice, **not an open-source license grant**. This must be
resolved before an open-source release. [Third-party notes](docs/assets.md) retain
separate licenses/attribution and unresolved data access limits. Existing Git
history is unchanged. See [retention decisions](docs/retention.md) and
[official publication guidance](docs/publication_checklist.md).

```bibtex
@misc{gao2026raptorridgeadaptivelogisticprobes,
  title={RAPTOR: Ridge-Adaptive Logistic Probes},
  author={Ziqi Gao and Yaotian Zhu and Qingcheng Zeng and Xu Zhao and Ziqing Wang and Feng Ruan and Kaize Ding},
  year={2026},
  eprint={2602.00158},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2602.00158v4}
}
```
