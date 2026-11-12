# Accuracy protocol and fresh reproduction

The default primary grid has six models, six datasets, five methods, and outer
split seeds 42–46. Llama-3.3-70B is deferred and excluded. These commands prepare
new inputs and splits; they require no historical results, frozen internal
validation decisions, cluster directory, or historical Git HEAD.

```bash
python scripts/prepare_accuracy_multiseed.py --emb-dir embeddings_all \
  --results-root exp_results/accuracy_release --task-dir exp_results/accuracy_release/tasks
python scripts/run_accuracy.py --manifest exp_results/accuracy_release/tasks/manifest.json --dry-run
python scripts/run_accuracy.py --manifest exp_results/accuracy_release/tasks/manifest.json
python scripts/aggregate_accuracy_multiseed.py \
  --manifest exp_results/accuracy_release/tasks/manifest.json \
  --out-json exp_results/accuracy_release/summary.json \
  --out-csv exp_results/accuracy_release/summary.csv
python scripts/plotting/plot_accuracy_multiseed.py \
  --summary exp_results/accuracy_release/summary.json --out-dir plots/accuracy_release
```

Use `--models`, `--datasets`, `--methods`, `--seeds`, and `--protocols` on the
preparation command to select an explicit grid. A single task can run through
`python scripts/run_accuracy_task.py --task-file <prepared.jsonl> --task-index 0`.
A completed layer is restored only when its split, embedding, method seed,
configuration, scientific source, and environment identity match. Atomic files
and unique per-method/per-layer paths isolate seeds and methods. Corrupt or
incompatible checkpoints are preserved with a `.rejected.*` suffix before a
recomputation. Signals are observed after a completed atomic layer checkpoint;
the CLI exits 75 so a scheduler may resume explicitly. No job is submitted.

Preparation, execution and aggregation require the fixed physical layer count
of each benchmark model. A consecutive prefix is incomplete even if a manifest
or cache declares that shorter count. Existing model, dataset, layer-count and
feature-dimension metadata must match the actual input. Metadata-free historical
caches still require all physical layers. Tiny custom offline fixtures require
`--synthetic-canary`; this labels their manifests/results and cannot shorten a
known benchmark model.

Splits are stratified by row: 20% test, then 20% of the remainder validation
(approximately 64/16/20). Every method shares the exact stored indices in one
model–dataset–seed setting. Class blocks in embeddings remain positive first,
then negative. Preprocessing is fitted only to candidate training rows.

| Method ID | Candidate and validation | Final test probe |
|---|---|---|
| `singlelr` (RAPTOR) | Train-subset StandardScaler; 100 warm-start C candidates in `logspace(-4,2,100)`, max_iter 1000; highest outer-validation accuracy, first C on exact ties | Fresh lbfgs/L2 refit on train+validation at selected C, retaining train-only scaler; max_iter 1000 |
| `fixed_reg` | Same native candidate with predeclared sklearn C=1; no regularization search | Same native refit and unchanged C=1 |
| `xrfm` | Shared outer validation explicitly supplied; native six-candidate regularization/bandwidth/centering search; 10 RFM iterations; standardized features | Selected native model; no refit |
| `gcs` | Native 1000 sampled probes from train-subset; internal training fraction .1 and validation fraction .3; max_iter100, early_loops10, threshold .9; 1000 sampled readouts | Identical sampled probes score outer validation and test; mean individual-probe accuracy; no refit |
| `dim_midpoint` (CAA accuracy) | Raw FP64 class means on train-subset; W=mu_positive−mu_negative; midpoint=(mu_positive+mu_negative)/2; score=W dot (h−midpoint); score>=0 predicts positive | Re-estimate raw class means on train+validation; no logistic regression or threshold tuning |

For the native sklearn/lbfgs implementation, fixed C=1 minimizes an objective
equivalent to `sum(logistic_loss) + ||w||²/(2*C)` with an unpenalized intercept.
For the paper's mean-loss objective with ridge term `lambda*||w||²/2`,
`lambda=1/(C*n_fit)`. The implementation preserves C at refit, so this normalized
lambda changes with the fitted sample count. It is not a sample-independent
`1/C`. This also explains Fixed-reg's pre-refit lambda=1/n_train.

Each layer retains its own train-only, pre-refit outer-validation accuracy.
The best layer maximizes those validation scores; exact ties select the lowest
stored layer index. Test scores never select C, a layer, a retry, or a run.
All fixed layers contribute equally to each seed's layer-average accuracy.
Only after reducing layers to one scalar per outer seed do summaries compute
arithmetic mean and sample SD (`ddof=1`). RAPTOR-minus-baseline comparisons use
matched seed differences and a two-sided 95% Student-t interval with n−1 degrees
of freedom. The exact seeds, completed count, and every seed value are saved.
Missing layers or seeds produce explicit incomplete cells; the aggregation CLI
exits nonzero by default. Plotting consumes these stored summaries.

The separate `grouped_split_sensitivity` protocol is restricted to RAPTOR, xRFM,
and GCS. CounterFact uses `subject`; Cities uses `city`. CSV IDs are loaded in
positive-then-negative order and truncated to actual cached class counts.
`--group-data-dir <directory>` or `RAPTOR_DATA_DIR` selects prepared source CSVs.
GroupShuffleSplit first reserves 20% of groups for test, then 20% of remaining
groups for validation (seed+1 for the inner step); realized row proportions may
vary. Group intersections across every partition pair must be empty. Grouped
results have their own protocol directories, tables and labels and do not
replace the primary rowwise five-method results.

The numerical cores were selectively consolidated from audited scientific
commit `68faa5f84ce7b6a09c4f18fdd55b45156b219a1d`. Historical midpoint production
used `253d8a7456a3c0533ba69050d6dfe4d1604511b3`; the published helper is the
reviewed equivalent, without historical import/reuse orchestration. New runs
record the current release scientific-package hash, observed commit if Git is
available, complete method configurations, embedding SHA-256, and environment.
Package hashing is independent of installation directory; Git HEAD alone is
not a prerequisite for running an installed wheel. This release has small
synthetic/protocol tests; a fresh full-grid training run was not performed.
