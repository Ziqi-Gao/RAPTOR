# Computational cost: one five-method workflow

The public runner measures RAPTOR, xRFM, GCS, Fixed-reg, and CAA from raw cached
embeddings. It prepares its own inputs, persistent row splits, and run identity.
It requires no paper plot, previous accuracy result, frozen validation decision,
particular Git HEAD, or historical result directory.

The full configuration is `configs/timing.yaml`: six models, six datasets,
seeds 42–46, and every physical layer. Llama-3.3-70B remains deferred. There are
1,608 setting–layer pairs, 8,040 layer–seed tasks, and 40,200 method observations.
Each task runs all five methods in a deterministic rotated order. Input rows
remain positive then negative, matching the embedding cache. Shared rowwise
outer splits have approximately64/16/20% train/validation/test.

## Native procedures and budgets

| Method | Native work inside each measured layer | Final fit |
|---|---|---|
| RAPTOR | Train-only scaler; 100 ascending logspace C values from1e-4 to100, warm-start logistic fits, each max_iter1000; own outer-validation C selection | Refit train+validation, same train-only scaler, max_iter1000 |
| xRFM | Train-only standardization; six trials: reg.001 × bandwidth1/10/100 × center_grads true/false; each at most10 native RFM iterations with native validation stopping; AGOP direction | No refit; selected candidate scores test |
| GCS |1,000 native subsamples of train-subset at fraction.1; internal validation fraction.3; logistic max_iter100; native early_loops10/val_thresh.9; 1,000 ring samples | No refit; identical sampled probes score outer validation and test |
| Fixed-reg | Native sklearn logistic C=1; train-only scaler; candidate and pre-refit outer-validation scoring; no regularization search | Native second fit on train+validation, same train-only scaler |
| CAA | Raw FP64 class means, raw mean difference, Euclidean midpoint, pre-refit validation; no regression | Recompute means/midpoint on train+validation; score test |

These budgets are defined centrally in
`raptor.experiments.accuracy_multiseed.METHOD_HPARAMETERS`. The timing workflow
rejects changed budgets within a prepared manifest. Native GCS stopping and
native xRFM early stopping remain part of their procedures; setting a maximum
does not assert that every fit reached it. Convergence-limit outputs are retained.
The passive xRFM ledger forwards the actual native calls and rejects timing
observations if any of its six search trials failed or disappeared.

The runner times each full native layer pipeline even when that layer would
not win on validation. Fresh validation scores determine a separately stored
lowest-layer tie-broken decision during aggregation; neither test scores nor a
historical decision filter timed layers. This experiment produces cost results;
it does not replace the full accuracy reporting workflow.

## Timing and resource boundary

The published implementation uses a portable CPU1 protocol: one allowed affinity
CPU, Torch intra/inter-op threads1, and native thread pools limited to1. It selects
an allowed CPU at runtime and requires no named cluster host. Set
`CUDA_VISIBLE_DEVICES=''` before starting Python so the native xRFM adapter uses
CPU. This implementation has no validated GPU timing route.

A single monotonic `perf_counter_ns` interval includes partition slicing,
train-only preprocessing, the complete native candidate search, outer validation,
prescribed refit/no-refit, concept/readout construction, and validation/test
scoring. The process CPU clock is saved too. Native return-value assembly and
passive xRFM observer overhead remain **inside** the interval. Imports, loading,
input hashing, common synthetic kernel warmup, resource collection,
checkpointing, aggregation, cross-layer selection, LLM extraction and queueing
are outside. Loading/validation, native module preload, and warmup durations are recorded separately.
The lazily referenced CAA module is explicitly loaded before any method clock.
Do not substitute job elapsed time divided by layers.

Every record contains CPU model, backend, precision description, affinity IDs and
count, effective Torch and native-pool threads, installed distribution versions and
actual imported runtime versions, Python/platform,
CUDA availability/build, clock metadata, method and split seeds, hyperparameters,
ordered split identity, embedding content identity, release source identity,
installed xRFM Python-source digest, run identity and native diagnostics.
Resource snapshots before and after each method must agree. Strict aggregation
rejects mixed CPU, backend, thread, precision, dependency or software classes.
Affinity CPU IDs and hostnames remain recorded without requiring the same
physical CPU across separate worker invocations.

## Commands

Install the timing and plotting dependencies with `pip install -e ".[xrfm,plots,test]"`, as described in the main README.
The following commands start from a fresh checkout and your prepared embedding
cache; no historical experiment artifacts are needed.

```bash
# Header-only full-grid inventory, counts and output-path uniqueness; no training.
python scripts/run_timing.py plan --config configs/timing.yaml \
  --embeddings embeddings_all --output outputs/timing_input_plan.json

# Content-hash the raw caches and create independent shared splits/manifests.
python scripts/run_timing.py prepare --config configs/timing.yaml \
  --embeddings embeddings_all --seeds 42,43,44,45,46 \
  --output outputs/timing_new_run

# Validate input/split identities and the complete task inventory; no training.
python scripts/run_timing.py validate \
  --manifest outputs/timing_new_run/manifest.json

# One checkpoint-safe task: setting, seed, physical layer; all five methods.
CUDA_VISIBLE_DEVICES='' python scripts/run_timing.py run \
  --manifest outputs/timing_new_run/manifest.json --task-index 0

# Full sequential rerun, deliberately explicit and potentially expensive.
CUDA_VISIBLE_DEVICES='' python scripts/run_timing.py run \
  --manifest outputs/timing_new_run/manifest.json --all

# Repeat the same run command to resume verified complete units.
python scripts/run_timing.py aggregate \
  --manifest outputs/timing_new_run/manifest.json
python scripts/plot_timing.py --aggregate outputs/timing_new_run/aggregate.json \
  --output outputs/timing_new_run/timing

# Offline synthetic execution of ALL actual methods, with full native budgets.
CUDA_VISIBLE_DEVICES='' python scripts/run_timing.py canary \
  --output outputs/timing_synthetic_canary
```

Manifest input references are relative to its directory. Preserve the relative
layout when transferring a run. Source compatibility uses SHA256 of release
Python files and installed xRFM source, rather than enforcing an old HEAD.
The Git commit is descriptive metadata where available. Editing executable
source requires preparing a new run; no old results are silently grandfathered.

A worker hashes each embedding file once per invocation outside all method
clocks, then validates each loaded layer and split. Results are atomic per
layer–method, with file locks against concurrent duplicate workers. Corrupt or
incompatible checkpoints are renamed as rejected records before recomputation;
failures are saved explicitly and never count as success. Interrupt/termination
signals request a stop after the current native method completes and its atomic
checkpoint flushes. The runner does not submit or automatically requeue jobs.

## Aggregation and interpretation

The strict aggregate first takes the median runtime over the **complete layer
set** within each setting–method–seed. It then takes the equal mean of the five
seed medians to produce one setting point. The displayed method diamond is the
median of all 36 setting points. Setting summaries preserve all five seed values,
seed IDs and sample SD (`ddof=1`). An absent, failed, corrupt or incompatible
unit fails aggregation. The synthetic canary is explicitly scoped synthetic and
cannot be confused with a36-setting result. Plots use log seconds and export
PNG/PDF/SVG.

The offline canary uses800 synthetic rows, 6 features, 2 layers, 5 seeds and all 50
actual native method observations. It writes records, immediately resumes all 50
without recomputation, aggregates and plots. It validates execution and protocol,
not scientific benchmark values or speed ordering.

## Relationship to reported timing

Scientific source navigation: public integration68faa5f; three-method native
fair-timing producer afff336; Fixed-reg/CAA timing scientific producer a699dcf
and final integration842ac59. The new public workflow reuses the native fitting
functions, removes historical decision/HEAD locks, and prepares fresh decisions.
It is a new rerun workflow, not a claim that its full grid has been executed.

The reported five-method timing presentation combines three original-paper
vector-recovered methods with later CPU1 Fixed-reg/CAA results. The latter
contain 16,080 native records across 36 settings and 5 seeds, with medians
0.26893072895 seconds for CAA and 8.8333318099 seconds for Fixed-reg under their
reported hierarchy. The historical new three-method fair-timing strict snapshot
was incomplete (21,804 of 24,120 accepted records; 87 of 108 setting–method points).
It does not establish completion of a unified five-method benchmark.

The author confirmed that timing hardware and procedure were unchanged; this is
author-provided information. Available original three-method logs do not
independently establish their exact resources, clocks, seeds or producer
identity. A new runner or its synthetic canary cannot retroactively provide that
evidence. Do not infer cross-source speed ratios from the combined reported
figure. New strict aggregation never imports historical reference points.
