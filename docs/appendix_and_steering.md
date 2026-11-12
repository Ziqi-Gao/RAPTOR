# Appendix and steering workflows

These commands use an installed `raptor` package. None submits cluster jobs. Generation commands load a real model only when executed; judge `prepare`, `verify` and aggregation commands are offline. Full experiments and paid judging were not run as part of release preparation.

## Linear separability

Pass explicit model/dataset embedding files to avoid ambiguous dataset discovery:

```bash
python scripts/check_separability.py \
  --embeddings embeddings_all/Qwen-Qwen2.5-7B-Instruct_STSA_embeddings.npz \
  --out exp_results/separability/new_run
```

The native kernels standardize all rows of each input cache, try LinearSVC at C=10,000/1,000,000/100,000,000 and run a 2,000-epoch perceptron. SVM warnings and exceptions remain visible. This is an in-sample finite-budget diagnostic; zero error is not a certificate of exact separability. The historical table's model/subset provenance is missing, so this command does not assert reproduction of its numbers.

## High-dimensional structure diagnostic

```bash
python scripts/run_structure_grid.py plan --out exp_results/structure/new_run
python scripts/run_structure_grid.py run --out exp_results/structure/new_run
python scripts/run_structure_grid.py aggregate --out exp_results/structure/new_run
```

`configs/structure_validation.json` explicitly lists two models, three datasets, layers 10/20, six fractions and five subsample seeds. The 360 fraction–repeat shards write individual completion identities. `--task-index INDEX` runs a single shard for external scheduling. Aggregation rejects missing, corrupt or mixed shards. Each setting's result averages repeats within fraction, computes correlations across the fraction sweep separately for each C, and reports the largest Spearman correlation, with lowest C breaking exact ties. This reproduces the diagnostic's descriptive selection procedure, not the main benchmark's model-selection protocol.

The released six fractions are an explicit fresh grid; the historical fraction list was not recovered. The native plug-in and cross-fitting computations are retained. CSV fields retain legacy `delta=p/n` and also provide `aspect_ratio_n_over_p`. `paper_inverse_C_label` is labelled literally; a mean-loss ridge strength is `1/(n*C)` and varies with training size. Default diagnostic fit/intercept and C-grid settings are listed separately from main RAPTOR. No Gaussian fixed-point solver is required by the published paper.

## Original steering and actual control measurements

The original normalized-direction intervention is retained as `mode=legacy_unit` in the single official steering runner. Native prompt builders remain in the retained scientific library and have their original regression coverage. Prepare fresh frozen trial IDs with:

```bash
python scripts/prepare_steering_prompts.py --dataset STSA --direction neutral_to_negative \
  --count 64 --out outputs/prompts/STSA_negative.jsonl
```

Sarcasm additionally requires `--data` pointing to the legitimately obtained JSONL source. These are explicitly prepared fresh trial IDs; historical original IDs were not recovered. Reliability filtering requires pre-refit outer-validation and never reads test scores. Existing reported control statistics remain in `docs/reported_results/steering_controls.csv`.

For portable execution with actual per-step measurements, export this method's own probe from complete fresh accuracy checkpoints. Fill `configs/steering_binding.example.json` with model/dataset/method, split and embedding SHA256 values, and the verified matching activation site. The binding is a statement about your extraction/model coordinates; changing a site name does not validate the coordinates.

```bash
python scripts/prepare_steering_probes.py export \
  --checkpoints RESULTS/checkpoints/singlelr/layer_*.npz \
  --binding MY_BINDING.json --out outputs/probes/raptor.npz
python scripts/steer.py \
  --config configs/steering.example.json --probe outputs/probes/raptor.npz \
  --prompts MY_PROMPTS.jsonl --out outputs/steering/new_run --plan-only
python scripts/steer.py \
  --config configs/steering.example.json --probe outputs/probes/raptor.npz \
  --prompts MY_PROMPTS.jsonl --out outputs/steering/new_run
```

`MY_PROMPTS.jsonl` has one object per prompt, with unique integer `prompt_id` and nonempty `text`. Keep that file fixed. Full generation seeds are recorded as `base_seed + prompt_id*100000`; use the same prompt IDs and seed list across methods. Complete physical layer mappings are required in the own-probe export. No xRFM probe is accepted.

The supplied example uses `mode=legacy_unit`, preserving the original normalized-direction classifier convention with zero global bias and block-output hooks. `mode=raw_calibrated` uses each method's own raw W/B and the approved GCAV formula, with a separately verified matching activation site. It is the optional newer generation identity, not a silent replacement for old control-table provenance. Raw CAA scalar calibration can be prepared independently with `prepare_steering_probes.py fit-caa --embedding EMB.npz --split SPLIT.npz --binding CAA_BINDING.json --out CAA.npz`; its reliability mask freezes before native train+validation refitting.

Every accepted prompt–seed record contains Original and steered text, token IDs, full seed and actual pre/post probabilities and alpha measurements for each active layer and step. Complete records are written atomically; identity-matched resumes skip them. `control_summary.json` uses the first forward step, one layer–prompt observation per active layer per prompt–seed. All-token trajectories remain available but are a different denominator. Success is measured after the actual finite-precision hook, so it may differ from the algebraic ideal. Missing historical trajectories cannot be reconstructed from text or populated with published numbers.

The paper's eight task/direction settings are CounterFact truth→lie and lie→truth, HateXplain neutral→hate and nonhate, Sarcasm neutral→sarcastic and sincere, and STSA neutral→negative and positive. Configure each task independently using its native prompt template; targets are .0001 for away and .9999 for towards. Historical thresholds are .7 for CounterFact and .8 for STSA-positive, with filtering disabled for the other four conditions and STSA-negative. The exact old prompt identities, model/run binding and control trajectories remain unavailable; the portable command enables new controlled runs, without claiming exact numerical replay of those old tables.

## Original external judge: four dimensions, 1–10

```bash
python scripts/paper_steering_judge.py collect \
  --runs outputs/steering/raptor_run outputs/steering/gcs_run \
  --out outputs/paper_judge/GENERATIONS.json
python scripts/paper_steering_judge.py prepare \
  --generations outputs/paper_judge/GENERATIONS.json --out outputs/paper_judge/new_run
python scripts/paper_steering_judge.py aggregate \
  --run-dir outputs/paper_judge/new_run --scores SCORES.json --provenance PROVENANCE.json
```

`collect` reads the two actual completed `steer.py` output directories. It verifies their manifests, every atomic unit and payload digest, final control summaries, full prompt-by-seed coverage, shared model/revision/split/embedding/prompt/configuration identity, and exact equality of shared Original text and token IDs. Method-specific probes and active layer sets remain independent. Missing, corrupt, mixed or unindexed units, inconsistent references and existing output paths are rejected. The operation only copies saved text into one Original/GCS/RAPTOR triple per full-seed pair and records source hashes in a provenance sidecar; it generates no text or scores and never chooses a better reference. Input run artifacts remain unchanged.

`GENERATIONS.json` is the collected array of complete Original/GCS/RAPTOR records, with `model_id`, `task`, `direction`, `prompt_id`, full integer `generation_seed`, `method`, `prompt_text` and `text`. A caller may also provide that explicit flat schema directly. Each pair must contain all three methods. `prepare` requires a new output directory and exports anonymous requests and a local identity mapping. This is a new portable transport; the historical judge's exact prompt and exclusion transcript were not recovered.

`SCORES.json` is an array keyed by `record_id`. Each item requires explicit boolean `evaluable` and `valid`, a nonempty `reason`, and `target_sentiment_strength`, `output_quality`, `naturalness`, `overall_steering_success` scores in [1,10]. Invalid records may have null scores; Original quality/naturalness may be null. Do not infer scores from keywords or delete failures. `PROVENANCE.json` requires `judge_model`, `execution_context`, `scoring_date`, `rubric_description` and `source_kind`; record the actual human/API/conversation process, not a wished-for producer. Aggregation retains every input record and reports only common-valid pairs, with excluded-pair counts. Published reported values, including the 160-pair table, remain separate.

## Optional six-metric judging

This experiment is not in public paper v4. Its API runner requires a fresh `DELIVERY/records.jsonl` export containing five methods, full generation-seed pairing and source identities. Each row contains `record_id` and `pair_id` (canonical JSON SHA256), the five pairing fields, `method`, `prompt_text`, `raw_output`, and `source_identity` with `checkpoint_revision`, `tokenizer_revision`, `chat_template_sha256`, and `decoding_config`. The complete optional grid is 9,000 records, 1,800 five-method pairs and 30 conditions. Original stays the unsteered reference.

```bash
python scripts/judge_steering_six_metrics.py prepare --delivery DELIVERY --run-dir outputs/optional_judge/new_run
python scripts/judge_steering_six_metrics.py verify --run-dir outputs/optional_judge/new_run
```

Preparation writes the rubric, immutable anonymous Stage A/B requests, response schemas and an endpoint/model/pricing/budget template. It does not call an API. After explicit configuration, `freeze`, `canary` and `run` provide a new API execution path with request/response metadata, bounded technical retries, accounting and checkpoint validation. `canary` and `run` incur actual requests when a user executes them. This release preparation did not execute them. A new API execution has its own provenance; it cannot reproduce the older shared-conversation Terra scores or targeted Sol overlay merely by matching a rubric.

Stage A scores target/quality/naturalness/overall without Original; Stage B uses Original for preservation/drift. Scores are six separate 1–5 dimensions. Null reasons, per-metric denominators and common-valid paired differences remain explicit. Original is not a fifth intervention method, Fixed-reg uses its own parameters, and CAA generation is scalar-logistic+GCAV rather than midpoint. No composite is a default outcome.
