# Retained and removed release content

The research checkout, other worktrees, results, frozen producers, manifests and
logs remain untouched. Only this new release branch is reorganized. Nothing is
hidden in a public legacy/archive directory.

| Category | Release decision / replacement |
|---|---|
| Native probe algorithms | Keep tested native RAPTOR, Fixed-reg, xRFM adapter, GCS sampler, CAA midpoint, and explicit calibrated steering helper under src/raptor. Preserve budgets, coordinates and refit rules. |
| Accuracy | One five-method multiseed preparation/task/run/aggregate workflow; grouped scope three methods. Shared sample-summary and plotting interfaces. |
| Robustness | One five-method per-perturbation checkpoint runner; full denominators and coordinate labels. No paper backfill masquerading as new runs. |
| Timing | One portable five-method pipeline with independently prepared manifest and strict current-run aggregation. |
| Embeddings | One extraction entrypoint, per-sample checkpoints, explicit same-module tap. Data prepare/identity validator retained. |
| Steering and appendix | Retain published original controls, old four-metricjudge identity, separability, structure diagnostics, and necessary optional calibrated workflow; see paper_map.csv. |
| Raw benchmark files | Five tracked raw data copies removed pending exact redistribution rights; precise transformations, upstream sources and SHA-256 references retained. Original research files unchanged. |
| Internal experiment operations | Do not copy session controllers, rescue submission scripts, handoffs, historical frozen validations, logs, jobs, notebook duplicates, or machine paths into the public tree. |
| Slurm | Remove the two outdated public entrypoints; full commands are scheduler-independent. No uncalibrated new scheduler scripts are published, and no cluster work was submitted. |
| Historical numeric results | Keep sanitized small numeric reference CSVs and provenance separate from new outputs. No raw personal/third-party text. |

Explicit obsolete tracked paths removed:

- `scripts/run_experiments.py`
- `scripts/run_layer_task.py`
- `scripts/run_gcs.py`
- `scripts/run_raptor.py`
- `scripts/run_xrfm.py`
- `scripts/save_embeddings.py`
- `src/raptor/experiments/benchmark.py`
- `src/raptor/experiments/layer_task.py`
- `src/raptor/experiments/run_xrfm.py`
- `slurm/run_benchmark_array.slurm`
- `slurm/run_layer_task.slurm`
- `scripts/plotting/plot_acc_curves.py`
- `scripts/plotting/plot_acc_diff_heatmaps.py`
- `scripts/plotting/plot_time_by_method_box.py`
- `scripts/plotting/plot_time_by_method_violin.py`
- `scripts/plotting/plot_time_fork_dumbbell.py`
- `scripts/plotting/plot_time_per_layer_vs_scale.py`
- `scripts/plotting/plot_time_summary_bar.py`
- `scripts/plotting/plot_timing_curves.py`

Native run_raptor/run_gcs modules remain reusable implementation dependencies of
the unified runners. Their duplicate command wrappers are removed. Changes do not
rewrite historical Git commits or claim previously committed raw data have vanished
from history.


## Steering and appendix file decisions

| Files | Decision and dependency |
|---|---|
| `scripts/steer.py`, `steering/controls.py` | Sole documented generation/control workflow. Own-probe inputs, measured pre/post trajectories and atomic prompt-seed resume. Legacy-unit and calibrated modes have separate explicit identities. |
| `steering/gcav.py`, `configs/steering_kernel_identity.json` | Retain approved four GCAV kernel bodies from e3b924e with source/AST hashes; no historical manifest or absolute path required. |
| `steering/generate.py` | Retain native scientific library and original regression tests, including original prompt builders and legacy behavior. Its old CLI is not a second documented reproduction workflow. |
| `scripts/prepare_steering_prompts.py`, `steering/prompts.py` | Fresh fixed trial IDs through the retained native templates; source text required only where the native template uses it. No old prompt identity is fabricated. |
| `scripts/prepare_steering_probes.py`, `steering/prepare_probes.py`, `steering/scalar_calibration.py` | Export actual own-method raw probes with pre-refit validation; retain native epsilon1e-6 scalar-logistic CAA calibration independently of midpoint accuracy. |
| `scripts/paper_steering_judge.py`, `steering/paper_judge.py` | Published four-metric 1–10 score transport and aggregation, explicit validity and actual judge provenance. Old raw score provenance remains unverified. |
| `steering/judge_six_*`, `steering/rubric/*`, `scripts/judge_steering_six_metrics.py` | Optional later six-metric workflow, absent from published v4 default experiments. Fresh API input/config identity and checkpoints; historical shared-conversation scores are not relabelled. |
| `scripts/evaluate_steering.py`, `steering/evaluate.py` | Remove the unrelated two-metric Joyfulness/Coherence evaluator. It did not produce the published four-metric judge table. |
| `experiments/check_separability.py` | Retain native embedding SVM/perceptron kernels; remove unrelated HaluBench/toxic-chat/RFM hidden-state loaders and ambiguous dataset discovery. Expose explicit cache inputs and actual all-row diagnostic scope. |
| `experiments/structure_validation.py`, `experiments/appendix.py`, `scripts/run_structure_grid.py` | Retain native cross-fit/plug-in computation, add explicit ratio naming and restartable complete-grid orchestration. Declare the fresh fraction list because historical exact flags were not recovered. |
| `docs/reported_results/steering_controls.csv`, `paper_external_judge.csv` | Small numeric published references with source hashes; never substitute them into new execution output. No raw generated text is redistributed. |

The duplicate provisional `measure_steering.py` entrypoint was removed during
release preparation; its tested implementation is the sole `steer.py` command.
The complete native generation library and its existing tests remain present.

## HateXplain documentation and loader assets

- `dataset/hatexplain/README.md`: replace the copied upstream dataset card,
  including its raw `post_tokens` example, with a short project-authored source
  and attribution pointer. No raw post/sample is retained. Exact preparation
  and unresolved data redistribution terms remain documented in `docs/data.md`.
- `dataset/hatexplain/.gitattributes`: remove stale upstream Git LFS patterns;
  no retained implementation or preparation command depends on that file.
- `dataset/hatexplain/hatexplain.py` and
  `third_party/licenses/Apache-2.0.txt`: preserve the historical builder, original
  attribution, and complete Apache-2.0 license without byte changes. Loader-code
  licensing does not resolve licensing of the dataset or copied card.
