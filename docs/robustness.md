# Directional robustness protocol

The reported scope is three models (Gemma-7B-it, Qwen2.5-3B-Instruct,
Qwen2.5-32B-Instruct), six datasets, and five methods: 18 settings and 768
stored layers per method. Each layer uses K=20 independent 20% training-pool
drops. This is a fixed-seed perturbation experiment, not five outer accuracy
seeds. Stored layer 0 corresponds to displayed physical layer 1.

```bash
python scripts/run_robustness.py prepare --emb-dir embeddings_all \
  --output exp_results/robustness_release
python scripts/run_robustness.py run --manifest exp_results/robustness_release/manifest.json --dry-run
python scripts/run_robustness.py run --manifest exp_results/robustness_release/manifest.json
python scripts/run_robustness.py aggregate --manifest exp_results/robustness_release/manifest.json
python scripts/plotting/plot_robustness.py \
  --summary exp_results/robustness_release/summary.json --output plots/robustness_release.pdf
```

`run --task-index N` executes one prepared layer-method task. Every expensive
perturbation is checkpointed atomically, and RAPTOR/Fixed-reg also save a native
candidate before refit. Resume verifies source, environment, embedding bytes,
configuration, run seed and exact ordered indices. Corrupt or incompatible
files (including truncated NPZ archives and frozen candidates) are retained
under `.rejected.*` before repair. Technical exceptions are saved as
`status=error`; the run exits 2 if execution errors or invalid directions remain.
After fixing the execution problem, repeat the same `run --manifest ...` command
(and `--task-index N` if used). Resume retries each error once per invocation with
the identical perturbation, split, seed and fit budget, retaining the previous
error checkpoint byte for byte under `run_NN.npz.error.*`. Matching frozen native
candidates remain reusable if an exception happened during refit. Persistent
errors keep producing a nonzero exit and a new archived attempt; there is no
automatic retry loop within one invocation.

Finite nonconvergence outputs are restored unchanged and never retried according
to their scores. Zero or nonfinite returned directions are also retained without
retry and keep the fixed grid incomplete; resume cannot improve those scientific
outcomes by replacing them. Error history is not a new perturbation and never
enters the pair denominator.
A timeout signal stops after the current atomic perturbation; exit 75 supports
explicit scheduler resumption. The runner does not submit or change jobs.

The outer seed 42 stratified 80% train pool is fixed. For stored layer l and
perturbation r, `RandomState(42+1000*l+r)` draws the retained pool without
replacement after dropping `round(.2*n_pool)` rows. Its next integer determines
the train-subset/validation split. Method fitting seed is `42+r`. Every new-run
artifact saves retained, train-subset, validation, final-refit and test indices
plus their hashes and fitting seeds. The historical RNG recipe is preserved;
this cannot retroactively prove identity with unavailable historical indices.

| Method | Native fit and direction used in cosine |
|---|---|
| RAPTOR | Train-only scaler/C search; retained-pool refit with frozen scaler; raw W, excluding b |
| Fixed-reg | Same candidate/refit, fixed sklearn C=1 and max_iter1000; raw W, excluding b |
| GCS | Native subsampling from the retained pool, not accuracy's train-subset; raw sampled-ensemble mean W, excluding b |
| CAA (`caa`) | Re-estimated raw class mean difference on the complete retained pool; no midpoint classification or scalar calibration |
| xRFM | Native internal validation on retained pool; standardized-space AGOP eigenvector, preserving the reported historical coordinate identity |

Every pair of genuine directions contributes: K(K−1)/2=190 unordered absolute
cosines per layer. Nonfinite or zero directions mark the layer incomplete;
they cannot reduce the denominator. Finite iteration-limit fits remain in all
190 pairs. Setting `Mean` equally averages the full fixed layer set; setting
`Max` is its maximum layer statistic. Cross-setting diamonds are medians of
the 18 setting values. Neither layers nor direction pairs become outer-seed
replicates. Results compare explicitly labelled method-native coordinates;
the historical results do not support strict five-method paired inference.

RAPTOR/Fixed-reg retain candidate and final iteration-hit counts. New GCS runs
retain each genuine optimizer attempt's iteration and threshold status; the
optional diagnostics do not alter the RNG, fit, early-stop or return values.
xRFM reports its fixed native RFM budget and selected configuration, without
inventing an optimizer-convergence certificate. Nonconverged finite outputs
produce `complete_with_nonconvergence`, while an incomplete fixed grid prevents
a complete setting summary.

The published Fixed-reg robustness production at commit
`ff52cb105602b3c737f27cfcc7226944513d0431` has 16 finite final fits at the
iteration limit in Gemma/STSA's first layer (17 stage hits including one
candidate). Those directions define that setting's Max and enter Mean. They
remain valid finite-budget results and must not be discarded, retried to a
better solution, or described as all converged. Historical raw directions,
perturbation identity limitations and standardized xRFM coordinates remain
distinct from the richer records written by a new release run.

The new public runner consolidates the reviewed native algorithms and RNG
recipe into one portable workflow, replacing internal replay-manifest and
frozen-worktree dependencies. It does not import historical results. Reduced
K or fitting budgets are accepted only with an explicit `synthetic_smoke:true`
configuration and remain labelled as such. Small tests validate behavior;
the full 18-setting release run was not performed.
