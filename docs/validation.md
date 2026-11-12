# Release validation — 12 November 2026

This record separates tested code behavior, executable workflows and reproduced
reported summaries. No complete scientific benchmark was rerun. Compact results
and evidence digests are saved in [validation/evidence.json](validation/evidence.json).
Raw logs, caches, generated text and machine paths are excluded from this report.

## Installed package and environment

The archived Git index tree `cb326684168d3ea6e10fed373bc8a9c718e52de1` was used to
build `raptor-probes 0.2.0`. This audit documentation was added afterward. The wheel
SHA256 is `cf3aa418501b0f2f3fe6ccccabba82175e808743ba1851a905c45393b609ad20`.

The final build used a clean environment without host site packages after an
initial host-build setuptools conflict. The scientific Python-package source
identity is `96d57688f2689f845dbc497c84e1f0079fa123547087f9e24ae8a0da4e5fb399`
(the final timing manifest's location-independent source digest).

Two installation checks have different isolation guarantees:

- A **clean core environment**, without host packages or Torch, installed the
  wheel noneditably. Core imports and reported-result recomputation passed:
  420 accuracy summaries, including descriptive dataset means, and 288 paired
  comparisons. No `PYTHONPATH` or another worktree supplied package source.
- A **numerical runtime environment** installed the same wheel noneditably and
  used its existing numerical/model dependencies. It was not isolated from host
  dependency packages. All 41 packaged Python/rubric files matched the archived
  source byte for byte, and all 20 public script `--help` checks passed without
  `PYTHONPATH`. An additional audit checked the flags of 28 documented README and
  paper-map commands; their entrypoints and flags matched the actual CLIs.

The installed-wheel suite passed **91 tests**, with two warnings, in 37.46 seconds.
The wheel includes the author-license pending notice, Apache-2.0
and retained MIT notices, and both optional judge rubric resources.

Local verification used Python 3.9.21, NumPy 1.26.4, SciPy 1.13.1,
scikit-learn 1.6.1, pandas 2.2.3, threadpoolctl 3.6.0, xRFM 0.4.3,
PyTorch distribution 2.8.0 (actual runtime `2.8.0+cu128`), Transformers 4.45.0,
Datasets 3.5.0 and Matplotlib 3.9.4. Timing used an Intel Xeon Platinum 8592+
CPU, one affinity CPU and one effective Torch/native-pool thread. The installed
Torch build supports CUDA 12.8; these local checks executed on CPU with Python
3.9.21. [GitHub Actions](https://github.com/Ziqi-Gao/RAPTOR/actions/runs/43231370620)
completed successfully with Python 3.11 for commit
`3184a978563c516678e223f87c32849a267870b1`. This records the prior commit's CI;
local Python 3.11 tests and CI for this later documentation update are not claimed.
A command-scoped C++ runtime
search-path correction was needed for a local embedding CLI check; no global
loader configuration changed. The declared Python range is
3.9–3.12, reflecting the pinned numerical dependencies.

## Actual offline execution

| Check | Observed result | Scope |
|---|---|---|
| Five-method accuracy | 50 native layer–method–seed results; resume restored all 50 with no refits; complete aggregation, ten mean/SD cells, eight paired comparisons and plots | Synthetic cache, two layers, seeds 42–46; full native per-layer budgets |
| Five-method timing | 50 native results; all 50 restored without recomputation; strict aggregation and PNG/PDF/SVG exports passed | Synthetic 800-row, six-feature cache; two layers, five seeds; actual installed xRFM and full native budgets |
| Native output equivalence | Exact validation accuracy, test accuracy and concept-array SHA256 matches for all five methods | Final installed wheel, same synthetic layer/split/seed, against native scientific baseline `68faa5f` |
| Timing interruption | SIGUSR1 during actual GCS work flushed the active method; three units survived, then resume restored three and computed two | Synthetic task, actual native fitting |
| Corrupt timing checkpoint | Invalid checksum rejected; one actual CAA result recomputed and four verified units restored | Synthetic task; no historical result fallback |
| Robustness | Real finite-budget Fixed-reg fitting/checkpoint/resume and native GCS diagnostic equivalence passed | Small synthetic inputs; iteration-limit outputs retained |
| Structure diagnostic | Four genuine fraction–seed shards ran, resumed and strictly aggregated | Two fractions and two subsample seeds; full 360-shard grid not run |
| Reported-result recomputation | 900 reported seed rows validated, 420 summaries and 288 paired comparisons recomputed; reported robustness and mixed-source timing medians rederived | Published numeric reference inputs; no new training |
| Original artifact spot check | Five methods' saved choices and summaries checked against all 64 layers of original NPZ artifacts | Qwen2.5-32B/Cities, seed 42; each choice maximizes its own pre-refit validation |

The five-method canaries execute actual native methods. They establish execution,
protocol and resume behavior, rather than benchmark accuracy or a speed ranking.
The focused accuracy/signal/corruption/structure checks used isolated release-source
snapshots. Native equivalence was additionally confirmed against the final
installed wheel; its suite and timing canary are identified separately above. Tests cover shared splits, seed isolation,
validation-only selection and deterministic ties, native refit differences,
midpoint readout, sample SD and paired statistics, corrupt/missing/mixed records,
group separation, robustness coordinates/intercepts/full denominators, and
finite-budget outcomes. Synthetic hook and telemetry checks did not load an LLM.

## Full-grid inventory without training

Header-only inspection of all 36 real embedding caches confirmed the complete
physical layer sets and planned output uniqueness. It did not rehash the large
embedding contents or execute the formal grids. Fresh run preparation performs
its own content and split-identity checks.

| Planned workflow | Checked denominator |
|---|---|
| Primary accuracy | 40,200 layer–seed–method units |
| Three-method grouped sensitivity | 8,040 units; 60 actual-CSV grouped splits checked for zero entity overlap |
| Combined accuracy paths | 48,240 unique unit targets |
| Five-method timing | 8,040 layer–seed tasks; 40,200 method observations |
| Directional robustness | 18 settings, 768 layers per method, 3,840 layer–method tasks, 76,800 directions and 729,600 cosine pairs |

## Final manuscript recheck

The author-supplied final TeX ZIP has SHA256
`2a1a6ad2a02886f10048b3ab1580d63604bb29ca9bbcd0136902069da21dc2d5`.
All 12 active inputs among its 66 files match the audited v4 files byte for byte;
54 files are inactive. Static checks found 93 labels without duplicates or
unresolved references, and 99 bibliography entries with 36 distinct cited keys,
no missing cited keys and no duplicate bibliography keys. This checks source
inputs and static references, not a new compilation or typesetting certificate.

The read-only numeric recheck compared **1,618 displayed values**: 420 compact
main accuracy entries, 840 appendix accuracy means/SDs, 168 grouped sensitivity
means/SDs and 190 robustness statistics. All agree within their declared rounding
precision; six float-format boundary cases are exactly half a displayed unit.
All 288 accuracy heatmap colors and the robustness PDF's 180 points and ten
median diamonds match the released numeric references. Layerwise figure checks
cover layout, complete ranges and 180 curves (8,040 values); their raw seed means
were not independently regenerated. The timing PDF's 180 setting circles and five
median diamonds also match the
released mixed-source reference within PDF-coordinate precision (maximum
relative point deviation about `2.22e-6`). These checks confirm the printed
inputs, not historical resource equivalence or a newly matched timing run.

[manuscript_audit.md](manuscript_audit.md) records exact manuscript locations,
methods and preserved limitations. [validation/manuscript_identity.json](validation/manuscript_identity.json)
contains the active-input hashes, counts and compact numeric results. No
scientific source, result, manuscript or figure was changed for this recheck;
no fitting, generation, judging, extraction or cluster work was performed.

## Provided workflows and remaining limits

The repository provides fresh prepare/run/resume/aggregate/plot commands for the
formal grids, data and embedding preparation, steering/control measurement,
separability, structure diagnostics and optional judge transport. Full accuracy,
grouped sensitivity, robustness, timing and structure reruns were not executed.
Neither large-model activation extraction nor real LLM generation, paid judging,
GPU validation or cluster submission/modification/cancellation was performed.

The owner deferred the author-code software license: [LICENSE](../LICENSE) is a
pending-status notice, not an open-source grant. Some dataset redistribution terms
remain unresolved, so raw benchmark text is withheld. The author has now supplied
the final TeX ZIP. Its 12 active inputs match the audited public v4 files byte for
byte; its 54 inactive files do not enter the document. The archives themselves
are different. No fresh TeX compilation was performed.

Historical scientific limits remain as documented in [provenance.md](provenance.md):
the reported five-method timing figure mixes sources, original three-method
clock/resource/seed logs are unavailable, and the later three-method CPU1 grid
is incomplete. The author's unchanged-hardware/procedure statement remains
separate from verifiable logs. Old robustness perturbation pairing, steering
alpha/probability trajectories, original judge individual scores and some appendix
producer/subset bindings cannot be established by these new tests. Reported
finite iteration-limit directions remain in their original denominators.
