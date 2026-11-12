# Final manuscript source and result audit

On 12 November 2026, the author supplied the final TeX ZIP with SHA256
`2a1a6ad2a02886f10048b3ab1580d63604bb29ca9bbcd0136902069da21dc2d5`.
This closes the earlier missing-attachment limitation. All **12 active inputs**
among its 66 files match the corresponding [public arXiv v4](https://arxiv.org/abs/2602.00158v4)
files byte for byte. The remaining 54 files are inactive. The ZIP and public
source archive differ; active-input identity does not mean identical archives.
The [identity manifest](validation/manuscript_identity.json) records the active
filenames and hashes without republishing the paper or figures.

Static checks resolved every local input and found **93 labels**, no duplicate
labels or unresolved references, **99 bibliography entries** and **36 distinct
cited keys**, with no missing cited keys or duplicate bibliography keys.
`main_ralop_v2.pdf`, used at `main.tex:166`, is the one-page overview figure,
not a compiled manuscript. Inactive seven-model assets do not expand the active
six-model experiment grid. **No fresh TeX compilation was performed.**

## Confirmed protocol and numeric checks

| Source location | Confirmed scope |
|---|---|
| `main.tex:582–596,1728–1751` | Six models, six datasets, five methods, seeds 42–46; approximately 64/16/20 splits; shared outer validation, pre-refit layer selection, deterministic ties and complete-layer averages. |
| `main.tex:265–292` | Preserving native C at refit implements the stated sample-size-adjusted mean-loss ridge coefficient, `lambda=1/(n*C)`. Native refit/no-refit rules remain method specific. |
| `main.tex:1756–1781,1797–1869` | Separate three-method subject/city-grouped sensitivity; within-seed six-model means precede SD and paired intervals. |
| `main.tex:618–629`; `robustness_full.tex:5–8,52–56` | 18 settings, 768 layers per method, K=20, 20% drops and 190 absolute-cosine pairs per layer. Native raw versus standardized AGOP coordinates and all 16 Fixed-reg iteration-limit directions are explicitly retained. |

The comparison checked **1,618 displayed values**: 420 compact main accuracy
entries, 840 appendix accuracy means/SDs, 168 grouped means/SDs and 190 robustness
statistics. All agree within their stated rounding precision. Six float-format
comparisons land at half a displayed unit; these are rounding boundaries. Main
accuracy highlights and narrative win counts agree. Separately, arithmetic
recomputation validated 900 primary seed rows, 420 summaries and 288 paired
comparisons without fitting. Grouped values were checked against the accepted
historical summaries; they are not additional released seed-level CSV records.

Existing PDF vectors and layouts were also checked:

- All 288 accuracy heatmap colors match the released RAPTOR-minus-baseline
  values and the ±15 percentage-point palette (maximum RGB error `3.02e-8`).
- All 180 robustness points and ten median diamonds match the released reference
  (maximum absolute point deviation `6.61e-8`).
- All 180 timing points and five median diamonds match the mixed-source reference
  (maximum relative point deviation `2.22e-6`).
- The two layerwise figures contain 36 panels, 180 method curves and 8,040
  mean-layer values with complete layer ranges, six-model order and five-method
  legends. Their raw seed-level means were **not independently regenerated**.

These checks inspect the printed tables and existing figures; they do not
reproduce the original experiments. Numeric consistency cannot establish every
scientific claim or the provenance of historical runs.

## Specific limitations that remain

The final attachment introduces no changes to the previously audited active v4
inputs. Existing manuscript/code differences and missing evidence remain:

- **Timing:** the active figure (`main.tex:692–700`) combines recovered original
  three-method points with newer CPU1 CAA/Fixed-reg points. Its metadata labels
  this mixture. The same-hardware/pipeline statement at `main.tex:679–680` is
  not independently established by the recovered clocks, threads, allocations
  or old seed aggregation. New-two values use five-seed means of full-layer
  medians; the older aggregation is unbound. The later three-method CPU1 grid
  is incomplete. The [unified workflow](timing.md) records new executions with
  explicit resources; it does not repair the historical figure's provenance.
- **Steering:** the adopted study remains Original/GCS/RAPTOR with four metrics
  on a 1–10 scale, 300 cases, 265 described as evaluable and 160 fully valid
  (`main.tex:704–753,1920–2169,2210–2239`). The later calibrated four-method,
  six-metric study is optional. Historical control trajectories, individual
  judge scores and exclusion records remain unavailable. The printed raw
  affine-logit formula (`main.tex:352–372,1958–2030`) matches `raw_calibrated`;
  known `legacy_unit` behavior uses unit directions and zero bias. The old
  table's producer is not established by either mode's presence.
- **Separability and structure:** the training-set description
  (`main.tex:1906–1911`) differs from retained all-cache fitting; the original
  model/subset/producer is unverified. Structure results
  (`main.tex:2175–2206,2243–2250`) retain a legacy `p/n` field versus the printed
  `n/p`. The paper's table caption calls `lambda=1/C` equivalent (`main.tex:2177`),
  conflicting with its mean-loss objective (`main.tex:268–275`): native sklearn
  fits imply `lambda=1/(n*C)`. Old inverse-C values are not relabeled. Historical
  flags/fractions remain unbound; best-over-C correlations are exploratory.
  A fixed-p sweep cannot independently distinguish dependence on n/p from
  dependence on n (`main.tex:801–802,951–954`).
- **Theory scope:** the analytic derivation keeps its explicit uniqueness
  assumption and cited optimizer-convergence result
  (`main.tex:1013–1017,1571–1572,1602–1608`). The stylized independent-noise
  approximation (`main.tex:934–940`) does not guarantee stability under
  overlapping ablations. No separate numerical-theory solver is an adopted
  experiment in the active manuscript.
- **Permissions and execution:** author-code licensing remains deferred and
  some data redistribution terms remain unresolved. This audit performed no
  full benchmark rerun, extraction, generation, judge calls or TeX compilation.

Receiving the attachment closes its access and active-input identity questions.
It does not settle the issues above or certify the manuscript as fully correct.
[Provenance](provenance.md), the [publication checklist](publication_checklist.md)
and [validation evidence](validation.md) distinguish completed checks from
outstanding work. No manuscript, figure, scientific code or numeric result was
changed during this documentation update.
