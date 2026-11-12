# Publication checks and scope

Checked against the official [Code and Data Submission Guidelines](https://neurips.cc/public/guides/CodeSubmissionPolicy),
[Paper Checklist](https://neurips.cc/public/guides/PaperChecklist), and
[2026 Main Track Handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook)
on 2026-11-12. These pages encourage reproducibility and disclosure; this
repository's directory organization is a project choice, not a NeurIPS-mandated
template or conference certification.

The release retains every experiment identifiable in the public camera-ready
v4 body/appendix and separates published reported numbers from new runs. See
`paper_map.csv`, `provenance.md`, and `validation.md`. The author-supplied final
TeX ZIP is now available and checked: all 12 active inputs match public v4 byte
for byte, while the archives themselves differ. The attachment's 54 inactive
files do not expand the paper's experiment scope. Static checks found no missing
local inputs, unresolved references, duplicate labels or unresolved cited keys.
See [manuscript_audit.md](manuscript_audit.md). No fresh TeX compilation was run.

Completed checks establish source identity, selected numeric/figure consistency,
and the code behavior described in `validation.md`. The author-code license is
still deferred, some data terms and historical execution evidence remain
unresolved, and full experimental reruns have not been completed. The manuscript's
checklist answers are author statements, not a substitute for those missing records.

No Slurm jobs, full benchmark, LLM extraction/generation, or paid judge calls are
part of release validation. No main-branch update, merge, tag, Release, or history
rewrite is authorized by this PR. Those remain maintainer publication actions.
