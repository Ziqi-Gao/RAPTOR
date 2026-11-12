# Publication checks and scope

Checked against the official [Code and Data Submission Guidelines](https://neurips.cc/public/guides/CodeSubmissionPolicy),
[Paper Checklist](https://neurips.cc/public/guides/PaperChecklist), and
[2026 Main Track Handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook)
on 2026-11-12. These pages encourage reproducibility and disclosure; this
repository's directory organization is a project choice, not a NeurIPS-mandated
template or conference certification.

The release retains every experiment identifiable in the public camera-ready
v4 body/appendix and separates published reported numbers from new runs. See
`paper_map.csv`, `provenance.md`, and `validation.md`. The exact author-named
Windows archive has not yet been compared byte-for-byte with publicv4.

No Slurm jobs, full benchmark, LLM extraction/generation, or paid judge calls are
part of release validation. No main-branch update, merge, tag, Release, or history
rewrite is authorized by this PR. Those remain maintainer publication actions.
