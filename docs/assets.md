# Assets, versions, and redistribution

Author-owned code license is pending the repository owner's decision. A paper's
CC-BY license is not automatically a software license. No third-party asset is
relicensed by this repository.

| Asset actually used | Status and source |
|---|---|
| xRFM adapter dependency | Tested against PyPI `xrfm==0.4.3`; [upstream](https://github.com/dmbeaglehole/xRFM), MIT. Installed Python source digest is recorded by timing manifests. The old README pin `773fae8` identifies a `setup.py` declaring 0.1.0 and does not identify the installed 0.4.3 production package; do not claim equivalence. Its historical absence of direct-URL provenance is retained. xRFM is installed as a dependency, not vendored wholesale. |
| Historical HateXplain builder | `dataset/hatexplain/hatexplain.py` retains Hugging Face authors' Apache2.0 header. Full license in `third_party/licenses/Apache-2.0.txt`. Data licensing is separate from loader-code licensing. |
| Native probing implementations | RAPTOR/Fixed-reg, GCS sampling adapter and CAA readouts are integrated research implementations. GCS fidelity is unchanged; this cleanup does not certify equivalence to all upstream defaults. |
| Earlier neural controller dependency | Earlier history included code under Daniel Beaglehole's2025 MIT license. Attribution text is retained under `third_party/licenses/neural_controllers-MIT.txt`; no controller repository is copied into this release. |
| Tracing | Native PyTorch module hooks replace the optional baukit import with the same tap. No baukit source is redistributed; its optional old fallback is not an alternative scientific protocol. |
| SST/STSA | [SST card](https://huggingface.co/datasets/stanfordnlp/sst/blob/main/README.md) lists unknown license; specific binary release terms not established. Text withheld. |
| Cities | [benchmark](https://github.com/saprmarks/geometry-of-truth) derives from [GeoNames](https://download.geonames.org/export/dump/readme.txt), CC-BY4.0; separate benchmark CSV terms not established. Text withheld. |
| CounterFact | [ROME](https://rome.baulab.info/) and derived Geometry-of-Truth CSV; redistribution permission for this exact derived file not established. Text withheld. |
| CommonClaim | [repository](https://github.com/Algorithmic-Alignment-Lab/CommonClaim) has no explicit dataset license located. Text withheld. |
| HateXplain | [data card](https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain/blob/main/README.md) lists CC-BY4.0 metadata and MIT licensing prose; exact data terms unresolved. Raw data withheld. |
| Sarcasm | [author dataset release](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) lists CC-BY4.0; obtain there and retain attribution. Text omitted from release. |
| Llama 3.1 | [Meta license](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/blob/main/LICENSE), community license and acceptable-use policy; weights not included. |
| Gemma7B | [Gemma terms](https://ai.google.dev/gemma/terms), weights not included. |
| Qwen2.5-3B | [model license](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE), Qwen Research License. |
| Qwen2.5-7B/32B | [7B license](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE), Apache2.0. |

The public paper is [arXiv2602.00158v4](https://arxiv.org/abs/2602.00158v4),
marked camera-ready and accepted at NeurIPS 2026 by its authors. The paper source
is CC-BY4.0; this repository links it rather than copying the full manuscript.
Only numeric reported-result summaries are candidates for redistribution; source
texts, generated outputs, prompts with personal data, hidden states, and internal
review logs are excluded. Existing Git history still contains earlier benchmark
copies. No history rewrite was performed; maintainers must resolve any historical
redistribution concern separately.
