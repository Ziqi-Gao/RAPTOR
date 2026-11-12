# Data and embeddings

The release does not redistribute benchmark text, model weights, generated user
text, or activations. Obtain the files below from their original publishers under
the applicable terms. Removing files from this release does not remove earlier
Git history. `configs/reported_data_identity.json` records the exact SHA-256 of
the five local files used in the research workspace; it is an identity reference,
not a license grant. An upstream file with a different hash is a new input version.

| Dataset | Obtain / local filename | Exact transformation and row identity |
|---|---|---|
| STSA | [binary SST release](https://github.com/harvardnlp/sent-conv-torch/tree/master/data), `stsa.binary.train` | Read nonempty lines, split first space into integer label and text; label1 positive, otherwise negative. Preserve source order within each class; no truncation. Original source version is pinned by the local-file hash, not an asserted upstream commit. |
| Sarcasm | [author release](https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection), rename `Sarcasm_Headlines_Dataset.json` to `sarcasm.json` | JSONL `headline`; `is_sarcastic=1` positive. First3000 positive and first3000 negative records independently, source order. |
| CounterFact | [Geometry of Truth benchmark](https://github.com/saprmarks/geometry-of-truth/tree/main/datasets), `counterfact.csv`; underlying [CounterFact](https://rome.baulab.info/) | CSV `statement`, integer `label`0/1; first2000 of each class independently. Grouped sensitivity uses `subject`, positive groups then negative groups with the same truncation. Do not substitute raw ROME JSON for this derived CSV. |
| Cities | [Geometry of Truth benchmark](https://github.com/saprmarks/geometry-of-truth/tree/main/datasets), `cities.csv` | CSV `statement`, integer `label`0/1, all rows. Groups are `city`. Source order within class. |
| Common | [CommonClaim](https://github.com/Algorithmic-Alignment-Lab/CommonClaim), `common_claim.csv` | `examples` text; literal `label` strings `True`/`False`. Other labels omitted. First3000 of each class independently. |
| HateXplain | [original Data directory](https://github.com/hate-alert/HateXplain/tree/master/Data), `hatexplain/dataset.json` and `hatexplain/post_id_divisions.json` | Use the ordered `train` IDs from divisions (not every record / alphabetical ID order). Join `post_tokens` with spaces; majority of labels0=hatespeech, 1=normal, 2=offensive; discard ties/missing labels. Positive majority0/2, negative1; all remaining rows. The retained Apache-licensed historical builder provides the equivalent HF loader. |

The sources above are access instructions, not assurances that every dataset
permits redistribution. STSA, derived CounterFact/Cities, and CommonClaim have
unresolved distribution terms. HateXplain's data card has conflicting CC-BY4.0
metadata and MIT prose. Sarcasm's author release on Kaggle lists CC-BY4.0.
Consult [asset notes](assets.md) before redistributing any data.

```bash
python scripts/prepare_data.py --data-root dataset --output outputs/data_identity.json
# Require the exact five historical local file bytes where a reference exists:
python scripts/prepare_data.py --data-root dataset --output outputs/data_identity.json \
  --reference configs/reported_data_identity.json
```

This command reads local data only, checks both classes and group IDs, and saves
file hashes and ordered text/label/group hashes. It fails on missing files.
HateXplain's historical raw upstream revision was not captured in the local
five-file reference; new runs record both downloaded files. The strict reference
option does not certify that missing historical revision.

All embeddings concatenate **positive then negative**. There is no shuffle before
this concatenation. A cached NPZ has paired `X_pos_0`/`X_neg_0` through every stored
layer, metadata `meta_model`, `meta_dataset`, `meta_layers`, and `meta_dim`;
labels are reconstructed as positive1 then negative0. Stored layer0 is physical
layer1, and the full set is required for the accuracy/timing denominator. Counts
and array identity are checked when building experiment manifests. Group IDs must
be reconstructed from exactly the same source file and transformation.

Legacy NPZ caches remain usable when explicitly supplied to the accuracy or
timing workflows. Extraction skips an existing cache only when its complete
extraction identity matches; its skip validator rejects legacy caches missing
that identity. To extract again, choose a fresh `--emb_dir` or explicitly pass
`--force` to regenerate the cache.

The extraction tap is the **output of each transformer block's
`post_attention_layernorm`, at the final token of the input**. Cached vectors are
FP32. It is not automatically the block's final residual output. The release uses
native PyTorch hooks at exactly these module names; the previous optional fallback
to `output_hidden_states` could change the tap and is removed. Models without
these modules fail explicitly. No new model extraction was performed for release.

```bash
# Local dataset paths use this optional override; defaults to ./dataset.
export RAPTOR_DATA_DIR="$PWD/dataset"
python scripts/run_embeddings.py --models Qwen/Qwen2.5-3B-Instruct \
  --datasets STSA --model_path models --emb_dir embeddings_all --quant 32
# Full six-model/six-dataset grid, after obtaining model access:
python scripts/run_embeddings.py --models all --datasets all \
  --model_path models --emb_dir embeddings_all --quant 32
```

The native model-specific precision overrides are Llama 3.1-70B=bfloat16 and
Qwen2.5-32B=4bit NF4; remaining models follow the selected precision (default32).
These are distinct from FP32 storage and from probe arithmetic precision.
Authenticate for gated Gemma/Llama models outside the repository. CPU execution
is supported for small extraction examples, but large models need appropriate
memory/GPU hardware; no full extraction or GPU test is claimed. Embedding
checkpoints save each completed sample. The final archive is atomically published.
