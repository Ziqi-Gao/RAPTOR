# Numeric paper references

These small numeric tables are reported-result references. They contain no raw
benchmark text, generated responses, or activations. `manifest.json` binds their
source digests, producer identities, and known limits. They are not outputs of the
new release runner. The accuracy input has900method-seed rows; the cell reference
has360setting-metric cells plus60six-dataset descriptive means. Robustness has
90setting-method rows, and mixed historical timing has180setting-method rows.

```bash
python scripts/recompute_reported.py --input reported_results \
  --output outputs/reported_recomputed.json
```

The command verifies all released input digests, all shared split identities,
five seeds, sample SD and420published summaries, then uses the same seed-summary
and paired Student-t code as new accuracy runs. It recomputes robustness setting
medians and the mixed timing reference's36setting medians. It performs no fitting.
It does not reconstruct unavailable original raw logs or steering trajectories.
The new timing aggregator deliberately cannot load these mixed historical points.
