# RAPTOR public release maintenance

Read README.md, configs/default.yaml and the relevant docs before changing science.
Keep one public workflow per experiment. Preserve native fit budgets, refit rules,
layer sets, coordinates, and preprocessing. Test scores never select layers or
hyperparameters. Shared outer seeds42–46 are not robustness perturbations.
Keep reported_results separate from new outputs; never silently backfill a run.
Do not commit raw data/text, weights, embeddings, logs, machine paths or secrets.
Run offline unit tests and the real five-method synthetic timing canary after
scientific changes. Full benchmarks, model generation, paid judges and cluster
submissions require explicit user authorization. No default CI may call them.
The author-code license remains pending owner confirmation; third-party terms
are separate. Never merge, tag, release, or rewrite history without authorization.
