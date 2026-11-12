"""Execute a freshly prepared accuracy manifest, including verified resumes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from raptor.experiments.accuracy_multiseed import execute_task, file_sha256, install_signal_handlers, protocol_methods, ROW_PROTOCOL, GROUPED_PROTOCOL
from raptor.experiments.accuracy_aggregation import _frozen_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("logs/accuracy_tasks/manifest.json"))
    parser.add_argument("--dry-run", action="store_true", help="Verify the full frozen task grid without fitting probes.")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    units = _frozen_tasks(manifest)
    expected = {(protocol, e["model"], e["dataset"], seed, method, layer)
                for protocol in manifest["protocol"].get("protocols", [ROW_PROTOCOL, GROUPED_PROTOCOL])
                for e in manifest["embeddings"]
                if protocol == ROW_PROTOCOL or e["dataset"] in {"counterfact", "cities"}
                for seed in manifest["protocol"]["outer_seeds"]
                for method in protocol_methods(manifest, protocol)
                for layer in range(e["layers"])}
    if set(units) != expected:
        raise ValueError("frozen tasks do not cover the complete declared grid and layer set")
    if len(units) != manifest["total_layer_checkpoints"]:
        raise ValueError("manifest layer count differs from frozen task units")
    targets = {(task["results_root"], protocol, task["model_tag"], dataset, seed, method, layer)
               for (protocol, model, dataset, seed, method, layer), task in units.items()}
    if len(targets) != len(units):
        raise ValueError("accuracy tasks have colliding output targets")
    # Verify data identity, even for a dry run. Preparation never imports old results.
    for embedding in manifest["embeddings"]:
        if file_sha256(Path(embedding["path"])) != embedding["sha256"]:
            raise ValueError("embedding bytes differ from prepared manifest")
    tasks = []
    for entry in manifest["task_files"].values():
        tasks.extend(json.loads(line) for line in Path(entry["path"]).read_text().splitlines() if line.strip())
    print(json.dumps({"dry_run": args.dry_run, "tasks": len(tasks), "layer_checkpoints": len(units),
                      "outer_seeds": manifest["protocol"]["outer_seeds"], "grid": manifest["grid"]}, sort_keys=True))
    if args.dry_run:
        return
    install_signal_handlers()
    for task in tasks:
        outcome = execute_task(task)
        if outcome["stop_requested"]:
            raise SystemExit(75)


if __name__ == "__main__":
    main()
