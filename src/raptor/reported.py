"""Recompute public numeric reference tables; never treated as fresh experiment runs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from raptor.experiments.accuracy_aggregation import sample_summary, paired_summary
from raptor.experiments.accuracy_multiseed import atomic_json

METRICS = ("best_layer_test_accuracy", "average_layer_test_accuracy")

def recompute(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["kind"] != "reported_results_reference_only":
        raise ValueError("Requires an explicit reported-results reference manifest")
    for name, item in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != item["released_sha256"]:
            raise ValueError(f"Reported input digest mismatch: {name}")
    seeds = manifest["seeds"]
    rows = list(csv.DictReader((directory / "accuracy_seeds.csv").open()))
    values = {}; splits = {}
    for row in rows:
        seed = int(row["seed"]); key = (row["model"], row["dataset"], row["method"])
        if seed not in seeds or seed in values.get(key, {}):
            raise ValueError("Unexpected or duplicate outer seed")
        values.setdefault(key, {})[seed] = row
        split_key = (row["model"], row["dataset"], seed)
        if split_key in splits and splits[split_key] != row["split_sha256"]:
            raise ValueError("Methods did not share the same split")
        splits[split_key] = row["split_sha256"]
    if len(values) != 180 or len(splits) != 180 or any(set(v) != set(seeds) for v in values.values()):
        raise ValueError("Incomplete6model6dataset5method5seed reported accuracy grid")
    cells = []; paired = []
    for (model, dataset, method), by_seed in sorted(values.items()):
        for metric in METRICS:
            scores = {seed: float(row[metric + "_percent"]) for seed, row in by_seed.items()}
            if not all(np.isfinite(v) and 0 <= v <= 100 for v in scores.values()):
                raise ValueError("Invalid accuracy percentage")
            cells.append(dict(model=model, dataset=dataset, method=method, metric=metric, **sample_summary(scores,seeds)))
            if method != "singlelr":
                left = {seed: float(values[(model,dataset,"singlelr")][seed][metric+"_percent"]) for seed in seeds}
                paired.append(dict(model=model,dataset=dataset,metric=metric,**paired_summary(left,scores,seeds,comparison="singlelr_minus_"+method)))
    lookup = {(c["model"],c["dataset"],c["method"],c["metric"]):c for c in cells}
    reference = list(csv.DictReader((directory/"accuracy_cells.csv").open()))
    checked=0
    for row in reference:
        key=(row["model"],row["dataset"],row["method"],row["metric"])
        if row["dataset"] == "six_dataset_mean":
            scores={seed: float(np.mean([float(v[seed][row["metric"]+"_percent"]) for (m,d,method),v in values.items() if m==row["model"] and method==row["method"]])) for seed in seeds}
            observed=sample_summary(scores,seeds)
        else:
            observed=lookup[key]
        for field,ref_field in (("mean","mean_percent"),("sample_std_ddof1","sample_sd_percentage_points")):
            if not np.isclose(observed[field],float(row[ref_field]),rtol=0,atol=1e-9):
                raise ValueError(f"Reported summary mismatch: {key} {field}")
        checked+=1
    robustness=list(csv.DictReader((directory/"robustness_settings.csv").open()))
    if len(robustness)!=90 or any(int(v["ablations_per_layer"])!=20 for v in robustness):
        raise ValueError("Incomplete18setting5method robustness reference")
    robustness_medians={method:{metric:float(np.median([float(v[metric]) for v in robustness if v["method"]==method])) for metric in ("mean","max")} for method in sorted({v["method"] for v in robustness})}
    timing=list(csv.DictReader((directory/"timing_reported_mixed.csv").open()))
    if len(timing)!=180: raise ValueError("Incomplete mixed timing reference")
    timing_medians={method:float(np.median([float(v["time_seconds"]) for v in timing if v["method_id"]==method])) for method in sorted({v["method_id"] for v in timing})}
    return {"schema_version":1,"kind":"recomputed_reported_numeric_reference","status":"complete",
            "new_training":False,"outer_seeds":seeds,"units":"accuracy_percent_and_difference_percentage_points",
            "verified_accuracy_cells_including_dataset_means":checked,"cells":cells,"paired":paired,
            "robustness_setting_medians":robustness_medians,"mixed_timing_reference_medians_seconds":timing_medians,
            "timing_limitation":"Mixed historical provenance; not a matched timing benchmark or new-run aggregate.",
            "source_manifest_sha256":hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest()}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("reported_results"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    result=recompute(args.input);atomic_json(args.output,result)
    print(json.dumps({"status":result["status"],"verified_accuracy_cells":result["verified_accuracy_cells_including_dataset_means"],"paired_cells":len(result["paired"]),"new_training":False}))
    return 0

if __name__ == "__main__": raise SystemExit(main())
