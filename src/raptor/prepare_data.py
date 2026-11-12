"""Validate user-obtained benchmark files and fingerprint exact embedding row order."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from raptor.core import DEFAULT_DATASETS
from raptor.data import DataProcessing
from raptor.embedding_checkpoint import hash_embedding_inputs

FILENAMES = {"STSA": "stsa.binary.train", "sarcasm": "sarcasm.json",
             "hatexplain": "hatexplain", "counterfact": "counterfact.csv",
             "cities": "cities.csv", "common": "common_claim.csv"}

def inspect_dataset(data_root: Path, dataset: str) -> dict:
    path = data_root / FILENAMES[dataset]
    if dataset == "hatexplain":
        inputs = [path / "dataset.json", path / "post_id_divisions.json"]
    else:
        inputs = [path]
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        raise FileNotFoundError("Obtain the documented upstream inputs first: " + ", ".join(missing))
    pos, neg, _, _ = DataProcessing(str(path), dataset).dispatcher()
    if not pos or not neg or not all(isinstance(x, str) for x in pos + neg):
        raise ValueError(f"{dataset}: both classes must contain text strings")
    result = {"dataset": dataset, "positive_count": len(pos), "negative_count": len(neg),
              "row_order": "positive_then_negative_preserving_source_order_within_class",
              "ordered_text_labels_sha256": hash_embedding_inputs(pos, neg),
              "files": {p.relative_to(data_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    if dataset in ("cities", "counterfact"):
        import pandas as pd
        key = "city" if dataset == "cities" else "subject"
        df = pd.read_csv(path)
        if key not in df or df[key].isna().any():
            raise ValueError(f"{dataset}: missing group column or group IDs: {key}")
        groups = []
        for label, size in ((1, len(pos)), (0, len(neg))):
            groups.extend(df.loc[df.label == label, key].iloc[:size].astype(str).tolist())
        if len(groups) != len(pos) + len(neg):
            raise ValueError("Group IDs do not match the embedding row order")
        result.update(group_key=key, group_count=len(set(groups)),
                      ordered_groups_sha256=hashlib.sha256(json.dumps(groups, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest())
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Optional configs/reported_data_identity.json: require the historic local file bytes")
    args = parser.parse_args(argv)
    datasets = args.datasets.split(",")
    if not datasets or len(datasets) != len(set(datasets)) or any(d not in FILENAMES for d in datasets):
        parser.error("datasets must be unique canonical benchmark names")
    rows = [inspect_dataset(args.data_root, name) for name in datasets]
    if args.reference:
        reference = json.loads(args.reference.read_text())["files"]
        for row in rows:
            for name, digest in row["files"].items():
                key = "dataset/" + name
                if key in reference and reference[key]["sha256"] != digest:
                    raise ValueError(f"Historical input identity mismatch: {name}")
    payload = {"schema_version": 1, "kind": "new_input_manifest", "datasets": rows,
               "note": "A fingerprint validates local preparation, not permission to redistribute or a paper reproduction."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(args.output)
    print(json.dumps({"datasets": len(rows), "rows": sum(r["positive_count"] + r["negative_count"] for r in rows)}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
