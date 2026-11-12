#!/usr/bin/env python3
"""Aggregate validation-selected Table 1 rebuttal results across outer seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from raptor.experiments.accuracy_aggregation import aggregate_accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("logs/accuracy_tasks/manifest.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("exp_results/accuracy_multiseed/summary.json"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("exp_results/accuracy_multiseed/summary.csv"),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write incomplete coverage and exit successfully.",
    )
    args = parser.parse_args()
    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    payload = aggregate_accuracy(
        manifest,
        summary_json=args.out_json,
        summary_csv=args.out_csv,
    )
    coverage = payload["coverage"]
    print(
        f"accuracy aggregation status={payload['status']} "
        f"complete_cells={coverage['complete_summary_cells']}/"
        f"{coverage['expected_summary_cells']} "
        f"missing_layers={coverage['missing_or_invalid_layer_checkpoints']}"
    )
    if payload["status"] != "complete" and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
