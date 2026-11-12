"""Coverage-first aggregation of accepted per-unit judge checkpoints."""
from __future__ import annotations

import collections
from pathlib import Path
import statistics

from .judge_six_protocol import METHODS, METRICS, METRICS_A, METRICS_B, require
from .judge_six_metrics import (
    atomic_text, budget_entries, canonical, load_checkpoint, load_frozen, read_json, read_jsonl,
    sha_file, unit_dir, utc_now, verify_prepared, write_csv, write_json, write_jsonl,
)

DISPLAY = {
    "original": "Original", "raptor": "RAPTOR", "gcs": "GCS", "fixed_reg": "fixed-reg",
    "dim_gcav": "Mean-difference direction with scalar logistic calibration + GCAV (CAA-style)",
}


def sample_stats(values):
    return {"mean": statistics.mean(values) if values else None,
            "sample_sd_ddof1": statistics.stdev(values) if len(values) >= 2 else None}


def pending_item(status, reason):
    return {"score": None, "reason": reason, "applicability": "pending",
            "null_reason": reason, "status": status}


def collect_stage(run_dir, stage, frozen):
    results, states, failures = {}, {}, []
    for unit in read_jsonl(Path(run_dir) / ("stage_%s_inputs.jsonl" % stage)):
        rid = unit["record_id"]
        checkpoint = None
        if frozen:
            try:
                checkpoint = load_checkpoint(run_dir, unit, frozen)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                states[rid] = "technical_failed"
                failures.append({"stage": stage, "record_id": rid, "reason": "invalid_checkpoint", "error": str(exc)})
                continue
        if checkpoint and checkpoint["status"] == "complete":
            states[rid] = "complete"
            results[rid] = checkpoint["parsed"]
        elif checkpoint or any((unit_dir(run_dir, unit) / "attempts").glob("*/started.json")):
            states[rid] = "technical_failed"
            failures.append({"stage": stage, "record_id": rid,
                             "reason": "attempted_without_valid_completion", "checkpoint": bool(checkpoint)})
        else:
            states[rid] = "missing"
    return results, states, failures


def merge_scores(records, a, b, states_a, states_b):
    merged = []
    for rid, source in sorted(records.items()):
        row = {k: source[k] for k in (
            "record_id", "pair_id", "model_id", "task", "direction", "method", "prompt_id", "generation_seed")}
        row["method_display_name"] = DISPLAY[source["method"]]
        row.update(stage_A_status=states_a[rid],
                   stage_B_status="not_applicable_self_reference" if source["method"] == "original" else states_b[rid],
                   factual_status=a.get(rid, {}).get("factual_status"), factual_reason=a.get(rid, {}).get("factual_reason"),
                   reference_comparability=b.get(rid, {}).get("reference_comparability"),
                   reference_comparability_reason=b.get(rid, {}).get("reference_comparability_reason"))
        if source["method"] == "original":
            row["reference_comparability"] = "not_applicable_self_reference"
            row["reference_comparability_reason"] = "not_applicable_self_reference"
        for metric in METRICS:
            if source["method"] == "original" and metric in METRICS_B:
                item = {"score": None, "reason": "not_applicable_self_reference", "applicability": "not_applicable",
                        "null_reason": "not_applicable_self_reference", "status": "N/A"}
            else:
                parsed = a.get(rid) if metric in METRICS_A else b.get(rid)
                state = states_a[rid] if metric in METRICS_A else states_b[rid]
                if parsed:
                    item = dict(parsed["scores"][metric])
                    item["status"] = "valid" if item["score"] is not None else "N/A"
                else:
                    item = pending_item(state, "judge_not_run" if state == "missing" else "technical_failure_or_interruption")
            row[metric] = item["score"]
            for key in ("reason", "applicability", "null_reason", "status"):
                row[metric + "_" + key] = item[key]
        merged.append(row)
    return merged


def summarise(merged):
    groups = collections.defaultdict(list)
    for row in merged:
        groups[tuple(row[k] for k in ("model_id", "task", "direction", "method"))].append(row)
    coverage, summary = [], []
    for key, rows in sorted(groups.items()):
        for metric in METRICS:
            valid = [row[metric] for row in rows if row[metric + "_status"] == "valid"]
            counts = collections.Counter(row[metric + "_status"] for row in rows)
            reasons = collections.Counter(row[metric + "_null_reason"] for row in rows if row[metric + "_status"] == "N/A")
            item = dict(zip(("model_id", "task", "direction", "method"), key))
            item.update(metric=metric, expected=len(rows), completed=counts["valid"] + counts["N/A"],
                        valid=len(valid), **{"N/A": counts["N/A"]}, technical_failed=counts["technical_failed"],
                        missing=counts["missing"], judge_units_expected=0 if key[-1] == "original" and metric in METRICS_B else len(rows))
            require(item["completed"] + item["technical_failed"] + item["missing"] == item["expected"], "coverage denominator")
            coverage.append(item)
            summary.append({**item, **sample_stats(valid), **{"score_%s_count" % i: valid.count(i) for i in range(1, 6)},
                            "na_reasons": dict(reasons),
                            "sd_interpretation": "generation-record distribution; not outer-split seed uncertainty",
                            "favorable_sign": "lower" if metric == "semantic_drift" else "higher"})
    return coverage, summary


def paired_differences(merged):
    groups = collections.defaultdict(dict)
    strata = collections.defaultdict(set)
    for row in merged:
        groups[row["pair_id"]][row["method"]] = row
        strata[tuple(row[k] for k in ("model_id", "task", "direction"))].add(row["pair_id"])
    output = []
    for stratum, pair_ids in sorted(strata.items()):
        for baseline in ("original", "gcs", "fixed_reg", "dim_gcav"):
            for metric in METRICS:
                if baseline == "original" and metric in METRICS_B:
                    continue
                differences = []
                excluded = collections.Counter()
                for pid in sorted(pair_ids):
                    left, right = groups[pid]["raptor"], groups[pid][baseline]
                    statuses = {left[metric + "_status"], right[metric + "_status"]}
                    if statuses == {"valid"}:
                        differences.append(left[metric] - right[metric])
                    elif "technical_failed" in statuses:
                        excluded["technical_failed_pairs"] += 1
                    elif "missing" in statuses:
                        excluded["missing_pairs"] += 1
                    else:
                        excluded["not_evaluable_pairs"] += 1
                row = dict(zip(("model_id", "task", "direction"), stratum))
                row.update(comparison="RAPTOR_minus_" + baseline, baseline=baseline, metric=metric,
                           expected_pairs=len(pair_ids), paired_count=len(differences), missing_or_invalid_pairs=len(pair_ids) - len(differences),
                           technical_failed_pairs=excluded["technical_failed_pairs"], missing_pairs=excluded["missing_pairs"],
                           not_evaluable_pairs=excluded["not_evaluable_pairs"],
                           mean_paired_difference=sample_stats(differences)["mean"],
                           sample_sd_paired_difference_ddof1=sample_stats(differences)["sample_sd_ddof1"],
                           favorable_sign="negative" if metric == "semantic_drift" else "positive",
                           inference="descriptive paired record differences; no significance or independence claim")
                output.append(row)
    return output


def cost_summary(run_dir):
    entries = list(budget_entries(run_dir).values())
    return {
        "currency": "USD", "requests_reserved_including_unknown_outcomes": len(entries),
        "usage_known_requests": sum("usage" in r for r in entries),
        "prompt_tokens_reported": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in entries),
        "completion_tokens_reported": sum(r.get("usage", {}).get("completion_tokens", 0) for r in entries),
        "usage_cost_at_frozen_rates_usd": sum(r.get("charged_usd", 0) for r in entries),
        "unknown_outcome_or_usage_reservation_usd": sum(r["reserved_usd"] for r in entries if "charged_usd" not in r),
        "cost_plus_unknown_reservations_usd": sum(r.get("charged_usd", r["reserved_usd"]) for r in entries),
        "interpretation": "conservative usage cost at supplied frozen rates; provider invoice not available; no cache discount assumed",
    }


def aggregate(run_dir):
    run_dir = Path(run_dir)
    prep = verify_prepared(run_dir)
    frozen = load_frozen(run_dir) if (run_dir / "frozen_config.json").exists() else None
    records = {r["record_id"]: r for r in read_jsonl(run_dir / "local_records.jsonl")}
    a, states_a, failures_a = collect_stage(run_dir, "A", frozen)
    b, states_b, failures_b = collect_stage(run_dir, "B", frozen)
    merged = merge_scores(records, a, b, states_a, states_b)
    coverage, summary = summarise(merged)
    pairs = paired_differences(merged)
    require(len(merged) == len(records), "merged score count")
    complete = len(a) == len(states_a) and len(b) == len(states_b)
    status = "COMPLETE" if complete else ("AWAITING_JUDGE_CONFIGURATION" if not frozen else "INCOMPLETE")
    cost = cost_summary(run_dir)
    totals = collections.Counter()
    for row in coverage:
        totals.update({k: row[k] for k in ("expected", "completed", "valid", "N/A", "technical_failed", "missing")})
    result = {
        "status": status, "created_at_utc": utc_now(), "records": len(merged),
        "stages": {stage: {"expected": len(states), "completed": len(scores),
                            "technical_failed": sum(v == "technical_failed" for v in states.values()),
                            "missing": sum(v == "missing" for v in states.values())}
                   for stage, states, scores in (("A", states_a, a), ("B", states_b, b))},
        "metric_slots": dict(totals), "cost": cost,
        "frozen_config_identity": frozen["identity_sha256"] if frozen else None,
        "original_reference_metric_slots_are_local_na": 2 * sum(r["method"] == "original" for r in merged),
        "judge_score_status": "all_expected_units_accepted_including_legitimate_nulls" if complete else "NOT_COMPLETE",
    }
    write_jsonl(run_dir / "scores.jsonl", merged)
    write_csv(run_dir / "scores.csv", list(merged[0]), merged)
    write_csv(run_dir / "coverage.csv", list(coverage[0]), coverage)
    write_csv(run_dir / "summary.csv", list(summary[0]), summary)
    write_csv(run_dir / "paired_differences.csv", list(pairs[0]), pairs)
    write_jsonl(run_dir / "technical_failures.jsonl", failures_a + failures_b)
    write_json(run_dir / "cost_summary.json", cost)
    write_json(run_dir / "completion.json", result)
    write_report(run_dir, result, prep, frozen, summary, pairs)
    outputs = ("scores.jsonl", "scores.csv", "coverage.csv", "summary.csv", "paired_differences.csv",
               "technical_failures.jsonl", "cost_summary.json", "completion.json", "FINAL_REPORT.md")
    write_json(run_dir / "result_manifest.json", {"status": status, "at_utc": utc_now(),
               "sha256": {name: sha_file(run_dir / name) for name in outputs}})
    return result


def write_report(run_dir, result, prep, frozen, summary, pairs):
    a, b = result["stages"]["A"], result["stages"]["B"]
    cost = result["cost"]
    lines = [
        "# Six-metric steering judge: actual delivery status",
        "", "Status: **%s**. Generated at %s." % (result["status"], result["created_at_utc"]), "",
        "Stage A accepted %s/%s independent content units; missing %s; technical failures/unfinished attempts %s."
        % (a["completed"], a["expected"], a["missing"], a["technical_failed"]),
        "Stage B accepted %s/%s paired units; missing %s; technical failures/unfinished attempts %s."
        % (b["completed"], b["expected"], b["missing"], b["technical_failed"]), "",
        "scores.jsonl and scores.csv contain %s rows. Pending/failed scores are explicit nulls with pending applicability and a technical/missing status; they are not valid N/A ratings."
        % result["records"],
        "Original's %s reference-metric slots are local N/A (not_applicable_self_reference), excluded from comparison rankings."
        % result["original_reference_metric_slots_are_local_na"], "",
        "## Frozen protocol and source identity", "",
        "Input audit passed: unique record IDs, complete five-method pairs, full generation seeds, generation JSONL SHA-256, exact text and prompt/checkpoint/tokenizer/chat-template/decoding identity. See source_audit.json and inputs_eda_report.md for every hash and all input/run identities.",
        "Targets: STSA neutral_to_negative = negative sentiment; CounterFact lie_to_truth = factual truth; CounterFact truth_to_lie = factual falsity. The same target rule applies to Original.",
        "Each anonymous Stage A/B row is one independent context. B always uses the locally paired Original as reference. No method/model/path/probe labels are in judge payloads. Frozen shuffle seed: %s." % prep["shuffle_seed"],
        "Rubric and schemas: RUBRIC.md, schema_A.json, schema_B.json. File hashes and code identity: preparation.json. Source code snapshot: provenance/.",
        "Code commit: `%s`; branch `%s`. Environment: `%s`, Python `%s`." % (
            prep["code_identity"]["git_commit"], prep["code_identity"]["git_branch"],
            prep["environment"]["executable"], prep["environment"]["python"].splitlines()[0]), "",
    ]
    if frozen:
        config = frozen["config"]
        lines += ["Judge request model: `%s`; required actual response model/version: `%s`." % (config["model"], config["expected_response_model"]),
                  "Exact request parameters, capability evidence, retry policy, pricing and budget are in frozen_config.json; credentials are read only from the named environment variable and are absent from saved requests.",
                  "Only bounded network/format/contract failures are retried. Model drift, authentication/configuration errors, and provider refusals halt. Accepted ratings are never retried based on score or method ranking."]
    else:
        lines += ["**No real judge model/version, usable credential, or budget was supplied. No real canary or scoring API call has been made.**",
                  "CONFIG_TO_COMPLETE.json is an unfilled template, not a frozen execution configuration. Rubric, schemas, input ordering and source/code identities are already prepared; judge configuration must be supplied and frozen before a real canary."]
    lines += ["", "## Costs and coverage", "",
              "Requests reserved (including uncertain interrupted outcomes): %s. Usage-known requests: %s. Reported input/output tokens: %s/%s."
              % (cost["requests_reserved_including_unknown_outcomes"], cost["usage_known_requests"], cost["prompt_tokens_reported"], cost["completion_tokens_reported"]),
              "Usage cost at frozen rates: USD %.6f. Unknown-outcome reservations: USD %.6f. These are conservative accounting values, not a provider invoice."
              % (cost["usage_cost_at_frozen_rates_usd"], cost["unknown_outcome_or_usage_reservation_usd"]),
              "coverage.csv partitions every metric slot into valid, legitimate N/A, technical_failed, or missing; completed = valid + N/A. judge_units_expected is zero for Original self-reference metrics. completion.json separately checks the Stage A and Stage B unit denominators.",
              "summary.csv reports each metric separately with mean, sample SD (ddof=1), 1–5 counts, valid count, N/A count and reasons. technical_failures.jsonl and per-unit attempts preserve failed/unfinished records.",
              "paired_differences.csv uses matching pair IDs and only pairs where both scores are valid. RAPTOR-minus-baseline is favorable when positive for five metrics, negative for semantic_drift. No Original self-scores enter reference-metric comparisons.", "",
              "## Findings and limits", ""]
    if not a["completed"] and not b["completed"]:
        lines += ["**No method effect has been measured by a real judge in this run.** All statistical result cells remain empty; no method advantage, disadvantage, factual success rate, or overall winner can be concluded. Offline synthetic test responses are not included in scientific scores."]
    else:
        lines += ["The following descriptive RAPTOR-minus-baseline means use only currently valid matched records. Read them with valid/missing/N/A coverage in the CSVs; incomplete coverage cannot support a full-grid conclusion.", "",
                  "| Model | Task/direction | Baseline | Metric | Paired n | RAPTOR minus baseline |", "|---|---|---|---|---:|---:|"]
        for row in pairs:
            if row["paired_count"]:
                lines.append("| %s | %s/%s | %s | %s | %s | %.4f |" % (
                    row["model_id"], row["task"], row["direction"], row["baseline"], row["metric"],
                    row["paired_count"], row["mean_paired_difference"]))
    lines += ["", "A higher target score can coexist with lower quality/naturalness/preservation or higher drift. No composite of the six metrics is defined. Free-generation entity changes are descriptive differences from Original, not necessarily violations of prompt-specified content.",
              "CounterFact truth labels are this judge's knowledge judgments, not external factual ground truth. Unverifiable factual claims have null target/overall ratings; refusals/nonfactual failures remain scoreable. Reference comparisons may have high/limited/none comparability.",
              "Sample SD describes the distribution of generation records. Repeated trial IDs/templates and generation base seeds do not establish independent entity samples, five outer split replicates, statistical significance, or causal effects.",
              "dim_gcav means mean-difference direction with scalar logistic calibration + GCAV, a CAA-style baseline; this is not a full original-CAA reproduction. This delivery contains no xRFM generation.",
              "Original raw files remain unchanged. No text was regenerated, no probe trained, no manuscript modified, and no cluster jobs submitted/canceled or timing configuration changed.", "",
              "## Reproduction and continuation", "", "Use the Python interpreter recorded above and scripts/judge_steering_six_metrics.py. Commands and paths are in RUN_COMMANDS.md. Complete CONFIG_TO_COMPLETE.json with endpoint, pinned model/version, supported request settings, credential environment variable, pricing and budget. Then freeze, run the real canary, run the full grid, and verify. Compatible canary scores remain in the same unit checkpoints.",
              "All saved actual requests exclude Authorization. Responses, parsed ratings, request/response identity, errors, retries, usage and atomic checkpoints reside in units/{A,B}/{record_id}/. Resume validates input/config/rubric/schema/code identity and restores accepted units without new calls.", ""]
    atomic_text(Path(run_dir) / "FINAL_REPORT.md", "\n".join(lines))
