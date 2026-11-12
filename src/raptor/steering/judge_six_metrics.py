"""Prepare and run reproducible, blinded six-metric judging of existing text.

No model loading, generation, training, Slurm operations, or manuscript edits.
Only ``canary`` and ``run`` make network calls, after an explicit configuration
has been frozen. All other commands are offline and use the standard library.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import datetime
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import shlex
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from .judge_six_protocol import (
    METHODS, METRICS_A, METRICS_B, PAIR_FIELDS, SHUFFLE_SEED, TARGETS,
    ProtocolError, require, response_schema, strict_json, validate_response,
)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_DIR = Path(__file__).with_name("rubric")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            require(line.strip(), "empty JSONL line: %s:%s" % (path, number))
            yield strict_json(line)


def write_jsonl(path, rows):
    atomic_text(path, "".join(canonical(row) + "\n" for row in rows))


def write_csv(path, fields, rows):
    f = io.StringIO(newline="")
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: canonical(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    atomic_text(path, f.getvalue())


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def code_identity():
    files = [Path(__file__), Path(__file__).with_name("judge_six_protocol.py"),
             Path(__file__).with_name("judge_six_summary.py")]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "installed-package-no-git"
    return {"git_commit": commit, "git_branch": "recorded-by-commit-and-source-digests",
            "files": {p.name: sha_file(p) for p in files}}


def unique_index(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        require(value not in result, "duplicate " + key + ": " + str(value))
        result[value] = row
    return result


def pair_key(row):
    return tuple(row[k] for k in PAIR_FIELDS)


def audit_delivery(delivery, production=True):
    """Audit a fresh portable JSONL export, never load historical session paths.

    records.jsonl contains complete own-generation records and a source_identity
    with checkpoint/tokenizer revisions, chat-template digest and decoding config.
    The historical conversation ratings are a separate reported-results source.
    """
    delivery = Path(delivery).resolve()
    source = delivery / "records.jsonl"
    flat = unique_index(read_jsonl(source), "record_id")
    require(bool(flat), "empty generation export")
    pairs, counts, identities = {}, collections.Counter(), {}
    for rid, row in flat.items():
        require(bool(re.fullmatch(r"[0-9a-f]{64}",rid)), "invalid record ID")
        require(row["method"] in METHODS, "unknown generation method")
        require(type(row["generation_seed"]) is int, "full generation seed required")
        require((row["task"],row["direction"]) in TARGETS, "unsupported task/direction")
        require(isinstance(row["prompt_text"],str) and isinstance(row["raw_output"],str), "invalid text")
        source_identity = row["source_identity"]
        for key in ("checkpoint_revision","tokenizer_revision","chat_template_sha256","decoding_config"):
            require(source_identity.get(key) is not None, "missing source identity: " + key)
        key = pair_key(row)
        pid = digest(key)
        require(row["pair_id"] == pid, "pair ID/full seed mismatch")
        pair_identity = dict(source_identity, prompt_text=row["prompt_text"])
        require(identities.setdefault(key,pair_identity)==pair_identity, "paired generation input identity mismatch")
        pair = pairs.setdefault(pid,{**{k:row[k] for k in PAIR_FIELDS},"pair_id":pid,"record_ids":{}})
        require(row["method"] not in pair["record_ids"], "duplicate method in pair")
        pair["record_ids"][row["method"]]=rid
        row["target_property"] = TARGETS[(row["task"],row["direction"])]
        row["checkpoint_revision"] = source_identity["checkpoint_revision"]
        row["finish_reason"] = row.get("finish_reason","unrecorded")
        counts[(row["model_id"],row["task"],row["direction"],row["method"])]+=1
    require(all(set(p["record_ids"])==set(METHODS) for p in pairs.values()), "incomplete five-method pairs")
    if production:
        require(len(flat)==9000 and len(pairs)==1800 and len(counts)==30, "optional full grid requires9000records/1800pairs/30conditions")
    coverage=[dict(zip(("model_id","task","direction","method"),k), records=v) for k,v in sorted(counts.items())]
    audit=dict(status="PASS",checked_at_utc=utc_now(),delivery=str(delivery),records=len(flat),pairs=len(pairs),
               source_files=1,source_sha_matches=1,identity_mismatches=0,effective_decoding_mismatches=0,
               all_text_preserved_exactly=True,generation_seed_full_values_used=True,
               missing_effective_decoding_metadata=[],source_and_payload_sha256={str(source):sha_file(source)},coverage=coverage)
    return flat,pairs,audit


def make_units(records, pairs):
    units = {"A": [], "B": []}
    for rid, row in sorted(records.items()):
        task_key = (row["task"], row["direction"])
        base = {"record_id": rid, "task": row["task"], "direction": row["direction"],
                "target_property": TARGETS[task_key], "prompt_text": row["prompt_text"]}
        units["A"].append({**base, "stage": "A", "candidate": row["raw_output"]})
        if row["method"] != "original":
            ref_id = pairs[row["pair_id"]]["record_ids"]["original"]
            require(ref_id in records, "missing Original reference")
            reference = records[ref_id]
            require(pair_key(row) == pair_key(reference), "Original pairing mismatch")
            units["B"].append({**base, "stage": "B", "candidate": row["raw_output"],
                               "reference": reference["raw_output"]})
    for stage in units:
        random.Random(SHUFFLE_SEED).shuffle(units[stage])
    return units


def prepare(delivery, run_dir, production=True):
    run_dir = Path(run_dir).resolve()
    require(not run_dir.exists(), "new run directory required; use existing run commands to resume")
    records, pairs, audit = audit_delivery(delivery, production=production)
    units = make_units(records, pairs)
    run_dir.mkdir(parents=True)
    write_json(run_dir / "source_audit.json", audit)
    atomic_text(run_dir / "RUBRIC.md", (PROTOCOL_DIR / "RUBRIC.md").read_text(encoding="utf-8"))
    write_jsonl(run_dir / "local_records.jsonl", sorted(records.values(), key=lambda r: r["record_id"]))
    id_fields = ["record_id", "pair_id", "model_id", "task", "direction", "method", "prompt_id", "generation_seed"]
    write_csv(run_dir / "local_id_map.csv", id_fields,
              [{key: row[key] for key in id_fields} for row in records.values()])
    canary = []
    workload = {}
    for stage, rows in units.items():
        write_jsonl(run_dir / ("stage_%s_inputs.jsonl" % stage), rows)
        write_json(run_dir / ("schema_%s.json" % stage), response_schema(stage))
        seen = set()
        for row in rows:
            source = records[row["record_id"]]
            key = tuple(source[k] for k in ("model_id", "task", "direction"))
            if key not in seen:
                canary.append({"stage": stage, "record_id": row["record_id"], "stratum": list(key)})
                seen.add(key)
        sizes = [len(canonical(r).encode("utf-8")) for r in rows]
        workload[stage] = {"units": len(rows), "unit_utf8_bytes_total": sum(sizes),
                           "unit_utf8_bytes_min": min(sizes), "unit_utf8_bytes_max": max(sizes)}
        write_json(run_dir / ("stage_%s_manifest.json" % stage), {
            "stage": stage, "units": len(rows), "shuffle_seed": SHUFFLE_SEED,
            "input_sha256": sha_file(run_dir / ("stage_%s_inputs.jsonl" % stage)),
            "schema_sha256": sha_file(run_dir / ("schema_%s.json" % stage)),
            "unit_identities_sha256": digest([digest(row) for row in rows]),
            "one_independent_context_per_unit": True,
        })
    write_json(run_dir / "canary_manifest.json", {"units": canary,
               "selection_rule": "first frozen-random-order unit in each model/task/direction stratum, separately per stage"})
    write_json(run_dir / "workload.json", {"stages": workload, "judge_calls_so_far": 0,
               "price_estimate": None, "reason": "model, request settings, pricing, and budget not supplied"})
    identity = code_identity()
    for relative in identity["files"]:
        atomic_text(run_dir / "provenance" / relative, (Path(__file__).parent / relative).read_text(encoding="utf-8"))
    atomic_text(run_dir / "CONFIG_TO_COMPLETE.json", (PROTOCOL_DIR / "judge_config.template.json").read_text())
    write_run_instructions(run_dir, Path(delivery).resolve())
    report_eda(run_dir, records, audit)
    tracked = ["source_audit.json", "RUBRIC.md", "PROTOCOL.md", "RUN_COMMANDS.md", "local_records.jsonl", "local_id_map.csv", "canary_manifest.json"]
    tracked += ["stage_%s_%s.json%s" % (s, kind, "l" if kind == "inputs" else "")
                for s in ("A", "B") for kind in ("inputs", "manifest")]
    tracked += ["schema_%s.json" % s for s in ("A", "B")]
    write_json(run_dir / "preparation.json", {
        "protocol": "steering_six_metrics_v1", "created_at_utc": utc_now(),
        "delivery": str(Path(delivery).resolve()), "code_identity": identity,
        "environment": {"python": sys.version, "executable": sys.executable, "platform": sys.platform},
        "command": sys.argv, "shuffle_seed": SHUFFLE_SEED,
        "files_sha256": {name: sha_file(run_dir / name) for name in tracked},
    })
    return {"status": "OFFLINE_PREPARED_JUDGE_NOT_RUN", "records": len(records),
            "stage_A_units": len(units["A"]), "stage_B_units": len(units["B"]), "run_dir": str(run_dir)}


def write_run_instructions(run_dir, delivery):
    command = "%s -m raptor.steering.judge_six_metrics" % shlex.quote(sys.executable)
    path = shlex.quote(str(run_dir))
    config = shlex.quote(str(run_dir / "CONFIG_TO_COMPLETE.json"))
    commands = """# Actual commands and continuation

Preparation command recorded by this run:

```bash
{command} prepare --delivery {delivery} --run-dir {run}
```

Preparation requires a fresh output directory. Reuse the existing run below
for continuation, and edit only CONFIG_TO_COMPLETE.json to fill explicit
model/endpoint/version, supported parameter evidence, token cap, prices and
USD/request limits. Load the credential through the named environment variable
or an existing secure shell configuration; never put its value in this file,
the config JSON, logs, code, or chat. Model capability/pricing evidence is
checked after a model is chosen. No model or temperature/seed is defaulted.

Freeze configuration, then run the minimal real 12-unit canary (six strata
in each of two stages). Its compatible accepted responses are final scores:

```bash
{command} freeze --run-dir {run} --config {config}
{command} canary --run-dir {run}
{command} run --run-dir {run}
{command} verify --run-dir {run}
```

The same canary/run commands resume; complete accepted units are restored.
No extra vote or score-based retry is possible. These commands run a single
CPU-light HTTPS client; they do not load local language models, generate text,
change timing experiments, or submit/cancel Slurm jobs.

Offline reaggregation and source-integrity verification:

```bash
{command} aggregate --run-dir {run}
{command} verify --run-dir {run}
```

aggregate/verify exit 2 until all 9,000 Stage A and 7,200 Stage B units have
valid accepted checkpoints (legitimate metric N/A does not count as failure).
Preparation exits 0 for a successful offline preparation only; it does not
claim scoring completion. A full run or canary with technical failures exits
nonzero. SIGINT/SIGTERM/SIGUSR1 requests a stop after the current durable unit;
interrupted network outcomes remain budget-reserved and bounded by retries.
""".format(command=command, delivery=shlex.quote(str(delivery)), run=path, config=config)
    atomic_text(run_dir / "RUN_COMMANDS.md", commands)
    atomic_text(run_dir / "PROTOCOL.md", """# Execution protocol: six-metric steering judgment v1

RUBRIC.md defines all anchors and direction/null rules. Stage A independently
scores four absolute metrics on 9,000 anonymous candidates. Stage B scores
two reference-relative metrics on 7,200 nonOriginal candidates. Original's
two self-reference metrics are local N/A. Contexts never include other units,
method/model identity, probe metadata, previous scores, or rankings. The fixed
shuffle seed is 20261104; input/schema/rubric hashes are in preparation.json.

Configuration is pending until frozen_config.json is written. The required
actual response model must identify the chosen version, rather than allowing
an unverified floating alias. No seed, temperature or other sampling settings
are sent unless explicitly configured and listed as supported with capability
evidence. The API's actual supported determinism controls are documented in
that configuration; byte-identical repeated scores are not promised.

The HTTPS chat completions adapter sends a single system/user exchange with
strict JSON schema and an explicit output-token cap. For an OpenAI endpoint,
the official reference is
https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
and structured-output guidance is
https://developers.openai.com/api/docs/guides/structured-outputs .
Other compatible endpoints must have their own capability/pricing evidence.

Before the first call, freeze model/version, endpoint, request parameters,
capability/pricing evidence, retry policy, budget, and all prepared identities.
Run the 12 real units selected before observing scores, covering both models
and all three task/directions in each phase. Reload every canary checkpoint
and verify it against its saved request and raw response. Null handling is
also tested offline with deliberately unverifiable/noncomparable fixtures;
live N/A labels are never forced for a canary. Synthetic tests are temporary
and cannot enter scores.jsonl. The real canary gates the full run.

Every actual attempt atomically saves its anonymous request (without the
Authorization header), raw response body/JSON, HTTP request ID, parsed result,
contract error or network/HTTP failure, usage and status. A per-unit checkpoint
binds input, preparation, model/config, rubric, schema and code identities.
One run-wide advisory lock prevents simultaneous writers. Raw responses can
reconstruct a lost final checkpoint without a new request.

Technical retry policy is frozen: normally at most three attempts per unit,
2/8-second backoffs, and only the declared transient HTTP statuses, network
errors, malformed/incomplete JSON and violations of the frozen response
contract. Refusals by the candidate are scored, not retried. Refusals by the
provider, actual model/version mismatch, or permanent authentication/request
errors halt for diagnosis. No accepted rating is retried for a low score,
unfavorable method ordering, unverifiability, or limited reference comparison.
Attempt limits persist across process restarts. Changed input/config/code or
corrupt accepted checkpoints are rejected, not silently mixed or overwritten.

The USD cap is checked before every request with conservative reservation.
For a confirmed byte-level BPE tokenizer, the upper input bound is the UTF-8
byte length of the entire serialized request (including schema) plus 1,024
framing tokens. Output reservation uses the full completion-token cap,
including reasoning tokens when charged as completion tokens. Actual usage
settles at the frozen rates with no cache discount; missing usage and unknown
network outcomes retain their full reservation. The transactional SQLite
ledger uses full synchronization and small updates. A request-count cap also
bounds retries. This is conservative cost accounting, not an invoice or a
model-independent tokenization theorem. Different billing/tokenizer families
require a suitable adapter before use.

All 9,000 records remain in merged output, including empty/refused/truncated
text and failures. Coverage partitions each metric into valid numeric, legal
N/A, technical_failed, or missing. Missing/failed nulls have pending
applicability and never enter N/A counts. Completion is assessed separately
for 9,000 Stage A units and 7,200 Stage B units. Summary statistics use only
valid numeric ratings, show all denominators and N/A reasons, and keep each
metric separate. Paired comparisons require both values within the same full
pair key; no Original self-comparison enters reference-metric contrasts.

Means/SDs and paired differences describe these generation records. They do
not establish independent trials, outer-split uncertainty, significance, or
an external factual gold standard. Original-relative free-generation content
differences have the descriptive limits specified in the rubric. Full claims
remain unavailable until mechanical and evaluable coverage are both reported.
""")


def report_eda(run_dir, records, audit):
    outputs = [r["raw_output"] for r in records.values()]
    groups = collections.defaultdict(list)
    for r in records.values():
        groups[(r["model_id"], r["task"], r["direction"])].append(r)
    rows = []
    for key, values in sorted(groups.items()):
        rows.append("| %s | %s | %s | %s | %s | %s |" % (
            *key, len(values), len({r["prompt_text"] for r in values}), len({r["raw_output"] for r in values})))
    text = """# Existing steering inputs: offline data audit

Generated at {now}. Source directory: `{delivery}`.

JSONL holds one structured object per record; CSV is parsed with a CSV parser
so embedded newlines remain in cells. Analysis uses Python's standard json/csv
libraries. All source and export SHA-256 values are in source_audit.json.

Revalidated {records} unique record IDs, {pairs} complete five-method pairs,
{files} original source files and their packaged copies. Prompt, checkpoint,
tokenizer revision, chat template, quantization, and configured decoding match
within every pair. Available effective decoding metadata also matches.

Empty/whitespace outputs: {empty}. Distinct exact output strings: {distinct}.
Repeated strings remain separate generation records. Output characters range
from {minimum} to {maximum}. No score, linguistic success, refusal, or factual
status is inferred by this mechanical audit. Text is never filtered or changed.

| Model | Task | Direction | Records | Distinct prompt texts | Distinct output texts |
|---|---|---|---:|---:|---:|
{rows}

Missing effective runtime metadata: `{missing}`. These legacy omissions remain
explicit; configured checkpoint/tokenizer/decoding metadata is available and
matched. No runtime information is fabricated.

CounterFact trial IDs repeat free-generation templates: trial IDs are not
independent entity questions. Use full generation_seed values for pairing;
neither those seeds nor the fixed probe seed are five outer split seeds.
Next analysis must use separate Stage A and Stage B contexts and distinguish
legal N/A from technical failures. Original-relative comparison has descriptive
limits for details left unspecified by the prompt. Report record SDs without
independent-sample inference. No new text generation is needed.
""".format(now=utc_now(), delivery=audit["delivery"],
           records=len(records), pairs=audit["pairs"], files=audit["source_files"], empty=sum(not o.strip() for o in outputs),
           distinct=len(set(outputs)), minimum=min(map(len, outputs)), maximum=max(map(len, outputs)),
           rows="\n".join(rows), missing=canonical(audit["missing_effective_decoding_metadata"]))
    atomic_text(run_dir / "inputs_eda_report.md", text)


def verify_prepared(run_dir, check_code=True, check_sources=False):
    run_dir = Path(run_dir)
    prep = read_json(run_dir / "preparation.json")
    for name, expected in prep["files_sha256"].items():
        require(sha_file(run_dir / name) == expected, "prepared file changed: " + name)
    if check_code:
        require(code_identity() == prep["code_identity"], "judge code/commit changed since preparation")
    if check_sources:
        audit = read_json(run_dir / "source_audit.json")
        for path, expected in audit["source_and_payload_sha256"].items():
            require(sha_file(path) == expected, "original input changed: " + path)
    return prep


def validate_config(config):
    expected = set(read_json(PROTOCOL_DIR / "judge_config.template.json"))
    require(set(config) == expected, "configuration fields differ from template")
    require(config["provider"] == "chat_completions_https", "unsupported provider adapter")
    url = urllib.parse.urlparse(config["endpoint"] or "")
    require(url.scheme == "https" and url.hostname and not url.username and not url.password and not url.query and not url.fragment,
            "endpoint must be an HTTPS chat completions URL without credentials/query")
    for field in ("model", "expected_response_model", "api_key_env", "capability_evidence"):
        require(isinstance(config[field], str) and config[field].strip(), "missing config: " + field)
    require(re.fullmatch(r"[A-Z][A-Z0-9_]*", config["api_key_env"]) is not None, "credential environment name")
    require(config["tokenizer_family"] == "byte_level_bpe", "cost reservation requires confirmed byte-level BPE tokenizer")
    require(type(config["max_completion_tokens"]) is int and config["max_completion_tokens"] > 0, "output token cap required")
    require(isinstance(config["request_parameters"], dict), "request parameters")
    allowed = {"temperature", "seed", "top_p", "reasoning_effort", "store", "service_tier"}
    params = config["request_parameters"]
    require(set(params) <= allowed, "unsupported parameter or possible credential/identity leak")
    require(isinstance(config["supported_parameters"], list) and set(params) <= set(config["supported_parameters"]),
            "parameters must be explicitly verified as supported")
    require(config["timeout_seconds"] > 0, "request timeout")
    retry = config["retry"]
    require(set(retry) == {"max_attempts_per_unit", "backoff_seconds", "retry_http_status"}, "retry fields")
    require(type(retry["max_attempts_per_unit"]) is int and 1 <= retry["max_attempts_per_unit"] <= 5, "bounded retry count")
    require(len(retry["backoff_seconds"]) >= retry["max_attempts_per_unit"] - 1 and
            all(type(x) in (int, float) and 0 <= x <= 60 for x in retry["backoff_seconds"]), "bounded backoff")
    require(set(retry["retry_http_status"]) <= {408, 409, 429, 500, 502, 503, 504}, "only predefined technical HTTP retries")
    budget = config["budget"]
    require(set(budget) == {"currency", "max_usd", "input_usd_per_million", "output_usd_per_million", "pricing_evidence", "max_requests"}, "budget fields")
    require(budget["currency"] == "USD" and isinstance(budget["pricing_evidence"], str) and budget["pricing_evidence"].strip(), "pricing evidence required")
    for field in ("max_usd", "input_usd_per_million", "output_usd_per_million"):
        x = budget[field]
        require(type(x) in (int, float) and math.isfinite(x) and x >= 0, "missing/invalid budget: " + field)
    require(type(budget["max_requests"]) is int and budget["max_requests"] > 0, "request budget required")
    return config


def freeze_config(run_dir, config_path):
    run_dir = Path(run_dir)
    prep = verify_prepared(run_dir, check_sources=True)
    config = validate_config(read_json(config_path))
    require(not (run_dir / "frozen_config.json").exists(), "configuration already frozen")
    frozen = {"config": config, "preparation_sha256": sha_file(run_dir / "preparation.json"),
              "code_identity": prep["code_identity"], "frozen_at_utc": utc_now()}
    frozen["identity_sha256"] = digest(frozen)
    write_json(run_dir / "frozen_config.json", frozen)
    return {"status": "CONFIG_FROZEN_NO_CALLS", "model": config["model"], "identity_sha256": frozen["identity_sha256"]}


def load_frozen(run_dir):
    run_dir = Path(run_dir)
    verify_prepared(run_dir)
    frozen = read_json(run_dir / "frozen_config.json")
    require(digest({k: v for k, v in frozen.items() if k != "identity_sha256"}) == frozen["identity_sha256"], "frozen configuration corrupt")
    require(frozen["preparation_sha256"] == sha_file(run_dir / "preparation.json"), "preparation/config mismatch")
    validate_config(frozen["config"])
    return frozen


def request_body(run_dir, unit, config):
    stage = unit["stage"]
    return {
        "model": config["model"], "max_completion_tokens": config["max_completion_tokens"],
        **config["request_parameters"],
        "messages": [
            {"role": "system", "content": Path(run_dir, "RUBRIC.md").read_text(encoding="utf-8") +
             "\nThis request is Stage " + stage + ". Judge only that stage's metrics. Output JSON."},
            {"role": "user", "content": canonical(unit)},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "steering_stage_" + stage, "strict": True, "schema": response_schema(stage)}},
    }


def unit_identity(unit, frozen):
    return {"input_sha256": digest(unit), "frozen_identity_sha256": frozen["identity_sha256"],
            "record_id": unit["record_id"], "stage": unit["stage"]}


def unit_dir(run_dir, unit):
    require(unit["stage"] in ("A", "B") and re.fullmatch(r"[0-9a-f]{64}", unit["record_id"]), "unsafe unit path")
    return Path(run_dir) / "units" / unit["stage"] / unit["record_id"]


def load_checkpoint(run_dir, unit, frozen):
    path = unit_dir(run_dir, unit) / "checkpoint.json"
    if not path.exists():
        return None
    checkpoint = read_json(path)
    require(checkpoint["identity"] == unit_identity(unit, frozen), "checkpoint identity mismatch")
    require(checkpoint["status"] in ("complete", "technical_failed"), "checkpoint status")
    if checkpoint["status"] == "complete":
        validate_response(checkpoint["parsed"], unit)
        require(checkpoint["parsed_sha256"] == digest(checkpoint["parsed"]), "checkpoint parsed data corrupt")
        attempt = unit_dir(run_dir, unit) / "attempts" / ("%04d" % checkpoint["accepted_attempt"])
        raw = read_json(attempt / "response.json")
        parsed = parse_api_response(raw, unit, frozen["config"])
        require(parsed == checkpoint["parsed"], "checkpoint/raw response mismatch")
        require(read_json(attempt / "identity.json") == unit_identity(unit, frozen), "attempt identity mismatch")
        require(read_json(attempt / "request.json") == request_body(run_dir, unit, frozen["config"]), "request/checkpoint mismatch")
    return checkpoint


def parse_api_response(raw, unit, config):
    require(isinstance(raw, dict), "API response must be an object")
    require(raw.get("model") == config["expected_response_model"], "actual judge model/version mismatch")
    require(isinstance(raw.get("choices"), list) and len(raw["choices"]) == 1, "one response required")
    choice = raw["choices"][0]
    require(isinstance(choice, dict), "response choice must be an object")
    require(choice.get("finish_reason") == "stop", "response incomplete/refused (no outcome-based retry)")
    message = choice.get("message", {})
    require(isinstance(message, dict), "response message must be an object")
    require(not message.get("refusal"), "provider refusal (no automatic retry)")
    require(isinstance(message.get("content"), str), "response text missing")
    return validate_response(strict_json(message["content"]), unit)


class BudgetStop(RuntimeError):
    pass


class RunStop(RuntimeError):
    pass


@contextlib.contextmanager
def run_lock(run_dir):
    path = Path(run_dir) / ".writer.lock"
    with path.open("a") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ProtocolError("another process is writing this run")
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def reserved_cost(body, config):
    # For confirmed byte-level BPE, one token cannot be smaller than one UTF-8
    # byte; include the entire serialized schema/body plus 1024 framing tokens.
    # This is deliberately conservative, not an estimate of expected usage.
    bound = len(canonical(body).encode("utf-8")) + 1024
    budget = config["budget"]
    cost = (bound * budget["input_usd_per_million"] +
            config["max_completion_tokens"] * budget["output_usd_per_million"]) / 1_000_000
    return bound, cost


@contextlib.contextmanager
def budget_db(run_dir):
    # Small transactional updates avoid rewriting a 16,200-entry JSON ledger
    # after every API call. DELETE journal + FULL sync is suitable for the
    # single-writer run lock and does not require shared-memory WAL files.
    connection = sqlite3.connect(str(Path(run_dir) / "budget_ledger.sqlite3"))
    try:
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE IF NOT EXISTS attempts (path TEXT PRIMARY KEY, entry_json TEXT NOT NULL, charged REAL NOT NULL)")
        with connection:
            yield connection
    finally:
        connection.close()


def budget_entries(run_dir):
    if not (Path(run_dir) / "budget_ledger.sqlite3").exists():
        return {}
    with budget_db(run_dir) as db:
        return {path: strict_json(value) for path, value in db.execute("SELECT path, entry_json FROM attempts")}


def has_reservation(run_dir, relative):
    if not (Path(run_dir) / "budget_ledger.sqlite3").exists():
        return False
    with budget_db(run_dir) as db:
        return db.execute("SELECT 1 FROM attempts WHERE path=?", (relative,)).fetchone() is not None


def budget_reserve(run_dir, body, config, attempt_relative):
    bound, reservation = reserved_cost(body, config)
    with budget_db(run_dir) as db:
        require(db.execute("SELECT 1 FROM attempts WHERE path=?", (attempt_relative,)).fetchone() is None,
                "duplicate request reservation")
        count, charged = db.execute("SELECT COUNT(*), COALESCE(SUM(charged), 0) FROM attempts").fetchone()
        if count >= config["budget"]["max_requests"] or charged + reservation > config["budget"]["max_usd"] + 1e-12:
            raise BudgetStop("request or conservative USD budget cap reached before request")
        entry = {"reserved_usd": reservation, "input_token_upper_bound": bound,
                 "state": "reserved_unknown_outcome", "reserved_at_utc": utc_now(), "request_sha256": digest(body)}
        db.execute("INSERT INTO attempts VALUES (?, ?, ?)", (attempt_relative, canonical(entry), reservation))


def settle_budget(run_dir, attempt_relative, raw, config):
    exceeded = False
    with budget_db(run_dir) as db:
        item = db.execute("SELECT entry_json FROM attempts WHERE path=?", (attempt_relative,)).fetchone()
        require(item is not None, "response without request cost reservation")
        entry = strict_json(item[0])
        usage = raw.get("usage") or {}
        incoming, outgoing = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if type(incoming) is int and type(outgoing) is int and incoming >= 0 and outgoing >= 0:
            entry["usage"] = usage
            exceeded = incoming > entry["input_token_upper_bound"] or outgoing > config["max_completion_tokens"]
            b = config["budget"]
            entry["charged_usd"] = (incoming * b["input_usd_per_million"] + outgoing * b["output_usd_per_million"]) / 1_000_000
            entry["state"] = "usage_bound_exceeded" if exceeded else "usage_cost_at_frozen_rates_no_cache_discount"
        else:
            entry["state"] = "usage_missing_reserved_worst_case"
        db.execute("UPDATE attempts SET entry_json=?, charged=? WHERE path=?",
                   (canonical(entry), entry.get("charged_usd", entry["reserved_usd"]), attempt_relative))
    if exceeded:
        raise RunStop("usage exceeded frozen token bounds; halt and audit budget")


def http_call(config, body):
    key = os.environ.get(config["api_key_env"])
    require(bool(key), "credential not set: " + config["api_key_env"])
    request = urllib.request.Request(config["endpoint"], data=canonical(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    # Do not forward an Authorization header through redirects.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(request, timeout=config["timeout_seconds"]) as response:
        return response.read().decode("utf-8"), {"status": response.status,
                "request_id": response.headers.get("x-request-id")}


def redact(text, config):
    key = os.environ.get(config["api_key_env"])
    return text.replace(key, "[REDACTED]") if key else text


def process_unit(run_dir, unit, frozen, transport=http_call, sleep=time.sleep):
    """Never repeat an accepted score; attempts are bounded across restarts."""
    run_dir = Path(run_dir)
    config = frozen["config"]
    existing = load_checkpoint(run_dir, unit, frozen)
    if existing:
        return "restored_" + existing["status"]
    directory = unit_dir(run_dir, unit)
    attempts_root = directory / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    identity = unit_identity(unit, frozen)
    body = request_body(run_dir, unit, config)
    retry = config["retry"]
    # Reconstruct a checkpoint from a durable valid raw response after a crash.
    attempts = sorted(p for p in attempts_root.iterdir() if p.is_dir() and p.name.isdigit())
    for attempt in attempts:
        if not (attempt / "identity.json").exists():
            raise ProtocolError("partial attempt identity; audit required")
        require(read_json(attempt / "identity.json") == identity, "attempt input/config mismatch")
        require(read_json(attempt / "request.json") == body, "attempt request mismatch")
        if (attempt / "response.json").exists():
            raw = read_json(attempt / "response.json")
            settle_budget(run_dir, str(attempt.relative_to(run_dir)), raw, config)
            if raw.get("model") != config["expected_response_model"] or any(
                (c.get("message") or {}).get("refusal") or c.get("finish_reason") == "content_filter"
                for c in raw.get("choices", [])):
                raise RunStop("saved response has model mismatch or provider refusal")
            try:
                parsed = parse_api_response(raw, unit, config)
            except (ValueError, TypeError, KeyError):
                continue
            write_json(attempt / "parsed.json", parsed)
            save_complete(directory, identity, parsed, int(attempt.name))
            return "recovered_from_raw_response"
    used = max([int(p.name) for p in attempts] or [0])
    # A fatal error remains fatal after restart; do not turn it into a retry.
    if attempts and (attempts[-1] / "error.json").exists():
        error = read_json(attempts[-1] / "error.json")
        if error.get("fatal"):
            raise RunStop("previous fatal judge/configuration error; see saved error")
    if attempts and not (attempts[-1] / "outcome.json").exists():
        # A network/crash outcome is unknown and conservatively budgeted.
        write_json(attempts[-1] / "outcome.json", {"status": "interrupted_unknown_outcome", "technical": True})
    for number in range(used + 1, retry["max_attempts_per_unit"] + 1):
        if number > 1:
            sleep(retry["backoff_seconds"][number - 2])
        attempt = attempts_root / ("%04d" % number)
        # Reserve before creating an attempt: hitting the cap must not consume
        # a retry, and a crash after reservation still counts the possible cost.
        relative = str(attempt.relative_to(run_dir))
        if has_reservation(run_dir, relative):
            raise RunStop("orphaned cost reservation; audit before any duplicate request")
        budget_reserve(run_dir, body, config, relative)
        attempt.mkdir()
        write_json(attempt / "identity.json", identity)
        write_json(attempt / "request.json", body)
        write_json(attempt / "started.json", {"at_utc": utc_now(), "endpoint": config["endpoint"]})
        try:
            raw_text, metadata = transport(config, body)
        except urllib.error.HTTPError as exc:
            raw_text = redact(exc.read().decode("utf-8", errors="replace"), config)
            atomic_text(attempt / "response_body.txt", raw_text)
            fatal = exc.code not in retry["retry_http_status"]
            write_json(attempt / "error.json", {"category": "http_error", "http_status": exc.code, "fatal": fatal})
            write_json(attempt / "outcome.json", {"status": "technical_error", "retryable": not fatal})
            if fatal:
                raise RunStop("nonretryable HTTP status %s; details saved without credentials" % exc.code)
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionError, socket.timeout, OSError) as exc:
            write_json(attempt / "error.json", {"category": "network_error", "type": type(exc).__name__, "fatal": False})
            write_json(attempt / "outcome.json", {"status": "technical_error", "retryable": True})
            continue
        raw_text = redact(raw_text, config)
        atomic_text(attempt / "response_body.txt", raw_text)
        write_json(attempt / "http_metadata.json", metadata)
        try:
            raw = strict_json(raw_text)
            require(isinstance(raw, dict), "API response must be an object")
        except (ValueError, TypeError) as exc:
            write_json(attempt / "error.json", {"category": "invalid_api_json", "type": type(exc).__name__, "fatal": False})
            write_json(attempt / "outcome.json", {"status": "technical_error", "retryable": True})
            continue
        write_json(attempt / "response.json", raw)
        settle_budget(run_dir, relative, raw, config)
        model_mismatch = raw.get("model") != config["expected_response_model"]
        choices = raw.get("choices") or []
        refusal = any(isinstance(c, dict) and (
            (isinstance(c.get("message"), dict) and c["message"].get("refusal")) or c.get("finish_reason") == "content_filter") for c in choices)
        if model_mismatch or refusal:
            write_json(attempt / "error.json", {"category": "model_mismatch" if model_mismatch else "provider_refusal", "fatal": True})
            write_json(attempt / "outcome.json", {"status": "fatal", "retryable": False})
            raise RunStop("judge model mismatch or provider refusal; no automatic rejudging")
        try:
            parsed = parse_api_response(raw, unit, config)
        except (ValueError, TypeError, KeyError) as exc:
            write_json(attempt / "error.json", {"category": "response_contract_error", "reason": redact(str(exc), config), "fatal": False})
            write_json(attempt / "outcome.json", {"status": "technical_error", "retryable": True})
            continue
        write_json(attempt / "parsed.json", parsed)
        write_json(attempt / "outcome.json", {"status": "complete", "retryable": False})
        save_complete(directory, identity, parsed, number)
        return "computed_complete"
    write_json(directory / "checkpoint.json", {"identity": identity, "status": "technical_failed",
               "attempts": retry["max_attempts_per_unit"], "at_utc": utc_now()})
    return "technical_failed"


def save_complete(directory, identity, parsed, number):
    write_json(directory / "checkpoint.json", {"identity": identity, "status": "complete", "parsed": parsed,
               "parsed_sha256": digest(parsed), "accepted_attempt": number, "at_utc": utc_now()})


def run_judge(run_dir, canary=False, limit=None):
    run_dir = Path(run_dir)
    frozen = load_frozen(run_dir)
    config = frozen["config"]
    require(bool(os.environ.get(config["api_key_env"])), "credential not set: " + config["api_key_env"])
    if not canary:
        report = read_json(run_dir / "canary_status.json")
        require(report["status"] == "PASS" and report["frozen_identity_sha256"] == frozen["identity_sha256"], "real canary must pass before full run")
    selected = {(r["stage"], r["record_id"]) for r in read_json(run_dir / "canary_manifest.json")["units"]}
    counts = collections.Counter()
    stop = {"requested": False}
    previous = {}

    def on_signal(signum, frame):
        stop["requested"] = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        previous[sig] = signal.signal(sig, on_signal)
    try:
        with run_lock(run_dir):
            verify_prepared(run_dir, check_sources=True)
            for stage in ("A", "B"):
                for unit in read_jsonl(run_dir / ("stage_%s_inputs.jsonl" % stage)):
                    if canary and (stage, unit["record_id"]) not in selected:
                        continue
                    if stop["requested"] or (limit is not None and sum(counts.values()) >= limit):
                        raise RunStop("stopped between atomic units; restart restores completed work")
                    status = process_unit(run_dir, unit, frozen)
                    counts[status] += 1
                    if sum(counts.values()) % 25 == 0 or canary:
                        print(canonical({"at_utc": utc_now(), "stage": stage, "status": status, "counts": counts}), flush=True)
            if canary:
                failures = []
                for stage in ("A", "B"):
                    for unit in read_jsonl(run_dir / ("stage_%s_inputs.jsonl" % stage)):
                        if (stage, unit["record_id"]) in selected:
                            checkpoint = load_checkpoint(run_dir, unit, frozen)
                            if not checkpoint or checkpoint["status"] != "complete":
                                failures.append((stage, unit["record_id"]))
                report = {"status": "PASS" if not failures else "INCOMPLETE",
                          "frozen_identity_sha256": frozen["identity_sha256"], "units": len(selected), "failures": failures,
                          "resume_validation": "every checkpoint reloaded and matched raw response, input, request, config and schema",
                          "null_rule_validation": "offline contract tests; live canary nulls are never forced",
                          "at_utc": utc_now()}
                write_json(run_dir / "canary_status.json", report)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return dict(counts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prepare_p = sub.add_parser("prepare", help="offline input verification and anonymous stage construction")
    prepare_p.add_argument("--delivery", required=True, type=Path)
    prepare_p.add_argument("--run-dir", required=True, type=Path)
    freeze_p = sub.add_parser("freeze", help="freeze explicit judge configuration without calling API")
    freeze_p.add_argument("--run-dir", required=True, type=Path)
    freeze_p.add_argument("--config", required=True, type=Path)
    for name in ("canary", "run", "aggregate", "verify"):
        sp = sub.add_parser(name)
        sp.add_argument("--run-dir", required=True, type=Path)
        if name in ("canary", "run"):
            sp.add_argument("--limit", type=int)
    args = p.parse_args(argv)
    result, code = None, 0
    try:
        if args.command == "prepare":
            result = prepare(args.delivery, args.run_dir)
            from .judge_six_summary import aggregate
            aggregate(args.run_dir)
        elif args.command == "freeze":
            with run_lock(args.run_dir):
                result = freeze_config(args.run_dir, args.config)
        elif args.command in ("canary", "run"):
            try:
                result = run_judge(args.run_dir, canary=args.command == "canary", limit=args.limit)
            finally:
                from .judge_six_summary import aggregate
                completion = aggregate(args.run_dir)
            if args.command == "canary":
                passed = read_json(args.run_dir / "canary_status.json")["status"] == "PASS"
                code = 0 if passed else 2
                result = {"status": "CANARY_PASS" if passed else "CANARY_INCOMPLETE", "counts": result,
                          "full_grid_status": completion["status"]}
            else:
                code = 0 if completion["status"] == "COMPLETE" else 2
                result = {"status": completion["status"], "counts": result}
        elif args.command == "aggregate":
            from .judge_six_summary import aggregate
            with run_lock(args.run_dir):
                result = aggregate(args.run_dir)
            code = 0 if result["status"] == "COMPLETE" else 2
        elif args.command == "verify":
            verify_prepared(args.run_dir, check_sources=True)
            from .judge_six_summary import aggregate
            with run_lock(args.run_dir):
                result = aggregate(args.run_dir)
            code = 0 if result["status"] == "COMPLETE" else 2
    except (ProtocolError, BudgetStop, RunStop, FileNotFoundError) as exc:
        print(canonical({"status": "STOPPED_OR_INCOMPLETE", "error": str(exc)}), file=sys.stderr)
        return 2
    print(canonical(result), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
