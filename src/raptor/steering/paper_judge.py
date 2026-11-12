"""Offline export/import for the paper's separate four-dimensional 1–10 judge.

No model/API is called. Imported semantic scores need an explicit provenance
record. This protocol is distinct from optional six-dimensional 1–5 judging.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
from .controls import atomic_json, checkpoint_valid, digest, file_hash, summarize_trajectories

METHODS = ("original", "gcs", "raptor")
METRICS = ("target_sentiment_strength", "output_quality", "naturalness", "overall_steering_success")


def _new_json(path, value):
    """Atomic create, refusing replacement even if another collector won a race."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def collect(run_dirs, output):
    """Normalize two complete measured RAPTOR/GCS runs; never infer text or scores."""
    output = Path(output)
    provenance_path = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or provenance_path.exists():
        raise ValueError("refusing to overwrite collected generations or provenance")
    if len(run_dirs) != 2 or len({Path(p).resolve() for p in run_dirs}) != 2:
        raise ValueError("provide exactly two distinct completed RAPTOR and GCS run directories")
    shared_fields = ("model", "dataset", "direction", "revision", "dtype", "device", "chat_template",
                     "target_probability", "mode", "activation_site", "decoding", "generation_seeds", "reliability")
    shared = None
    methods, sources = {}, []
    for directory in run_dirs:
        directory = Path(directory)
        manifest_path, records_path = directory / "manifest.json", directory / "generations.json"
        summary_path = directory / "control_summary.json"
        manifest = json.loads(manifest_path.read_text())
        identity, probe = manifest["identity"], manifest["probe_metadata"]
        if identity.get("protocol") != "measured_gcav_v1": raise ValueError("unsupported controls run protocol")
        method = {"singlelr": "raptor", "raptor": "raptor", "gcs": "gcs"}.get(probe["method"])
        if method is None or method in methods: raise ValueError("require one own RAPTOR run and one own GCS run")
        config = identity["config"]
        for field, key in (("model", "model"), ("dataset", "dataset"), ("activation_site", "activation_site")):
            if probe[field] != config[key]: raise ValueError("probe/run identity mismatch: " + field)
        source_identity = dict(config={key: config[key] for key in shared_fields},
                               split_sha256=probe["split_sha256"], embedding_sha256=probe["embedding_sha256"],
                               prompts_sha256=identity["prompts_sha256"], source_sha256=identity["source_sha256"])
        if shared is None: shared = source_identity
        elif shared != source_identity: raise ValueError("RAPTOR/GCS model, split, prompts, seed or generation configuration mismatch")
        seeds = config["generation_seeds"]
        if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("invalid generation seed list")
        records = json.loads(records_path.read_text())
        if not isinstance(records, list) or not records or len(records) != manifest["record_count"]:
            raise ValueError("incomplete controls generation export")
        indexed, prompt_texts, unit_sources = {}, {}, []
        for record in records:
            if record["method"] != probe["method"] or any(record[key] != config[expected] for key, expected in
                    (("model_id", "model"), ("task", "dataset"), ("direction", "direction"))):
                raise ValueError("generation record differs from its own run identity")
            prompt, seed = record["prompt_id"], record["generation_seed"]
            if type(prompt) is not int or prompt < 0 or type(seed) is not int or seed - prompt * 100000 not in seeds:
                raise ValueError("invalid prompt identity or full generation seed")
            key = (record["model_id"], record["task"], record["direction"], prompt, seed)
            if key in indexed: raise ValueError("duplicate controls prompt/full-seed record")
            if any(not isinstance(record[field], str) for field in ("prompt_text", "original_output", "steered_output")):
                raise ValueError("generation texts must be strings, including retained empty outputs")
            if prompt_texts.setdefault(prompt, record["prompt_text"]) != record["prompt_text"]:
                raise ValueError("prompt ID changes text within a run")
            unit_key = dict(prompt_id=prompt, generation_seed=seed)
            unit_path = directory / "units" / (digest(unit_key) + ".json")
            unit_identity = dict(run_sha256=digest(identity), **unit_key)
            if not checkpoint_valid(unit_path, unit_identity): raise ValueError("missing/corrupt/incompatible completed controls unit")
            if json.loads(unit_path.read_text())["payload"] != record:
                raise ValueError("generation export differs from completed unit payload")
            indexed[key] = record
            unit_sources.append(dict(file=unit_path.name, sha256=file_hash(unit_path)))
        if len(prompt_texts) != manifest["prompt_count"] or len(records) != len(prompt_texts) * len(seeds):
            raise ValueError("incomplete prompt-by-seed grid")
        if {path.name for path in (directory / "units").glob("*.json")} != {item["file"] for item in unit_sources}:
            raise ValueError("unindexed controls units outside the declared complete grid")
        if json.loads(summary_path.read_text()) != summarize_trajectories(records):
            raise ValueError("controls completion summary differs from actual saved trajectories")
        methods[method] = indexed
        sources.append(dict(method=method, directory=str(directory.resolve()),
                            manifest_sha256=file_hash(manifest_path), generations_sha256=file_hash(records_path),
                            control_summary_sha256=file_hash(summary_path), unit_sources=unit_sources))
    if set(methods) != {"raptor", "gcs"} or set(methods["raptor"]) != set(methods["gcs"]):
        raise ValueError("missing cross-method prompt/full-seed pairs")
    rows = []
    for key in sorted(methods["raptor"]):
        raptor, gcs = methods["raptor"][key], methods["gcs"][key]
        if (raptor["prompt_text"] != gcs["prompt_text"] or raptor["original_output"] != gcs["original_output"]
                or raptor["original_token_ids"] != gcs["original_token_ids"]):
            raise ValueError("shared Original or prompt differs across methods; no normalization or reference selection allowed")
        base = {field: raptor[field] for field in ("model_id", "task", "direction", "prompt_id", "generation_seed", "prompt_text")}
        rows.append(dict(base, method="original", text=raptor["original_output"], token_ids=raptor["original_token_ids"]))
        for method in ("gcs", "raptor"):
            record = methods[method][key]
            rows.append(dict(base, method=method, text=record["steered_output"], token_ids=record["token_ids"]))
    provenance = dict(schema="paper_judge_collected_controls_v1", records=len(rows), pairs=len(rows)//3,
                      collected_payload_sha256=digest(rows),
                      shared_generation_identity=shared, sources=sources, new_generation=False, judge_called=False,
                      historical_paper_numbers_reproduced=False)
    _new_json(provenance_path, provenance)
    _new_json(output, rows)
    return dict(records=len(rows), pairs=len(rows)//3, generations=str(output), provenance=str(provenance_path))


def requests(generations, output):
    """Input rows contain model,task,direction,prompt_id,full generation_seed,method,text."""
    pairs, rows = {}, []
    for record in generations:
        pair = [record[k] for k in ("model_id","task","direction","prompt_id","generation_seed")]
        if record["method"] not in METHODS: raise ValueError("paper protocol supports Original/GCS/RAPTOR")
        pair_id = digest(pair); record_id = digest([*pair,record["method"]])
        if record["method"] in pairs.setdefault(pair_id,{}): raise ValueError("duplicate pair/method")
        pairs[pair_id][record["method"]]=record_id
        rows.append(dict(record_id=record_id,pair_id=pair_id,**record))
    if any(set(methods)!=set(METHODS) for methods in pairs.values()): raise ValueError("incomplete method pairs")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    mapping=output/"local_records.json"
    atomic_json(mapping,rows)
    items=[dict(record_id=r["record_id"],prompt=r["prompt_text"],candidate=r["text"],
                scoring=dict(scale=[1,10],metrics=list(METRICS),
                             validity="Record evaluable and valid independently, with reasons; retain every record")) for r in rows]
    # Ordering is stable anonymous hash order, with no method/model labels.
    atomic_json(output/"anonymous_requests.json", sorted(items,key=lambda r:r["record_id"]))
    atomic_json(output/"request_identity.json", dict(schema="paper_four_metrics_1to10_v1",
                records_sha256=file_hash(mapping),records=len(rows),pairs=len(pairs),
                caveat="New portable score transport; original historical prompt/score provenance not verified."))


def aggregate(run_dir, scores_path, provenance_path):
    run_dir=Path(run_dir); identity=json.loads((run_dir/"request_identity.json").read_text())
    records_path=run_dir/"local_records.json"
    if file_hash(records_path)!=identity["records_sha256"]:raise ValueError("record identity mismatch")
    records=json.loads(records_path.read_text());scores=json.loads(Path(scores_path).read_text())
    provenance=json.loads(Path(provenance_path).read_text())
    for field in ("judge_model","execution_context","scoring_date","rubric_description","source_kind"):
        if not provenance.get(field):raise ValueError("missing judge provenance "+field)
    by_id={r["record_id"]:r for r in scores}
    if len(by_id)!=len(scores) or set(by_id)!={r["record_id"] for r in records}:raise ValueError("missing/duplicate/extra score rows")
    pairs={}
    for record in records:
        score=by_id[record["record_id"]]
        if type(score.get("evaluable")) is not bool or type(score.get("valid")) is not bool or not score.get("reason"):
            raise ValueError("every score needs explicit evaluable/valid flags and reason")
        if score["valid"] and not score["evaluable"]:raise ValueError("valid requires evaluable")
        for metric in METRICS:
            value=score.get(metric)
            allowed_na=record["method"]=="original" and metric in {"output_quality","naturalness"}
            if value is None and (not score["valid"] or allowed_na): continue
            if type(value) not in (int,float) or not 1<=value<=10:raise ValueError("1–10 score required: "+metric)
        pairs.setdefault(record["pair_id"],[]).append(record)
    complete={key for key,rs in pairs.items() if all(by_id[r["record_id"]]["valid"] for r in rs)}
    summary=[]
    for method in METHODS:
        rs=[r for r in records if r["method"]==method and r["pair_id"] in complete]
        for metric in METRICS:
            values=[by_id[r["record_id"]][metric] for r in rs if by_id[r["record_id"]].get(metric) is not None]
            summary.append(dict(method=method,metric=metric,valid_N=len(values),
                mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values)>1 else None))
    result=dict(protocol="paper_four_metrics_1to10_v1",records=len(records),pairs=len(pairs),
                common_valid_pairs=len(complete),excluded_pairs=len(pairs)-len(complete),
                all_records_retained=True,score_sha256=file_hash(scores_path),provenance=provenance,
                summary=summary,historical_paper_numbers_reproduced=False)
    atomic_json(run_dir/"summary.json",result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("collect");a.add_argument("--runs",required=True,nargs="+",type=Path);a.add_argument("--out",required=True,type=Path)
    a=sub.add_parser("prepare");a.add_argument("--generations",required=True,type=Path);a.add_argument("--out",required=True,type=Path)
    a=sub.add_parser("aggregate");a.add_argument("--run-dir",required=True,type=Path);a.add_argument("--scores",required=True,type=Path);a.add_argument("--provenance",required=True,type=Path)
    args=p.parse_args(argv)
    if args.command=="collect":print(json.dumps(collect(args.runs,args.out),indent=2))
    elif args.command=="prepare":
        generations=json.loads(args.generations.read_text())
        provenance_path=args.generations.with_suffix(args.generations.suffix+".meta.json")
        if provenance_path.exists():
            provenance=json.loads(provenance_path.read_text())
            if provenance.get("schema")!="paper_judge_collected_controls_v1" or provenance.get("collected_payload_sha256")!=digest(generations):
                raise ValueError("collected generation payload differs from its source provenance")
        requests(generations,args.out)
    else:print(json.dumps(aggregate(args.run_dir,args.scores,args.provenance),indent=2))


if __name__=="__main__":main()
