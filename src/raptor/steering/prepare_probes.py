"""Export own-method probes or fit native CAA scalar calibration for steering."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import tempfile
import numpy as np
from .controls import file_hash, load_bundle


def save(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise ValueError("use a new probe output path")
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
            handle.flush(); os.fsync(handle.fileno())
        load_bundle(temporary)
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def export_checkpoints(paths, binding, output):
    """Never reads test accuracy; exports raw concept W and separate intercept."""
    from raptor.experiments.accuracy_preparation import validate_layer_set
    if binding["method"] not in {"singlelr", "gcs", "fixed_reg"}:
        raise ValueError("supported steering methods are singlelr/gcs/fixed_reg; no xRFM steering")
    finals, selection_count = [], 0
    for path in map(Path, paths):
        if re.fullmatch(r"layer_[0-9]+\.selection\.npz", path.name):
            selection_count += 1
            continue
        match = re.fullmatch(r"layer_(0|[1-9][0-9]*)\.npz", path.name)
        if match is None:
            raise ValueError(f"expected final layer_<integer>.npz checkpoint: {path}")
        finals.append((path, int(match.group(1))))
    if selection_count:
        print(f"[steering] excluded {selection_count} selection snapshot(s); "
              "only final layer_<integer>.npz checkpoints are exported", flush=True)
    if not finals:
        raise ValueError("no final layer_<integer>.npz checkpoints supplied")
    rows, sources, identities = {}, [], []
    for path, filename_layer in finals:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"].item()))
            identity = meta["identity"]
            if meta.get("status") != "complete": raise ValueError(f"incomplete final probe checkpoint: {path}")
            for key in ("model", "dataset", "method", "split_sha256", "embedding_sha256"):
                if identity[key] != binding[key]: raise ValueError("checkpoint/binding mismatch " + key)
            # Only explicitly audited candidate-validation checkpoints are accepted.
            details = meta.get("method_details", {})
            validation_source = (str(data["validation_source"].item()) if "validation_source" in data
                                 else details.get("validation_source", meta.get("validation_source")))
            if validation_source not in {"pre_refit_train_only", "shared_outer_validation_train_only"}:
                raise ValueError("checkpoint lacks explicit pre-refit validation provenance")
            concept = np.asarray(data["concept"], dtype=np.float64)
            val = float(data["validation_accuracy"])
        layer = int(identity["layer"])
        if layer != filename_layer:
            raise ValueError(f"checkpoint filename/identity layer mismatch: {path}")
        count, dim = identity.get("embedding_layers"), identity.get("embedding_dim")
        if type(count) is not int or count < 3 or type(dim) is not int or dim < 1:
            raise ValueError("checkpoint requires physical embedding_layers and embedding_dim")
        validate_layer_set(identity["model"], range(count), synthetic=identity.get("scope") == "synthetic")
        if concept.ndim != 1 or concept.shape != (dim + 1,):
            raise ValueError(f"probe dimension does not match embedding_dim: {path}")
        if layer in rows: raise ValueError("duplicate probe layer")
        rows[layer] = (concept, val)
        sources.append(dict(path=str(path), sha256=file_hash(path)))
        identities.append(identity)
    base = {k:v for k,v in identities[0].items() if k != "layer"}
    if any({k:v for k,v in identity.items() if k != "layer"} != base for identity in identities):
        raise ValueError("mixed probe identities")
    layers = list(range(base["embedding_layers"]))
    if set(rows) != set(layers):
        raise ValueError(f"complete physical layer set required: expected {len(layers)}, "
                         f"missing={sorted(set(layers) - set(rows))}, extra={sorted(set(rows) - set(layers))}")
    arrays = dict(W=np.stack([rows[i][0][:-1] for i in layers]),
                  B=np.asarray([rows[i][0][-1] for i in layers]),
                  val_acc_pre_refit=np.asarray([rows[i][1] for i in layers]))
    metadata = dict(binding, schema="raptor.steering.own_probe.v1", validation_source="pre_refit_train_only",
                    source_identity=base, sources=sources)
    save(output, arrays, metadata)


def fit_caa(embedding, split, binding, output):
    from raptor.experiments.accuracy_preparation import inspect_embedding
    from .scalar_calibration import fit_mean_difference
    from .gcav import select_active_layers
    inspected = inspect_embedding(Path(embedding), binding["model"], binding["dataset"],
                                  synthetic=binding.get("scope") == "synthetic")
    if file_hash(embedding) != binding["embedding_sha256"]: raise ValueError("embedding SHA mismatch")
    with np.load(split, allow_pickle=False) as data:
        split_meta = json.loads(str(data["metadata_json"].item()))
        if split_meta["split_sha256"] != binding["split_sha256"]: raise ValueError("split SHA mismatch")
        tr, val = data["train_sub_idx"], data["val_idx"]
    candidates, validation = [], []
    with np.load(embedding, allow_pickle=False) as data:
        layers = list(range(inspected["layers"]))
        for layer in layers:
            pos, neg = data[f"X_pos_{layer}"], data[f"X_neg_{layer}"]
            hidden = np.vstack([pos,neg]); labels=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
            candidate = fit_mean_difference(hidden, labels, tr)
            validation.append(float(np.mean((hidden[val]@candidate["W"]+candidate["b"] >= 0) == labels[val])))
            candidates.append(candidate)
        policy = binding["reliability"]
        active = select_active_layers([c["direction"] for c in candidates], validation,
                    layers=binding.get("layers"), accuracy_threshold=policy["threshold"],
                    reliability_enabled=policy["enabled"], reliability_comparison=policy["comparison"])
        # Native generation rule: freeze own candidate mask before any final refit.
        final = list(candidates)
        for layer in active:
            pos,neg=data[f"X_pos_{layer}"],data[f"X_neg_{layer}"]
            final[layer]=fit_mean_difference(np.vstack([pos,neg]), np.r_[np.ones(len(pos)),np.zeros(len(neg))], np.r_[tr,val])
    arrays=dict(W=np.stack([c["W"] for c in final]),B=np.asarray([c["b"] for c in final]),val_acc_pre_refit=np.asarray(validation))
    metadata=dict(binding,schema="raptor.steering.own_probe.v1",method="dim_gcav",
                  validation_source="pre_refit_train_only", active_layers=active,
                  final_refit_layers=active, epsilon=1e-6,
                  source_identity=dict(scalar_calibration_sha256=file_hash(Path(__file__).with_name("scalar_calibration.py")),
                                       split_file_sha256=file_hash(split), direction="raw positive-minus-negative unit mean difference"))
    save(output, arrays, metadata)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    export=sub.add_parser("export"); export.add_argument("--checkpoints",type=Path,nargs="+",required=True)
    caa=sub.add_parser("fit-caa"); caa.add_argument("--embedding",type=Path,required=True); caa.add_argument("--split",type=Path,required=True)
    for s in (export,caa):
        s.add_argument("--binding",type=Path,required=True); s.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv); binding=json.loads(a.binding.read_text())
    if a.command=="export": export_checkpoints(a.checkpoints,binding,a.out)
    else: fit_caa(a.embedding,a.split,binding,a.out)


if __name__=="__main__": main()
