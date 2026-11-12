"""Restartable orchestration of the 12-setting structure diagnostic.

Each training-fraction/repeat shard calls the retained native implementation.
Completed compatible shards are reused; a setting summary requires every shard.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from functools import lru_cache
import numpy as np
from scipy.stats import pearsonr, spearmanr
from raptor.steering.controls import atomic_json, digest, file_hash


@lru_cache(maxsize=64)
def _cached_embedding_hash(path, size, mtime_ns):
    return file_hash(path)


def embedding_hash(path):
    info=Path(path).stat()
    return _cached_embedding_hash(str(path),info.st_size,info.st_mtime_ns)


def tasks(config, embeddings, output):
    result=[]
    source=file_hash(Path(__file__).with_name("structure_validation.py"))
    for model in config["models"]:
        tag=model.replace("/","-")
        for dataset in config["datasets"]:
            embedding=Path(embeddings)/f"{tag}_{dataset}_embeddings.npz"
            for layer in config["layers"]:
                for fraction in config["train_fracs"]:
                    for seed in config["subsample_seeds"]:
                        identity=dict(model=model,dataset=dataset,layer=layer,fraction=fraction,
                                      subsample_seed=seed,config=config,source_sha256=source)
                        path=Path(output)/tag/dataset/f"layer_{layer}"/f"frac_{fraction}_seed_{seed}"
                        result.append(dict(identity=identity,embedding=str(embedding),output=str(path)))
    paths=[task["output"] for task in result]
    if len(set(paths))!=len(paths): raise ValueError("duplicate tasks/output paths")
    return result


def run_task(task):
    identity=dict(task["identity"],embedding_sha256=embedding_hash(task["embedding"]))
    output=Path(task["output"]);done=output/"complete.json";result=output/"acc_structure_validation_expA.json"
    if done.exists():
        previous=json.loads(done.read_text())
        if previous["identity"]==identity and result.is_file() and previous["result_sha256"]==file_hash(result):
            return "restored"
        raise ValueError("incompatible/corrupt structure checkpoint; choose a new output root or inspect it")
    output.mkdir(parents=True,exist_ok=True)
    config=identity["config"]
    command=[sys.executable,"-m","raptor.experiments.structure_validation","--emb_npz",task["embedding"],
        "--layer",str(identity["layer"]),"--out",str(output),"--exp","A",
        "--train_fracs",str(identity["fraction"]),"--repeats","1","--subsample_seed",str(identity["subsample_seed"]),
        "--eval_seed",str(config["eval_seed"]),"--eval_frac",str(config["eval_frac"]),
        "--seed",str(config["crossfit_seed"]),"--kfold",str(config["kfold"]),
        "--Cs",",".join(map(str,config["Cs"])),"--oracle_Cs",",".join(map(str,config["oracle_Cs"])),
        "--oracle",config["oracle"],"--sigma_floor",str(config["sigma_floor"])]
    if config["fit_intercept"]:command.append("--fit_intercept")
    command.append("--standardize" if config["standardize"] else "--no_standardize")
    command.append("--calibrate_oracle" if config["calibrate_oracle"] else "--no_calibrate_oracle")
    subprocess.run(command,check=True)
    data=json.loads(result.read_text())
    if data.get("mode")!="expA" or data["repeats"]!=1:raise ValueError("invalid structure shard")
    atomic_json(done,dict(identity=identity,result_sha256=file_hash(result),command=command))
    return "computed"


def aggregate(all_tasks, output):
    settings={};missing=[]
    for task in all_tasks:
        identity=task["identity"];root=Path(task["output"]);done=root/"complete.json";path=root/"acc_structure_validation_expA.json"
        if not done.exists() or not path.exists():missing.append(task["output"]);continue
        cp=json.loads(done.read_text())
        if cp["identity"]!={**identity,"embedding_sha256":embedding_hash(task["embedding"])} or cp["result_sha256"]!=file_hash(path):
            raise ValueError("mixed/corrupt structure shard")
        data=json.loads(path.read_text());key=(identity["model"],identity["dataset"],identity["layer"])
        settings.setdefault(key,[]).extend(data["mean_rows"])
    if missing:raise ValueError(f"incomplete structure grid: {len(missing)} shards missing")
    rows=[]
    for (model,dataset,layer),data in settings.items():
        config=all_tasks[0]["identity"]["config"];candidates=[]
        for C in config["Cs"]:
            true=[];pred=[]
            for fraction in config["train_fracs"]:
                matches=[r for r in data if r["C"]==C and r["frac"]==fraction]
                if len(matches)!=len(config["subsample_seeds"]):raise ValueError("incomplete fraction repeat coverage")
                true.append(np.mean([r["true_acc_mean"] for r in matches]));pred.append(np.mean([r["pred_acc_mean"] for r in matches]))
            candidates.append(dict(C=C,spearman=float(spearmanr(true,pred).statistic),pearson=float(pearsonr(true,pred).statistic)))
        valid=[r for r in candidates if np.isfinite(r["spearman"]) and np.isfinite(r["pearson"])]
        if not valid:raise ValueError("undefined correlations; no finite candidate to summarize")
        best=min(valid,key=lambda r:(-r["spearman"],r["C"]))
        rows.append(dict(model=model,dataset=dataset,layer=layer,**best,
                         paper_inverse_C_label=1/best["C"],
                         interpretation="descriptive best correlation over C; evaluated sweep, not independent test selection"))
    destination=Path(output)/"structure_summary.csv"
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("w",newline="") as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return dict(settings=len(rows),shards=len(all_tasks),summary=str(destination),historical_paper_numbers_reproduced=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["plan","run","aggregate"])
    p.add_argument("--config",type=Path,default=Path("configs/structure_validation.json"))
    p.add_argument("--embeddings",type=Path,default=Path("embeddings_all"))
    p.add_argument("--out",type=Path,required=True);p.add_argument("--task-index",type=int)
    a=p.parse_args(argv);config=json.loads(a.config.read_text());all_tasks=tasks(config,a.embeddings,a.out)
    if a.command=="plan":
        report=dict(tasks=len(all_tasks),settings=len(config["models"])*len(config["datasets"])*len(config["layers"]),
                    missing_embeddings=sorted({t["embedding"] for t in all_tasks if not Path(t["embedding"]).is_file()}),
                    output_paths_unique=True,historical_fraction_list_verified=config["historical_fraction_list_verified"])
        print(json.dumps(report,indent=2));return
    if a.command=="aggregate":print(json.dumps(aggregate(all_tasks,a.out),indent=2));return
    chosen=all_tasks if a.task_index is None else [all_tasks[a.task_index]]
    for task in chosen:print(task["output"],run_task(task))


if __name__=="__main__":main()
