#!/usr/bin/env python3
"""Plot stored scalar summaries, paired differences, or complete seed layer curves."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"singlelr": "RAPTOR", "xrfm": "xRFM", "gcs": "GCS", "fixed_reg": "Fixed-reg", "dim_midpoint": "CAA midpoint"}
COLORS = {"singlelr": "#7b3294", "xrfm": "#008837", "gcs": "#e66101", "fixed_reg": "#3574b1", "dim_midpoint": "#b3588b"}


def _cells(payload, args):
    return [cell for cell in payload["cells"] if cell["protocol"] == args.protocol
            and cell["metric"] == args.metric and cell["status"] == "complete"]


def heatmaps(payload, args):
    cells = _cells(payload, args)
    if not cells:
        raise ValueError("no complete accuracy cells for requested protocol/metric")
    models, datasets = sorted({c["model"] for c in cells}), sorted({c["dataset"] for c in cells})
    methods = [m for m in LABELS if any(c["method"] == m for c in cells)]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.5*len(methods), max(3.5, .55*len(models))), constrained_layout=True)
    for ax, method in zip(np.atleast_1d(axes), methods):
        lookup = {(c["model"], c["dataset"]): c["mean"] for c in cells if c["method"] == method}
        matrix = np.array([[lookup.get((model, dataset), np.nan) for dataset in datasets] for model in models])
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(LABELS[method]); ax.set_xticks(range(len(datasets)), datasets, rotation=45, ha="right")
        ax.set_yticks(range(len(models)), [m.rsplit("/",1)[-1] for m in models])
        for i in range(len(models)):
            for j in range(len(datasets)):
                if np.isfinite(matrix[i,j]):
                    ax.text(j,i,f"{matrix[i,j]:.3f}",ha="center",va="center",fontsize=7)
    fig.colorbar(image, ax=np.atleast_1d(axes).tolist(), label="mean test accuracy")
    fig.savefig(args.out_dir/f"{args.protocol}_{args.metric}.pdf");plt.close(fig)
    lookup = {(c["model"],c["dataset"],c["method"]):c for c in cells}
    with (args.out_dir/f"{args.protocol}_{args.metric}.md").open("w") as out:
        out.write("| Model | Dataset | "+" | ".join(LABELS[m]+" mean±sd" for m in methods)+" |\n")
        out.write("|---|---|"+"---:|"*len(methods)+"\n")
        for model in models:
            for dataset in datasets:
                values=[]
                for method in methods:
                    cell=lookup.get((model,dataset,method))
                    values.append(f"{cell['mean']:.4f} ± {cell['sample_std_ddof1']:.4f}" if cell else "—")
                out.write(f"| {model} | {dataset} | "+" | ".join(values)+" |\n")


def differences(payload, args):
    rows = [row for row in payload["paired_comparisons"] if row["protocol"] == args.protocol
            and row["metric"] == args.metric and row["status"] == "complete"]
    if not rows:
        raise ValueError("no complete paired seed comparisons")
    models, datasets = sorted({r["model"] for r in rows}), sorted({r["dataset"] for r in rows})
    baselines=[m for m in LABELS if m != "singlelr" and any(r["baseline"] == m for r in rows)]
    values=[100*r["mean_paired_difference"] for r in rows]
    limit=max(1, max(abs(v) for v in values))
    fig, axes=plt.subplots(1,len(baselines),figsize=(4.5*len(baselines),max(3.5,.55*len(models))),constrained_layout=True)
    for ax, baseline in zip(np.atleast_1d(axes),baselines):
        lookup={(r["model"],r["dataset"]):r for r in rows if r["baseline"] == baseline}
        matrix=np.array([[100*lookup[(m,d)]["mean_paired_difference"] if (m,d) in lookup else np.nan for d in datasets] for m in models])
        image=ax.imshow(matrix,vmin=-limit,vmax=limit,cmap="PiYG")
        ax.set_title("RAPTOR − "+LABELS[baseline]);ax.set_xticks(range(len(datasets)),datasets,rotation=45,ha="right")
        ax.set_yticks(range(len(models)),[m.rsplit("/",1)[-1] for m in models])
        for i,m in enumerate(models):
            for j,d in enumerate(datasets):
                if (m,d) in lookup:
                    ci=lookup[(m,d)]["confidence_interval_95"]
                    marker="*" if ci and (ci["lower"]>0 or ci["upper"]<0) else ""
                    ax.text(j,i,f"{matrix[i,j]:+.2f}{marker}",ha="center",va="center",fontsize=7)
    fig.colorbar(image,ax=np.atleast_1d(axes).tolist(),label="paired outer-seed mean difference (percentage points); * 95% CI excludes zero")
    fig.savefig(args.out_dir/f"{args.protocol}_{args.metric}_differences.pdf");plt.close(fig)


def curves(payload, args):
    rows=[r for r in payload.get("layer_results",[]) if r["protocol"] == args.protocol]
    if not rows:
        raise ValueError("no layer_results retained in summary; scalar reported values cannot reconstruct curves")
    seeds=payload["requested_outer_seeds"]
    models,datasets=sorted({r["model"] for r in rows}),sorted({r["dataset"] for r in rows})
    fig,axes=plt.subplots(len(models),len(datasets),figsize=(4*len(datasets),2.8*len(models)),squeeze=False,constrained_layout=True)
    drew=False
    for i,model in enumerate(models):
        for j,dataset in enumerate(datasets):
            ax=axes[i,j]
            for method in LABELS:
                selected=[r for r in rows if (r["model"],r["dataset"],r["method"]) == (model,dataset,method)]
                if not selected:
                    continue
                by_seed={r["seed"]:r for r in selected}
                if len(by_seed)!=len(selected) or set(by_seed)!=set(seeds):
                    raise ValueError("layer curve requires exactly the requested outer seeds")
                layers=by_seed[seeds[0]]["layer_indices"]
                if any(r["layer_indices"]!=layers for r in selected):
                    raise ValueError("layer curves have inconsistent evaluated layer sets")
                values=np.asarray([by_seed[s]["test_accuracy"] for s in seeds])
                mean=values.mean(axis=0);sd=values.std(axis=0,ddof=1) if len(seeds)>1 else np.zeros_like(mean)
                ax.plot(layers,mean,label=LABELS[method],color=COLORS[method]);ax.fill_between(layers,mean-sd,mean+sd,alpha=.13,color=COLORS[method])
                drew=True
            ax.set_title(model.rsplit("/",1)[-1]+" / "+dataset,fontsize=9)
            ax.set_xlabel("stored layer index");ax.set_ylabel("test accuracy (mean ± sample SD)");ax.set_ylim(0,1)
    if not drew:
        raise ValueError("no complete layer curves")
    handles,labels=next(ax.get_legend_handles_labels() for row in axes for ax in row if ax.get_legend_handles_labels()[0])
    fig.legend(handles,labels,loc="outside lower center",ncol=len(labels))
    fig.savefig(args.out_dir/f"{args.protocol}_layerwise_accuracy.pdf");plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",type=Path,default=Path("exp_results/accuracy_multiseed/summary.json"))
    parser.add_argument("--protocol",choices=["rowwise","grouped_split_sensitivity"],default="rowwise")
    parser.add_argument("--metric",choices=["best_layer_test_accuracy","average_layer_test_accuracy"],default="best_layer_test_accuracy")
    parser.add_argument("--view",choices=["heatmap","differences","curves","all"],default="heatmap")
    parser.add_argument("--out-dir",type=Path,default=Path("plots/accuracy_multiseed"))
    args=parser.parse_args();payload=json.loads(args.summary.read_text());args.out_dir.mkdir(parents=True,exist_ok=True)
    for view, function in (("heatmap",heatmaps),("differences",differences),("curves",curves)):
        if args.view in (view,"all"):
            function(payload,args)

if __name__ == "__main__":
    main()
