#!/usr/bin/env python3
"""Plot complete native-coordinate robustness setting Mean/Max distributions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("plots/robustness_release.pdf"))
    args=parser.parse_args();payload=json.loads(args.summary.read_text())
    if payload["status"] not in {"complete","complete_with_nonconvergence"}:
        raise ValueError("cannot plot a pruned or incomplete robustness grid")
    labels={"singlelr":"RAPTOR","xrfm":"xRFM","gcs":"GCS","fixed_reg":"Fixed-reg","caa":"CAA"}
    methods=[m for m in labels if any(r["method"] == m for r in payload["settings"])]
    fig,axes=plt.subplots(2,1,figsize=(8,7),constrained_layout=True)
    for ax,metric in zip(axes,("Max","Mean")):
        for position,method in enumerate(methods):
            rows=[r for r in payload["settings"] if r["method"] == method]
            if any(r["status"] != "complete" or r[metric] is None for r in rows):
                raise ValueError("robustness settings must all be complete")
            values=np.array([r[metric] for r in rows])
            offsets=np.linspace(-.12,.12,len(values))
            ax.scatter(np.full(len(values),position)+offsets,values,s=18,alpha=.6)
            ax.scatter(position,np.median(values),marker="D",s=70,color="black",zorder=4)
        ax.set_xticks(range(len(methods)),[labels[m] for m in methods]);ax.set_ylim(0,1.02)
        ax.set_ylabel(metric+" absolute pairwise cosine");ax.grid(axis="y",alpha=.2)
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output);plt.close(fig)

if __name__ == "__main__":
    main()
