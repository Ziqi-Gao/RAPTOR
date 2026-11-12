"""Prepare fresh fixed trial IDs using the retained native paper templates.

Historical frozen trial IDs were not recovered. Native prompt builder bodies
remain in generate.py; this command changes only transport and provenance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .controls import atomic_json, file_hash


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=['STSA','sarcasm','hatexplain','counterfact'])
    p.add_argument('--direction',required=True)
    p.add_argument('--data',type=Path)
    p.add_argument('--count',type=int,default=64)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    from .generate import (get_dataset_template, load_dataset_samples,
                           build_dataset_prompt_specs, build_fixed_total_prompt_specs)
    template=get_dataset_template(a.dataset)
    if a.direction not in template['steering_configs']: raise ValueError('unsupported task direction')
    if a.count<=0:raise ValueError('count must be positive')
    instruction=template['steering_configs'][a.direction].get('instruction',template['instruction'])
    if '{text}' in instruction:
        if a.data is None:raise ValueError('this task uses source text; provide --data')
        pos,neg=load_dataset_samples(a.dataset,a.data,(a.count+1)//2,a.count//2,seed=a.seed)
        sentences=(pos+neg)[:a.count]
        if len(sentences)!=a.count:raise ValueError('insufficient source sentences')
        specs=build_dataset_prompt_specs(sentences,[a.direction],a.dataset)
    else:
        specs=build_fixed_total_prompt_specs(a.count,[a.direction],a.dataset)
    if a.out.exists():raise ValueError('refusing to overwrite frozen prompt IDs')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(''.join(json.dumps(dict(prompt_id=i,text=spec.prompt),ensure_ascii=False)+'\n' for i,spec in enumerate(specs)))
    atomic_json(a.out.with_suffix(a.out.suffix+'.meta.json'),dict(dataset=a.dataset,direction=a.direction,count=a.count,seed=a.seed,
        prompt_sha256=file_hash(a.out),source_sha256=file_hash(a.data) if a.data else None,
        historical_prompt_identity_verified=False,template_source_commit='68faa5f84ce7b6a09c4f18fdd55b45156b219a1d'))


if __name__=='__main__':main()
