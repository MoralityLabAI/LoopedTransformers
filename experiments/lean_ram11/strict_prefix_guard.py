"""RAM-11P posthoc corrected-label CPU residual-head experiment.

Repeats the RAM-09B family-balanced frozen-trunk head fit using ONLY
original observed exact-prefix closing routes as positive. Earlier quotient
witnesses whose composite script never appeared in the collector are censored.
The split has already been inspected, so this is exploratory and cannot prove
RAM-08 or RAM-10 theorem generalization.
"""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'lean_ram09'))
sys.path.insert(0,str(HERE.parent/'lean_ram09b'))
from offline_graph import load_archive,source_splits
from offline_fit import originals,split_internal
from ram09b_balanced_cpu import precompute, sample_pairs
from prefix_audit import PrefixGraph,prefix_key

ALPHA=.35
STEPS=200
BATCH=32
LR=.003
REG=.02
SEED=20261115


def clean_examples(graph,base):
    by=defaultdict(list)
    for i,r in enumerate(graph.rows):by[prefix_key(r)].append(i)
    examples=[]
    for node,ids in sorted(by.items()):
        positives=[];negative=defaultdict(list)
        for idx in ids:
            row=graph.rows[idx]
            if row['label']==0:negative[row['family']].append(idx)
            else:
                distance=graph.action_distance(row)
                if distance is not None and distance<=4:positives.append((distance,idx))
        if not positives or not negative:continue
        distance=min(d for d,_ in positives)
        pos=max((i for d,i in positives if d==distance),key=lambda i:(float(base[i]),-i))
        failed={f:max(v,key=lambda i:(float(base[i]),-i)) for f,v in negative.items()}
        examples.append(dict(state=node,pos=pos,pos_family=graph.rows[pos]['family'],distance=distance,hard_failed_by_family=failed))
    return examples


def metrics(graph,original,adapted):
    groups=defaultdict(list)
    for i,r in enumerate(graph.rows):groups[prefix_key(r)].append(i)
    out={}
    for name,scores in [('original',original),('corrected_prefix_guard',adapted)]:
        m=Counter()
        for node,ids in groups.items():
            index=max(ids,key=lambda i:(float(scores[i]),-i))
            row=graph.rows[index]
            m['states']+=1
            m['top1_useful']+=int(row['label']>0)
            m['top1_closes_goals']+=int(row['label']==3)
            d=graph.action_distance(row)
            m['top1_exact_prefix_witness']+=int(d is not None)
            if node in graph.distance:
                m['states_with_exact_prefix_witness']+=1
                m['winner_has_prefix_witness']+=int(d is not None)
                m['winner_uses_shortest_prefix_witness']+=int(d==graph.distance[node])
        out[name]=dict(m)
    guaranteed=0
    for indices in groups.values():
        best=max(indices,key=lambda i:(float(original[i]),-i))
        second=max((float(original[j]) for j in indices if j!=best),default=float('-inf'))
        if float(original[best])-second>2*ALPHA:
            guaranteed+=1
            recomputed=max(indices,key=lambda i:(float(adapted[i]),-i))
            if recomputed!=best:raise AssertionError('Theorem: margin > 2 alpha must preserve winning action')
    out['rank_certification']=dict(alpha=ALPHA,margin=2*ALPHA,verified_preserved_state_count=guaranteed)
    return out


def run(archive:Path,out:Path)->dict:
    if out.exists():raise FileExistsError('Existing output: preserve previous original pilot result')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rows,audit=load_archive(archive)
    train,dev=source_splits(rows)
    fit,hold=split_internal(train)
    splits={'fit':PrefixGraph(fit),'internal_holdback':PrefixGraph(hold),'old_ram08_dev':PrefixGraph(dev)}
    if [len(splits[x].root_sha) for x in ('fit','internal_holdback','old_ram08_dev')]!=[42,15,5]:
        raise AssertionError('RAM09 original root split changed')
    receipt=dict(campaign='RAM11P_posthoc_exact_prefix_censored_frozen_head',
        description='RAM09B original sampler and optimizer, corrected prefix-only positive label',
        historical_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        original_compiler_observations=audit['checked'],
        original_pef_proof_cases_scored=0,complete_new_lean_theorem_certificates=0,
        already_inspected_internal_holdback=True,already_used_dev_for_RAM08_checkpoint_selection=True,
        source_trust='historical original independent Lean effect replays; no exact full theorem source audit',
        split_roots={'fit':42,'holdback':15,'old_dev':5},
        optimizer=dict(alpha=ALPHA,steps=STEPS,batch=BATCH,lr=LR,regularizer=REG,seed=SEED),
        models={},status='running')
    out.mkdir(parents=True,exist_ok=False)
    try:
        for name,model,sha in originals(archive):
            for p in model.parameters():p.requires_grad_(False)
            feats,base=precompute(model,splits['fit'].rows)
            examples=clean_examples(splits['fit'],base)
            if len(examples)<30:raise ValueError('too few clean exact-prefix states for training')
            torch.manual_seed(SEED)
            head=nn.Linear(model.config.width,1)
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
            opt=torch.optim.AdamW(head.parameters(),lr=LR,weight_decay=.01)
            rng=random.Random(SEED);snapshots=[];sampled=Counter();start=time.perf_counter()
            for step in range(1,STEPS+1):
                positive,negative,stats=sample_pairs(examples,rng,BATCH)
                sampled.update(stats)
                pcor=torch.tanh(head(feats[positive]).squeeze(-1))
                ncor=torch.tanh(head(feats[negative]).squeeze(-1))
                delta=(base[positive]-base[negative])+ALPHA*(pcor-ncor)
                loss=F.softplus(-delta).mean()+REG*(pcor.square().mean()+ncor.square().mean())
                if not torch.isfinite(loss):raise FloatingPointError('RAM11P loss invalid')
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                if step in (1,50,100,150,200):snapshots.append([step,round(float(loss.detach()),6)])
            evaluated={}
            with torch.no_grad():
                for label in ['internal_holdback','old_ram08_dev']:
                    g=splits[label]
                    z,b=precompute(model,g.rows)
                    new=b+ALPHA*torch.tanh(head(z).squeeze(-1))
                    if torch.max(torch.abs(new-b)).item()>ALPHA+1e-5:raise AssertionError('rank score bound exceeded')
                    evaluated[label]=metrics(g,b,new)
            path=out/(name+'_prefix_head.pt')
            torch.save(dict(schema='RAM11P-posthoc-v1',name=name,original_base_sha=sha,alpha=ALPHA,head=head.state_dict()),path)
            receipt['models'][name]=dict(original_sha=sha,head_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                fit_seconds=round(time.perf_counter()-start,3),eligible_fit_states=len(examples),
                positive_families=dict(sorted(Counter(e['pos_family'] for e in examples).items())),
                sampled_families=dict(sorted(sampled.items())),loss_snapshots=snapshots,scores=evaluated)
            b=evaluated['internal_holdback']['original']['winner_has_prefix_witness']
            v=evaluated['internal_holdback']['corrected_prefix_guard']['winner_has_prefix_witness']
            print('RAM11P_FROZEN_CPU',name,b,v, 'out_of',evaluated['internal_holdback']['original']['states_with_exact_prefix_witness'],flush=True)
        receipt['status']='complete'
    except Exception as e:
        receipt['status']='aborted';receipt['error']=repr(e)
        (out/'results.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
        raise
    (out/'results.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.archive,a.out)
