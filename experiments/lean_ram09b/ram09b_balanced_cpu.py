"""RAM-09B fixed-protocol family-balanced retrospective Lean policy study.

Original RAM-08 checkpoint tensors and 20,930 independently verified old
MathCan compiler effects are immutable. The old 15-root holdback is reused,
so this is exploratory (not PEF theorem transfer).
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
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
# Repository layout: experiments/lean_ram09b/ram09b_balanced_cpu.py
# imports audited graph + model code from sibling RAM-09 stage0 directory.
RAM09=HERE if (HERE/'offline_graph.py').exists() else HERE.parent/'lean_ram09'
sys.path.insert(0,str(RAM09))
from offline_graph import load_archive, source_splits, ObservedClosureGraph, key, ARCHIVE_SHA
from offline_fit import originals, split_internal, score, encode
from guarded_probe import embedded

ALPHA=0.35
STEPS=200
BATCH=32
SEED=20261115
LR=0.003
REG=.02


def precompute(model, rows):
    """Keep all original model output weights frozen, including embeddings."""
    model.eval()
    f=[];b=[]
    with torch.no_grad():
        for offset in range(0,len(rows),64):
            encoded=encode(rows[offset:offset+64])
            f.append(embedded(model,encoded).detach().cpu())
            b.append(score(model,encoded).detach().cpu())
    return torch.cat(f), torch.cat(b)


def eligible_examples(graph, base):
    """Build a state-level, family-stratifiable clean loss sample.

    A selected positive must have a bounded original compiler-observed route
    to closure. All negative scripts have original label0. Each family keeps
    its hardest negative under *frozen base* score, using stable row order.
    """
    assert len(base)==len(graph.rows)
    grouped=defaultdict(list)
    for i,row in enumerate(graph.rows):grouped[key(row)].append(i)
    examples=[]
    for state,indices in sorted(grouped.items()):
        good=[];negative=defaultdict(list)
        for i in indices:
            row=graph.rows[i]
            if row['label']==0:
                negative[row['family']].append(i)
            else:
                distance=graph.action_distance(row)
                if distance is not None and 1<=distance<=4:
                    good.append((distance,i))
        if not good or not negative:continue
        # Prevent target family frequency from following candidate multiplicity.
        shortest=min(x[0] for x in good)
        pos=max((i for d,i in good if d==shortest),key=lambda i:(float(base[i]),-i))
        chosen={f:max(ids,key=lambda i:(float(base[i]),-i)) for f,ids in negative.items()}
        assert graph.action_distance(graph.rows[pos])==shortest
        assert all(graph.rows[idx]['label']==0 for idx in chosen.values())
        examples.append(dict(state=state,pos=pos,pos_family=graph.rows[pos]['family'],
                             distance=shortest,hard_failed_by_family=chosen))
    return examples


def sample_pairs(examples,rng,count):
    """Uniform positive family, uniform state in family, uniform failed family."""
    by_family=defaultdict(list)
    for item in examples:by_family[item['pos_family']].append(item)
    families=sorted(by_family)
    if not families:raise AssertionError('no verified observed closure vs compiler-failed examples')
    pos=[];neg=[];diag=Counter()
    for _ in range(count):
        family=families[rng.randrange(len(families))]
        item=by_family[family][rng.randrange(len(by_family[family]))]
        failure_families=sorted(item['hard_failed_by_family'])
        rejected_family=failure_families[rng.randrange(len(failure_families))]
        pos.append(item['pos']);neg.append(item['hard_failed_by_family'][rejected_family])
        diag['pos/'+family]+=1;diag['neg/'+rejected_family]+=1
        if family==rejected_family:diag['within_family']+=1
    return torch.tensor(pos,dtype=torch.long),torch.tensor(neg,dtype=torch.long),diag


def state_metrics(graph, base, altered):
    """Single-action offline scoreboard, with witnessed-path label censoring."""
    if len(base)!=len(graph.rows) or len(altered)!=len(graph.rows):
        raise AssertionError('rank scoreboard dimension mismatch')
    grouped=defaultdict(list)
    for i,row in enumerate(graph.rows):grouped[key(row)].append(i)
    metrics={}
    for label,scores in [('original',base),('balanced_guard',altered)]:
        c=Counter();chosen=Counter()
        for state,idxs in grouped.items():
            winner=max(idxs,key=lambda j:(float(scores[j]),-j))
            row=graph.rows[winner]
            distance=graph.action_distance(row)
            c['states']+=1
            c['top1_useful']+=row['label']>0
            c['top1_closes']+=row['label']==3
            c['top1_observed_witness']+=distance is not None and distance<=4
            if state in graph.distance:
                c['states_with_observed_witness']+=1
                c['winner_on_witness_state']+=distance is not None and distance<=4
                c['winner_shortest_when_witnessed']+=distance==graph.distance[state]
            chosen[row['family']]+=1
        metrics[label]=dict(**c,chosen_families=dict(sorted(chosen.items())))
    min_margin=2*ALPHA
    guaranteed=0;exact=0
    for idxs in grouped.values():
        base_winner=max(idxs,key=lambda j:(float(base[j]),-j))
        competitor=max((float(base[j]) for j in idxs if j!=base_winner),default=float('-inf'))
        if float(base[base_winner])-competitor>min_margin:
            guaranteed+=1
            new_winner=max(idxs,key=lambda j:(float(altered[j]),-j))
            if base_winner==new_winner:exact+=1
            else:raise AssertionError('RAM09B rank-margin invariance violated')
    metrics['margin_certificate']=dict(alpha=ALPHA,margin=2*ALPHA,
             strict_high_margin_states=guaranteed,preserved=exact)
    return metrics


def run(archive,out):
    if out.exists():raise ValueError('Output already exists. Preserve any earlier trial.')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rows,audit=load_archive(archive)
    train,old_dev=source_splits(rows)
    fit,hold=split_internal(train)
    fit_graph=ObservedClosureGraph(fit)
    hold_graph=ObservedClosureGraph(hold)
    dev_graph=ObservedClosureGraph(old_dev)
    if len(fit_graph.theorems)!=42 or len(hold_graph.theorems)!=15 or len(dev_graph.theorems)!=5:
        raise AssertionError('RAM09 frozen root split is altered')
    if set(fit_graph.theorems)&(set(hold_graph.theorems)|set(dev_graph.theorems)):
        raise AssertionError('lookahead source-root leakage')
    # Commit frozen registration file independently before fitting.
    protocol=HERE/'PROTOCOL.md'
    if not protocol.is_file():
        # Can run standalone from replicated RAM09 source: explicit opt-in in local sandbox.
        raise FileNotFoundError('RAM09B PROTOCOL.md missing from source directory')
    report=dict(campaign='RAM09B_exploratory_balanced_failures',status='running',
        original_archive_sha256=ARCHIVE_SHA,original_compiler_effects=audit['checked'],
        original_source_only='c84e2025334e9a967c1297425a490f98d5b3d189',
        prospective_pef_theorems_read=0,prospective_pef_case_scores=0,
        original_mathcan_dev_previously_used_for_selection=True,
        original_ram09_holdback_previously_inspected=True,
        optimizer=dict(steps=STEPS,batch=BATCH,lr=LR,reg=REG,alpha=ALPHA,seed=SEED),
        source_roots=dict(fit=42,internal_holdback=15,original_development=5),
        models={})
    out.mkdir(parents=True,exist_ok=False)
    try:
        for name,model,base_sha in originals(archive):
            model.eval()
            for p in model.parameters():p.requires_grad_(False)
            features,base=precompute(model,fit_graph.rows)
            examples=eligible_examples(fit_graph,base)
            if len(examples)<50:raise AssertionError('too few eligible old source states')
            positive_fams=Counter(x['pos_family'] for x in examples)
            neg_fams=Counter(f for x in examples for f in x['hard_failed_by_family'])
            fit_head=nn.Linear(model.config.width,1)
            nn.init.zeros_(fit_head.weight);nn.init.zeros_(fit_head.bias)
            opt=torch.optim.AdamW(fit_head.parameters(),lr=LR,weight_decay=.01)
            rng=random.Random(SEED)
            snapshots=[];sampled=Counter();began=time.perf_counter()
            for step in range(1,STEPS+1):
                pos,neg,diagnostic=sample_pairs(examples,rng,BATCH)
                sampled.update(diagnostic)
                pos_corr=torch.tanh(fit_head(features[pos]).squeeze(-1))
                neg_corr=torch.tanh(fit_head(features[neg]).squeeze(-1))
                dif=(base[pos]-base[neg])+ALPHA*(pos_corr-neg_corr)
                loss=F.softplus(-dif).mean()+REG*(pos_corr.square().mean()+neg_corr.square().mean())
                if not torch.isfinite(loss):raise FloatingPointError('RAM09B nonfinite loss')
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                if step in (1,50,100,150,200):snapshots.append([step,round(float(loss.detach()),6)])
            split_metrics={}
            with torch.no_grad():
                for label,g in [('internal_holdback',hold_graph),('old_development',dev_graph)]:
                    feats,unmodified=precompute(model,g.rows)
                    adapted=unmodified+ALPHA*torch.tanh(fit_head(feats).squeeze(-1))
                    if torch.max(torch.abs(adapted-unmodified)).item()>ALPHA+1e-5:
                        raise AssertionError('RAM09B rank perturbation bound violated')
                    split_metrics[label]=state_metrics(g,unmodified,adapted)
            checkpoint=out/(name+'_head.pt')
            torch.save(dict(schema='RAM09B-posthoc-20261115',name=name,base_sha256=base_sha,
                            head_state_dict=fit_head.state_dict(),alpha=ALPHA),checkpoint)
            report['models'][name]=dict(base_checkpoint_sha256=base_sha,
                 head_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 fit_seconds=round(time.perf_counter()-began,3),
                 eligible_fit_states=len(examples),eligible_fit_positive_families=dict(sorted(positive_fams.items())),
                 eligible_fit_failure_families=dict(sorted(neg_fams.items())),
                 actual_sampled_families=dict(sorted(sampled.items())),
                 loss=snapshots,split_metrics=split_metrics)
            print('RAM09B_FINISHED_MODEL',name,json.dumps({
                 'before':split_metrics['internal_holdback']['original']['winner_on_witness_state'],
                 'after':split_metrics['internal_holdback']['balanced_guard']['winner_on_witness_state'],
                 'positive_families':len(positive_fams),
                 'negative_families':len(neg_fams)},sort_keys=True),flush=True)
        report['status']='complete'
        before=[report['models'][name]['split_metrics']['internal_holdback']['original']['winner_on_witness_state'] for name in ('loop32','loop64','flat64')]
        after=[report['models'][name]['split_metrics']['internal_holdback']['balanced_guard']['winner_on_witness_state'] for name in ('loop32','loop64','flat64')]
        report['descriptive_target_met']=sum(a>b for a,b in zip(after,before))>=2 and min((a-b for a,b in zip(after,before)))>=-1
    except BaseException as exc:
        report['status']='aborted';report['error']=repr(exc)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        raise
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('RAM09B_COMPLETE',json.dumps({'descriptive_target_met':report['descriptive_target_met'],'before':before,'after':after},sort_keys=True))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    run(a.archive,a.out)