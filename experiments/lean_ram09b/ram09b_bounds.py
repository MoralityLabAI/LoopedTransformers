"""Certified rank-feasibility intervals for bounded tactic-score corrections.

A state with an observed closing path is a censored graph diagnostic, NOT a
new Lean theorem certificate. This is a post-score explanatory analysis.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
RAM09=HERE if (HERE/'offline_graph.py').exists() else HERE.parent/'lean_ram09'
sys.path.insert(0,str(RAM09))

import torch
from offline_graph import load_archive,source_splits,ObservedClosureGraph,key
from offline_fit import originals,split_internal
from ram09b_balanced_cpu import precompute,ALPHA


def witness_feasibility(graph,score,alpha):
    if len(graph.rows)!=len(score):raise AssertionError('one score per compiler effect required')
    groups=defaultdict(list)
    for i,r in enumerate(graph.rows):groups[key(r)].append(i)
    out=Counter();blocked=Counter();errors=[];near=[];protected=Counter()
    for state,ids in groups.items():
        if state not in graph.distance:continue
        w=[i for i in ids if graph.action_distance(graph.rows[i]) is not None and graph.action_distance(graph.rows[i])<=4]
        if not w:raise AssertionError('closure reachability graph disagrees with state effects')
        n=[i for i in ids if i not in set(w)]
        top_w=max(w,key=lambda i:(float(score[i]),-i))
        max_w=float(score[top_w]);top_n=max(n,key=lambda i:(float(score[i]),-i)) if n else None
        max_n=float(score[top_n]) if top_n is not None else float('-inf')
        original=max(ids,key=lambda i:(float(score[i]),-i))
        is_witness=original in w
        delta=max_n-max_w
        out['witness_states']+=1
        out['original_winner_is_witness']+=is_witness
        out['necessarily_wrong_under_any_bounded_delta']+=delta>=2*alpha
        out['possibly_witness_under_bounded_delta']+=delta<2*alpha
        out['guaranteed_witness_under_every_bounded_delta']+=delta< -2*alpha
        out['nonwitness_winner_and_legal_flip']+=not is_witness and delta<2*alpha
        out['nonwitness_winner_and_unflippable']+=not is_witness and delta>=2*alpha
        if not is_witness:
            c=graph.rows[top_n]['family']
            blocked[c]+=int(delta>=2*alpha)
            near.append(dict(state_sha256=state[2],winner_family=c,
                strongest_witness_family=graph.rows[top_w]['family'],
                winner_script=graph.rows[top_n]['script'],
                witness_script=graph.rows[top_w]['script'],
                score_gap=round(delta,5),unflippable=delta>=2*alpha))
        if is_witness and delta< -2*alpha:
            protected[graph.rows[top_w]['family']]+=1
    if out['necessarily_wrong_under_any_bounded_delta']+out['possibly_witness_under_bounded_delta']!=out['witness_states']:
        raise AssertionError('upper bound must partition witnessed states')
    out['maximum_witness_states_under_independent_bounded_corrections']=out['possibly_witness_under_bounded_delta']
    out['potential_gain_above_original']=out['maximum_witness_states_under_independent_bounded_corrections']-out['original_winner_is_witness']
    return dict(counts=dict(out),blocked_winner_families=dict(sorted(blocked.items())),
                robust_witness_families=dict(sorted(protected.items())),
                wrong_winner_gap_examples=sorted(near,key=lambda x:-x['score_gap'])[:8])


def run(archive,out):
    if out.exists():raise ValueError('output already exists, preserve previous analysis')
    torch.set_num_threads(2)
    rows,_=load_archive(archive)
    train,dev=source_splits(rows)
    fit,hold=split_internal(train)
    results=dict(campaign='RAM09B_postscore_rank_feasibility_audit',
       role='explanatory',original_PEF_theorems_read=0,
       bound='s_max_nonwitness-s_max_witness >= 2alpha implies no bounded perturbation can select an observed witness',
       alpha=ALPHA,
       caveat='Upper bound allows independent per-candidate score perturbations, more power than a shared learned linear head; original MathCan training data already exposed these roots to RAM08',
       models={})
    for name,model,sha in originals(archive):
        results['models'][name]={}
        for split,subset in [('internal_holdback',hold),('old_development',dev)]:
            graph=ObservedClosureGraph(subset)
            _,score=precompute(model,graph.rows)
            report=witness_feasibility(graph,score,ALPHA)
            results['models'][name][split]=report
            print('RAM09B_BOUND',name,split,json.dumps(report['counts'],sort_keys=True),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2)+'\n')
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.archive,a.out)