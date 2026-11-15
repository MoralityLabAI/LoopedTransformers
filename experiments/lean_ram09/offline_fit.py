"""RAM-09 retrospective CPU pilot: proof-closure lookahead pairwise fine-tuning.

Uses unchanged frozen RAM-08 weights and *original TRAIN-only* replay labels.
Internal holdback is theorem-root disjoint; the original old DEV is only a
post-fit diagnostic. New PEF theorems and compiler goal states are never read.
All unknown successor effects are censored, not relabeled failure.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time
import zipfile

import torch
import torch.nn.functional as F
from offline_graph import ObservedClosureGraph, load_archive, source_splits, key, ARCHIVE_SHA
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'lean_ram08'))
from model import Policy, Config, encode

BASE_MODELS={
  'loop32':(29,'4a88bc80456743a89a70882911060cc90e76785a5ae1d435a0e46a8f8c09778f'),
  'loop64':(43,'5cbd327904714ac698897e1c2333cae9ec4125aeb70e7c56cd6236db8a737dac'),
  'flat64':(29,'2aa101edbd56f6563ebb2edcfc8c61c07dcc027559ea4cdc551187d6ec5303d8'),
}
SPLIT_SALT='RAM09-GRAPH-20261115/'

def is_holdback(root: tuple[str,str]):
    return int(hashlib.sha256((SPLIT_SALT+':'.join(root)).encode()).hexdigest(),16)%5==0

def split_internal(original_train):
    fit, hold = [], []
    for r in original_train:
        (hold if is_holdback((r['source_file'],r['theorem'])) else fit).append(r)
    if {r['state_sha256'] for r in fit}&{r['state_sha256'] for r in hold}:
        raise AssertionError('inner train and holdback have overlapping states')
    if {(r['source_file'],r['theorem']) for r in fit}&{(r['source_file'],r['theorem']) for r in hold}:
        raise AssertionError('inner split has overlapping theorem roots')
    return fit,hold

def originals(path):
    with zipfile.ZipFile(path) as z:
        for name,(seed,sha) in BASE_MODELS.items():
            blob=z.read(f'training/{name}_s{seed}/best.pt')
            if hashlib.sha256(blob).hexdigest()!=sha:raise AssertionError('RAM08 frozen checkpoint hash')
            data=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
            if data['schema']!='RAM08-v1' or data['seed']!=seed or data['model_config']['name']!=name:
                raise AssertionError('RAM08 checkpoint model/schema/seed mismatch')
            model=Policy(Config(**data['model_config'])).eval()
            model.load_state_dict(data['state_dict'])
            yield name, model, sha

def pairs_from_graph(g):
    """Pair one finite closure witness with a verified failed candidate.

    Unobserved successors and previously valid but uncompleted paths never
    become negatives. This is supervised with closed-world evidence only.
    """
    pairs=[]
    for state,rows in sorted(g.by_state.items()):
        positive=[(g.action_distance(row),row) for row in rows]
        positive=[(d,row) for d,row in positive if d is not None and d<=4]
        negative=[row for row in rows if row['label']==0]
        if not positive or not negative:continue
        smallest=min(d for d,_ in positive)
        good=min((row for d,row in positive if d==smallest),key=lambda r:r['script'])
        # Four distinct independent original compiler-rejected actions.
        negatives=sorted(negative,key=lambda r:r['script'])[:4]
        for rejected in negatives:pairs.append((good,rejected,smallest))
    return pairs

def score(model, batch):
    logits,delta=model(batch)
    p=logits.softmax(-1)
    return 3*p[:,3]+1.5*p[:,1]+.3*p[:,2]-.6*p[:,0]+.4*delta.clamp(-3,3)

@torch.no_grad()
def eval_states(model,graph):
    model.eval()
    all_rows=graph.rows
    predictions=[]
    for start in range(0,len(all_rows),64):
        predictions.extend(score(model,encode(all_rows[start:start+64])).tolist())
    by_state=defaultdict(list)
    for row,v in zip(all_rows,predictions):by_state[key(row)].append((row,v))
    metrics=Counter()
    for state,items in by_state.items():
        metrics['states']+=1
        selected=max(items,key=lambda rv:rv[1])[0]
        metrics['top1_useful']+=int(selected['label']>0)
        metrics['top1_closed']+=int(selected['label']==3)
        d=graph.action_distance(selected)
        metrics['top1_reaches_observed_closure']+=int(d is not None and d<=4)
        if state in graph.distance:
            metrics['witness_states']+=1
            metrics['top1_witness_given_available']+=int(d is not None and d<=4)
            metrics['top1_shortest_witness_given_available']+=int(d==graph.distance[state])
    return dict(metrics)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--seed',type=int,default=20261115)
    a=parser.parse_args()
    if a.out.exists():parser.error('output directory already exists; preserve old independent run')
    if (a.steps,a.batch)!=(300,32):parser.error('stage0 original design fixes 300 updates and 32 pair batch')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    random.seed(a.seed);torch.manual_seed(a.seed)
    rows,_=load_archive(a.archive)
    train,original_dev=source_splits(rows)
    fit,hold=split_internal(train)
    gfit=ObservedClosureGraph(fit);ghold=ObservedClosureGraph(hold)
    pairs=pairs_from_graph(gfit)
    if len(pairs)<100:raise AssertionError('insufficient independent original witness-versus-fail pairs')
    a.out.mkdir(parents=True,exist_ok=False)
    report=dict(campaign='RAM09_postfreeze_retrospective_pairwise_lookahead_pilot',
      source_archive_sha256=ARCHIVE_SHA,
      original_train_theorems=len(gfit.theorems),internal_holdback_theorems=len(ghold.theorems),
      train_states_with_observed_witness=len(gfit.distance),
      holdback_states_with_observed_witness=len(ghold.distance),
      train_pairs=len(pairs),original_dev_was_used_in_ram08_selection=True,
      original_pef_cases_scored=0,step_count=a.steps,batch_pairs=a.batch,seed=a.seed,
      gradient_rule='softplus(failed_action_score - best_observed_closure_witness_score)',
      invalid_action_labels='original compiler failure (0) only',
      missing_successor_labels='censored', models={})
    with torch.no_grad():
        pass
    for name,model,sha in originals(a.archive):
        torch.manual_seed(a.seed)
        before=dict(train_holdback=eval_states(model,ghold),old_development=eval_states(model,ObservedClosureGraph(original_dev)))
        # Evaluate the original small network's *same* ranker on a fresh copy.
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
        rng=random.Random(a.seed)
        losses=[];t0=time.perf_counter()
        for step in range(a.steps):
            batch=[pairs[rng.randrange(len(pairs))] for _ in range(a.batch)]
            pos=[p[0] for p in batch]; neg=[p[1] for p in batch]
            model.train();optimizer.zero_grad(set_to_none=True)
            pos_score=score(model,encode(pos))
            neg_score=score(model,encode(neg))
            loss=F.softplus(neg_score-pos_score).mean()
            if not torch.isfinite(loss):raise FloatingPointError('invalid RAM09 pairwise loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if (step+1)%50==0:losses.append([step+1,round(float(loss.detach()),6)])
        after=dict(train_holdback=eval_states(model,ghold),old_development=eval_states(model,ObservedClosureGraph(original_dev)))
        target=a.out/(name+'_lookahead.pt')
        # Immutable original optimizer and weights are NOT written back.
        torch.save(dict(schema='RAM09-stage0-retrospective-v1',name=name,seed=a.seed,
            base_checkpoint_sha256=sha,steps=a.steps,state_dict=model.state_dict(),
            original_train_source='c84e2025334e9a967c1297425a490f98d5b3d189'),target)
        report['models'][name]=dict(base_checkpoint_sha256=sha,training_seconds=round(time.perf_counter()-t0,3),
          weights_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
          training_loss_snapshots=losses,before=before,after=after)
        print('RAM09_CPU_FIT',name,json.dumps(report['models'][name],sort_keys=True),flush=True)
    (a.out/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('RAM09_STAGE0_COMPLETE',json.dumps({k:report[k] for k in ['train_pairs','train_states_with_observed_witness','holdback_states_with_observed_witness','original_pef_cases_scored']},sort_keys=True))

if __name__=='__main__': main()
