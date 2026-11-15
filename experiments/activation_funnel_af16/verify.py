"""Independent AF-16 host-proof audit from encoded incidence, with source-linked replay.

This verifies the public edge relation directly from tensors. It uses an
independent reachability/BFS oracle rather than the training shortest-path DP.
"""
from __future__ import annotations
from collections import deque
from functools import lru_cache
import argparse,json
from pathlib import Path
import torch
from experiments.activation_funnel_af05 import gym as base
from . import gym,experiment as ex


def public_graph(case):
    graph=gym.encode(case)
    incidence=graph.prem_up+graph.prem_down;heads=graph.head_up+graph.head_down
    rules={}
    for i in range(base.RULES):
        if float(graph.rule_features[i,2])<.5:continue
        ps=tuple(int(j) for j in torch.where(incidence[i]>.5)[0].tolist())
        dest=tuple(int(j) for j in torch.where(heads[i]>.5)[0].tolist())
        if len(dest)!=1 or not ps or len(ps)>2:raise ValueError('invalid AF16 public Horn incidence')
        rules[i]=(ps,dest[0])
    init=frozenset(int(j) for j in torch.where(graph.atom_features[:,0]>.5)[0].tolist())
    goal=int(graph.read_slots[2]);return rules,init,goal

@lru_cache(maxsize=4000)
def independent_reference(case):
    rules,init,goal=public_graph(case)
    # Independent, breadth-first shortest-goal search; unseen cases use only public physical incidence.
    q=deque([(init,0)]);seen={init};distance=None
    while q:
        known,t=q.popleft()
        if goal in known:distance=t;break
        for ps,h in rules.values():
            if h not in known and set(ps)<=known:
                nxt=known|{h}
                if nxt not in seen:seen.add(nxt);q.append((nxt,t+1))
    closure=set(init)
    for _ in range(base.ATOMS+1):
        new=closure|{h for ps,h in rules.values() if set(ps)<=closure}
        if new==closure:break
        closure=new
    else:raise RuntimeError('Horn closure did not stabilize')
    if (goal in closure)!=(distance is not None):raise ValueError('independent shortest/closure mismatch')
    return rules,init,goal,distance,closure


def verify_row(case,row):
    if (row['case_id'],row['split'])!=(case.id,case.split):raise ValueError('case identity changed')
    edges,init,goal,dist,closure=independent_reference(case)
    mapping,_=gym.physical_maps(case.permutation)
    inv={physical:logical for logical,physical in enumerate(mapping)}
    if row['initial']!=sorted(inv[j] for j in init):raise ValueError('initial fact altered')
    if row['goal_truth']!=int(goal in closure):raise ValueError('Horn oracle goal target altered')
    limit=2 if dist is None else dist+1
    if row['budget']!=limit or row['optimal_steps']!=dist:raise ValueError('minimum Horn budget altered')
    if len(row['actions'])>limit:raise ValueError('action budget exceeded')
    known=set(init);stopped=False
    for at,entry in enumerate(row['actions']):
        if entry['before']!=sorted(inv[j] for j in known):raise ValueError('claimed predecessor state altered')
        available=sorted(i for i,(ps,h) in edges.items() if h not in known and set(ps)<=known)
        if entry['enabled']!=available:raise ValueError('public rule eligibility changed')
        action=entry['action']
        if action==gym.STOP:
            if entry['proof'] is not None:raise ValueError('STOP fabricated a proof edge')
            stopped=True
        else:
            if action not in available:raise ValueError('illegal Horn action accepted')
            prem,head=edges[action];proof=entry['proof']
            if not isinstance(proof,dict) or proof.get('rule_slot')!=action or proof.get('head')!=inv[head] or \
                    sorted(proof.get('premises',[]))!=sorted(inv[x] for x in prem):
                raise ValueError('proof edge not bound to public incidence')
            known.add(head)
        if entry['after']!=sorted(inv[x] for x in known):raise ValueError('claimed successor state altered')
        logits=entry.get('raw_logits')
        if logits is not None:
            if len(logits)!=base.RULES+1 or not all(isinstance(x,(float,int)) and abs(x)<1e20 for x in logits):
                raise ValueError('raw model logits invalid')
            legal=available+[gym.STOP]
            predicted=max(legal,key=lambda k:logits[k])
            if predicted!=action:raise ValueError('chosen action disagrees with model logits')
        if stopped or goal in known:
            if at!=len(row['actions'])-1:raise ValueError('action after terminal episode')
            break
    got=goal in known
    if bool(row['solved'])!=got or bool(row['stopped'])!=(stopped or got):
        raise ValueError('terminal proof status changed')
    if row['final']!=sorted(inv[x] for x in known) or row['used_actions']!=len(row['actions']):
        raise ValueError('final proof state changed')
    if row['host_accepted_false_proof'] or (got and goal not in closure):
        raise ValueError('false Horn proof accepted')
    return True


def audit(seed,training,evaluation,out,replays=1,smoke=False):
    training=Path(training);evaluation=Path(evaluation)
    rec,models=ex.load_models(training,seed,smoke)
    path=evaluation/'episodes.jsonl';summary=json.loads((evaluation/'evaluation.json').read_text())
    if summary['experiment']!='AF-16' or summary['seed']!=seed or summary['smoke']!=smoke:
        raise ValueError('AF16 scoring metadata mismatch')
    if summary['training_sha256']!=ex.sha(training/'train_receipt.json') or summary['source_hashes']!=ex.hashes() or \
       summary['manifest_sha256']!=ex.check_manifest() or summary['episode_sha256']!=ex.sha(path):
        raise ValueError('AF16 evaluation/source hashes mismatch')
    rows=[json.loads(r) for r in path.read_text().splitlines()]
    index={}
    for row in rows:
        key=(row['arm'],row['split'],row['case_id'])
        if key in index:raise ValueError('duplicate AF16 episode row')
        index[key]=row
    splits=('dev',) if smoke else ('novel_layout','novel_depth4','novel_depth5')
    expected=len(summary['scores'])*sum(len(gym.cases_for(s)) for s in splits)
    if len(rows)!=expected or len(rows)!=summary['rows']:raise ValueError('incomplete AF16 episode roster')
    sampled=0
    for arm in summary['scores']:
        for split in splits:
            cases=gym.cases_for(split);batch=[]
            for case in cases:
                row=index.get((arm,split,case.id))
                if row is None:raise ValueError('missing AF16 scored case')
                verify_row(case,row);batch.append(row)
            if ex.summarize(batch)!=summary['scores'][arm][split]:raise ValueError('AF16 saved scores differ from rows')
            if arm not in models:continue
            for case in list(cases[:replays])+list(cases[-replays:]):
                row=index[(arm,split,case.id)]
                for move in row['actions']:
                    x,bits,gate=gym.observations([case],[frozenset(move['before'])])
                    with torch.no_grad():out_logits=models[arm](x,bits)[0]
                    observed=torch.tensor(move['raw_logits'])
                    if not torch.allclose(out_logits,observed,atol=1e-5,rtol=0):
                        raise ValueError('AF16 independent raw-logit checkpoint replay mismatch')
                    sampled+=1
    result={'experiment':'AF-16','seed':seed,'passed':True,'episodes':len(rows),
            'forward_replays':sampled,'source_hashes':ex.hashes(),
            'training_sha256':ex.sha(training/'train_receipt.json'),
            'evaluation_sha256':ex.sha(evaluation/'evaluation.json')}
    ex.write(out,result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',required=True,type=int)
    p.add_argument('--training',required=True,type=Path)
    p.add_argument('--evaluation',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    print(json.dumps(audit(a.seed,a.training,a.evaluation,a.out,smoke=a.smoke)))
if __name__=='__main__':main()
