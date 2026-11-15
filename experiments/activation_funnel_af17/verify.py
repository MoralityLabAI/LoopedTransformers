"""Independent AF-17 Horn proof and Bellman-state audit from public incidence.

Truth reconstruction reads physical rule tensors and performs BFS over legal
symbolic states. It does not call the trained Q target or policy DP to certify
proof reachability, budgets, or recorded steps.
"""
from __future__ import annotations
from collections import deque
from functools import lru_cache
from pathlib import Path
import argparse,json,math
import torch
from experiments.activation_funnel_af05 import gym as base
from . import gym,experiment as ex


def public_graph(case):
    x=gym.encode(case);prem=x.prem_up+x.prem_down;head=x.head_up+x.head_down
    rules={}
    for k in range(base.RULES):
        if float(x.rule_features[k,2])<.5:continue
        p=tuple(int(i) for i in torch.where(prem[k]>.5)[0].tolist())
        h=tuple(int(i) for i in torch.where(head[k]>.5)[0].tolist())
        if not p or len(p)>2 or len(h)!=1:raise ValueError('invalid AF17 physical Horn edge')
        rules[k]=(p,h[0])
    init=frozenset(int(i) for i in torch.where(x.atom_features[:,0]>.5)[0].tolist())
    goal=int(x.read_slots[2]);return rules,init,goal


def search(rules,known,goal):
    """Minimum proof fires from physical known atoms, independent BFS."""
    known=frozenset(known)
    if goal in known:return 0
    q=deque([(known,0)]);seen={known}
    while q:
        state,depth=q.popleft()
        for prem,h in rules.values():
            if h in state or not set(prem)<=state:continue
            nxt=state|{h}
            if goal in nxt:return depth+1
            if nxt not in seen:seen.add(nxt);q.append((nxt,depth+1))
    return None


@lru_cache(maxsize=3600)
def independent_reference(case):
    rules,initial,goal=public_graph(case)
    closure=set(initial)
    for _ in range(base.ATOMS+1):
        nxt=closure|{h for prem,h in rules.values() if set(prem)<=closure}
        if nxt==closure:break
        closure=nxt
    else:raise ValueError('Horn closure did not stabilize')
    d=search(rules,initial,goal)
    if (d is None)!=(goal not in closure):raise ValueError('BFS and closure disagree')
    return rules,initial,goal,d,frozenset(closure)


def independent_costs(case,state):
    rules,_,goal,_,_=independent_reference(case)
    amap,_=gym.physical_maps(case.permutation)
    known=frozenset(amap[i] for i in state)
    base_distance=search(rules,known,goal)
    out={}
    for j,(prem,h) in rules.items():
        if h in known or not set(prem)<=known:continue
        next_d=search(rules,known|{h},goal)
        out[j]=float(gym.CAP if next_d is None else min(gym.CAP,1+next_d))
    out[gym.STOP]=0. if base_distance is None or base_distance==0 else float(gym.CAP)
    return base_distance,out


def verify_row(case,row):
    if (row['case_id'],row['split'])!=(case.id,case.split):raise ValueError('case ID or split changed')
    rules,init,goal,shortest,closure=independent_reference(case)
    amap,_=gym.physical_maps(case.permutation)
    inverse={p:i for i,p in enumerate(amap)}
    if row['initial']!=sorted(inverse[k] for k in init):raise ValueError('initial facts changed')
    if row['goal_truth']!=int(goal in closure):raise ValueError('goal truth changed')
    budget=2 if shortest is None else shortest+1
    if row['budget']!=budget or row['optimal_steps']!=shortest:raise ValueError('shortest-proof budget altered')
    if len(row['actions'])>budget:raise ValueError('action budget overflow')
    known=set(init);stopped=False
    for idx,entry in enumerate(row['actions']):
        if entry['before']!=sorted(inverse[k] for k in known):raise ValueError('claimed before-state changed')
        enabled=sorted(i for i,(prem,h) in rules.items() if h not in known and set(prem)<=known)
        if entry['enabled']!=enabled:raise ValueError('host enabled-action roster changed')
        action=entry['action']
        if action==gym.STOP:
            if entry['proof'] is not None:raise ValueError('STOP forged proof edge')
            stopped=True
        else:
            if action not in enabled:raise ValueError('illegal Horn rule accepted')
            p,h=rules[action];proof=entry['proof']
            if not isinstance(proof,dict) or proof.get('rule_slot')!=action or \
               proof.get('head')!=inverse[h] or sorted(proof.get('premises',[]))!=sorted(inverse[i] for i in p):
                raise ValueError('proof edge not bound to actual public incidence')
            known.add(h)
        if entry['after']!=sorted(inverse[k] for k in known):raise ValueError('claimed successor state changed')
        raw=entry['raw_logits']
        if raw is not None:
            if len(raw)!=base.RULES+1 or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in raw):
                raise ValueError('invalid raw policy logits')
            allowed=enabled+[gym.STOP]
            if max(allowed,key=lambda a:raw[a])!=action:raise ValueError('model argmax disagrees with chosen action')
        if stopped or goal in known:
            if idx!=len(row['actions'])-1:raise ValueError('actions after termination')
    solved=goal in known
    if bool(row['solved'])!=solved or bool(row['stopped'])!=(stopped or solved):raise ValueError('terminal state changed')
    if row['final']!=sorted(inverse[k] for k in known) or row['used_actions']!=len(row['actions']):raise ValueError('final state changed')
    early_stop=bool(shortest is not None and not solved and stopped)
    if bool(row['early_stop'])!=early_stop:raise ValueError('early STOP evidence changed')
    if row['host_accepted_false_proof'] or (solved and goal not in closure):raise ValueError('unsound proof accepted')
    return True


def audit(seed,training,evaluation,out,replays=1,smoke=False):
    training=Path(training);evaluation=Path(evaluation)
    receipt,models=ex.load_models(training,seed,smoke)
    summary=json.loads((evaluation/'evaluation.json').read_text())
    if (summary['experiment'],summary['seed'],summary['smoke'])!=('AF-17',seed,smoke):raise ValueError('AF17 evaluation identity changed')
    if summary['training_sha256']!=ex.sha(training/'train_receipt.json') or \
       summary['source_hashes']!=ex.hashes() or summary['manifest_sha256']!=ex.check_manifest() or \
       summary['episode_sha256']!=ex.sha(evaluation/'episodes.jsonl'):
        raise ValueError('evaluation provenance changed')
    rows=[json.loads(line) for line in (evaluation/'episodes.jsonl').read_text().splitlines()]
    index={}
    for row in rows:
        key=(row['arm'],row['split'],row['case_id'])
        if key in index:raise ValueError('duplicate episode in scored roster')
        index[key]=row
    families=('dev',) if smoke else ('held_layout','held_depth4','held_depth5')
    expected=len(summary['scores'])*sum(len(gym.cases_for(s)) for s in families)
    if len(rows)!=expected or summary['rows']!=expected:raise ValueError('incomplete AF17 episode roster')
    replay=0
    for arm in summary['scores']:
        for split in families:
            cases=gym.cases_for(split)
            observed=[]
            for c in cases:
                row=index.get((arm,split,c.id))
                if row is None:raise ValueError('missing AF17 scored case')
                verify_row(c,row);observed.append(row)
            if ex.summarize(observed)!=summary['scores'][arm][split]:raise ValueError('saved summary differs from verified row scores')
            if arm not in models:continue
            for c in list(cases[:replays])+list(cases[-replays:]):
                row=index[(arm,split,c.id)]
                for step in row['actions']:
                    x,known,gate=gym.observations([c],[frozenset(step['before'])])
                    with torch.no_grad():raw=models[arm](x,known)[0]
                    if not torch.allclose(raw,torch.tensor(step['raw_logits']),atol=1e-5,rtol=0):
                        raise ValueError('checkpoint raw-logit replay mismatch')
                    replay+=1
    # Independently audit a bounded deterministic sample of the train-only
    # Bellman label receipt by reconstructing costs from physical graph BFS.
    labels=[json.loads(r) for r in (training/'train_label_receipt.jsonl').read_text().splitlines()]
    source=list(ex.training_states()[0]+ex.training_states()[1])
    if len(labels)!=len(source):raise ValueError('Bellman label source roster incomplete')
    label_checks=0
    for i in sorted({0,len(labels)-1,*range(0,len(labels),max(1,len(labels)//80))}):
        c,state=source[i];r=labels[i]
        if (r['case_id'],r['state'])!=(c.id,sorted(state)):raise ValueError('Bellman label source changed')
        independent,costs=independent_costs(c,state)
        if r['d']!=independent or {int(k):v for k,v in r['action_costs'].items()}!=costs:
            raise ValueError('Bellman label disagrees with independent BFS')
        label_checks+=1
    report={'experiment':'AF-17','seed':seed,'passed':True,'episodes':len(rows),
       'forward_replays':replay,'independent_cost_label_checks':label_checks,
       'train_sha256':ex.sha(training/'train_receipt.json'),
       'evaluation_sha256':ex.sha(evaluation/'evaluation.json'),
       'source_hashes':ex.hashes(),'false_accepted_proofs':0}
    ex.write(out,report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();print(json.dumps(audit(a.seed,a.training,a.evaluation,a.out,smoke=a.smoke)))
if __name__=='__main__':main()
