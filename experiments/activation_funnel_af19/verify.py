"""AF-19 independent BFS, physical graph proof trace and causal-cone audit.

All derivation/reachability labels are rebuilt from public physical incidence;
they never come from the policy's internal Bellman source or its predictions.
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
    batch=gym.encode(case);prem=batch.prem_up+batch.prem_down;head=batch.head_up+batch.head_down
    rules={}
    for j in range(base.RULES):
        if float(batch.rule_features[j,2])<.5:continue
        p=tuple(int(i) for i in torch.where(prem[j]>.5)[0].tolist())
        h=tuple(int(i) for i in torch.where(head[j]>.5)[0].tolist())
        if not p or len(p)>2 or len(h)!=1:raise ValueError('invalid public Horn edge')
        rules[j]=(p,h[0])
    initial=frozenset(int(i) for i in torch.where(batch.atom_features[:,0]>.5)[0].tolist())
    return rules,initial,int(batch.read_slots[2])


def bfs(rules,known,goal):
    known=frozenset(known)
    if goal in known:return 0
    q=deque([(known,0)]);seen={known}
    while q:
        state,d=q.popleft()
        for p,h in rules.values():
            if h in state or not set(p)<=state:continue
            nxt=state|{h}
            if goal in nxt:return d+1
            if nxt not in seen:seen.add(nxt);q.append((nxt,d+1))
    return None


@lru_cache(maxsize=2000)
def independent_reference(case):
    rules,initial,goal=public_graph(case)
    closure=set(initial)
    for _ in range(base.ATOMS+1):
        new=closure|{h for p,h in rules.values() if set(p)<=closure}
        if new==closure:break
        closure=new
    else:raise ValueError('Horn closure failed to reach fixed point')
    d=bfs(rules,initial,goal)
    if (d is None)!=(goal not in closure):raise ValueError('closure and BFS disagree')
    return rules,initial,goal,d,frozenset(closure)


def independent_costs(case,state,cap):
    rules,_,goal,_,_=independent_reference(case)
    amap,_=gym.physical_maps(case.permutation)
    known=frozenset(amap[i] for i in state)
    d=bfs(rules,known,goal)
    costs={}
    for j,(p,h) in rules.items():
        if h in known or not set(p)<=known:continue
        rem=bfs(rules,known|{h},goal)
        costs[j]=float(cap if rem is None else min(cap,1+rem))
    costs[gym.STOP]=0. if d is None or d==0 else float(cap)
    return d,costs


def verify_row(case,row):
    if (row['case_id'],row['split'])!=(case.id,case.split):raise ValueError('case identity changed')
    rules,initial,goal,d,closure=independent_reference(case)
    amap,_=gym.physical_maps(case.permutation);inv={p:i for i,p in enumerate(amap)}
    if row['initial']!=sorted(inv[k] for k in initial):raise ValueError('initial facts changed')
    if row['goal_truth']!=int(goal in closure):raise ValueError('goal truth changed')
    budget=2 if d is None else d+1
    if row['budget']!=budget or row['optimal_steps']!=d:raise ValueError('shortest budget changed')
    if len(row['actions'])>budget:raise ValueError('episode exceeded budget')
    known=set(initial);stopped=False
    for t,step in enumerate(row['actions']):
        if step['before']!=sorted(inv[k] for k in known):raise ValueError('before-state changed')
        legal=sorted(j for j,(p,h) in rules.items() if h not in known and set(p)<=known)
        if step['enabled']!=legal:raise ValueError('host enabled-action roster changed')
        act=step['action'];proof=step['proof']
        if act==gym.STOP:
            if proof is not None:raise ValueError('STOP forged proof edge')
            stopped=True
        else:
            if act not in legal:raise ValueError('inadmissible accepted Horn action')
            p,h=rules[act]
            if not isinstance(proof,dict) or proof.get('rule_slot')!=act or \
               proof.get('head')!=inv[h] or sorted(proof.get('premises',[]))!=sorted(inv[k] for k in p):
                raise ValueError('proof not bound to physical Horn source')
            known.add(h)
        if step['after']!=sorted(inv[k] for k in known):raise ValueError('after-state changed')
        raw=step['raw_logits']
        if raw is not None:
            if len(raw)!=base.RULES+1 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in raw):
                raise ValueError('nonfinite or malformed raw policy logits')
            if max(legal+[gym.STOP],key=lambda a:raw[a])!=act:
                raise ValueError('model argmax mismatches host action')
        if stopped or goal in known:
            if t!=len(row['actions'])-1:raise ValueError('continuation after terminal action')
    solved=goal in known
    if row['solved']!=solved or row['stopped']!=(stopped or solved):raise ValueError('terminal flags changed')
    if row['final']!=sorted(inv[k] for k in known) or row['used_actions']!=len(row['actions']):
        raise ValueError('final state changed')
    early=bool(d is not None and not solved and stopped)
    if row['early_stop']!=early:raise ValueError('premature STOP bit changed')
    if row['host_accepted_false_proof'] or solved and goal not in closure:raise ValueError('unsound proof')


def physical_graph_distance(case,start_atom,goal_atom):
    rules,_,_,_,_=independent_reference(case)
    adj={('a',i):set() for i in range(base.ATOMS)}
    for j,(p,h) in rules.items():
        adj[('r',j)]=set()
        for i in (*p,h):
            adj[('a',i)].add(('r',j));adj[('r',j)].add(('a',i))
    q=deque([(start_atom,0)]);seen={start_atom}
    while q:
        u,d=q.popleft()
        if u==goal_atom:return d
        for v in adj[u]:
            if v not in seen:seen.add(v);q.append((v,d+1))
    return None


def independent_rule_masks(case):
    """Separate logical-to-physical backward graph search, without model masks."""
    rules,_,goal,_,_=independent_reference(case)
    local=frozenset(j for j,(prem,h) in rules.items() if h==goal)
    atoms={goal};ancestor=set()
    for _ in range(base.ATOMS+1):
        new={j for j,(prem,h) in rules.items() if h in atoms}
        if new==ancestor:break
        ancestor|=new
        for j in new:atoms.update(rules[j][0])
    else:raise ValueError('independent ancestry failed to settle')
    if not local <= ancestor:raise ValueError('local direct goal rule missing from ancestry')
    return local,frozenset(ancestor)


def audit(seed,training,evaluation,out,replays=1,smoke=False):
    training=Path(training);evaluation=Path(evaluation)
    receipt,models=ex.load_models(training,seed,smoke)
    summary=json.loads((evaluation/'evaluation.json').read_text())
    if (summary['experiment'],summary['seed'],summary['smoke'])!=('AF-19',seed,smoke):
        raise ValueError('AF19 evaluation identity changed')
    if summary['train_sha256']!=ex.sha(training/'train_receipt.json') or \
       summary['source_hashes']!=ex.hashes() or summary['manifest_sha256']!=ex.check_manifest() or \
       summary['episodes_sha256']!=ex.sha(evaluation/'episodes.jsonl') or \
       summary['cone_sha256']!=ex.sha(evaluation/'causal_cone.jsonl'):
        raise ValueError('source or evaluation hash changed')
    rows=[json.loads(l) for l in (evaluation/'episodes.jsonl').read_text().splitlines()]
    index={}
    for row in rows:
        key=(row['config'],row['split'],row['case_id'])
        if key in index:raise ValueError('duplicate scored episode')
        index[key]=row
    families=('dev',) if smoke else ('held_layout','held_depth4','held_depth5')
    expected=len(summary['scores'])*sum(len(gym.cases_for(s)) for s in families)
    if len(rows)!=expected or summary['rows']!=expected:raise ValueError('incomplete AF19 roster')
    replay_count=0
    for arm in summary['scores']:
        for sp in families:
            cases=gym.cases_for(sp);observed=[]
            for c in cases:
                key=(arm,sp,c.id)
                if key not in index:raise ValueError('missing scored case')
                row=index[key];verify_row(c,row);observed.append(row)
            if summary['scores'][arm][sp]!=ex.summarize(observed):
                raise ValueError('recorded score differs from independent proof audit')
            if arm not in models:continue
            for c in list(cases[:replays])+list(cases[-replays:]):
                row=index[(arm,sp,c.id)]
                for step in row['actions']:
                    batch,known,gate=gym.observations([c],[frozenset(step['before'])])
                    with torch.no_grad():raw=models[arm](batch,known)[0]
                    if not torch.allclose(raw,torch.tensor(step['raw_logits']),atol=1e-5,rtol=0):
                        raise ValueError('AF19 checkpoint forward replay differs')
                    replay_count+=1
    labels=[json.loads(l) for l in (training/'train_cost_receipt.jsonl').read_text().splitlines()]
    sources=list(ex.training_states()[0]+ex.training_states()[1])
    if len(labels)!=len(sources):raise ValueError('missing training cost labels')
    cost_checks=0
    for i in sorted({0,len(labels)-1,*range(0,len(labels),max(1,len(labels)//80))}):
        c,state=sources[i];r=labels[i]
        if (r['case_id'],r['state'])!=(c.id,sorted(state)):raise ValueError('cost-label source changed')
        for cap in (16,):
            d,costs=independent_costs(c,state,cap)
            if r['distance']!=d or {int(k):v for k,v in r[f'cost_cap{cap}'].items()}!=costs:
                raise ValueError('Bellman label differs from independent BFS')
            cost_checks+=1
    probes=[json.loads(l) for l in (evaluation/'causal_cone.jsonl').read_text().splitlines()]
    if len(probes)!=len(models)*3 or summary['probes']!=len(probes):
        raise ValueError('incomplete cone probe roster')
    probe_replays=0
    for name,model in models.items():
        group=[r for r in probes if r['config']==name]
        if len(group)!=3:raise ValueError('missing model probe')
        for row,(a,b) in zip(sorted(group,key=lambda x:x['depth']),gym.probe_pairs()):
            rules,_,g,_,_=independent_reference(a)
            apos,_=gym.physical_maps(a.permutation)
            distance=physical_graph_distance(a,('a',apos[0]),('a',g))
            if distance!=row['graph_distance'] or row['outside_local_cone']!=(distance>2*model.loops):
                raise ValueError('causal cone distance mismatch')
            x,k,_=gym.observations([a,b],[gym.initial_facts(a),gym.initial_facts(b)])
            with torch.no_grad():raw,d,reach=model.full(x,k)
            actual={'stop_difference':float((raw[0,-1]-raw[1,-1]).abs()),
                'reach_difference':float((reach[0]-reach[1]).abs()),
                'distance_difference':float((d[0]-d[1]).abs())}
            for field,v in actual.items():
                if abs(v-row[field])>1e-6:raise ValueError('causal probe saved logit changed')
            if model.mode=='local' and distance>2*model.loops and any(v>1e-6 for v in actual.values()):
                raise ValueError('unphysical goal-local influence beyond recurrence cone')
            from .model import relevance_masks
            local_mask,ancestor_mask=relevance_masks(x)
            independent_local,independent_ancestor=independent_rule_masks(a)
            if set(torch.where(local_mask[0])[0].tolist())!=independent_local or \
                set(torch.where(ancestor_mask[0])[0].tolist())!=independent_ancestor:
                raise ValueError('independent graph ancestry disagrees with model readout masks')
            if row['readout_mode']!=model.mode or row['local_rule_count']!=int(local_mask[0].sum()) or \
                row['ancestor_rule_count']!=int(ancestor_mask[0].sum()):
                raise ValueError('graph-ancestry mask evidence inconsistent')
            if row['irrelevant_perturbation_max']>1e-5:
                raise ValueError('irrelevant padded-rule control violated')
            absent=row['padded_rule_slot']
            from dataclasses import fields
            alt=base.GraphBatch(*(getattr(x,f.name).clone() for f in fields(base.GraphBatch)))
            alt.rule_features[:,absent,:]=torch.tensor([1.7,-2.3,0.0])
            with torch.no_grad():raw_alt,dist_alt,reach_alt=model.full(alt,k)
            observed=max(float((raw_alt[:,-1]-raw[:,-1]).abs().max()),
                float((dist_alt-d).abs().max()),float((reach_alt-reach).abs().max()))
            if abs(observed-row['irrelevant_perturbation_max'])>1e-6:
                raise ValueError('independent irrelevant-perturbation replay mismatch')
            if row['truth_a']!=0 or row['truth_b']!=1:raise ValueError('paired Horn truth changed')
            probe_replays+=1
    rec={'experiment':'AF-19','seed':seed,'passed':True,'episodes':len(rows),
        'forward_replays':replay_count,'independent_bfs_cost_checks':cost_checks,
        'cone_probe_replays':probe_replays,'host_false_proofs':0,'readout_mask_replays':probe_replays,
        'training_sha256':ex.sha(training/'train_receipt.json'),
        'evaluation_sha256':ex.sha(evaluation/'evaluation.json'),
        'source_hashes':ex.hashes()}
    ex.write(out,rec);return rec

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',required=True,type=int)
    p.add_argument('--training',required=True,type=Path);p.add_argument('--evaluation',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();print(json.dumps(audit(a.seed,a.training,a.evaluation,a.out,smoke=a.smoke)))
if __name__=='__main__':main()
