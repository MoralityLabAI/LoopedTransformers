"""Independent AF-22 physical-Horn BFS, proof, backup, and weight replay.

The independent proof oracle reads encoded public incidence. It never calls
AF-22's logical_rules, optimal_remaining, source proof policy, or label roster.
A separate sample replays saved neural outputs from the exact Torch weights.
"""
from __future__ import annotations
import argparse,json,math
from collections import deque
from pathlib import Path
import torch
from experiments.activation_funnel_af05 import gym as base
from . import gym,planner,experiment as ex


def public_graph(c):
    x=gym.encode(c);prem=x.prem_up+x.prem_down;head=x.head_up+x.head_down
    rules={}
    for i in range(base.RULES):
        if x.rule_features[i,2]<.5:continue
        pp=tuple(torch.where(prem[i]>.5)[0].tolist())
        hh=tuple(torch.where(head[i]>.5)[0].tolist())
        if not pp or len(pp)>2 or len(hh)!=1:raise ValueError('corrupt public incidence')
        rules[i]=(pp,hh[0])
    initial=frozenset(torch.where(x.atom_features[:,0]>.5)[0].tolist())
    return rules,initial,int(x.read_slots[2])


def shortest_bfs(rules,s,goal):
    state=frozenset(s)
    if goal in state:return 0
    queue=deque([(state,0)]);seen={state}
    while queue:
        prev,d=queue.popleft()
        for p,h in rules.values():
            if h in prev or not set(p)<=prev:continue
            nxt=prev|{h}
            if goal in nxt:return d+1
            if nxt not in seen:seen.add(nxt);queue.append((nxt,d+1))
    return None


def audit_episode(c,row):
    rules,initial,goal=public_graph(c)
    amap,_=gym.physical_maps(c.permutation)
    tophys=lambda states:frozenset(amap[a] for a in states)
    fromphys=lambda states:sorted(amap.index(a) for a in states)
    d=shortest_bfs(rules,initial,goal)
    if row['case_id']!=c.id or row['split']!=c.split:raise ValueError('episode identity mismatch')
    if row['initial']!=fromphys(initial):raise ValueError('initial facts differ from public graph')
    if row['optimal_steps']!=d or row['budget']!=(d+1 if d is not None else 2):
        raise ValueError('independent shortest-proof budget mismatch')
    if row['goal_truth']!=int(d is not None):raise ValueError('Horn truth differs')
    if len(row['actions'])>row['budget']:raise ValueError('proof exceeds exact budget')
    physical=set(initial);stopped=False;used_nodes=0
    for index,step in enumerate(row['actions']):
        if step['before']!=fromphys(physical):raise ValueError('before proof state differs')
        legal=sorted(i for i,(p,h) in rules.items() if h not in physical and set(p)<=physical)
        if step['enabled']!=legal:raise ValueError('legal action mask differs')
        act=step['action']
        if row['mode'] in planner.MODES:
            if step['plan_trace'] is None or step['decision_scores'] is None:
                raise ValueError('missing Bellman action ledger')
            # Check every numerical min-plus backup from the stored value ledger.
            selection=planner.check_trace_math(c,frozenset(step['before']),
                         step['plan_trace'],{row['mode']:step['decision_scores']})
            if selection[row['mode']]!=act:raise ValueError('Bellman selected action differs')
            used_nodes+=step['plan_trace']['nodes_forward_count']
        else:
            if step['plan_trace'] is not None or step['decision_scores'] is not None:
                raise ValueError('symbolic baseline acquired neural trace')
            if row['mode']=='fixed':expected=legal[0] if legal else gym.STOP
            elif row['mode']=='shortest':
                if d is None:expected=gym.STOP
                else:
                    rem=shortest_bfs(rules,physical,goal)
                    if rem==0:expected=gym.STOP
                    else:
                        choices=[i for i in legal if shortest_bfs(rules,
                            frozenset(physical)|{rules[i][1]},goal)==rem-1]
                        expected=choices[0]
            else:raise ValueError('unknown baseline')
            if act!=expected:raise ValueError('symbolic policy differs')
        if act==gym.STOP:
            if step['proof'] is not None:raise ValueError('STOP carried proof')
            stopped=True
        else:
            if act not in legal:raise ValueError('unsupported Horn rule accepted')
            p,h=rules[act]
            proof=step['proof']
            if not isinstance(proof,dict) or proof.get('rule_slot')!=act or \
                proof.get('head')!=amap.index(h) or sorted(proof.get('premises',[]))!=sorted(amap.index(k) for k in p):
                raise ValueError('invalid host rule proof')
            physical.add(h)
        if step['after']!=fromphys(physical):raise ValueError('post-action proof state differs')
        if stopped or goal in physical:
            if index!=len(row['actions'])-1:raise ValueError('continued after terminal')
    solved=goal in physical
    if row['final']!=fromphys(physical) or row['solved']!=solved or \
        row['early_stop']!=bool(d is not None and not solved and stopped) or \
        row['stopped']!=(stopped or solved) or row['used_actions']!=len(row['actions']):
        raise ValueError('incorrect terminal record')
    if row['host_accepted_false_proof'] or (solved and d is None):
        raise ValueError('false accepted goal')
    return used_nodes


def audit(seed,training,evaluation,out,smoke=False,replays=2):
    training=Path(training);evaluation=Path(evaluation)
    tr,model=ex.load_model(training,seed,smoke)
    ev=json.loads((evaluation/'evaluation.json').read_text())
    if (ev['experiment'],ev['seed'],ev['smoke'])!=('AF-22',seed,smoke) or \
       ev['source_hashes']!=ex.hashes() or ev['manifest_sha256']!=ex.check_manifest() or \
       ev['training_sha256']!=ex.sha(training/'train_receipt.json') or \
       ev['episodes_sha256']!=ex.sha(evaluation/'episodes.jsonl'):
        raise ValueError('scored source or checksum mismatch')
    rows=[json.loads(x) for x in (evaluation/'episodes.jsonl').read_text().splitlines()]
    families=('dev',) if smoke else ('held_layout','held_depth4','held_depth5')
    expected=sum(len(gym.cases_for(s)) for s in families)*(len(planner.MODES)+len(ex.BASELINES))
    if len(rows)!=expected or ev['episode_count']!=expected:raise ValueError('missing scored rows')
    index={};checks=0;node_checks=0;forward_replays=0
    for row in rows:
        key=(row['mode'],row['split'],row['case_id'])
        if key in index:raise ValueError('duplicate host proof episode')
        index[key]=row
    for split in families:
        cases=gym.cases_for(split)
        for mode in (*planner.MODES,*ex.BASELINES):
            roster=[]
            for case in cases:
                key=(mode,split,case.id)
                if key not in index:raise ValueError('missing registered proof case')
                row=index[key];node_checks+=audit_episode(case,row)
                roster.append(row);checks+=1
            if ex.summarize(roster)!=ev['scores'][split][mode]:
                raise ValueError('independent summary disagrees')
        # Replay deterministic first and last cases under same frozen weights.
        for case in (list(cases[:replays])+list(cases[-replays:])):
            for mode in planner.MODES:
                record=index[(mode,split,case.id)]
                for step in record['actions']:
                    c=planner.decisions(model,[case],[frozenset(step['before'])])[0]
                    saved=step['decision_scores']
                    if max(abs(x-y) for x,y in zip(c['scores'][mode],saved))>1e-5:
                        raise ValueError('actual checkpoint Bellman forward replay differs')
                    forward_replays+=1
    rec={'experiment':'AF-22','seed':seed,'verified':True,
         'proof_rows':checks,'planner_nodes_checked':node_checks,
         'checkpoint_forward_replays':forward_replays,'false_accepted_proofs':0,
         'evaluation_sha256':ex.sha(evaluation/'evaluation.json'),
         'training_sha256':ex.sha(training/'train_receipt.json'),
         'source_hashes':ex.hashes()}
    ex.write(out,rec)
    return rec


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    print(json.dumps(audit(a.seed,a.training,a.evaluation,a.out,a.smoke)))
if __name__=='__main__':main()
