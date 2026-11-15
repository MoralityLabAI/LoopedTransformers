"""Independent AF-15 receipt replay: physical incidence, legal proof edges, and learned logits."""
from __future__ import annotations
import argparse,hashlib,json,collections
from pathlib import Path
import torch
from experiments.activation_funnel_af05 import gym as base
from . import gym,model as policy,experiment as ex


def physical_edges(case):
    """Independently recover rule semantics from the public incidence tensors."""
    graph=gym.encode(case)
    premises=graph.prem_up+graph.prem_down;heads=graph.head_up+graph.head_down
    edges={}
    for j in range(base.RULES):
        if float(graph.rule_features[j,2])<0.5:continue
        ps=tuple(int(i) for i in torch.where(premises[j]>0.5)[0].tolist())
        h=tuple(int(i) for i in torch.where(heads[j]>0.5)[0].tolist())
        if len(h)!=1:raise ValueError('bad public Horn rule head')
        edges[j]=(ps,h[0])
    return edges


def independent_enabled(edges,known_phys):
    known=set(known_phys)
    return sorted(j for j,(p,h) in edges.items() if h not in known and set(p)<=known)


def verify_episode(case,row):
    if row['case_id']!=case.id or row['split']!=case.split:raise ValueError('AF-15 case mismatch')
    slots,_=gym.physical_maps(case.permutation)
    src=set(base.initial_facts(case));known={slots[k] for k in src}
    graph=gym.encode(case)
    goal=int(graph.read_slots[2]);edge=physical_edges(case)
    if row['initial']!=sorted(src):raise ValueError('AF-15 initial fact mutation')
    reference=base.oracle(case)[0][2]
    if row['goal_truth']!=reference:raise ValueError('AF-15 oracle goal mutated')
    distance=gym.optimal_remaining(case,src)
    budget=distance+1 if distance<10**6 else 2
    if row['budget']!=budget or row['optimal_steps']!=(None if distance>=10**6 else distance):
        raise ValueError('AF-15 proof budget mutated')
    if len(row['actions'])>budget:raise ValueError('AF-15 action budget exceeded')
    stopped=False
    for t,action in enumerate(row['actions']):
        before={slots[k] for k in action['before']}
        if before!=known:raise ValueError('AF-15 claimed prior state altered')
        eligible=independent_enabled(edge,known)
        if eligible!=action['enabled']:raise ValueError('AF-15 host eligibility mutated')
        choice=action['action']
        if choice==base.RULES:
            if action['proof'] is not None:raise ValueError('STOP claimed unsupported proof')
            stopped=True
        else:
            if choice not in eligible:raise ValueError('AF-15 accepted an invalid Horn rule')
            ps,h=edge[choice]
            if action['proof'] is None:raise ValueError('missing Horn proof edge')
            proof=action['proof']
            if (proof.get('rule_slot')!=choice or proof.get('head')!=slots.index(h) or
                sorted(proof.get('premises',[]))!=sorted(slots.index(p) for p in ps)):
                raise ValueError('AF-15 proof edge is not source-bound')
            known.add(h)
        if action['after']!=sorted(slots.index(k) for k in known):raise ValueError('AF-15 claimed next state altered')
        if stopped or goal in known:
            if t!=len(row['actions'])-1:raise ValueError('AF-15 extra actions after terminal proof')
            break
    got=goal in known
    if bool(row['solved'])!=got or bool(row['stopped'])!=(stopped or got):
        raise ValueError('AF-15 terminal state altered')
    if row['final']!=sorted(slots.index(k) for k in known):
        raise ValueError('AF-15 final state changed')
    if row['used_actions']!=len(row['actions']) or row['host_accepted_false_proof']:
        raise ValueError('AF-15 unsupported proof credit')
    if got and not reference:raise ValueError('AF-15 false proof accepted')
    return True


def audit(seed,training,evaluation,out,replays=2,smoke=False):
    training=Path(training);evaluation=Path(evaluation)
    rec,models=ex.load_models(training,seed,smoke)
    ep=evaluation/'episodes.jsonl';ev=evaluation/'evaluation.json'
    summary=json.loads(ev.read_text())
    if summary['seed']!=seed or summary['training_sha256']!=ex.sha(training/'train_receipt.json'):
        raise ValueError('evaluation provenance changed')
    if summary['source_hashes']!=ex.hashes() or summary['manifest_sha256']!=ex.check_manifest() or summary['episodes_sha256']!=ex.sha(ep):
        raise ValueError('AF-15 evaluation evidence hash mismatch')
    lines=[json.loads(r) for r in ep.read_text().splitlines()]
    index={}
    for row in lines:
        k=(row['arm'],row['split'],row['case_id'])
        if k in index:raise ValueError('AF-15 duplicate score row')
        index[k]=row
    splits=('dev',) if smoke else gym.FRESH
    if len(lines)!=len(summary['scores'])*sum(len(gym.cases_for(s)) for s in splits) or summary['rows']!=len(lines):
        raise ValueError('AF-15 evaluation episode roster incomplete')
    checks=0;forward=0
    for arm in summary['scores']:
        for split in splits:
            cases=gym.cases_for(split);rows=[]
            for case in cases:
                r=index.get((arm,split,case.id))
                if r is None:raise ValueError('AF-15 missing episode')
                verify_episode(case,r);rows.append(r);checks+=1
            recomputed=ex.summarize(rows)
            if recomputed!=summary['scores'][arm][split]:raise ValueError('AF-15 scores do not match raw episodes')
            if arm in models:
                for case in (cases[:replays]+cases[-replays:]):
                    row=index[arm,split,case.id]
                    for event in row['actions']:
                        if event['raw_logits'] is None:raise ValueError('AF-15 neural model logit row missing')
                        x,bits,gate=policy.observations([case],[frozenset(event['before'])])
                        with torch.no_grad():observed=models[arm](x,bits)[0].tolist()
                        if max(abs(a-b) for a,b in zip(observed,event['raw_logits']))>2e-5:
                            raise ValueError('AF-15 checkpoint forward replay mismatch')
                        forward+=1
    result={'experiment':'AF-15','seed':seed,'passed':True,'episodes':len(lines),
         'source_sha256':ex.sha(ex.PROTOCOL),'episode_sha256':ex.sha(ep),
         'verified_semantic_rows':checks,'forward_replays':forward,
         'training_sha256':ex.sha(training/'train_receipt.json'),
         'evaluation_sha256':ex.sha(ev)}
    ex.write(out,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();print(json.dumps(audit(a.seed,a.training,a.evaluation,a.out,smoke=a.smoke)))
if __name__=='__main__':main()
