"""Independent AF-07 recurrence-locality auditor; independently recomputes labels and scores.

Results are validated from retained per-case logits and the deterministic Horn
oracle, not copied from the evaluator's aggregate values. It also compares
whole-code patches to separately saved full donor forward predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
import math
from .core import paired_traces, trace_metrics
from pathlib import Path

import torch

from experiments.activation_funnel_af05 import gym
from . import experiment as ex

TOL = 5e-6


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def close(a,b,tol=TOL):
    if isinstance(a,list) and isinstance(b,list):
        return len(a)==len(b) and all(close(x,y,tol) for x,y in zip(a,b))
    return abs(a-b)<=tol


def verify_one(root, training, seed, width, smoke=False, replay_samples=6):
    root, training=Path(root),Path(training)
    ex.validate_manifest()
    entry=json.loads((root/'evaluation.json').read_text())
    if ex.sha_file(root/'evaluation.json') != (root/'evaluation.sha256').read_text().strip():
        raise ValueError('evaluation header mutation')
    if (entry['seed'],entry['width'],entry['smoke']) != (seed,width,smoke):
        raise ValueError('wrong evaluation identity')
    if entry['source_train_receipt']!=ex.sha_file(training/'train_receipt.json') or entry['source']!=ex.source_receipt():
        raise ValueError('evaluation source is not the checked training run')
    for name,digest in entry['files'].items():
        if digest!=ex.sha_file(root/name):raise ValueError('corrupt evaluation file '+name)
    clean=read_records(root/'clean_predictions.jsonl')
    patch=read_records(root/'patch_predictions.jsonl')
    trajectories=read_records(root/'trajectory_predictions.jsonl')
    splits=('dev',) if smoke else ('held_composition','held_layout','held_depth4','held_depth5')
    valid={(arm,split,case.id):case for arm in ex.ARMS
           for split in splits for case in gym.cases_for(split)}
    expected_total=3*sum(len(gym.cases_for(s)) for s in splits)
    if len(clean)!=expected_total or len(patch)!=expected_total*2:
        raise ValueError('missing or duplicate predictions')
    clean_index={}
    for r in clean:
        key=(r['arm'],r['split'],r['id'])
        if key in clean_index or key not in valid:
            raise ValueError('unexpected or duplicate clean row')
        case=valid[key];target=gym.oracle(case)[0]
        if r['target']!=list(target) or r.get('pre_loops')!=ex.EVAL_LOOPS[r['split']] or (r['seed'],r['width'])!=(seed,width):
            raise ValueError('clean truth/identity disagreement')
        if len(r['logits'])!=3 or any(len(x)!=2 for x in r['logits']):
            raise ValueError('missing original logits')
        clean_index[key]=r
    for arm in ex.ARMS:
        for split in splits:
            records=[clean_index[(arm,split,c.id)] for c in gym.cases_for(split)]
            logits=torch.tensor([r['logits'] for r in records])
            targets=torch.tensor([r['target'] for r in records])
            measured=ex.metrics(logits,targets)
            reported=entry['scores'][arm][split]
            for k in ('goal_accuracy','subgoal_exact','all_exact','false_goal_attempts'):
                if abs(measured[k]-reported[k])>TOL:
                    raise ValueError('clean metric mismatch '+k)
    patch_index={}
    for r in patch:
        key=(r['arm'],r['split'],r['recipient_id'],r['selected'])
        if key in patch_index or r['selected'] not in (0,1):
            raise ValueError('duplicate patched recipient or bad branch selector')
        case=valid.get((r['arm'],r['split'],r['recipient_id']))
        if case is None:raise ValueError('unknown recipient')
        cs=gym.cases_for(r['split'])
        look={c.id:i for i,c in enumerate(cs)}
        i=look[case.id]
        donor_ix,oracle=gym.donors_for(r['split'])[i,r['selected']]
        donor=cs[donor_ix]
        if r['donor_id']!=donor.id or r['oracle']!=list(oracle) or r.get('pre_loops')!=ex.EVAL_LOOPS[r['split']] or \
           r['recipient_truth']!=list(gym.oracle(case)[0]):
            raise ValueError('patched truth/donor mismatch')
        if len(r['patched_subgoal_code'])!=width:
            raise ValueError('bad carried-code width')
        d_logit=clean_index[(r['arm'],r['split'],donor.id)]['logits']
        if not close(r['whole_logits'],d_logit,2e-5):
            raise ValueError('whole-code patch does not match donor forward logits')
        if (r['seed'],r['width'])!=(seed,width):raise ValueError('cross-seed score leakage')
        patch_index[key]=r
    if len(patch_index)!=len(patch):raise ValueError('patch record roster incomplete')
    for arm in ex.ARMS:
        for split in splits:
            rows=[patch_index[arm,split,c.id,j] for c in gym.cases_for(split) for j in (0,1)]
            target=torch.tensor([r['oracle'] for r in rows])
            original=torch.tensor([r['recipient_truth'] for r in rows])
            pred=torch.tensor([r['addressed_logits'] for r in rows]).argmax(-1)
            other=torch.tensor([r['other_logits'] for r in rows]).argmax(-1)
            selectors=torch.tensor([r['selected'] for r in rows])
            good=pred==target
            flip=target[:,2]!=original[:,2]
            if not flip.any():raise ValueError('uninformative counterfactual family')
            measured={
                'pairs':len(rows), 'informative_goal_pairs':int(flip.sum()),
                'addressed_subgoal_accuracy':float(good.gather(1,selectors[:,None]).float().mean()),
                'untouched_subgoal_accuracy':float(good.gather(1,(1-selectors)[:,None]).float().mean()),
                'addressed_goal_accuracy':float(good[:,2].float().mean()),
                'informative_goal_accuracy':float(good[flip,2].float().mean()),
                'other_patch_goal_retention':float((other[:,2]==original[:,2]).float().mean()),
                'other_patch_subgoal_exact':float((other[:,:2]==original[:,:2]).all(-1).float().mean())}
            reported=entry['patches'][arm][split]
            for k,v in measured.items():
                if abs(v-reported[k])>TOL:raise ValueError('recomputed patch metric mismatch '+k)
            if reported['whole_direct_max_logit_error']>2e-5:
                raise ValueError('source transplant numerical mismatch')
    # Recompute recurrence-resolved predictions and statistics independently from
    # the retained original logits and Horn oracle. Trace hashes bind the raw
    # numerical record; selected checkpoints are re-executed below.
    stage_index={}
    expected_trace=0
    for arm in ex.ARMS:
        for split in splits:
            expected_trace+=len(gym.cases_for(split))*2*(2 if split in ('held_depth4','held_depth5') and not smoke else 1)
    if len(trajectories)!=expected_trace:
        raise ValueError('missing or duplicated recurrence traces')
    for r in trajectories:
        key=(r['arm'],r['split'],r['recipient_id'],r['selected'],r['post_loops'])
        if key in stage_index or r['selected'] not in (0,1):
            raise ValueError('duplicate staged recipient/selection/horizon')
        if r['post_loops'] not in ((3,5) if r['split'] in ('held_depth4','held_depth5') and not smoke else (3,)):
            raise ValueError('unregistered recurrence horizon')
        known=patch_index.get(key[:4])
        if known is None or r['oracle']!=known['oracle'] or r['donor_id']!=known['donor_id'] or \
           r['recipient_truth']!=known['recipient_truth'] or (r['seed'],r['width'])!=(seed,width):
            raise ValueError('staged source, recipient or Horn oracle changed')
        if r['pre_loops']!=ex.EVAL_LOOPS[r['split']]:
            raise ValueError('altered upstream recurrence count')
        steps=r['post_loops'];protected=1-r['selected']
        if any(len(r[k])!=steps for k in ('clean_q_steps','patched_q_steps','patched_goal_steps',
           'protected_q_correct_by_step','patched_q_correct_by_step','protected_code_l2_by_step')):
            raise ValueError('missing recurrence step')
        for t in range(steps):
            q=r['patched_q_steps'][t]
            clean_q=r['clean_q_steps'][t]
            if len(q)!=2 or len(clean_q)!=2 or any(len(x)!=2 for x in q+clean_q):
                raise ValueError('bad intermediate Q logit shape')
            if len(r['patched_goal_steps'][t])!=2:
                raise ValueError('bad intermediate goal logit shape')
            correct=[int(q[b][1]>q[b][0])==r['oracle'][b] for b in (0,1)]
            if r['patched_q_correct_by_step'][t]!=correct:
                raise ValueError('intermediate subgoal target mismatch')
            if r['protected_q_correct_by_step'][t]!=(int(q[protected][1]>q[protected][0])==r['recipient_truth'][protected]):
                raise ValueError('untouched Q semantic mismatch')
            drift=r['protected_code_l2_by_step'][t]
            if not isinstance(drift,(int,float)) or not math.isfinite(drift) or drift<0:
                raise ValueError('invalid protected-code displacement')
        if steps==3:
            if not close(r['patched_q_steps'][-1]+[r['patched_goal_steps'][-1]], known['addressed_logits'],2e-5):
                raise ValueError('stage3 readout differs from registered native patch output')
            orig=clean_index[(r['arm'],r['split'],r['recipient_id'])]['logits']
            if not close(r['clean_q_steps'][-1],orig[:2],2e-5):
                raise ValueError('clean stage3 readout differs from original')
        stage_index[key]=r
    for arm in ex.ARMS:
        for split in splits:
            for steps,reported in entry['trajectories'][arm][split].items():
                horizon=int(steps)
                group=[stage_index[arm,split,c.id,b,horizon]
                       for c in gym.cases_for(split) for b in (0,1)]
                actual=ex.trajectory_summary(group)
                for key in actual:
                    if not close(actual[key],reported[key],5e-6):
                        raise ValueError('recurrence-resolved aggregate mismatch: '+key)
    # Fresh sample from checkpoint validates saved latents, logits and derived recurrence traces.
    _,models=ex.load_frozen(training,seed,width,smoke)
    replays=0; stage_replays=0
    for arm in ex.ARMS:
        model = models[arm]
        for split in splits:
            cs=gym.cases_for(split)
            ds=ex.Dataset(split)
            for i in range(min(replay_samples,len(cs))):
                selection=i%2
                receiver=ds.get([i])
                donor,truth=ds.pair([i],[selection])
                with torch.no_grad():
                    output=model.patched(receiver,donor,torch.tensor([selection]),pre_loops=ex.EVAL_LOOPS[split])
                r=patch_index[(arm,split,cs[i].id,selection)]
                actual=torch.cat([output.subgoals,output.goal.unsqueeze(1)],1)[0].tolist()
                code=output.pre.az[0,receiver.read_slots[0,selection]].tolist()
                if not close(actual,r['addressed_logits'],2e-5):
                    raise ValueError('Torch checkpoint intervention replay mismatch')
                if not close(code,r['patched_subgoal_code'],2e-5):
                    raise ValueError('saved causal code replay mismatch')
                replays+=1
                for post_loops in ((3,5) if split in ('held_depth4','held_depth5') and not smoke else (3,)):
                    with torch.no_grad():
                        clean_tr,patched_tr=paired_traces(model,receiver,donor,
                            torch.tensor([selection]),torch.ones(1,dtype=torch.bool),
                            ex.EVAL_LOOPS[split],post_loops)
                        stats=trace_metrics(clean_tr,patched_tr,receiver.targets,truth,
                            torch.tensor([selection]),torch.ones(1,dtype=torch.bool))
                    saved=stage_index[arm,split,cs[i].id,selection,post_loops]
                    for key,data in [('clean_q_steps',clean_tr.q_steps[0].tolist()),
                                     ('patched_q_steps',patched_tr.q_steps[0].tolist()),
                                     ('patched_goal_steps',patched_tr.goal_steps[0].tolist()),
                                     ('protected_code_l2_by_step',stats['protected_code_l2_by_step'][0])]:
                        if not close(saved[key],data,2e-5):
                            raise ValueError('Torch recurrence trace replay mismatch: '+key)
                    stage_replays+=1
    return {'passed':True,'seed':seed,'width':width,
            'clean_rows':len(clean),'counterfactual_rows':len(patch),
            'recomputed_score_tables':len(ex.ARMS)*len(splits),
            'independent_torch_patch_replays':replays,
            'recurrence_trace_rows':len(trajectories),
            'independent_torch_trace_replays':stage_replays,
            'schema':'af07-independent-v1',
            'source_train_receipt':entry['source_train_receipt'],
            'evaluation_sha256':ex.sha_file(root/'evaluation.json')}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--width',type=int,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    result=verify_one(a.evaluation,a.train,a.seed,a.width,a.smoke)
    ex.json_write(a.out,result)
    print(json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__':main()
