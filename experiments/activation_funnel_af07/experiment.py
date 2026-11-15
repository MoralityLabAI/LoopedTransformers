"""AF-07: recurrence-resolved subgoal locality under matched causal training.

All fitting reads only the parent AF-05 train/dev corpus. The original TypedLoop,
AF-05 graph model and Horn oracle are imported verbatim. Final checkpoints are
frozen before a separate evaluation command can access held-out cases.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.model import GraphLooper, parameter_count
from experiments.activation_funnel_af05 import experiment as parent
from experiments.activation_funnel_af06.core import mixed_targets, ramp_mask
from .core import paired_traces, locality_components, step_status_loss, trace_metrics

SEEDS=(17,29,43)
WIDTHS=(1,2)
ARMS=('status','ramp','locality')
STAGE_WARM='warmup'
WARM_STEPS=480
CONT_STEPS=160
BATCH_SIZE=32
LR=.002
WD=.01
EVAL_LOOPS=parent.EVAL_LOOPS
PINNED_MANIFEST=Path(__file__).with_name('corpus_manifest.json')
sha_file=parent.sha_file
json_write=parent.json_write
jsonl_write=parent.jsonl_write
Dataset=parent.Dataset
metrics=parent.metrics
logits_ce=parent.logits_ce
score_counterfactual=parent.score_counterfactual


def validate_manifest():
    current=gym.manifest()
    pinned=json.loads(PINNED_MANIFEST.read_text())
    if pinned!=current or pinned!=json.loads(parent.PINNED_MANIFEST.read_text()):
        raise ValueError('AF-07 source, oracle, or frozen corpus mismatch')
    if gym.digest({k:v for k,v in pinned.items() if k!='manifest_sha256'})!=pinned['manifest_sha256']:
        raise ValueError('AF-07 corpus self-digest mismatch')
    return sha_file(PINNED_MANIFEST)


def source_receipt():
    root=Path(__file__).resolve().parents[2]
    paths=[Path(__file__).with_name(f) for f in
        ('experiment.py','core.py','verify.py','aggregate.py','PROTOCOL.json','corpus_manifest.json')]
    paths += [root/'experiments/activation_funnel_af06'/f for f in
              ('experiment.py','core.py','corpus_manifest.json')]
    paths += [root/'experiments/activation_funnel_af05'/f for f in
              ('gym.py','model.py','experiment.py','corpus_manifest.json')]
    paths += [root/'experiments/activation_funnel_af02/model.py',
              root/'experiments/activation_funnel/model.py',
              root/'experiments/lean_reasoner/train.py',
              root/'experiments/lean_reasoner/proof_data.py']
    return {str(p.relative_to(root)):sha_file(p) for p in paths}


def sample_batches(data,depth_indices,step,seed,width,phase):
    """Deterministic sampler with paired, depth-specific recurrent schedules."""
    generator=torch.Generator().manual_seed(seed+4000*width+(121 if phase=='warmup' else 329))
    control=torch.Generator().manual_seed(seed+4000*width+(791 if phase=='warmup' else 899))
    for t in range(step):
        for depth in (2,3):
            torch.randint(len(depth_indices[depth]),(BATCH_SIZE//2,),generator=generator)
        torch.randperm(BATCH_SIZE,generator=control)
    while True:
        chosen={depth:[depth_indices[depth][i] for i in torch.randint(
            len(depth_indices[depth]),(BATCH_SIZE//2,),generator=generator).tolist()]
            for depth in (2,3)}
        # The shared permutation controls the exact number of addressed patches.
        perm=torch.randperm(BATCH_SIZE,generator=control)
        yield chosen,perm


def split_tensor_predictions(model, ds):
    """Dev and train eligibility use the correct 4/6 native calls per depth."""
    model.eval()
    logits=torch.empty((len(ds.cases),3,2),dtype=torch.float32)
    with torch.no_grad():
        for depth,loops in ((2,4),(3,6)):
            indices=[i for i,case in enumerate(ds.cases) if case.depth==depth]
            if not indices:continue
            out=model(ds.get(indices),pre_loops=loops)
            record=torch.cat([out.subgoals,out.goal.unsqueeze(1)],1)
            logits[torch.tensor(indices)]=record
    return metrics(logits,ds.batch.targets),logits


def model_loss(model,receiver,donor,selected,hybrid,addressed,pre_loops,arm):
    # All arms run *exactly* two pre-stage and two post-stage trajectories.
    # The labels are the independently computed finite Horn oracle, never a
    # prediction of the current model. Every post-stage Q state is supervised.
    clean,patched=paired_traces(model,receiver,donor,selected,addressed,pre_loops)
    target=mixed_targets(receiver.targets,hybrid,addressed)
    loss=logits_ce(clean.goal,receiver.targets[:,2])
    loss=loss+.5*logits_ce(clean.initial_q,receiver.targets[:,:2])
    loss=loss+.25*logits_ce(clean.q,receiver.targets[:,:2])
    loss=loss+.10*(F.softmax(clean.q,-1)-F.softmax(clean.initial_q,-1).detach()).square().mean()
    loss=loss+.75*logits_ce(patched.goal,target[:,2])
    loss=loss+.30*logits_ce(patched.q,target[:,:2])
    # Identical per-step Q supervision for all comparators. The locality arm
    # alone adds the transport-locality loss, including every recurrent step.
    loss=loss+.15*(step_status_loss(clean,receiver.targets)+step_status_loss(patched,target))
    if arm=='locality':
        probability,code=locality_components(clean,patched,selected,addressed)
        loss=loss+.25*probability+.12*code
    return loss,patched


def train_phase(model, optimizer, data, depth_indices, seed,width, phase,steps,arm=None):
    if phase not in ('warmup','continuation'):
        raise ValueError('bad phase')
    torch.set_num_threads(2)
    model.train()
    generator=sample_batches(data,depth_indices,0,seed,width,phase)
    history=[]; first_cell_grad=None
    num_addressed=0
    t0=time.perf_counter()
    for step in range(steps):
        depth_batches,permutation=next(generator)
        if phase=='warmup' or arm=='status':
            mask=torch.zeros(BATCH_SIZE,dtype=torch.bool)
        elif arm in ('ramp','locality'):
            mask=ramp_mask(BATCH_SIZE,step,steps,permutation)
        else:raise ValueError('unknown continuation arm')
        num_addressed+=int(mask.sum())
        optimizer.zero_grad(set_to_none=True)
        loss_total=None
        for group_i,(depth,loops) in enumerate(((2,4),(3,6))):
            selected_cases=depth_batches[depth]
            selectors=torch.tensor([(i+step+group_i)%2 for i in range(BATCH_SIZE//2)],dtype=torch.long)
            recipient=data.get(selected_cases)
            donor,hybrid=data.pair(selected_cases,selectors.tolist())
            loss,_=model_loss(model,recipient,donor,selectors,hybrid,
                              mask[group_i*(BATCH_SIZE//2):(group_i+1)*(BATCH_SIZE//2)],loops,arm or 'status')
            weighted=loss*.5
            loss_total=weighted if loss_total is None else loss_total+weighted
            weighted.backward()
        if first_cell_grad is None:
            first_cell_grad=float(sum(p.grad.detach().square().sum().item()
                        for p in model.cell.parameters() if p.grad is not None)**.5)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step%40==0 or step==steps-1:
            history.append([step,float(loss_total.detach()),int(mask.sum())])
    return {'steps':steps,'loss_history':history,'first_native_grad_l2':first_cell_grad,
            'addressed_rows':num_addressed,'total_patch_rows':steps*BATCH_SIZE,
            'elapsed_seconds':time.perf_counter()-t0}


def _save_model(out,name,model,seed,width,phase,warm_steps,cont_steps,file_roster):
    package={'model':model.state_dict(),'seed':seed,'width':width,'arm':name,'phase':phase,
             'warm_steps':warm_steps,'cont_steps':cont_steps,
             'sources':source_receipt(),'corpus_sha256':sha_file(PINNED_MANIFEST)}
    pt=out/f'{name}.pt';npz=out/f'{name}.npz'
    torch.save(package,pt)
    np.savez(npz,**{key:t.detach().cpu().numpy() for key,t in model.state_dict().items()})
    file_roster[pt.name]=sha_file(pt);file_roster[npz.name]=sha_file(npz)


def train_one(out,seed,width,warm_steps=WARM_STEPS,cont_steps=CONT_STEPS,smoke=False):
    if seed not in SEEDS or width not in WIDTHS or (not smoke and (warm_steps,cont_steps)!=(WARM_STEPS,CONT_STEPS)):
        raise ValueError('different run from frozen AF-07 protocol')
    if warm_steps<1 or cont_steps<2:raise ValueError('insufficient schedule length')
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('fresh output required')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(seed+1000*width);torch.manual_seed(seed+1000*width)
    manifest=validate_manifest()
    train,dev=Dataset('train'),Dataset('dev')
    depth_indices={d:[i for i,c in enumerate(train.cases) if c.depth==d] for d in (2,3)}
    if {d:len(v) for d,v in depth_indices.items()}!={2:336,3:336}:
        raise ValueError('modified training depth coverage')
    warm_model=GraphLooper(width)
    optimizer=torch.optim.AdamW(warm_model.parameters(),lr=LR,weight_decay=WD)
    warm_log=train_phase(warm_model,optimizer,train,depth_indices,seed,width,'warmup',warm_steps)
    frozen={key:v.clone().detach() for key,v in warm_model.state_dict().items()}
    pretrain_scores={split:split_tensor_predictions(warm_model,ds)[0]
                     for split,ds in (('train',train),('dev',dev))}
    file_roster={}
    _save_model(out,STAGE_WARM,warm_model,seed,width,'warmup',warm_steps,cont_steps,file_roster)
    fit={'warmup':{'fit_metrics':pretrain_scores,**warm_log}}
    rows=[]
    for name in (STAGE_WARM,)+ARMS:
        if name==STAGE_WARM:
            evaluated=warm_model
        else:
            evaluated=GraphLooper(width)
            evaluated.load_state_dict(copy.deepcopy(frozen))
            assert all(torch.equal(value,frozen[key]) for key,value in evaluated.state_dict().items())
            optimizer=torch.optim.AdamW(evaluated.parameters(),lr=LR,weight_decay=WD)
            run_log=train_phase(evaluated,optimizer,train,depth_indices,seed,width,
                                'continuation',cont_steps,arm=name)
            scores={split:split_tensor_predictions(evaluated,ds)[0]
                    for split,ds in (('train',train),('dev',dev))}
            fit[name]={'fit_metrics':scores,'starts_from_shared_warmup':True,**run_log}
            _save_model(out,name,evaluated,seed,width,'continuation',warm_steps,cont_steps,file_roster)
        for split,ds in (('train',train),('dev',dev)):
            score,logits=split_tensor_predictions(evaluated,ds)
            for case,target,values in zip(ds.cases,ds.batch.targets.tolist(),logits.tolist()):
                rows.append({'seed':seed,'width':width,'arm':name,'split':split,
                             'id':case.id,'target':target,'logits':values,
                             'pre_loops':4 if case.depth==2 else 6})
        print(json.dumps({'stage':'fit','seed':seed,'width':width,'arm':name,
            'dev':fit[name]['fit_metrics']['dev']}),flush=True)
    jsonl_write(out/'train_dev_predictions.jsonl',rows)
    file_roster['train_dev_predictions.jsonl']=sha_file(out/'train_dev_predictions.jsonl')
    rec={'experiment':'AF-07','seed':seed,'width':width,'warm_steps':warm_steps,
        'cont_steps':cont_steps,'engineering_smoke':smoke,'arms':fit,'sources':source_receipt(),
        'corpus_sha256':manifest,'files':file_roster,
        'train_dev_predictions':len(rows),'fit_splits':['train','dev']}
    json_write(out/'train_receipt.json',rec)
    (out/'train_receipt.sha256').write_text(sha_file(out/'train_receipt.json')+'\n')
    return rec


def load_frozen(out,seed,width,smoke=False):
    out=Path(out)
    if sha_file(out/'train_receipt.json')!=(out/'train_receipt.sha256').read_text().strip():
        raise ValueError('training receipt checksum mismatch')
    rec=json.loads((out/'train_receipt.json').read_text())
    if (rec['seed'],rec['width'],rec['engineering_smoke'])!=(seed,width,smoke):
        raise ValueError('wrong training identity')
    if rec['corpus_sha256']!=validate_manifest() or rec['sources']!=source_receipt():
        raise ValueError('source changed after training')
    if set(rec['arms'])!=set((STAGE_WARM,)+ARMS):
        raise ValueError('model arm roster changed')
    if rec['train_dev_predictions']!=len((STAGE_WARM,)+ARMS)*(672+224):
        raise ValueError('missing train/dev predictions')
    for name,sha in rec['files'].items():
        if sha_file(out/name)!=sha:raise ValueError('corrupted checkpoint/predictions: '+name)
    models={}
    for name in (STAGE_WARM,)+ARMS:
        pkg=torch.load(out/f'{name}.pt',map_location='cpu',weights_only=True)
        if (pkg['seed'],pkg['width'],pkg['arm'],pkg['corpus_sha256'])!=(seed,width,name,rec['corpus_sha256']):
            raise ValueError('checkpoint metadata mismatch')
        if pkg['sources']!=source_receipt():raise ValueError('checkpoint source pin mismatch')
        model=GraphLooper(width)
        model.load_state_dict(pkg['model']);model.eval()
        with np.load(out/f'{name}.npz',allow_pickle=False) as saved:
            if set(saved.files)!=set(model.state_dict()):raise ValueError('incomplete named-array exports')
            if any(not np.array_equal(saved[k],v.detach().cpu().numpy())
                   for k,v in model.state_dict().items()):raise ValueError('array weights mismatch')
        models[name]=model
    return rec,models


def trajectory_records(model, ds, pre_loops, post_loops=3, chunk=64):
    """Retain exact per-step logits, protected Q drift and all donor identities."""
    rows=[]
    total=len(ds.cases)*2
    for first in range(0,total,chunk):
        ids=list(range(first,min(first+chunk,total)))
        indices=[k//2 for k in ids]
        selects=[k%2 for k in ids]
        receiver=ds.get(indices)
        donor,composed=ds.pair(indices,selects)
        selected=torch.tensor(selects,dtype=torch.long)
        addressed=torch.ones(len(selected),dtype=torch.bool)
        with torch.no_grad():
            clean,patch=paired_traces(model,receiver,donor,selected,addressed,pre_loops,post_loops)
            independent=model(receiver,pre_loops=pre_loops) if post_loops==3 else None
            if independent is not None:
                err=max((clean.goal-independent.goal).abs().max().item(),
                        (clean.q-independent.subgoals).abs().max().item())
                if err>2e-5:raise RuntimeError('clean trajectory source replay differs')
            rec=trace_metrics(clean,patch,receiver.targets,composed,selected,addressed)
        for j,i in enumerate(indices):
            donor_idx,_=ds.donors[i,selects[j]]
            rows.append({
                'recipient_id':ds.cases[i].id,'donor_id':ds.cases[donor_idx].id,
                'selected':selects[j],'pre_loops':pre_loops,'post_loops':post_loops,
                'oracle':composed[j].tolist(),'recipient_truth':receiver.targets[j].tolist(),
                'clean_q_steps':clean.q_steps[j].tolist(),
                'patched_q_steps':patch.q_steps[j].tolist(),
                'patched_goal_steps':patch.goal_steps[j].tolist(),
                'protected_q_correct_by_step':rec['protected_q_correct_by_step'][j],
                'patched_q_correct_by_step':rec['patched_q_correct_by_step'][j],
                'protected_code_l2_by_step':rec['protected_code_l2_by_step'][j],
            })
    return rows


def trajectory_summary(rows):
    if not rows:raise ValueError('empty stage trace roster')
    steps=rows[0]['post_loops']
    if any(r['post_loops']!=steps for r in rows):
        raise ValueError('mixed trace horizons')
    return {'pairs':len(rows), 'post_loops':steps,
            'protected_q_step_accuracy':[sum(r['protected_q_correct_by_step'][k] for r in rows)/len(rows)
                                          for k in range(steps)],
            'patched_q_pair_exact_by_step':[sum(all(r['patched_q_correct_by_step'][k]) for r in rows)/len(rows)
                                            for k in range(steps)],
            'mean_protected_code_l2_by_step':[sum(r['protected_code_l2_by_step'][k] for r in rows)/len(rows)
                                              for k in range(steps)]}

def evaluate_one(training,out,seed,width,smoke=False):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('fresh evaluation output required')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    rec,models=load_frozen(training,seed,width,smoke)
    splits=('dev',) if smoke else ('held_composition','held_layout','held_depth4','held_depth5')
    result={'experiment':'AF-07','seed':seed,'width':width,'smoke':smoke,
            'source_train_receipt':sha_file(Path(training)/'train_receipt.json'),
            'source':source_receipt(),'training':rec['arms'],
            'scores':{},'patches':{},'trajectories':{},'shared_warmup_metrics':rec['arms']['warmup']['fit_metrics']}
    rows=[];patchrows=[];tracerows=[]
    for arm,model in models.items():
        if arm==STAGE_WARM:continue
        result['scores'][arm]={};result['patches'][arm]={};result['trajectories'][arm]={}
        for split in splits:
            ds=Dataset(split)
            loops=EVAL_LOOPS[split]
            with torch.no_grad():
                clean,lg=parent.get_predictions(model,ds,pre_loops=loops)
                patch,records=score_counterfactual(model,ds,pre_loops=loops)
            if patch['whole_direct_max_logit_error']>2e-5:raise RuntimeError('whole-donor mismatch')
            result['scores'][arm][split]=clean;result['patches'][arm][split]=patch
            result['trajectories'][arm][split]={}
            for post_steps in ((3,5) if split in ('held_depth4','held_depth5') and not smoke else (3,)):
                trows=trajectory_records(model,ds,pre_loops=loops,post_loops=post_steps)
                if post_steps==3:
                    pmap={(r['recipient_id'],r['selected']):r for r in records}
                    for tr in trows:
                        old=pmap[tr['recipient_id'],tr['selected']]
                        predicted=torch.tensor(tr['patched_q_steps'][-1]+[tr['patched_goal_steps'][-1]])
                        if not torch.allclose(predicted,torch.tensor(old['addressed_logits']),atol=2e-5,rtol=0):
                            raise RuntimeError('trajectory and AF-05 patch outputs disagree')
                result['trajectories'][arm][split][str(post_steps)]=trajectory_summary(trows)
                tracerows.extend({'seed':seed,'width':width,'arm':arm,'split':split,**r} for r in trows)
            for c,target,logits in zip(ds.cases,ds.batch.targets.tolist(),lg.tolist()):
                rows.append({'seed':seed,'width':width,'arm':arm,'split':split,'id':c.id,
                    'pre_loops':loops,'target':target,'logits':logits})
            for r in records:
                patchrows.append({'seed':seed,'width':width,'arm':arm,'split':split,**r})
    jsonl_write(out/'clean_predictions.jsonl',rows)
    jsonl_write(out/'patch_predictions.jsonl',patchrows)
    jsonl_write(out/'trajectory_predictions.jsonl',tracerows)
    result['clean_rows']=len(rows);result['patch_rows']=len(patchrows);result['trajectory_rows']=len(tracerows)
    result['files']={p.name:sha_file(p) for p in
         (out/'clean_predictions.jsonl',out/'patch_predictions.jsonl',out/'trajectory_predictions.jsonl')}
    json_write(out/'evaluation.json',result)
    (out/'evaluation.sha256').write_text(sha_file(out/'evaluation.json')+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['manifest','lean','train','evaluate'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=gym.SPLITS,default='held_depth5')
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--width',type=int,default=2)
    p.add_argument('--warm-steps',type=int,default=WARM_STEPS)
    p.add_argument('--cont-steps',type=int,default=CONT_STEPS)
    p.add_argument('--training',type=Path)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.command=='manifest':
        a.out.parent.mkdir(parents=True,exist_ok=True);json_write(a.out,gym.manifest())
    elif a.command=='lean':
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(gym.lean_certificate(a.split))
    elif a.command=='train':train_one(a.out,a.seed,a.width,a.warm_steps,a.cont_steps,a.smoke)
    elif a.command=='evaluate':
        if a.training is None:p.error('evaluate requires --training')
        evaluate_one(a.training,a.out,a.seed,a.width,a.smoke)

if __name__=='__main__':main()
