"""AF-05 train/score with frozen case manifests, exact interchange oracle and controls.

Run `python -m experiments.activation_funnel_af05.experiment ...` from root.
Fit reads train + dev only; held-out data is accessed solely by `evaluate`.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import gym
from .model import GraphLooper, parameter_count

ARMS = ('task', 'status', 'causal')
SEEDS = (17, 29, 43)
WIDTHS = (1, 2)
STEPS = 240
EVAL_LOOPS = {'held_composition': 6, 'held_layout': 6, 'held_depth4': 8, 'held_depth5': 10, 'dev': 6}
BATCH_SIZE = 32
LR = .002
WD = .01
WEIGHTS = {'q_pre': .5, 'q_post': .25, 'patch_goal': .75, 'patch_subgoal': .3}
PINNED_MANIFEST = Path(__file__).with_name('corpus_manifest.json')


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, x):
    Path(path).write_text(json.dumps(x, sort_keys=True, indent=2) + '\n')


def jsonl_write(path, rows):
    Path(path).write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))


def validate_manifest():
    receipt = gym.manifest()
    fixed = json.loads(PINNED_MANIFEST.read_text())
    if fixed != receipt or gym.digest({k:v for k,v in fixed.items() if k!='manifest_sha256'}) != fixed['manifest_sha256']:
        raise ValueError('AF-05 corpus receipt changed; abort before fitting')
    return sha_file(PINNED_MANIFEST)


def source_receipt():
    root=Path(__file__).resolve().parents[2]
    paths = [Path(__file__), Path(__file__).with_name('gym.py'),
             Path(__file__).with_name('model.py'),
             Path(__file__).with_name('verify.py'),
             Path(__file__).with_name('aggregate.py'),
             Path(__file__).with_name('PROTOCOL.json'), PINNED_MANIFEST,
             root/'experiments/lean_reasoner/train.py',
             root/'experiments/lean_reasoner/proof_data.py',
             root/'experiments/activation_funnel/model.py',
             root/'experiments/activation_funnel_af02/model.py']
    return {str(p.relative_to(root) if p.is_relative_to(root) else p):sha_file(p) for p in paths}


class Dataset:
    def __init__(self, split):
        self.cases = gym.cases_for(split)
        self.batch = gym.stack_cases(self.cases)
        self.donors = gym.donors_for(split)

    def get(self, indices):
        ids = torch.tensor(indices, dtype=torch.long)
        return gym.GraphBatch(*(getattr(self.batch, f.name).index_select(0, ids)
                                for f in fields(gym.GraphBatch)))

    def pair(self, indices, choices):
        ids, expected = [], []
        for i, select in zip(indices, choices):
            j, truth = self.donors[i,int(select)]
            ids.append(j); expected.append(truth)
        return self.get(ids), torch.tensor(expected, dtype=torch.long)


def logits_ce(logits, targets):
    return F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))


def metrics(answers, labels):
    pred = answers.argmax(-1)
    correct = pred == labels
    return {'n':int(len(labels)), 'goal_accuracy':float(correct[:, 2].float().mean()),
            'subgoal_exact':float(correct[:, :2].all(-1).float().mean()),
            'all_exact':float(correct.all(-1).float().mean()),
            'per_subgoal': [float(correct[:,j].float().mean()) for j in (0,1)],
            'false_goal_attempts':int(((pred[:, 2] == 1) & (labels[:, 2] == 0)).sum())}


def get_predictions(model, data, pre_loops=6):
    model.eval()
    with torch.no_grad():
        out = model(data.batch, pre_loops=pre_loops)
        concat = torch.cat([out.subgoals, out.goal.unsqueeze(1)], 1)
        score = metrics(concat, data.batch.targets)
    return score, concat.detach().cpu()


def train_one(out, seed, width, steps=STEPS, smoke=False):
    if seed not in SEEDS or width not in WIDTHS or (steps != STEPS and not smoke):
        raise ValueError('training configuration differs from frozen protocol')
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('require fresh output for immutable AF-05 run')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    validate_manifest()
    data, dev = Dataset('train'), Dataset('dev')
    training_depths = {depth:[i for i,c in enumerate(data.cases) if c.depth == depth] for depth in (2,3)}
    assert all(len(v) == 336 for v in training_depths.values())
    logs, rows, file_roster, receipts = {}, [], {}, {}
    baseline = None
    for arm in ARMS:
        random.seed(seed + 1000 * width)
        torch.manual_seed(seed + 1000 * width)
        model = GraphLooper(width)
        if baseline is None:
            baseline = {k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            assert all(torch.equal(v, baseline[k]) for k,v in model.state_dict().items())
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        rng = torch.Generator().manual_seed(seed + width * 4000 + 123)
        series=[]
        first_native_grad = None
        t0=time.perf_counter()
        for step in range(steps):
            selected = [training_depths[depth][j] for depth in (2, 3)
                        for j in torch.randint(len(training_depths[depth]), (BATCH_SIZE // 2,), generator=rng).tolist()]
            pre_loops = 4 if step % 2 == 0 else 6
            selection = [(i+step)%2 for i in range(BATCH_SIZE)]
            receiver = data.get(selected)
            donor, hybrid_truth = data.pair(selected,selection)
            which = torch.tensor(selection,dtype=torch.long)
            clean=model(receiver, pre_loops=pre_loops)
            # Every arm runs two graph prefixes and an extra suffix per update.
            # The status comparator patches an irrelevant branch and trains on
            # unchanged labels, giving the same number of cell calls as causal.
            patch_condition = 'addressed' if arm == 'causal' else ('other' if arm == 'status' else 'self')
            swapped=model.patched(receiver,donor,which,condition=patch_condition,pre_loops=pre_loops)
            loss=logits_ce(clean.goal,receiver.targets[:,2])
            if arm != 'task':
                loss=loss+WEIGHTS['q_pre']*logits_ce(clean.initial_subgoals,receiver.targets[:,:2])
                loss=loss+WEIGHTS['q_post']*logits_ce(clean.subgoals,receiver.targets[:,:2])
                # Shared depth-commutation regularizer; no extra native forwards.
                loss=loss+.10*(F.softmax(clean.subgoals,-1)-F.softmax(clean.initial_subgoals,-1).detach()).square().mean()
            patch_truth = hybrid_truth if arm == 'causal' else receiver.targets
            loss=loss+WEIGHTS['patch_goal']*logits_ce(swapped.goal,patch_truth[:,2])
            if arm != 'task':
                loss=loss+WEIGHTS['patch_subgoal']*logits_ce(swapped.subgoals,patch_truth[:,:2])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if first_native_grad is None:
                first_native_grad = float(sum(p.grad.detach().square().sum().item()
                                               for p in model.cell.parameters() if p.grad is not None)**.5)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if step%40==0 or step==steps-1:
                series.append([step,float(loss.detach())])
        scores={}
        for split, ds in [('train',data),('dev',dev)]:
            score, logit=get_predictions(model,ds)
            scores[split]=score
            for case,label,values in zip(ds.cases,ds.batch.targets.tolist(),logit.tolist()):
                rows.append({'seed':seed,'width':width,'arm':arm,'split':split,
                             'id':case.id,'target':label,'logits':values})
        payload={'model':model.state_dict(),'seed':seed,'width':width,'arm':arm,
                 'steps':steps,'source':source_receipt(),'corpus_sha256':sha_file(PINNED_MANIFEST)}
        file_pt=out/f'{arm}.pt';torch.save(payload,file_pt)
        file_npz=out/f'{arm}.npz'
        np.savez(file_npz, **{k:v.detach().cpu().numpy()
                            for k,v in model.state_dict().items()})
        for p in (file_pt,file_npz):file_roster[p.name]=sha_file(p)
        receipts[arm]={'seed':seed,'width':width,'arm':arm,'steps':steps,'loss_history':series,
                       'first_native_grad_l2':first_native_grad,
                       'train_prefix_steps': [4, 6], 'balanced_train_depths': [2, 3],
                       'resources':parameter_count(model), 'fit_metrics':scores,
                       'elapsed_seconds':time.perf_counter()-t0}
        print(json.dumps({'event':'trained','seed':seed,'width':width,'arm':arm,
                          'dev':scores['dev'],'seconds':receipts[arm]['elapsed_seconds']}),flush=True)
    preds=out/'train_dev_predictions.jsonl';jsonl_write(preds,rows)
    file_roster[preds.name]=sha_file(preds)
    rec={'experiment':'AF-05','seed':seed,'width':width,'steps':steps,
         'engineering_smoke':smoke,'arms':receipts,'sources':source_receipt(),
         'corpus_sha256':sha_file(PINNED_MANIFEST),'files':file_roster,
         'train_dev_predictions':len(rows),'eligible_splits':['train','dev']}
    json_write(out/'train_receipt.json',rec)
    (out/'train_receipt.sha256').write_text(sha_file(out/'train_receipt.json')+'\n')
    return rec


def load_frozen(out, seed, width, smoke=False):
    out=Path(out)
    if sha_file(out/'train_receipt.json') != (out/'train_receipt.sha256').read_text().strip():
        raise ValueError('tampered training receipt')
    rec=json.loads((out/'train_receipt.json').read_text())
    if (rec['seed'],rec['width'],rec['engineering_smoke']) != (seed,width,smoke):
        raise ValueError('training identifier mismatch')
    if rec['corpus_sha256'] != validate_manifest() or rec['sources'] != source_receipt():
        raise ValueError('source or dataset changed after training')
    for name,sha in rec['files'].items():
        if sha_file(out/name)!=sha:raise ValueError('changed artifact '+name)
    result={}
    for arm in ARMS:
        payload=torch.load(out/f'{arm}.pt',map_location='cpu',weights_only=True)
        if any(payload[k]!=rec[k] for k in ('seed','width','steps')) or payload['arm']!=arm:
            raise ValueError('checkpoint header mismatch')
        model=GraphLooper(width)
        model.load_state_dict(payload['model']);model.eval()
        with np.load(out/f'{arm}.npz',allow_pickle=False) as arr:
            if set(arr.files)!=set(model.state_dict()):raise ValueError('named-array roster mismatch')
            for k,v in model.state_dict().items():
                if not np.array_equal(v.numpy(),arr[k]):raise ValueError('named array differs: '+k)
        result[arm]=model
    return rec,result


def score_counterfactual(model, data, chunk=128, retain=True, pre_loops=6):
    rows=[]; max_whole=0.
    labels_all=[]; pred_addr=[]; pred_other=[]; pred_whole=[]
    for start in range(0,len(data.cases)*2,chunk):
        ordered=[(i//2,i%2) for i in range(start,min(len(data.cases)*2,start+chunk))]
        idx=[t[0] for t in ordered]; selects=[t[1] for t in ordered]
        receiver=data.get(idx)
        donor, oracle_target = data.pair(idx,selects)
        which=torch.tensor(selects,dtype=torch.long)
        with torch.no_grad():
            addr=model.patched(receiver,donor,which,condition='addressed',pre_loops=pre_loops)
            other=model.patched(receiver,donor,which,condition='other',pre_loops=pre_loops)
            whole=model.patched(receiver,donor,which,condition='whole',pre_loops=pre_loops)
            direct=model(donor,pre_loops=pre_loops)
            fullmax=max((whole.goal-direct.goal).abs().max().item(),
                        (whole.subgoals-direct.subgoals).abs().max().item())
            max_whole=max(max_whole,fullmax)
            addressed=torch.cat((addr.subgoals,addr.goal.unsqueeze(1)),1)
            otherlog=torch.cat((other.subgoals,other.goal.unsqueeze(1)),1)
            wholelog=torch.cat((whole.subgoals,whole.goal.unsqueeze(1)),1)
        labels_all.append(oracle_target)
        pred_addr.append(addressed.cpu())
        pred_other.append(otherlog.cpu())
        pred_whole.append(wholelog.cpu())
        if retain:
            for j,(case_i,sel) in enumerate(ordered):
                case=data.cases[case_i]
                donor_i,_=data.donors[case_i,sel]
                rows.append({'recipient_id':case.id,'donor_id':data.cases[donor_i].id,
                             'selected':sel,'pre_loops':pre_loops, 'oracle':oracle_target[j].tolist(),
                             'recipient_truth':receiver.targets[j].tolist(),
                             'addressed_logits':addressed[j].tolist(),
                             'other_logits':otherlog[j].tolist(),
                             'whole_logits':wholelog[j].tolist(),
                             'patched_subgoal_code':addr.pre.az[j,receiver.read_slots[j,sel]].tolist(),
                             'whole_replay_error_max':fullmax})
    expected=torch.cat(labels_all)
    ans=torch.cat(pred_addr).argmax(-1)
    other=torch.cat(pred_other).argmax(-1)
    selected=torch.tensor([i%2 for i in range(len(data.cases)*2)])
    otheridx=1-selected
    orig=data.batch.targets.repeat_interleave(2,0)
    semantic_change=expected[:,2]!=orig[:,2]
    good=ans==expected
    neg=other==orig
    summary={'pairs':len(expected), 'informative_goal_pairs':int(semantic_change.sum()),
             'addressed_subgoal_accuracy':float(good.gather(1,selected[:,None]).float().mean()),
             'untouched_subgoal_accuracy':float(good.gather(1,otheridx[:,None]).float().mean()),
             'addressed_goal_accuracy':float(good[:,2].float().mean()),
             'informative_goal_accuracy':float(good[semantic_change,2].float().mean()),
             'other_patch_goal_retention':float(neg[:,2].float().mean()),
             'other_patch_subgoal_exact':float(neg[:,:2].all(-1).float().mean()),
             'whole_direct_max_logit_error':max_whole,
             'observed_changed_goal':int(semantic_change.sum())}
    return summary,rows


def evaluate_one(training,out,seed,width,smoke=False):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('require fresh evaluation output')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    rec,models=load_frozen(training,seed,width,smoke)
    splits=['held_composition','held_layout','held_depth4','held_depth5'] if not smoke else ['dev']
    result={'seed':seed,'width':width,'smoke':smoke,'source_train_receipt':sha_file(Path(training)/'train_receipt.json'),
            'source':source_receipt(),'training':rec['arms'],'scores':{},'patches':{}}
    rows=[];patchrows=[]
    for arm,model in models.items():
        result['scores'][arm]={};result['patches'][arm]={}
        for split in splits:
            ds=Dataset(split)
            loops=EVAL_LOOPS[split]
            clean,lg=get_predictions(model,ds,pre_loops=loops)
            patch,ps=score_counterfactual(model,ds,pre_loops=loops)
            if patch['whole_direct_max_logit_error'] > 2e-5:
                raise RuntimeError('whole-state/direct-donor replay failure')
            result['scores'][arm][split]=clean
            result['patches'][arm][split]=patch
            for c,truth,logit in zip(ds.cases,ds.batch.targets.tolist(),lg.tolist()):
                rows.append({'seed':seed,'width':width,'arm':arm,'split':split,
                             'id':c.id,'pre_loops':loops,'target':truth,'logits':logit})
            for x in ps:patchrows.append({'seed':seed,'width':width,'arm':arm,'split':split,**x})
        print(json.dumps({'event':'evaluated','seed':seed,'width':width,'arm':arm,
                          'splits':result['scores'][arm]}),flush=True)
    jsonl_write(out/'clean_predictions.jsonl',rows)
    jsonl_write(out/'patch_predictions.jsonl',patchrows)
    result['clean_rows']=len(rows);result['patch_rows']=len(patchrows)
    result['files']={p.name:sha_file(p) for p in
                     (out/'clean_predictions.jsonl',out/'patch_predictions.jsonl')}
    json_write(out/'evaluation.json',result)
    (out/'evaluation.sha256').write_text(sha_file(out/'evaluation.json')+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['manifest','lean','train','evaluate'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=gym.SPLITS,default='held_depth')
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--width',type=int,default=2)
    p.add_argument('--steps',type=int,default=STEPS)
    p.add_argument('--training',type=Path)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.command=='manifest':
        a.out.parent.mkdir(parents=True,exist_ok=True);json_write(a.out,gym.manifest())
    elif a.command=='lean':
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(gym.lean_certificate(a.split))
    elif a.command=='train':train_one(a.out,a.seed,a.width,a.steps,a.smoke)
    else:
        if not a.training:p.error('evaluate requires --training')
        evaluate_one(a.training,a.out,a.seed,a.width,a.smoke)

if __name__=='__main__':main()
