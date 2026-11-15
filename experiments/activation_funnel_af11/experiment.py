"""AF-11 preregistered Boolean-influence continuation of qualified AF-06 TypedLoop.

The original warm-up checkpoint is source-bound. A *shared* cut-adaptation
checkpoint precedes all experimental branches. Frozen train/dev fit and a
separate evaluate command enforce the held-out boundary. No previous scored
held-out inputs enter the training objective or hyperparameter selection.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.model import GraphLooper
from experiments.activation_funnel_af05 import experiment as parent
from experiments.activation_funnel_af06 import experiment as af06
from experiments.activation_funnel_af09.experiment import source_from_zip
from experiments.activation_funnel_af07.core import trace_suffix
from experiments.activation_funnel_af10.core import latch_suffix
from . import data
from .core import influence_loss, replay, signed_goal_difference, soft_signed_delta

SEEDS=(17,29)
WIDTH=2
ARMS=('status_cut','counterfactual_cut','influence_cut','influence_ordinary')
ADAPT_STEPS=160
CONT_STEPS=160
BATCH=32
LR=.0008
WD=.01
FAMILIES=data.FRESH
PRE={2:4,3:6,4:8,5:10}
SOURCE={'17':'c8cf6c6c8264cf313e755c4d3008c26ae72cdd6ef20583fadc112bf02169ffdd',
        '29':'660b71acd5dc95bbfd200e55f206f609dcf6cc141ccdbb131d74bb7e5f0d4ebf'}
PROTO=Path(__file__).with_name('PROTOCOL.json')
FRESH_MANIFEST=Path(__file__).with_name('fresh_manifest.json')
CE=parent.logits_ce
sha_file=parent.sha_file
json_write=parent.json_write
jsonl_write=parent.jsonl_write


def frozen_sources():
    root=Path(__file__).resolve().parents[2]
    dirs={
      'activation_funnel_af11':('data.py','core.py','experiment.py','verify.py','aggregate.py','PROTOCOL.json','fresh_manifest.json'),
      'activation_funnel_af10':('core.py',),
      'activation_funnel_af09':('core.py','experiment.py','PROTOCOL.json'),
      'activation_funnel_af06':('core.py','experiment.py','PROTOCOL.json'),
      'activation_funnel_af05':('gym.py','model.py','experiment.py','corpus_manifest.json'),
      'activation_funnel_af02':('model.py',),
      'activation_funnel':('model.py',),
      'lean_reasoner':('train.py','proof_data.py')}
    p=[root/'experiments'/d/n for d,files in dirs.items() for n in files]
    return {str(x.relative_to(root)):sha_file(x) for x in p}


def verify_fresh_manifest():
    now=data.fresh_manifest()
    fixed=json.loads(FRESH_MANIFEST.read_text())
    if now!=fixed or gym.digest({k:v for k,v in fixed.items() if k!='manifest_sha256'})!=fixed['manifest_sha256']:
        raise ValueError('AF-11 frozen fresh-case receipt mismatch')
    return sha_file(FRESH_MANIFEST)


def check_source(artifact,seed):
    if seed not in SEEDS or sha_file(artifact)!=SOURCE[str(seed)]:
        raise ValueError('AF-11 qualified parent ZIP mismatch')
    return source_from_zip(artifact,seed)


def split_score(model,ds,*,cut):
    model.eval()
    logits=torch.empty(len(ds.cases),3,2)
    with torch.no_grad():
        for depth in (2,3):
            indices=[i for i,c in enumerate(ds.cases) if c.depth==depth]
            if not indices:continue
            for offset in range(0,len(indices),32):
                ix=indices[offset:offset+32]
                chunk=ds.get(ix)
                start=model.prefix(chunk,PRE[depth])
                out=(latch_suffix(model,chunk,start,3) if cut else trace_suffix(model,chunk,start,3))
                logits[ix]=torch.cat([out.q,out.goal.unsqueeze(1)],1).detach()
    return parent.metrics(logits,ds.batch.targets),logits


def qualifications(metrics_by_split,minimum=.90):
    return all(metrics_by_split[x]['goal_accuracy']>=minimum and
               metrics_by_split[x]['subgoal_exact']>=minimum for x in ('train','dev'))


def batch_stream(seed,offset=0):
    """One deterministic balanced sampler; continuation arms replay exact draws."""
    rng=torch.Generator().manual_seed(11011+seed+offset*101)
    train=data.Dataset('train')
    bins={d:[i for i,c in enumerate(train.cases) if c.depth==d] for d in (2,3)}
    if [len(bins[d]) for d in (2,3)]!=[336,336]:raise ValueError('train split changed')
    while True:
        for d in (2,3):
            ix=torch.randint(len(bins[d]),(BATCH//2,),generator=rng).tolist()
            indices=[bins[d][i] for i in ix]
            selected=torch.randint(2,(BATCH//2,),generator=rng)
            rec=train.get(indices)
            donors,composed=train.pair(indices,selected.tolist())
            gates=torch.tensor([int(train.cases[i].gate=='or') for i in indices])
            yield rec,donors,selected,composed,gates,d


def loss_for(model,rec,donor,selected,hybrid,gates,depth,arm):
    cut=arm!='influence_ordinary'
    pair=replay(model,rec,donor,selected,PRE[depth],cut=cut)
    y=rec.targets
    clean,patched,ctrl=pair.clean,pair.patched,pair.control
    loss=(CE(clean.goal,y[:,2])+.5*CE(clean.initial_q,y[:,:2])
         +.35*CE(clean.q,y[:,:2])
         +.20*CE(ctrl.goal,y[:,2])+.15*CE(ctrl.q,y[:,:2]))
    # A strong counterfactual control supervises the same exact donor target.
    # The status-only arm performs equal recurrent forwards but omits this term.
    if arm!='status_cut':
        loss=loss+.75*CE(patched.goal,hybrid[:,2])+.3*CE(patched.q,hybrid[:,:2])
    else:
        loss=loss+.2*CE(patched.q,hybrid[:,:2]) # status readout, without goal-influence supervision
    parts={'task':float(loss.detach())}
    if arm.startswith('influence_'):
        gap,margin,info=influence_loss(clean.goal,patched.goal,y,hybrid,selected,gates)
        loss=loss+.4*gap+.2*margin
        parts.update({'influence_mse':float(gap.detach()),'influence_margin':float(margin.detach()),**info})
    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite AF-11 loss')
    return loss,parts


def _train_steps(model,seed,steps,arm,stage):
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    stream=batch_stream(seed,offset=0 if stage=='shared' else 1)
    initial_grad=None;history=[];start=time.perf_counter()
    # Balanced depth2 and depth3 episodes per update. Exact forward roster
    # matches across arms for a given stage and seed, including donor sampling.
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        stats={}
        for _ in range(2):
            rec,donor,selected,hybrid,gate,depth=next(stream)
            loss,parts=loss_for(model,rec,donor,selected,hybrid,gate,depth,arm)
            (loss/2).backward()
            for k,v in parts.items():stats[k]=stats.get(k,0.)+v/2
        if initial_grad is None:
            initial_grad=float(sum(p.grad.square().sum().item() for p in model.cell.parameters()
                    if p.grad is not None)**.5)
            if initial_grad<=0:raise RuntimeError('native recurrent cell receives no gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError(f'nonfinite AF11 {arm} at update {step}')
        if step%40==0 or step==steps-1:history.append({'step':step,**stats})
    return {'updates':steps,'gradient_l2_first_step':initial_grad,
            'train_seconds':time.perf_counter()-start,'loss_history':history,
            'actual_native_calls_per_row':2*PRE[2]+3*3}


def save_model(out,name,model,seed,parent_sha,roster):
    weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    p=out/(name+'.pt');q=out/(name+'.npz')
    torch.save({'experiment':'AF-11','seed':seed,'width':WIDTH,'arm':name,
                'parent_warmup_sha256':parent_sha,'state_dict':weights},p)
    np.savez(q,**{k:v.numpy() for k,v in weights.items()})
    roster[p.name]=sha_file(p);roster[q.name]=sha_file(q)


def train(artifact,out,seed,adapt=ADAPT_STEPS,steps=CONT_STEPS,smoke=False):
    if seed not in SEEDS or (not smoke and (adapt,steps)!=(ADAPT_STEPS,CONT_STEPS)) or min(adapt,steps)<2:
        raise ValueError('nonregistered AF-11 training budget')
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('fresh output directory required')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    random.seed(13011+seed);torch.manual_seed(13011+seed)
    manifest_sha=verify_fresh_manifest()
    parent_receipt,original,parent_zip,warm_sha,parent_training_sha=check_source(artifact,seed)
    source_start={k:v.detach().clone() for k,v in original.state_dict().items()}
    warm=GraphLooper(WIDTH);warm.load_state_dict(source_start)
    shared=_train_steps(warm,seed,adapt,'counterfactual_cut','shared')
    shared_state={k:v.detach().clone() for k,v in warm.state_dict().items()}
    ds={n:data.Dataset(n) for n in ('train','dev')}
    results={};records=[];hashes={}
    shared_fit={n:split_score(warm,s,cut=True)[0] for n,s in ds.items()}
    save_model(out,'shared_cut',warm,seed,warm_sha,hashes)
    for arm in ARMS:
        model=GraphLooper(WIDTH);model.load_state_dict(shared_state)
        if not all(torch.equal(shared_state[k],v) for k,v in model.state_dict().items()):
            raise ValueError('continuation source differs')
        log=_train_steps(model,seed,steps,arm,'continuation')
        scores={}
        for n,s in ds.items():
            metric,logits=split_score(model,s,cut=arm!='influence_ordinary')
            scores[n]=metric
            for c,values,y in zip(s.cases,logits.tolist(),s.batch.targets.tolist()):
                records.append({'arm':arm,'split':n,'case_id':c.id,'logits':values,'truth':y})
        results[arm]={'fit_metrics':scores,**log}
        save_model(out,arm,model,seed,warm_sha,hashes)
        print(json.dumps({'stage':'continuation_complete','seed':seed,'arm':arm,
                'dev':scores['dev'],'elapsed':log['train_seconds']}),flush=True)
    jsonl_write(out/'train_dev_predictions.jsonl',records)
    hashes['train_dev_predictions.jsonl']=sha_file(out/'train_dev_predictions.jsonl')
    receipt={'experiment':'AF-11','seed':seed,'smoke':smoke,'adapt_updates':adapt,
             'continuation_updates':steps,'models':list(ARMS),
             'parent_training_sha256':parent_training_sha,'parent_archive_sha256':parent_zip,
             'original_warmup_sha256':warm_sha,
             'qualified_AF06_original':parent_receipt['arms']['warmup']['fit_metrics'],
             'shared_cut_fit_metrics':shared_fit,'shared_cut_log':shared,
             'arms':results,'files':hashes,'parent_source': 'AF-06',
             'fresh_manifest_sha256':manifest_sha,'source_hashes':frozen_sources(),
             'train_dev_predictions':len(records)}
    json_write(out/'train_receipt.json',receipt)
    (out/'train_receipt.sha256').write_text(sha_file(out/'train_receipt.json')+'\n')
    return receipt


def load_models(folder,seed,smoke=False):
    folder=Path(folder);receipt_path=folder/'train_receipt.json'
    if sha_file(receipt_path)!=(folder/'train_receipt.sha256').read_text().strip():
        raise ValueError('AF-11 altered receipt')
    rec=json.loads(receipt_path.read_text())
    if (rec['experiment'],rec['seed'],rec['smoke'])!=('AF-11',seed,smoke):
        raise ValueError('AF-11 training identity mismatch')
    if rec['fresh_manifest_sha256']!=verify_fresh_manifest() or rec['source_hashes']!=frozen_sources():
        raise ValueError('AF-11 source changed after training')
    if set(rec['files'])!=set([n+'.'+ext for n in ('shared_cut',)+ARMS for ext in ('pt','npz')]+['train_dev_predictions.jsonl']):
        raise ValueError('AF-11 frozen export roster mismatch')
    out={}
    for n in ('shared_cut',)+ARMS:
        p=folder/(n+'.pt');npz=folder/(n+'.npz')
        if any(sha_file(f)!=rec['files'][f.name] for f in (p,npz)):
            raise ValueError('AF-11 source weight digest mismatch')
        pkg=torch.load(p,map_location='cpu',weights_only=True)
        if (pkg['experiment'],pkg['seed'],pkg['arm'],pkg['parent_warmup_sha256'])!=('AF-11',seed,n,rec['original_warmup_sha256']):
            raise ValueError('AF-11 saved model identity mismatch')
        m=GraphLooper(WIDTH);m.load_state_dict(pkg['state_dict']);m.eval()
        with np.load(npz,allow_pickle=False) as f:
            if set(f.files)!=set(pkg['state_dict']) or any(not np.array_equal(
                   f[k],v.numpy()) for k,v in pkg['state_dict'].items()):
                raise ValueError('AF-11 Torch/NumPy export mismatch')
        if any(not torch.isfinite(v).all() for v in m.state_dict().values()):
            raise FloatingPointError('AF-11 nonfinite saved model')
        out[n]=m
    if sha_file(folder/'train_dev_predictions.jsonl')!=rec['files']['train_dev_predictions.jsonl']:
        raise ValueError('AF11 train/dev prediction digest mismatch')
    return rec,out


def evaluate(training,out,seed,smoke=False):
    training=Path(training);out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('AF-11 evaluation requires fresh output')
    out.mkdir(parents=True,exist_ok=True)
    rec,models=load_models(training,seed,smoke)
    torch.set_num_threads(2)
    rows=[];clean=[];summaries={}
    for arm in ARMS:
        model=models[arm];cut=arm!='influence_ordinary'
        summaries[arm]={}
        for split in (('dev',) if smoke else FAMILIES):
            ds=data.Dataset(split)
            n=len(ds.cases);counts={'n':n,'patch_rows':0,'informative':0,'correct_flip':0,
                 'unaffected_goal_retention':0,'unaffected_n':0,'patched_goal_correct':0,
                 'prefix_q_exact':0,'clean_goal_correct':0,'all_patched_q_exact':0,
                 'prefix_correct_informative':0,'prefix_correct_informative_success':0,
                 'signed_delta_abs_error_sum':0.,'false_goal':0,'patched_other_q_correct':0}
            for offset in range(0,n,16):
                indices=list(range(offset,min(offset+16,n)))
                for j in (0,1):
                    rx=ds.get(indices)
                    dx,hybrid=ds.pair(indices,[j]*len(indices))
                    choose=torch.full((len(indices),),j,dtype=torch.long)
                    gates=torch.tensor([int(ds.cases[i].gate=='or') for i in indices])
                    with torch.no_grad():
                        pair=replay(model,rx,dx,choose,PRE[ds.cases[indices[0]].depth],cut=cut)
                        delta,inf=signed_goal_difference(rx.targets,hybrid,choose,gates)
                        prob_delta=soft_signed_delta(pair.clean.goal,pair.patched.goal)
                        prefix=pair.clean.initial_q.argmax(-1)
                        goal=pair.patched.goal.argmax(-1)
                        other=1-choose
                        correct_prefix=(prefix==rx.targets[:,:2]).all(-1)
                        good_goal=(goal==hybrid[:,2]);effective=inf.bool()
                        otherq=pair.patched.q.argmax(-1).gather(1,other[:,None]).squeeze(-1)
                        othercorrect=otherq==rx.targets[:,:2].gather(1,other[:,None]).squeeze(-1)
                        clean_goal=pair.clean.goal.argmax(-1)==rx.targets[:,2]
                    counts['patch_rows']+=len(indices)
                    counts['informative']+=int(effective.sum());counts['correct_flip']+=int((effective&good_goal).sum())
                    counts['unaffected_n']+=int((~effective).sum())
                    counts['unaffected_goal_retention']+=int(((~effective)&good_goal).sum())
                    counts['patched_goal_correct']+=int(good_goal.sum());counts['prefix_q_exact']+=int(correct_prefix.sum())
                    counts['clean_goal_correct']+=int(clean_goal.sum())
                    counts['all_patched_q_exact']+=int((pair.patched.q.argmax(-1)==hybrid[:,:2]).all(-1).sum())
                    counts['prefix_correct_informative']+=int((correct_prefix&effective).sum())
                    counts['prefix_correct_informative_success']+=int((correct_prefix&effective&good_goal).sum())
                    counts['signed_delta_abs_error_sum']+=float((prob_delta-delta).abs().sum())
                    counts['false_goal']+=int(((goal==1)&(hybrid[:,2]==0)).sum())
                    counts['patched_other_q_correct']+=int(othercorrect.sum())
                    for k,i in enumerate(indices):
                        rows.append({'arm':arm,'split':split,'case_id':ds.cases[i].id,'selected':j,
                          'recipient':rx.targets[k].tolist(),'donor_case_id':ds.cases[ds.donors[i,j][0]].id,
                          'composed':hybrid[k].tolist(),'gate':int(gates[k]),
                          'influence':int(inf[k]),'goal_delta':int(delta[k]),
                          'clean_goal_logits':pair.clean.goal[k].tolist(),
                          'patch_goal_logits':pair.patched.goal[k].tolist(),
                          'control_goal_logits':pair.control.goal[k].tolist(),
                          'prefix_q_logits':pair.clean.initial_q[k].tolist(),
                          'patched_q_logits':pair.patched.q[k].tolist(),
                          'protected_cut_code_difference':float((pair.patched.q_codes[k]-pair.patched.q_codes[k,0:1]).abs().max()) if cut else None})
            counts['informative_flip_accuracy']=counts['correct_flip']/max(1,counts['informative'])
            counts['irrelevant_patch_goal_retention']=counts['unaffected_goal_retention']/max(1,counts['unaffected_n'])
            counts['overall_patched_goal_accuracy']=counts['patched_goal_correct']/counts['patch_rows']
            counts['clean_goal_accuracy']=counts['clean_goal_correct']/counts['patch_rows']
            counts['prefix_joint_q_accuracy']=counts['prefix_q_exact']/counts['patch_rows']
            counts['prefix_conditioned_flip_accuracy']=(counts['prefix_correct_informative_success']/max(1,counts['prefix_correct_informative']))
            counts['patched_other_q_accuracy']=counts['patched_other_q_correct']/counts['patch_rows']
            counts['signed_delta_mae']=counts['signed_delta_abs_error_sum']/counts['patch_rows']
            summaries[arm][split]=counts
            print(json.dumps({'phase':'evaluated','seed':seed,'arm':arm,'split':split,
                              'informative_goal_flip_accuracy':counts['informative_flip_accuracy']}),flush=True)
    jsonl_write(out/'counterfactuals.jsonl',rows)
    s={'experiment':'AF-11','seed':seed,'smoke':smoke,'fresh_manifest_sha256':rec['fresh_manifest_sha256'],
       'source_train_receipt_sha256':sha_file(training/'train_receipt.json'),
       'training_qualified':{a:qualifications(rec['arms'][a]['fit_metrics']) for a in ARMS},
       'shared_cut_qualified':qualifications(rec['shared_cut_fit_metrics']),
       'scores':summaries,'counterfactual_rows':len(rows),
       'counterfactual_sha256':sha_file(out/'counterfactuals.jsonl')}
    json_write(out/'evaluation.json',s)
    return s


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['train','evaluate'])
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--source-zip',type=Path)
    p.add_argument('--train',type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--adapt-steps',type=int,default=ADAPT_STEPS)
    p.add_argument('--steps',type=int,default=CONT_STEPS)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.command=='train':
        if not a.source_zip:p.error('--source-zip is required for training')
        train(a.source_zip,a.out,a.seed,a.adapt_steps,a.steps,a.smoke)
    else:
        if not a.train:p.error('--train required for evaluation')
        evaluate(a.train,a.out,a.seed,a.smoke)
if __name__=='__main__':main()
