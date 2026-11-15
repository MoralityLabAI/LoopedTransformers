"""AF-08: continuation from independently qualified AF-06 warm-up checkpoints.

The original AF-05 Horn oracle, AF-06 training receipts and AF-07 trajectory
code stay immutable. Fit reads train/dev; held-out cases are loaded only by the
separate evaluate command after fixed-step checkpoint export.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import time
import tempfile
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.model import GraphLooper,parameter_count
from experiments.activation_funnel_af05 import experiment as base
from experiments.activation_funnel_af06 import experiment as af06
from experiments.activation_funnel_af07 import experiment as af07
from experiments.activation_funnel_af07.core import locality_components, step_status_loss, trace_suffix
from experiments.activation_funnel_af07.experiment import trajectory_records, trajectory_summary
from .core import triple_traces, class_separation, selected_code_change, qualified, code_separation_scores

SEEDS=(17,29)
WIDTH=2
ARMS=('status','locality','noncollapse')
STEPS=160
BATCH_SIZE=32
LR=0.0008
WD=0.01
PINNED_MANIFEST=Path(__file__).with_name('corpus_manifest.json')
# These are exact GitHub Actions artifact ZIP SHA-256 values, not model pickle digests.
SOURCE_ARTIFACTS={
 17:{'id':14116102941,'sha256':'c8cf6c6c8264cf313e755c4d3008c26ae72cdd6ef20583fadc112bf02169ffdd'},
 29:{'id':14115887129,'sha256':'660b71acd5dc95bbfd200e55f206f609dcf6cc141ccdbb131d74bb7e5f0d4ebf'},
}
PARENT_COMMIT='36b83eb83bc116dd4e938213edeba3057ffa9016'
sha_file=base.sha_file
json_write=base.json_write
jsonl_write=base.jsonl_write
Dataset=base.Dataset
metrics=base.metrics
logits_ce=base.logits_ce
EVAL_LOOPS=base.EVAL_LOOPS


def source_receipt():
    root=Path(__file__).resolve().parents[2]
    dirs={
      'activation_funnel_af08':('core.py','experiment.py','verify.py','aggregate.py','PROTOCOL.json','corpus_manifest.json'),
      'activation_funnel_af07':('core.py','experiment.py','corpus_manifest.json'),
      'activation_funnel_af06':('core.py','experiment.py','corpus_manifest.json'),
      'activation_funnel_af05':('gym.py','model.py','experiment.py','corpus_manifest.json'),
      'activation_funnel_af02':('model.py',),
      'activation_funnel':('model.py',),
      'lean_reasoner':('train.py','proof_data.py')}
    paths=[root/'experiments'/d/n for d,ns in dirs.items() for n in ns]
    return {str(p.relative_to(root)):sha_file(p) for p in paths}


def validate_manifest():
    h=sha_file(PINNED_MANIFEST)
    m=json.loads(PINNED_MANIFEST.read_text())
    if m!=gym.manifest() or m!=json.loads(base.PINNED_MANIFEST.read_text()) or \
       gym.digest({k:v for k,v in m.items() if k!='manifest_sha256'})!=m['manifest_sha256']:
        raise ValueError('AF-08 exact frozen Horn corpus mismatch')
    return h


def source_from_zip(artifact,seed):
    """Validate complete AF-06 training artifact before loading executable weights."""
    artifact=Path(artifact)
    actual=sha_file(artifact)
    if actual!=SOURCE_ARTIFACTS[seed]['sha256']:
        raise ValueError('original AF-06 ZIP checksum mismatch')
    validate_manifest()
    with tempfile.TemporaryDirectory(prefix='af08-source-') as temp:
        path=Path(temp)
        with zipfile.ZipFile(artifact) as z:
            if z.testzip() is not None:raise ValueError('AF-06 ZIP member CRC failure')
            names=z.namelist()
            if len(names)!=14 or len(set(names))!=14 or any(
                n.startswith('/') or '..' in Path(n).parts for n in names):
                raise ValueError('AF-06 ZIP member roster mismatch')
            if z.read('source_revision.txt').decode().strip()!=PARENT_COMMIT:
                raise ValueError('AF-06 source commit mismatch')
            for member in names:
                if member.startswith('train/'):
                    dest=path/member;dest.parent.mkdir(exist_ok=True)
                    dest.write_bytes(z.read(member))
        rec,models=af06.load_frozen(path/'train',seed,WIDTH,smoke=False)
        status=rec['arms']['warmup']['fit_metrics']
        if not qualified(status,.90):
            raise ValueError('source warm-up is not qualified on train and dev')
        model=models['warmup']
        return rec,model,actual,sha_file(path/'train'/'warmup.pt'),sha_file(path/'train'/'train_receipt.json')


def split_scores(model,ds):
    return af06.split_tensor_predictions(model,ds)


def loss_for(model,anchor,receiver,donor,selected,hybrid,addressed,loops,arm):
    trio=triple_traces(model,receiver,donor,selected,addressed,loops)
    target=torch.where(addressed[:,None],hybrid,receiver.targets)
    loss=logits_ce(trio.clean.goal,receiver.targets[:,2])
    loss=loss+.5*logits_ce(trio.clean.initial_q,receiver.targets[:,:2])
    loss=loss+.25*logits_ce(trio.clean.q,receiver.targets[:,:2])
    loss=loss+.15*step_status_loss(trio.clean,receiver.targets)
    loss=loss+.75*logits_ce(trio.patched.goal,target[:,2])
    loss=loss+.30*logits_ce(trio.patched.q,target[:,:2])
    loss=loss+.15*step_status_loss(trio.patched,target)
    loss=loss+.20*logits_ce(trio.control.goal,receiver.targets[:,2])
    loss=loss+.05*step_status_loss(trio.control,receiver.targets)
    # Same weight-distance anchor in every continuation. An actually qualified
    # seed supplies the reference; it is not trained from held-out labels.
    model_dist=torch.stack([((p-ref)**2).mean() for p,ref in zip(model.parameters(),anchor)]).mean()
    loss=loss+.025*model_dist
    components={'base':loss.detach().item()}
    if arm in ('locality','noncollapse'):
        kl,code=locality_components(trio.clean,trio.patched,selected,addressed)
        loss=loss+.25*kl+.04*code
        components.update(kl=float(kl.detach()),code=float(code.detach()))
    if arm=='noncollapse':
        prototypes=class_separation(trio.clean.q_codes,receiver.targets[:,:2],margin=.45)
        changed=selected_code_change(trio,selected,receiver.targets,hybrid,margin=.35)
        loss=loss+.3*prototypes+.15*changed
        components.update(separation=float(prototypes.detach()),change=float(changed.detach()))
    return loss,components


def _train_continuation(model,anchor,data,depth_idx,seed,arm,steps):
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    gen=af07.sample_batches(data,depth_idx,0,seed,WIDTH,'continuation')
    history=[]; gradients=None; exposure=0; begin=time.perf_counter()
    model.train()
    for step in range(steps):
        selected,perm=next(gen)
        if arm=='status':mask=torch.zeros(BATCH_SIZE,dtype=torch.bool)
        else:mask=af07.ramp_mask(BATCH_SIZE,step,steps,perm)
        exposure+=int(mask.sum())
        optimizer.zero_grad(set_to_none=True)
        component_log={}
        for group,(depth,loops) in enumerate(((2,4),(3,6))):
            indices=selected[depth]
            targets=torch.tensor([(i+step+group)%2 for i in range(BATCH_SIZE//2)],dtype=torch.long)
            recipient=data.get(indices)
            donor,hybrid=data.pair(indices,targets.tolist())
            active=mask[group*16:(group+1)*16]
            loss,parts=loss_for(model,anchor,recipient,donor,targets,hybrid,active,loops,arm)
            (loss*.5).backward()
            for k,v in parts.items():component_log[k]=component_log.get(k,0.)+v*.5
        if gradients is None:
            gradients=float(sum(p.grad.detach().square().sum().item()
                                for p in model.cell.parameters() if p.grad is not None)**.5)
            if gradients<=0:raise RuntimeError('intervention gradient failed to reach native cell')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step%40==0 or step==steps-1:
            history.append({'step':step,'loss':component_log,'addressed':int(mask.sum())})
    return {'steps':steps,'first_native_cell_gradient_l2':gradients,
            'addressed_rows':exposure,'total_patch_rows':steps*BATCH_SIZE,
            'training_time_seconds':time.perf_counter()-begin,'loss_history':history}


def _save_model(folder,name,model,seed,source_sha,files):
    p=folder/f'{name}.pt';q=folder/f'{name}.npz'
    torch.save({'experiment':'AF-08','seed':seed,'width':WIDTH,'arm':name,
                'source_sha256':source_sha,'model':model.state_dict(),
                'source_hashes':source_receipt()},p)
    np.savez(q,**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    files[p.name]=sha_file(p);files[q.name]=sha_file(q)


def train_one(artifact,output,seed,steps=STEPS,smoke=False):
    if seed not in SEEDS or (not smoke and steps!=STEPS) or steps<2:
        raise ValueError('AF-08 frozen training settings mismatch')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('fresh AF-08 training output required')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(seed+8000);torch.manual_seed(seed+8000)
    rec,initial,zip_sha,warm_sha,receipt_sha=source_from_zip(artifact,seed)
    warm={k:v.detach().clone() for k,v in initial.state_dict().items()}
    anchor=[p.detach().clone() for p in initial.parameters()]
    datasets={s:Dataset(s) for s in ('train','dev')}
    depth_idx={d:[i for i,c in enumerate(datasets['train'].cases) if c.depth==d] for d in (2,3)}
    if {d:len(v) for d,v in depth_idx.items()}!={2:336,3:336}:
        raise ValueError('wrong training depth roster')
    file_roster={};history={};records=[]
    for arm in ARMS:
        model=GraphLooper(WIDTH);model.load_state_dict(warm)
        if not all(torch.equal(model.state_dict()[k],v) for k,v in warm.items()):
            raise ValueError('changed source before continuation')
        log=_train_continuation(model,anchor,datasets['train'],depth_idx,seed,arm,steps)
        fit={}
        for split,ds in datasets.items():
            score,logits=split_scores(model,ds)
            fit[split]=score
            for c,l,y in zip(ds.cases,logits.tolist(),ds.batch.targets.tolist()):
                records.append({'seed':seed,'width':WIDTH,'arm':arm,'split':split,
                                'id':c.id,'logits':l,'target':y})
        history[arm]={'fit_metrics':fit,'starts_from_qualified_af06_warmup':True,**log}
        _save_model(output,arm,model,seed,warm_sha,file_roster)
        print(json.dumps({'stage':'trained','seed':seed,'arm':arm,
                          'dev':fit['dev'],'steps':steps}),flush=True)
    jsonl_write(output/'train_dev_predictions.jsonl',records)
    file_roster['train_dev_predictions.jsonl']=sha_file(output/'train_dev_predictions.jsonl')
    result={'experiment':'AF-08','seed':seed,'width':WIDTH,'smoke':smoke,'steps':steps,
            'parent_commit':PARENT_COMMIT,'parent_artifact_id':SOURCE_ARTIFACTS[seed]['id'],
            'source_zip_sha256':zip_sha,'parent_train_receipt_sha256':receipt_sha,
            'qualified_parent_warmup_sha256':warm_sha,
            'qualified_parent_metrics':rec['arms']['warmup']['fit_metrics'],
            'original_warmup_optimization_steps':480,'shared_continuation_source':True,
            'source_hashes':source_receipt(),'corpus_sha256':validate_manifest(),
            'arms':history,'training_rows':len(records),'files':file_roster}
    json_write(output/'train_receipt.json',result)
    (output/'train_receipt.sha256').write_text(sha_file(output/'train_receipt.json')+'\n')
    return result


def load_models(folder,seed,smoke=False):
    folder=Path(folder)
    if sha_file(folder/'train_receipt.json')!=(folder/'train_receipt.sha256').read_text().strip():
        raise ValueError('AF-08 train receipt checksum mismatch')
    rec=json.loads((folder/'train_receipt.json').read_text())
    if (rec['seed'],rec['width'],rec['smoke'],rec['parent_commit'])!=(seed,WIDTH,smoke,PARENT_COMMIT):
        raise ValueError('AF-08 wrong frozen run identity')
    if rec['source_hashes']!=source_receipt() or rec['corpus_sha256']!=validate_manifest():
        raise ValueError('AF-08 frozen source/corpus changed')
    if set(rec['arms'])!=set(ARMS) or rec['training_rows']!=len(ARMS)*(672+224):
        raise ValueError('AF-08 training roster incomplete')
    if rec['source_zip_sha256']!=SOURCE_ARTIFACTS[seed]['sha256'] or not qualified(rec['qualified_parent_metrics']):
        raise ValueError('AF-08 parent no longer qualified')
    if set(rec['files'])!=set(['train_dev_predictions.jsonl']+[f'{a}.{ext}' for a in ARMS for ext in ('pt','npz')]):
        raise ValueError('AF-08 checkpoint/export roster incomplete')
    for file,sha in rec['files'].items():
        if sha_file(folder/file)!=sha:raise ValueError('AF-08 frozen checkpoint digest disagreement')
    out={}
    for arm in ARMS:
        pkg=torch.load(folder/f'{arm}.pt',map_location='cpu',weights_only=True)
        if (pkg['experiment'],pkg['seed'],pkg['arm'],pkg['source_sha256'])!=('AF-08',seed,arm,rec['qualified_parent_warmup_sha256']):
            raise ValueError('AF-08 checkpoint identity changed')
        if pkg['source_hashes']!=source_receipt():raise ValueError('AF-08 checkpoint source changed')
        model=GraphLooper(WIDTH);model.load_state_dict(pkg['model']);model.eval()
        with np.load(folder/f'{arm}.npz',allow_pickle=False) as z:
            if set(z.files)!=set(model.state_dict()):raise ValueError('AF-08 export keys changed')
            if any(not np.array_equal(z[k],v.detach().numpy()) for k,v in model.state_dict().items()):
                raise ValueError('AF-08 export tensors changed')
        out[arm]=model
    return rec,out


def separation_for_model(model,ds,pre_loops,post_loops=3,chunk=96):
    codes=[];targets=[]
    with torch.no_grad():
        for start in range(0,len(ds.cases),chunk):
            ix=list(range(start,min(start+chunk,len(ds.cases))))
            b=ds.get(ix);p=model.prefix(b,pre_loops)
            trace=trace_suffix(model,b,p,post_loops)
            codes.append(trace.q_codes)
            targets.append(b.targets[:,:2])
        return code_separation_scores(torch.cat(codes),torch.cat(targets))


def evaluate_one(training,output,seed,smoke=False):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('fresh AF-08 scoring output required')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    rec,models=load_models(training,seed,smoke)
    splits=('dev',) if smoke else ('held_composition','held_layout','held_depth4','held_depth5')
    scored={'experiment':'AF-08','seed':seed,'width':WIDTH,'smoke':smoke,
            'source_hashes':source_receipt(),'train_receipt_sha256':sha_file(Path(training)/'train_receipt.json'),
            'fits':rec['arms'],'qualified_parent_metrics':rec['qualified_parent_metrics'],
            'scores':{},'patches':{},'trajectories':{},'separation':{}}
    clean_rows=[];patch_rows=[];stage_rows=[]
    for arm,model in models.items():
        scored['scores'][arm]={};scored['patches'][arm]={};scored['trajectories'][arm]={};scored['separation'][arm]={}
        for split in splits:
            ds=Dataset(split);loops=EVAL_LOOPS[split]
            with torch.no_grad():
                clean,logits=base.get_predictions(model,ds,pre_loops=loops)
                patch,records=base.score_counterfactual(model,ds,pre_loops=loops)
                stages={}
                for post in ((3,5) if split in ('held_depth4','held_depth5') and not smoke else (3,)):
                    rows=trajectory_records(model,ds,loops,post_loops=post)
                    stages[str(post)]=trajectory_summary(rows)
                    stage_rows.extend({'arm':arm,'split':split,'seed':seed,'width':WIDTH,**x} for x in rows)
                sep=separation_for_model(model,ds,loops)
            if patch['whole_direct_max_logit_error']>2e-5:
                raise ValueError('original model whole-donor replay discrepancy')
            scored['scores'][arm][split]=clean;scored['patches'][arm][split]=patch
            scored['trajectories'][arm][split]=stages;scored['separation'][arm][split]=sep
            for c,y,t in zip(ds.cases,logits.tolist(),ds.batch.targets.tolist()):
                clean_rows.append({'seed':seed,'width':WIDTH,'arm':arm,'split':split,
                                   'id':c.id,'pre_loops':loops,'logits':y,'target':t})
            for row in records:
                patch_rows.append({'seed':seed,'width':WIDTH,'arm':arm,'split':split,**row})
        print(json.dumps({'stage':'evaluated','seed':seed,'arm':arm}),flush=True)
    for name,rows in [('clean_predictions.jsonl',clean_rows),('patch_predictions.jsonl',patch_rows),('stage_predictions.jsonl',stage_rows)]:
        jsonl_write(output/name,rows)
    scored['clean_rows']=len(clean_rows);scored['patch_rows']=len(patch_rows);scored['stage_rows']=len(stage_rows)
    scored['files']={name:sha_file(output/name) for name in ('clean_predictions.jsonl','patch_predictions.jsonl','stage_predictions.jsonl')}
    json_write(output/'evaluation.json',scored)
    (output/'evaluation.sha256').write_text(sha_file(output/'evaluation.json')+'\n')
    return scored


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('manifest','lean','source','train','evaluate'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,default=17)
    p.add_argument('--width',type=int,choices=(WIDTH,),default=WIDTH)
    p.add_argument('--source-zip',type=Path)
    p.add_argument('--training',type=Path)
    p.add_argument('--split',choices=gym.SPLITS,default='dev')
    p.add_argument('--steps',type=int,default=STEPS)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.command=='manifest':json_write(a.out,gym.manifest())
    elif a.command=='lean':a.out.write_text(gym.lean_certificate(a.split))
    elif a.command=='source':
        if a.source_zip is None:p.error('source requires --source-zip')
        r,_,h,w,t=source_from_zip(a.source_zip,a.seed)
        json_write(a.out,{'seed':a.seed,'artifact':h,'warmup':w,'receipt':t,
                          'train':r['arms']['warmup']['fit_metrics']['train'],
                          'dev':r['arms']['warmup']['fit_metrics']['dev']})
    elif a.command=='train':
        if a.source_zip is None:p.error('train requires --source-zip')
        train_one(a.source_zip,a.out,a.seed,a.steps,a.smoke)
    else:
        if a.training is None:p.error('evaluate requires --training')
        evaluate_one(a.training,a.out,a.seed,a.smoke)

if __name__=='__main__':main()
