"""Audited branch-balanced RAM-06 recurrent training, with dev-only selection."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import platform
import random
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch import nn
import torch.nn.functional as F
import model as m
import ram08_core as c
from runtime import ResourceGuard,ResourceLimit

def load_data(root,config,config_sha=None):
    audit=json.loads((root/'independent_audit.json').read_text())
    collection=json.loads((root/'collection_report.json').read_text())
    assert audit['status']=='passed' and collection['status']=='complete'
    assert audit['dataset_sha256']==collection['dataset_sha256']==c.digest(root/'transitions.jsonl')
    if config_sha is not None:assert audit['config_sha256']==collection['config_sha256']==config_sha
    rows=c.read_rows(root/'transitions.jsonl');dev=[r for r in rows if r['development']]
    assert all(r['development']==(r['source_file']+':'+r['theorem'] in config['development']) for r in rows)
    forbidden={r['state_sha256'] for r in dev}
    excluded=[r for r in rows if not r['development'] and r['state_sha256'] in forbidden]
    train=[r for r in rows if not r['development'] and r['state_sha256'] not in forbidden]
    train_theorems={(r['source_file'],r['theorem']) for r in train};dev_theorems={(r['source_file'],r['theorem']) for r in dev}
    assert not train_theorems&dev_theorems
    assert all(len(r['before']['goals'])>0 for r in rows)
    counts={name:dict(rows=len(part),theorems=len({(r['source_file'],r['theorem']) for r in part}),
       states=len({r['state_sha256'] for r in part}),multi_input_rows=sum(len(r['before']['goals'])>1 for r in part),
       multi_input_states=len({r['state_sha256'] for r in part if len(r['before']['goals'])>1}),
       outcomes=dict(Counter(r['label'] for r in part))) for name,part in [('train',train),('development',dev)]}
    assert counts['train']['multi_input_states']>0 and counts['development']['multi_input_states']>0
    return train,dev,dict(**counts,excluded_training_rows=len(excluded),dataset_sha256=audit['dataset_sha256'],
      independent_audit_sha256=c.digest(root/'independent_audit.json'),excluded_state_hashes=sorted({r['state_sha256'] for r in excluded}))

def data_tensors(rows):
    x=m.encode(rows);y=torch.tensor([r['label'] for r in rows]);delta=torch.tensor([r['delta'] for r in rows],dtype=torch.float32)
    weights=(len(rows)/(4*torch.bincount(y,minlength=4).float().clamp_min(1))).sqrt();weights/=weights.mean()
    multi=torch.tensor([i for i,r in enumerate(rows) if len(r['before']['goals'])>1]);single=torch.tensor([i for i,r in enumerate(rows) if len(r['before']['goals'])==1])
    typed_positive=torch.tensor([i for i,r in enumerate(rows) if r['family'] in ('exact_typed','exact_local') and r['label']>0],dtype=torch.long)
    if len(typed_positive)<1:
        raise AssertionError('zero independently verified naturally mined exact positives in original MathCan TRAIN roots')
    return x,y,delta,weights,(single,multi,typed_positive)

def set_seed(seed):
    random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def update(model,opt,data,generator,config,guard):
    x,y,delta,w,(single,multi,typed_positive)=data;device=next(model.parameters()).device
    size=config['microbatch']*config['accumulation']
    n_multi=size//2
    n_exact=round(size*config['exact_positive_fraction'])
    n_single=size-n_multi-n_exact
    if n_single<=0 or n_exact<=0:
        raise AssertionError('invalid pre-registered exact/multigoal batch fractions')
    ids=torch.cat([
        single[torch.randint(len(single),(n_single,),generator=generator)],
        multi[torch.randint(len(multi),(n_multi,),generator=generator)],
        typed_positive[torch.randint(len(typed_positive),(n_exact,),generator=generator)]])
    ids=ids[torch.randperm(size,generator=generator)];denominator=w[y[ids]].sum().to(device)
    opt.zero_grad(set_to_none=True);total=0.
    for ix in ids.split(config['microbatch']):
        guard.check();logits,pred=model({k:v[ix].to(device) for k,v in x.items()})
        ce=F.cross_entropy(logits,y[ix].to(device),weight=w.to(device),reduction='sum')/denominator
        auxiliary=F.smooth_l1_loss(pred,delta[ix].to(device),reduction='sum')/size
        loss=ce+config['delta_loss_weight']*auxiliary
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward();total+=float(loss.detach())
    nn.utils.clip_grad_norm_(model.parameters(),config['clip'],error_if_nonfinite=True);opt.step();guard.check()
    return total

@torch.no_grad()
def metrics(model,rows,weights):
    model.eval();device=next(model.parameters()).device;outs=[];deltas=[]
    for i in range(0,len(rows),16):
        logits,delta=model({k:v.to(device) for k,v in m.encode(rows[i:i+16]).items()});outs.append(logits.cpu());deltas.append(delta.cpu())
    logits=torch.cat(outs);pred_delta=torch.cat(deltas);y=torch.tensor([r['label'] for r in rows]);delta=torch.tensor([r['delta'] for r in rows])
    prediction=logits.argmax(-1);confusion=torch.bincount(4*y+prediction,minlength=16).reshape(4,4)
    p=logits.softmax(-1);score=3*p[:,3]+1.5*p[:,1]+.3*p[:,2]-.6*p[:,0]+.4*pred_delta.clamp(-3,3)
    by_state=defaultdict(list)
    for i,r in enumerate(rows):by_state[r['state_sha256']].append(i)
    selected=[max(indices,key=lambda i:(float(score[i]),-i)) for indices in by_state.values()]
    classes={}
    for i,name in enumerate(('failed_unchanged','goal_count_reduced','other_progress','all_goals_closed')):
        support=int(confusion[i].sum());chosen=int(confusion[:,i].sum());tp=int(confusion[i,i])
        classes[name]=dict(support=support,precision=tp/chosen if chosen else None,recall=tp/support if support else None)
    losses=F.cross_entropy(logits,y,weight=weights,reduction='none');groups={};selection=[]
    for name,multiple in [('single',False),('multi',True)]:
        ids=torch.tensor([i for i,r in enumerate(rows) if (len(r['before']['goals'])>1)==multiple])
        ce=float(losses[ids].sum()/weights[y[ids]].sum()) if len(ids) else None
        aux=float(F.smooth_l1_loss(pred_delta[ids],delta[ids].float())) if len(ids) else None
        groups[name]=dict(rows=len(ids),weighted_ce=ce,delta_smooth_l1=aux)
        if ce is not None:selection.append(ce+.2*aux)
    families={}
    for family in c.FAMILIES:
        ids=[i for i,r in enumerate(rows) if r['family']==family];positive=sum(int(y[i]>0) for i in ids)
        families[family]=dict(rows=len(ids),positive=positive,positive_recall=sum(int(y[i]>0 and prediction[i]>0) for i in ids)/positive if positive else None)
    return dict(selection_loss=sum(selection)/len(selection),unweighted_ce=float(F.cross_entropy(logits,y)),
      delta_mae=float((pred_delta-delta).abs().mean()),confusion=confusion.tolist(),classes=classes,input_groups=groups,
      states=len(by_state),top1_usable=sum(int(y[i]>0) for i in selected),top1_closed=sum(int(y[i]==3) for i in selected),
      families=families,kernel_certified_proofs=0)

def calibrate(data,config,guard,out):
    results=[]
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        set_seed(17);model=m.Policy(m.Config('loop64',64,4)).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
        g=torch.Generator().manual_seed(1017)
        for _ in range(10):update(model,opt,data,g,config,guard)
        if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        began=time.perf_counter()
        for _ in range(30):update(model,opt,data,g,config,guard)
        if device=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-began
        results.append(dict(device=device,updates=30,seconds=elapsed,updates_per_second=30/elapsed,
          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0))
        del model,opt;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    speedup=results[1]['updates_per_second']/results[0]['updates_per_second'] if len(results)>1 else None
    result=dict(trials=results,cuda_speedup=speedup,selected_device='cuda' if speedup and speedup>=1.15 else 'cpu',precision='float32')
    c.write_json(out/'calibration.json',result);print('RAM08_CALIBRATION',json.dumps(result),flush=True)
    return result

def load_checkpoint(path):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data['schema']!='RAM08-v1':raise ValueError('unknown checkpoint schema')
    model=m.Policy(m.Config(**data['model_config']));model.load_state_dict(data['state_dict']);return model.eval()

def fit(train,dev,data,model_config,seed,config,device,out,guard,fingerprint):
    out.mkdir(parents=True,exist_ok=False);set_seed(seed)
    model=m.Policy(model_config).to(device);opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    g=torch.Generator().manual_seed(seed+1000);best=float('inf');best_step=0;stale=0;history=[];began=time.perf_counter()
    for step in range(1,config['updates']+1):
        model.train();sample_loss=update(model,opt,data,g,config,guard)
        if step%config['validation_interval']:continue
        tr=metrics(model,train,data[3]);dv=metrics(model,dev,data[3]);score=dv['selection_loss']
        checkpoint=dict(schema='RAM08-v1',model_config=asdict(model_config),seed=seed,step=step,
          state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),
          sampler_rng=g.get_state(),torch_rng=torch.get_rng_state(),training_config=config,**fingerprint)
        torch.save(checkpoint,out/'last.pt')
        if score<best-1e-4:best=score;best_step=step;stale=0;torch.save(checkpoint,out/'best.pt')
        else:stale+=1
        record=dict(step=step,sampled_loss=sample_loss,train=tr,dev=dv)
        c.append_json(out/'validation.jsonl',record);history.append(dict(step=step,train=tr['selection_loss'],dev=score))
        print('RAM08_VALIDATE',model_config.name,seed,step,round(score,5),best_step,dv['top1_usable'],dv['states'],flush=True)
        if stale>=config['patience']:break
    seconds=time.perf_counter()-began
    last=load_checkpoint(out/'last.pt');model=model.cpu().eval();sample={k:v[:8] for k,v in data[0].items()}
    with torch.no_grad():
        assert all(torch.equal(a,b) for a,b in zip(model(sample),last(sample)))
    selected=load_checkpoint(out/'best.pt')
    receipt=dict(name=model_config.name,seed=seed,parameters=sum(p.numel() for p in selected.parameters()),
      updates=step,best_step=best_step,seconds=seconds,device=device,history=history,train=metrics(selected,train,data[3]),
      dev=metrics(selected,dev,data[3]),checkpoint=str(out/'best.pt'),checkpoint_sha256=c.digest(out/'best.pt'),
      last_checkpoint_sha256=c.digest(out/'last.pt'),reload_exact=True,resources=guard.receipt())
    c.write_json(out/'receipt.json',receipt)
    del model,opt,last,selected;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);config=json.loads(a.config.read_text())
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.625);torch.backends.cuda.matmul.allow_tf32=False
    report=dict(schema='RAM08-v1',models=[],config=config,config_sha256=c.digest(a.config),python=platform.python_version(),torch=str(torch.__version__))
    try:
        with ResourceGuard(a.out/'resources.jsonl',torch) as guard:
            train,dev,audit=load_data(a.data,config,report['config_sha256']);report['data']=audit;c.write_json(a.out/'data_audit.json',audit)
            exact_wins=[r for r in train if r['family'] in ('exact_typed','exact_local') and r['label']>0]
            typed_wins=[r for r in train if r['family']=='exact_typed' and r['label']>0]
            report['natural_exact']={
              'positive_actions':len(exact_wins),
              'typed_positive_actions':len(typed_wins),
              'theorem_roots':sorted({r['source_file']+':'+r['theorem'] for r in exact_wins}),
              'positive_train_source_only':all(r['source_file'].startswith('MathCan/') for r in exact_wins),
              'no_synthetic_auxiliary':all('RAM06/' not in r['source_file'] for r in train+dev),
            }
            if not exact_wins or not report['natural_exact']['positive_train_source_only']:
                raise AssertionError('original MathCan source has no independently replayed positive exact')
            print('RAM08_NATURAL_EXACT_GATE',json.dumps(report['natural_exact'],sort_keys=True),flush=True)
            data=data_tensors(train);calibration=calibrate(data,config['training'],guard,a.out);report['calibration']=calibration
            fingerprint=dict(config_sha256=report['config_sha256'],dataset_sha256=audit['dataset_sha256'])
            for architecture in config['models']:
                for seed in config['seeds']:
                    current=dict(config['training']);attempt=0
                    while True:
                        destination=a.out/f"{architecture['name']}_s{seed}" if not attempt else a.out/'retries'/f"{architecture['name']}_s{seed}_micro{current['microbatch']}"
                        try:receipt=fit(train,dev,data,m.Config(**architecture),seed,current,calibration['selected_device'],destination,guard,fingerprint);break
                        except (MemoryError,torch.cuda.OutOfMemoryError,ResourceLimit) as error:
                            c.append_json(a.out/'memory_failures.jsonl',dict(model=architecture['name'],seed=seed,error=repr(error),config=current));error.__traceback__=None
                        gc.collect()
                        if torch.cuda.is_available():torch.cuda.empty_cache()
                        if current['microbatch']==1:raise ResourceLimit('microbatch1 failed')
                        current=dict(current,microbatch=current['microbatch']//2,accumulation=current['accumulation']*2);attempt+=1
                        if guard.error is not None:guard.recover()
                    report['models'].append(receipt);c.write_json(a.out/'training_report.json',report)
            report['selected']={name:min([r for r in report['models'] if r['name']==name],key=lambda r:(r['dev']['selection_loss'],r['seed'])) for name in ('loop32','loop64','flat64')}
            report['resources']=guard.receipt();report['status']='complete'
    except BaseException as error:
        report['status']='aborted';report['error']=repr(error);c.write_json(a.out/'training_report.json',report);raise
    c.write_json(a.out/'training_report.json',report)
    print('RAM08_TRAINING_COMPLETE',json.dumps({k:dict(seed=v['seed'],loss=v['dev']['selection_loss']) for k,v in report['selected'].items()}),flush=True)
if __name__=='__main__':main()
