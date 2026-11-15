"""RAM-10 gated, fresh-source, CPU-only nine-model training and DEV-only freeze.

Only original new MathCan TRAIN/DEV compiler observations may enter this step.
TEST theorem files must be absent and PEF scores remain independent.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import hashlib
import os
import psutil
import json
from pathlib import Path
import random
import sys
import time
import torch
import torch.nn.functional as F
from torch import nn
from ram10_core import rows,sha_file,stable_json,registered_ram08_core,RAM08


def original_model(config):
    registered_ram08_core(config)
    if str(RAM08) not in sys.path:sys.path.insert(0,str(RAM08))
    import model as m
    return m


def load_training_data(path,config,config_path):
    audit=json.loads((path/'independent_audit.json').read_text())
    gate=json.loads((path/'training_gate.json').read_text())
    coll=json.loads((path/'collection_report.json').read_text())
    digest=sha_file(path/'transitions.jsonl')
    if not gate['admitted'] or audit['status']!='passed' or coll['status']!='complete':
        raise ValueError('original training gate or independent Lean replay failed')
    if digest!=audit['dataset_sha256'] or digest!=gate['dataset_sha256'] or digest!=coll['dataset_sha256']:
        raise ValueError('source dataset hash mismatch')
    if any(receipt['config_sha256']!=sha_file(config_path) for receipt in (coll,gate)):
        raise ValueError('preregistered configuration mismatch')
    data=rows(path/'transitions.jsonl')
    tr=[r for r in data if r['partition']=='train']
    dv=[r for r in data if r['partition']=='dev']
    if not tr or not dv:raise AssertionError('real train or development data missing')
    forbidden=set(config['modules']['test'])|set(config['excluded_pef_modules'])
    if any(Path(r['source_file']).stem in forbidden for r in data):raise AssertionError('TRAIN contaminated with forbidden original source')
    if {(r['source_file'],r['theorem']) for r in tr}&{(r['source_file'],r['theorem']) for r in dv}:
        raise AssertionError('original theorem root crosses TRAIN/DEV split')
    if {r['state_sha256'] for r in tr}&{r['state_sha256'] for r in dv}:
        raise AssertionError('original compiler state crosses TRAIN/DEV split')
    return tr,dv,digest,gate


def encoded_rows(model,examples):
    t=model.encode(examples)
    label=torch.tensor([r['label'] for r in examples],dtype=torch.long)
    delta=torch.tensor([r['delta'] for r in examples],dtype=torch.float32)
    return t,label,delta


def sampling_indices(train,seed):
    tr_rng=torch.Generator().manual_seed(seed+1000)
    original=torch.arange(len(train),dtype=torch.long)
    multi=torch.tensor([i for i,r in enumerate(train) if len(r['before']['goals'])>1],dtype=torch.long)
    exact=torch.tensor([i for i,r in enumerate(train) if r['family'] in ('exact_typed','exact_local') and r['label']>0],dtype=torch.long)
    rw=torch.tensor([i for i,r in enumerate(train) if r['family']=='rw_local' and r['label']>0],dtype=torch.long)
    if min(len(multi),len(exact),len(rw))==0:
        raise AssertionError('original multi-goal, exact-positive, or rw-positive TRAIN reservoir absent')
    while True:
        a=multi[torch.randint(len(multi),(32,),generator=tr_rng)]
        b=exact[torch.randint(len(exact),(6,),generator=tr_rng)]
        c=rw[torch.randint(len(rw),(6,),generator=tr_rng)]
        d=original[torch.randint(len(original),(20,),generator=tr_rng)]
        ids=torch.cat([a,b,c,d])
        yield ids[torch.randperm(64,generator=tr_rng)],tr_rng.get_state()


@torch.no_grad()
def development_score(model,encoded,labels,delta,weights,rows_dev):
    model.eval();preds=[];diff=[]
    n=len(labels)
    for start in range(0,n,32):
        logits,d=model({k:v[start:start+32] for k,v in encoded.items()})
        preds.append(logits);diff.append(d)
    logits=torch.cat(preds);estimated=torch.cat(diff)
    losses=F.cross_entropy(logits,labels,weight=weights,reduction='none')
    parts=[]
    for count in (1,2):
        ids=torch.tensor([i for i,r in enumerate(rows_dev) if (len(r['before']['goals'])>1)==(count==2)],dtype=torch.long)
        if len(ids):
            ce=float(losses[ids].sum()/weights[labels[ids]].sum())
            aux=float(F.smooth_l1_loss(estimated[ids],delta[ids]))
            parts.append(ce+0.2*aux)
    if not parts:raise AssertionError('no original development states')
    p=logits.softmax(-1)
    utility=3*p[:,3]+1.5*p[:,1]+.3*p[:,2]-.6*p[:,0]+.4*estimated.clamp(-3,3)
    by_state=defaultdict(list)
    for i,row in enumerate(rows_dev):by_state[(row['source_file'],row['theorem'],row['state_sha256'])].append(i)
    chosen=[max(v,key=lambda i:(float(utility[i]),-i)) for v in by_state.values()]
    return dict(selection_loss=sum(parts)/len(parts),states=len(by_state),
                top1_usable=sum(int(labels[i]>0) for i in chosen),
                top1_closed=sum(int(labels[i]==3) for i in chosen),
                per_class_support=dict(Counter(labels.tolist())))


def process_tree_rss_bytes():
    proc=psutil.Process(os.getpid())
    rss=0
    for child in [proc]+proc.children(recursive=True):
        try:rss+=child.memory_info().rss
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return rss


def train(source,config_path,out):
    config=json.loads(Path(config_path).read_text());source=Path(source);out=Path(out)
    if out.exists():raise ValueError('output already exists; protect first evaluation')
    tr,dev,data_sha,gate=load_training_data(source,config,Path(config_path))
    m=original_model(config)
    train_encoded,y,d=encoded_rows(m,tr)
    dev_encoded,yd,dd=encoded_rows(m,dev)
    weights=(len(y)/(4*torch.bincount(y,minlength=4).float().clamp_min(1))).sqrt();weights/=weights.mean()
    torch.set_num_threads(config['operational']['cpu_threads']);torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True,exist_ok=False)
    receipt=dict(experiment='RAM10_new_mathcan_original_train_only',status='training',
        original_mathcan_source=config['source_mathcan'],
        config_sha256=sha_file(config_path),dataset_sha256=data_sha,
        independently_replayed_compiler_effects=gate['independently_replayed'],
        train_original_tactic_observations=len(tr),dev_original_tactic_observations=len(dev),
        fresh_test_proof_cases_scored=0,development_only_selection=True,
        models=[],selected={},gate=gate)
    stable_json(out/'training_report.json',receipt)
    for spec in config['training']['models']:
        for seed in config['training']['seeds']:
            random.seed(seed);torch.manual_seed(seed)
            model=m.Policy(m.Config(**spec))
            count=sum(p.numel() for p in model.parameters())
            if count>=250000:raise AssertionError('registered compact model exceeds bound')
            opt=torch.optim.AdamW(model.parameters(),lr=config['training']['lr'],weight_decay=config['training']['decay'])
            sampler=sampling_indices(tr,seed)
            checkpoint_dir=out/f"{spec['name']}_s{seed}";checkpoint_dir.mkdir()
            best=float('inf');best_step=0;stale=0;history=[];began=time.perf_counter();peak_rss=0
            for step in range(1,config['training']['updates']+1):
                if step%25==1:
                    peak_rss=max(peak_rss,process_tree_rss_bytes())
                    if peak_rss>config['operational']['rss_gib']*(1024**3):
                        raise MemoryError('original registered six-GiB process-tree limit')
                model.train();ix,sampler_rng=next(sampler)
                denom=weights[y[ix]].sum()
                opt.zero_grad(set_to_none=True)
                for block in ix.split(config['training']['microbatch']):
                    logits,pred=model({k:v[block] for k,v in train_encoded.items()})
                    ce=F.cross_entropy(logits,y[block],weight=weights,reduction='sum')/denom
                    delta_loss=F.smooth_l1_loss(pred,d[block],reduction='sum')/len(ix)
                    loss=ce+config['training']['delta_weight']*delta_loss
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite original train loss')
                    loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),config['training']['clip'],error_if_nonfinite=True)
                opt.step()
                if step%config['training']['val_interval']!=0:continue
                metrics=development_score(model,dev_encoded,yd,dd,weights,dev)
                score=metrics['selection_loss']
                ckpt=dict(schema='RAM10-v1',model_config=asdict(model.config),seed=seed,
                          step=step,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                          optimizer=opt.state_dict(),sampler_rng=sampler_rng,
                          torch_rng=torch.get_rng_state(),data_sha256=data_sha,
                          config_sha256=sha_file(config_path),original_mathcan_sha=config['source_mathcan'])
                torch.save(ckpt,checkpoint_dir/'last.pt')
                if score<best-1e-4:
                    best=score;best_step=step;stale=0
                    torch.save(ckpt,checkpoint_dir/'best.pt')
                else:stale+=1
                history.append(dict(step=step,dev=metrics))
                print('RAM10_DEV',spec['name'],seed,step,round(score,5),metrics['states'],flush=True)
                if stale>=config['training']['patience']:break
            best_file=checkpoint_dir/'best.pt'
            report=dict(name=spec['name'],seed=seed,parameters=count,best_step=best_step,
                        last_step=step,best_score=best,best_sha256=sha_file(best_file),
                        last_sha256=sha_file(checkpoint_dir/'last.pt'),
                        seconds=time.perf_counter()-began,peak_process_tree_rss_bytes=peak_rss,history=history,
                        relative_best_file=str(best_file.relative_to(out)))
            receipt['models'].append(report);stable_json(out/'training_report.json',receipt)
    for spec in config['training']['models']:
        alternatives=[x for x in receipt['models'] if x['name']==spec['name']]
        winner=min(alternatives,key=lambda x:(x['best_score'],x['seed']))
        receipt['selected'][spec['name']]=dict(seed=winner['seed'],step=winner['best_step'],
                                               sha256=winner['best_sha256'],file=winner['relative_best_file'])
    receipt['status']='complete';receipt['frozen_model_count']=len(receipt['models'])
    stable_json(out/'training_report.json',receipt)
    print('RAM10_FROZEN_DEV_SELECTED',json.dumps(receipt['selected'],sort_keys=True),flush=True)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();train(args.source,args.config,args.out)
