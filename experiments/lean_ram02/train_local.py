"""RAM-02 local campaign: verified frozen labels, bounded resources, dev selection."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
import torch.nn.functional as F
import model_data as m
from resources import ResourceGuard, ResourceLimit, GIB

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf8', newline='\n')
    temp.replace(path)

def append_json(path, value):
    with Path(path).open('a', encoding='utf8', newline='\n') as f:
        f.write(json.dumps(value, ensure_ascii=False)+'\n')

def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def setup(config):
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(min(.625, config['max_cuda_allocated_gib']*GIB/torch.cuda.get_device_properties(0).total_memory))
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False

def smaller_microbatch(config):
    """Resource recovery changes the microbatch while preserving effective batch."""
    size=config['microbatch']
    if size<=1: return None
    effective=size*config['accumulation']
    reduced=size//2
    if effective%reduced: raise ValueError('cannot preserve effective batch')
    return dict(config,microbatch=reduced,accumulation=effective//reduced)

def memory_failure(error):
    return isinstance(error,(MemoryError,torch.cuda.OutOfMemoryError,ResourceLimit)) or (
        isinstance(error,RuntimeError) and any(text in str(error).lower() for text in
        ('out of memory',"can't allocate memory",'cannot allocate memory','cublas_status_alloc_failed')))

def fit_with_backoff(train,dev,model_config,seed,config,device,root,guard,config_hash):
    current=dict(config)
    failures=[]
    while True:
        destination=root if not failures else root/'memory-retries'/f'{model_config.name}_s{seed}_micro{current["microbatch"]}'
        try:
            receipt=fit(train,dev,model_config,seed,current,device,destination,guard,config_hash)
            receipt['memory_retries']=failures
            receipt['effective_training_config']=current
            for item in receipt['checkpoints'].values():
                item['relative_path']=str(Path(item['path']).relative_to(root)).replace('\\','/')
            write_json(Path(receipt['checkpoints']['best.pt']['path']).parent/'receipt.json',receipt)
            return receipt
        except (RuntimeError,MemoryError) as exc:
            if not memory_failure(exc): raise
            error=repr(exc)
            exc.__traceback__=None
        # Exit the exception block to release the failed model/optimizer frames.
        failure=dict(error=error,microbatch=current['microbatch'],accumulation=current['accumulation'],device=device)
        failures.append(failure)
        append_json(root/'memory-retries.jsonl',dict(model=model_config.name,seed=seed,**failure))
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        reduced=smaller_microbatch(current)
        if reduced is None: raise ResourceLimit('microbatch1 failed; preserved attempts: '+error)
        # Retry only after live hardware has recovered above every safety floor.
        if guard.error is not None: guard.recover()
        else:
            guard.sample(gpu=True)
            guard.check()
        current=reduced

def split_data(root):
    states, rows = m.load_data(root)
    train = [r for r in rows if r['source_file'] not in m.DEV_MODULES]
    dev = [r for r in rows if r['source_file'] in m.DEV_MODULES]
    assert len({r['stage_id'] for r in train}) == 39
    assert len({r['stage_id'] for r in dev}) == 12
    assert not ({(r['source_file'],r['theorem']) for r in train} & {(r['source_file'],r['theorem']) for r in dev})
    positive = Counter(r['family'] for r in train if r['label'] > 0)
    assert all(positive[f] > 0 for f in ('exact_local','rw_local','rcases_local','induction_list'))
    frequency = {}
    for family in m.FAMILIES:
        q = [r for r in train if r['family']==family]
        frequency[family] = sum(2*(r['label']==3)+(r['label']==1) for r in q)/max(1,len(q))
    native, freq = m.cached_metrics(None, dev), m.cached_metrics(None, dev, frequency)
    assert native['top1_executable'] == 3 and freq['top1_executable'] == 6
    return train, dev, dict(states=len(states), probes=len(rows), train_states=39, dev_states=12,
        train_rows=len(train), dev_rows=len(dev), positive_training_support=dict(positive),
        state_sha256=m.STATE_SHA, effect_semantics_sha256=m.EFFECT_SHA,
        original_zip_sha256='d91a12d725ca0a611b4de308c80365bfd3888d88449b03b2c5f07137a1c2025a',
        original_fc12_checkpoint_hashes={p.name:m.digest(p) for p in root.glob('fc12_seed*.pt')},
        native=native, frequency=freq, frequency_weights=frequency)

def tensors(rows):
    x = m.encode(rows)
    y = torch.tensor([r['label'] for r in rows], dtype=torch.long)
    counts = torch.bincount(y,minlength=4).float()
    weights = (len(rows)/(4*counts.clamp_min(1))).sqrt()
    return x, y, weights/weights.mean()

def update(model, opt, x, y, weights, generator, config, guard, mms=None, step=0):
    device = next(model.parameters()).device
    batch = config['microbatch']*config['accumulation']
    ix = torch.randint(len(y), (batch,), generator=generator)
    denominator = weights[y[ix]].sum().to(device)
    opt.zero_grad(set_to_none=True)
    total = 0.
    for ids in ix.split(config['microbatch']):
        guard.check()
        from contextlib import nullcontext
        context = (mms.capture(ids, step, model.config.loops // 2, model.config.loops)
                   if mms is not None else nullcontext())
        with context:
            logits = model({k:v[ids].to(device) for k,v in x.items()})
            loss = F.cross_entropy(logits, y[ids].to(device), weight=weights.to(device), reduction='sum')/denominator
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
            loss.backward()
        total += float(loss.detach())
    norm = nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
    opt.step()
    guard.check()
    return total, float(norm)

@torch.no_grad()
def metrics(model, rows, microbatch=16):
    model.eval()
    device = next(model.parameters()).device
    logits = torch.cat([model({k:v.to(device) for k,v in m.encode(rows[i:i+microbatch]).items()}).cpu()
                        for i in range(0,len(rows),microbatch)])
    y = torch.tensor([r['label'] for r in rows])
    pred = logits.argmax(-1)
    confusion = torch.bincount(4*y+pred,minlength=16).reshape(4,4)
    classes = {}
    for i,name in enumerate(m.OUTCOMES):
        tp = int(confusion[i,i]); n = int(confusion[i].sum()); chosen = int(confusion[:,i].sum())
        classes[name] = dict(support=n, precision=tp/chosen if chosen else 0., recall=tp/n if n else 0.)
    per_family = {}
    for family in m.FAMILIES:
        ids = [i for i,r in enumerate(rows) if r['family']==family]
        per_family[family] = dict(rows=len(ids), positives=sum(int(y[i]>0) for i in ids),
            accuracy=sum(int(y[i]==pred[i]) for i in ids)/len(ids) if ids else None,
            positive_recall=sum(int(y[i]>0 and pred[i]>0) for i in ids)/max(1,sum(int(y[i]>0) for i in ids)))
    probs = logits.softmax(-1)
    groups = {}
    for i,r in enumerate(rows): groups.setdefault(r['stage_id'],[]).append(i)
    choices = {}
    for mode, score in [('one_goal',2*probs[:,3]+probs[:,1]),('multigoal',2*probs[:,3]+probs[:,1]+probs[:,2])]:
        selected = [max(ids,key=lambda i:(float(score[i]),-i)) for ids in groups.values()]
        choices[mode] = dict(states=len(groups),top1_usable=sum(int(y[i]>0) for i in selected),
            top1_executable=sum(int(y[i] in (1,3)) for i in selected),
            top1_scratch_closed=sum(int(y[i]==3) for i in selected),
            selected=[dict(stage_id=rows[i]['stage_id'],family=rows[i]['family'],script=rows[i]['script'],label=int(y[i])) for i in selected])
    return dict(loss=float(F.cross_entropy(logits,y)),accuracy=float((y==pred).float().mean()),
        confusion=confusion.tolist(), classes=classes, per_family=per_family, choices=choices,
        kernel_certified_proofs=0)

def calibrate(train, config, out, guard):
    x,y,w = tensors(train)
    results = []
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        seed_all(17)
        model = m.CandidatePolicy(m.Config('loop64',64,4)).to(device).train()
        opt = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        generator = torch.Generator().manual_seed(1017)
        for _ in range(config['calibration_warmup']): update(model,opt,x,y,w,generator,config,guard)
        if device=='cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        losses = []
        for _ in range(config['calibration_updates']): losses.append(update(model,opt,x,y,w,generator,config,guard)[0])
        if device=='cuda': torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        row = dict(device=device, updates=config['calibration_updates'],seconds=elapsed,
                   updates_per_second=config['calibration_updates']/elapsed, losses=losses,
                   cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                   configuration='loop64/4 passes; microbatch16 accumulation4; float32',
                   resources=guard.receipt())
        results.append(row)
        print('RAM02_CALIBRATION',json.dumps(row),flush=True)
        del model,opt
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    speedup = results[1]['updates_per_second']/results[0]['updates_per_second'] if len(results)>1 else None
    chosen = 'cuda' if speedup is not None and speedup>=config['cuda_min_speedup'] else 'cpu'
    receipt = dict(trials=results,cuda_speedup=speedup,selected_device=chosen,
        policy=f"CUDA requires >= {config['cuda_min_speedup']}x speedup; otherwise CPU",
        warmup_updates=config['calibration_warmup'], precision='float32',
        deviations=['Reuse installed PyTorch 2.6.0+cu124 instead of upgrading to RAM-01 2.10.0.',
                    'FP16 is optional and not enabled; compact-model FP32 calibration is the device decision.'])
    write_json(out/'calibration.json',receipt)
    return receipt

def load_checkpoint(path):
    value = torch.load(path,map_location='cpu',weights_only=True)
    if value['state_sha256']!=m.STATE_SHA or value['effect_sha256']!=m.EFFECT_SHA:
        raise ValueError('checkpoint dataset fingerprint differs')
    model = m.CandidatePolicy(m.Config(**value['model_config']))
    model.load_state_dict(value['state_dict'])
    return model.eval()

def checkpoint_path(root,record,which='best.pt'):
    """Receipts retain original locations; copied artifacts also load portably."""
    entry=record['checkpoints'][which]
    candidates=[root/entry['relative_path']] if entry.get('relative_path') else []
    candidates += [root/f"{record['name']}_s{record['seed']}"/which,Path(entry['path'])]
    for path in candidates:
        if path.is_file() and m.digest(path)==entry['sha256']:return path
    raise ValueError('missing or modified checkpoint: '+str(candidates[0]))

def fit(train, dev, model_config, seed, config, device, root, guard, config_hash):
    out = root/f'{model_config.name}_s{seed}'
    out.mkdir(parents=True,exist_ok=False)
    seed_all(seed)
    model = m.CandidatePolicy(model_config).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(seed+1000)
    x,y,w = tensors(train)
    mms = None
    if config.get('mms_mib',0):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
        from research_gym.interp.mms import MechanisticMemory
        if device != 'cpu' or not model_config.loops:
            raise ValueError('MMS requires a CPU recurrent ranker')
        mms = MechanisticMemory(model.cell[-1], mib=config['mms_mib'],
                mode=config['mms_mode'], ratio=config['mms_ratio'], rank=config['mms_rank'],
                topk=config['mms_topk'], refresh=config['mms_refresh'],
                max_age=config['mms_max_age'], map_file=config['mms_map'], map_sha=config['mms_map_sha'])
    start = time.perf_counter()
    best_loss, best_step, stale, history = float('inf'),0,0,[]
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    def checkpoint(step):
        return dict(schema='RAM02-v1',model_config=asdict(model_config),seed=seed,step=step,
            state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            optimizer=opt.state_dict(),sampler_rng=generator.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [],
            state_sha256=m.STATE_SHA,effect_sha256=m.EFFECT_SHA,config_sha256=config_hash,
            training_config=config, trained_states=39,development_states=12,
            **({'mms': mms.state_dict()} if mms is not None else {}))
    for step in range(1,config['max_updates']+1):
        model.train()
        loss,norm = update(model,opt,x,y,w,generator,config,guard,mms=mms,step=step)
        if step % config['validation_interval'] == 0:
            train_score, dev_score = metrics(model,train),metrics(model,dev)
            row = dict(step=step, sampled_weighted_loss=loss, gradient_norm=norm,
                       train=train_score, dev=dev_score, elapsed_seconds=time.perf_counter()-start)
            append_json(out/'validation.jsonl',row)
            history.append(dict(step=step,train_loss=train_score['loss'],dev_loss=dev_score['loss']))
            torch.save(checkpoint(step),out/'last.pt')
            if dev_score['loss'] < best_loss-config['early_stopping_min_delta']:
                best_loss,best_step,stale = dev_score['loss'],step,0
                torch.save(checkpoint(step),out/'best.pt')
            else: stale += 1
            print('RAM02_VALIDATE',json.dumps(dict(model=model_config.name,seed=seed,step=step,
                train_loss=train_score['loss'],dev_loss=dev_score['loss'],best_step=best_step,
                dev_top1=dev_score['choices']['multigoal']['top1_usable'])),flush=True)
            if stale>=config['early_stopping_patience']: break
    elapsed = time.perf_counter()-start
    model = model.cpu().eval()
    restored = load_checkpoint(out/'last.pt')
    sample = {k:v[:8] for k,v in x.items()}
    exact = torch.equal(model(sample),restored(sample))
    assert exact, 'last checkpoint CPU reload differs'
    best = load_checkpoint(out/'best.pt')
    receipt = dict(name=model_config.name,seed=seed,parameters=sum(p.numel() for p in best.parameters()),
        device=device,precision='float32',optimizer_steps=step,best_step=best_step,
        stopped_early=step<config['max_updates'],elapsed_seconds=elapsed,updates_per_second=step/elapsed,
        train=metrics(best,train),dev=metrics(best,dev),history=history,
        resources=guard.receipt(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
        checkpoints={name:dict(path=str(out/name),sha256=m.digest(out/name),bytes=(out/name).stat().st_size)
                     for name in ('best.pt','last.pt')},cpu_reload_exact=exact,config_sha256=config_hash)
    ablations=[]
    if model_config.loops:
        for passes in config['pass_ablation']:
            best.config = m.Config(model_config.name,model_config.width,passes)
            for _ in range(3): metrics(best,dev)
            began=time.perf_counter()
            score=metrics(best,dev)
            ablations.append(dict(passes=passes,seconds=time.perf_counter()-began,metrics=score))
    receipt['inference_pass_ablation']=ablations
    if mms is not None: receipt['mms']=mms.receipt()
    write_json(out/'receipt.json',receipt)
    del model,opt,restored,best
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config_3050.yaml'))
    p.add_argument('--calibrate-only',action='store_true')
    p.add_argument('--include-flat',action='store_true')
    p.add_argument('--mms-mib',type=float,default=0)
    p.add_argument('--mms-mode',choices=('observe','dense','bpd','sham'),default='bpd')
    p.add_argument('--mms-ratio',type=float,default=.2)
    p.add_argument('--mms-rank',type=int,default=16)
    p.add_argument('--mms-topk',type=int,default=4)
    p.add_argument('--mms-refresh',type=int,default=128)
    p.add_argument('--mms-max-age',type=int,default=32768)
    p.add_argument('--mms-map',default='')
    args=p.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    config_hash=m.digest(args.config)
    if args.mms_mib:
        if args.include_flat or args.mms_mib < 0:
            p.error('MMS requires positive budget and recurrent arms only')
        import hashlib, sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
        from research_gym.interp.mms import file_sha256
        config.update({k:v for k,v in vars(args).items() if k.startswith('mms_')})
        config['mms_map_sha']=file_sha256(args.mms_map) if args.mms_map else ''
        config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    elif args.mms_map:
        p.error('MMS map requires positive MMS budget')
    args.out.mkdir(parents=True,exist_ok=False)
    setup(config)
    report=dict(experiment='RAM02_local_training',config=config,config_sha256=config_hash,
                python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform(),
                cuda_available=torch.cuda.is_available(),models=[],kernel_certified_proofs=0)
    write_json(args.out/'frozen_config.json',report)
    guard=ResourceGuard(args.out/'resources.jsonl',torch,
        max_rss_gib=config['max_tree_rss_gib'],max_allocated_gib=config['max_cuda_allocated_gib'],
        min_available_gib=config['minimum_available_ram_gib'],min_gpu_free_mib=config['minimum_dedicated_vram_free_mib'])
    try:
        with guard:
            train,dev,audit=split_data(args.data)
            report['data_audit']=audit
            write_json(args.out/'data_audit.json',audit)
            calibration=calibrate(train,config,args.out,guard)
            report['calibration']=calibration
            report['training_device']='cpu' if config.get('mms_mib',0) else calibration['selected_device']
            write_json(args.out/'training_report.json',report)
            if not args.calibrate_only:
                architectures=config['architectures']+([config['control']] if args.include_flat else [])
                for c in architectures:
                    for seed in config['seeds']:
                        receipt=fit_with_backoff(train,dev,m.Config(**c),seed,config,report['training_device'],args.out,guard,report['config_sha256'])
                        report['models'].append(receipt)
                        write_json(args.out/'training_report.json',report)
                        print('RAM02_FIT_COMPLETE',c['name'],seed,receipt['optimizer_steps'],flush=True)
        report['resources']=guard.receipt()
        report['status']='complete'
    except BaseException as exc:
        report['status']='aborted'
        report['error']=repr(exc)
        report['resources']=guard.receipt()
        write_json(args.out/'training_report.json',report)
        raise
    write_json(args.out/'training_report.json',report)

if __name__=='__main__': main()
