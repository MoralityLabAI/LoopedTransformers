"""PEF-14: new-seed distribution-aware quantization calibration.

This study uses six freshly trained trunks, a fresh context split, and
probe-only forecasts sealed before any heldout model-state inference.
Protocol first committed at e63ad7c93109713decdd69ad662c52e916d0a8b3.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from research.pef07 import pef07 as p7

REGISTRATION = 'e63ad7c93109713decdd69ad662c52e916d0a8b3'
PARENT = '2bb8ec38edfadaac1c6927e6577bf21563d5af9c'
SALT = b'pef14-independent-v1:2026-11-15:'
SEEDS = (211, 223, 239)
TASKS = ('coalition', 'threshold')
COUNTS = {'train': 160, 'probe': 48, 'heldout': 48}
TANGENTS = ('random', 'pair01')
ANGLES = (0.25, 4.0)
STEPS = (0.009, 0.018, 0.045, 0.07)
TRAIN_STEPS = 450
D = 32
N = 768


def canonical(x):
    return (json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha_bytes(x):
    return hashlib.sha256(x).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def tensor_hash(t):
    return sha_bytes(t.detach().cpu().contiguous().numpy().tobytes())


def split_contexts():
    contexts = [tuple((n >> i)&1 for i in range(8)) for n in range(256)]
    contexts.sort(key=lambda c: (hashlib.sha256(SALT+bytes(c)).digest(), bytes(c)))
    result = {k: [] for k in COUNTS}
    assignment = {}
    cur = 0
    for kind, count in COUNTS.items():
        for c in contexts[cur:cur+count]:
            assignment[''.join(map(str,c))] = kind
            result[kind].extend(tuple((m >> i)&1 for i in range(4))+c for m in range(16))
        cur += count
    assert cur == 256
    groups = [set(result[k]) for k in COUNTS]
    assert len(set.union(*groups)) == 4096 and sum(map(len, groups)) == 4096
    for kind, rows in result.items():
        assert len(rows) == 16*COUNTS[kind]
        for j in range(0, len(rows), 16):
            assert len({x[4:] for x in rows[j:j+16]}) == 1
            assert len({x[:4] for x in rows[j:j+16]}) == 16
    manifest = {'campaign': 'PEF-14', 'salt': SALT.decode(), 'context_counts': COUNTS,
                'history_counts': {k: len(v) for k,v in result.items()},
                'sorted_contexts': [''.join(map(str,c)) for c in contexts],
                'context_assignment': assignment, 'focal_mask_order': list(range(16)),
                'closed_focal_hypercube': True}
    manifest['sha256'] = sha_bytes(canonical(manifest))
    return {k: torch.tensor(rows, dtype=torch.long) for k,rows in result.items()}, manifest


def parent_model():
    """PEF-07 retains two equivalent names in repo and local evidence releases."""
    model_type=getattr(p7,'CoalitionMemoryLooper',None)
    if model_type is None:
        model_type=p7.Looper
    return model_type()


def model_digest(model):
    routine=getattr(p7,'tensor_hash',None)
    if routine is None:
        routine=p7.model_digest
    return routine(model)


def ridge_digest(fit):
    routine=getattr(p7,'digest_ridge',None)
    if routine is None:
        routine=p7.ridge_digest
    return routine(fit)


def class_truth(bits, task):
    routine=getattr(p7,'targets',None)
    if routine is None:
        routine=p7.target
    return routine(bits,task)


def frozen_path(out, seed, task):
    return Path(out)/f'seed{seed}.{task}.pef14.frozen.pt'


def train_single(seed, task, data, out):
    """Exactly the PEF-07 schedule; resume only verified finished checkpoints."""
    existing = frozen_path(out,seed,task)
    if existing.exists():
        payload = torch.load(existing,map_location='cpu',weights_only=True)
        if (payload['seed'],payload['task'],payload['steps']) != (seed,task,TRAIN_STEPS):
            raise AssertionError('cached checkpoint metadata mismatch')
        model = parent_model().eval()
        model.load_state_dict(payload['model'],strict=True)
        if model_digest(model)!=payload['final_sha256']:
            raise AssertionError('cached final parameter digest mismatch')
        opt_state=payload['optimizer']['state']
        if len(opt_state)!=len(list(model.parameters())) or not all(
                int(t['step'].item())==TRAIN_STEPS for t in opt_state.values()):
            raise AssertionError('cached optimizer has incomplete training steps')
        return {'seed':seed,'task':task,'steps':TRAIN_STEPS,'file':existing.name,
                'checkpoint_sha256':sha_file(existing),
                'initial_weights_sha256':payload['initial_sha256'],
                'frozen_weights_sha256':payload['final_sha256'],
                'first_train_loss':None,'final_train_loss':None,
                'loss_receipts_available':False,'verified_finished_checkpoint_reused':True}
    torch.manual_seed(seed + (10000 if task=='threshold' else 0))
    model = parent_model().train()
    initial_hash = model_digest(model)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0.0)
    gen = torch.Generator(device='cpu').manual_seed(seed*7919+59+(100000 if task=='threshold' else 0))
    first_loss = last_loss = None
    for step in range(TRAIN_STEPS):
        batch = data[torch.randint(len(data), (128,), generator=gen)]
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model.logits(batch), class_truth(batch,task))
        if not bool(torch.isfinite(loss)):
            raise AssertionError('nonfinite loss')
        if step == 0:
            first_loss = float(loss.detach())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(norm)):
            raise AssertionError('nonfinite grad')
        opt.step()
        last_loss = float(loss.detach())
    model.eval()
    weights_hash = model_digest(model)
    path = frozen_path(out,seed,task)
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'seed': seed, 'task': task, 'steps':TRAIN_STEPS,
                'initial_sha256': initial_hash, 'final_sha256': weights_hash},path)
    return {'seed':seed,'task':task,'steps':TRAIN_STEPS,
            'file':path.name,'checkpoint_sha256':sha_file(path),
            'initial_weights_sha256':initial_hash,'frozen_weights_sha256':weights_hash,
            'first_train_loss':first_loss,'final_train_loss':last_loss}


def checked_model(out, record):
    path=Path(out)/record['file']
    if sha_file(path)!=record['checkpoint_sha256']:
        raise AssertionError('checkpoint byte SHA256 mismatch')
    source = torch.load(path, map_location='cpu', weights_only=True)
    if (source['seed'],source['task'],source['steps']) != (record['seed'],record['task'],TRAIN_STEPS):
        raise AssertionError('checkpoint provenance mismatch')
    model = parent_model().eval()
    model.load_state_dict(source['model'], strict=True)
    if model_digest(model)!=record['frozen_weights_sha256']:
        raise AssertionError('checkpoint parameter SHA256 mismatch')
    return model


@torch.no_grad()
def at_eight(model, bits):
    h = model.memory_init[None,:].expand(len(bits), -1)
    for k in range(8):
        h = model.step(h,model.bit_embedding(bits[:,k]))
    return h


def after_eight(model, bits, start):
    h = start
    for k in range(8,12):
        h = model.step(h,model.bit_embedding(bits[:,k]))
    return h


def stable_sign(v):
    v=v.clone()
    return -v if float(v[int(v.abs().argmax())])<0 else v


def unit(v):
    n = v.norm()
    if not torch.isfinite(n) or float(n)<1e-12:
        raise AssertionError('degenerate direction')
    return v/n


def canonical_orthogonal(g):
    candidates = torch.eye(D,dtype=torch.float64)
    orthos=candidates-(candidates@g)[:,None]*g[None,:]
    idx=int(orthos.norm(dim=1).argmax())
    return stable_sign(unit(orthos[idx]))


def geometry(h8, seed, task):
    """Independent of labels and output-head weights."""
    x=h8.double()
    mu=x.mean(0)
    _,singular,right=torch.linalg.svd(x-mu,full_matrices=False)
    g=stable_sign(unit(right[0]))
    cube=x.reshape(-1,16,D)
    signals=torch.stack([cube[:,mask+3]-cube[:,mask+1]-cube[:,mask+2]+cube[:,mask]
                         for mask in (0,4,8,12)],dim=1).reshape(-1,D)
    _,pair_sing,pair_right=torch.linalg.svd(signals,full_matrices=False)
    pv=unit(pair_right[0]) if float(pair_sing[0])>=1e-10 else canonical_orthogonal(g)
    pv=pv if float(pv@g)>=0 else -pv
    resid = pv-(pv@g)*g
    pair_degenerate=float(resid.norm())<1e-9 or float(pair_sing[0])<1e-10
    u_pair = canonical_orthogonal(g) if pair_degenerate else unit(resid)
    gen=torch.Generator(device='cpu').manual_seed(14014+1000*seed+(100000 if task=='threshold' else 0))
    raw=torch.randn(D, generator=gen, dtype=torch.float64)
    random_ortho=raw-(raw@g)*g
    u_random=canonical_orthogonal(g) if float(random_ortho.norm())<1e-9 else unit(random_ortho)
    scale=float((((x-mu)@g)**2).mean().sqrt())
    if scale<=1e-10:
        raise AssertionError('probe principal-component variance vanished')
    return {'mu':mu,'g':g,'tangents':{'random':u_random,'pair01':u_pair},
            'pc_scale':scale,'pair_degenerate':pair_degenerate,
            'pc_singular':float(singular[0]),'pair_singular':float(pair_sing[0])}


def rotate(g,u,angle):
    radians=math.radians(angle)
    v=unit(math.cos(radians)*g+math.sin(radians)*u)
    w=unit(-math.sin(radians)*g+math.cos(radians)*u)
    if abs(float(v@w))>1e-9:
        raise AssertionError('invalid orthogonal angle basis')
    return v,w


def project(x,mu,v):
    x=x.double()
    return (x-((x-mu)@v)[:,None]*v[None,:]).float()


def ridge_margin(states,fit):
    avg,std,beta=fit
    x=(states.double()-avg)/std
    return x@beta[:-1]+beta[-1]-.5


def probe_margin_j(model,bits,z,fit,w):
    with torch.enable_grad():
        active=z.detach().requires_grad_(True)
        margin=ridge_margin(after_eight(model,bits,active),fit)
        grad=torch.autograd.grad(margin.sum(),active)[0]
    return margin.detach().double(), grad.detach().double()@w


def quant_error(q,step):
    if not math.isfinite(step) or step<=0:
        raise ValueError('positive finite quantizer step required')
    return step*torch.round(q.double()/step)-q.double()


def uniform_probability(f,j,y,step):
    signed=(2*y.long()-1).double()*f.double()
    coeff=j.double().abs()
    raw=(.5 + signed/(step*coeff.clamp_min(1e-300))).clamp(0,1)
    return torch.where(coeff<=1e-14, ((f>=0).long()==y.long()).double(), raw)


def phase_forecasts(f,j,y,q,step):
    """All 16 focal assignments kept inside every one of 48 contexts."""
    if not (len(f)==len(j)==len(y)==len(q)==N):
        raise ValueError('expected 768 complete probe cells')
    d=quant_error(q,step).reshape(48,16)
    a=f.double().reshape(48,16)
    b=j.double().reshape(48,16)
    truth=y.long().reshape(48,16)
    own=((a+b*d>=0).long()==truth).double()
    uniform=uniform_probability(f,j,y,step).reshape(48,16)
    context_cross=torch.empty((48,16),dtype=torch.float64)
    other=~torch.eye(48,dtype=torch.bool)
    for mask in range(16):
        linear=a[:,mask,None]+b[:,mask,None]*d[None,:,mask]
        classes=((linear>=0).long()==truth[:,mask,None]).double()
        context_cross[:,mask]=classes[other].reshape(48,47).mean(1)
    mixed=(47*context_cross+8*uniform)/55
    per_context=mixed.mean(1)
    fraction=float(per_context.mean())
    sd=float(per_context.std(unbiased=True))
    halfwidth=1.645*math.sqrt(2)*sd/math.sqrt(48)
    return {'distribution_mixture':fraction,
            'conditional_cross':float(context_cross.mean()),
            'uniform_phase':float(uniform.mean()),
            'exact_probe_phase':float(own.mean()),
            'clean_probe_constant':float(((f>=0).long()==y.long()).double().mean()),
            'interval_90_low':max(0.0,fraction-halfwidth),
            'interval_90_high':min(1.0,fraction+halfwidth),
            'mixture_context_probabilities':per_context.tolist(),
            'probe_phase_mean':float(d.mean()),
            'probe_phase_std':float(d.std(unbiased=False))}


def freeze_training(catalog,manifest,root):
    records=[]
    for seed in SEEDS:
        for task in TASKS:
            rec=train_single(seed,task,catalog['train'],root)
            rec['source_sha256']=manifest['sha256']
            records.append(rec)
            print('PEF14_SEALED_MODEL',seed,task,rec['checkpoint_sha256'],flush=True)
    ledger={'campaign':'PEF-14','registration_commit':REGISTRATION,'source_sha256':manifest['sha256'],
            'all_six_models_sealed_before_probe':True,'seeds':list(SEEDS),
            'tasks':list(TASKS),'steps':TRAIN_STEPS,'torch_version':str(torch.__version__),
            'cpu_capability':torch.backends.cpu.get_cpu_capability(),
            'onednn_max_cpu_isa':os.environ.get('DNNL_MAX_CPU_ISA'),
            'checkpoints':records}
    path=root/'fresh_training_ledger.json'
    path.write_bytes(canonical(ledger))
    return ledger


def freeze_probes(catalog,train_ledger,root):
    """This function never references catalog['heldout']."""
    probe=catalog['probe']
    records=[]
    for rec in train_ledger['checkpoints']:
        seed,task=rec['seed'],rec['task']
        model=checked_model(root,rec)
        weights_before=model_digest(model)
        h8=at_eight(model,probe)
        info=geometry(h8,seed,task)
        truth=class_truth(probe,task)
        conditions=[]
        for tangent in TANGENTS:
            u=info['tangents'][tangent]
            for angle in ANGLES:
                v,w=rotate(info['g'],u,angle)
                z=project(h8,info['mu'],v)
                with torch.no_grad():
                    final=after_eight(model,probe,z)
                    fit=p7.fit_ridge(final,truth)
                f,j=probe_margin_j(model,probe,z,fit,w)
                q=(z.double()-info['mu'])@w
                if not (torch.isfinite(f).all() and torch.isfinite(j).all() and torch.isfinite(q).all()):
                    raise AssertionError('nonfinite probe features')
                cond={'tangent':tangent,'angle_deg':angle,'projector_v':v.tolist(),
                      'retained_w':w.tolist(), 'fit':{'mean':fit[0].tolist(),
                      'std':fit[1].tolist(),'beta':fit[2].tolist()},
                      'fit_sha256':ridge_digest(fit),
                      'q':q.tolist(),'margin':f.tolist(),'jacobian':j.tolist(),
                      'labels':truth.tolist(),'forecasts':[]}
                for relative in STEPS:
                    pred=phase_forecasts(f,j,truth,q,relative*info['pc_scale'])
                    cond['forecasts'].append({'relative_step':relative,**pred})
                conditions.append(cond)
        if model_digest(model)!=weights_before:
            raise AssertionError('probe estimation modified frozen trunk')
        records.append({'seed':seed,'task':task,'checkpoint_sha256':rec['checkpoint_sha256'],
                        'center':info['mu'].tolist(),'global_pc':info['g'].tolist(),
                        'tangents':{n:info['tangents'][n].tolist() for n in TANGENTS},
                        'pair_degenerate':info['pair_degenerate'],
                        'pc_singular':info['pc_singular'],'pair_singular':info['pair_singular'],
                        'pc_scale':info['pc_scale'],'conditions':conditions})
        print('PEF14_PROBE_READY',seed,task,flush=True)
    ledger={'campaign':'PEF-14','registration_commit':REGISTRATION,
            'training_ledger_sha256':sha_file(root/'fresh_training_ledger.json'),
            'source_sha256':train_ledger['source_sha256'],
            'probe_histories':768,'heldout_histories':768,'forecast_sealed_before_heldout':True,
            'seeds':list(SEEDS),'tangents':list(TANGENTS), 'angles_deg':list(ANGLES),
            'relative_steps':list(STEPS),'models':records}
    path=root/'sealed_probe_forecasts.json'
    path.write_bytes(canonical(ledger))
    print('PEF14_ALL_FORECASTS_FROZEN',sha_file(path),flush=True)
    return ledger


def validate_frozen(root,catalog,manifest):
    training_path=root/'fresh_training_ledger.json'
    probe_path=root/'sealed_probe_forecasts.json'
    training=json.loads(training_path.read_text())
    probe=json.loads(probe_path.read_text())
    assert training['source_sha256']==probe['source_sha256']==manifest['sha256']
    assert probe['training_ledger_sha256']==sha_file(training_path)
    assert probe['forecast_sealed_before_heldout']
    assert len(training['checkpoints'])==len(probe['models'])==6
    assert {(r['seed'],r['task']) for r in training['checkpoints']}=={(s,t) for s in SEEDS for t in TASKS}
    for rec in training['checkpoints']:
        checked_model(root,rec)
        row=next(x for x in probe['models'] if (x['seed'],x['task'])==(rec['seed'],rec['task']))
        assert row['checkpoint_sha256']==rec['checkpoint_sha256'] and len(row['conditions'])==4
        for cond in row['conditions']:
            fit=tuple(torch.tensor(cond['fit'][k],dtype=torch.float64) for k in ('mean','std','beta'))
            assert ridge_digest(fit)==cond['fit_sha256']
            assert len(cond['forecasts'])==4
    return training,probe


def heldout_scoring(catalog,manifest,root):
    """Called only after the entire probe forecast ledger exists on disk."""
    training,probe=validate_frozen(root,catalog,manifest)
    held=catalog['heldout']
    nrows=[]
    models=[]
    for rec in training['checkpoints']:
        seed,task=rec['seed'],rec['task']
        model=checked_model(root,rec)
        before=model_digest(model)
        before_file=sha_file(root/rec['file'])
        prow=next(x for x in probe['models'] if (x['seed'],x['task'])==(seed,task))
        mu=torch.tensor(prow['center'],dtype=torch.float64)
        h8=at_eight(model,held)
        truth=class_truth(held,task)
        with torch.no_grad():
            native=(model.head(after_eight(model,held,h8)).argmax(1))
        original_correct=int((native==truth).sum())
        rows=[]
        for cond in prow['conditions']:
            v=torch.tensor(cond['projector_v'],dtype=torch.float64)
            w=torch.tensor(cond['retained_w'],dtype=torch.float64)
            fit=tuple(torch.tensor(cond['fit'][k],dtype=torch.float64) for k in ('mean','std','beta'))
            z=project(h8,mu,v)
            q=(z.double()-mu)@w
            with torch.no_grad():
                clean=(ridge_margin(after_eight(model,held,z),fit)>=0).long()
            clean_correct=int((clean==truth).sum())
            for forecast in cond['forecasts']:
                rel=forecast['relative_step']; step=rel*prow['pc_scale']
                d=quant_error(q,step)
                quantized=(z.double()+d[:,None]*w[None,:]).float()
                with torch.no_grad():
                    margin=ridge_margin(after_eight(model,held,quantized),fit)
                    predicted=(margin>=0).long()
                correct=int((predicted==truth).sum())
                wrong=(predicted!=truth).reshape(48,16).sum(0).tolist()
                val=correct/768
                subset={'seed':seed,'task':task,'tangent':cond['tangent'],
                        'angle_deg':cond['angle_deg'],'relative_step':rel,
                        'clean_correct':clean_correct,'native_correct':original_correct,
                        'correct':correct,'total':N,'accuracy':val,
                        'wrong_by_focal_mask':list(map(int,wrong)),
                        'class0_recall':float((predicted[truth==0]==0).float().mean()),
                        'class1_recall':float((predicted[truth==1]==1).float().mean()),
                        'mean_abs_heldout_phase_error':float(d.abs().mean()),
                        'forecast':{k:forecast[k] for k in ('distribution_mixture',
                          'uniform_phase','exact_probe_phase','conditional_cross',
                          'clean_probe_constant','interval_90_low','interval_90_high')}}
                rows.append(subset);nrows.append(subset)
        assert model_digest(model)==before and sha_file(root/rec['file'])==before_file
        models.append({'seed':seed,'task':task,'checkpoint_sha256':rec['checkpoint_sha256'],
                       'native_h12_correct':original_correct,'conditions':rows})
        print('PEF14_HELDOUT',seed,task,'native',original_correct,flush=True)
    all_decisions=[]
    for task in TASKS:
        task_seeds=[]
        for seed in SEEDS:
            records=[x for x in nrows if x['seed']==seed and x['task']==task]
            errors={name:sum(abs(x['forecast'][name]-x['accuracy']) for x in records)/len(records)
                    for name in ('distribution_mixture','uniform_phase','exact_probe_phase',
                                 'conditional_cross','clean_probe_constant')}
            covered=sum(x['forecast']['interval_90_low']<=x['accuracy']<=
                        x['forecast']['interval_90_high'] for x in records)
            success=errors['distribution_mixture']<=.030 and errors['distribution_mixture']<=.9*errors['uniform_phase']
            task_seeds.append({'seed':seed,'task':task,'n_conditions':len(records),
                               'mae':errors,'interval_90_covered':covered,
                               'registered_seed_criterion':bool(success)})
        all_decisions.extend(task_seeds)
    count=sum(x['registered_seed_criterion'] for x in all_decisions if x['task']=='coalition')
    summary={'campaign':'PEF-14','registered_protocol_commit':REGISTRATION,
             'parent_pef13_head':PARENT,'new_context_salt':SALT.decode(),
             'source_sha256':manifest['sha256'],
             'training_ledger_sha256':sha_file(root/'fresh_training_ledger.json'),
             'sealed_probe_forecast_sha256':sha_file(root/'sealed_probe_forecasts.json'),
             'all_six_models_frozen_before_probe':True,'all_forecasts_frozen_before_heldout':True,
             'training_seeds':list(SEEDS),'registered_successful_coalition_seeds':count,
             'primary_decision':'criterion_met' if count>=2 else 'criterion_not_met',
             'decisions':all_decisions,'models':models}
    (root/'summary.json').write_bytes(canonical(summary))
    with (root/'condition_scores.csv').open('w',newline='') as fh:
        writer=csv.writer(fh);writer.writerow(['seed','task','tangent','angle_deg','relative_step','correct',
            'clean_correct','native_correct','mixture','uniform','exact_probe','conditional','clean_constant','interval_low','interval_high'])
        for x in nrows:
            f=x['forecast'];writer.writerow([x['seed'],x['task'],x['tangent'],x['angle_deg'],x['relative_step'],
               x['correct'],x['clean_correct'],x['native_correct'],f['distribution_mixture'],
               f['uniform_phase'],f['exact_probe_phase'],f['conditional_cross'],
               f['clean_probe_constant'],f['interval_90_low'],f['interval_90_high']])
    return summary


def run(out):
    os.environ.setdefault('DNNL_MAX_CPU_ISA','AVX2')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    catalog,manifest=split_contexts()
    (root/'fresh_source_split.json').write_bytes(canonical(manifest))
    ledger=freeze_training(catalog,manifest,root)
    freeze_probes(catalog,ledger,root)
    result=heldout_scoring(catalog,manifest,root)
    print(json.dumps({'campaign':'PEF-14','decision':result['primary_decision'],
          'seeds':result['registered_successful_coalition_seeds'],
          'seed_mae':[r for r in result['decisions']]},indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    run(args.out)


if __name__=='__main__':
    main()
