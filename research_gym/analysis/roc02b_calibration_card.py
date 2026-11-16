"""ROC-02B: calibration-only full rank-card and matched perturbation audit.

Registered in configs/roc02b_calibration_observer_card_v1.json before new ranks/nulls.
Uses the frozen ROC-02 train+calibration data and existing model checkpoints.
NEVER accesses a sealed test file, label, record, or derived target.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from research_gym.analysis.roc02_tied_train import TiedRuleModel, SEEDS, DEPTHS, WIDTH, tensors, rows
except ImportError:
    from roc02_tied_train import TiedRuleModel, SEEDS, DEPTHS, WIDTH, tensors, rows

RANKS = (1,2,3,4,7,8,12,16,32)
NULL_SEEDS = tuple(range(941001,941129))
SHADOW_INDICES=(0,1)
RADIUS = .2
CHECKPOINT_NAME='roc02_R{depth}_S{seed}_step240.pt'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stage_model(model, x, mask):
    """Return midpoint and final states, using identical float32 update order."""
    h=x.new_zeros(x.shape[0],WIDTH)
    for t in range(x.shape[1]):
        h=h+mask[:,t,None]*(.25*model.event_embed(x[:,t])+model.delta(h))
    for _ in range(model.depth//2):
        h=h+model.delta(h)
    mid=h.clone()
    for _ in range(model.depth//2):
        h=h+model.delta(h)
    return mid,h

def suffix(model,h):
    for _ in range(model.depth//2):
        h=h+model.delta(h)
    return h

def tangent(model,h):
    """Chronologically ordered 128x128 suffix derivative in float64."""
    w1=model.block_in.weight.detach().double()
    b1=model.block_in.bias.detach().double()
    w2=model.block_out.weight.detach().double()
    eye=torch.eye(WIDTH,dtype=torch.float64)
    z=h.detach().double().clone()
    J=eye.clone()
    for _ in range(model.depth//2):
        a=torch.tanh(w1@z+b1)
        J=(eye+.25*(w2*(1-a*a)[None,:])@w1)@J
        z=z+.25*(w2@a+model.block_out.bias.detach().double())
    return J

def orth(base, k):
    U=torch.linalg.qr(base[:,:k],mode='reduced')[0]
    return U

def observer_bases(train_h,train_labels):
    H=train_h.double()
    H=H-H.mean(0)
    Y=F.one_hot(train_labels,8).double()
    Y-=Y.mean(0)
    cross=H.T@Y
    cx=torch.linalg.svd(cross,full_matrices=False).U
    pca=torch.linalg.svd(H,full_matrices=False).Vh.T
    filled=torch.linalg.qr(torch.cat((cx,pca),dim=1),mode='reduced')[0]
    bases={}
    for k in RANKS:
        # Strongest cross-covariance rank-k; for k>8 fill with train-only PCA.
        # QR of a concatenated matrix retains the first 8-dimensional span.
        bases[('supervised',k)]=orth(filled,k)
        bases[('pca',k)]=orth(pca,k)
    return bases

def jacobian_metrics(D,U,head_j):
    k=U.shape[1]
    A=U.T@D@U
    UD=U.T@D
    DU=D@U
    asq=float(torch.sum(A*A))
    bsq=max(0.,float(torch.sum(UD*UD))-asq)
    csq=max(0.,float(torch.sum(DU*DU))-asq)
    Df=float(torch.linalg.norm(D))
    cap=(float(torch.sum((head_j@U)**2))/float(torch.sum(head_j**2))) if float(torch.sum(head_j**2))>1e-16 else None
    return {'A':math.sqrt(asq),'B':math.sqrt(bsq),'C':math.sqrt(csq),
            'a':math.sqrt(asq)/Df if Df>1e-16 else None,
            'b':math.sqrt(bsq)/Df if Df>1e-16 else None,
            'c':math.sqrt(csq)/Df if Df>1e-16 else None,
            'B_per_sqrtk':math.sqrt(bsq/k),
            'D_frob':Df,'s':cap}

def null_seed(cksha,idx,k,seed):
    digest=hashlib.sha256(f'{cksha}|{idx}|{k}|{seed}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8],'big')%(2**63-1)

def random_U(cksha,idx,k,seed):
    g=torch.Generator(device='cpu').manual_seed(null_seed(cksha,idx,k,seed))
    z=torch.randn(WIDTH,k,device='cpu',dtype=torch.float64,generator=g)
    return torch.linalg.qr(z,mode='reduced')[0]

def fingerprint(U):
    # 16x16 projector patch makes basis-gauge/sign ambiguity harmless.
    P=U[:16,:]@U[:16,:].T
    return hashlib.sha256(P.float().numpy().tobytes()).hexdigest()

def empirical(x,arr,low):
    if x is None or any(a is None for a in arr) or len(arr)!=128:
        return None
    count=sum(a<=x if low else a>=x for a in arr)
    return (1+count)/129.

def holm(p):
    order=sorted((x,i) for i,x in enumerate(p) if x is not None)
    out=[None]*len(p)
    max_so_far=0.
    # Holm over ALL SIX, including undefined tests: undefined abstains;
    # keep family size 6 for conservative adjustment.
    for rank,(v,i) in enumerate(order):
        max_so_far=max(max_so_far,min(1.,(6-rank)*v))
        out[i]=max_so_far
    return out

def null_calibrate(actual,randoms):
    keys=('a','b','c','s')
    vals={k:sorted([x[k] for x in randoms if x[k] is not None]) for k in keys}
    meds={k:(v[63]+v[64])/2 if len(v)==128 else None for k,v in vals.items()}
    p=[empirical(actual['a'],vals['a'],True),empirical(actual['b'],vals['b'],True),
       empirical(actual['c'],vals['c'],True),empirical(actual['b'],vals['b'],False),
       empirical(actual['s'],vals['s'],False),empirical(actual['s'],vals['s'],True)]
    q=holm(p)
    r={k:(actual[k]/meds[k] if actual[k] is not None and meds[k] not in (None,0) else None) for k in ('a','b','c')}
    s=actual['s'];ms=meds['s']
    if s is not None and ms is not None and ms>0 and ms<1:
        r['s_low']=s/ms; r['s_high']=(s-ms)/(1-ms)
    else:
        r['s_low']=r['s_high']=None
    ab=lambda idx,condition: q[idx] is not None and q[idx]<=.05 and condition
    tags=[]
    if ab(1,r['b'] is not None and r['b']<=.2):tags.append('closed_quotient')
    if ab(2,r['c'] is not None and r['c']<=.2):tags.append('invariant_subspace')
    if ab(0,r['a'] is not None and r['a']<=.2) and ab(1,r['b'] is not None and r['b']<=.2):
        if ab(4,r['s_high'] is not None and r['s_high']>=.8):tags.append('latched_memory')
        if ab(5,r['s_low'] is not None and r['s_low']<=.2):tags.append('irrelevant')
    if ab(3,r['b'] is not None and r['b']>=1.2) and ab(0,r['a'] is not None and r['a']<=.2) and ab(2,r['c'] is not None and r['c']<=.2):tags.append('output_register')
    return {'medians':meds,'ratios':r,'p':p,'q':q,'tags':tags or ['unclassified']}

def ridge_predict(train_h,train_y,eval_h,eval_y,U):
    U=U.float()
    center=train_h.mean(0)
    trainZ=(train_h-center)@U
    valZ=(eval_h-center)@U
    target=F.one_hot(train_y,8).float()
    meanY=target.mean(0)
    eye=torch.eye(U.shape[1],device=U.device)
    head=torch.linalg.solve(trainZ.T@trainZ+eye,trainZ.T@(target-meanY))
    out=meanY+valZ@head
    err=float(F.mse_loss(out,F.one_hot(eval_y,8).float()))
    acc=float((out.argmax(-1)==eval_y).float().mean())
    return err,acc

def direction(U,arm,g):
    k=U.shape[1]
    if arm=='observed':
        v=torch.randn(k,dtype=torch.float64,generator=g)
        return U@v
    raw=torch.randn(WIDTH,dtype=torch.float64,generator=g)
    if arm=='dark':raw=raw-U@(U.T@raw)
    return raw

def causal_interventions(model,h0,U,cksha,idx,k,T,truth):
    output={}
    h0=h0.detach().float()
    with torch.no_grad():
        final=suffix(model,h0.unsqueeze(0))[0]
        orig_logits=model.readout(final)
        def margin(z):
            return float(z[truth]-torch.max(torch.cat([z[:truth],z[truth+1:]])))
        base_margin=margin(orig_logits)
        frozen_R=model.readout.weight.detach().double()
        for arm in ('observed','dark','isotropic'):
            token=f'{cksha}|{idx}|{k}|causal|{arm}'
            g=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'big')%(2**63-1))
            v=direction(U,arm,g)
            norm=float(torch.linalg.norm(v))
            if norm<1e-9:raise ValueError('invalid random direction')
            delta=(RADIUS/norm*v).float()
            if abs(float(torch.linalg.norm(delta))-RADIUS)>1e-6:raise AssertionError('norm mismatch')
            moved=suffix(model,(h0+delta).unsqueeze(0))[0]
            dz=(moved-final).double()
            dlog=model.readout(moved)-orig_logits
            output[arm+'_delta_logits']=float(torch.linalg.norm(dlog))
            output[arm+'_delta_obs']=float(torch.linalg.norm(U.T@dz))
            output[arm+'_delta_dark']=float(torch.linalg.norm(dz-U@(U.T@dz)))
            predicted=frozen_R@(T@delta.double())
            output[arm+'_predicted_delta_logits']=float(torch.linalg.norm(predicted))
            output[arm+'_tangent_residual_logits']=float(torch.linalg.norm(dlog.double()-predicted))
            output[arm+'_task_margin_delta']=margin(model.readout(moved))-base_margin
        # Readout ablation is a separate control; it is not norm-matched.
        ablated=final.double()-U@(U.T@final.double())
        output['readout_ablate_delta_logits']=float(torch.linalg.norm(model.readout(ablated.float())-orig_logits))
    return output

def load_and_verify(seed,depth,source_dir,ckdir):
    ck=Path(ckdir)/CHECKPOINT_NAME.format(seed=seed,depth=depth)
    name=f'roc02_R{depth}_S{seed}.json'
    candidates=[Path(ckdir).parent/'receipts'/name, Path(source_dir)/'train_full'/name, Path(ckdir).parent/name]
    receipt=next((p for p in candidates if p.is_file()),None)
    if receipt is None:raise FileNotFoundError('ROC-02 checkpoint receipt required')
    a=json.loads(receipt.read_text())
    checksum=sha(ck)
    if checksum!=a['checkpoint_sha256']:raise RuntimeError('SHA mismatch '+ck.name)
    obj=torch.load(ck,weights_only=True,map_location='cpu')
    if (obj['seed'],obj['depth'],obj['steps'])!=(seed,depth,240):raise RuntimeError('checkpoint identity')
    for split in ('train','calibration'):
        if sha(Path(source_dir)/'data'/f'{split}_inputs.jsonl')!=obj['source_sha256'][split]:
            raise RuntimeError('source mismatch '+split)
    torch.manual_seed(seed)
    model=TiedRuleModel(depth)
    model.load_state_dict(obj['state_dict'])
    model.eval()
    return ck,checksum,model

def one(seed,depth,source_dir,ckdir,outdir):
    torch.set_num_threads(1)
    ck,checksum,model=load_and_verify(seed,depth,source_dir,ckdir)
    train=tensors(rows(Path(source_dir)/'data'/'train_inputs.jsonl'))
    dev=tensors(rows(Path(source_dir)/'data'/'calibration_inputs.jsonl'))
    with torch.no_grad():
        ht,_=stage_model(model,train[0],train[1]);hd,_=stage_model(model,dev[0],dev[1])
    bases=observer_bases(ht,train[2])
    modelscore=float((model(*dev[:2]).argmax(-1)==dev[2]).float().mean())
    A=[];B=[];N=[];D=[]
    for idx in SHADOW_INDICES:
        J=tangent(model,hd[idx]);T=J;Dmat=J-torch.eye(WIDTH,dtype=torch.float64)
        R=model.readout.weight.detach().double()
        GR=R@T
        hfloat=hd[idx]
        for k in RANKS:
            allnull=[]
            basis_null={}
            for nullseed in NULL_SEEDS:
                U=random_U(checksum,idx,k,nullseed)
                metrics=jacobian_metrics(Dmat,U,GR)
                metrics.update({'seed':seed,'depth':depth,'state_index':idx,'rank':k,
                                'null_seed':nullseed,'subspace_sha256':fingerprint(U),
                                'checkpoint_sha256':checksum})
                allnull.append(metrics)
            if len({x['subspace_sha256'] for x in allnull})<100:raise AssertionError('non-distinct nulls')
            N.extend(allnull)
            for arm in ('supervised','pca'):
                U=bases[(arm,k)]
                m=jacobian_metrics(Dmat,U,GR)
                calibrated=null_calibrate(m,allnull)
                m.update({'seed':seed,'depth':depth,'state_index':idx,'rank':k,'arm':arm,
                          'checkpoint_sha256':checksum,
                          'a_to_null':calibrated['ratios']['a'],
                          'b_to_null':calibrated['ratios']['b'],
                          'c_to_null':calibrated['ratios']['c'],
                          's_to_null':(m['s']/calibrated['medians']['s'] if calibrated['medians']['s'] else None),
                          'p':calibrated['p'],'q':calibrated['q'],'tags':calibrated['tags'],
                          'null_median_B':calibrated['medians']['b'],
                          'null_median_s':calibrated['medians']['s']})
                m.update(causal_interventions(model,hfloat,U,checksum,idx,k,T,int(dev[2][idx])))
                B.append(m)
            D.append({'seed':seed,'depth':depth,'rank':k,'state_index':idx,'D_frob':float(torch.linalg.norm(Dmat)),
                      'H_trace':float(torch.sum(GR*GR))})
    for arm in ('supervised','pca'):
        for k in RANKS:
            U=bases[(arm,k)]
            mse,acc=ridge_predict(ht,train[2],hd,dev[2],U)
            sub=[r for r in B if r['arm']==arm and r['rank']==k]
            A.append({'seed':seed,'depth':depth,'rank':k,'arm':arm,'calibration_mse':mse,'calibration_accuracy':acc,
                      'original_network_accuracy':modelscore,
                      'B_per_sqrtk':sum(r['B_per_sqrtk'] for r in sub)/len(sub),
                      'B_to_null':sum(r['b_to_null'] for r in sub)/len(sub),
                      'H_capture':sum(r['s'] for r in sub)/len(sub),
                      'checkpoint_sha256':checksum})
    outdir.mkdir(parents=True,exist_ok=True)
    payload={'study':'ROC-02B-calibration-only-full-observer-card-v1','seed':seed,'depth':depth,
             'checkpoint_sha256':checksum,'calibration_only':True,'shadow_indices':list(SHADOW_INDICES),
             'cells':A,'observed_states':B,'nulls':N,'base_norms':D}
    path=outdir/f'roc02b_R{depth}_S{seed}.json.gz'
    with path.open('wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,compresslevel=6,mtime=0) as gz:
            with io.TextIOWrapper(gz,encoding='utf-8') as txt:
                json.dump(payload,txt,sort_keys=True,separators=(',',':'))
    return A,B,N,D,sha(path),modelscore

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',required=True,type=int,choices=SEEDS)
    p.add_argument('--depth',required=True,type=int,choices=DEPTHS)
    p.add_argument('--source-dir',required=True,type=Path)
    p.add_argument('--checkpoint-dir',required=True,type=Path)
    p.add_argument('--out-dir',required=True,type=Path)
    a=p.parse_args()
    cells,observed,nulls,original,checkpoint,accuracy=one(a.seed,a.depth,a.source_dir,a.checkpoint_dir,a.out_dir)
    print(json.dumps({'seed':a.seed,'depth':a.depth,'cells':len(cells),'observed':len(observed),
                      'nulls':len(nulls),'sha256':checkpoint,'model_cal_accuracy':accuracy},sort_keys=True))

if __name__=='__main__':main()