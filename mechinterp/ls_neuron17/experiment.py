"""Frozen-code versus full QAT scale recovery for the reset/wrap mod4 template."""
import argparse, hashlib, itertools, json, math
from pathlib import Path
import torch
from construction import scalar_weights
from parent_train import step, NAMES
SEEDS=tuple(range(419,431)); DOSES=(5,15); ARMS=('frozen','scale','full'); STEPS=800; QIDS=(0,2,3,4)

def init(seed,dose):
    w,s=scalar_weights(4);p=[torch.tensor(w[k],dtype=torch.float32) for k in NAMES];rms=[v.square().mean().sqrt().clamp_min(.125) for v in p];g=torch.Generator().manual_seed(12010000+seed)
    return [v+dose/100*r*torch.randn(v.shape,generator=g) for v,r in zip(p,rms)],rms,s

def decode(ps,ls,codes=None,ste=False):
    out=list(ps);cs=[];ss=[]
    for j,i in enumerate(QIDS):
        s=ls[j].exp().clamp(1e-6,128.);ss.append(s);z=(ps[i]/s).clamp(-7,7);c=z.round() if codes is None else codes[j];cs.append(c.detach())
        out[i]=(z+(c-z).detach())*s if ste and codes is None else c*s
    return out,cs,ss

def centers(n):
    c=torch.tensor(list(itertools.product(range(4),(0,1),(0,1),(0,1))),dtype=torch.float32);p=c[:,0].repeat_interleave(16);x=c[:,1:].repeat_interleave(16,0);y=c[:,:3].sum(-1).remainder(4).repeat_interleave(16)-2
    h=p-2+torch.linspace(-.125+.125/16,.125-.125/16,16).repeat(len(c));return h.expand(n,-1),x.expand(n,-1,-1),y.expand(n,-1)

def enclosure(q,h,x,y):
    z,j=step(q,h,x,True);curv=.2*(q[4].abs()*q[2][:,:,0].square()).sum(-1,keepdim=True);return (z-y).abs()+j.abs()*(.125/16)+.5*curv*(.125/16)**2

def evaluate(q,seed):
    g=torch.Generator().manual_seed(12030000+seed);r={}
    with torch.no_grad():
        for t in (8,128):
            x=torch.randint(2,(1,256,t,3),generator=g).float();y=x[:,:,:,:2].sum(-1).cumsum(-1).remainder(4);h=torch.full(x.shape[:2],-2.);ok=[];finite=True
            for j in range(t):
                h=step(q,h,x[:,:,j]);finite=finite and bool(torch.isfinite(h).all());pred=(h+2.5).floor().clamp(0,3);ok.append((pred==y[:,:,j])&torch.isfinite(h))
            o=torch.stack(ok,2);r[str(t)]={'accuracy':float(o[:,:,-1].float().mean()),'all_prefix':float(o.all(-1).float().mean()),'finite':finite,'final_correct':o[0,:,-1].int().tolist(),'all_prefix_correct':o[0].all(-1).int().tolist()}
    return r

def train(dose,arm,out):
    n=len(SEEDS);_,rms,s=init(SEEDS[0],dose);raw=list(zip(*(init(seed,dose)[0] for seed in SEEDS)));ps=[torch.nn.Parameter(torch.stack(v)/r) for v,r in zip(raw,rms)];physical=lambda:[v*r for v,r in zip(ps,rms)]
    ls=[torch.nn.Parameter(torch.full((n,)+(1,)*(ps[i].ndim-1),math.log(v),dtype=torch.float32)) for i,v in zip(QIDS,s)];h,x,y=centers(n);out.mkdir(exist_ok=True,parents=True)
    initial=decode(physical(),ls)[1];fixed=[v.clone() for v in initial] if arm in ('scale','frozen') else None
    trainable=ps if arm=='full' else [ps[1],ps[5]]
    opt=torch.optim.Adam([{'params':trainable,'lr':.002},{'params':ls,'lr':.001}]);limit=0 if arm=='frozen' else STEPS
    for t in range(limit+1):
        if t in (0,limit):
            with torch.no_grad():
                p=physical();q,codes,ss=decode(p,ls,fixed);bounds=enclosure(q,h,x,y).amax(1)
                for j,seed in enumerate(SEEDS):
                    r={'dose':dose,'arm':arm,'seed':seed,'step':t,'fp32_weights':{k:v[j].tolist() for k,v in zip(NAMES,p)},'q4_weights':{k:v[j].tolist() for k,v in zip(NAMES,q)},'codes':{NAMES[i]:v[j].long().tolist() for i,v in zip(QIDS,codes)},'scales':{NAMES[i]:float(v[j].reshape(-1)[0]) for i,v in zip(QIDS,ss)},'bound':float(bounds[j]),'eval':evaluate([v[j:j+1] for v in q],seed)}
                    (out/f'{dose}-{arm}-{seed}-{t}.json').write_text(json.dumps(r,separators=(',',':'))+'\n')
                print(dose,arm,t,'bound passes',int((bounds<=3/32).sum()),'median',float(bounds.median()),flush=True)
        if t==limit:break
        if t==400:
            for group in opt.param_groups:group['lr']*=.5
        q,_,_=decode(physical(),ls,fixed,True);z=step(q,h,x);err=enclosure(q,h,x,y);loss=((z-y).square().mean(1)+10*(err-3/32).clamp_min(0).square().mean(1)).sum();assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();ap=trainable+ls;norm=sum(v.grad.reshape(n,-1).square().sum(1) for v in ap).sqrt().clamp_min(1e-12);f=(1/norm).clamp_max(1)
        for v in ap:v.grad.mul_(f.reshape((n,)+(1,)*(v.ndim-1)))
        opt.step()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--dose',type=int,choices=DOSES,required=True);a.add_argument('--arm',choices=ARMS,required=True);a.add_argument('--out',type=Path,default=Path('results'));v=a.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);train(v.dose,v.arm,v.out)
