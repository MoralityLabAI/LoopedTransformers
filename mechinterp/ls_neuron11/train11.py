"""Fresh parity recovery replication and four-state Q4 modular transfer."""
import argparse, hashlib, itertools, json, math
from pathlib import Path
import torch
from train10 import template as parity_template
from train09 import decode,scales,step,NAMES,QIDS
PSEEDS=tuple(range(379,403)); MSEEDS=tuple(range(403,415)); STEPS=800
ARMS={'parity':('struct_05','disrupt_05'),'mod4':('struct_05','disrupt_05','struct_15','disrupt_15','random')}

def base(task):
    if task=='parity':return parity_template()[0],0.,2,None
    # h=p-5/2, s=h+a+b; 6 possible sums. Each adjacent jump is a two-GELU ramp.
    u=[];ub=[];down=[]
    for j,c in zip(range(-2,3),(1,1,1,-3,1)):
        u += [[30.,30.,30.,0.]]*2
        ub += [-30*j+10.,-30*j-10.]
        down += [c/4,-c/4]
    p=[torch.tensor(z,dtype=torch.float32) for z in ([-1.,0.,0.,0.],[-2.5],u,ub,down,[0.])]
    return p,-2.5,4,(.25,5.,10.,.25)

def init(task,seed,arm):
    p,off,m,design=base(task);rms=[v.square().mean().sqrt().clamp_min(.125) for v in p]
    # Replicate LS10 exactly for parity, including the noise generator domain.
    g=torch.Generator().manual_seed(10010000+seed if task=='parity' else 11010000+seed)
    noise=[torch.randn(v.shape,generator=g) for v in p]
    if arm=='random':
        out=[s*n for s,n in zip(rms,noise)];out[0][0]-=1
    else:
        kind,dose=arm.split('_');sigma=int(dose)/100
        if kind=='disrupt':p[4]=p[4][[2,3,0,1] if task=='parity' else [2,3,0,1,6,7,4,5,8,9]]
        out=[v+sigma*s*n for v,s,n in zip(p,rms,noise)]
    return out,rms

def centers(task,n):
    _,off,m,_=base(task);c=torch.tensor(list(itertools.product(range(m),(0,1),(0,1),(0,1))),dtype=torch.float32)
    p=c[:,0].repeat_interleave(16);x=c[:,1:].repeat_interleave(16,0);y=c[:,:3].sum(-1).remainder(m).repeat_interleave(16)+off
    h=p+off+torch.linspace(-.125+.125/16,.125-.125/16,16).repeat(len(c))
    return h.expand(n,-1),x.expand(n,-1,-1),y.expand(n,-1)

def enclosure(q,h,x,y):
    z,j=step(q,h,x,True);curv=.2*(q[4].abs()*q[2][:,:,0].square()).sum(-1,keepdim=True)
    return (z-y).abs()+j.abs()*(.125/16)+.5*curv*(.125/16)**2

def evaluate(q,task,seed):
    _,off,m,_=base(task);g=torch.Generator().manual_seed(11030000+seed);r={}
    for t in (8,128):
        x=torch.randint(2,(1,256,t,3),generator=g).float();y=x[:,:,:,:2].sum(-1).cumsum(-1).remainder(m);h=torch.full(x.shape[:2],off);ok=[];finite=True
        with torch.no_grad():
            for j in range(t):
                h=step(q,h,x[:,:,j]);finite=finite and bool(torch.isfinite(h).all());pred=(h-off+.5).floor().clamp(0,m-1) if m>2 else (h>.5).float();ok.append((pred==y[:,:,j])&torch.isfinite(h))
        o=torch.stack(ok,2);r[str(t)]={'accuracy':float(o[:,:,-1].float().mean()),'all_prefix':float(o.all(-1).float().mean()),'finite':finite,'final_correct':o[0,:,-1].int().tolist(),'all_prefix_correct':o[0].all(-1).int().tolist()}
    return r

def train(task,arm,out):
    seeds=PSEEDS if task=='parity' else MSEEDS;n=len(seeds);_,rms=init(task,seeds[0],arm);raw=list(zip(*(init(task,s,arm)[0] for s in seeds)));ps=[torch.nn.Parameter(torch.stack(v)/r) for v,r in zip(raw,rms)];physical=lambda:[v*r for v,r in zip(ps,rms)]
    design=base(task)[3]; ss=scales(physical(),4) if design is None else [torch.full_like(s,v) for s,v in zip(scales(physical(),4),design)];ls=[torch.nn.Parameter(s.log()) for s in ss];opt=torch.optim.Adam([{'params':ps,'lr':.002},{'params':ls,'lr':.001}]);h,x,y=centers(task,n);out.mkdir(exist_ok=True,parents=True)
    for t in range(STEPS+1):
        if t in (0,STEPS):
            with torch.no_grad():
                p=physical();q,codes,s=decode(p,4,ls);bounds=enclosure(q,h,x,y).amax(1)
                for j,seed in enumerate(seeds):
                    row={'task':task,'arm':arm,'seed':seed,'step':t,'fp32_weights':{k:v[j].tolist() for k,v in zip(NAMES,p)},'q4_weights':{k:v[j].tolist() for k,v in zip(NAMES,q)},'codes':{NAMES[i]:v[j].long().tolist() for i,v in zip(QIDS,codes)},'scales':{NAMES[i]:float(v[j].reshape(-1)[0]) for i,v in zip(QIDS,s)},'bound':float(bounds[j]),'eval':evaluate([v[j:j+1] for v in q],task,seed)}
                    (out/f'{task}-{arm}-{seed}-{t}.json').write_text(json.dumps(row,separators=(',',':'))+'\n')
                print(task,arm,t,'bound passes',int((bounds<=3/32).sum()),'median bound',float(bounds.median()),flush=True)
        if t==STEPS:break
        if t==400:
            for group in opt.param_groups:group['lr']*=.5
        q,_,_=decode(physical(),4,ls,True);z=step(q,h,x);err=enclosure(q,h,x,y);loss=((z-y).square().mean(1)+10*(err-3/32).clamp_min(0).square().mean(1)).sum();assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();ap=ps+ls;norm=sum(v.grad.reshape(n,-1).square().sum(1) for v in ap).sqrt().clamp_min(1e-12);f=(1/norm).clamp_max(1)
        for v in ap:v.grad.mul_(f.reshape((n,)+(1,)*(v.ndim-1)))
        opt.step()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--task',choices=ARMS,required=True);a.add_argument('--arm',required=True);a.add_argument('--out',type=Path,default=Path('results'));a=a.parse_args();assert a.arm in ARMS[a.task];torch.set_num_threads(1);torch.use_deterministic_algorithms(True);train(a.task,a.arm,a.out)
