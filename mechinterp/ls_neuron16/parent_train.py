"""LS07: scalar 4-GELU QAT, supervised and interval loss, paired ensemble."""
import argparse, hashlib, itertools, json, math, time
from pathlib import Path
import torch
import torch.nn.functional as F
SEEDS=(101,103,107,109,113,127,131,137,139,149,151,157)
ARMS=('qat','sample','bound')
STEPS=1800; BATCH=64; T=8; R=1/8; TARGET=3/32; K=8; AUX=5.
NAMES=('skip','skip_bias','up','up_bias','down','down_bias')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def init(seed):
    g=torch.Generator().manual_seed(seed)
    skip=torch.randn(4,generator=g)*.1;skip[0]-=.9
    return (skip,torch.zeros(1),torch.randn(4,4,generator=g)*2,torch.randn(4,generator=g)*2,torch.randn(4,generator=g)*.5,torch.zeros(1))
def quant(p,bits):
    dims=tuple(range(1,p.ndim));a=p.detach().abs().amax(dim=dims,keepdim=True);level=(1<<(bits-1))-1;s=(a/level).clamp_min(1e-12);return (p/s).round().clamp(-level,level)*s

def params(ps,bits,ste=False):
    if bits==32:return ps
    qs=[quant(p,bits) for p in ps]
    return [q.detach()+(p-p.detach()) for p,q in zip(ps,qs)] if ste else qs

def step(ps,h,x,deriv=False):
    sw,sb,u,ub,v,vb=ps
    z=torch.cat((h.unsqueeze(-1),x),-1)
    pre=torch.bmm(z,u.transpose(1,2))+ub[:,None,:]
    out=h+(z*sw[:,None,:]).sum(-1)+sb+.2*((F.gelu(pre)*v[:,None,:]).sum(-1)+vb)
    if not deriv:return out
    gp=.5*(1+torch.erf(pre/math.sqrt(2)))+pre*torch.exp(-pre.square()/2)/math.sqrt(2*math.pi)
    jac=1+sw[:,0,None]+.2*(gp*v[:,None,:]*u[:,None,:,0]).sum(-1)
    return out,jac

def rollout(ps,x):
    h=torch.zeros(x.shape[:2]);hs=[]
    for t in range(x.shape[2]):h=step(ps,h,x[:,:,t]);hs.append(h)
    return torch.stack(hs,2)

def train_batch(gs,t=T,n=BATCH):
    x=torch.stack([torch.randint(2,(n,t,3),generator=g) for g in gs]).float();y=x[:,:,:,:2].sum(-1).cumsum(-1).remainder(2);return x,y

def inv_inputs(n):
    a=torch.tensor(list(itertools.product((0.,1.),repeat=4)));p=a[:,0].repeat_interleave(K);x=a[:,1:].repeat_interleave(K,0);y=a[:,:3].sum(-1).remainder(2).repeat_interleave(K);centers=p+torch.linspace(-R+R/K,R-R/K,K).repeat(16)
    return centers.expand(n,-1),x.expand(n,-1,-1),y.expand(n,-1)

def bound(ps,h,x,y):
    o,j=step(ps,h,x,True);_,_,u,_,v,_=ps
    # |GELU''| <= 1 globally; Taylor remainder on each scalar subinterval.
    curv=.2*(v.abs()*u[:,:,0].square()).sum(-1,keepdim=True)
    return (o-y).abs()+j.abs()*(R/K)+.5*curv*(R/K)**2

def evaluate(ps,seed):
    g=torch.Generator().manual_seed(900000+seed);rows={}
    for t in (4,8,16,32,64):
        x=torch.randint(2,(1,512,t,3),generator=g).float();y=x[:,:,:,:2].sum(-1).cumsum(-1).remainder(2)
        with torch.no_grad():h=rollout(ps,x);ok=(h>.5)==y.bool();finite=torch.isfinite(h).all();rows[str(t)]={'accuracy':float(ok[:,:,-1].float().mean()),'all_prefix':float(ok.all(-1).float().mean()),'finite':bool(finite),'final_correct':ok[0,:,-1].int().tolist()}
    return rows

def train(arm,out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);n=len(SEEDS);gs=[torch.Generator().manual_seed(100000+s) for s in SEEDS];ig=[torch.Generator().manual_seed(200000+s) for s in SEEDS];raw=list(zip(*(init(s) for s in SEEDS)));ps=[torch.nn.Parameter(torch.stack(t)) for t in raw];o=torch.optim.Adam(ps,lr=.015);c,x,y=inv_inputs(n);sampler=[hashlib.sha256() for _ in SEEDS];start=time.monotonic()
    for t in range(STEPS):
        bits=32 if t<600 else 6 if t<900 else 4
        if t==1200:o.param_groups[0]['lr']=.005
        q=params(ps,bits,True);xx,yy=train_batch(gs)
        for j in range(n):sampler[j].update(xx[j].numpy().tobytes())
        h=rollout(q,xx);task=(h-yy).square().mean((1,2))
        jitter=torch.stack([torch.rand(16*K,generator=g)*2-1 for g in ig])*R/K
        if arm=='sample':excess=(step(q,c+jitter,x)-y).abs()
        elif arm=='bound':excess=bound(q,c,x,y)
        else:excess=torch.zeros_like(y)
        aux=(excess-TARGET).clamp_min(0).square().mean(1)
        loss=(task+AUX*aux).sum();assert torch.isfinite(loss), (arm,t)
        o.zero_grad();loss.backward()
        norms=sum(p.grad.reshape(n,-1).square().sum(1) for p in ps).sqrt().clamp_min(1e-12);scale=(1/norms).clamp_max(1)
        for p in ps:p.grad.mul_(scale.reshape((n,)+(1,)*(p.ndim-1)))
        o.step()
        if t+1 in (600,900,1200,1800):print(arm,t+1,'task',task.detach().tolist(),'aux',aux.detach().tolist(),flush=True)
    out.mkdir(parents=True,exist_ok=True);rows=[]
    for j,seed in enumerate(SEEDS):
        p=[v[j:j+1].detach().clone() for v in ps];q=params(p,4);weights={name:v[0].tolist() for name,v in zip(NAMES,q)};f=out/f'{arm}-{seed}.json';r={'arm':arm,'seed':seed,'sampler_sha256':sampler[j].hexdigest(),'fp32_weights':{name:v[0].tolist() for name,v in zip(NAMES,p)},'q4_weights':weights,'fp32':evaluate(p,seed),'q4':evaluate(q,seed)};f.write_text(json.dumps(r,separators=(',',':'))+'\n');rows.append(r)
    print(arm,'seconds',time.monotonic()-start,'T8 Q4',[round(r['q4']['8']['accuracy'],3) for r in rows],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--out',type=Path,default=Path('results'));a=p.parse_args();train(a.arm,a.out)
