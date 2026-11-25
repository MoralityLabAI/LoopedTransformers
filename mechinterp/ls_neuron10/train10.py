"""LS10 direct Q4 regional synthesis from perturbed structural seeds."""
from train09 import centers, enclosure, decode, scales, step, rollout, torch, Path, json, time, hashlib, NAMES, QIDS
from verify import positive
import argparse
SEEDS=(307,311,313,317,331,337,347,349,353,359,367,373)
DOSES=(.01,.05,.15,.30)
ARMS=tuple(f'{kind}_{int(s*100):02}' for kind in ('struct','disrupt') for s in DOSES)+('random',)
STEPS=800

def template():
    w=positive();p=[torch.tensor(w[k],dtype=torch.float32) for k in NAMES]
    rms=[v.square().mean().sqrt().clamp_min(.125) for v in p]
    return p,rms

def init10(seed,arm):
    p,rms=template();g=torch.Generator().manual_seed(10010000+seed)
    noise=[torch.randn(v.shape,generator=g) for v in p]
    if arm=='random':
        out=[s*n for s,n in zip(rms,noise)];out[0][0]-=1
    else:
        kind,dose=arm.split('_');sigma=int(dose)/100
        if kind=='disrupt':p[4]=p[4][[2,3,0,1]]
        out=[v+sigma*s*n for v,s,n in zip(p,rms,noise)]
    return out

def eval10(ps,seed):
    g=torch.Generator().manual_seed(10030000+seed);rows={}
    for t in (8,128):
        x=torch.randint(2,(1,256,t,3),generator=g).float();y=x[:,:,:,:2].sum(-1).cumsum(-1).remainder(2)
        with torch.no_grad():h=rollout(ps,x);ok=(h>.5)==y.bool();finite=bool(torch.isfinite(h).all())
        rows[str(t)]={'accuracy':float(ok[:,:,-1].float().mean()),'all_prefix':float(ok.all(-1).float().mean()),'finite':finite,'final_correct':ok[0,:,-1].int().tolist(),'all_prefix_correct':ok[0].all(-1).int().tolist()}
    return rows

def train(arm,out):
    n=len(SEEDS);_,rms=template();raw=list(zip(*(init10(s,arm) for s in SEEDS)));ps=[torch.nn.Parameter(torch.stack(v)/r) for v,r in zip(raw,rms)];physical=lambda:[v*r for v,r in zip(ps,rms)]
    ls=[torch.nn.Parameter(s.log()) for s in scales(physical(),4)];opt=torch.optim.Adam([{'params':ps,'lr':.002},{'params':ls,'lr':.001}]);h,x,y=centers(n);out.mkdir(exist_ok=True,parents=True)
    for t in range(STEPS+1):
        if t in (0,STEPS):
            with torch.no_grad():
                p=physical();q,codes,ss=decode(p,4,ls);bound=enclosure(q,h,x,y).amax(1)
                for j,seed in enumerate(SEEDS):
                    pp=[v[j:j+1].detach() for v in p];qq=[v[j:j+1].detach() for v in q];r={'arm':arm,'seed':seed,'step':t,'fp32_weights':{k:v[0].tolist() for k,v in zip(NAMES,pp)},'q4_weights':{k:v[0].tolist() for k,v in zip(NAMES,qq)},'codes':{NAMES[i]:v[j].long().tolist() for i,v in zip(QIDS,codes)},'scales':{NAMES[i]:float(v[j].reshape(-1)[0]) for i,v in zip(QIDS,ss)},'float_bound_max':float(bound[j]),'q4':eval10(qq,seed)};(out/f'{arm}-{seed}-{t}.json').write_text(json.dumps(r,separators=(',',':'))+'\n')
                print(arm,t,'bound',bound.tolist(),flush=True)
        if t==STEPS:break
        if t==400:
            for group in opt.param_groups:group['lr']*=.5
        q,_,_=decode(physical(),4,ls,True);z=step(q,h,x);err=enclosure(q,h,x,y);loss=((z-y).square().mean(1)+10*(err-3/32).clamp_min(0).square().mean(1)).sum();assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();allp=ps+ls;norms=sum(v.grad.reshape(n,-1).square().sum(1) for v in allp).sqrt().clamp_min(1e-12);factor=(1/norms).clamp_max(1)
        for v in allp:v.grad.mul_(factor.reshape((n,)+(1,)*(v.ndim-1)))
        opt.step()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--arm',choices=ARMS,required=True);a.add_argument('--out',type=Path,default=Path('results'));args=a.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);train(args.arm,args.out)
