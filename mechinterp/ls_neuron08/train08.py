"""LS08 paired factorial: mean/max enclosure penalty x fixed/learnable scale QAT."""
from parent_train import *
SEEDS=(163,167,173,179,181,191,193,197,199,211,223,227)
ARMS=('mean_fixed','max_fixed','mean_learn','max_learn')
STEPS=2400; BATCH=64; T=8; K08=16; AUX08=10.; TARGET08=3/32
QIDS=(0,2,3,4)
def scales(ps,bits):
    return [ps[i].detach().abs().amax(dim=tuple(range(1,ps[i].ndim)),keepdim=True).clamp_min(1e-7)/((1<<(bits-1))-1) for i in QIDS]
def decode(ps,bits,ls=None,ste=False):
    if bits==32:return list(ps),None,None
    lim=(1<<(bits-1))-1; ss=scales(ps,bits) if ls is None else [z.exp().clamp(1e-6,10.) for z in ls]
    out=list(ps);codes=[]
    for i,s in zip(QIDS,ss):
        u=ps[i]/s;cl=u.clamp(-lim,lim);c=cl.round();codes.append(c.detach())
        out[i]=(cl+(c-cl).detach())*s if ste else c*s
    return out,codes,ss
def centers(n):
    a=torch.tensor(list(itertools.product((0.,1.),repeat=4)));p=a[:,0].repeat_interleave(K08);x=a[:,1:].repeat_interleave(K08,0);y=a[:,:3].sum(-1).remainder(2).repeat_interleave(K08);c=p+torch.linspace(-R+R/K08,R-R/K08,K08).repeat(16)
    return c.expand(n,-1),x.expand(n,-1,-1),y.expand(n,-1)
def enclosure(ps,h,x,y):
    o,j=step(ps,h,x,True);_,_,u,_,v,_=ps
    curv=.2*(v.abs()*u[:,:,0].square()).sum(-1,keepdim=True)
    return (o-y).abs()+j.abs()*(R/K08)+.5*curv*(R/K08)**2
def eval08(ps,seed):
    g=torch.Generator().manual_seed(8030000+seed);rows={}
    for t in (4,8,16,32,64,128):
        x=torch.randint(2,(1,512,t,3),generator=g).float();y=x[:,:,:,:2].sum(-1).cumsum(-1).remainder(2)
        with torch.no_grad():h=rollout(ps,x);ok=(h>.5)==y.bool();finite=torch.isfinite(h).all();rows[str(t)]={'accuracy':float(ok[:,:,-1].float().mean()),'all_prefix':float(ok.all(-1).float().mean()),'finite':bool(finite),'final_correct':ok[0,:,-1].int().tolist(),'all_prefix_correct':ok[0].all(-1).int().tolist()}
    return rows
def train08(arm,out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);n=len(SEEDS);gs=[torch.Generator().manual_seed(8010000+s) for s in SEEDS];raw=list(zip(*(init(s) for s in SEEDS)));ps=[torch.nn.Parameter(torch.stack(t)) for t in raw];learn=arm.endswith('learn');ls=None;o=torch.optim.Adam(ps,lr=.015);c,x,y=centers(n);sampler=[hashlib.sha256() for _ in SEEDS];start=time.monotonic()
    for t in range(STEPS):
        bits=32 if t<600 else 6 if t<900 else 4
        if t in (600,900):
            ls=[torch.nn.Parameter(s.log()) for s in scales(ps,bits)] if learn else None
            groups=[{'params':ps,'lr':.01}]+([{'params':ls,'lr':.003}] if learn else []);o=torch.optim.Adam(groups)
        if t==1800:
            o.param_groups[0]['lr']=.003
            if learn:o.param_groups[1]['lr']=.001
        q,_,_=decode(ps,bits,ls,True);xx,yy=train_batch(gs,t=T,n=BATCH)
        for j in range(n):sampler[j].update(xx[j].numpy().tobytes())
        h=rollout(q,xx);task=(h-yy).square().mean((1,2));err=enclosure(q,c,x,y);exc=(err-TARGET08).clamp_min(0).square();aux=exc.amax(1) if arm.startswith('max') else exc.mean(1)
        loss=(task+AUX08*aux).sum();assert torch.isfinite(loss),(arm,t)
        o.zero_grad();loss.backward();allp=ps+(ls or []);norms=sum(p.grad.reshape(n,-1).square().sum(1) for p in allp).sqrt().clamp_min(1e-12);fac=(1/norms).clamp_max(1)
        for p in allp:p.grad.mul_(fac.reshape((n,)+(1,)*(p.ndim-1)))
        o.step()
        if t+1 in (600,900,1800,2400):print(arm,t+1,'maxbound',[round(v,4) for v in err.detach().amax(1).tolist()],flush=True)
    out.mkdir(parents=True,exist_ok=True);q,codes,ss=decode(ps,4,ls);final_bounds=enclosure(q,c,x,y).detach().amax(1)
    for j,seed in enumerate(SEEDS):
        p=[v[j:j+1].detach().clone() for v in ps];qp=[v[j:j+1].detach().clone() for v in q];f=out/f'{arm}-{seed}.json';r={'arm':arm,'seed':seed,'sampler_sha256':sampler[j].hexdigest(),'fp32_weights':{name:v[0].tolist() for name,v in zip(NAMES,p)},'q4_weights':{name:v[0].tolist() for name,v in zip(NAMES,qp)},'codes':{NAMES[i]:v[j].to(torch.int64).tolist() for i,v in zip(QIDS,codes)},'scales':{NAMES[i]:float(v[j].reshape(-1)[0].detach()) for i,v in zip(QIDS,ss)},'float_bound_max':float(final_bounds[j]),'fp32':eval08(p,seed),'q4':eval08(qp,seed)};f.write_text(json.dumps(r,separators=(',',':'))+'\n')
    print(arm,'seconds',time.monotonic()-start,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--out',type=Path,default=Path('results'));a=p.parse_args();train08(a.arm,a.out)
