from parent_pilot import *
import copy
ARMS=('fixed4','curriculum','long8','kd_long8','prog_salience','prog_random','budget_curriculum')
CUTS={120:32,240:16,360:8,480:4}
HORIZONS=(4,6,8,10,12)
B=128
BASE=1200
def cost(w):return 320+35*w
PROG=sum(120*4*cost(w) for w in (64,32,16,8))+720*8*cost(4)
SCRATCH=480*4*cost(4)+720*8*cost(4)
EXTRA=round((PROG-SCRATCH)/(8*cost(4)))
def fresh(width,seed):
    torch.manual_seed(seed);m=Model('tied','affine');m.width=width;m.up=nn.ModuleList([nn.Linear(D+3,width)]);m.down=nn.ModuleList([nn.Linear(width,D)]);return m

def stream(g,contexts,n,t):
    x=torch.randint(2,(n,t,3),generator=g).float();c=torch.tensor(contexts)[torch.randint(len(contexts),(n,),generator=g)]
    x[:,:4,2]=((c[:,None]>>torch.arange(4))&1).float()
    y=x[:,:,:2].long().sum(-1).cumsum(1).remainder(2).float()
    return x,y

def eval_data(seed,contexts,t):
    g=torch.Generator().manual_seed(990000+seed*100+t)
    rows=[stream(g,[c],512,t) for c in contexts];return torch.cat([r[0] for r in rows]),torch.cat([r[1] for r in rows]),contexts

def compact(m,ids):
    n=fresh(len(ids),0)
    with torch.no_grad():
        n.init.copy_(m.init);n.skip.load_state_dict(m.skip.state_dict());n.read.load_state_dict(m.read.state_dict());n.up[0].weight.copy_(m.up[0].weight[ids]);n.up[0].bias.copy_(m.up[0].bias[ids]);n.down[0].weight.copy_(m.down[0].weight[:,ids]);n.down[0].bias.copy_(m.down[0].bias)
    return n

def prune(m,k,arm,seed,step,x,y):
    if arm=='prog_salience':
        z,a=m(x,trace=True);g=torch.autograd.grad(F.binary_cross_entropy_with_logits(z,y),a);score=torch.stack([(v*d).abs().sum(0) for v,d in zip(a,g)]).sum(0).detach();ids=score.argsort(descending=True,stable=True)[:k].tolist()
    else:ids=torch.randperm(m.width,generator=torch.Generator().manual_seed(780000+seed*1000+step))[:k].tolist()
    n=compact(m,ids);mask=torch.zeros(m.width);mask[ids]=1
    with torch.no_grad():error=float((m(x,masks=[mask]*x.shape[1])[0]-n(x)[0]).abs().max())
    assert error<.002,error
    return n,dict(step=step,keep=ids,compact_max_error=error)

def opt(m):return torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=0.)
def quant(m,bits):
    q=copy.deepcopy(m)
    with torch.no_grad():
        for p in q.parameters():
            a=p.abs().max();lev=2**(bits-1)-1
            if a>0:p.copy_((p/a*lev).round().clamp(-lev,lev)*a/lev)
    return q

def measure(m,data):
    return {str(t):evaluate(m,d) for t,d in data.items()}

def run(seed,arm,out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);out.mkdir(exist_ok=True,parents=True)
    d=split(seed);train=d['train'][2];cal=d['cal'][2];dev=d['dev'][2];assert len(set(train+cal+dev))==16
    p=Path(__file__).parent/'parents'/'affine_long'/f'affine-tied-{seed}.pt';p_sha=sha(p.read_bytes());assert json.loads((p.parent.parent/'manifest.json').read_text())[str(p.relative_to(p.parent.parent))]==p_sha
    teacher=Model('tied','affine');teacher.load_state_dict(torch.load(p,weights_only=True));teacher.eval()
    data={t:eval_data(seed,dev,t) for t in HORIZONS};cx,cy=stream(torch.Generator().manual_seed(880000+seed),cal,512,8)
    progressive=arm.startswith('prog_');m=fresh(64 if progressive else 4,seed);o=opt(m);gen=torch.Generator().manual_seed(770000+seed);sampler=hashlib.sha256();ledger=[];mac=teachmac=selmac=0;steps=BASE+EXTRA if arm=='budget_curriculum' else BASE;start=time.monotonic()
    for step in range(steps):
        if step in CUTS:
            if progressive:
                selmac+=len(cx)*4*cost(m.width) if arm=='prog_salience' else 0
                m,row=prune(m,CUTS[step],arm,seed,step,cx[:,:4],cy[:,:4]);ledger.append(row)
            o=opt(m)
        x,y=stream(gen,train,B,8);sampler.update(x.numpy().tobytes());h=4 if arm=='fixed4' or (arm in ('curriculum','prog_salience','prog_random','budget_curriculum') and step<480) else 8
        x=x[:,:h];y=y[:,:h];z,_=m(x);loss=F.binary_cross_entropy_with_logits(z,y);mac+=B*h*cost(m.width)
        if arm=='kd_long8':
            with torch.no_grad():soft=(teacher(x)[0]/2).sigmoid();teachmac+=B*h*cost(64)
            loss=.5*loss+2*F.binary_cross_entropy_with_logits(z/2,soft)
        o.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step()
    cp=out/f'{seed}-{arm}.pt';torch.save(m.state_dict(),cp);m.eval();r=dict(seed=seed,arm=arm,steps=steps,width=m.width,params=sum(p.numel() for p in m.parameters()),checkpoint=cp.name,checkpoint_sha256=sha(cp.read_bytes()),parent_sha256=p_sha,sampler_sha256=sampler.hexdigest(),train_contexts=train,cal_contexts=cal,dev_contexts=dev,student_mac_proxy=mac,teacher_mac_proxy=teachmac,selection_forward_mac_proxy=selmac,prune=ledger,eval=measure(m,data),teacher=measure(teacher,data),removed=evaluate(m,data[4],remove=True),quant={str(b):measure(quant(m,b),{t:data[t] for t in (8,12)}) for b in (8,4)},seconds=time.monotonic()-start)
    (out/f'{seed}-{arm}.json').write_text(json.dumps(r,separators=(',',':'))+'\n');print(seed,arm,steps,[(t,round(r['eval'][str(t)]['acc']*100,2)) for t in HORIZONS],round(r['seconds'],2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--out',type=Path,default=Path('results'));a=ap.parse_args();run(a.seed,a.arm,a.out)
