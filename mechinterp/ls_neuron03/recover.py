"""LS-NEURON-03: exact greedy ablation and matched sparse recovery."""
from parent_pilot import *
STEPS=(0,120,600,1200)
WIDTHS=(64,48,32,16,8,4,2,1)
ARMS=('greedy','salience','random0','random1','reset_mlp','scratch','freeze_mlp','freeze_bypass')

def states_logits(m,x,mask=None):
    h=m.init.expand(len(x),-1);hs=[];zs=[]
    for t in range(x.shape[1]):
        q=torch.cat((h,x[:,t]),-1);a=F.gelu(m.up[0](q));a=a if mask is None else a*mask
        h=h+m.skip[0](q)+.2*m.down[0](a);hs.append(h);zs.append(m.read(h).squeeze(-1))
    return torch.stack(zs,1),torch.stack(hs,1)

@torch.no_grad()
def candidate_losses(m,x,y,active):
    vals=[]
    for chunk in torch.tensor(active).split(8):
        masks=torch.zeros(len(chunk),1,W);masks[:,:,active]=1;masks[torch.arange(len(chunk)),0,chunk]=0
        h=m.init.expand(len(chunk),len(x),D);loss=torch.zeros(len(chunk))
        for t in range(T):
            q=torch.cat((h,x[None,:,t].expand(len(chunk),-1,-1)),-1)
            h=h+m.skip[0](q)+.2*m.down[0](F.gelu(m.up[0](q))*masks)
            z=m.read(h).squeeze(-1);loss+=F.binary_cross_entropy_with_logits(z,y[:,t].expand_as(z),reduction='none').mean(1)/T
        vals.extend(loss.tolist())
    return vals

def select(m,x,y,seed):
    active=list(range(W));path={W:active.copy()};ledger=[]
    for _ in range(W-1):
        losses=candidate_losses(m,x,y,active);idx=min(range(len(active)),key=lambda j:(losses[j],active[j]));ledger.append(dict(active=active.copy(),losses=losses,removed=active[idx]));active.pop(idx);path[len(active)]=active.copy()
    z,a=m(x,trace=True);g=torch.autograd.grad(F.binary_cross_entropy_with_logits(z,y),a);sal=torch.stack([(v*d).abs().sum(0) for v,d in zip(a,g)]).sum(0).detach();order=sal.argsort(descending=True,stable=True).tolist()
    rnd=[torch.randperm(W,generator=torch.Generator().manual_seed(67000+seed+i)).tolist() for i in range(2)]
    return path,ledger,order,rnd

def fresh(width,seed):
    torch.manual_seed(seed);m=Model('tied','affine');m.width=width;m.up=nn.ModuleList([nn.Linear(D+3,width)]);m.down=nn.ModuleList([nn.Linear(width,D)]);return m

def transplant(parent,ids,seed,reset=False):
    m=fresh(len(ids),seed)
    with torch.no_grad():
        m.init.copy_(parent.init);m.skip.load_state_dict(parent.skip.state_dict());m.read.load_state_dict(parent.read.state_dict())
        if not reset:
            m.up[0].weight.copy_(parent.up[0].weight[ids]);m.up[0].bias.copy_(parent.up[0].bias[ids]);m.down[0].weight.copy_(parent.down[0].weight[:,ids]);m.down[0].bias.copy_(parent.down[0].bias)
    return m

@torch.no_grad()
def measure(m,parent,data):
    x,y,ctx=data;z,h=states_logits(m,x);p,ph=states_logits(parent,x);ok=z.gt(0)==y.bool();pr=z.gt(0)==p.gt(0);a=z.flatten();b=p.flatten();corr=float(F.cosine_similarity(a-a.mean(),b-b.mean(),dim=0));return dict(acc=float(ok[:,-1].float().mean()),prefix_acc=ok.float().mean(0).tolist(),correct=ok[:,-1].int().tolist(),parent_agreement=float(pr[:,-1].float().mean()),logit_corr=corr,state_rel_rms=float((h-ph).square().mean().sqrt()/ph.square().mean().sqrt().clamp_min(1e-9)),contexts=ctx)

def run(seed,base,out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);out.mkdir(parents=True,exist_ok=True)
    parent=Model('tied','affine');p=base/'affine_long'/f'affine-tied-{seed}.pt';digest=sha(p.read_bytes());assert json.loads((base/'manifest.json').read_text())[str(p.relative_to(base))]==digest;parent.load_state_dict(torch.load(p,weights_only=True));parent.eval();d=split(seed);x,y,_=d['cal'];x,y=x[:512],y[:512];shadow=(d['cal'][0][512:],d['cal'][1][512:],d['cal'][2][2:])
    data={k:split(seed,k)['dev'] for k in (2,4,6,8)}
    path,ledger,order,rnd=select(parent,x,y,seed);selection=dict(seed=seed,parent_sha256=digest,greedy_path=path,candidate_ledger=ledger,salience_order=order,random_orders=rnd,cal_contexts=d['cal'][2][:2]);(out/f'selection-{seed}.json').write_text(json.dumps(selection,separators=(',',':'))+'\n')
    print('SELECTED',seed,path[4],flush=True);sweep=[]
    parent_metrics={k:measure(parent,parent,v) for k,v in data.items()};removed=evaluate(parent,d['dev'],remove=True)
    for w in WIDTHS:
        for kind,ids in [('greedy',path[w]),('salience',order[:w]),('random0',rnd[0][:w]),('random1',rnd[1][:w])]:
            m=transplant(parent,ids,seed);mask=torch.zeros(W);mask[ids]=1
            with torch.no_grad():delta=float((m(x)[0]-parent(x,masks=[mask]*T)[0]).abs().max());assert delta<.001,delta
            sweep.append(dict(width=w,kind=kind,ids=ids,mask_compact_max_delta=delta,shadow=measure(m,parent,shadow),dev=measure(m,parent,data[4])))
    results=dict(seed=seed,parent_sha256=digest,parent=parent_metrics,remove_mlp=removed,sweep=sweep,arms=[])
    for arm in ARMS:
        ids={'salience':order[:4],'random0':rnd[0][:4],'random1':rnd[1][:4]}.get(arm,path[4]);m=fresh(4,seed) if arm=='scratch' else transplant(parent,ids,seed,reset=arm=='reset_mlp')
        if arm=='freeze_mlp':
            for param in list(m.up.parameters())+list(m.down.parameters()):param.requires_grad_(False)
        if arm=='freeze_bypass':
            for param in [m.init]+list(m.skip.parameters())+list(m.read.parameters()):param.requires_grad_(False)
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.003,weight_decay=0.);g=torch.Generator().manual_seed(20000+seed);tx,ty,_=d['train'];sampler=hashlib.sha256();rows=[]
        for step in range(STEPS[-1]+1):
            if step in STEPS:
                cp=out/f'{seed}-{arm}-{step}.pt';torch.save(m.state_dict(),cp);hs=(2,4,6,8) if step in (0,1200) else (4,8);rows.append(dict(step=step,checkpoint=cp.name,checkpoint_sha256=sha(cp.read_bytes()),sampler_sha256=sampler.hexdigest(),eval={k:measure(m,parent,data[k]) for k in hs}))
            if step==STEPS[-1]:break
            ii=torch.randint(len(tx),(128,),generator=g);sampler.update(ii.numpy().tobytes());z,_=m(tx[ii]);loss=F.binary_cross_entropy_with_logits(z,ty[ii]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        results['arms'].append(dict(arm=arm,ids=None if arm=='scratch' else ids,params=sum(p.numel() for p in m.parameters()),trainable=sum(p.numel() for p in m.parameters() if p.requires_grad),rows=rows));(out/f'results-{seed}.json').write_text(json.dumps(results,separators=(',',':'))+'\n');print(seed,arm,[(r['step'],round(r['eval'][4]['acc'],4),round(r['eval'][8]['acc'],4)) for r in rows],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True,choices=SEEDS[:3]);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();run(a.seed,a.base,a.out)
