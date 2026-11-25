from train import *
from torch.func import functional_call,jacrev
SEEDS=(11,23,37)
PARENT_ARMS=ARMS
QARMS=('fp','qat4','qat6to4')
H=(4,8,12,16,24)
STEPS=1200
P=Path(__file__).parent

def load_parent(seed,arm):
    name=f'{seed}-{arm}.pt';p=P/'parents'/name;digest=sha(p.read_bytes());assert digest==json.loads((P/'parents/manifest.json').read_text())['extended/'+name]
    m=fresh(4,seed);m.load_state_dict(torch.load(p,weights_only=True));return m.eval(),digest

def data(seed,part,t,n,domain):
    ctx=split(seed)[part][2];g=torch.Generator().manual_seed(domain+100*seed+t);rows=[stream(g,[c],n,t) for c in ctx];return torch.cat([r[0] for r in rows]),torch.cat([r[1] for r in rows]),ctx

def step(m,h,x):
    q=torch.cat((h,x),-1);return h+m.skip[0](q)+.2*m.down[0](F.gelu(m.up[0](q)))

def trace(m,x,h=None):
    if h is None:h=m.init.expand(len(x),-1)
    states=[h];zs=[]
    for t in range(x.shape[1]):
        h=step(m,h,x[:,t]);states.append(h);zs.append(m.read(h).squeeze(-1))
    return torch.stack(states,1),torch.stack(zs,1)

def jac(m,h,x):
    u=m.up[0](torch.cat((h,x),-1));der=.5*(1+torch.erf(u/math.sqrt(2)))+u*torch.exp(-u.square()/2)/math.sqrt(2*math.pi)
    return torch.eye(D)+m.skip[0].weight[:,:D]+.2*((m.down[0].weight*der.unsqueeze(-2))@m.up[0].weight[:,:D])

def stats(a):
    a=a.detach().flatten();return dict(mean=float(a.mean()),median=float(a.median()),p90=float(torch.quantile(a,.9)),max=float(a.max()))

def correct(z,y):return (z.gt(0)==y.bool()).int()
def summary(ok):return dict(acc=float(ok[:,-1].float().mean()),all_suffix=float(ok.all(1).float().mean()),final=ok[:,-1].tolist(),all_correct=ok.all(1).int().tolist())

def diagnostics(m,seed):
    xc,yc,_=data(seed,'cal',16,128,1500000);x,y,ctx=data(seed,'dev',16,128,1600000)
    with torch.no_grad():
        hc,_=trace(m,xc);h,z=trace(m,x);assert torch.allclose(z,m(x)[0],atol=1e-5,rtol=1e-5)
        means=torch.stack([hc[yc[:,7]==k,8].mean(0) for k in (0,1)]);gap=(means[1]-means[0]).norm();assert gap>0
        within=(hc[:,8]-means[yc[:,7].long()]).square().sum(-1).mean().sqrt();p=y[:,7].long();guess=torch.cdist(h[:,8],means).argmin(1);suffix=x[:,8:];target=y[:,8:]
        orig=h[:,8];same=means[p];opp=means[1-p];delta=opp-orig;g=torch.Generator().manual_seed(1700000+seed);dirs=torch.randn((4,len(x),D),generator=g);dirs=dirs/dirs.norm(dim=-1,keepdim=True);sham=orig+dirs[0]*delta.norm(dim=-1,keepdim=True)
        interventions={}
        for name,state in [('clean',orig),('same_centroid',same),('opposite_centroid',opp),('matched_random',sham)]:
            _,v=trace(m,suffix,state);interventions[name]=dict(original=summary(correct(v,target)),flipped=summary(correct(v,1-target)),dose_norm=stats((state-orig).norm(dim=-1)))
        dose_rows=[]
        for dose in (0,.05,.1,.25,.5,1):
            ok=torch.stack([correct(trace(m,suffix,orig+dose*gap*direction)[1],target) for direction in dirs]);clean=correct(z[:,8:],target).bool();den=int(clean[:,-1].sum())*len(dirs);stable=(ok[:,:,-1].bool()&clean[None,:,-1]).sum().item()/max(1,den)
            dose_rows.append(dict(dose=dose,acc=float(ok[:,:,-1].float().mean()),conditional_final_survival=stable,clean_correct_denom=den,all_suffix=float(ok.all(-1).float().mean()),final=ok[:,:,-1].tolist()))
        J=jac(m,h[:,8:16].reshape(-1,D),x[:,8:].reshape(-1,3)).reshape(len(x),8,D,D).double();radius=torch.linalg.eigvals(J).abs().amax(-1);sv=torch.linalg.svdvals(J)[...,0];product=torch.eye(D,dtype=torch.float64).expand(len(x),-1,-1)
        for t in range(8):product=J[:,t]@product
        result=dict(contexts=ctx,n=len(x),centroid_gap=float(gap),within_rms=float(within),gap_over_within=float(gap/within.clamp_min(1e-9)),centroid_accuracy=float((guess==p).float().mean()),margin=stats((2*y[:,7]-1)*z[:,7]),radius=stats(radius),singular=stats(sv),suffix_product_gain=stats(torch.linalg.svdvals(product)[:,0]),interventions=interventions,perturbations=dose_rows)
    errs=[]
    for i in range(4):
        state=h[i,8].detach();inp=x[i,8];actual=jacrev(lambda hh:step(m,hh,inp))(state);pred=jac(m,state,inp);errs.append(float((actual-pred).abs().max().detach()))
    result['jacobian_max_error']=max(errs);assert max(errs)<2e-5
    return result

def do_diag(seed,arm,out):
    m,p_sha=load_parent(seed,arm);r=dict(seed=seed,arm=arm,parent_sha=p_sha,fp=diagnostics(m,seed),q4=diagnostics(quant(m,4),seed));(out/f'{seed}-{arm}.json').write_text(json.dumps(r,separators=(',',':'))+'\n');print('DYN',seed,arm,[(k,round(r[k]['interventions']['clean']['original']['acc'],4),round(r[k]['interventions']['opposite_centroid']['flipped']['acc'],4),round(r[k]['suffix_product_gain']['median'],3)) for k in ('fp','q4')],flush=True)

def ste(v,bits):
    a=v.detach().abs().max();lev=2**(bits-1)-1;scale=(a/lev).clamp_min(1e-12);q=(v/scale).round().clamp(-lev,lev)*scale;return v+(q-v).detach()

def qa_eval(m,seed):
    d={t:data(seed,'dev',t,512,1800000) for t in H};return {str(b):measure(m if b==32 else quant(m,b),d) for b in (32,8,4,2)}

def do_qat(seed,arm,out):
    m,p_sha=load_parent(seed,'curriculum');m.train();g=torch.Generator().manual_seed(1900000+seed);ctx=split(seed)['train'][2];o=torch.optim.AdamW(m.parameters(),lr=.0015,weight_decay=0.);sample=hashlib.sha256();initial=qa_eval(m,seed);start=time.monotonic()
    for i in range(STEPS):
        x,y=stream(g,ctx,128,8);sample.update(x.numpy().tobytes());bits=32 if arm=='fp' else 6 if arm=='qat6to4' and i<400 else 4
        z,_=m(x) if bits==32 else functional_call(m,{k:ste(v,bits) for k,v in m.named_parameters()},(x,));loss=F.binary_cross_entropy_with_logits(z,y);o.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step()
    m.eval();cp=out/f'{seed}-{arm}.pt';torch.save(m.state_dict(),cp);r=dict(seed=seed,arm=arm,parent_sha=p_sha,checkpoint_sha=sha(cp.read_bytes()),params=sum(p.numel() for p in m.parameters()),updates=STEPS,sampler_sha=sample.hexdigest(),initial=initial,final=qa_eval(m,seed),fp_dynamics=diagnostics(m,seed),q4_dynamics=diagnostics(quant(m,4),seed),seconds=time.monotonic()-start);(out/f'{seed}-{arm}.json').write_text(json.dumps(r,separators=(',',':'))+'\n');print('QAT',seed,arm,[(b,round(r['final'][b]['12']['acc'],4),round(r['final'][b]['24']['acc'],4)) for b in ('32','4')],r['seconds'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=('diag','qat'),required=True);ap.add_argument('--seed',type=int,required=True,choices=SEEDS);ap.add_argument('--arm',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);a.out.mkdir(parents=True,exist_ok=True);(do_diag if a.task=='diag' else do_qat)(a.seed,a.arm,a.out)
