"""Audit every stored endpoint, exact witness/certificate and selected modular stress runs."""
from verify11 import *
from collections import Counter

def audit_one(row):
    torch.set_num_threads(1);task,arm,seed,t=[row[k] for k in ('task','arm','seed','step')];p=Path(f'results/{task}-{arm}-{seed}-{t}.json');assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'];r=json.loads(p.read_text());assert all(r[k]==row[k] for k in ('task','arm','seed','step'))
    ps=[torch.tensor(r['fp32_weights'][k],dtype=torch.float32)[None] for k in NAMES];q=[torch.tensor(r['q4_weights'][k],dtype=torch.float32)[None] for k in NAMES]
    for i in QIDS:
        code=torch.tensor(r['codes'][NAMES[i]],dtype=torch.long);s=torch.tensor(r['scales'][NAMES[i]],dtype=torch.float32);assert s>0 and code.abs().max()<=7 and torch.equal(q[i][0],code.float()*s) and torch.equal(code,(ps[i][0]/s).clamp(-7,7).round().long())
    for i in (1,5):assert torch.equal(ps[i],q[i])
    if t==0:
        assert all(torch.allclose(a[0],b,atol=1e-6,rtol=1e-6) for a,b in zip(ps,init(task,seed,arm)[0]));design=base(task)[3];ss=scales(ps,4) if design is None else [torch.full_like(s,v) for s,v in zip(scales(ps,4),design)];assert all(torch.allclose(s.log().exp().clamp(1e-6,10.),torch.tensor(r['scales'][NAMES[i]]),rtol=1e-6,atol=1e-6) for i,s in zip(QIDS,ss))
    assert evaluate(q,task,seed)==r['eval'];h,x,y=centers(task,1);assert abs(float(enclosure(q,h,x,y).max())-r['bound'])<1e-4
    c=row['certificate'];w=r['q4_weights']
    if c['verdict']=='POINT_VIOLATION':
        a,b,n=c['input'];_,off,m=cases(task);p=c['p'];h=Q(c['h']);assert abs(h-off-p)<=R and (p+a+b)%m==c['target'];lo,hi=map(Q,c['target_inner']);assert (lo,hi)==(off+c['target']-R+EPS,off+c['target']+R-EPS);val=wf(w,box(h),a,b,n);assert val.out()==c['output'] and (val.lo>hi or val.hi<lo)
    else:assert verify(task,w)==c
    return c['verdict']

def stress(w,seed):
    q=[torch.tensor(w[k],dtype=torch.float32)[None] for k in NAMES];g=torch.Generator().manual_seed(11100000+seed);n=2048;h=torch.full((1,n),-2.5);p=torch.zeros(1,n,dtype=torch.long);bad=0;maxerr=0.
    with torch.no_grad():
        for _ in range(512):
            x=torch.randint(2,(1,n,3),generator=g).float();p=(p+x[:,:,:2].long().sum(-1))%4;h=step(q,h,x)+(torch.randint(2,(1,n),generator=g)*2-1).float()/32;bad+=int((((h+3).floor().clamp(0,3).long()!=p)|~torch.isfinite(h)).sum());maxerr=max(maxerr,float((h-(p-2.5)).abs().max()))
    return {'histories':n,'visits':512,'transitions':n*512,'errors':bad,'max_region_error':maxerr}

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);d=json.loads(Path('certificates.json').read_text());rows=d['rows'];expected={(task,arm,s,t) for task,arms in ARMS.items() for arm in arms for s in (PSEEDS if task=='parity' else MSEEDS) for t in (0,800)};assert {(r['task'],r['arm'],r['seed'],r['step']) for r in rows}==expected and len(rows)==len(expected)
    for task in ARMS:
        w={k:v.tolist() for k,v in zip(NAMES,base(task)[0])};assert verify(task,w)==d['positive_controls'][task]
    stored=json.loads(Path('construction.json').read_text());assert constructive()==stored or json.loads(Path('construction.json').read_text())==stored
    with ProcessPoolExecutor(max_workers=4) as ex:status=list(ex.map(audit_one,rows,chunksize=1))
    sc={'handcrafted':stress(stored['weights'],0)}
    for r in rows:
        if r['task']=='mod4' and r['step']==800 and r['certificate']['verdict']=='CERTIFIED':
            seed=r['seed'];w=json.loads(Path(f"results/mod4-{r['arm']}-{seed}-800.json").read_text())['q4_weights'];sc[str(seed)]=stress(w,seed)
    assert all(v['errors']==0 and v['max_region_error']<=.125 for v in sc.values())
    receipt={'verdict':'PASS','endpoints':len(status),'numerical_replays':2*len(status),'certificate_replays':dict(Counter(status)),'positive_controls':2,'modular_stress':sc};Path('audit_results.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt,flush=True)
if __name__=='__main__':main()
