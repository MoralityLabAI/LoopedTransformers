"""Frozen fixed-region verifier and receipt audit."""
from train10 import *
from verify import exact,fixed_violation,positive,wf,box,Q
from concurrent.futures import ProcessPoolExecutor
REGS=[(-Q(1,8),Q(1,8)),(Q(7,8),Q(9,8))];EPS=Q(1,32)
def cert(w):
    v=fixed_violation(w,REGS,EPS)
    return {'verdict':'POINT_VIOLATION','witness':v} if v else exact(w,REGS,EPS)
def job(path):
    torch.set_num_threads(1);r=json.loads(path.read_text());w=r['q4_weights'];c=cert(w);return dict(arm=r['arm'],seed=r['seed'],step=r['step'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),certificate=c)
def audit_one(r):
    path=Path('results')/f"{r['arm']}-{r['seed']}-{r['step']}.json";assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];d=json.loads(path.read_text());p=[torch.tensor(d['fp32_weights'][k],dtype=torch.float32)[None] for k in NAMES];q=[torch.tensor(d['q4_weights'][k],dtype=torch.float32)[None] for k in NAMES]
    for i in QIDS:
        code=torch.tensor(d['codes'][NAMES[i]],dtype=torch.long);s=torch.tensor(d['scales'][NAMES[i]],dtype=torch.float32);assert s>0 and code.abs().max()<=7 and torch.equal(q[i][0],code.float()*s) and torch.equal(code,(p[i][0]/s).clamp(-7,7).round().long())
    for i in (1,5):assert torch.equal(p[i],q[i])
    if r['step']==0:
        assert all(torch.allclose(a[0],b,atol=1e-6,rtol=1e-6) for a,b in zip(p,init10(r['seed'],r['arm'])))
        expect=[v.log().exp().clamp(1e-6,10.) for v in scales(p,4)];assert all(torch.allclose(a,torch.tensor(d['scales'][NAMES[i]]),atol=1e-6,rtol=1e-6) for i,a in zip(QIDS,expect))
    assert eval10(q,d['seed'])==d['q4'];h,x,y=centers(1);assert abs(float(enclosure(q,h,x,y).max())-d['float_bound_max'])<1e-4
    c=r['certificate'];w=d['q4_weights']
    if c['verdict']=='POINT_VIOLATION':
        z=c['witness'];a,b,n=z['input'];assert z['target']==z['p']^a^b;value=wf(w,box(Q(z['h'])),a,b,n);assert value.out()==z['output'];lo,hi=map(Q,z['target_inner']);assert value.lo>hi or value.hi<lo
    else:assert cert(w)==c
    return c['verdict']
def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);a=argparse.ArgumentParser();a.add_argument('--audit',action='store_true');args=a.parse_args();paths=sorted(Path('results').glob('*.json'));assert len(paths)==len(ARMS)*len(SEEDS)*2
    if not args.audit:
        pos=cert(positive());assert pos['verdict']=='CERTIFIED'
        with ProcessPoolExecutor(max_workers=4) as ex:rows=list(ex.map(job,paths,chunksize=1))
        Path('certificates.json').write_text(json.dumps({'positive_control':pos,'rows':rows},indent=2)+'\n');print('verified',len(rows),flush=True)
    else:
        d=json.loads(Path('certificates.json').read_text());assert cert(positive())==d['positive_control'];assert len({(r['arm'],r['seed'],r['step']) for r in d['rows']})==len(paths)
        with ProcessPoolExecutor(max_workers=4) as ex:status=list(ex.map(audit_one,d['rows'],chunksize=1))
        counts={s:status.count(s) for s in set(status)};receipt=dict(verdict='PASS',endpoints=len(status),numerical_replays=2*len(status),certificate_replays=counts,positive_control='CERTIFIED');Path('audit_results.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt,flush=True)
if __name__=='__main__':main()
