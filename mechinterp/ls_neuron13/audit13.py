"""Receipt replay, code-transition audit, exact fixed-region certificates."""
from repair13 import *
from restart13 import EXT_ARMS
from audit import verify_scalar
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def flat_codes(r):return np.concatenate([np.asarray(r['codes'][NAMES[i]]).ravel() for i in old.QIDS]).astype(np.int64)
def cert(w):return verify_scalar(w)
def audit_baseline(r):
    p=[torch.tensor(r['fp32_weights'][k],dtype=torch.float32)[None] for k in NAMES]
    q=[torch.tensor(r['q4_weights'][k],dtype=torch.float32)[None] for k in NAMES]
    for i in old.QIDS:
        c=torch.tensor(r['codes'][NAMES[i]],dtype=torch.float32);s=torch.tensor(r['scales'][NAMES[i]],dtype=torch.float32)
        assert c.abs().max()<=7 and s>0 and torch.equal(q[i][0],c*s) and torch.equal(c,(p[i][0]/s).clamp(-7,7).round())
    for i in (1,5):assert torch.equal(p[i],q[i])
    assert measure(r['q4_weights'],r['seed'])==dict(eval=r['eval'],float_bound=r['bound'])

def audit_repair(r,base):
    seed=r['seed'];dose=r['dose'];arm=r['arm'];c,s,o,db=setup(seed,dose)
    assert r['initial']['codes']==c.tolist() and r['initial']['weights']==base['q4_weights']
    assert r['initial']['scales']==s.tolist() and r['initial']['offset']==o
    assert measure(r['initial']['weights'],seed)=={k:r['initial'][k] for k in ('eval','float_bound')}
    assert len(r['ledger'])==ROUNDS+1;prev=r['ledger'][0];assert prev['codes']==c.tolist()
    edits=0
    for j,row in enumerate(r['ledger'][1:],1):
        assert row['round']==j;oldc=np.array(prev['codes']);newc=np.array(row['codes']);changed=not np.array_equal(oldc,newc);edits+=changed
        assert (np.abs(newc)<=7).all() and row['fit']['loss']<=prev['fit']['loss']+1e-8
        if arm.startswith('continuous'):assert not changed
        elif changed:
            moves,names=proposals(oldc,arm.startswith('paired'));name=row['move'].split('@')[0];assert name in names and np.array_equal(newc,moves[names.index(name)])
        prev=row
    assert r['final']['codes']==prev['codes'] and r['final']['scales']==prev['scales'] and r['final']['offset']==prev['offset']
    nc=np.array(prev['codes']);assert r['hamming']==int(np.count_nonzero(nc!=c)) and r['code_l1']==int(abs(nc-c).sum())
    assert r['final']['weights']==decoded(nc,np.array(prev['scales']),prev['offset'],db)
    assert measure(r['final']['weights'],seed)=={k:r['final'][k] for k in ('eval','float_bound')}
    assert r['search_evaluations']==sum(row['candidates'] for row in r['ledger'][1:])
    if arm in ARMS:assert r['fit_nfev']==sum(row['fit']['nfev'] for row in r['ledger'])
    else:assert r['fit_nfev']==r['ledger'][0]['fit']['nfev']+sum(t['fit']['nfev'] for row in r['ledger'][1:] for t in row['shortlist'])
    return edits

def job(a):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);seed,dose,arm=a
    basepath=Path(f'baseline/{dose}-frozen-{seed}-0.json');base=json.loads(basepath.read_text());audit_baseline(base);initial=base['q4_weights'];edits=0
    if arm=='frozen':path=basepath;final=initial;endpoint=base['eval'];hd=l1=search=fit=0
    elif arm=='full':
        ip=Path(f'baseline/{dose}-full-{seed}-0.json');ir=json.loads(ip.read_text());assert ir['q4_weights']==initial;audit_baseline(ir)
        path=Path(f'baseline/{dose}-full-{seed}-800.json');r=json.loads(path.read_text());audit_baseline(r);final=r['q4_weights'];endpoint=r['eval'];c=flat_codes(base);nc=flat_codes(r);hd=int(np.count_nonzero(c!=nc));l1=int(abs(c-nc).sum());search=0;fit=800
    else:
        path=Path('repairs' if arm in ARMS else 'restarts')/f'{dose}-{arm}-{seed}.json';r=json.loads(path.read_text());edits=audit_repair(r,base);final=r['final']['weights'];endpoint=r['final']['eval'];hd=r['hamming'];l1=r['code_l1'];search=r['search_evaluations'];fit=r['fit_nfev']
    ci=cert(initial);cf=ci if final==initial else cert(final)
    return dict(seed=seed,dose=dose,arm=arm,receipt_sha256=sha(path),initial=ci,final=cf,final_eval=endpoint,hamming=hd,code_l1=l1,accepted_edits=edits,search=search,fit=fit)

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert len(list(Path('baseline').glob('*.json')))==72 and len(list(Path('repairs').glob('*.json')))==72 and len(list(Path('restarts').glob('*.json')))==36
    control=cert(old.scalar_weights(4)[0]);assert control['verdict']=='CERTIFIED'
    jobs=[(s,d,a) for d in DOSES for s in SEEDS for a in ('frozen','full')+ARMS]+[(s,15,a) for s in SEEDS for a in EXT_ARMS]
    with ProcessPoolExecutor(max_workers=4) as ex:rows=list(ex.map(job,jobs,chunksize=1))
    assert len(rows)==156
    initial={(r['dose'],r['seed']):r['initial'] for r in rows if r['arm']=='frozen'}
    assert all(r['initial']==initial[r['dose'],r['seed']] for r in rows)
    Path('certificates.json').write_text(json.dumps({'positive_control':control,'rows':rows},indent=2)+'\n')
    summary=[]
    for d in DOSES:
        for a in ('frozen','full')+ARMS+(EXT_ARMS if d==15 else ()):
            r=[q for q in rows if q['dose']==d and q['arm']==a];assert len(r)==12;i=[q['initial']['verdict']=='CERTIFIED' for q in r];f=[q['final']['verdict']=='CERTIFIED' for q in r];s=dict(dose=d,arm=a,initial=sum(i),final=sum(f),new=sum((not b and v and q['initial']['verdict']=='POINT_VIOLATION') for q,b,v in zip(r,i,f)),lost=sum(b and not v for b,v in zip(i,f)),T8=sum(q['final_eval']['8']['accuracy'] for q in r)/12,T128=sum(q['final_eval']['128']['accuracy'] for q in r)/12,T128_all=sum(q['final_eval']['128']['all_prefix'] for q in r)/12,hamming=sum(q['hamming'] for q in r)/12,search=sum(q['search'] for q in r),fit=sum(q['fit'] for q in r),new_seeds=[q['seed'] for q,b,v in zip(r,i,f) if not b and v and q['initial']['verdict']=='POINT_VIOLATION']);summary.append(s);print(s,flush=True)
    out=dict(verdict='PASS',paired_rows=len(rows),unique_initial_conditions=len(initial),endpoint_verdicts=dict(Counter(r['final']['verdict'] for r in rows)),initial_verdicts=dict(Counter(c['verdict'] for c in initial.values())),stored_receipt_files=180,numerical_horizon_conditions=2*(72+2*72+2*36),accepted_edits=sum(r['accepted_edits'] for r in rows),positive_control='CERTIFIED',summary=summary);Path('audit_results.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS', {k:v for k,v in out.items() if k!='summary'},flush=True)
if __name__=='__main__':main()
