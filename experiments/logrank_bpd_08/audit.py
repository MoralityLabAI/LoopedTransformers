"""Independent saved-route replay and exact rational coordinate certificates."""
import sys, hashlib, json, itertools
from fractions import Fraction
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unpack(ids,k): return 2*((np.asarray(ids,dtype=np.uint64)[:,None]>>np.arange(k,dtype=np.uint64))&1).astype(np.int8)-1
def eval_program(x,p):
    op,idx,sign=p; vals=x[:,idx]*np.asarray(sign)
    return vals.prod(1) if op=='parity' else vals.min(1)
def component_certificate(v):
    w=list(map(Fraction,v)); i=max(range(len(w)),key=lambda j:abs(w[j])); l1=sum(map(abs,w)); margin=2*abs(w[i])-l1
    # Exact rational dominance plus conservative margin for ordinary fp32 summation.
    safe=margin>l1/Fraction(65536) and margin>0
    return dict(coordinate=i,sign=1 if w[i]>0 else -1,margin=float(margin),safe=bool(safe))
def equivalent(p,q):
    ix=sorted(set(p[1])|set(q[1])); z=np.array(list(itertools.product((-1,1),repeat=len(ix))))
    def local(p):return p[0],[ix.index(i) for i in p[1]],p[2]
    return bool(np.array_equal(eval_program(z,local(p)),eval_program(z,local(q))))
def audit(path):
    r=json.loads(Path(path).read_text()); k=r['k']; assert r['source_sha256'] in {sha(Path(__file__).with_name(n)) for n in ('pilot.py','teacher.py')}; assert r['dictionary_sha256']==sha(Path(__file__).with_name('dictionary.py')); rows=[]
    for t in r['results']:
        tr,te=t['train_ids'],t['heldout_ids']; assert len(set(tr))==256 and len(set(te))==1024 and not set(tr)&set(te); assert sorted(t['codes'])==list(range(k)); assert len(t['truth'])==k
        x,z=unpack(tr,k),unpack(te,k); labels=np.stack([eval_program(z,p) for p in t['truth']],1); assert ''.join('1' if b else '0' for b in (labels>0).flat)==t['labels']; train=np.stack([eval_program(x,p) for p in t['truth']],1)
        fit_correct=(np.stack([eval_program(x,p) for p in t['fitted']],1)==train).sum(0); assert fit_correct.tolist()==t['fit_correct']; assert all(equivalent(p,q) for p,q in zip(t['fitted'],t['truth']))
        for a in t['arms']:
            V=np.array(a['V'],dtype=np.float64); assert V.shape==(k,k) and np.isfinite(V).all(); cc=[component_certificate(V[:,i].tolist()) for i in range(k)]; predictions=[]; certs=[]; route_match=[]; reads=0
            for j,(p,route) in enumerate(zip(t['truth'],a['query_routes'])):
                idx,ss,op=route['indices'],route['signs'],route['op']; assert 1<=len(idx)<=3 and len(ss)==len(idx) and all(0<=i<k for i in idx) and all(s in (-1,1) for s in ss) and op in ('parity','and'); reads+=len(idx)
                literal=np.where(z@V[:,idx]>=0,1,-1)*np.array(ss); pred=literal.prod(1) if op=='parity' else literal.min(1); predictions.append(pred)
                chosen=[cc[i] for i in idx]; cert=all(c['safe'] for c in chosen) and equivalent((op,[c['coordinate'] for c in chosen],[s*c['sign'] for s,c in zip(ss,chosen)]),p); certs.append(bool(cert)); fit=t['fitted'][j]; route_match.append(op==fit[0] and idx==fit[1] and ss==fit[2])
                if cert: assert np.array_equal(pred,labels[:,j])
            pred=np.stack(predictions,1); saved=np.array([1 if c=='1' else -1 for c in a['predictions']]).reshape(-1,k); mismatch=int((saved!=pred).sum()); assert mismatch==0; ok=saved==labels; assert int(ok.sum())==a['correct'] and int(ok.all(1).sum())==a['exact_rows']; assert sum(route_match)==a['exact_instruction_queries']; assert abs(reads/k-a['mean_active_reads'])<1e-6
            rows.append(dict(k=k,root=t['root'],arm=a['arm'],correct=a['correct'],total=a['total'],exact_rows=a['exact_rows'],exact_instruction_queries=a['exact_instruction_queries'],certified_queries=sum(certs),certificates=certs,dominant_components=sum(c['safe'] for c in cc),component_certificates=cc,mean_active_reads=reads/k,replay_mismatches=mismatch,parameters=a['parameters'],trainable_parameters=a['trainable_parameters']))
    return dict(source_sha256=r['source_sha256'],receipt_sha256=sha(path),models=len(rows),decisions=sum(a['total'] for a in rows),status='PASS',rows=rows)
if __name__=='__main__': print(json.dumps([audit(p) for p in sys.argv[1:]],indent=2))
