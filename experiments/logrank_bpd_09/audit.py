"""Saved-output and exact Boolean certificate audit. No optimizer replay."""
import hashlib, itertools, json, sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def to_cols(ids,k): return [sum(((v>>j)&1)<<i for i,v in enumerate(ids)) for j in range(k)]
def execute(p,cols,n):
    allbits=(1<<n)-1; a=allbits
    for i,o,s in p:
        v=cols[i] if s==1 else allbits^cols[i]
        a=allbits^a^v if o==0 else a&v
    return a

def actual(w,p):
    return [(w['slot_coord'][w['token_slot'][t]],o,s*w['slot_sign'][w['token_slot'][t]]) for t,o,s in p]
def equiv(p,q):
    ix=sorted({i for i,_,_ in p+q}); n=1<<len(ix); cols=to_cols(range(n),len(ix)); loc=lambda a:[(ix.index(i),o,s) for i,o,s in a]
    return execute(loc(p),cols,n)==execute(loc(q),cols,n)
def columns(v):
    out=[]
    for row in zip(*v):
        w=list(map(Fraction,row)); i=max(range(len(w)),key=lambda j:abs(w[j])); l1=sum(map(abs,w)); m=2*abs(w[i])-l1
        out.append((i,1 if w[i]>=0 else -1,m>0 and m>l1/Fraction(65536)))
    return out

def as_bits(s): return sum((c=='1')<<i for i,c in enumerate(s))
def audit(paths):
    rows=[]; roots=[]; cases=0; dense_cert=0; snap_cert=0
    for path in paths:
        r=json.loads(Path(path).read_text()); assert r['source_sha256']==sha(Path(__file__).with_name('pilot.py')); k=r['k']
        for w in r['roots']:
            assert sorted(w['addresses'])==list(range(128)); assert sorted(w['slot_coord'])==sorted(w['token_slot'])==list(range(k)); assert len(set(w['train_ids']))==len(set(w['test_ids']))==256 and not set(w['train_ids'])&set(w['test_ids']); ps=w['programs']; assert len({tuple(map(tuple,p)) for p in ps})==128; assert all(1<=len(p)<=3 for p in ps[:96]) and all(4<=len(p)<=5 for p in ps[96:]); truth=[actual(w,p) for p in ps]; cols=to_cols(w['test_ids'],k); labels=[execute(p,cols,256) for p in truth[64:]]; assert labels==[as_bits(w['labels'][j::64]) for j in range(64)]
            same_seen=[any(equiv(p,q) for q in truth[:64] if {i for i,_,_ in p}=={i for i,_,_ in q}) for p in truth[64:]]; roots.append(dict(k=k,root=w['root'],semantically_seen=sum(same_seen),new_same=32-sum(same_seen[:32]),new_deep=32-sum(same_seen[32:])))
            x=np.array([[1 if (v>>j)&1 else -1 for j in range(k)] for v in w['test_ids']],dtype=np.float64)
            for a in w['arms']:
                v=np.array(a['V']); assert v.shape==(k,k) and np.isfinite(v).all(); cc=columns(a['V']); route=a['route']; go=a['go']; assert len(route)==len(go)==k+1 and all(0<=i<k for i in route); projection=x@v; cert=[]; scert=[]; pred=[]; snap=[]
                for p,t in zip(ps[64:],truth[64:]):
                    used=[]
                    for token,o,s in p+[[k,0,1]]*(6-len(p)):
                        if not go[token]:break
                        used.append([route[token],o,s])
                    j=len(pred); assert used==a['executed'][j]; local=[(cc[i][0],o,s*cc[i][1]) for i,o,s in used]; exact=equiv(local,t); scert.append(exact); cert.append(exact and all(cc[i][2] for i,_,_ in used)); s=execute(local,cols,256); assert s==as_bits(a['snap_predictions'][j::64]); snap.append(s); acc=np.ones(256,dtype=np.int8)
                    for i,o,sign in used:
                        lit=np.where(projection[:,i]>=0,1,-1)*sign; acc=acc*lit if o==0 else np.minimum(acc,lit)
                    q=sum(int(b>0)<<i for i,b in enumerate(acc)); assert q==as_bits(a['predictions'][j::64]); pred.append(q)
                    if cert[-1]: assert q==labels[j]
                    if scert[-1]: assert s==labels[j]
                correct=[256-(p^q).bit_count() for p,q in zip(pred,labels)]; snapped=[256-(p^q).bit_count() for p,q in zip(snap,labels)]; assert sum(correct[:32])==a['correct_same'] and sum(correct[32:])==a['correct_deep']; assert sum(snapped[:32])==a['snap_same'] and sum(snapped[32:])==a['snap_deep']; assert sum((p^q).bit_count() for p,q in zip(pred,snap))==a['snap_disagreements']; assert a['replay_mismatches']==0
                novel=[i for i in range(64) if not same_seen[i]]; rows.append(dict(k=k,root=w['root'],rate=a['rate'],arm=a['arm'],correct_same=a['correct_same'],correct_deep=a['correct_deep'],snap_same=a['snap_same'],snap_deep=a['snap_deep'],dense_cert=sum(cert),snap_cert=sum(scert),dense_cert_deep=sum(cert[32:]),snap_cert_deep=sum(scert[32:]),stop_exact=go==[True]*k+[False],route_slot_matches=sum(x==y for x,y in zip(route,w['token_slot'])),dominant=sum(c[2] for c in cc),novel_correct=sum(correct[i] for i in novel),novel_total=256*len(novel),snap_disagreements=a['snap_disagreements'],noise=a['noise'])); cases+=64*256; dense_cert+=sum(cert); snap_cert+=sum(scert)
    return dict(status='PASS',source_sha256=sha(Path(__file__).with_name('pilot.py')),input_sha256={Path(p).name:sha(p) for p in paths},scored_decisions=cases,projected_decisions=cases,models_and_controls=len(rows),dense_certificates=dense_cert,snap_certificates=snap_cert,roots=roots,rows=rows,scope='saved dense bank + hard route replay, signed projection, exact rational dominance and support truth tables; no optimizer or native BPD training replay')
if __name__=='__main__':print(json.dumps(audit(sys.argv[1:]),indent=2))
