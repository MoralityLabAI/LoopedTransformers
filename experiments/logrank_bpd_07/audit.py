"""Independent bitset replay and exact candidate-fit audit."""
import itertools,json,hashlib,sys
from pathlib import Path
import numpy as np

def columns(ids,k):return [sum(((x>>j)&1)<<i for i,x in enumerate(ids)) for j in range(k)]
def bits(p,cols,n):
    op,ix,sign=p; full=(1<<n)-1
    if op=='const':return full if sign[0]==1 else 0
    val=full
    for j,s in zip(ix,sign):
        lit=cols[j] if s==1 else full^cols[j]
        val=(full^val^lit) if op=='parity' else val&lit
    return val

def candidates(k,d):
    if d==0:
        return [('const',(),(s,)) for s in (1,-1)]
    a=[]
    for ix in itertools.combinations(range(k),d):
        a.extend(('parity',ix,(s,)+(1,)*(d-1)) for s in (1,-1))
        if d>1:a.extend(('and',ix,ss) for ss in itertools.product((1,-1),repeat=d))
    return a

def encode(text):return sum((c=='1')<<i for i,c in enumerate(text))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(path):
    r=json.loads(Path(path).read_text()); assert sha(Path(__file__).with_name('pilot.py'))==r['source_sha256']; nrows=0; certs=0; comparisons=0; max_delta=0.; all_roots=[]
    for root in r['results']:
        k=root['k'];tr,te=root['train_ids'],root['heldout_ids'];n,m=len(tr),len(te);assert len(set(tr))==n and len(set(te))==m and not(set(tr)&set(te)); assert sorted(root['query_codes'])==list(range(k)) and root['bit_cap']==k.bit_length()
        a,b=columns(tr,k),columns(te,k); labels=[bits(p,a,n) for p in root['true_programs']];test=[bits(p,b,m) for p in root['true_programs']]
        assert [encode(root['labels'][j::k]) for j in range(k)]==test
        best=[n+1]*k; ties=[0]*k; count=0
        for d in range(4):
            for p in candidates(k,d):
                t=bits(p,a,n);count+=1
                for j,v in enumerate(labels):
                    err=(v^t).bit_count();comparisons+=1
                    if err<best[j]:best[j]=err;ties[j]=1
                    elif err==best[j]:ties[j]+=1
            if d==0:continue
            arm=root['arms'][d-1];assert arm['candidate_count']==count and arm['ties']==ties and arm['train_correct']==[n-e for e in best]; assert arm['reads']==[len(p[1]) for p in arm['programs']]
            for j,p in enumerate(arm['programs']):
                assert (bits(p,a,n)^labels[j]).bit_count()==best[j]; assert bits(p,b,m)==encode(arm['predictions'][j::k]); assert arm['certificates'][j]==(p==root['true_programs'][j]); certs+=arm['certificates'][j]
        x=2*((np.asarray(te,dtype=np.uint64)[:,None]>>np.arange(k,dtype=np.uint64))&1).astype(np.float64)-1
        for arm in root['arms']:
            out=arm['predictions']; truth=root['labels']; assert len(out)==len(truth)==m*k; ok=[p==t for p,t in zip(out,truth)]; assert sum(ok)==arm['correct'] and sum(all(ok[i:i+k]) for i in range(0,len(ok),k))==arm['exact_rows'];nrows+=len(ok)
            if arm['arm'].startswith('rank'):
                logits=x@np.array(arm['weight'])+np.array(arm['bias']); pred=''.join('1' if v>=0 else '0' for v in logits.flat);assert pred==out
        all_roots.append({'k':k,'root':root['root'],'depth3_certificates':root['arms'][2]['certified_queries']})
    return dict(status='PASS',source_sha256=r['source_sha256'],receipt_sha256=sha(path),scored_decisions=nrows,exact_program_certificates_all_caps=certs,candidate_label_comparisons=comparisons,roots=all_roots,scope='bitset candidate scoring, labels, predictions, saved affine head replay, certificate equality; no optimizer replay or native BPD execution')
if __name__=='__main__':print(json.dumps(main(sys.argv[1]),indent=2))
