"""Independent finite-integer checker for fresh source worlds only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

def multiply(a,b):
    out=[0]*len(a)
    for i,x in enumerate(a):
        for j,y in enumerate(b[:len(a)-i]):out[i+j]+=x*y
    return out

def unary(a,op):
    a=list(a)
    if op=='I':return a
    if op=='N':return [(-1)**i*x for i,x in enumerate(a)]
    if op=='S':
        out=[0]*len(a)
        for i in range((len(a)+1)//2):out[2*i]=a[i]
        return out
    if op=='R':
        # Formal geometric series 1/(1+q)=sum (-q)^k, truncated exactly.
        q=[0]+[-x for x in a[1:]];term=[1]+[0]*(len(a)-1);out=term.copy()
        for _ in range(1,len(a)):
            term=multiply(term,q);out=[x+y for x,y in zip(out,term)]
        return out
    raise ValueError(op)

def execute(a,topology,g):
    left=unary(a,g[0]);right=unary(left if topology=='cascade' else a,g[1])
    if topology=='additive':mixed=[1]+[x+y for x,y in zip(left[1:],right[1:])]
    elif topology=='hadamard':mixed=[1]+[x*y for x,y in zip(left[1:],right[1:])]
    else:mixed=multiply(left,right)
    return unary(mixed,g[2])

def verify(cohort):
    root=Path(cohort);labels={r['identifier']:r for r in map(json.loads,(root/'host_labels.jsonl').read_text().splitlines())}
    public={r['case_id']:r for r in map(json.loads,(root/'public.jsonl').read_text().splitlines())}
    pairs=json.loads((root/'pairs.json').read_text());checked=[]
    for cid,c in labels.items():
        assert execute(c['query'],c['topology'],c['source_graph'])==c['future']
        assert all(execute(f,c['topology'],c['source_graph'])==y for f,y in c['demos'])
        p=public[cid]
        assert set(p)=={'case_id','topology','observed_degree','demos','query','public_sha256'}
        assert p['query']==c['query'][:13] and p['topology']==c['topology']
        assert p['demos']==[{'input':f[:13],'output':y[:13]} for f,y in c['demos']]
        signature=hashlib.sha256(json.dumps({k:v for k,v in p.items() if k not in ['case_id','public_sha256']},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert signature==p['public_sha256']
        checked.append(cid)
    for pair in pairs:
        c,d=labels[pair['recipient']],labels[pair['donor']]
        assert c['query']==d['query'] and c['topology']==d['topology'] and [f for f,_ in c['demos']]==[f for f,_ in d['demos']]
        assert Counter(c['source_graph'])==Counter(d['source_graph']) and c['future']!=d['future']
    return {'status':'passed','independent_checker':'finite geometric inverse and double-loop convolution; no EvalMath import',
        'cohort_worlds_verified':len(checked),'source_pairs_verified':len(pairs),'verified_case_ids':checked,
        'all_admitted_source_pairs_retained':True,'sealed_test_accessed':False,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'files_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['host_labels.jsonl','public.jsonl','pairs.json']}}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=verify(a.cohort);Path(a.out).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
