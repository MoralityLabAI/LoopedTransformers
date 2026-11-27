"""Post-run same-source preflight diagnostic; no new likelihood evaluation."""
import json, random
from collections import Counter
from pathlib import Path
from pilot18 import condition, cut_degree, mask, bits, plan, sha
H=Path(__file__).resolve().parent
r=json.loads((H/'receipt.json').read_text());counts=Counter();rows=[]
for w in r['worlds']:
    if w['n']!=48 or w['family']!='dense':continue
    for j,q in enumerate(w['queries']):
        rr,_,active=condition(w['rows'],[0]*len(w['rows']),(1<<48)-1,q,0)
        menu=[(0,c) for c in (3,5,7)]+[(cut_degree(rr,active,k),3) for k in (1,2)]+[(mask(random.Random(w['seed']+18018+j).sample(bits(active),2)),3)]
        for cut,c in menu:
            a=active^cut;work,slots,splits,scope=plan([x&a for x in rr],a,c);work*=2*(1<<cut.bit_count());s=slots>45*48;t=work>900*48
            counts['total']+=1;counts['slot_reject']+=s;counts['work_reject']+=t;counts['both']+=s and t;counts['feasible']+=not(s or t)
            rows.append(dict(seed=w['seed'],query=j,cut=cut,scope=c,work=work,slots=slots,splits=splits,slot_reject=s,work_reject=t))
out=dict(evidence='post-run same-source structural preflight diagnostic; no additional likelihood work',receipt_sha256=sha(H/'receipt.json'),counts=dict(counts),rows=rows)
(H/'diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');print(out['counts'])
