"""Cross-check all 12D exact frontier-DP references against direct state enumeration."""
import json,hashlib
from fractions import Fraction as F
from pathlib import Path
from audit16 import frontier_exact,group
H=Path(__file__).resolve().parent
r=json.loads((H/'receipt.json').read_text());checks=0;states=0
for w in r['worlds']:
    if w['n']!=12:continue
    for p in (.15,.35):
        obs=[(a&w['hidden']).bit_count()%2^int(u<p) for a,u in zip(w['rows'],w['uniforms'])];u,v=F(str(p)).as_integer_ratio();v-=u;m=len(obs);weights=[u**e*v**(m-e) for e in range(m+1)]
        likelihood=[weights[sum(((a&h).bit_count()%2)!=b for a,b in zip(w['rows'],obs))] for h in range(4096)];states+=len(likelihood);z=sum(likelihood);g=group(w['rows'],obs,p);assert z==frontier_exact(g,12)[0];checks+=1
        for q in w['queries']:
            z1=sum(v for h,v in enumerate(likelihood) if (h&q).bit_count()%2);assert z1==frontier_exact(g,12,q)[0];checks+=1
out=dict(status='PASS',states=states,partition_and_event_crosschecks=checks,receipt_sha256=hashlib.sha256((H/'receipt.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(H/'exhaustive.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
