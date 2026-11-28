"""15B: adaptive likelihood-weighted matroid basis, on exposed 15 worlds."""
import json
from collections import defaultdict
from pathlib import Path
from fractions import Fraction as F
import pilot15 as p
H=Path(__file__).resolve().parent
class Weighted(p.Bounds):
    def __init__(self,rows,obs,r,noise,mode):
        super().__init__(rows,obs,r,noise,mode)
        self.residual=[]
        for d in range(r+1):
            g=defaultdict(list);mask=(1<<d)-1
            for (a,y),n in self.counts.items():
                if a&~mask:g[a&~mask].append((a&mask,y,n))
            self.residual.append([(a,t,sum(n for _,_,n in t)) for a,t in g.items()])
    def bound(self,d,h):
        fixed,_,nfree,_=self.level[d];e=sum(n for a,y,n in fixed if p.parity(a&h)!=y);lo=hi=self.pu[e]*self.pv[self.m-nfree-e];items=[]
        for a,ts,n in self.residual[d]:
            e=sum(c for pre,y,c in ts if p.parity(pre&h)!=y);x=self.pu[e]*self.pv[n-e];y=self.pu[n-e]*self.pv[e];items.append((-abs(n-2*e),a,x,y))
        piv={}
        for _,a,x,y in sorted(items):
            v=a
            while v and v.bit_length()-1 in piv:v^=piv[v.bit_length()-1]
            if v:piv[v.bit_length()-1]=v;lo*=x+y;hi*=x+y
            else:lo*=min(x,y);hi*=max(x,y)
        free=self.r-d-len(piv);return lo<<free,hi<<free

def main():
    rows=[];old=p.Bounds;p.Bounds=Weighted
    for r in (8,21,45):
        rec=json.loads((H/f'receipt{r}.json').read_text());out=[]
        for w in rec['worlds']:
            a=p.search(w['rows'],w['obs'],w['rank'],w['noise'],'weighted',w['modal'],w['targets'],w['truth'],w['weights']);w['arms']=[a];out.append(w)
            s=a['snapshots'][-1];print(w['d'],w['seed'],w['noise'],s['retained_lower'],s['coverage'],s['leaves'],flush=True)
        (H/f'weighted{r}.json').write_text(json.dumps(dict(evidence='adaptive exposed-root diagnostic',source_sha256=p.sha(__file__),pilot_sha256=p.sha(H/'pilot15.py'),protocol_sha256=p.sha(H/'PROTOCOL.md'),worlds=out),separators=(',',':'))+'\n')
    p.Bounds=old
if __name__=='__main__':main()
