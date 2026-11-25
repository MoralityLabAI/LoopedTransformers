"""Independent exact matrix/interval audit of the constructed certificate."""
from fractions import Fraction as Q
from pathlib import Path
from itertools import product
from math import factorial
import json
P=Path(__file__).parent

def main():
    w=json.loads((P/'constructed_weights.json').read_text());r=json.loads((P/'certificate_results.json').read_text());d=len(w['init']);assert d==16
    radius=Q(r['region_radius']);eps=Q(r['disturbance_radius']);beta=Q(1,5)
    assert all(Q(v)==0 for v in w['init']+w['skip.0.bias']+w['down.0.bias'])
    for i,row in enumerate(w['skip.0.weight']):
        assert len(row)==d+3 and all(Q(row[j])+int(i==j)==0 for j in range(d))
        assert row[d:]==[0,int(i==0),0]
    assert w['read.weight']==[[1]+[0]*(d-1)] and Q(w['read.bias'][0])==-Q(1,2)
    assert all(v==0 for row in w['up.0.weight'] for v in row[1:d])
    assert all(v==0 for row in w['down.0.weight'][1:] for v in row)
    gain=beta*sum(abs(Q(v)) for v in w['down.0.weight'][0]);assert gain==Q(1,4)
    assert sum(Q(8**k,factorial(k)) for k in range(10))>1024
    err=gain/Q(1024);assert err==Q(r['gelu_output_error_bound']) and err+eps==radius
    count=0
    for p,a,b,n in product((0,1),repeat=4):
        bounds=[]
        for u,c in zip(w['up.0.weight'],w['up.0.bias']):
            v=Q(u[0])*p+Q(u[d])*a+Q(u[d+1])*b+Q(u[d+2])*n+Q(c);rad=abs(Q(u[0]))*radius
            lo,hi=v-rad,v+rad;assert lo>=4 or hi<=-4;bounds.append((max(lo,0),max(hi,0)))
        # Every ReLU stays on one affine branch over the entire region.
        vals=[]
        for h in (Q(p)-radius,Q(p)+radius):
            z=[max(Q(0),Q(u[0])*h+Q(u[d])*a+Q(u[d+1])*b+Q(u[d+2])*n+Q(c)) for u,c in zip(w['up.0.weight'],w['up.0.bias'])]
            vals.append(Q(b)+beta*sum(Q(v)*zz for v,zz in zip(w['down.0.weight'][0],z)))
        assert vals==[p^a^b,p^a^b]
        count+=1
    assert count==16 and Q(r['readout_margin'])==Q(1,2)-radius
    # ReLU plateau derivative is zero; |GELU'-ReLU'| <= (4+1/4)*exp(-8).
    derivative_bound=beta*sum(abs(Q(v)*Q(u[0])) for v,u in zip(w['down.0.weight'][0],w['up.0.weight']))*Q(17,4*1024)
    assert derivative_bound==Q(17,512)<1
    print(f'PASS: {count} exact matrix transition cases, additive disturbance {eps}, contraction bound {derivative_bound}, readout margin {Q(1,2)-radius}')
if __name__=='__main__':main()
