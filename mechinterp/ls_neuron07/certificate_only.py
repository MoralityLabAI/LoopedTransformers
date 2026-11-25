"""Standalone exact-real certificate of the repaired, compacted seed-107 cell."""
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
from rational import I,box,gelu
import json

def main():
    w=json.loads(Path(__file__).with_name('compact_weights.json').read_text());assert len(w['down'])==len(w['up'])==len(w['up_bias'])==3
    assert all(len(row)==4 for row in w['up'])
    def transition(h,a,b,n):
        x=[h,box(a),box(b),box(n)]
        pre=[sum((box(q)*v for q,v in zip(row,x)),box(c)) for row,c in zip(w['up'],w['up_bias'])]
        skip=[Q(1)+Q(w['skip'][0])]+list(map(Q,w['skip'][1:]))
        return sum((q*v for q,v in zip(skip,x)),box(w['skip_bias'][0]))+Q(1,5)*sum((Q(q)*gelu(z) for q,z in zip(w['down'],pre)),box(w['down_bias'][0]))
    R=Q(1,8);eps=Q(1,256);inner=R-eps;leaves=0;boxes=0
    for p,a,b,n in product((0,1),repeat=4):
        t=p^a^b;todo=[(Q(p)-R,Q(p)+R,0)];seen=0
        while todo:
            l,u,d=todo.pop();out=transition(I(l,u),a,b,n);boxes+=1
            if out.lo>=t-inner and out.hi<=t+inner:leaves+=1;seen+=1;continue
            assert d<14,'inclusion unresolved';m=(l+u)/2;todo.extend([(l,m,d+1),(m,u,d+1)])
        assert seen>0
    assert Q(0)>=-R and Q(0)<=R and Q(1,2)-R==Q(3,8)
    print(f'PASS: all 16 region/input transitions, {boxes} rational boxes and {leaves} leaves, disturbance {eps}, readout margin 3/8; induction from h=0 gives arbitrary-depth correctness.')
if __name__=='__main__':main()
