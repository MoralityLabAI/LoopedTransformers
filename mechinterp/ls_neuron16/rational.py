"""Exact rational outward-rounded enclosures for the selected real GELU recurrence.
Only stdlib. GELU(z)=z*Phi(z), beta=1/5 exactly. Not a floating-point kernel proof.
"""
from fractions import Fraction as Q
from dataclasses import dataclass
from pathlib import Path
import json, math, hashlib
P=Path(__file__).parent
G=1<<80

def floor_grid(x): return Q(x.numerator*G//x.denominator,G)
def ceil_grid(x): return -floor_grid(-x)
@dataclass(frozen=True)
class I:
    lo:Q
    hi:Q
    def __post_init__(self):
        assert self.lo<=self.hi
    def __add__(a,b):
        b=box(b);return outward(a.lo+b.lo,a.hi+b.hi)
    __radd__=__add__
    def __mul__(a,b):
        b=box(b);xs=[a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi];return outward(min(xs),max(xs))
    __rmul__=__mul__
    def __neg__(a):return I(-a.hi,-a.lo)
    def __sub__(a,b):return a+-box(b)
    def out(a):return [str(a.lo),str(a.hi)]

def outward(lo,hi):return I(floor_grid(lo),ceil_grid(hi))
def box(x):return x if isinstance(x,I) else outward(Q(x),Q(x))

def atan_inv(n,terms=40):
    s=sum((Q((-1)**k,(2*k+1)*n**(2*k+1)) for k in range(terms)),Q(0));next_term=Q((-1)**terms,(2*terms+1)*n**(2*terms+1))
    return I(min(s,s+next_term),max(s,s+next_term))
# Machin's identity pi=16 atan(1/5)-4 atan(1/239), alternating-series bounds.
a,b=atan_inv(5),atan_inv(239)
pi_lo,pi_hi=16*a.lo-4*b.hi,16*a.hi-4*b.lo
CLO=Q('0.398942280401432677939946059934');CHI=Q('0.398942280401432677939946059935')
assert 2*pi_hi*CLO*CLO < 1 < 2*pi_lo*CHI*CHI
# e^8 > 1024, by positive Taylor partial sum, hence e^-32 < 2^-40.
assert sum((Q(8**k,math.factorial(k)) for k in range(10)),Q(0))>1024

def cdf_pos(x):
    assert x>=0
    if x>=8:return I(1-Q(1,1<<40),Q(1)) # Mills ratio: tail <= phi(x)/x <= exp(-32)
    # integral_0^x exp(-t^2/2) dt. Taylor with Lagrange remainder integrated.
    # term k = (-1)^k x^(2k+1)/(2^k k! (2k+1))
    term=x;total=term;k=0
    while True:
        nxt=-term*x*x*Q(2*k+1,2*(k+1)*(2*k+3))
        if abs(nxt)<=Q(1,1<<90):break
        total+=nxt;term=nxt;k+=1
        assert k<240
    err=abs(nxt)
    lo,hi=max(Q(0),total-err),total+err
    return outward(max(Q(1,2),Q(1,2)+CLO*lo),min(Q(1),Q(1,2)+CHI*hi))

def cdf(x):
    if x>=0:return cdf_pos(x)
    p=cdf_pos(-x);return I(1-p.hi,1-p.lo)
def gelu(x):
    # Phi is monotone even though GELU is not; enclose the product independently.
    return x*I(cdf(x.lo).lo,cdf(x.hi).hi)
def affine(w,b,x):return [sum((box(a)*v for a,v in zip(row,x)),box(c)) for row,c in zip(w,b)]
def read_weights(path):
    # JSON floats are exact decimal representations that round-trip to the dyadic stored value.
    w=json.loads(path.read_text());return w

def forward(w,inputs):
    h=[box(v) for v in w['init']];logits=[]
    for x in inputs:
        v=h+[box(t) for t in x];skip=affine(w['skip.0.weight'],w['skip.0.bias'],v)
        a=[gelu(t) for t in affine(w['up.0.weight'],w['up.0.bias'],v)]
        down=affine(w['down.0.weight'],w['down.0.bias'],a)
        h=[p+s+Q(1,5)*d for p,s,d in zip(h,skip,down)]
        logits.append(affine(w['read.weight'],w['read.bias'],h)[0])
    return logits

def main():
    data=json.loads((P/'witness_data.json').read_text());w={}
    for k,t in data['tensors'].items():
        vals=[float.fromhex(v) for v in t['values']];flat=[vals[int(i,16)] for i in t['codes']];shape=t['shape']
        assert len(flat)==math.prod(shape)
        w[k]=flat if len(shape)==1 else [flat[i:i+shape[1]] for i in range(0,len(flat),shape[1])]
    serialized=(json.dumps(w,sort_keys=True)+'\n').encode();assert hashlib.sha256(serialized).hexdigest()==data['q4_weights_sha256']
    if (P/'q4_weights.json').exists():assert (P/'q4_weights.json').read_bytes()==serialized
    r=data['witness'];assert r
    ys=forward(w,r['inputs']);par=0;rows=[]
    for x,y in zip(r['inputs'],ys):
        par^=x[0]^x[1];margin=y if par else -y
        rows.append(dict(input=x,parity=par,logit=y.out(),margin=margin.out(),logit_decimal=[float(y.lo),float(y.hi)]))
    assert par==r['parity'] and (ys[-1].hi<0 if par else ys[-1].lo>0)
    assert all(Q(row['margin'][0])>0 for row in rows[:-1])
    result=dict(semantics='Exact real arithmetic with stored FP32-quantized dyadic weights and exact beta=1/5, exact GELU',grid_denominator=G,rows=rows,verdict='CERTIFIED_REACHABLE_COUNTEREXAMPLE')
    (P/'interval_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result['verdict'],rows[-1]['logit_decimal'],flush=True)
if __name__=='__main__':main()
