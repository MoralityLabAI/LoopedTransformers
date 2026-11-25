"""Exact rational certificate for a constructed four-GELU parity recurrence.
The mathematical GELU tail bound is documented in README.md. No training.
"""
import itertools,json,math
from fractions import Fraction as Q
from pathlib import Path
P=Path(__file__).parent
R=Q(1,8);K=32;BETA=Q(1,5);ETA=Q(1,1024);ERR=4*Q(1,16)*ETA;EPS=R-ERR

def ramp(s):return 2*(max(Q(0),s-Q(1,4))-max(Q(0),s-Q(3,4)))

def weights(d=16):
    # q = [state, a,b,nuisance]. Only the first state coordinate is used.
    def vector(h,a,b,n=0):return [h]+[0]*(d-1)+[a,b,n]
    def bypass(i):return [-int(j==i) for j in range(d)]+[0,int(i==0),0]
    return {'init':[0]*d,'read.weight':[[1]+[0]*(d-1)],'read.bias':[-.5],
        'skip.0.weight':[bypass(i) for i in range(d)],'skip.0.bias':[0]*d,
        'up.0.weight':[vector(K,K,-K)]*2+[vector(K,K,K)]*2,
        'up.0.bias':[-8,-24,-40,-56],
        'down.0.weight':[[.3125,-.3125,-.3125,.3125]]+[[0]*4 for _ in range(d-1)],'down.0.bias':[0]*d}

def flatten(x):return [v for t in x for v in flatten(t)] if isinstance(x,list) else [Q(x)]

def certify():
    assert sum((Q(8**k,math.factorial(k)) for k in range(10)),Q(0))>1024
    # Mills: |GELU(u)-ReLU(u)|=|u|*Phi(-|u|)<=phi(|u|)<exp(-8)<1/1024 for |u|>=4.
    w=weights();levels={};n=0
    for name,t in w.items():
        flat=flatten(t);n+=len(flat);mx=max(map(abs,flat));codes=[Q(7)*v/mx if mx else Q(0) for v in flat]
        assert all(c.denominator==1 and abs(c)<=7 for c in codes)
        levels[name]={'scale':str(mx/7),'levels':sorted(set(int(c) for c in codes))}
    rows=[]
    for p,a,b,nuisance in itertools.product((0,1),repeat=4):
        hlo,hhi=Q(p)-R,Q(p)+R;s1=(hlo+a-b,hhi+a-b);s2=(hlo+a+b-1,hhi+a+b-1)
        zs=[(K*(s[0]-theta),K*(s[1]-theta)) for s in (s1,s2) for theta in (Q(1,4),Q(3,4))]
        assert all(lo>=4 or hi<=-4 for lo,hi in zs)
        r1=tuple(map(ramp,s1));r2=tuple(map(ramp,s2));y=p^a^b
        assert r1[0]==r1[1] and r2[0]==r2[1] and b+r1[0]-r2[0]==y
        # Exact ReLU expression is constant on the input interval; four GELU errors bounded by ERR.
        assert ERR+EPS==R and R<Q(1,2)
        rows.append(dict(p=p,input=[a,b,nuisance],target=y,preactivation_intervals=[[str(lo),str(hi)] for lo,hi in zs],ramp_values=[str(r1[0]),str(r2[0])],output_enclosure_with_disturbance=[str(Q(y)-R),str(Q(y)+R)]))
    assert n==513 and sum(len(flatten(t)) for t in weights(1).values())==33
    result=dict(verdict='INDUCTIVE_CERTIFICATE',semantics='exact-real GELU and exact rational beta 1/5, additive per-step Linf state disturbance',state_dimension=16,mlp_width=4,parameters=n,scalar_parameters=33,region_radius=str(R),gelu_output_error_bound=str(ERR),disturbance_radius=str(EPS),readout_margin=str(Q(1,2)-R),weight_levels=levels,transition_rows=rows)
    (P/'constructed_weights.json').write_text(json.dumps(w,sort_keys=True)+'\n');(P/'certificate_results.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(rows),'region/input cases, epsilon',EPS,'margin',Q(1,2)-R)
if __name__=='__main__':certify()
