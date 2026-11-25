"""Rational verification of reset/wrap and shared-carry modular cells."""
from fractions import Fraction as Q
from itertools import product
from math import ceil, log2, factorial
from pathlib import Path
import json
R=Q(1,8); BETA=Q(1,5); EPS=Q(1,32)

def zmat(r,c): return [[Q(0) for _ in range(c)] for _ in range(r)]
def model(d,w,centers,scale,skip,offset,ramps,K,delta,bits):
    # ramp = (integer affine direction, threshold, vector output coefficient)
    U=[];b=[];V=zmat(d,2*len(ramps));
    for i,(direction,threshold,coef) in enumerate(ramps):
        for sign in (1,-1):
            U.append([K*Q(v) for v in direction]);b.append(K*(-threshold+sign*delta))
        for j in range(d):
            V[j][2*i]=Q(coef[j])/(BETA*2*K*delta);V[j][2*i+1]=-V[j][2*i]
    return dict(d=d,w=w,centers=centers,scales=scale,bits=bits,skip=skip,offset=offset,U=U,b=b,V=V,ramps=ramps,K=K,delta=delta)

def scalar(m,K=32):
    K=Q(K);offset=-Q(m,2);delta=Q(1,4);ramps=[]
    for j in range(1,m):ramps.append(([1,0,0,0],offset+j-Q(1,2),[1]))
    ramps.append(([1,1,1,0],offset+m-Q(1,2),[-m]))
    bits=1+ceil(log2(2*m))
    return model(1,2*m,[[offset+p] for p in range(m)], [Q(1),K,K/4,Q(10)/K], [[-1,1,1,0]],[offset],ramps,K,delta,bits)

def binary(k,L=1,K=512):
    L=Q(L);K=Q(K);delta=L/8;ramps=[];skip=zmat(k,k+3)
    for j in range(k):
        skip[j][j]=-1
        if j==0:skip[j][k]=skip[j][k+1]=L
    for j in range(k):
        u=[0]*(k+3);u[j]=1;c=[0]*k;c[j]=L;ramps.append((u,-L/2,c))
    for j in range(k):
        u=[0]*(k+3);u[0]=1
        for l in range(1,j+1):u[l]=2
        u[k]=u[k+1]=L;c=[0]*k;c[j]=-2*L
        if j+1<k:c[j+1]=L
        ramps.append((u,L/2,c))
    return model(k,4*k,[[L*(((p>>j)&1)-1) for j in range(k)] for p in range(1<<k)], [Q(1),K,K*L/8,Q(20)/K],skip,[-L]*k,ramps,K,delta,4)

def flat(x):
    if isinstance(x,(list,tuple)):return [v for y in x for v in flat(y)]
    return [Q(x)]

def certify(mdl,rho=R):
    d=mdl['d'];w=mdl['w'];centers=mdl['centers'];K=mdl['K'];delta=mdl['delta'];m=len(centers)
    codes={};lim=2**(mdl['bits']-1)-1
    for name,scale in zip(('skip','U','b','V'),mdl['scales']):
        c=[x/scale for x in flat(mdl[name])];assert all(x.denominator==1 and abs(x)<=lim for x in c),(name,c);codes[name]=sorted(set(map(int,c)))
    assert len(mdl['ramps'])*2==w
    # |GELU(z)-ReLU(z)|=|z|Phi(-|z|)<=phi(|z|)<2/5.
    # For |z|>=4, this is < e^-8 < 1/1024. Pairwise difference is bounded by the same eta.
    assert sum(Q(8**i,factorial(i)) for i in range(10))>1024
    # phi(0)<2/5 from pi>25/8 (e.g. Machin bound in rational.py).
    from rational import pi_lo
    assert pi_lo>Q(25,8)
    err=[Q(0)]*d;min_abs=None;cases=0;relu_cases=[]
    for p,a,b,n in product(range(m), (0,1),(0,1),(0,1)):
        c=centers[p];t=centers[(p+a+b)%m];x=c+[Q(a),Q(b),Q(n)];v=[];es=[Q(0)]*d;states=[]
        for direction,theta,coef in mdl['ramps']:
            z=sum(Q(u)*xj for u,xj in zip(direction,x))-theta
            rad=rho*sum(abs(Q(u)) for u in direction[:d]);lo=z-rad;hi=z+rad
            # Plateau even at a hinge endpoint; exact linear extension of ReLU.
            assert hi<=-delta or lo>=delta,(p,a,b,n,z,rad,theta,delta)
            s=int(lo>=delta);states.append(s);v.append(s)
            sep=K*(lo-delta if s else -delta-hi);min_abs=sep if min_abs is None else min(sep,min_abs)
            eta=Q(1,1024) if sep>=4 else Q(2,5)
            for j in range(d):es[j]+=abs(Q(coef[j]))*eta/(2*K*delta)
        for j in range(d):
            # Actual residual+skip cancels every state coordinate; all ramps plateau.
            assert all(Q(mdl['skip'][j][i])+int(i==j)==0 for i in range(d))
            val=mdl['offset'][j]+sum(Q(u)*xj for u,xj in zip(mdl['skip'][j][d:],x[d:]))+sum(Q(ramp[2][j])*state for ramp,state in zip(mdl['ramps'],v))
            assert val==t[j],(p,a,b,n,j,val,t[j]);err[j]=max(err[j],es[j])
        cases+=1;relu_cases.append([p,a,b,n,(p+a+b)%m,states])
    eps=rho-max(err);assert eps>=EPS
    sep=min(max(abs(a-b) for a,b in zip(x,y)) for i,x in enumerate(centers) for y in centers[i+1:]);assert rho<sep/2
    count=sum(len(flat(mdl[name])) for name in ('skip','offset','U','b','V'))+d # down bias zero
    return {'states':m,'state_dim':d,'width':w,'weight_bits':mdl['bits'],'recurrence_slots':count,'radius':str(rho),'min_preactivation_abs':str(min_abs),'max_coordinate_error':[str(e) for e in err],'epsilon':str(eps),'readout_margin':str(sep/2-rho),'state_separation':str(sep),'cases':cases,'codes':codes,'relu_cases':relu_cases}

def scalar_weights(m=4):
    a=scalar(m);return {'skip':[float(v) for v in a['skip'][0]],'skip_bias':[float(a['offset'][0])],'up':[[float(v) for v in row] for row in a['U']],'up_bias':[float(v) for v in a['b']],'down':[float(v) for v in a['V'][0]],'down_bias':[0.]},[float(v) for v in a['scales']]

def main():
    results={}
    for m in (2,3,4,8,16):results[f'scalar_{m}']=certify(scalar(m))
    for L,kmax in ((1,2),(4,6)):
        for k in range(1,kmax+1):results[f'binary_L{L}_k{k}']=certify(binary(k,L))
    for key,r in results.items():print(key,'W',r['width'],'bits',r['weight_bits'],'eps',r['epsilon'],'cases',r['cases'])
    Path('construction_results.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
