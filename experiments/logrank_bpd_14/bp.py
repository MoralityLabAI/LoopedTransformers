"""Damped BP from LOGRANK-BPD-11, 40-iteration cap."""
import math
import numpy as np
def bp(rows,y,p,r):
    ll={};scale=30. if p==0 else math.log((1-p)/p)
    for a,b in zip(rows,y):ll[a]=ll.get(a,0.)+(1-2*b)*scale
    keys=sorted(ll);d=max(q.bit_count() for q in keys);idx=np.zeros((len(keys),d),dtype=int);mask=np.zeros_like(idx,dtype=bool)
    for j,q in enumerate(keys):
        ix=[i for i in range(r) if q>>i&1];idx[j,:len(ix)]=ix;mask[j,:len(ix)]=True
    v=np.zeros(idx.shape);f=v.copy();sign=np.tanh(np.array([ll[q] for q in keys])/2)[:,None];edges=idx[mask];iters=40
    for step in range(40):
        t=np.where(mask,np.tanh(v/2),1.);left=np.concatenate((np.ones((len(keys),1)),np.cumprod(t[:,:-1],axis=1)),axis=1);right=np.concatenate((np.cumprod(t[:,:0:-1],axis=1)[:,::-1],np.ones((len(keys),1))),axis=1)
        new=np.where(mask,2*np.arctanh(np.clip(sign*left*right,-1+1e-12,1-1e-12)),0.)
        delta=np.max(abs(new-f));f=.5*f+.5*new;total=np.bincount(edges,weights=f[mask],minlength=r);v=np.where(mask,total[idx]-f,0.)
        if delta<1e-6:iters=step+1;break
    return np.tanh(total/2),iters
