"""Noisy parity observation budget and rank expansion."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
BUDGETS=(2,4,8,16,32)
def dot(a,b): return (int(a)&int(b)).bit_count()&1
def world(k,r,seed):
    g=np.random.default_rng(seed); h=int(g.bit_generator.random_raw())&((1<<k)-1); piv={};basis=[]
    while len(basis)<r+3:
        a=int(g.bit_generator.random_raw())&((1<<k)-1);a^=a.bit_count()&1;b=a
        while b and b.bit_length()-1 in piv:b^=piv[b.bit_length()-1]
        if b:piv[b.bit_length()-1]=b;basis.append(a)
    state=sum(dot(b,h)<<i for i,b in enumerate(basis));tests=set()
    while len(tests)<32:
        q=int(g.integers(1,1<<r,dtype=np.uint64))
        if q.bit_count()>=2:tests.add(q)
    test=sorted(tests);rows={}
    for arm in ('repeat','sparse','dense'):
        a=[1<<i for i in range(r)]
        while len(a)<32*r:
            if arm=='repeat':q=1<<(len(a)%r)
            elif arm=='sparse':q=sum(1<<int(i) for i in g.choice(r,int(g.integers(2,4)),replace=False))
            else:q=int(g.integers(1,1<<r,dtype=np.uint64))
            if q not in tests:a.append(q)
        rows[arm]=a
    targets=[];pool=[]
    if r<=12:
        for j in range(3):
            a=g.choice(np.arange(1,1<<r),56,replace=False).tolist();targets.append(a[:8]);pool.append(a[8:])
    extra=[]
    while len(extra)<3:
        q=int(g.integers(1,1<<r,dtype=np.uint64))
        if q not in tests and q not in rows['dense'][:4*r] and q not in extra:extra.append(q)
    return dict(k=k,r=r,seed=seed,basis=basis,hidden=h,state=state,test=test,rows=rows,uniforms=g.random(32*r).tolist(),extra_uniforms=g.random(3).tolist(),targets=targets,pool=pool,within_extra=extra)
def walsh(p):
    a=p.copy();h=1
    while h<len(a):
        b=a.reshape(-1,2*h);l=b[:,:h].copy();v=b[:,h:].copy();b[:,:h]=l+v;b[:,h:]=l-v;h*=2
    return a
def exact(rows,y,p,r):
    n=1<<r;states=np.arange(n);parity=np.array([i.bit_count()&1 for i in range(n)],dtype=np.int16);e=np.zeros(n,dtype=np.int16)
    for a,b in zip(rows,y):e+=parity[states&a]!=b
    w=(e==0).astype(float) if p==0 else np.exp((e-e.min())*math.log(p/(1-p)))
    assert w.sum()>0;w/=w.sum();return w,walsh(w)
def bp(rows,y,p,r):
    ll={};scale=30. if p==0 else math.log((1-p)/p)
    for a,b in zip(rows,y):ll[a]=ll.get(a,0.)+(1-2*b)*scale
    keys=sorted(ll);d=max(q.bit_count() for q in keys);idx=np.zeros((len(keys),d),dtype=int);mask=np.zeros_like(idx,dtype=bool)
    for j,q in enumerate(keys):
        ix=[i for i in range(r) if q>>i&1];idx[j,:len(ix)]=ix;mask[j,:len(ix)]=True
    v=np.zeros(idx.shape);f=v.copy();sign=np.tanh(np.array([ll[q] for q in keys])/2)[:,None];edges=idx[mask];iters=60
    for step in range(60):
        t=np.where(mask,np.tanh(v/2),1.);left=np.concatenate((np.ones((len(keys),1)),np.cumprod(t[:,:-1],axis=1)),axis=1);right=np.concatenate((np.cumprod(t[:,:0:-1],axis=1)[:,::-1],np.ones((len(keys),1))),axis=1)
        new=np.where(mask,2*np.arctanh(np.clip(sign*left*right,-1+1e-12,1-1e-12)),0.)
        delta=np.max(abs(new-f));f=.5*f+.5*new;total=np.bincount(edges,weights=f[mask],minlength=r);v=np.where(mask,total[idx]-f,0.)
        if delta<1e-6:iters=step+1;break
    return np.tanh(total/2),iters
def bp_queries(mu,qs):return np.array([np.prod([mu[i] for i in range(len(mu)) if q>>i&1]) for q in qs])
def score(mu,labels):
    p=np.clip((1-np.asarray(mu))/2,0,1);y=np.asarray(labels);pred=(p>.5+1e-12).astype(int);take=np.maximum(p,1-p)>=.9-1e-12
    return dict(prob=p.tolist(),correct=int((pred==y).sum()),covered=int(take.sum()),covered_correct=int(((pred==y)&take).sum()),brier=float(np.mean((p-y)**2)))
def state_from_mu(mu):return sum(int(v< -2e-12)<<i for i,v in enumerate(mu))
def run():
    result=[]
    for k,r,nroots in [(16,8,6),(16,12,6),(32,8,6),(32,12,6),(64,24,4),(64,48,4)]:
        for i in range(nroots):
            w=world(k,r,110000+1000*k+10*r+i);s=w['state'];test=w['test'];labels=[dot(q,s) for q in test];w['labels']=labels;w['results']=[];w['expansion']=[]
            for p in ((0.,.15,.35) if r<=12 else (.15,.35)):
                for arm in (('repeat','sparse','dense') if r<=12 else ('repeat','sparse')):
                    a=w['rows'][arm];y=[dot(q,s)^int(u<p) for q,u in zip(a,w['uniforms'])]
                    for b in BUDGETS:
                        m=b*r;rows=a[:m];obs=y[:m];t=time.perf_counter();mu,it=bp(rows,obs,p,r);dt=time.perf_counter()-t;z=state_from_mu(mu);rec=dict(noise=p,arm=arm,budget=b,bp=score(bp_queries(mu,test),labels),bp_state=z,bp_block=z==s%(1<<r),bp_iterations=it,bp_seconds=dt,bp_mu=mu.tolist())
                        if r<=12:
                            t=time.perf_counter();post,wx=exact(rows,obs,p,r);dt=time.perf_counter()-t;z=int(np.flatnonzero(post>=post.max()-1e-15)[0]);rec.update(exact=score(wx[test],labels),exact_state=z,exact_block=z==s%(1<<r),exact_seconds=dt)
                            if arm=='dense' and b==4:
                                er=[];target=[(1<<(r+j))^q for j,ts in enumerate(w['targets']) for q in ts];truth=[dot(q,s) for q in target];rho=1-2*p
                                for policy in ('within','expand_random','expand_target'):
                                    if policy=='within':
                                        calls=w['within_extra'];outs=[dot(q,s)^int(u<p) for q,u in zip(calls,w['extra_uniforms'])];_,after=exact(rows+calls,obs+outs,p,r);outside=np.zeros(len(target))
                                    else:
                                        offsets=[]
                                        for ts,pool in zip(w['targets'],w['pool']):
                                            gain=[float(np.mean(wx[np.array(ts)^q]**2)) for q in pool];offsets.append(pool[0] if policy=='expand_random' else pool[int(np.argmax(gain))])
                                        calls=[(1<<(r+j))^q for j,q in enumerate(offsets)];outs=[dot(q,s)^int(u<p) for q,u in zip(calls,w['extra_uniforms'])];after=wx;outside=np.concatenate([rho*(1-2*v)*wx[np.array(ts)^q] for ts,q,v in zip(w['targets'],offsets,outs)])
                                    er.append(dict(policy=policy,calls=calls,obs=outs,inspan=score(after[test],labels),outside=score(outside,truth),rank_after=r+(0 if policy=='within' else 3)))
                                w['expansion'].append(dict(noise=p,truth=truth,initial=rec['exact'],policies=er))
                        w['results'].append(rec)
            result.append(w);print(k,r,w['seed'],flush=True)
            Path('receipt.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),worlds=result),indent=2)+'\n')
if __name__=='__main__':run()
