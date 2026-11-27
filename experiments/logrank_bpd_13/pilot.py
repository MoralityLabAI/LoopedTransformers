"""Costed noisy parity acquisition with public basis induction."""
import json, hashlib, sys, time
from pathlib import Path
import numpy as np
from bp import bp
H=Path(__file__).resolve().parent; B=24; NOISE=(.15,.35)
ARMS=('exact_stop','exact_fixed','raw_bp','induced_bp','induced_fixed','planted_bp','random_induced','learned_induced'); LARGE=('raw_bp','induced_bp','planted_bp','random_induced')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dot(a,b):return (int(a)&int(b)).bit_count()%2
def lift(rows,c):
    a=0
    while c:
        b=c&-c;a^=rows[b.bit_length()-1];c^=b
    return a
class Frame:
    def __init__(self,rows):
        self.piv={};self.rows=[]
        for a in rows:
            a=int(a);v=a;c=0
            while v and v.bit_length()-1 in self.piv:
                b,t=self.piv[v.bit_length()-1];v^=b;c^=t
            if v:self.piv[v.bit_length()-1]=(v,c^(1<<len(self.rows)));self.rows.append(a)
    def coord(self,a):
        a=int(a);c=0
        while a:
            v=self.piv.get(a.bit_length()-1)
            if v is None:return None
            b,t=v;a^=b;c^=t
        return c
def walsh(a):
    a=a.copy();h=1
    while h<len(a):
        x=a.reshape(-1,2*h);l=x[:,:h].copy();r=x[:,h:].copy();x[:,:h]=l+r;x[:,h:]=l-r;h*=2
    return a
def world(r,i):
    d=r+3;seed=130000+100*r+i;g=np.random.default_rng(seed);lim=1<<d;old=1<<r;planted=[]
    while len(planted)<d:
        a=int(g.integers(1,lim,dtype=np.uint64))
        if Frame(planted).coord(a) is None:planted.append(a)
    def sample(lo,hi,n,exclude):
        out=[];seen=set(exclude)
        while len(out)<n:
            q=int(g.integers(lo,hi,dtype=np.uint64))
            if q not in seen:out.append(q);seen.add(q)
        return out
    target=sample(1,old,8,[1<<j for j in range(r)]);pool=sample(1,old,16,target)
    for j in range(3):
        off=sample(0,old,16,[]);target += [(1<<(r+j))^q for q in off[:8]];pool += [(1<<(r+j))^q for q in off[8:]]
    seen=set(pool+target)
    while len(pool)<48:
        a=int(g.integers(old,lim,dtype=np.uint64))
        if (a>>r).bit_count()>=2 and a not in seen:pool.append(a);seen.add(a)
    initial=[1<<j for j in range(r)]
    while len(initial)<4*r:
        a=sum(1<<int(j) for j in g.choice(r,int(g.integers(2,4)),replace=False))
        if a not in target:initial.append(a)
    g.shuffle(pool);cost=g.integers(1,4,len(pool)).tolist();mass=.25 if i%2==0 else .75
    p,init,qs=[[lift(planted,q) for q in a] for a in (pool,initial,target)]
    assert not set(qs)&(set(init)|set(p)) and len(set(p))==48
    induced=Frame(init+[p[j] for j in sorted(range(len(p)),key=lambda j:(cost[j],j))]).rows;assert len(induced)==d
    return dict(r=r,d=d,seed=seed,hidden=int(g.integers(lim,dtype=np.uint64)),planted=planted,induced=induced,initial=init,pool=p,costs=cost,targets=qs,weights=[mass/8]*8+[(1-mass)/24]*24,old_mass=mass,uniforms=g.random(len(init)+B).tolist(),random_order=g.permutation(48).tolist())
def exact_setup(w):
    f=Frame(w['induced']);n=1<<w['d'];s=np.arange(n);par=np.array([int(i).bit_count()%2 for i in s]);return f,s,par,np.ones(n)/n
def upd(f,s,par,p,a,y,noise):
    p=p*np.where(par[s&f.coord(a)]==y,1-noise,noise);return p/p.sum()
def exact_moment(f,p):
    mu=walsh(p);return lambda a:np.array([mu[f.coord(int(q))] for q in np.asarray(a).flat]).reshape(np.shape(a))
def bp_moment(w,rows,y,noise,kind):
    d=w['d'];f=Frame([1<<j for j in range(d)] if kind=='raw' else w[kind]);sp=Frame(rows);mu,it=bp([f.coord(a) for a in rows],y,noise,d);cache={}
    def moment(a):
        a=np.asarray(a);out=[]
        for q in a.flat:
            q=int(q)
            if q not in cache:
                v=0.
                if sp.coord(q) is not None:
                    c=f.coord(q);v=1.
                    while c:
                        b=c&-c;v*=mu[b.bit_length()-1];c^=b
                cache[q]=float(v)
            out.append(cache[q])
        return np.array(out).reshape(a.shape)
    return moment,it
def gain(m,q,a,wt,p):
    a=np.array(a,dtype=np.int64);q=np.array(q,dtype=np.int64);ma=m(a);mq=m(q);rho=1-2*p;cov=m(q[:,None]^a)-mq[:,None]*ma[None,:]
    return rho*rho*(wt@cov**2)/(4*(1-(rho*ma)**2))
def features(w,m,rows,ids,p,spent):
    q=w['targets'];wt=np.array(w['weights']);a=[w['pool'][j] for j in ids];c=np.array([w['costs'][j] for j in ids]);sp=Frame(rows);f=Frame(w['induced']);g=gain(m,q,a,wt,p);ma=abs(m(a));unc=float(wt@(1-m(q)**2));inside=[sp.coord(v) is not None for v in q];x=[]
    for i,v in enumerate(a):
        x.append([g[i]/c[i],g[i],c[i],float(sp.coord(v) is None),sum(z for u,z,t in zip(q,wt,inside) if not t and sp.coord(u^v) is not None),sum(z for u,z in zip(q,wt) if sp.coord(u^v) is not None),ma[i],f.coord(v).bit_count()/w['d'],(B-spent)/B,p,len(sp.rows)/w['d'],unc,w['old_mass']])
    return np.array(x)
def score(m,w):
    p=np.clip((1-m(w['targets']))/2,0,1);y=np.array([dot(q,w['hidden']) for q in w['targets']]);wt=np.array(w['weights']);take=np.maximum(p,1-p)>=.9-1e-12;ok=(p>.5+1e-12)==y
    return dict(prob=p.tolist(),brier=float(wt@(p-y)**2),risk=float(wt@(p*(1-p))),accuracy=float(wt@ok),old_correct=int(ok[:8].sum()),new_correct=int(ok[8:].sum()),coverage=float(wt@take),covered_correct=float(wt@(take&ok)))
def run(w,noise,arm,model=None,collect=False):
    rows=w['initial'].copy();obs=[dot(a,w['hidden'])^int(u<noise) for a,u in zip(rows,w['uniforms'])];initial=obs.copy();ids=list(range(48));spent=0;trace=[];samples=[];elapsed=0.;its=0;exact=w['d']<=11
    if exact:
        f,s,par,p=exact_setup(w)
        for a,y in zip(rows,obs):p=upd(f,s,par,p,a,y,noise)
    kind='raw' if arm=='raw_bp' else 'planted' if arm=='planted_bp' else 'induced';fixed=arm.endswith('_fixed')
    while True:
        start=time.perf_counter()
        if arm.startswith('exact'):m=exact_moment(f,p)
        else:m,it=bp_moment(w,rows,obs,noise,kind);its+=it
        final=score(m,w);elapsed+=time.perf_counter()-start;avail=[j for j in ids if w['costs'][j]+spent<=B]
        if not avail or (not fixed and final['coverage']>=.8-1e-12 and not collect):reason='confidence' if avail else 'budget';break
        start=time.perf_counter();pool=[w['pool'][j] for j in avail];c=np.array([w['costs'][j] for j in avail]);wt=np.array(w['weights'])
        if collect:
            bm,_=bp_moment(w,rows,obs,noise,'induced');samples.append((features(w,bm,rows,avail,noise,spent),gain(exact_moment(f,p),w['targets'],pool,wt,noise)/c))
        if arm.startswith('exact'):v=gain(m,w['targets'],pool,wt,noise)/c
        elif arm=='random_induced':v=np.array([-w['random_order'].index(j) for j in avail])
        elif arm=='learned_induced':v=((features(w,m,rows,avail,noise,spent)-model['mean'])/model['std'])@model['coef']
        else:v=gain(m,w['targets'],pool,wt,noise)/c
        j=avail[int(v.argmax())];elapsed+=time.perf_counter()-start;a=w['pool'][j];c=w['costs'][j];y=dot(a,w['hidden'])^int(w['uniforms'][len(w['initial'])+len(trace)]<noise);trace.append(dict(j=j,query=a,obs=y,cost=c,expands=Frame(rows).coord(a) is None,coverage=final['coverage'],score=float(v.max())));spent+=c;ids.remove(j);rows.append(a);obs.append(y)
        if exact:p=upd(f,s,par,p,a,y,noise)
    out=dict(arm=arm,noise=noise,spent=spent,reason=reason,rank=len(Frame(rows).rows),initial_obs=initial,trace=trace,final=final,seconds=elapsed,bp_iterations=its)
    if exact:
        ref=score(exact_moment(f,p),w);out['reference']=ref;out['posterior_regret']=float(np.array(w['weights'])@(np.array(final['prob'])-np.array(ref['prob']))**2)
    if arm=='exact_fixed':out['same_trace']={b:score(bp_moment(w,rows,obs,noise,b)[0],w) for b in ('raw','induced','planted')}
    return out,samples
def train():
    x=[];y=[];seeds=[]
    for r in (6,8):
        for i in range(6):
            w=world(r,i);seeds.append(w['seed'])
            for p in NOISE:
                for a in ('exact_fixed','random_induced'):
                    _,data=run(w,p,a,collect=True);x.extend(a for a,b in data);y.extend(b for a,b in data)
    x=np.concatenate(x);y=np.concatenate(y);mean=x.mean(0);std=x.std(0);std[std<1e-12]=1;z=(x-mean)/std;coef=np.linalg.solve(z.T@z+10*np.eye(z.shape[1]),z.T@(y-y.mean()));model=dict(mean=mean.tolist(),std=std.tolist(),coef=coef.tolist(),intercept=float(y.mean()),samples=len(y),rmse=float(np.sqrt(np.mean((z@coef+y.mean()-y)**2))),train_seeds=seeds,source_sha256=sha(__file__),protocol_sha256=sha(H/'PROTOCOL.md'),bp_sha256=sha(H/'bp.py'));(H/'model.json').write_text(json.dumps(model,indent=2)+'\n');print('TRAIN',len(y),model['rmse'],flush=True)
def evaluate(large=False):
    model=json.loads((H/'model.json').read_text());assert model['source_sha256']==sha(__file__);m={k:np.array(model[k]) for k in ('mean','std','coef')};worlds=[]
    for r in ((21,45) if large else (6,8)):
        for i in (range(200,204) if large else range(100,112)):
            w=world(r,i);w['runs']=[]
            for p in NOISE:
                for a in (LARGE if large else ARMS):w['runs'].append(run(w,p,a,m)[0])
            worlds.append(w);print('EVAL',r,w['seed'],flush=True);(H/('scale.json' if large else 'receipt.json')).write_text(json.dumps(dict(source_sha256=sha(__file__),protocol_sha256=sha(H/'PROTOCOL.md'),model_sha256=sha(H/'model.json'),worlds=worlds),separators=(',',':'))+'\n')
if __name__=='__main__':
    if sys.argv[1]=='train':train()
    else:evaluate(sys.argv[1]=='scale')
