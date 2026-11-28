"""Independent integer posterior, edge-list BP, budget and gauge replay."""
import json, hashlib, math
from fractions import Fraction
from pathlib import Path
from collections import Counter
import numpy as np
H=Path(__file__).resolve().parent
B=24
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dot(a,b):return (int(a)&int(b)).bit_count()%2
class Basis:
    # LSB pivots, independently implemented coordinate tracking.
    def __init__(self,rows):
        self.piv={};self.rows=[]
        for a in rows:
            a=int(a);v=a;c=1<<len(self.rows)
            while v:
                i=(v&-v).bit_length()-1
                if i not in self.piv:self.piv[i]=(v,c);self.rows.append(a);break
                b,t=self.piv[i];v^=b;c^=t
    def coord(self,a):
        a=int(a);c=0
        while a:
            i=(a&-a).bit_length()-1
            if i not in self.piv:return None
            v,t=self.piv[i];a^=v;c^=t
        return c

def edge_bp(rows,obs,p,d):
    ll={};scale=math.log((1-p)/p)
    for q,b in zip(rows,obs):ll[q]=ll.get(q,0.)+(1-2*b)*scale
    keys=sorted(ll);fi=[];vi=[]
    for j,q in enumerate(keys):
        for i in range(d):
            if q>>i&1:fi.append(j);vi.append(i)
    fi=np.array(fi);vi=np.array(vi);f=np.zeros(len(fi));v=f.copy();n=len(keys);sg=np.tanh(np.array([ll[q] for q in keys])/2);iters=40
    for t in range(40):
        a=np.tanh(v/2);zero=a==0;prod=np.ones(n);np.multiply.at(prod,fi,np.where(zero,1.,a));zc=np.bincount(fi,weights=zero,minlength=n);z=zc[fi]-zero
        exc=np.divide(prod[fi],a,out=prod[fi].copy(),where=~zero);exc=np.where(z==0,exc,0.);new=2*np.arctanh(np.clip(sg[fi]*exc,-1+1e-12,1-1e-12));delta=np.max(abs(new-f));f=(f+new)/2;total=np.bincount(vi,weights=f,minlength=d);v=total[vi]-f
        if delta<1e-6:iters=t+1;break
    return np.tanh(total/2),iters

def bp_prob(w,rows,obs,p,kind):
    d=w['d'];f=Basis([1<<i for i in range(d)] if kind=='raw' else w[kind]);s=Basis(rows);mu,it=edge_bp([f.coord(q) for q in rows],obs,p,d)
    cache={}
    def moment(q):
        q=int(q)
        if q not in cache:
            c=f.coord(q);cache[q]=0. if s.coord(q) is None else math.prod(mu[i] for i in range(d) if c>>i&1)
        return cache[q]
    return moment,it

def exact_moment(rows,obs,p,d):
    n=1<<d;states=np.arange(n);par=np.fromiter((i.bit_count()%2 for i in range(n)),np.int8,n);errors=np.zeros(n,dtype=int)
    for a,b in zip(rows,obs):errors+=(par[states&a]!=b)
    u=Fraction(str(p));m=len(rows);weights=[u.numerator**i*(u.denominator-u.numerator)**(m-i) for i in range(m+1)];f=np.bincount(errors,minlength=m+1);den=sum(int(c)*v for c,v in zip(f,weights));assert den>0
    # Independent physical-coordinate integer Walsh transform.
    z=np.array([weights[e] for e in errors],dtype=object);h=1
    while h<n:
        for i in range(0,n,2*h):
            a=z[i:i+h].copy();b=z[i+h:i+2*h].copy();z[i:i+h]=a+b;z[i+h:i+2*h]=a-b
        h*=2
    assert z[0]==den
    return lambda q:float(Fraction(z[int(q)],den))

def check_score(moment,w,s):
    p=np.clip(np.array([(1-moment(q))/2 for q in w['targets']]),0,1);y=np.array([dot(q,w['hidden']) for q in w['targets']]);wt=np.array(w['weights']);saved=np.array(s['prob']);delta=float(np.max(abs(p-saved)));assert delta<2e-8,delta
    pred=saved>.5+1e-12;ok=pred==y;take=np.maximum(saved,1-saved)>=.9-1e-12
    for k,v in dict(brier=wt@(saved-y)**2,risk=wt@(saved*(1-saved)),accuracy=wt@ok,old_correct=sum(ok[:8]),new_correct=sum(ok[8:]),coverage=wt@take,covered_correct=wt@(take&ok)).items():assert abs(s[k]-v)<2e-12,(k,s[k],v)
    return delta


def run14():
    from pilot14 import candidates, frames, temper
    from bp import bp
    receipt=json.loads((H/'receipt.json').read_text());assert receipt['source_sha256']==sha(H/'pilot14.py') and receipt['base_sha256']==sha(H/'base13.py') and receipt['bp_sha256']==sha(H/'bp.py') and receipt['protocol_sha256']==sha(H/'PROTOCOL.md')
    assert len(receipt['worlds'])==36 and len({w['seed'] for w in receipt['worlds']})==36
    counts=Counter();maxerr=0.;rows_out=[];mass_out=[]
    for w in receipt['worlds']:
        assert w['split']==('cal' if 300<=w['seed']-(130000+100*w['r'])<=303 else 'dev')
        y=np.array([dot(q,w['hidden']) for q in w['targets']]);wt=np.array(w['weights']);assert not set(w['targets'])&(set(w['initial'])|set(w['pool']))
        for x in w['runs']:
            t=x['trace'];p=x['noise'];rs=w['initial'].copy();obs=t['initial_obs'].copy();assert obs==[dot(a,w['hidden'])^int(u<p) for a,u in zip(rs,w['uniforms'])];spent=0;seen=set()
            for j,step in enumerate(t['trace']):
                i=step['j'];assert i not in seen and step['query']==w['pool'][i] and step['cost']==w['costs'][i] and step['obs']==dot(step['query'],w['hidden'])^int(w['uniforms'][len(w['initial'])+j]<p)
                assert step['expands']==(Basis(rs).coord(step['query']) is None);seen.add(i);spent+=step['cost'];rs.append(step['query']);obs.append(step['obs']);counts['trace_steps']+=1
            assert spent==t['spent']<=24 and not any(spent+w['costs'][i]<=24 for i in range(48) if i not in seen)
            R=len(Basis(rs).rows);assert R==x['rank'];canon=Basis(x['frames'][0]);assert canon.rows==x['frames'][0];assert all(canon.coord(q)<1<<R for q in rs)
            fs,_=frames(w,rs,obs);assert [f.rows for f in fs]==x['frames'];mus=[];replayed=[]
            for f in x['frames']:
                f=Basis(f);assert len(f.rows)==w['d'];mu,_=edge_bp([f.coord(q) for q in rs],obs,p,w['d']);m=np.array([0. if f.coord(q)>=1<<R else math.prod(mu[i] for i in range(R) if f.coord(q)>>i&1) for q in w['targets']]);replayed.append(m);counts['independent_frame_moments']+=32
            maxerr=max(maxerr,float(np.max(np.abs(np.array(replayed)-x['frame_moments']))));assert np.allclose(replayed,x['frame_moments'],atol=2e-10,rtol=0)
            # Same-source replay of candidate generation only; likelihood weights independently computed below.
            for f in fs:mus.append(bp([f.coord(q) for q in rs],obs,p,w['d'])[0])
            for n,u,random,name in [(1,8,False,'repair1'),(4,6,False,'repair4'),(8,8,False,'repair8'),(8,8,True,'random_repair8')]:
                a=x['decoders'][name];raw=[candidates(f,fs[0],mu,R,u,random,w['seed']+i+17000) for i,(f,mu) in enumerate(zip(fs[:n],mus[:n]))];assert sorted(set(int(v) for z in raw for v in z))==a['state_ids'];assert a['proposals']==sum(len(z) for z in raw);counts['candidate_replays']+=1
            u=Fraction(str(p));m=len(rs);weights=[u.numerator**i*(u.denominator-u.numerator)**(m-i) for i in range(m+1)];row_masks=[canon.coord(q) for q in rs];target_masks=[canon.coord(q) for q in w['targets']];all_states=set()
            for a in x['decoders'].values():all_states.update(a.get('state_ids',[]))
            if 'exact' in x['decoders']:all_states.update(range(1<<R))
            # Independently evaluate exact integer likelihood for every unique state in the union.
            ew={h:weights[sum(((h&q).bit_count()&1)!=b for q,b in zip(row_masks,obs))] for h in all_states};counts['unique_state_likelihoods']+=len(ew)
            full=sum(ew.values()) if 'exact' in x['decoders'] else None
            for name,a in x['decoders'].items():
                saved=np.array(a['prob']);assert len(saved)==32 and np.all((saved>=0)&(saved<=1));assert all(abs(saved[j]-.5)<1e-12 for j,q in enumerate(target_masks) if q>=1<<R)
                if name.startswith('repair') and 'state_ids' in a or name in ('random_repair8','exact'):
                    states=a.get('state_ids',list(range(1<<R)) if name=='exact' else []);assert len(states)==a['states']==len(set(states));den=sum(ew[h] for h in states);assert den>0
                    prob=np.array([.5 if q>=1<<R else float(Fraction(sum(ew[h] for h in states if (h&q).bit_count()&1),den)) for q in target_masks]);delta=float(np.max(abs(prob-saved)));maxerr=max(maxerr,delta);assert delta<2e-12;counts['integer_posterior_probabilities']+=32
                    if full is not None:mass_out.append(dict(seed=w['seed'],split=w['split'],noise=p,arm=name,mass=float(Fraction(den,full)),max_probability_error=float(np.max(abs(saved-np.array(x['decoders']['exact']['prob']))))))
                if name in ('observed','ensemble4','ensemble8'):
                    n={'observed':1,'ensemble4':4,'ensemble8':8}[name];assert np.allclose(saved,(1-np.mean(replayed[:n],axis=0))/2,atol=2e-10,rtol=0);counts['ensemble_probabilities']+=32
                if name=='fixed_induced':
                    f,_=bp_prob(w,rs,obs,p,'induced');assert np.allclose(saved,[(1-f(q))/2 for q in w['targets']],atol=2e-10,rtol=0);counts['independent_baseline_probabilities']+=32
                if name in ('repair_guard','repair_calibrated'):
                    pr=temper(np.array(x['decoders']['repair8']['prob']),receipt['temperatures'][str(p)]['temperature']);assert np.allclose(pr,saved,atol=2e-12,rtol=0)
                ok=(saved>.5+1e-12)==y;take=np.maximum(saved,1-saved)>=.9-1e-12
                if name=='repair_guard':
                    ms=np.array(x['frame_moments']);take&=(np.all(ms>=.4-1e-12,axis=0)&(saved<.5))|(np.all(ms<=-.4+1e-12,axis=0)&(saved>.5))
                for k,v in dict(brier=wt@(saved-y)**2,accuracy=wt@ok,coverage=wt@take,covered_correct=wt@(take&ok)).items():assert abs(a[k]-v)<2e-12
                counts['score_checks']+=32;rows_out.append(dict(seed=w['seed'],split=w['split'],d=w['d'],noise=p,arm=name,cost=spent,**{k:v for k,v in a.items() if k not in ('prob','state_ids')}))
            counts['conditions']+=1
        print('AUDIT',w['seed'],flush=True)
    for p in (.15,.35):
        losses=[]
        for temp in (1.,2.,4.,8.):
            vals=[]
            for w in receipt['worlds']:
                if w['split']!='cal':continue
                x=next(t for t in w['runs'] if t['noise']==p);y=np.array([dot(q,w['hidden']) for q in w['targets']]);pr=temper(np.array(x['decoders']['repair8']['prob']),temp);vals.append(np.array(w['weights'])@(pr-y)**2)
            losses.append(float(np.mean(vals)))
        assert np.allclose(losses,receipt['temperatures'][str(p)]['cal_losses'],atol=1e-12,rtol=0);assert receipt['temperatures'][str(p)]['temperature']==(1.,2.,4.,8.)[int(np.argmin(losses))]
    result=dict(status='PASS',**counts,max_probability_or_moment_error=maxerr,receipt_sha256=sha(H/'receipt.json'),source_sha256=sha(__file__),scope='Independent edge-list BP and integer likelihood; same-source candidate generation and temperature replay; no independent acquisition-choice replay.',retained_mass=mass_out,rows=rows_out)
    (H/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ('rows','retained_mass')})
if __name__=='__main__':run14()
