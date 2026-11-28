"""Independent exact integer-likelihood and GF(2) witness audit."""
import hashlib,json,math
from fractions import Fraction
from pathlib import Path


def parity(a,b): return (a&b).bit_count()%2

def solve(rows,rhs,k):
    eq=[[a,b] for a,b in zip(rows,rhs)]; piv=[]; pos=0
    for i in range(k):
        p=next((j for j in range(pos,len(eq)) if eq[j][0]>>i&1),None)
        if p is None: continue
        eq[p],eq[pos]=eq[pos],eq[p]
        for j in range(len(eq)):
            if j!=pos and eq[j][0]>>i&1: eq[j][0]^=eq[pos][0]; eq[j][1]^=eq[pos][1]
        piv.append(i);pos+=1
    if any(not a and b for a,b in eq): return None
    return sum(eq[j][1]<<i for j,i in enumerate(piv))

def qmask(q,n): return sum(parity(q,b)<<b for b in range(n))

def lift(q,basis):
    a=0
    for i,b in enumerate(basis):
        if q>>i&1:a^=b
    return a

def integer_post(c,o,rate,n):
    x=sum(b<<i for i,b in enumerate(o)); groups={}
    for b in range(n):
        e=(sum(parity(q,b)<<i for i,q in enumerate(c))^x).bit_count();groups[e]=groups.get(e,0)|(1<<b)
    p=Fraction(str(rate)); u=p.numerator;v=p.denominator-u
    weights={e:u**e*v**(len(c)-e) for e in groups};den=sum(weights[e]*g.bit_count() for e,g in groups.items());assert den>0
    def prob(mask):return Fraction(sum(weights[e]*(g&mask).bit_count() for e,g in groups.items()),den)
    return prob

def audit(path):
    r=json.loads(Path(path).read_text()); assert r['source_sha256']==hashlib.sha256(Path(__file__).with_name('pilot.py').read_bytes()).hexdigest(); worlds={}; decisions=0; probs_checked=0; entropy_checks=0; outcomes=[]
    for a in r['rows']:
        k,rank,seed,noise=a['k'],a['r'],a['seed'],a['noise']; basis=a['basis'];n=1<<rank; masks=[lift(i,basis) for i in range(n)]; assert len(set(masks))==n and all(x.bit_count()%2==0 for x in masks);c=a['coeff'];test=a['test'];pool=a['pool'];outside=a['outside'];h=a['h'];obs=a['obs'];y=a['test_y'];yo=a['outside_y']; assert c[:rank]==[1<<i for i in range(rank)] and len(c)==4*rank and all(0<i<n for i in c);assert len(set(test))==32 and len(set(pool))==48 and not(set(c)&(set(test)|set(pool))) and not set(test)&set(pool);assert len(set(outside))==32 and not set(outside)&set(masks) and all(x.bit_count()%2==0 and x for x in outside); assert y==[parity(masks[q],h) for q in test] and yo==[parity(q,h) for q in outside];assert obs==[parity(masks[q],h)^f for q,f in zip(c,a['flips'])]
        if seed not in worlds:
            witnesses=[]
            for q in outside:
                g=solve(basis+[q],[0]*rank+[1],k); assert g is not None and all(parity(x,g)==0 for x in basis) and parity(q,g)==1; witnesses.append(g)
            worlds[seed]=dict(k=k,r=rank,ambiguity_witnesses=witnesses)
        b=sum(obs[i]<<i for i in range(rank));assert a['basis_correct']==sum(parity(q,b)==v for q,v in zip(test,y)); bh=a['hard_h'];assert a['hard_correct']==sum(parity(masks[q],bh)==v for q,v in zip(test,y));assert a['hard_outside_correct']==sum(parity(q,bh)==v for q,v in zip(outside,yo));assert min(a['hard_train_errors'])==sum(parity(masks[q],bh)!=v for q,v in zip(c,obs))
        cache={q:qmask(q,n) for q in test+pool};
        def check(p,s):
            nonlocal decisions,probs_checked
            pr=[p(cache[q]) for q in test]; assert all(abs(float(x)-z)<2e-12 for x,z in zip(pr,s['prob'])); pred=s['pred'];assert all((x==int(z>Fraction(1,2))) or abs(z-Fraction(1,2))<Fraction(1,10**12) for x,z in zip(pred,pr));take=[max(x,1-x)>=Fraction(9,10) for x in pr];assert s['correct']==sum(x==z for x,z in zip(pred,y));assert s['covered']==sum(take);assert s['covered_correct']==sum(t and x==z for t,x,z in zip(take,pred,y));assert abs(s['brier']-sum(float((p-v)**2) for p,v in zip(pr,y))/len(y))<2e-12; decisions+=len(y);probs_checked+=len(pr)
        post=integer_post(c,obs,noise,n);check(post,a['posterior']);
        for method,tr in a['active'].items():
            remain=set(pool); cc=list(c);oo=list(obs);p=post
            for row in tr:
                q=row['query'];assert q in remain
                if method=='entropy':
                    best=min(abs(p(cache[t])-Fraction(1,2)) for t in remain); assert float(abs(p(cache[q])-Fraction(1,2))-best)<2e-12;entropy_checks+=1
                remain.remove(q);cc.append(q);oo.append(row['obs']);p=integer_post(cc,oo,noise,n);check(p,row['score'])
        for i,(e,z) in enumerate(zip(a['active']['entropy'],a['active']['random'])): assert (e['obs']^parity(masks[e['query']],h))==(z['obs']^parity(masks[z['query']],h))
        outcomes.append(dict(seed=seed,k=k,r=rank,noise=noise,basis=a['basis_correct'],hard=a['hard_correct'],hard_outside=a['hard_outside_correct'],bayes=a['posterior']['correct'],covered=a['posterior']['covered'],covered_correct=a['posterior']['covered_correct'],entropy=a['active']['entropy'][-1]['score']['correct'],random=a['active']['random'][-1]['score']['correct']))
    return dict(status='PASS',worlds=len(worlds),conditions=len(outcomes),exact_posterior_probabilities=probs_checked,entropy_policy_steps=entropy_checks,ambiguity_witnesses=sum(len(x['ambiguity_witnesses']) for x in worlds.values()),receipt_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),world_receipts=worlds,rows=outcomes,scope='exact rational posterior, query separation, ambiguity witnesses, saved hard sign predictions; no optimizer replay')
if __name__=='__main__':print(json.dumps(audit('receipt.json'),indent=2))
