"""Adaptive follow-up on exposed 10 roots: target-aware posterior acquisition."""
import hashlib,json
from pathlib import Path
import numpy as np
from pilot import dot,span,posterior,score,probs


def walsh(p):
    a=p.copy(); h=1
    while h<len(a):
        b=a.reshape(-1,2*h);l=b[:,:h].copy();r=b[:,h:].copy();b[:,:h]=l+r;b[:,h:]=l-r;h*=2
    return a

def value(mu,test,pool,noise):
    q=np.array(test);a=np.array(pool);rho=1-2*noise
    cov=mu[q[:,None]^a]-mu[q,None]*mu[a];den=1-(rho*mu[a])**2
    return np.divide(rho*rho*(cov**2).mean(0),4*den,out=np.zeros(len(a)),where=den>1e-12)

def run():
    receipt=json.loads(Path('receipt.json').read_text());rows=[];tables={}
    for w in receipt['rows']:
        rank=w['r'];n=1<<rank;noise=w['noise'];h=w['h'];masks=span(w['basis']);test=w['test'];y=np.array(w['test_y']);c=list(w['coeff']);o=list(w['obs']);pool=list(w['pool']);trace=[]
        if rank not in tables:
            b=np.array([i.bit_count()%2 for i in range(n)],dtype=np.uint8);tables[rank]=b[np.arange(n)[:,None]&np.arange(n)]
        t=tables[rank];p=posterior(t,c,o,noise)
        for old in w['active']['entropy']:
            mu=walsh(p);v=value(mu,test,pool,noise);j=int(v.argmax());a=pool.pop(j);flip=old['obs']^dot(masks[old['query']],h);b=dot(masks[a],h)^flip
            # Check expected decrease directly from both hypothetical outcomes.
            py=probs(t,test,p);pa=float(probs(t,[a],p)[0]);pobs=noise+(1-2*noise)*pa;gain=0.
            for obs,weight in ((1,pobs),(0,1-pobs)):
                if weight>1e-12:
                    pp=p*np.where(t[a]==obs,1-noise,noise)/weight;new=probs(t,test,pp);gain+=weight*float(np.mean((new-py)**2))
            assert abs(gain-v[j])<2e-12
            c.append(a);o.append(b);p=posterior(t,c,o,noise);trace.append(dict(query=a,obs=b,expected_brier_reduction=float(v[j]),conditional_update_check=gain,score=score(probs(t,test,p),y)))
        rows.append(dict(seed=w['seed'],k=w['k'],r=rank,noise=noise,trace=trace))
    return dict(evidence='adaptive target-aware probe diagnostic; exposed roots; public test masks used, no test answers in selection',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=hashlib.sha256(Path('receipt.json').read_bytes()).hexdigest(),rows=rows)
if __name__=='__main__':Path('acquisition.json').write_text(json.dumps(run(),indent=2)+'\n')
