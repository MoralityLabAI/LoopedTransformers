"""Adaptive stronger compiler controls for LOGRANK-BPD-09; exposed roots."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from audit import actual, equiv, execute, to_cols
from pilot import bits, observations, truth

def run(paths):
    rows=[]
    for path in paths:
        for w in json.load(open(path))['roots']:
            k=w['k']; x=bits(w['train_ids'],k); p=w['programs']; labels=truth(x,p[:64],w['token_slot'],w['slot_coord'],w['slot_sign']); tr=to_cols(w['test_ids'],k); gt=[actual(w,t) for t in p[64:]]
            def score(rate,arm,coord,sign,route):
                pred=[[(int(coord[route[t]]),o,int(s*sign[route[t]])) for t,o,s in q] for q in p[64:]]; cert=[equiv(a,b) for a,b in zip(pred,gt)]; correct=[256-(execute(a,tr,256)^execute(b,tr,256)).bit_count() for a,b in zip(pred,gt)]
                rows.append(dict(k=k,root=w['root'],rate=rate,arm=arm,correct_same=sum(correct[:32]),correct_deep=sum(correct[32:]),certified=sum(cert),certified_deep=sum(cert[32:]),coord=coord.tolist(),sign=sign.tolist(),route=route.tolist()))
            # Uses the same clean singleton task labels provided to the neural arms.
            assert all(q==[[t,0,1]] for t,q in enumerate(p[:k])); votes=x.T@labels[:,:k]; idx=abs(votes).argmax(0); sign=np.where(votes[idx,np.arange(k)]>=0,1,-1)
            assert np.all(abs(votes[idx,np.arange(k)])==len(x)); assert np.all((abs(votes)==len(x)).sum(0)==1)
            score(None,'clean_singleton_compiler',idx,sign,np.arange(k))
            for rate in (0.,.15,.35):
                noisy,tok,lab,_=observations(w,x,rate); votes=x.T@noisy; i,j=linear_sum_assignment(-abs(votes)); coord=np.empty(k,dtype=int);coord[j]=i; sign=np.where(votes[coord,np.arange(k)]>=0,1,-1); counts=np.zeros((k,k),dtype=int);np.add.at(counts,(tok,lab),1); i,j=linear_sum_assignment(-counts);route=np.empty(k,dtype=int);route[i]=j;score(rate,'bijective_teacher_compiler',coord,sign,route)
    return dict(evidence='adaptive exposed-root diagnostic; assumes known signed permutation and known singleton descriptor semantics',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows)
if __name__=='__main__':print(json.dumps(run(sys.argv[1:]),indent=2))
