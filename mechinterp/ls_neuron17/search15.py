"""LS15: paired conditional minimax fitting, numerical proposals, exact terminal gate."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import itertools, math, json, hashlib, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.special import ndtr
from scipy.optimize import linprog
import torch
import repair13 as parent
from audit import verify_scalar
from construction import scalar_weights
from parent_train import NAMES
SEEDS=tuple(range(469,493)); ARMS=('mean_ls','max_ls','max_lp','bound_lp'); ROUNDS=12; TARGET=3/32
CASES=np.array(list(itertools.product(range(4),(0,1),(0,1),(0,1))))
X=np.column_stack(((CASES[:,0,None]-2+np.linspace(-.125,.125,17)).ravel(),np.repeat(CASES[:,1:],17,axis=0))).astype(float)
Y=np.repeat((CASES[:,:3].sum(1)%4)-2,17); H=X[:,0]; R=H-Y; D=1/64

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def curvature(z):return abs(2-z*z)*np.exp(-z*z/2)/math.sqrt(2*math.pi)
def pads(pre,coeff):
    z=pre.reshape(32,17,8);lo=np.minimum(z[:,:-1],z[:,1:]);hi=np.maximum(z[:,:-1],z[:,1:]);m=np.maximum(curvature(lo),curvature(hi))
    for v in (-2.,0.,2.):m=np.maximum(m,np.where((lo<=v)&(v<=hi),curvature(v),0))
    p=(D*D/40)*np.sum(m*coeff,axis=-1);return np.maximum(np.pad(p,((0,0),(1,0))),np.pad(p,((0,0),(0,1)))).ravel()
def metric(w):
    u=np.asarray(w['up'],float);v=np.asarray(w['down'],float);z=X@u.T+np.asarray(w['up_bias']);e=R+X@np.asarray(w['skip'])+w['skip_bias'][0]+.2*((z*ndtr(z))@v+w['down_bias'][0]);pad=pads(z,abs(v)*u[:,0]**2)
    return {'mean':float(parent.score(e)),'max':float(abs(e).max()),'bound':float((abs(e)+pad).max())}
def state(c,s,o,db):return {'codes':c.tolist(),'scales':s.tolist(),'offset':float(o),'scores':metric(parent.decoded(c,s,o,db))}
def lp_matrix(c,s,o,db,with_pad):
    sk,u,b,v=parent.unpack(c);w=parent.decoded(c,s,o,db);u=np.asarray(w['up'],float);z=X@u.T+np.asarray(w['up_bias']);A=X@sk;B=.2*((z*ndtr(z))@v);k=pads(z,abs(v)*u[:,0]**2) if with_pad else np.zeros_like(R)
    M=np.vstack([np.column_stack((sign*A,sign*B+k,np.full_like(R,sign),-np.ones_like(R))) for sign in (1.,-1.)]);return M,np.r_[-R,R]
def lp_fit(c,s,o,db,with_pad):
    M,rhs=lp_matrix(c,s,o,db,with_pad);obj=np.array([0.,0.,0.,1.]);bounds=[(parent.LO,parent.HI),(parent.LO,parent.HI),(-8.,8.),(0.,None)]
    res=linprog(obj,A_ub=M,b_ub=rhs,bounds=bounds,method='highs-ds',options={'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8,'maxiter':1000})
    info={'status':int(res.status),'nit':int(res.nit),'success':bool(res.success)}
    if not res.success:return s.copy(),o,info
    x=res.x;ss=s.copy();ss[[0,3]]=x[:2];y=res.ineqlin.marginals;lo=res.lower.marginals;hi=res.upper.marginals;dual=rhs@y+parent.LO*(lo[0]+lo[1])-8*lo[2]+parent.HI*(hi[0]+hi[1])+8*hi[2]
    info.update(x=x.tolist(),objective=float(res.fun),primal_violation=float(max(0.,(M@x-rhs).max(),parent.LO-x[0],parent.LO-x[1],x[0]-parent.HI,x[1]-parent.HI,abs(x[2])-8,-x[3])),dual_stationarity=float(abs(obj-M.T@y-lo-hi).max()),dual_gap=float(res.fun-dual))
    return ss,float(x[2]),info
def init(seed):
    c,s,o,db=parent.setup(seed,15);return {'seed':seed,'codes':c.tolist(),'scales':s.tolist(),'offset':o,'down_bias':db,'weights':parent.decoded(c,s,o,db)}
def candidates(c):
    cs,names=parent.proposals(c,True);assert len(cs)<=177;pad=177-len(cs);return np.concatenate((cs,np.repeat(c[None],pad,axis=0))),names+['padding']*pad
def shortlist(cs,rank):
    selected=[0];seen={tuple(cs[0])}
    for j in np.argsort(rank,kind='stable'):
        if tuple(cs[j]) not in seen:seen.add(tuple(cs[j]));selected.append(int(j))
        if len(selected)==3:break
    assert len(selected)==3;return selected
def search(initial,arm):
    start=time.monotonic();c=np.array(initial['codes']);si=np.array(initial['scales']);s=si.copy();o=initial['offset'];db=initial['down_bias'];key=arm.split('_')[0];c0=c.copy()
    s,o,fit=parent.fit(c,s,o);selected=state(c,s,o,db);ledger=[{'round':0,'fit':fit,'state':selected}];nfev=fit['nfev'];calls=1;secondary=0;slots=0
    for r in range(1,ROUNDS+1):
        cs,names=candidates(c);best=selected;move='incumbent';anchors=[]
        for anchor,ss in [('current',s),('initial',si)]:
            rank,coef,offs=parent.profile(cs,ss);slots+=len(cs);fits=[]
            for j in shortlist(cs,rank):
                z=ss.copy();z[[0,3]]=coef[j];q=cs[j];ns,no,rec=parent.fit(q,z,float(offs[j]));calls+=1;nfev+=rec['nfev'];first=state(q,ns,no,db)
                if arm.endswith('_lp'):ts,to,extra=lp_fit(q,ns,no,db,key=='bound')
                else:ts,to,extra=parent.fit(q,ns,no);nfev+=extra['nfev']
                secondary+=1;second=state(q,ts,to,db);winner=second if second['scores'][key]<first['scores'][key]-1e-12 else first
                fits.append({'slot':j,'move':names[j],'fit':rec,'first':first,'secondary':extra,'second':second,'selected_second':winner is second})
                if winner['scores'][key]<best['scores'][key]-1e-12:best=winner;move=f'{names[j]}@{anchor}'
            anchors.append({'anchor':anchor,'profile':rank.tolist(),'fits':fits})
        selected=best;c=np.array(best['codes']);s=np.array(best['scales']);o=best['offset'];ledger.append({'round':r,'state':best,'move':move,'anchors':anchors})
    assert (calls,secondary,slots)==(73,72,4248)
    return {'seed':initial['seed'],'arm':arm,'initial_sha256':hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest(),'final':selected|{'weights':parent.decoded(c,s,o,db)},'hamming':int(np.count_nonzero(c!=c0)),'first_calls':calls,'secondary_calls':secondary,'slots':slots,'ls_nfev':nfev,'ledger':ledger,'seconds':time.monotonic()-start}
def worker(job):
    seed,arm=job;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);initial=json.loads(Path(f'initial/{seed}.json').read_text());r=search(initial,arm);Path(f'results/{arm}-{seed}.json').write_text(json.dumps(r,separators=(',',':'))+'\n');print(seed,arm,r['final']['scores'],flush=True)
def main():
    for line in Path('lock.sha256').read_text().splitlines():d,p=line.split(maxsplit=1);assert sha(p)==d
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);Path('initial').mkdir();Path('results').mkdir()
    for seed in SEEDS:Path(f'initial/{seed}.json').write_text(json.dumps(init(seed),sort_keys=True)+'\n')
    with ProcessPoolExecutor(max_workers=4) as ex:list(ex.map(worker,[(s,a) for s in SEEDS for a in ARMS],chunksize=1))
if __name__=='__main__':main()
