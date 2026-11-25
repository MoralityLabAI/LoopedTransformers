"""Discrete/continuous repair of an eight-GELU mod4 cell."""
import argparse, itertools, json, math, time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.optimize import least_squares
import torch
import experiment as old
from parent_train import NAMES
SEEDS=tuple(range(433,445)); DOSES=(5,15); ARMS=('continuous','single','paired'); ROUNDS=16
CASES=np.array(list(itertools.product(range(4),(0,1),(0,1),(0,1))))
GRID=np.linspace(-.125,.125,9)
X=np.column_stack(((CASES[:,0,None]-2+GRID).ravel(),np.repeat(CASES[:,1:],len(GRID),axis=0))).astype(float)
Y=np.repeat((CASES[:,:3].sum(1)%4)-2,len(GRID)).astype(float)
H=X[:,0]; N=len(Y); TARGET=3/32; LO=1e-6; HI=128.

def setup(seed,dose):
    p,_,s=old.init(seed,dose); ps=[v[None] for v in p]; ls=[torch.full((1,)+(1,)*(ps[i].ndim-1),math.log(v),dtype=torch.float32) for i,v in zip(old.QIDS,s)]
    q,cs,ss=old.decode(ps,ls)
    codes=np.concatenate([v[0].numpy().astype(np.int64).ravel() for v in cs]); scales=np.array([float(v.flatten()[0]) for v in ss]); db=float(q[5][0,0]);offset=float(q[1][0,0])+.2*db
    return codes,scales,offset,db

def unpack(c):return c[...,0:4],c[...,4:36].reshape(c.shape[:-1]+(8,4)),c[...,36:44],c[...,44:52]
def score(e):return np.mean(e*e+10*np.maximum(np.abs(e)-TARGET,0)**2,axis=-1)

def fit(c,s,o):
    sk,u,b,v=unpack(c); skp=X@sk; up=X@u.T
    def residual(z,jac=False):
        ss=np.exp(z[:4]);z0=ss[1]*up+ss[2]*b; phi=ndtr(z0);gel=z0*phi;dg=phi+z0*np.exp(-.5*z0*z0)/math.sqrt(2*math.pi);down=gel@v
        e=H+ss[0]*skp+.2*ss[3]*down+z[4]-Y; excess=np.maximum(np.abs(e)-TARGET,0); w=np.sqrt(10.)
        if not jac:return np.r_[e,w*excess]
        J=np.column_stack((ss[0]*skp,.2*ss[3]*((dg*ss[1]*up)@v),.2*ss[3]*((dg*ss[2]*b)@v),.2*ss[3]*down,np.ones(N)))
        return np.vstack((J,w*(np.sign(e)*(excess>0))[:,None]*J))
    lb=np.r_[[math.log(LO)]*4,-8];ub=np.r_[[math.log(HI)]*4,8]; z=np.clip(np.r_[np.log(s),o],lb,ub)
    r=least_squares(residual,z,jac=lambda x:residual(x,True),bounds=(lb,ub),max_nfev=40,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    return np.exp(r.x[:4]),float(r.x[4]),dict(nfev=r.nfev,njev=r.njev,status=int(r.status),loss=float(np.sum(r.fun*r.fun)/N))

def proposals(c,paired):
    out=[c.copy()]; names=['noop']
    def add(q,name):
        if np.all(np.abs(q)<=7) and not np.array_equal(q,c):out.append(q);names.append(name)
    for i in range(len(c)):
        for d in (-1,1):
            q=c.copy();q[i]+=d;add(q,f'{i}:{d:+}')
    if paired:
        for i in range(0,8,2):
            a,b=4+4*i,4+4*(i+1)
            for source,target in ((a,b),(b,a)):
                q=c.copy();q[target:target+4]=c[source:source+4];add(q,f'copy_up:{source}:{target}')
            for d in (-1,1):
                for j in range(4):
                    q=c.copy();q[a+j]+=d;q[b+j]+=d;add(q,f'joint_up:{i}:{j}:{d:+}')
                q=c.copy();q[36+i]+=d;q[36+i+1]+=d;add(q,f'joint_bias:{i}:{d:+}')
                q=c.copy();q[44+i]+=d;q[44+i+1]-=d;add(q,f'opp_down:{i}:{d:+}')
            for i0,i1,sgn in ((i,i+1,1),(i+1,i,-1)):
                q=c.copy();q[36+i1]=c[36+i0]-2*sgn;add(q,f'gap2:{i0}:{i1}')
                q=c.copy();q[44+i1]=-c[44+i0];add(q,f'opp:{i0}:{i1}')
    keys=set();unique=[];labels=[]
    for q,label in zip(out,names):
        key=q.tobytes()
        if key not in keys:keys.add(key);unique.append(q);labels.append(label)
    return np.stack(unique),labels

def profile(cs,s):
    sk,u,b,v=unpack(cs);f0=sk@X.T;z=s[1]*np.einsum('kij,nj->kni',u,X)+s[2]*b[:,None,:];f1=.2*np.einsum('kni,ki->kn',z*ndtr(z),v)
    f=np.stack((f0,f1),-1);mu=f.mean(1);fc=f-mu[:,None,:];t=Y-H;tm=t.mean();tc=t-tm
    G=np.einsum('kni,knj->kij',fc,fc);g=np.einsum('kni,n->ki',fc,tc);K=len(cs)
    cand=[np.linalg.solve(G+1e-12*np.eye(2)[None],g[...,None]).squeeze(-1).clip(LO,HI)]
    for i in (0,1):
        j=1-i
        for v0 in (LO,HI):
            z=np.empty((K,2));z[:,i]=v0;z[:,j]=((g[:,j]-G[:,j,i]*v0)/np.maximum(G[:,j,j],1e-12)).clip(LO,HI);cand.append(z)
    c=np.stack(cand,1);o=(tm-np.einsum('kci,ki->kc',c,mu)).clip(-8,8);e=np.einsum('kni,kci->kcn',f,c)+o[:,:,None]+H-Y
    loss=score(e);idx=loss.argmin(1);ii=np.arange(K);return loss[ii,idx],c[ii,idx],o[ii,idx]

def decoded(c,s,o,db):
    sk,u,b,v=unpack(c);vals=[sk,np.array([o-.2*db]),u,b,v,np.array([db])];sc=[s[0],1,s[1],s[2],s[3],1]
    return {k:(np.asarray(v,dtype=np.float32)*np.float32(a)).tolist() for k,v,a in zip(NAMES,vals,sc)}

def measure(w,seed):
    q=[torch.tensor(w[k],dtype=torch.float32)[None] for k in NAMES];h,x,y=old.centers(1)
    return dict(eval=old.evaluate(q,seed),float_bound=float(old.enclosure(q,h,x,y).max()))

def run(seed,dose,arm,out):
    start=time.monotonic();c,s,o,db=setup(seed,dose);initial_c=c.copy();init_s=s.copy();init_o=o;initial=decoded(c,s,o,db);ledger=[];s,o,rec=fit(c,s,o);fits=rec['nfev'];count=0;ledger.append(dict(round=0,move='fit',fit=rec,scales=s.tolist(),offset=o,codes=c.tolist()))
    for r in range(1,ROUNDS+1):
        move='noop';n=0;profile_loss=None
        if arm!='continuous':
            cand,names=proposals(c,arm=='paired');n=len(cand);count+=n;l,coef,offs=profile(cand,s);j=int(np.argmin(l));profile_loss=float(l[j]);
            if j!=0 and l[j]<rec['loss']-1e-12:
                c=cand[j].copy();s=s.copy();s[[0,3]]=coef[j];o=float(offs[j]);move=names[j]
        s,o,rec=fit(c,s,o);fits+=rec['nfev'];ledger.append(dict(round=r,move=move,candidates=n,profile_loss=profile_loss,fit=rec,codes=c.tolist(),scales=s.tolist(),offset=o))
    final=decoded(c,s,o,db);row=dict(seed=seed,dose=dose,arm=arm,initial=dict(codes=initial_c.tolist(),scales=init_s.tolist(),offset=init_o,weights=initial,**measure(initial,seed)),final=dict(codes=c.tolist(),scales=s.tolist(),offset=o,weights=final,**measure(final,seed)),hamming=int(np.count_nonzero(c!=initial_c)),code_l1=int(abs(c-initial_c).sum()),search_evaluations=count,fit_nfev=fits,ledger=ledger,seconds=time.monotonic()-start)
    out.mkdir(parents=True,exist_ok=True);(out/f'{dose}-{arm}-{seed}.json').write_text(json.dumps(row,separators=(',',':'))+'\n');print(seed,dose,arm,'bound',round(row['final']['float_bound'],5),'moves',row['hamming'],'secs',round(row['seconds'],1),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True,choices=SEEDS);ap.add_argument('--dose',type=int,required=True,choices=DOSES);ap.add_argument('--arm',required=True,choices=ARMS);ap.add_argument('--out',type=Path,default=Path('repairs'));a=ap.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);run(a.seed,a.dose,a.arm,a.out)
