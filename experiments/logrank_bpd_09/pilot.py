"""Noisy component calibration and compositional transfer; development only."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOTS={16:(2309,2311,2333),32:(2401,2411,2417)}
NOISE=(0.,.15,.35)
STEPS=200
MAX=5
W=32

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bits(ids,k): return (2*((np.asarray(ids,dtype=np.uint64)[:,None]>>np.arange(k,dtype=np.uint64))&1).astype(np.int8)-1).astype(np.float32)
def truth(x,ps,ts,sc,ss):
    out=[]
    for p in ps:
        a=np.ones(len(x))
        for t,o,s in p:
            i=ts[t]; v=x[:,sc[i]]*ss[i]*s; a=a*v if o==0 else np.minimum(a,v)
        out.append(a)
    return np.stack(out,1).astype(np.float32)
def world(k,root):
    g=np.random.default_rng(root); ts=g.permutation(k); sc=g.permutation(k); ss=g.choice([-1,1],k)
    tr=[((t,0,1),) for t in range(k)]; used=set(tr)
    def add(n,lo,hi):
        out=[]
        while len(out)<n:
            d=int(g.integers(lo,hi+1)); tok=g.choice(k,d,replace=False)
            p=tuple((int(t),0 if i==0 else int(g.integers(2)),int(g.choice([-1,1]))) for i,t in enumerate(tok))
            if p not in used: used.add(p); out.append(p)
        return out
    tr+=add(64-k,1,3); same=add(32,1,3); deep=add(32,4,5); ps=tr+same+deep
    ids=list(dict.fromkeys(g.integers(0,1<<k,size=2048,dtype=np.uint64).tolist()))[:512]; assert len(ids)==512
    x,z=bits(ids[:256],k),bits(ids[256:],k)
    return dict(k=k,root=root,token_slot=ts.tolist(),slot_coord=sc.tolist(),slot_sign=ss.tolist(),programs=ps,addresses=g.permutation(128).tolist(),train_ids=ids[:256],test_ids=ids[256:]),x,z,truth(x,tr,ts,sc,ss),truth(z,same+deep,ts,sc,ss)
def observations(w,x,rate):
    k=w['k']; g=np.random.default_rng(w['root']+99173); y=x[:,w['slot_coord']]*np.array(w['slot_sign']); flips=g.random(y.shape)<rate
    noisy=np.where(flips,-y,y).astype(np.float32); tok=np.array([t for p in w['programs'][:64] for t,_,_ in p]); actual=np.array(w['token_slot'])[tok]; change=g.random(len(tok))<rate; bad=(actual+g.integers(1,k,len(tok)))%k
    return noisy,tok,np.where(change,bad,actual),dict(literal_flips=int(flips.sum()),literal_count=flips.size,route_flips=int(change.sum()),route_count=len(change))
def descriptor(ps,k):
    t=np.full((len(ps),MAX+1),k,dtype=np.int64); o=np.zeros_like(t); s=np.ones_like(t,dtype=np.float32)
    for j,p in enumerate(ps):
        for i,(a,b,c) in enumerate(p): t[j,i]=a;o[j,i]=b;s[j,i]=c
    return tuple(torch.as_tensor(v) for v in (t,o,s))
def onehot(z):
    p=z.softmax(-1);h=F.one_hot(z.argmax(-1),z.shape[-1]).float();return h+(p-p.detach())
def hard(z):
    p=z.tanh();return torch.where(z>=0,1.,-1.)+(p-p.detach())
def gate(z):
    p=z.sigmoid();return (z>=0).float()+(p-p.detach())
class Model(nn.Module):
    def __init__(self,k):
        super().__init__();self.k=k;self.emb=nn.Embedding(k+1,W);self.route=nn.Linear(W,k);self.go=nn.Linear(W,1);self.V=nn.Parameter(torch.linalg.qr(torch.randn(k,k))[0])
    def heads(self,t):
        e=self.emb(t);return self.route(e),self.go(e).squeeze(-1)
    def forward(self,x,t,op,sign):
        r,s=self.heads(t);mask=onehot(r);live=gate(s);lit=hard((mask*(x@self.V)[:,None,:]).sum(-1))*sign;a=torch.ones(len(x));run=torch.ones_like(a)
        for i in range(t.shape[1]):
            run=run*live[:,i];v=lit[:,i];candidate=torch.where(op[:,i]==0,a*v,torch.minimum(a,v));a=a+run*(candidate-a)
        return a

def train(w,x,yt,noisy,tokens,route,arm):
    k=w['k'];torch.manual_seed(w['root']+100);m=Model(k);opt=torch.optim.Adam(m.parameters(),lr=.02);g=torch.Generator().manual_seed(w['root']+200)
    x=torch.tensor(x);y=torch.tensor(yt);noisy=torch.tensor(noisy);tokens=torch.tensor(tokens);route=torch.tensor(route);t,o,s=descriptor(w['programs'][:64],k);ids=torch.arange(k+1);labels=(ids<k).float();start=time.monotonic()
    for step in range(STEPS):
        ix=torch.randint(len(x),(128,),generator=g);j=torch.randint(64,(128,),generator=g);ri=torch.randint(len(tokens),(128,),generator=g);r,_=m.heads(tokens[ri]);_,sl=m.heads(ids)
        aux=F.cross_entropy(r,route[ri])+F.binary_cross_entropy_with_logits(x[ix]@m.V,(noisy[ix]+1)/2)+F.binary_cross_entropy_with_logits(sl,labels)
        loss=aux if arm=='staged' and step<100 else aux+((m(x[ix],t[j],o[j],s[j])-y[ix,j])**2).mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
    with torch.no_grad():
        r,s=m.heads(ids);return m.V.detach().numpy().copy(),r.argmax(-1).numpy(),(s>=0).numpy(),m,time.monotonic()-start

def compile_teacher(w,x,noisy,tokens,route):
    k=w['k'];score=x.T@noisy;idx=abs(score).argmax(0);sg=np.where(score[idx,np.arange(k)]>=0,1,-1);v=np.zeros((k,k),dtype=np.float32);v[idx,np.arange(k)]=sg;counts=np.zeros((k,k),dtype=np.int64);np.add.at(counts,(tokens,route),1);return v,np.r_[counts.argmax(1),0],np.arange(k+1)<k

def replay(w,z,v,route,go):
    k=w['k'];a=z.astype(np.float64)@v.astype(np.float64);out=[];pr=[]
    for p in w['programs'][64:]:
        acc=np.ones(len(z));used=[]
        for t,o,s in list(p)+[(k,0,1)]*(MAX+1-len(p)):
            if not go[t]:break
            i=int(route[t]);v0=np.where(a[:,i]>=0,1,-1)*s;acc=acc*v0 if o==0 else np.minimum(acc,v0);used.append((i,o,s))
        out.append(acc);pr.append(used)
    return np.stack(out,1),pr

def snapped(v):
    k=len(v);i=abs(v).argmax(0);s=np.where(v[i,np.arange(k)]>=0,1,-1);p=np.zeros_like(v);p[i,np.arange(k)]=s;return p

def record(w,z,y,v,r,g,m,sec,arm,rate,noise):
    p,pr=replay(w,z,v,r,g);q,_=replay(w,z,snapped(v),r,g);mis=0
    if m is not None:
        t,o,s=descriptor(w['programs'][64:],w['k']);out=[]
        with torch.no_grad():
            for x in torch.tensor(z).split(32):
                n=len(x);a=m(x.repeat_interleave(64,0),t.repeat(n,1),o.repeat(n,1),s.repeat(n,1));assert ((a==1)|(a==-1)).all();out.append(a.reshape(n,64).numpy())
        mis=int((np.concatenate(out)!=p).sum());assert mis==0
    return dict(arm=arm,rate=rate,noise=noise,seconds=sec,V=v.tolist(),route=r.tolist(),go=g.tolist(),executed=pr,correct_same=int((p[:,:32]==y[:,:32]).sum()),correct_deep=int((p[:,32:]==y[:,32:]).sum()),snap_same=int((q[:,:32]==y[:,:32]).sum()),snap_deep=int((q[:,32:]==y[:,32:]).sum()),snap_disagreements=int((q!=p).sum()),predictions=''.join('1' if b else '0' for b in (p>0).flat),snap_predictions=''.join('1' if b else '0' for b in (q>0).flat),replay_mismatches=mis,parameters=sum(p.numel() for p in m.parameters()) if m is not None else None)
def run(k,out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);result=dict(k=k,steps=STEPS,source_sha256=sha(__file__),roots=[]);out=Path(out)
    for root in ROOTS[k]:
        w,x,z,yt,yz=world(k,root);w['labels']=''.join('1' if b else '0' for b in (yz>0).flat);w['arms']=[]
        for rate in NOISE:
            noisy,tok,route,noise=observations(w,x,rate)
            for arm in ('staged','joint','compiler'):
                if arm=='compiler':
                    start=time.monotonic();v,r,g=compile_teacher(w,x,noisy,tok,route);m=None;sec=time.monotonic()-start
                else:v,r,g,m,sec=train(w,x,yt,noisy,tok,route,arm)
                row=record(w,z,yz,v,r,g,m,sec,arm,rate,noise);w['arms'].append(row);print(k,root,rate,arm,row['correct_same'],row['correct_deep'],row['snap_same'],row['snap_deep'],flush=True)
        result['roots'].append(w);out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--k',type=int,choices=ROOTS,required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.k,a.out)
