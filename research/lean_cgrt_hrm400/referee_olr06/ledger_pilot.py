"""Referee pilot for the proposed OLR-06 gate (synthetic, no Lean).

Dependent witness ledger over GF(2). Each active obligation i has a hidden
valid branch w_i in {-1,+1}. The verifier-owned ledger holds, per obligation:
  parent pointer p(i) and an opaque edge type t_i in {-1,+1}   (structure)
  optional RELATIVE receipt  rel_i = w_i * w_p(i) * t_i          (dependent guard)
  optional ABSOLUTE receipt  abs_i = w_i                         (anchor, as in 05)
Target per obligation: w_i if it is determined by the receipts, else UNKNOWN.
Exact solution = parity propagation over the forest in BOTH directions.
"""
import sys, random, time, collections
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(1)
N=10; FEAT=7

def sample(rng, kmin, kmax, maxdepth, chain, p_rel=.85, p_abs=.25):
    while True:
        k=rng.randint(kmin,kmax); act=rng.sample(range(N),k)
        parent={}; depth={}
        for j,v in enumerate(act):
            if j==0 or rng.random()<(.1 if chain else .25): parent[v]=-1; depth[v]=0
            else:
                p=act[j-1] if (chain and rng.random()<.85) else rng.choice(act[:j])
                parent[v]=p; depth[v]=depth[p]+1
        if max(depth.values())<=maxdepth and (not chain or max(depth.values())>=min(5,k-2)): break
    w={v:rng.choice((-1,1)) for v in act}; t={v:rng.choice((-1,1)) for v in act}
    has_rel={v:(parent[v]>=0 and rng.random()<p_rel) for v in act}
    has_abs={v:rng.random()<p_abs for v in act}
    # exact solver: components of known relative edges; solved iff component has an anchor
    adj=collections.defaultdict(list)
    for v in act:
        if has_rel[v]: adj[v].append(parent[v]); adj[parent[v]].append(v)
    solved={}; dist={}
    for a in act:
        if has_abs[a] and a not in solved:
            solved[a]=True; dist[a]=0
    frontier=[a for a in act if has_abs[a]]
    while frontier:
        nxt=[]
        for u in frontier:
            for x in adj[u]:
                if x not in dist: dist[x]=dist[u]+1; nxt.append(x)
        frontier=nxt
    # top-down-only solvable: anchored ancestor-or-self with every relative edge known on the path
    td={}
    def topdown(v):
        if v in td: return td[v]
        r=has_abs[v] or (parent[v]>=0 and has_rel[v] and topdown(parent[v]))
        td[v]=r; return r
    for v in act: topdown(v)
    x=[[0.]*FEAT for _ in range(N)]; par=[0]*N; y=[3]*N; cat=[3]*N; hops=[0]*N
    for v in act:
        x[v]=[1., float(parent[v]>=0), float(t[v]) if parent[v]>=0 else 0.,
              float(has_rel[v]), float(w[v]*w[parent[v]]*t[v]) if has_rel[v] else 0.,
              float(has_abs[v]), float(w[v]) if has_abs[v] else 0.]
        par[v]=parent[v] if parent[v]>=0 else v
        if v in dist: y[v]=(w[v]+1)//2; cat[v]=0 if td[v] else 1; hops[v]=dist[v]
        else: y[v]=2; cat[v]=2
    return x,par,y,cat,hops

def bank(seed,n,**kw):
    rng=random.Random(seed); X=[];P=[];Y=[];C=[];H=[]
    for _ in range(n):
        x,p,y,c,h=sample(rng,**kw); X.append(x);P.append(p);Y.append(y);C.append(c);H.append(h)
    return (torch.tensor(X),torch.tensor(P),torch.tensor(Y),torch.tensor(C),torch.tensor(H))

class Structured(nn.Module):
    """Exact ledger rows -> T tied refinement steps over ledger incidence.
    mode: bi (parent->child and child->parent), td (parent->child only),
    bu (child->parent only), none (per-row reader = key-value table + MLP)."""
    def __init__(s,d=48,mode="bi"):
        super().__init__(); s.d,s.mode=d,mode
        s.enc=nn.Sequential(nn.Linear(FEAT,d),nn.GELU(),nn.Linear(d,d))
        s.down=nn.Linear(d,d,bias=False); s.up=nn.Linear(d,d,bias=False)
        s.cell=nn.Sequential(nn.Linear(4*d,2*d),nn.GELU(),nn.Linear(2*d,d)); s.norm=nn.LayerNorm(d)
        s.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,3))
    def forward(s,x,par,T,every=False):
        e=s.enc(x); h=e; B=x.shape[0]; outs=[]
        haspar=x[...,1:2]; idx=par.unsqueeze(-1).expand(-1,-1,s.d)
        for _ in range(T if s.mode!="none" else 1):
            md=torch.gather(s.down(h),1,idx)*haspar if s.mode in("bi","td") else torch.zeros_like(h)
            if s.mode in("bi","bu"):
                mu=torch.zeros_like(h).scatter_add(1,idx,s.up(h)*haspar)
            else: mu=torch.zeros_like(h)
            h=s.norm(h+s.cell(torch.cat((h,md,mu,e),-1)))
            if every: outs.append(s.head(h))
        return outs if every else s.head(h)

class Dense(nn.Module):
    """Ordinary transformer over the same ledger rows; the parent pointer and
    own address are given as one-hot input features (equal information)."""
    def __init__(s,d=48,tied=True,depth=8):
        super().__init__(); s.tied=tied
        s.enc=nn.Sequential(nn.Linear(FEAT+2*N,d),nn.GELU(),nn.Linear(d,d))
        mk=lambda: nn.ModuleDict(dict(attn=nn.MultiheadAttention(d,4,batch_first=True),
            ff=nn.Sequential(nn.Linear(2*d,2*d),nn.GELU(),nn.Linear(2*d,d)),n1=nn.LayerNorm(d),n2=nn.LayerNorm(d)))
        s.blocks=nn.ModuleList([mk()] if tied else [mk() for _ in range(depth)])
        s.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,3))
    def forward(s,x,par,T):
        eye=torch.eye(N).expand(x.shape[0],-1,-1)
        e=s.enc(torch.cat((x,eye,F.one_hot(par,N).float()*x[...,1:2]),-1)); h=e
        pad=x[...,0]<.5
        for k in range(T if s.tied else len(s.blocks)):
            b=s.blocks[0 if s.tied else k]
            a,_=b["attn"](h,h,h,key_padding_mask=pad,need_weights=False)
            h=b["n1"](h+a); h=b["n2"](h+b["ff"](torch.cat((h,e),-1)))
        return s.head(h)

def make(arm):
    if arm in("bi","td","bu","none"): return Structured(mode=arm)
    if arm=="bi_ds": return Structured(mode="bi")
    if arm=="dense_tied": return Dense(tied=True)
    if arm=="dense_untied": return Dense(tied=False)
    raise ValueError(arm)

def score(model,data,T):
    x,p,y,c,h=data
    with torch.no_grad(): z=model(x,p,T).argmax(-1)
    act=x[...,0]>.5; ok=(z==y)&act
    out={"all":float(ok.sum()/act.sum())}
    for name,k in (("topdown",0),("bottomup_only",1),("unknown",2)):
        m=act&(c==k); out[name]=float((ok&m).sum()/m.sum())
    m=act&(c<2)&(h>=5); out["hops>=5"]=float((ok&m).sum()/m.sum()) if m.any() else float("nan")
    return out

if __name__=="__main__":
    arms=sys.argv[1].split(","); seeds=[int(s) for s in sys.argv[2].split(",")]; steps=int(sys.argv[3])
    train=bank(1,20000,kmin=3,kmax=6,maxdepth=3,chain=False)
    dev=bank(2,3000,kmin=3,kmax=6,maxdepth=3,chain=False)
    deep=bank(3,3000,kmin=9,kmax=10,maxdepth=9,chain=True)
    for name,d in (("dev",dev),("deep",deep)):
        x,p,y,c,h=d; act=x[...,0]>.5; n=float(act.sum())
        f=[float((act&(c==k)).sum())/n for k in range(3)]
        print(f"{name}: slots {int(n)} topdown {f[0]:.3f} bottomup_only {f[1]:.3f} unknown {f[2]:.3f} max_hops {int(h.max())} | ceilings: exact 1.000, top-down-only {f[0]+.5*f[1]+f[2]:.3f}",flush=True)
    for arm in arms:
        for seed in seeds:
            torch.manual_seed(seed); m=make(arm); t0=time.time()
            opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=.005)
            g=torch.Generator().manual_seed(seed)
            for j in range(steps):
                ind=torch.randint(len(train[0]),(128,),generator=g)
                act=train[0][ind][...,0]>.5
                if arm=="bi_ds":
                    T=int(torch.randint(6,17,(1,),generator=g))
                    zs=m(train[0][ind],train[1][ind],T,every=True)[4:]
                    loss=sum(F.cross_entropy(z[act],train[2][ind][act]) for z in zs)/len(zs)
                else:
                    T=int(torch.randint(5,9,(1,),generator=g))
                    z=m(train[0][ind],train[1][ind],T)
                    loss=F.cross_entropy(z[act],train[2][ind][act])
                opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
            m.eval(); n=sum(p.numel() for p in m.parameters())
            r={f"dev@T8":score(m,dev,8),"deep@T8":score(m,deep,8),"deep@T16":score(m,deep,16),"deep@T32":score(m,deep,32)}
            print(f"{arm:13s} seed {seed} params {n} loss {float(loss):.3f} ({time.time()-t0:.0f}s)",flush=True)
            for k,v in r.items(): print("    ",k,{a:round(b,3) for a,b in v.items()},flush=True)
