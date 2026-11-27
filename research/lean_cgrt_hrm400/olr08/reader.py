"""Small CPU term/action graph residual ranker. Input is a lossy feature graph,
never used for proof identity. Native successful paths alone supply rank labels.
"""
import hashlib, math, random, json
from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F

ARMS=('bi','one_way','bi1','local','ff3','linear','sham')
SEEDS=(17,29,43)
W=40; B=128; VOCAB=2048; MAX_NODES=16000; MAX_EDGES=64000

def hid(s,n):return int.from_bytes(hashlib.blake2b(str(s).encode(),digest_size=8).digest(),'little')%n

def packet(graph, lemmas, actions, depth, prior):
    """Full available graph, no silent truncation. Oversize => explicit prior fallback."""
    nodes=list(graph['nodes']); edges=[tuple(e) for e in graph['edges']]; gcount=len(nodes)
    obs=graph['obligations']; targets=[o['target'] for o in obs]
    if not targets: raise ValueError('open obligations required')
    arg_roots=defaultdict(list); lemroot={}; lem_nodes={}
    for i,n in enumerate(nodes):
        if n.get('k') in ('fvar','const') and 'n' in n:arg_roots[n['n']].append(i)
    for lem in lemmas:
        name=lem['name']; start=len(nodes); nodes.extend(lem['nodes']);
        edges.extend((a+start,b+start,r) for a,b,r in lem['edges']);lemroot[name]=start+lem['root'];lem_nodes[name]=lem['nodes']
    an=[]; extra=[]; kinds=defaultdict(int)
    def bag(ns):
        x=torch.zeros(B)
        for n in ns:
            for k,v in sorted(n.items()):x[hid(k+'='+str(v),B)]+=1
        return x.log1p()/max(1,len(ns))**.5
    gb=bag(nodes[:gcount]); eb=bag([{'edge':r,'child':nodes[a].get('k'),'parent':nodes[b].get('k')} for a,b,r in edges if a<gcount and b<gcount])
    for j,(c,k,arg,tac) in enumerate(actions):
        ix=len(nodes);nodes.append({'k':'action','n':k,'arg':arg,'tactic':tac});an.append(ix)
        for target in targets:edges.append((target,ix,'obligation'))
        for a in arg_roots.get(arg,[]):edges.append((a,ix,'argument'))
        if arg in lemroot:edges.append((lemroot[arg],ix,'lemma'))
        rank=kinds[k];kinds[k]+=1
        sb=bag([nodes[a] for a in arg_roots.get(arg,[])]+lem_nodes.get(arg,[])+[{'kind':k,'arg':arg,'tactic':tac}])
        extra.append(torch.cat((gb,eb,sb,torch.tensor([depth/3,min(rank,100)/100,c/10,prior[j]/10,len(targets)/8,len(lem_nodes.get(arg,[]))/256]))))
    if len(nodes)>MAX_NODES or len(edges)>MAX_EDGES:return None
    if any(a<0 or b<0 or a>=len(nodes) or b>=len(nodes) for a,b,_ in edges):raise ValueError('invalid native graph edge')
    ids=torch.tensor([[hid('kind:'+str(n.get('k')),VOCAB),hid('name:'+str(n.get('n',n.get('arg',''))),VOCAB),hid(json.dumps(n,sort_keys=True),VOCAB)] for n in nodes],dtype=torch.long)
    e=torch.tensor([[a,b,hid(r,64)] for a,b,r in edges],dtype=torch.long).reshape(-1,3)
    return ids,e,torch.tensor(an),torch.stack(extra),torch.tensor(prior,dtype=torch.float32)

class Reader(nn.Module):
    def __init__(self,arm):
        super().__init__(); self.arm=arm
        if arm=='linear':self.direct=nn.Linear(3*B+6,1);return
        self.emb=nn.Embedding(VOCAB,W);self.rel=nn.Embedding(64,W)
        self.up=nn.Linear(W,W);self.down=nn.Linear(W,W)
        self.cell=nn.Sequential(nn.Linear(4*W,2*W),nn.GELU(),nn.Linear(2*W,W))
        self.norm=nn.LayerNorm(W);self.extra=nn.Linear(3*B+6,W)
        self.head=nn.Sequential(nn.Linear(2*W,W),nn.GELU(),nn.Linear(W,1))
    def forward(self,p):
        ids,e,act,x,prior=p
        if self.arm=='linear':return self.direct(x).squeeze(-1)
        initial=self.emb(ids).mean(1);h=initial
        c,b,r=e.T;rel=self.rel(r)
        # Child -> parent and parent -> child. Same operations for one-way;
        # the parent-to-child contribution alone is masked.
        dc=torch.bincount(c,minlength=len(h)).clamp_min(1).unsqueeze(1);db=torch.bincount(b,minlength=len(h)).clamp_min(1).unsqueeze(1)
        visits=0 if self.arm=='local' else (1 if self.arm=='bi1' else 3)
        for _ in range(visits):
            up=torch.zeros_like(h).index_add(0,b,self.up(h[c])+rel)/db
            down=torch.zeros_like(h).index_add(0,c,self.down(h[b])+rel)/dc
            up=up*float(self.arm!='ff3')
            down=down*float(self.arm not in ('one_way','ff3'))
            h=self.norm(h+self.cell(torch.cat((h,initial,up,down),1)))
        return self.head(torch.cat((h[act],F.gelu(self.extra(x))),1)).squeeze(-1)

@torch.no_grad()
def costs(model,p,alpha=1.):
    base=p[-1]; logits=-math.log(2)*base+alpha*model(p)
    return (-F.log_softmax(logits,dim=0)/math.log(2)).tolist()

def fit(frames,arm,seed,steps=400):
    torch.manual_seed(seed);m=Reader(arm);opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.005)
    rng=random.Random(seed+8081);sampler=[rng.randrange(len(frames)) for _ in range(steps)]
    # Shuffled reference action is fixed per frame and still belongs to its menu.
    labels=[random.Random(seed*1000003+i).randrange(len(p[-1])) if arm=='sham' else y for i,(p,y) in enumerate(frames)]
    initial=hash_weights(m);losses=[]
    for i in sampler:
        p,y=frames[i]; logits=-math.log(2)*p[-1]+m(p)
        loss=F.cross_entropy(logits[None],torch.tensor([labels[i]]))
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
    m.eval();return m,{'arm':arm,'seed':seed,'steps':steps,'parameters':sum(p.numel() for p in m.parameters()),'initial_sha256':initial,'sampler_sha256':hashlib.sha256(json.dumps(sampler).encode()).hexdigest(),'weights_sha256':hash_weights(m),'loss_first':losses[0],'loss_last':losses[-1]}

def hash_weights(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
