"""LS-NEURON-01 exploratory graph propagation pilot; PyTorch CPU."""
import argparse,json,math,hashlib
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
T,N,D,W=4,9,16,64
DOSES=[0,.5,.75,.9,.95,.99]
SEEDS=[11,23,37]

def worlds(seed,count):
    g=torch.Generator().manual_seed(seed); pred,mark,pairs=[],[],[]
    while len(pred)<count:
        p=torch.randint(N,(N,),generator=g);src=int(torch.randint(N,(),generator=g));reach=torch.arange(N)==src
        for _ in range(T): reach=reach|reach[p]
        pos=torch.where(reach)[0];pos=pos[pos!=src];neg=torch.where(~reach)[0]
        if not len(pos) or not len(neg): continue
        pred.append(p);mark.append(torch.arange(N)==src)
        pairs.append(torch.stack([pos[torch.randint(len(pos),(),generator=g)],neg[torch.randint(len(neg),(),generator=g)]]))
    return torch.stack(pred),torch.stack(mark).float(),torch.stack(pairs)

class Model(nn.Module):
    def __init__(self,banks,width):
        super().__init__();self.banks,self.width=banks,width
        self.emb=nn.Linear(1,D);self.norm=nn.LayerNorm(D);self.qkv=nn.Linear(D,3*D);self.proj=nn.Linear(D,D);self.mn=nn.LayerNorm(D)
        self.up=nn.ModuleList(nn.Linear(D,width) for _ in range(banks));self.down=nn.ModuleList(nn.Linear(width,D) for _ in range(banks));self.read=nn.Linear(D,1)
    def forward(self,p,s,idx,masks=None,steps=T,trace=False):
        h=self.emb(s.unsqueeze(-1));acts=[]
        for t in range(steps):
            q,k,v=self.qkv(self.norm(h)).chunk(3,-1);ix=p.unsqueeze(-1).expand_as(k);kp,vp=k.gather(1,ix),v.gather(1,ix)
            a=torch.sigmoid((q*(kp-k)).sum(-1,keepdim=True)/math.sqrt(D));h=h+self.proj(a*vp+(1-a)*v)
            b=t%self.banks;z=F.gelu(self.up[b](self.mn(h)))
            if trace: acts.append(z)
            if masks is not None: z=z*masks[t]
            h=h+self.down[b](z)
        return self.read(h).squeeze(-1).gather(1,idx),acts

def logits(model,data,masks=None,steps=T): return model(*data,masks=masks,steps=steps)[0]
def outcomes(z): return ((z[:,0]>0).int()+(z[:,1]<=0).int()).tolist()
def ranks(model,data):
    z,acts=model(*data,trace=True);y=torch.tensor([1.,0.]).expand_as(z);grads=torch.autograd.grad(F.binary_cross_entropy_with_logits(z,y),acts)
    return torch.stack([(a*g).abs().sum((0,1)) for a,g in zip(acts,grads)])

def masks_for(sal,banks,k,kind,seed,steps=T):
    width=sal.shape[1];g=torch.Generator().manual_seed(seed)
    if kind=='shared': order=sal.reshape(T//banks,banks,width).sum(0).argsort(descending=True)
    elif kind=='local': order=sal.argsort(descending=True)
    elif kind=='random': order=torch.stack([torch.randperm(width,generator=g) for _ in range(banks)])
    else: raise ValueError(kind)
    m=torch.zeros_like(order,dtype=torch.float).scatter_(1,order[:,:k],1.)
    return [m[t%len(m)] for t in range(steps)]

def run(seed,arm,updates):
    torch.manual_seed(seed);banks,width={'tied':(1,W),'untied':(T,W),'budget':(T,W//T)}[arm];model=Model(banks,width);opt=torch.optim.AdamW(model.parameters(),lr=.003)
    train=worlds(1000+seed,4096);cal=worlds(2000+seed,256);test=worlds(3000+seed,512);g=torch.Generator().manual_seed(4000+seed)
    for _ in range(updates):
        ix=torch.randint(len(train[0]),(64,),generator=g);batch=tuple(x[ix] for x in train);z=logits(model,batch)
        loss=F.binary_cross_entropy_with_logits(z,torch.tensor([1.,0.]).expand_as(z));opt.zero_grad();loss.backward();opt.step()
    model.eval();sal=ranks(model,cal).detach();rows=[]
    with torch.no_grad():
        for d in DOSES:
            k=max(1,round(width*(1-d)))
            for kind in ('shared','local','random'):
                for rep in range(4 if kind=='random' else 1):
                    m=masks_for(sal,banks,k,kind,5000+seed+rep);z=logits(model,test,m)
                    rows.append(dict(dose=d,keep=k,kind=kind,rep=rep,steps=T,root_outcomes=outcomes(z)))
        if banks==1:
            for steps in (1,2,8):
                for d in (0,.9,.95):
                    k=max(1,round(width*(1-d)));m=masks_for(sal,banks,k,'shared',0,steps)
                    rows.append(dict(dose=d,keep=k,kind='shared',rep=0,steps=steps,root_outcomes=outcomes(logits(model,test,m,steps))))
        stability=[]
        if banks==1:
            dense=logits(model,cal);dense=dense[:,0]-dense[:,1]
            for j in sal.sum(0).topk(8).indices.tolist():
                effects=[]
                for t in range(T):
                    m=torch.ones(T,width);m[t,j]=0;z=logits(model,cal,m);effects.append(dense-(z[:,0]-z[:,1]))
                cos=[float(F.cosine_similarity(effects[t],effects[t+1],dim=0)) if min(float(effects[t].norm()),float(effects[t+1].norm()))>1e-8 else None for t in range(T-1)]
                stability.append(dict(neuron=j,adjacent_cosines=cos))
    return dict(seed=seed,arm=arm,banks=banks,width=width,parameters=sum(p.numel() for p in model.parameters()),rows=rows,calibration_stability=stability)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--updates',type=int,default=240);ap.add_argument('--out',default='results.json');a=ap.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    result=dict(experiment='LS-NEURON-01',status='development_only',torch=torch.__version__,updates=a.updates,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),train_roots=4096,cal_roots=256,dev_roots=512,runs=[])
    for seed in SEEDS:
        for arm in ('tied','untied','budget'):
            r=run(seed,arm,a.updates);result['runs'].append(r);Path(a.out).write_text(json.dumps(result,separators=(',',':'))+'\n')
            print(seed,arm,r['parameters'],[(x['dose'],round(sum(x['root_outcomes'])/1024,4)) for x in r['rows'] if x['kind']=='shared' and x['steps']==T],flush=True)
if __name__=='__main__':main()
