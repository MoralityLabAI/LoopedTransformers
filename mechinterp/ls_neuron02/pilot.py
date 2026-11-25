"""LS-NEURON-02, development-only Boolean recurrent neuron pruning."""
import argparse,hashlib,json,math,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
D,W,T=16,64,4
SEEDS=(11,23,37,43,53,67)
ARMS=('tied','untied','budget','full_budget')
COUNTS=(0,1,2,3,4,6,8,12,16,24,32,48,64)
def sha(b):return hashlib.sha256(b).hexdigest()
def split(seed,steps=T):
    g=torch.Generator().manual_seed(10000+seed+100*steps)
    ctx=torch.randperm(2**steps,generator=g)
    focal=torch.arange(4**steps)
    if steps>T:focal=focal[torch.randperm(len(focal),generator=g)[:512]]
    a=((focal[:,None]>>torch.arange(0,2*steps,2))&1).float();b=((focal[:,None]>>torch.arange(1,2*steps,2))&1).float()
    y=((a.long()+b.long()).cumsum(1)%2).float()
    r={}
    for name,cs in [('train',ctx[:2**(steps-1)]),('cal',ctx[2**(steps-1):3*2**(steps-2)]),('dev',ctx[3*2**(steps-2):])]:
        if steps!=T and name!='dev':continue
        if steps>T:cs=cs[:4]
        r[name]=(torch.cat([torch.stack((a,b,((c>>torch.arange(steps))&1).float().expand_as(a)),-1) for c in cs]),y.repeat(len(cs),1),cs.tolist())
    return r
class Model(nn.Module):
    def __init__(self,arm,mode):
        super().__init__();self.mode=mode;self.banks=1 if arm=='tied' else T;self.full=T if arm=='full_budget' else 1
        self.width={'tied':W,'untied':W,'budget':W//T,'full_budget':8 if mode=='affine' else 1}[arm]
        self.init=nn.Parameter(torch.zeros(D));self.read=nn.Linear(D,1)
        if mode=='affine':
            self.skip=nn.ModuleList(nn.Linear(D+3,D) for _ in range(self.full));n=D+3
        else:
            self.emb=nn.Linear(3,D);self.role=nn.Parameter(torch.randn(2,D)*.02)
            self.norm=nn.ModuleList(nn.LayerNorm(D) for _ in range(self.full));self.qkv=nn.ModuleList(nn.Linear(D,3*D,bias=False) for _ in range(self.full));self.proj=nn.ModuleList(nn.Linear(D,D,bias=False) for _ in range(self.full));self.mn=nn.ModuleList(nn.LayerNorm(D) for _ in range(self.full));n=D
        self.up=nn.ModuleList(nn.Linear(n,self.width) for _ in range(self.banks));self.down=nn.ModuleList(nn.Linear(self.width,D) for _ in range(self.banks))
    def forward(self,x,masks=None,remove=False,trace=False):
        h=self.init.expand(len(x),-1);zs=[];acts=[]
        for t in range(x.shape[1]):
            k=t%self.full;b=t%self.banks
            if self.mode=='affine':q=torch.cat((h,x[:,t]),-1);base=h+self.skip[k](q)
            else:
                tokens=torch.stack((self.emb(x[:,t])+self.role[0],h+self.role[1]),1);n=self.norm[k](tokens);q,k0,v=self.qkv[k](n).reshape(len(x),2,3,4,D//4).unbind(2)
                scores=(q[:,1,None]*k0).sum(-1)/math.sqrt(D//4);att=self.proj[k]((scores.softmax(1).unsqueeze(-1)*v).sum(1).reshape(len(x),D));q=self.mn[k](n[:,1]+att);base=tokens[:,1]+.2*att
            a=F.gelu(self.up[b](q));acts.append(a)
            if masks is not None:a=a*masks[t]
            h=base if remove else base+.2*self.down[b](a);zs.append(self.read(h).squeeze(-1))
        return torch.stack(zs,1),acts if trace else None
def masks(sal,banks,k,kind,seed,steps=T):
    g=torch.Generator().manual_seed(seed);w=sal.shape[1]
    if kind=='global':order=sal.reshape(T//banks,banks,w).sum(0).argsort(descending=True)
    elif kind=='local':order=sal.argsort(descending=True)
    elif kind=='random':order=torch.stack([torch.randperm(w,generator=g) for _ in range(banks)])
    else:raise ValueError(kind)
    m=torch.zeros_like(order,dtype=torch.float).scatter_(1,order[:,:k],1.)
    return [m[t%len(m)] for t in range(steps)]
def evaluate(model,data,mask=None,remove=False):
    x,y,ctx=data
    with torch.no_grad():z=torch.cat([model(x[i:i+256],mask,remove)[0] for i in range(0,len(x),256)])
    ok=z.gt(0)==y.bool();return {'acc':float(ok[:,-1].float().mean()),'prefix_acc':ok.float().mean(0).tolist(),'correct':ok[:,-1].int().tolist(),'contexts':ctx}
def run(seed,arm,mode,updates,folder):
    torch.manual_seed(seed);model=Model(arm,mode);opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0.);d=split(seed);x,y,_=d['train'];g=torch.Generator().manual_seed(20000+seed);sampler=hashlib.sha256()
    for _ in range(updates):
        ids=torch.randint(len(x),(128,),generator=g);sampler.update(ids.numpy().tobytes());z,_=model(x[ids]);loss=F.binary_cross_entropy_with_logits(z,y[ids]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    model.eval();cp=folder/f'{mode}-{arm}-{seed}.pt';torch.save(model.state_dict(),cp)
    cx,cy,_=d['cal'];z,a=model(cx,trace=True);grad=torch.autograd.grad(F.binary_cross_entropy_with_logits(z,cy),a);sal=torch.stack([(v*g).abs().sum(0) for v,g in zip(a,grad)]).detach();rows=[]
    counts=sorted(set(COUNTS+(model.width, max(1,round(.1*model.width)),max(1,round(.05*model.width)))))
    for kind in ('global','local','random'):
        for k in counts:
            if k>model.width:continue
            for rep in range(3 if kind=='random' and k not in (0,model.width) else 1):
                m=masks(sal,model.banks,k,kind,30000+seed+rep)
                unique=int(torch.stack(m).amax(0).sum()) if kind=='local' and model.banks==1 else k*model.banks
                rows.append({'kind':kind,'keep':k,'unique':unique,'rep':rep,**evaluate(model,d['dev'],m)})
    dense=evaluate(model,d['dev']);removed=evaluate(model,d['dev'],remove=True);extrap=[]
    if arm=='tied':
        for steps in (2,6):
            data=split(seed,steps)['dev']
            for k in (1,3,6,model.width):extrap.append({'steps':steps,'keep':k,**evaluate(model,data,masks(sal,model.banks,k,'global',0,steps))})
    return {'seed':seed,'arm':arm,'mode':mode,'width':model.width,'banks':model.banks,'parameters':sum(p.numel() for p in model.parameters()),'sampler_sha256':sampler.hexdigest(),'checkpoint_sha256':sha(cp.read_bytes()),'train_contexts':d['train'][2],'cal_contexts':d['cal'][2],'dense':dense,'remove_mlp':removed,'rows':rows,'extrap':extrap,'last_loss':float(loss.detach())}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('affine','attention'),default='affine');ap.add_argument('--updates',type=int,default=600);ap.add_argument('--seeds',type=int,default=3);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);a.out.mkdir(parents=True,exist_ok=True)
    r={'experiment':'LS-NEURON-02','status':'development_only','source_sha256':sha(Path(__file__).read_bytes()),'torch':torch.__version__,'updates':a.updates,'runs':[]}
    for seed in SEEDS[:a.seeds]:
        for arm in ARMS:
            s=time.monotonic();x=run(seed,arm,a.mode,a.updates,a.out);r['runs'].append(x);(a.out/'results.json').write_text(json.dumps(r,separators=(',',':'))+'\n');print(seed,arm,x['parameters'],round(x['dense']['acc'],4),round(x['remove_mlp']['acc'],4),[(q['keep'],round(q['acc'],4)) for q in x['rows'] if q['kind']=='global'],round(time.monotonic()-s,1),flush=True)
if __name__=='__main__':main()
