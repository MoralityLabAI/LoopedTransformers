"""Small local predictor + explicit constrained residency admission."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from expert_lattice import ExpertLattice

LATTICE = ExpertLattice()


class Mesh(nn.Module):
    def __init__(self,width=48,communicate=True):
        super().__init__()
        self.communicate=communicate
        self.encoder=nn.Linear(256,width)
        self.layer=nn.Embedding(24,width)
        self.readout=nn.Sequential(nn.Linear(width*(3 if communicate else 1),width),
                                   nn.GELU(),nn.Linear(width,128))

    def forward(self,x):
        h=torch.tanh(self.encoder(x)+self.layer.weight)
        if self.communicate:
            b=h.reshape(-1,4,6,h.shape[-1]).mean(2)
            mesh=b.mean(1,keepdim=True).expand(-1,24,-1)
            h=torch.cat((h,b.repeat_interleave(6,1),mesh),dim=-1)
        return self.readout(h)


def admit(scores,old,k,b):
    """Exact maximizer of sum scores with k slots, <= b new IDs.
    Capacity/provenance is hard; learned scores are soft.
    """
    if not 8 <= k <= LATTICE.experts:
        raise ValueError("invalid admission state")
    return LATTICE.admit(scores, old, k, b)


def evaluate(data,scores,idx,k,b):
    last={}
    roots=defaultdict(lambda:[0.,0,0])
    order=sorted(idx,key=lambda i:(str(data["root"][i]),str(data["turn"][i]),int(data["start"][i])))
    for i in order:
        root,turn=str(data["root"][i]),str(data["turn"][i])
        for l in range(24):
            key=root,turn,l
            old=last.get(key)
            selected=admit(scores[i,l],old,k,b)
            n=int(data["visit_count"][i,l])
            roots[root][0]+=float(data["y"][i,l,list(sorted(selected))].sum())*n
            roots[root][1]+=n
            roots[root][2]+=len(selected-(old or frozenset()))
            last[key]=selected
    per={r:{"retained_mass":m/n,"visits":n,"admissions":a} for r,(m,n,a) in roots.items()}
    return {"root_mean_mass":float(np.mean([s["retained_mass"] for s in per.values()])),
            "admissions":sum(s["admissions"] for s in per.values()),"n_roots":len(per),"roots":per}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--resident",type=int,default=8)
    p.add_argument("--max-swaps",type=int,default=2)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--batch",type=int,default=32)
    a=p.parse_args()
    torch.manual_seed(a.seed)
    data=dict(np.load(a.data,allow_pickle=False))
    train=np.flatnonzero(data["split"]=="train")
    dev=np.flatnonzero(data["split"]=="dev")
    if not len(train) or not len(dev):
        raise ValueError("need train and dev")
    if set(data["root"][train])&set(data["root"][dev]) or set(data["family"][train])&set(data["family"][dev]):
        raise ValueError("split leakage")
    if data["x"].shape[1:]!=(24,256) or data["y"].shape[1:]!=(24,128):
        raise ValueError("invalid data shape")
    a.out.mkdir(parents=True,exist_ok=True)
    x=torch.from_numpy(data["x"]); y=torch.from_numpy(data["y"])
    # Weight each independent root equally in the training sampler.
    counts={r:int(np.sum(data["root"][train]==r)) for r in set(data["root"][train])}
    weights=torch.tensor([1/counts[data["root"][i]] for i in train],dtype=torch.double)
    report={}
    for name,scores in (("global",np.broadcast_to(data["y"][train].mean(0),data["y"].shape)),
                        ("prompt",data["x"][...,:128]),("history",data["x"][...,128:]),
                        ("hindsight_greedy",data["y"])):
        report[name]=evaluate(data,scores,dev,a.resident,a.max_swaps)
    # Hindsight with a local swap cap is only a greedy diagnostic; uncapped
    # per-segment oracle provides the mathematical fixed-segment ceiling.
    report["hindsight_uncapped"]=evaluate(data,data["y"],dev,a.resident,a.resident)
    for communicate,name,width in ((False,"independent",96),(True,"mesh",48)):
        torch.manual_seed(a.seed)
        model=Mesh(width=width,communicate=communicate).to(a.device)
        opt=torch.optim.AdamW(model.parameters(),lr=0.001)
        rng=torch.Generator().manual_seed(a.seed)
        for ep in range(a.epochs):
            model.train()
            order=train[torch.multinomial(weights,len(train),replacement=True,generator=rng).numpy()]
            for i in range(0,len(order),a.batch):
                j=order[i:i+a.batch]
                logits=model(x[j].to(a.device))
                loss=-(y[j].to(a.device)*logits.log_softmax(-1)).sum(-1).mean()
                opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            scores=torch.cat([model(x[i:i+a.batch].to(a.device)).softmax(-1).cpu()
                              for i in range(0,len(x),a.batch)]).numpy()
        report[name]=evaluate(data,scores,dev,a.resident,a.max_swaps)
        report[name]["parameters"]=sum(p.numel() for p in model.parameters())
        torch.save({"state":model.state_dict(),"width":width,"communicate":communicate,
                    "seed":a.seed,"resident":a.resident,"max_swaps":a.max_swaps},a.out/f"{name}.pt")
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print({k:{q:v for q,v in r.items() if q!="roots"} for k,r in report.items()})


if __name__=="__main__":
    main()
