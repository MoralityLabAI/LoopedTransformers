"""Causal next-use prediction for exact-route LD-MoE.

Prediction is made after seeing the current native selection; the target is
the next executed visit of the same physical layer. It ranks *evictable*
cache residents. It does not replace native expert IDs, predict before an
earlier prefetch deadline, or certify model quality. No future labels enter
the features or the replay predictor. Three equal-parameter variants differ
only in their causal message inputs.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ld_moe_cache import audit, next_pass, rows

MODES=("local","layer_mesh","pass_mesh")
WIDTH=9*128


def scatter(values,ids):
    a=np.zeros(128,dtype=np.float32)
    a[list(ids)]=values
    return a


def causal_features(events,support,mode):
    """Yield features before updating historical state for this visit.

    The present route is known to a demand-eviction policy. It would be
    leakage for predicting *this* visit before the router executes. Block
    messages contain only the latest already-observed block-end selection,
    never same-step pooled future layers.
    """
    if mode not in MODES:raise ValueError("unknown predictor")
    zero=np.zeros(128,dtype=np.float32)
    same={}; last={}; blocks={}
    for e in events:
        l,p=e["layer"],e["pass"]
        ids=support[l]
        current=scatter([1.]*len(e["experts"]),e["experts"])
        prev=last.get(l,zero)
        tag=(l,next_pass(l,p)) if mode=="pass_mesh" else (l,p)
        message=same.get(tag,zero)
        block_messages=[prev]*4 if mode=="local" else [blocks.get(b,zero) for b in range(4)]
        x=np.concatenate([scatter(e["scores"],ids),scatter(e["rank_scores"],ids),
                          current,prev,message,*block_messages])
        if x.shape!=(WIDTH,):raise AssertionError("feature contract")
        yield x,l,p
        same[l,p]=current
        last[l]=current
        if l%6==5:blocks[l//6]=current


def successor_indices(events):
    """Future use is a supervised target only; never called by inference."""
    last={}
    targets={}
    for i in range(len(events)-1,-1,-1):
        l=events[i]["layer"]
        if l in last:targets[i]=last[l]
        last[l]=i
    return targets


class Mesh(nn.Module):
    """Four independent block parameter banks with causal message features.

    Each node handles six physical layers; shared positional embeddings
    encode exact layer/pass. Identical parameter counts for all three arms.
    Communication is through observed routing summaries, not learned
    recurrent hidden-state exchange yet.
    """
    def __init__(self,width=32):
        super().__init__()
        self.layer=nn.Embedding(24,width)
        self.pass_id=nn.Embedding(3,width)
        self.encoder=nn.ModuleList(nn.Linear(WIDTH,width) for _ in range(4))
        self.head=nn.ModuleList(nn.Linear(width,128) for _ in range(4))

    def forward(self,x,layer,pass_id):
        out=x.new_zeros((len(x),128))
        for b in range(4):
            mask=(layer//6)==b
            if not torch.any(mask):continue
            h=torch.tanh(self.encoder[b](x[mask])+self.layer(layer[mask])+self.pass_id(pass_id[mask]))
            out[mask]=self.head[b](h)
        return out


def training_arrays(path,mode,per_case,cap,seed):
    if not 1<=per_case<=cap:raise ValueError("invalid sample cap")
    rng=np.random.default_rng(seed)
    x=[];layer=[];passes=[];targets=[];masks=[];roots=[]
    cases=set();splits={};identity=None
    for row in rows(path):
        if row["case_id"] in cases:raise ValueError("duplicate case")
        cases.add(row["case_id"])
        if splits.setdefault(row["root"],row["split"])!=row["split"]:raise ValueError("root leakage")
        key=(row["model_sha256"],row["runtime_revision"],json.dumps(row["expert_support"]))
        if identity is None:identity=key
        if key!=identity:raise ValueError("mixed teacher")
        if row["split"]!="train":continue
        e=row["routing_events"];s=row["expert_support"]
        next_idx=successor_indices(e)
        chosen=set(rng.choice(sorted(next_idx),min(per_case,len(next_idx)),replace=False).tolist()) if next_idx else set()
        if len(x)+len(chosen)>cap:raise ValueError("train sample cap exceeded; reduce --samples-per-case")
        for i,(v,l,p) in enumerate(causal_features(e,s,mode)):
            if i not in chosen:continue
            x.append(v);layer.append(l);passes.append(p)
            ids=e[next_idx[i]]["experts"]
            targets.append(scatter([1.]*len(ids),ids))
            masks.append(scatter([1.]*len(s[l]),s[l]))
            roots.append(row["root"])
    if not x or not any(s=="dev" for s in splits.values()):raise ValueError("need train and dev roots")
    return (np.stack(x),np.array(layer),np.array(passes),np.stack(targets),np.stack(masks),roots)


def fit(path,mode,*,width=32,epochs=8,batch=64,per_case=128,cap=32768,seed=17,device="cpu"):
    x,l,p,y,mask,roots=training_arrays(path,mode,per_case,cap,seed)
    torch.manual_seed(seed)
    model=Mesh(width).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=0.002)
    count=Counter(roots)
    sampling=torch.tensor([1/count[r] for r in roots],dtype=torch.double)
    rng=torch.Generator().manual_seed(seed)
    tensors=[torch.from_numpy(a) for a in (x,l,p,y,mask)]
    losses=[]
    for _ in range(epochs):
        model.train()
        order=torch.multinomial(sampling,len(x),replacement=True,generator=rng)
        epoch_sum=0.;n=0
        for start in range(0,len(order),batch):
            index=order[start:start+batch]
            a,b,c,target,available=[t[index].to(device) for t in tensors]
            logits=model(a,b,c)
            # Equal positive/negative class weight per visit; support-aware.
            pos=target.sum(-1,keepdim=True).clamp_min(1.)
            neg=(available-target).sum(-1,keepdim=True).clamp_min(1.)
            w=target/pos+(available-target)/neg
            loss=(nn.functional.binary_cross_entropy_with_logits(logits,target,reduction="none")*w).sum(-1).mean()
            opt.zero_grad();loss.backward();opt.step()
            epoch_sum+=float(loss.detach())*len(index);n+=len(index)
        losses.append(epoch_sum/n)
    return model,{"train_roots":len(count),"train_samples":len(x),"losses":losses,
                  "parameters":sum(v.numel() for v in model.parameters())}


def predict(model,row,mode,device="cpu",batch=64):
    """Returns one retention score vector per already-observed visit."""
    e=row["routing_events"];s=row["expert_support"]
    out=np.empty((len(e),128),dtype=np.float32)
    model.eval()
    buf=[];ls=[];ps=[];offset=0
    def flush():
        nonlocal offset
        if not buf:return
        with torch.no_grad():
            z=model(torch.from_numpy(np.stack(buf)).to(device),torch.tensor(ls,device=device),
                    torch.tensor(ps,device=device)).cpu().numpy()
        out[offset:offset+len(buf)]=z
        offset+=len(buf);buf.clear();ls.clear();ps.clear()
    for x,l,p in causal_features(e,s,mode):
        buf.append(x);ls.append(l);ps.append(p)
        if len(buf)>=batch:flush()
    flush()
    if offset!=len(e):raise AssertionError("incomplete forecast")
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routing",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--capacities",type=int,nargs="+",default=[8,12,16,24,32])
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--samples-per-case",type=int,default=128)
    p.add_argument("--sample-cap",type=int,default=32768)
    p.add_argument("--seed",type=int,default=17)
    a=p.parse_args()
    if a.epochs<1 or a.batch<1 or any(not 8<=k<=128 for k in a.capacities):
        p.error("invalid training budget")
    torch.set_num_threads(1)
    a.out.mkdir(parents=True,exist_ok=False)
    h=hashlib.sha256()
    with a.routing.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    digest=h.hexdigest()
    config={"routing_sha256":digest,"epochs":a.epochs,"batch":a.batch,
            "samples_per_case":a.samples_per_case,"sample_cap":a.sample_cap,
            "seed":a.seed,"device":a.device,"capacities":a.capacities,"width":32,
            "objective":"next_same_physical_layer_membership"}
    for mode in MODES:
        model,train=fit(a.routing,mode,epochs=a.epochs,batch=a.batch,per_case=a.samples_per_case,
                        cap=a.sample_cap,seed=a.seed,device=a.device)
        report=audit(a.routing,tuple(a.capacities),lambda row:predict(model,row,mode,a.device,a.batch))
        report.update(config=config,mode=mode,training=train)
        torch.save({"model":model.cpu().state_dict(),"config":config,"mode":mode},a.out/f"{mode}.pt")
        with (a.out/f"{mode}.json").open("x",encoding="utf8") as f:json.dump(report,f,indent=2,allow_nan=False)
        print(mode,train["parameters"],json.dumps([r for r in report["summary"] if r["policy"]=="forecast"]))


if __name__=="__main__":main()
