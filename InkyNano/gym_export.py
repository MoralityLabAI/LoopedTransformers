"""Cloud exporter from the pinned native eager teacher; no automatic upload.

Use a fresh output directory and an immutable, public/synthetic roster.
The full native loader is a pilot reference, not a memory-optimal extractor.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
import torch
from safetensors.torch import save_file
from inventory import MODEL, REV
from native_expert import audit_native, expert, factor_expert
from spectral import nested_svd, relative_error

TRANSFORMERS_REV = "d8f10a82213a3a9db48385aba3a4b57ab535de50"
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.(gate|experts)$")


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf8")


def modules(model):
    found={"gate":{},"experts":{}}
    for name,mod in model.named_modules():
        m=LAYER.search(name)
        if m:
            l,kind=int(m[1]),m[2]
            if l in found[kind]:
                raise ValueError("ambiguous decoder modules")
            found[kind][l]=mod
    if any(set(v)!=set(range(24)) for v in found.values()):
        raise ValueError("incomplete native physical layers")
    for l,m in found["experts"].items():
        if tuple(m.gate_up_proj.shape)!=(128,2304,2304) or tuple(m.down_proj.shape)!=(128,2304,1152):
            raise ValueError("unexpected native expert layout")
    return found


@torch.inference_mode()
def capture(model,routers,ids,device):
    visits={l:[] for l in routers}
    handles=[]
    def make(l):
        def hook(_m,_inp,out):
            _,weights,indices=out
            if indices.shape!=(len(ids),8) or weights.shape!=indices.shape:
                raise ValueError("unexpected router shape")
            visits[l].append((indices.detach().cpu(),weights.detach().float().cpu()))
        return hook
    try:
        for l,mod in routers.items():
            handles.append(mod.register_forward_hook(make(l)))
        tokens=torch.tensor([ids],device=device,dtype=torch.long)
        model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False)
    finally:
        for h in handles:
            h.remove()
    for l,v in visits.items():
        if len(v)!=(3 if 6<=l<18 else 1):
            raise ValueError("unexpected recurrent visit count")
        for ix,w in v:
            if torch.any(~torch.isfinite(w)) or torch.any(w<0) or torch.any(w.sum(-1)<=0):
                raise ValueError("invalid native weights")
    return visits


def prefix_check(model,routers,ids,p,device):
    if not 0<p<len(ids):
        raise ValueError("need nonempty prefix and continuation")
    short=capture(model,routers,ids[:p],device)
    full=capture(model,routers,ids,device)
    errors=[]
    for l in short:
        for (ix,w),(ix2,w2) in zip(short[l],full[l]):
            # Top-k is unsorted; compare by expert ID.
            a=ix.argsort(-1);b=ix2[:p].argsort(-1)
            if not torch.equal(ix.gather(-1,a),ix2[:p].gather(-1,b)):
                raise ValueError("prefix routing ID mismatch")
            errors.append(float((w.gather(-1,a)-w2[:p].gather(-1,b)).abs().max()))
    return max(errors)


def trace_export(model,tokenizer,routers,roster,out,device,max_tokens,prefix_tol):
    raw=[json.loads(l) for l in roster.read_text(encoding="utf8").splitlines() if l.strip()]
    if not raw:
        raise ValueError("empty roster")
    keys=set(); families={};roots={}
    prepared=[]
    for r in raw:
        if r["split"] not in ("train","dev") or r.get("source_kind") not in ("public","synthetic"):
            raise ValueError("only public/synthetic development roots allowed")
        key=r["root"],r.get("turn","0")
        if key in keys or families.setdefault(r["family"],r["split"])!=r["split"] or roots.setdefault(r["root"],r["split"])!=r["split"]:
            raise ValueError("roster leakage or duplicate")
        keys.add(key)
        # Explicit raw-text template for the first study; continuation IDs are
        # encoded separately so the boundary is exact. Chat-formatted roster
        # may supply the complete serialized prompt.
        p=tokenizer.encode(r["prompt"],add_special_tokens=True)
        c=tokenizer.encode(r["continuation"],add_special_tokens=False)
        if not p or not c or len(p)+len(c)>max_tokens:
            raise ValueError("invalid or oversized prompt/continuation")
        prepared.append((r,key,p+c,len(p)))
    dev=next((q for q in prepared if q[0]["split"]=="dev"),None)
    if dev is None:
        raise ValueError("need independent development root for prefix check")
    error=prefix_check(model,routers,dev[2],dev[3],device)
    if error>prefix_tol:
        raise ValueError(f"prefix parity error {error} exceeds {prefix_tol}")
    with (out/"roster.jsonl").open("w",encoding="utf8") as rf,(out/"trace.jsonl").open("w",encoding="utf8") as tf:
        for r,(root,turn),ids,p in prepared:
            visits=capture(model,routers,ids,device)
            rf.write(json.dumps({"root":root,"turn":turn,"family":r["family"],"split":r["split"],
                                 "source_kind":r["source_kind"],"source":r["source"],
                                 "token_ids":ids,"prompt_len":p})+"\n")
            for l,vs in visits.items():
                for pass_id,(indices,weights) in enumerate(vs):
                    for t,(ix,w) in enumerate(zip(indices.tolist(),weights.tolist())):
                        tf.write(json.dumps({"root":root,"turn":turn,"token":t,"layer":l,"pass":pass_id,
                                             "experts":ix,"weights":w,"phase":"prompt" if t<p else "continuation"})+"\n")
    dump(out/"trace_receipt.json",{"roots":len(roots),"turns":len(prepared),"prefix_max_abs":error,
                                    "prefix_tolerance":prefix_tol,"mode":"teacher_forced",
                                    "source_roster_sha256":hashlib.sha256(roster.read_bytes()).hexdigest()})


def pack4(w,block=64):
    """Symmetric signed INT4, last-axis blocks; low nibble precedes high.
    Return packed codes, FP16 scales, unpadded shape. No runtime speed claim.
    """
    shape=list(w.shape)
    w=w.float().contiguous()
    n=w.shape[-1]
    padded=((n+block-1)//block)*block
    w=torch.nn.functional.pad(w,(0,padded-n)).reshape(*w.shape[:-1],-1,block)
    scales=(w.abs().amax(-1)/7).clamp_min(2**-24).to(torch.float16)
    codes=(w/scales.float().unsqueeze(-1)).round().clamp(-7,7).to(torch.int16)+8
    codes=codes.to(torch.uint8).flatten(-2)
    return (codes[...,::2]|(codes[...,1::2]<<4)).contiguous(),scales.contiguous(),shape


def unpack4(packed,scale,shape,block=64):
    lo=packed&15;hi=packed>>4
    codes=torch.stack((lo,hi),dim=-1).flatten(-2).to(torch.float32)-8
    return (codes.reshape(*codes.shape[:-1],-1,block)*scale.float().unsqueeze(-1)).flatten(-2)[...,:shape[-1]]


@torch.inference_mode()
def spectral_export(experts,out,rank,fmt,atlas,parity_tol,seed):
    directory=out/"spectral";directory.mkdir()
    metrics=[];index={}
    rng=torch.Generator().manual_seed(seed)
    plan={l:torch.randperm(128,generator=rng)[:4].tolist() for l in (0,6,17,23)}
    if atlas:
        plan={l:list(range(128)) for l in range(24)}
    for l,ids in plan.items():
        tensors={};meta={}
        mod=experts[l]
        for e in ids:
            fused=mod.gate_up_proj[e];down=mod.down_proj[e]
            gate,up=fused.chunk(2,dim=0)
            x=torch.randn(16,2304,generator=rng).to(device=fused.device,dtype=fused.dtype)
            receipt=audit_native(mod,e,x)
            if receipt["relative_l2"]>parity_tol:
                raise ValueError(f"native expert parity failed: {l}/{e}")
            matrices=(gate,up,down)
            factors=[];packed_factors=[]
            projection={}
            for name,w in zip(("gate","up","down"),matrices):
                # Exact SVD for pilot; seeded randomized SVD for full atlas.
                left,right,s=nested_svd(w,rank,randomized=atlas,seed=seed+l*128+e)
                factors.append((left,right))
                entry=f"e{e:03d}.{name}"
                projection[name]={"rank_errors":{str(k):relative_error(w,left,right,k)
                                              for k in sorted(set([32,64,96,128,192,256,rank])) if k<=rank}}
                packed_pair=[]
                for side,mat in (("left",left),("right",right)):
                    key=entry+"."+side
                    if fmt=="int4":
                        q,scale,shape=pack4(mat)
                        tensors[key+".q"]=q.cpu();tensors[key+".scale"]=scale.cpu()
                        meta[key]={"shape":shape,"block":64}
                        packed_pair.append(unpack4(q,scale,shape))
                    else:
                        tensors[key]=mat.half().cpu().contiguous()
                        meta[key]={"shape":list(mat.shape)}
                        packed_pair.append(mat.half().float())
                packed_factors.append(tuple(packed_pair))
                projection[name]["stored_rank_error"]=relative_error(w,*packed_pair,rank)
                tensors[entry+".s"]=s.cpu()
            target=expert(x,fused,down).float()
            denom=torch.linalg.vector_norm(target).clamp_min(1e-20)
            nonlinear={}
            for k in sorted(set([32,64,96,128,192,256,rank])):
                if k<=rank:
                    pred=factor_expert(x.float(),*factors,k)
                    packed_pred=factor_expert(x.float(),*packed_factors,k)
                    nonlinear[str(k)]={"fp32_relative_l2":float(torch.linalg.vector_norm(pred-target)/denom),
                                      "stored_relative_l2":float(torch.linalg.vector_norm(packed_pred-target)/denom)}
            metrics.append({"layer":l,"expert":e,"parity":receipt,"projection":projection,
                            "nonlinear":nonlinear,"inputs":"seeded Gaussian only"})
        path=directory/f"layer_{l:02d}.safetensors"
        save_file(tensors,str(path),metadata={"format":fmt,"rank":str(rank),"revision":REV})
        index[str(l)]={"file":path.name,"experts":ids,"tensors":meta,"bytes":path.stat().st_size}
        dump(directory/"atlas_index.json",{"complete":False,"format":fmt,"rank":rank,"layers":index})
        dump(directory/"pilot_metrics.json",metrics)
    dump(directory/"atlas_index.json",{"complete":atlas,"format":fmt,"rank":rank,"layers":index})


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--inventory",type=Path,required=True)
    p.add_argument("--roster",type=Path)
    p.add_argument("--device",default="cuda")
    p.add_argument("--rank",type=int,default=256)
    p.add_argument("--format",choices=("fp16","int4"),default="fp16")
    p.add_argument("--max-tokens",type=int,default=512)
    p.add_argument("--prefix-tol",type=float,required=True)
    p.add_argument("--parity-tol",type=float,required=True)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--atlas",action="store_true")
    p.add_argument("--approved-atlas",action="store_true")
    p.add_argument("--trace-only",action="store_true")
    a=p.parse_args()
    if a.trace_only and not a.roster:
        raise ValueError("trace-only requires a roster")
    if a.atlas and not a.approved_atlas:
        raise ValueError("explicit full atlas approval required")
    if not 1<=a.rank<=1152 or not 0<a.parity_tol<1 or not 0<a.prefix_tol<1:
        raise ValueError("invalid rank/tolerance")
    inv=json.loads(a.inventory.read_text(encoding="utf8"))
    if inv["revision"]!=REV or inv["status"]!="metadata_only_native_parity_pending":
        raise ValueError("wrong inventory")
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"inventory.json").write_bytes(a.inventory.read_bytes())
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import transformers
    start=time.monotonic()
    # Pin exact implementation via the environment install, record its
    # package version and git receipt in the cloud run log.
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,dtype=torch.bfloat16,
                                               device_map={"":a.device}).eval()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REV)
    found=modules(model)
    if a.roster:
        trace_export(model,tokenizer,found["gate"],a.roster,a.out,a.device,a.max_tokens,a.prefix_tol)
    if not a.trace_only:
        spectral_export(found["experts"],a.out,a.rank,a.format,a.atlas,a.parity_tol,a.seed)
    dump(a.out/"export_receipt.json",{"model":MODEL,"revision":REV,"required_transformers_commit":TRANSFORMERS_REV,
                                     "transformers_version":transformers.__version__,"torch":str(torch.__version__),
                                     "device":a.device,"elapsed_seconds":time.monotonic()-start,
                                     "seed":a.seed,"atlas":a.atlas,"format":a.format,"rank":a.rank})


if __name__=="__main__":
    main()
