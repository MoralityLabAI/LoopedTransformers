"""LCD-05: packed recurrent-state precision with protected low-rank modes.

Actual int4/int8 payload plus FP16 scales and BF16 protected coefficients.
Native recurrent matmuls stay BF16. This is a state codec experiment, not
a low-bit weight model or custom fused kernel.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import torch
from research_gym.scripts import lcd01_huginn as base
from research_gym.scripts import lcd02_spectral as spectral
from research_gym.scripts import lcd03_anderson as aa

SCHEMA="lcd05-v1"
FIT_SEED,SELECT_SEED,TEST_SEED=2026112511,2026112512,2026112513
RANKS=(4,8,16)
DEPTH=16
REF=6
GROUP=64
RANDOM_SEED=271829

def bytes_of(t):
    return t.numel()*t.element_size()

def pack_quant(x,bits,group=GROUP):
    if bits not in (4,8) or group<=0 or group%2:
        raise ValueError("unsupported quantizer")
    x=x.float()
    d=x.shape[-1]
    pad=(-d)%group
    padded=torch.nn.functional.pad(x,(0,pad))
    chunks=padded.reshape(*x.shape[:-1],-1,group)
    qmax=(1<<(bits-1))-1
    scale=(chunks.abs().amax(-1,keepdim=True)/qmax).clamp_min(2.**-24).half()
    if not torch.isfinite(scale).all() or not torch.isfinite(x).all():
        raise ValueError("nonfinite quantizer input or FP16 scale overflow")
    q=(chunks/scale.float()).round().clamp(-qmax,qmax).to(torch.int8)
    if bits==4:
        v=(q.to(torch.int16)+8).to(torch.uint8)
        q=(v[...,::2] | (v[...,1::2]<<4)).contiguous()
    return dict(q=q,scale=scale,d=d,bits=bits,group=group)

def unpack_quant(p):
    q=p["q"]
    if p["bits"]==4:
        lo=(q&15).to(torch.int16)-8
        hi=(q>>4).to(torch.int16)-8
        q=torch.stack((lo,hi),dim=-1).flatten(-2)
    x=q.float()*p["scale"].float()
    return x.flatten(-2)[...,:p["d"]]

def pack(h,bits,u=None):
    x=h.float()
    coeff=None
    if u is not None:
        # Separate protected coordinates from the orthogonal residual.
        a=x@u
        coeff=a.bfloat16()
        x=x-a@u.T
    return dict(quant=pack_quant(x,bits),coeff=coeff)

def unpack(p,u=None,dtype=torch.bfloat16):
    residual=unpack_quant(p["quant"])
    if u is not None:
        # Quantizer error can leak into U. Remove it before restoring a.
        residual=residual-(residual@u)@u.T
        residual=residual+p["coeff"].float()@u.T
    return residual.to(dtype)

def payload_bytes(p):
    q=p["quant"]
    return bytes_of(q["q"])+bytes_of(q["scale"])+(0 if p["coeff"] is None else bytes_of(p["coeff"]))

def smoke():
    rng=torch.Generator().manual_seed(13)
    x=torch.randn(1,7,131,generator=rng)
    for bits in (4,8):
        p=pack_quant(x,bits)
        y=unpack_quant(p)
        assert y.shape==x.shape
        qmax=(1<<(bits-1))-1
        # Actual storage is nibble-packed for int4, byte per code for int8.
        assert p["q"].dtype==(torch.uint8 if bits==4 else torch.int8)
        assert p["q"].numel()==7*192*bits//8
        assert p["scale"].numel()==7*3
        assert (x-y).abs().max()<=p["scale"].float().max()*.501+1e-6
        assert torch.equal(unpack_quant(pack_quant(torch.zeros_like(x),bits)),torch.zeros_like(x))
        extremes=torch.tensor([[-float(qmax),-1.,0.,1.,float(qmax)]])
        assert torch.equal(unpack_quant(pack_quant(extremes,bits)),extremes)
    u,_=torch.linalg.qr(torch.randn(131,8,generator=rng))
    p=pack(x,4,u)
    y=unpack(p,u,torch.float32)
    assert torch.allclose(y@u,p["coeff"].float(),atol=1e-5)
    assert payload_bytes(p)==7*(96+6+16)
    assert payload_bytes(pack(x,4))<payload_bytes(p)<bytes_of(x.bfloat16())
    assert torch.equal(unpack(p,u),unpack(pack(x,4,u),u))
    # A synthetic protected signal plus orthogonal noise: measure projection
    # error rather than claiming any language-model accuracy from this test.
    a=(x@u).bfloat16().float()
    err=float(((y@u)-a).norm())
    assert err<1e-4
    print("LCD-05 packed codec and protected projection smoke PASS", "projection_error",err)

def config_grid():
    return [dict(method="bf16"),dict(method="uniform",bits=4),dict(method="uniform",bits=8)]+[
        dict(method=m,bits=4,rank=r) for m in ("contrast","state","random") for r in RANKS]

@torch.inference_mode()
def learn_bases(model,tok,roots):
    contrasts=[]
    states=[]
    for i,it in enumerate(roots):
        ss=base.trajectory(model,spectral.input_ids(tok,it),spectral.seed(it),DEPTH,(REF,))
        h=ss[DEPTH][0,-1].float().cpu()
        contrasts.append(h-ss[REF][0,-1].float().cpu())
        states.append(h)
        if (i+1)%8==0:
            print("LCD-05 basis fit",i+1,len(roots),flush=True)
    d=torch.stack(contrasts)
    h=torch.stack(states)
    return {k:u.tolist() for k,u in dict(
        contrast=spectral.get_basis(d,max(RANKS)),state=spectral.get_basis(h,max(RANKS)),
        random=spectral.get_basis(d,max(RANKS),RANDOM_SEED)).items()}

@torch.inference_mode()
def forward(model,ids,it,c,u):
    e,h,idx=aa.initialize(model,ids,spectral.seed(it))
    if h.dtype!=torch.bfloat16:
        raise ValueError("LCD-05 requires BF16 native recurrent state")
    native_bytes=bytes_of(h)
    encoded=None
    last_error=None
    for t in range(DEPTH):
        if encoded is not None:
            h=unpack(encoded,u)
            del encoded
            encoded=None
        h,idx=aa.step(model,e,h,idx,t)
        if h.dtype!=torch.bfloat16:
            raise ValueError("native recurrent dtype changed")
        if c["method"]!="bf16":
            encoded=pack(h,c["bits"],u)
            if t==DEPTH-1:
                # The last state is decoded for the coda; retain error as a
                # separate diagnostic and account its work in wall time.
                decoded=unpack(encoded,u)
                last_error=float((decoded.float()-h.float()).norm()/h.float().norm().clamp_min(1e-20))
                if not math.isfinite(last_error):
                    raise ValueError("nonfinite codec error")
                del decoded
            del h
    payload=native_bytes if encoded is None else payload_bytes(encoded)
    if encoded is not None:
        h=unpack(encoded,u)
        del encoded
    z=base.read(model,h)
    return z,dict(state_bytes=payload,native_state_bytes=native_bytes,
                  basis_bytes=0 if u is None else bytes_of(u),last_roundtrip_relative_error=last_error,
                  quantized_transitions=0 if c["method"]=="bf16" else DEPTH)

@torch.inference_mode()
def evaluate(model,tok,labels,cand,roots,arms,bases,meta,out,admission,timed=False):
    rows={k:[] for k in arms}
    # Keep basis arrays on host except the selected arm, to report actual
    # allocated basis residency. Upload not included in steady-state time.
    host={k:torch.tensor(v,dtype=torch.float32) for k,v in bases.items()}
    for i,it in enumerate(roots):
        ids=spectral.input_ids(tok,it)
        order=list(arms)
        random.Random(spectral.seed(it)).shuffle(order)
        for k in order:
            c=arms[k]
            u=host[c["method"]][:,:c["rank"]].contiguous().to("cuda") if "rank" in c else None
            base.sync()
            before=torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            z,info=forward(model,ids,it,c,u)
            base.sync()
            elapsed=(time.perf_counter()-start)*1000
            peak=torch.cuda.max_memory_allocated()
            r=base.row(z,cand,labels.index(it["answer"]))
            r.update(id=it["id"],family=it["family"],tokens=ids.shape[-1],loops=DEPTH,
                     coda_passes=1,latency_ms=elapsed,peak_bytes=peak,pre_forward_bytes=before,**info)
            rows[k].append(r)
            del z,u
        if (i+1)%8==0:
            print("LCD-05 eval",i+1,len(roots),flush=True)
            base.write(out,dict(meta=meta,status="IN_PROGRESS",admission=admission,arms=arms,rows=rows))
    return rows,{k:base.summarize(v) for k,v in rows.items()}

def choose(arms,summaries):
    def best(m):
        cs=[k for k,c in arms.items() if c["method"]==m]
        return arms[min(cs,key=lambda k:(-summaries[k]["accuracy"],summaries[k]["nll"],
                                           arms[k]["rank"],k))]
    p=best("contrast")
    r=p["rank"]
    return dict(bf16=dict(method="bf16"),q4=dict(method="uniform",bits=4),q8=dict(method="uniform",bits=8),
                contrast=p,random_matched=dict(method="random",bits=4,rank=r),
                state_matched=dict(method="state",bits=4,rank=r),
                random_tuned=best("random"),state_tuned=best("state"))

def costs(rr):
    s=sum(r["state_bytes"] for r in rr)
    n=sum(r["native_state_bytes"] for r in rr)
    return dict(payload_ratio=s/n,basis_bytes=max(r["basis_bytes"] for r in rr),
                total_forward_ms=sum(r["latency_ms"] for r in rr),peak_bytes=max(r["peak_bytes"] for r in rr),
                mean_incremental_peak=sum(r["peak_bytes"]-r["pre_forward_bytes"] for r in rr)/len(rr))

def main(a):
    model,tok,labels,cand,meta=spectral.load()
    meta.update(schema=SCHEMA,code_sha=hashlib.sha256(b"".join(Path(m.__file__).read_bytes()
                for m in (base,spectral,aa))+Path(__file__).read_bytes()).hexdigest(),
                depth=DEPTH,ref=REF,group=GROUP)
    fit=base.items(FIT_SEED,16)
    selection=base.items(SELECT_SEED,16)
    meta.update(fit_sha=base.digest(fit),select_sha=base.digest(selection))
    fs={r["id"] for r in fit}
    ss={r["id"] for r in selection}
    if len(fs)!=len(fit) or len(ss)!=len(selection) or fs&ss:
        raise ValueError("calibration split collision")
    ok,receipt=spectral.parity(model,tok,fit)
    if not ok:
        base.write(a.out,dict(meta=meta,status="BLOCKED_PARITY",admission=receipt))
        raise RuntimeError("native parity failed")
    grid={base.key(c):c for c in config_grid()}
    if a.stage=="fit":
        start=time.perf_counter()
        bases=learn_bases(model,tok,fit)
        base.sync()
        fit_seconds=time.perf_counter()-start
        start=time.perf_counter()
        rows,summaries=evaluate(model,tok,labels,cand,selection,grid,bases,meta,a.out,receipt)
        arms=choose(grid,summaries)
        frozen=dict(bases=bases,arms=arms)
        base.write(a.out,dict(meta=meta,status="FROZEN",admission=receipt,frozen=frozen,
            freeze_sha=base.digest(frozen),fit_elapsed_s=fit_seconds,
            selection_elapsed_s=time.perf_counter()-start,rows=rows,summaries=summaries))
        return
    f=json.loads(a.freeze.read_text(encoding="utf-8"))
    if f["status"]!="FROZEN" or f["meta"]!=meta or f["freeze_sha"]!=base.digest(f["frozen"]):
        raise ValueError("freeze mismatch")
    frozen=f["frozen"]
    arms=frozen["arms"]
    if arms!=choose(grid,f["summaries"]):
        raise ValueError("selection mismatch")
    for name,v in frozen["bases"].items():
        u=torch.tensor(v)
        if not torch.isfinite(u).all() or not torch.allclose(u.T@u,torch.eye(max(RANKS)),atol=1e-4):
            raise ValueError("invalid frozen basis "+name)
    test=base.items(TEST_SEED,64)
    ts={r["id"] for r in test}
    if len(ts)!=len(test) or ts&(fs|ss):
        raise ValueError("test split collision")
    meta["test_sha"]=base.digest(test)
    rows,summaries=evaluate(model,tok,labels,cand,test,arms,frozen["bases"],meta,a.out,receipt,timed=True)
    cs={k:costs(v) for k,v in rows.items()}
    p=rows["contrast"]
    comparisons={k:dict(paired=base.paired(p,v),latency_ratio=cs["contrast"]["total_forward_ms"]/cs[k]["total_forward_ms"])
                 for k,v in rows.items() if k!="contrast"}
    base.write(a.out,dict(meta=meta,status="PILOT_COMPLETE",admission=receipt,arms=arms,
                freeze_sha=f["freeze_sha"],rows=rows,summaries=summaries,costs=cs,comparisons=comparisons))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("smoke","fit","test"),required=True)
    p.add_argument("--out",type=Path)
    p.add_argument("--freeze",type=Path)
    a=p.parse_args()
    if a.stage=="smoke":
        smoke()
    else:
        if a.out is None or a.out.exists() or (a.stage=="test" and a.freeze is None):
            p.error("fresh output and test freeze required")
        main(a)
