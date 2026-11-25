"""LCD-06A: native TRM Sudoku transfer, unique-root generation, whole-grid verifier."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
import torch
from research_gym.scripts import lcd01_huginn as base
from research_gym.scripts import lcd03_anderson as aa

SCHEMA="lcd06a-v1"
CAL_SEED,TEST_SEED=2026112514,2026112515
HOLES=(44,48,52)
DEPTH,REF=8,4
DIGITS=set("123456789")

def verify(question,answer):
    if len(question)!=81 or len(answer)!=81 or any(c not in DIGITS|{"."} for c in question):
        return False
    if any(c not in DIGITS for c in answer) or any(q!="." and q!=a for q,a in zip(question,answer)):
        return False
    return all(set(answer[i*9:i*9+9])==DIGITS for i in range(9)) and all(
        set(answer[i::9])==DIGITS for i in range(9)) and all(
        {answer[9*(r+i)+c+j] for i in range(3) for j in range(3)}==DIGITS
        for r in (0,3,6) for c in (0,3,6))

def smoke():
    a="".join(str((r*3+r//3+c)%9+1) for r in range(9) for c in range(9))
    q="."*40+a[40:]
    assert verify(q,a)
    assert not verify(q,a[:40]+"0"+a[41:])
    assert not verify(q,a[:40]+str(int(a[40])%9+1)+a[41:])
    assert not verify(q,"1"*81)
    assert not verify(q,a[:-1])
    assert not verify("1"+q[1:],str(int(a[0])%9+1)+a[1:])
    print("LCD-06A independent Sudoku verifier smoke PASS")

def roots(seed,per_family,used):
    from mechinterp.dark_fiber_gyms import sudoku_roster as sr
    rng=random.Random(seed)
    rows=[]
    relabel,wl=used
    for family,holes in enumerate(HOLES,1):
        n=0
        for _ in range(5000):
            sol=sr.generate_diverse_solution(rng)
            q=sr.delete_clues(sol,81-holes,rng)
            if q is None or not verify(q,sol) or sr.count_solutions(q)!=1:
                continue
            if not sr._is_novel_and_register(q,relabel,wl):
                continue
            rows.append(dict(id=hashlib.sha256(q.encode()).hexdigest(),family=family,
                             holes=holes,question=q,answer=sol))
            n+=1
            if n==per_family:
                break
        if n!=per_family:
            raise RuntimeError(f"could not produce enough unique {holes}-hole roots: {n}")
    return rows

def grid():
    return [dict(method="scalar",dose=w) for w in (.25,.5,1.)]+[
        dict(method="anderson",window=m,ridge=r,dose=w)
        for m in (3,5) for r in (.001,.01) for w in (.25,.5,1.)]

@torch.inference_mode()
def forward(model,TS,q,c,device):
    e=model.encode_inputs(TS.encode_questions([q],device))
    h,l=model.init_states(1)
    depth=16 if c["method"]=="deep" else DEPTH
    history=[]
    source=c.get("source",c)
    ref=None
    for t in range(1,depth+1):
        h,l=model.act_step(h,l,e)
        if t==REF and c["method"] not in ("base","deep"):
            ref=torch.cat((h,l),-1)
        if "window" in source and DEPTH-source["window"]<=t<=DEPTH:
            history.append(torch.cat((h,l),-1))
    g=None
    rec=None
    if c["method"] not in ("base","deep"):
        x=torch.cat((h,l),-1)
        d=x.float()-ref.float()
        if source["method"]=="scalar":
            g=source["dose"]*d
            rec={}
        else:
            g,rec=aa.bounded_correction(history,source["ridge"],source["dose"])
        if c["method"]=="matched_random":
            g=aa.sham_direction(g,int(hashlib.sha256(q.encode()).hexdigest()[:8],16))
        if c["method"]=="matched_scalar":
            g,bad=aa.matched_direction(d,g)
            rec["degenerate"]=bad
        out=(x.float()+g).to(x.dtype)
        rec["joint_correction_norm"]=float(aa.norm(g))
        rec["readout_h_correction_norm"]=float(aa.norm(out[...,:h.shape[-1]].float()-h.float()))
        h,l=out.split(h.shape[-1],-1)
    return model.digit_logits(h)[0],rec

@torch.inference_mode()
def evaluate(model,TS,roots_,arms,device,out,meta):
    rows={k:[] for k in arms}
    for i,it in enumerate(roots_):
        order=list(arms)
        random.Random(int(it["id"][:8],16)).shuffle(order)
        for k in order:
            if device=="cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            z,rec=forward(model,TS,it["question"],arms[k],device)
            if device=="cuda":
                torch.cuda.synchronize()
            ms=(time.perf_counter()-start)*1000
            peak=torch.cuda.max_memory_allocated() if device=="cuda" else None
            if not torch.isfinite(z).all() or z.shape!=(81,9):
                raise ValueError("bad Sudoku logits")
            ans="".join(str(int(v)+1) for v in z.argmax(-1).cpu())
            mask=torch.tensor([q=="." for q in it["question"]],device=device)
            target=torch.tensor([int(a)-1 for a in it["answer"]],device=device)
            nll=-float(z.log_softmax(-1).gather(1,target[:,None])[:,0][mask].mean())
            valid=verify(it["question"],ans)
            rows[k].append(dict(id=it["id"],family=it["family"],holes=it["holes"],
                correct=valid and ans==it["answer"],valid=valid,predicted_grid=ans,nll=nll,
                missing_cell_accuracy=sum(a==b for q,a,b in zip(it["question"],ans,it["answer"]) if q==".")/it["holes"],
                loops=16 if arms[k]["method"]=="deep" else DEPTH,latency_ms=ms,peak_bytes=peak,
                intervention=rec))
        if (i+1)%8==0:
            print("LCD-06A",i+1,len(roots_),flush=True)
            base.write(out,dict(meta=meta,status="IN_PROGRESS",arms=arms,rows=rows,roster=roots_))
    return rows,{k:base.summarize(v) for k,v in rows.items()}

def select(arms,summaries):
    chosen={}
    for m in ("scalar","anderson"):
        keys=[k for k,c in arms.items() if c["method"]==m]
        chosen[m]=arms[min(keys,key=lambda k:(-summaries[k]["accuracy"],summaries[k]["nll"],k))]
    return dict(base8=dict(method="base"),base16=dict(method="deep"),scalar=chosen["scalar"],
                anderson=chosen["anderson"],scalar_random=dict(method="matched_random",source=chosen["scalar"]),
                aa_random=dict(method="matched_random",source=chosen["anderson"]),
                aa_scalar=dict(method="matched_scalar",source=chosen["anderson"]))

def main(a):
    # Existing loader validates the pinned checkpoint SHA before torch.load.
    os.environ["HF_HUB_OFFLINE"]="1"
    from mechinterp.loopscope02.scripts import trm_sudoku as TS
    device=a.device
    if device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    dtype=torch.bfloat16 if device=="cuda" else torch.float32
    model,sha=TS.load_trm_checkpoint(a.checkpoint,device,dtype=dtype)
    model.eval()
    used=(set(),{})
    cal=roots(CAL_SEED,12,used)
    meta=dict(schema=SCHEMA,checkpoint_sha=sha,checkpoint=a.checkpoint,device=device,
              dtype=str(dtype),torch=torch.__version__,cal_sha=base.digest(cal),
              code_sha=hashlib.sha256(Path(__file__).read_bytes()+Path(aa.__file__).read_bytes()+
                                       Path(TS.__file__).read_bytes()).hexdigest(),
              provenance="fresh synthetic diverse unique roots; no external training-corpus exclusion audit")
    configs={base.key(c):c for c in grid()}
    if a.stage=="calibrate":
        rows,summaries=evaluate(model,TS,cal,configs,device,a.out,meta)
        arms=select(configs,summaries)
        base.write(a.out,dict(meta=meta,status="FROZEN",freeze_sha=base.digest(arms),arms=arms,
                rows=rows,summaries=summaries,roster=cal))
        return
    f=json.loads(a.freeze.read_text(encoding="utf-8"))
    if f["status"]!="FROZEN" or f["meta"]!=meta or f["freeze_sha"]!=base.digest(f["arms"]):
        raise ValueError("invalid freeze")
    arms=f["arms"]
    if arms!=select(configs,f["summaries"]):
        raise ValueError("selection mismatch")
    test=roots(TEST_SEED,32,used)
    meta["test_sha"]=base.digest(test)
    rows,summaries=evaluate(model,TS,test,arms,device,a.out,meta)
    comparisons={m:{k:base.paired(rows[m],v) for k,v in rows.items() if k!=m}
                 for m in ("scalar","anderson")}
    base.write(a.out,dict(meta=meta,status="PILOT_COMPLETE",arms=arms,freeze_sha=f["freeze_sha"],
            rows=rows,summaries=summaries,comparisons=comparisons,roster=test))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("smoke","calibrate","test"),required=True)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--checkpoint",choices=("gaoxin492","yagizdevre"),default="gaoxin492")
    p.add_argument("--out",type=Path)
    p.add_argument("--freeze",type=Path)
    a=p.parse_args()
    if a.stage=="smoke":
        smoke()
    else:
        if a.out is None or a.out.exists() or (a.stage=="test" and a.freeze is None):
            p.error("need fresh output and test freeze")
        main(a)
