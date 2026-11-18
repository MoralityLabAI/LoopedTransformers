"""Deterministic pre-inference NAN-COMP-04 development source generator.

Pure stdlib. Produces public rows, host-only labels and source certificates.
Does not import model, EvalMath, earlier experiments' roster functions or any
held-out ARG-03 test. Never draws reserved confirmation-root source examples.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random

from .source import COEFFICIENTS, OPS, certificate, degrees, make_source, transform

HERE=Path(__file__).resolve().parent
PUBLIC_KEYS={"case_id","root_group","split","kind","m","tier","degree","demos","query","public_sha256"}


def canonical(obj)->str:
    return json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=True)


def fingerprint(obj)->str:
    return sha256(canonical(obj).encode("utf-8")).hexdigest()


def config()->dict:
    cfg=json.loads((HERE/"config.json").read_text(encoding="utf-8"))
    if cfg["experiment_id"]!="NAN-COMP-04" or cfg["BPD_transfer_enabled"] is not False:
        raise ValueError("experiment or safety contract drift")
    roots=cfg["calibration_roots"]+cfg["development_roots"]+cfg["reserved_confirmation_roots"]
    if len(roots)!=len(set(roots)):
        raise ValueError("root population overlap")
    if cfg["m_values"]!=[3,5] or cfg["observed_degrees"]!=["2m-1","2m","2m+2"]:
        raise ValueError("registered experiment geometry changed")
    return cfg


def _row(root:int,kind:str,tier:str,m:int,degree:int,demos:list,query:list,
         truth:str,split:str,oracle_answer):
    root_key=fingerprint({"study":"NAN-COMP-04","root":root})[:20]
    opaque_id=fingerprint({"study":"NAN-COMP-04","root":root,"kind":kind,
                           "tier":tier,"truth":truth})[:32]
    row={"case_id":"nc04:"+opaque_id,"root_group":root_key,"split":split,"kind":kind,"m":m,"tier":tier,
         "degree":degree,"demos":demos,"query":query}
    row["public_sha256"]=fingerprint({k:v for k,v in row.items()
                                      if k not in ("case_id","public_sha256")})
    hidden={"case_id":row["case_id"],"root":root,"root_key":root_key,
            "split":split,"kind":kind,"tier":tier,"truth":truth,
            "oracle_answer":oracle_answer,"public_sha256":row["public_sha256"]}
    return row,hidden


def build(cfg:dict|None=None):
    cfg=cfg or config()
    public=[];host=[];roots=[];source_keys=set()
    registered={"calibration":cfg["calibration_roots"],"development":cfg["development_roots"]}
    for split,roster in registered.items():
        for index,root in enumerate(roster):
            m=cfg["m_values"][index%2]
            rng=random.Random(root ^ cfg["source_seed"])
            a=rng.choice(COEFFICIENTS);c=rng.choice(COEFFICIENTS)
            a_other=-a;c_other=rng.choice(COEFFICIENTS)
            aq=rng.choice(COEFFICIENTS);cq=rng.choice(COEFFICIENTS)
            demos=[make_source(m,a,c),make_source(m,a_other,c_other)]
            query=make_source(m,aq,cq)
            identity=fingerprint({"m":m,"demos":demos,"query":query})
            if identity in source_keys:
                raise ValueError("Duplicate independent generator-root source; fail closed, do not resample")
            source_keys.add(identity)
            for f in [*demos,query]:
                cert=certificate(m,f[m],f[m+2])
                if cert["source"]!=f:
                    raise AssertionError("source certificate drift")
            vals={op:[transform(f,op) for f in demos] for op in OPS}
            tiers=degrees(m)
            if vals["N"][0][:tiers["early"]+1]!=vals["R"][0][:tiers["early"]+1]:
                raise AssertionError("early N/R collision changed")
            if any(vals["N"][i][:tiers["cross"]+1]==vals["R"][i][:tiers["cross"]+1]
                   for i in (0,1)):
                raise AssertionError("cross-term N/R difference missing")
            def insert(kind,tier,truth,degree,mode="shown",answer=None):
                rows=[]
                if kind=="classification":
                    for f,y in zip(demos,vals[truth],strict=True):
                        rows.append({"input":f[:degree+1],
                                     "output":y[:degree+1] if mode=="shown" else None})
                    question=[]
                else:
                    question=query[:degree+1]
                p,h=_row(root,kind,tier,m,degree,rows,question,truth,split,
                          truth if kind=="classification" else answer)
                public.append(p);host.append(h)
            for op in OPS:
                insert("classification","first",op,tiers["first"])
            for op in ("N","R"):
                insert("classification","early",op,tiers["early"])
                insert("classification","cross",op,tiers["cross"])
                insert("classification","erased",op,tiers["first"],mode="erased")
            insert("coefficient","query_cross","R",tiers["cross"],answer=2*aq*cq)
            roots.append({"root":root,"root_key":fingerprint({"study":"NAN-COMP-04",
                                                              "root":root})[:20],
                          "split":split,"m":m,"a":a,"c":c,"a_other":a_other,
                          "c_other":c_other,"query_a":aq,"query_c":cq,
                          "source_identity":identity})
    if len(public)!=len(host) or len({p["case_id"] for p in public})!=len(public):
        raise AssertionError("incomplete or duplicated source cases")
    if len(public)!=11*(len(cfg["calibration_roots"])+len(cfg["development_roots"])):
        raise AssertionError("expected 11 tasks per root")
    return {"public":public,"host":host,"roots":roots}


def save(out:Path,cohort:dict)->dict:
    if out.exists():
        raise FileExistsError("refuse to overwrite pre-inference source cohort")
    out.mkdir(parents=True,exist_ok=False)
    files={
        "public.jsonl":"".join(canonical(x)+"\n" for x in cohort["public"]),
        "host_only.jsonl":"".join(canonical(x)+"\n" for x in cohort["host"]),
        "roots.json":json.dumps(cohort["roots"],indent=2,sort_keys=True)+"\n",
    }
    for name,body in files.items():
        (out/name).write_text(body,encoding="utf-8",newline="\n")
    cfg=config()
    receipt={
        "schema":"nan_comp04.sourceonly.v1","model_inference":False,
        "BPD_transfer_attempted":False,"original_ARG03_168_test_opened":False,
        "reserved_confirmation_sources_generated":False,
        "calibration_roots":len(cfg["calibration_roots"]),
        "development_roots":len(cfg["development_roots"]),
        "worlds":len(cohort["public"]),
        "tasks":dict(Counter(p["kind"]+":"+p["tier"] for p in cohort["public"])),
        "source_code_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256":sha256((HERE/"config.json").read_bytes()).hexdigest(),
        "files_sha256":{n:sha256((out/n).read_bytes()).hexdigest() for n in files},
        "design":"two demos per root; early N/R collision, first evidence and cross-term, separate 2ac computation",
    }
    (out/"source_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",
                                            encoding="utf-8",newline="\n")
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    for part in a.out.parts:
        if part.lower() in ("test","heldout","test_full168"):
            raise ValueError("sealed/test path prohibited")
    print(json.dumps(save(a.out,build()),sort_keys=True))


if __name__=="__main__":
    main()
