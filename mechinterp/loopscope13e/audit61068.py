"""Independent NumPy-only audit of LGA-10E's original fresh 61068 capture.

No model, Torch, proposal search or precision-oracle import. Checks the
frozen per-edit LGA-10C native physical and scalar-resolution conjunction.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from mechinterp.loopscope13d.capture import open_capture
from mechinterp.loopscope13c.audit_receipt import audit_native

SCHEMA = "LGA10E/61068-independent-numpy-audit-v1"
ARMS=("frozen_baseline","fp32_sham_filter","mixed_donor_repair")
GOALS=(0,10,20,30,40,50,60,70)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(receipt,raw,out=None,*,expected_seed=61068):
    receipt,raw=Path(receipt),Path(raw)
    obj=json.loads(receipt.read_text(encoding="utf-8"))
    failures=[]
    def check(cond,msg):
        if not cond:
            failures.append(msg)
    check(obj.get("schema")=="LGA10E/61068-native-fp32-prospective-v1",
          "unexpected source receipt schema")
    check(obj.get("excluded_model_seed")==expected_seed
          and obj.get("excluded_source_seed")==expected_seed,
          "source or model seed mismatch")
    check(obj.get("n_cases")==64 and len(obj["cases"])==64
          and obj.get("n_independent_bases")==8 and obj.get("n_goals_per_base")==8,
          "source denominator mismatch")
    check(obj.get("registered_threshold")==61
          and obj.get("n_model_initializations")==1
          and obj.get("newly_trained_model_roots")==0
          and not obj.get("model_training")
          and not obj.get("task_correctness_scored"),
          "source scope or frozen threshold changed")
    check(obj.get("protected_science_seeds_unopened")==[31059,31064,31065],
          "science seed roster changed")
    check(obj.get("raw_sha256")==sha(raw),"raw SHA256 mismatch")
    totals=Counter()
    per_base={name:[0]*8 for name in ARMS}
    rows=[]
    with open_capture(raw,obj["capture"]) as cap:
        check(cap["head_weight"].shape==(10,48) and cap["head_projector"].shape==(48,48),
              "head physical dimensions differ")
        check(cap["source_inputs"].shape==(8,81) and cap["source_labels"].shape==(8,81),
              "source input or label shapes differ")
        head=np.asarray(cap["head_weight"])
        hp=np.asarray(cap["head_projector"])
        for idx,case in enumerate(obj["cases"]):
            b,g=divmod(idx,8)
            g=GOALS[g]
            prefix=f"case_{b:02}_{g:02}__"
            check(case["base"]==b and case["goal"]==g,"case order differs")
            h32=np.asarray(cap[f"base_{b:02}__hrec_fp32"])
            h64=np.asarray(cap[f"base_{b:02}__hrec_fp64"])
            check(h32.shape==(1,81,48) and h32.dtype==np.float32
                  and h64.shape==h32.shape and h64.dtype==np.float64,
                  "source reached state differs")
            if case["source_eligible"]:
                original=np.asarray(cap["source_inputs"][b])
                donor=np.asarray(cap[prefix+"common__source_input"])
                predecessor=(g-1)%81
                expected=original.copy()
                expected[predecessor]=expected[predecessor]%9+1
                check(np.array_equal(donor,expected),"legal donor source mutation differs")
                natural=np.asarray(cap[prefix+"common__natural_selected"],dtype=np.float64).reshape(96)
                hvp=np.asarray(cap[prefix+"common__target_hvp64"],dtype=np.float64).reshape(96)
                target=float(natural@hvp)
                check(np.isclose(target,case["target_curvature_original"],rtol=0,atol=1e-12),
                      "common fresh target HVP disagrees")
                rho=float(np.linalg.norm(natural))
                check(np.isclose(rho,case["natural_radius"],rtol=1e-8,atol=1e-11),
                      "common donor natural radius differs")
                if case.get("geometry_eligible") and (
                    prefix+"common__natural_k" in cap):
                    k=np.asarray(cap[prefix+"common__natural_k"],dtype=np.float64)
                    J=np.asarray(cap[prefix+"common__natural_jacobian64"],dtype=np.float64)
                    Jref=np.asarray(cap[prefix+"common__reference_jacobian64"],dtype=np.float64)
                    check(k.shape==(96,96) and J.shape==(10,96)
                          and Jref.shape==(10,96),"frozen Jacobian/K shape differs")
                else:
                    k=J=Jref=None
            else:
                natural=None
                target=None
                k=J=Jref=None
                check(all(not case["arms"][name]["native_fp32_ready"] for name in ARMS),
                      "source failure incorrectly passed")
            for name in ARMS:
                arm=case["arms"][name]
                if not case["source_eligible"] or arm["n_shams"]<0:
                    check(not arm["native_fp32_ready"],"ineligible source passed")
                if arm["n_shams"]==0 and prefix+name+"__selected" not in cap:
                    check(not arm["complete"] and not arm["native_fp32_ready"],
                          "missing arm has a passing endpoint")
                    continue
                selected=np.asarray(cap[prefix+name+"__selected"],dtype=np.float64).reshape(96)
                shams=np.asarray(cap[prefix+name+"__shams"],dtype=np.float64)
                check(shams.shape==(arm["n_shams"],96),"arm sham inventory differs")
                check(arm["n_shams"]<=8,"excess sham count")
                if natural is not None and J is not None:
                    deviation=float(np.linalg.norm(selected-natural))
                    limit=0.2*float(np.linalg.norm(natural))
                    check(deviation <= limit+1e-10,
                          "donor source deviation exceeds registered bound")
                    check(abs(np.linalg.norm(selected)-np.linalg.norm(natural)) <=
                          1e-7*max(1.0,np.linalg.norm(natural))+1e-12,
                          "donor natural dose changed")
                    check(np.linalg.norm(k@selected-selected) <=
                          1e-7*max(1.,np.linalg.norm(natural))+1e-10,
                          "donor head/energy tangent changed")
                    check(np.linalg.norm(J@(selected-natural)) <=
                          1e-8*max(1.,np.linalg.norm(J@natural))+1e-10,
                          "donor first-order margin changed")
                    if name in ("frozen_baseline","fp32_sham_filter"):
                        check(np.array_equal(selected,natural),
                              "fixed donor arm selected a changed source")
                if not arm["complete"]:
                    check(not arm["native_fp32_ready"],"partial sham plan passed")
                    continue
                check(arm["n_shams"]==8 and shams.shape==(8,96)
                      and arm["source_preserved"] and arm.get("selection_eligible"),
                      "complete arm selected without complete source plan")
                check("native" in arm,"complete arm omitted native endpoint")
                get=lambda key,p=prefix,name=name: cap[p+name+"__native__"+key]
                intended=np.concatenate([selected.reshape(1,96),shams],axis=0)
                if target is None or Jref is None:
                    check(False,"complete arm lacks common source target")
                    continue
                decoded=audit_native(arm["native"],get,check,target=target,
                                     intended=intended,h32=h32,h64=h64,
                                     reference_j=Jref,head_projector=hp,head=head)
                result=bool(decoded["native_ready"] and arm["source_preserved"]
                           and arm["complete"] and arm["selection_eligible"])
                check(result==arm["native_fp32_ready"],
                      f"native arm verdict differs: {b} {g} {name}")
                if name=="mixed_donor_repair" and arm["native_fp32_ready"]:
                    q64donor=float(arm["native"]["arms"][0]["q64_actual"])
                    tau=1e-8*max(1.,abs(target))
                    check(abs(q64donor-target)<=tau,
                          "repaired donor changed original curvature target")
                    chosen=case.get("repaired_donor_index")
                    selected_attempts=[a for a in case["donor_attempts"]
                                       if a.get("index")==chosen]
                    check(len(selected_attempts)==1 and selected_attempts[0]["passed"],
                          "repaired donor lacks selected-source witness")
                if result:
                    totals[name]+=1
                    per_base[name][b]+=1
                rows.append({"base":b,"goal":g,"arm":name,"ready":result,
                             "all_gates":decoded,"n_shams":arm["n_shams"]})
    for name in ARMS:
        expected=obj["arms"][name]
        check(expected["n_native_fp32_ready"]==totals[name],
              name+" summary count differs")
        check(expected["per_base"]==per_base[name],
              name+" per-base totals differ")
        check(expected["status"]==("PASS" if totals[name]>=61 else "FAIL"),
              name+" registered status differs")
    result={"schema":SCHEMA,"n_cases":64,"n_completed_arm_cases":len(rows),
            "status":"PASS" if not failures else "INTEGRITY_FAIL",
            "audited_counts":dict(totals),"audited_per_base":per_base,
            "n_failures":len(failures),"failures":failures,
            "torch_model_matcher_or_precision_oracle_imported":False,
            "preserved_raw_sha256":obj.get("raw_sha256"),
            "source_seed":expected_seed,
            "primary_repaired_endpoint":
                "PASS" if totals["mixed_donor_repair"]>=61 else "FAIL"}
    if out is not None:
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    if failures:
        raise AssertionError("Independent LGA-10E audit failures: "+
                             "; ".join(failures[:12]))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print("LGA10E_AUDIT",json.dumps(audit(a.receipt,a.raw,a.out),sort_keys=True))
