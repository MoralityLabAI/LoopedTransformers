"""Independent NumPy-only audit of LGA-10G's original fresh 61070 capture.

No model, Torch, proposal search or precision-oracle import. Checks the
frozen per-edit LGA-10C native physical and scalar-resolution conjunction.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from mechinterp.loopscope13d.capture import open_capture
from mechinterp.loopscope13c.audit_receipt import audit_native

SCHEMA = "LGA10G/61070-independent-numpy-audit-v1"
ARMS=("frozen_baseline","fixed_donor_oracle_filter","levelset_repair","levelset_sham_repair")
GOALS=(0,10,20,30,40,50,60,70)



def check_levelset_donors(case, cap, prefix, natural, target, h32, K, J, R, H, check):
    """Recalculate every archived donor's source geometry, Q and HVP receipts."""
    attempts=case.get("donor_attempts",[])
    if not attempts:
        check(case.get("repaired_donor_index") is None,
              "selected donor missing attempt receipts")
        return 0
    tau=1e-8*max(1.,abs(target))
    q0=float(natural@H@natural)
    rho=float(np.linalg.norm(natural))
    pair=h32[0,[(case["goal"]-1)%81,case["goal"]]].reshape(96)
    repaired=0
    for order,entry in enumerate(attempts):
        idx=entry["index"]
        check(idx==order and 0<=idx<=32,"donor attempts out of fixed order")
        if idx==0:
            check(entry["kind"]=="natural","first donor must be natural source")
        else:
            j,k=divmod(idx-1,8)
            check(entry["kind"]=="levelset" and entry["direction"]==j
                  and entry["s"]==(-.12,-.06,-.03,-.015,.015,.03,.06,.12)[k],
                  "prospective level-set candidate grid changed")
        key=prefix+f"repair__attempt_{idx:02d}__"
        exists=key+"intended" in cap
        if entry.get("root_reason") is not None:
            check(not exists and not entry["passed"],
                  "failed root has a physical edit or admitted donor")
            continue
        check(exists,"finite root candidate missing raw source vector")
        if not exists:
            continue
        z=np.asarray(cap[key+"intended"],dtype=np.float64).reshape(96)
        check(np.isfinite(z).all(),"nonfinite donor candidate")
        if idx==0:
            check(np.array_equal(z,natural),"source candidate zero changed")
        ratio=float(np.linalg.norm(z-natural))/max(rho,1e-30)
        dose=abs(float(np.linalg.norm(z))-rho)
        Kdiff=float(np.linalg.norm(K@z-z))
        Jdiff=float(np.linalg.norm(J@(z-natural)))
        levelerr=abs(float(z@H@z)-q0)
        for name,measured in (("relative_distance",ratio),("dose_error",dose),
                              ("k_error",Kdiff),("signed_j_error",Jdiff),
                              ("intended_levelset_error",levelerr)):
            check(np.isclose(measured,entry[name],rtol=1e-7,atol=1e-10),
                  "donor geometrical metric altered: "+name)
        if idx>0:
            check(levelerr<=1e-11*max(1.,abs(q0))+1e-13,
                  "off-curvature-surface donor represented as root")
            q=z-R@(R.T@z)
            qorig=natural-R@(R.T@natural)
            check(abs(np.linalg.norm(q)-np.linalg.norm(qorig)) <=
                  1e-8*max(1.,np.linalg.norm(qorig)),
                  "donor dark radius changed on solved sphere")
        valid=(ratio<=.2+1e-12 and
               dose<=1e-7*max(1.,rho)+1e-12 and
               Kdiff<=1e-7*max(1.,rho)+1e-12 and
               Jdiff<=1e-8*max(1.,np.linalg.norm(J@natural))+1e-12 and
               levelerr<=1e-11*max(1.,abs(q0))+1e-13)
        if not valid:
            check(not entry["passed"],"source-invalid donor passed")
            continue
        after=(pair+z.astype(np.float32)).astype(np.float32)
        actual=after.astype(np.float64)-pair.astype(np.float64)
        physicalerr=abs(float(actual@H@actual)-target)
        check(key+"actual" in cap and key+"probe32" in cap,
              "valid donor lacks FP32 physical source receipts")
        if key+"actual" not in cap:
            continue
        check(np.array_equal(cap[key+"actual"],actual)
              and np.array_equal(cap[key+"probe32"],actual.astype(np.float32)),
              "physical donor addition or probe differs")
        check(np.isclose(entry["physical_dense_curvature_error"],physicalerr,
                         rtol=1e-7,atol=1e-12),
              "physical dense curvature drift differs")
        if physicalerr>tau or not np.linalg.norm(actual)>0:
            check(not entry["passed"],"curvature-ineligible donor passed")
            continue
        if "q32_probe" not in entry:
            check(not entry["passed"],"donor without HVP oracle admitted")
            continue
        h32p=np.asarray(cap[key+"hvp32"],dtype=np.float64).reshape(96)
        h64p=np.asarray(cap[key+"hvp64_probe"],dtype=np.float64).reshape(96)
        h64a=np.asarray(cap[key+"hvp64_actual"],dtype=np.float64).reshape(96)
        p=actual.astype(np.float32).astype(np.float64)
        q32p=float(p@h32p)
        q64p=float(p@h64p)
        q64a=float(actual@h64a)
        for name,value in (("q32_probe",q32p),("q64_probe",q64p),
                           ("q64_actual",q64a),
                           ("probe_error",abs(q32p-q64p)),
                           ("target_error",abs(q64a-target)),
                           ("hvp_vector_error",float(np.linalg.norm(h32p-h64p)))):
            check(np.isclose(value,entry[name],rtol=1e-7,atol=1e-12),
                  "raw donor HVP mismatch: "+name)
        passed=(abs(q32p-q64p)<=tau and abs(q64a-target)<=tau
                and np.linalg.norm(h32p-h64p)<=
                    1e-6*max(1.,np.linalg.norm(h64p)))
        check(passed==entry["passed"],"donor HVP gate changed")
    admitted=[x for x in attempts if x.get("passed")]
    check(len(admitted)<=1 and (not admitted or admitted[-1]==attempts[-1]),
          "first-admissible donor policy broken")
    selected=case.get("repaired_donor_index")
    check((len(admitted)==1)==(selected is not None),
          "selected donor index missing or inconsistent")
    if admitted:
        check(admitted[0]["index"]==selected,
              "first accepted donor index differs")
        repaired=int(selected>0)
        for name in ("levelset_repair","levelset_sham_repair"):
            if prefix+name+"__selected" in cap:
                check(np.array_equal(
                    np.asarray(cap[prefix+name+"__selected"]).reshape(96),
                    np.asarray(cap[prefix+f"repair__attempt_{selected:02d}__intended"])),
                    "repaired donor differs from selected attempt: "+name)
    return repaired



SHAM_OFFSETS=(-.010,-.005,-.0025,.0025,.005,.010)

def correction_seed(*fields):
    text="LGA10G|"+"|".join(map(str,fields))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],"big")&((1<<63)-1)


def audit_sham_repairs(case,cap,prefix,h32,head,check):
    """Rebuild every saved local-sham root and actual edit using only NumPy.

    Parent proposal streams are taken from the first frozen 7×32 search.
    Orthogonal correction axes come from the raw NPY archive and are checked
    against their registered seed receipt, source sphere and dense curvature.
    """
    name="levelset_sham_repair"
    arm=case["arms"][name]
    if prefix+name+"__selected" not in cap:
        check(not arm["n_rescued_shams"] and not arm["n_rescue_trials"],
              "arm without source receipt contains sham corrections")
        return 0
    plan=arm.get("selection_receipt",{})
    records=plan.get("rescue_attempts",[])
    check(len(records)==arm["n_rescue_trials"],
          "correction trial count differs")
    geometry=prefix+name+"__repair_geometry__"
    for key in ("z0","q0","complement_basis","row_basis"):
        check(geometry+key in cap,"local curvature fibre missing: "+key)
    if any(geometry+k not in cap for k in ("z0","q0","complement_basis","row_basis")):
        return 0
    z0=np.asarray(cap[geometry+"z0"],dtype=np.float64).reshape(96)
    q0=np.asarray(cap[geometry+"q0"],dtype=np.float64).reshape(96)
    V=np.asarray(cap[geometry+"complement_basis"],dtype=np.float64)
    R=np.asarray(cap[geometry+"row_basis"],dtype=np.float64)
    H=np.asarray(cap[prefix+"common__natural_hessian"],dtype=np.float64)
    selected=np.asarray(cap[prefix+name+"__selected"],dtype=np.float64).reshape(96)
    check(V.shape==(96,64) and R.shape[0]==96 and H.shape==(96,96),
          "local correction geometry shape differs")
    if V.shape!=(96,64):
        return 0
    check(np.linalg.norm(V.T@V-np.eye(64))<=1e-7,
          "dark complement orthonormality failed")
    check(np.linalg.norm(R.T@V)<=1e-7,
          "dark complement overlaps full-J rowspace")
    check(np.linalg.norm(z0-R@(R.T@selected))<=1e-8,
          "sham donor rowspace changed")
    check(np.linalg.norm(q0-(selected-z0))<=1e-8,
          "sham donor dark source changed")
    r=float(np.linalg.norm(q0))
    target=float(plan["selected_curvature"])
    tau=1e-8*max(1.,abs(target))
    original=np.asarray(cap[prefix+name+"__selection__candidate_intended"],
                        dtype=np.float64)
    roots=np.asarray(cap[prefix+name+"__selection__rescue_root_vectors"],
                     dtype=np.float64)
    axes=np.asarray(cap[prefix+name+"__selection__rescue_root_axes"],
                    dtype=np.float64)
    measured=np.asarray(cap[prefix+name+"__selection__rescue_candidate_intended"],
                        dtype=np.float64)
    actuals=np.asarray(cap[prefix+name+"__selection__rescue_candidate_actual"],
                       dtype=np.float64)
    dense_hvp=np.asarray(cap[prefix+name+"__selection__rescue_candidate_hvp_dense"],
                         dtype=np.float64)
    check(roots.shape==(len(axes),96) and axes.shape[1:]==(64,),
          "local root vector/axis receipt count differs")
    check(measured.shape==(len(actuals),96)
          and measured.shape==dense_hvp.shape,
          "local physical candidate measurements differ")
    originals=plan.get("attempts",[])
    case_count={}
    n_rescued=0
    accepted_shams=np.asarray(cap[prefix+name+"__shams"],dtype=np.float64)
    donor=selected
    scalars=arm.get("filter_donor_scalars") or {}
    original_scores={}
    for record in originals:
        idx=record.get("candidate_index")
        phys=record.get("physical")
        if idx is None or not isinstance(phys,dict) or phys.get("passed"):
            continue
        needed=("trial_qnative","trial_q64_actual","trial_q32_probe","trial_q64_probe")
        if any(k not in phys for k in needed):
            continue
        dn=scalars.get("qnative");dp=scalars.get("qpromoted");t=scalars.get("q_tolerance")
        if dn is None or dp is None or t is None or t<=0:
            continue
        qn=float(phys["trial_qnative"]);qp=float(phys["trial_q64_actual"])
        q32=float(phys["trial_q32_probe"]);q64=float(phys["trial_q64_probe"])
        score=max(abs(qn-dn),abs(qp-dp),
                  abs((qn-dn)-(qp-dp)),abs(q32-q64))/t
        original_scores[(record["sham_index"],idx)]=score

    per_sham={}
    seen_accept={}
    seed_id=int(case.get("source_seed",61070))
    b,g=case["base"],case["goal"]
    for rec in records:
        k=int(rec["sham_index"])
        per_sham[k]=per_sham.get(k,0)+1
        check(0<=k<8 and per_sham[k]<=36,
              "local correction exceeded registered per-sham budget")
        rank=int(rec["parent_rank"])
        parent_index=int(rec["parent_candidate_index"])
        j=int(rec["j"])
        signed_offset=float(rec["s"])
        check(0<=rank<3 and j in (0,1)
              and signed_offset in SHAM_OFFSETS,
              "local correction grid changed")
        expected_seed=correction_seed(seed_id,b,g,k,parent_index,j)
        check(rec.get("seed")==expected_seed,
              "local correction deterministic seed receipt differs")
        check((k,parent_index) in original_scores,
              "corrected candidate was not rejected original parent")
        if (k,parent_index) in original_scores:
            check(np.isclose(rec["parent_score"],
                  original_scores[(k,parent_index)],rtol=1e-7,atol=1e-8),
                  "original parent rank score differs")
        check(0<=parent_index<len(original),"parent original candidate index missing")
        if not 0<=parent_index<len(original):
            continue
        parent=original[parent_index]
        root_index=rec.get("root_vector_index")
        if root_index is None:
            check(rec.get("candidate_index") is None and not rec["accepted"],
                  "failed local root admitted or consumed candidate index")
            continue
        check(0<=root_index<len(roots),
              "finite local root has invalid archive index")
        if not 0<=root_index<len(roots):
            continue
        root=roots[root_index]
        w=axes[root_index]
        check(abs(np.linalg.norm(w)-1.)<=1e-8,
              "correction tangent axis is not normalized")
        cosine=float((parent-z0)@q0)/(r*r)
        scale=r*math.sqrt(max(0.,1.-cosine*cosine))
        a=z0+cosine*q0
        y=V.T@(parent-a)
        y=y/np.linalg.norm(y)
        gradient=V.T@(H@(a+scale*(V@y)))
        e=gradient-y*float(y@gradient)
        e=e/np.linalg.norm(e)
        check(abs(w@y)<=1e-8 and abs(w@e)<=1e-8,
              "correction axis leaves two-plane source tangent")
        t=rec.get("t")
        check(isinstance(t,(int,float)) and abs(float(t))<=.08000000001,
              "correction t is out of precommitted bracket")
        if t is None:
            continue
        unit=(y+signed_offset*w+t*e)/math.sqrt(
            1.+signed_offset**2+t**2)
        reconstructed=a+scale*(V@unit)
        check(np.allclose(root,reconstructed,rtol=1e-9,atol=1e-9),
              "local correction reconstructed root differs")
        Q=float(root@H@root)
        check(abs(Q-target)<=1e-10*max(1.,abs(target))+1e-12,
              "local corrected source leaves curvature level")
        check(np.isclose(rec["root_residual"] if "root_residual" in rec else
                         rec.get("residual",abs(Q-target)),
                         abs(Q-target),atol=1e-11,rtol=1e-7),
              "root residual receipt differs")
        m=rec.get("candidate_index")
        if m is None:
            check(not rec["accepted"],"uncounted correction was accepted")
            continue
        check(0<=m<len(measured),"physical correction index invalid")
        if not 0<=m<len(measured):
            continue
        check(np.array_equal(root,measured[m]),
              "physical measured correction differs from numerical root")
        pair=h32[0,[(g-1)%81,g]].reshape(96)
        after=(pair+root.astype(np.float32)).astype(np.float32)
        delta=after.astype(np.float64)-pair.astype(np.float64)
        check(np.array_equal(actuals[m],delta),
              "corrected sham physical FP32 addition differs")
        check(np.allclose(dense_hvp[m],H@delta,rtol=1e-10,atol=1e-10),
              "correction dense HVP receipt differs")
        check(bool(rec["accepted"])==bool(rec["physical"]["passed"]),
              "correction accept flag disagrees with actual gate")
        if rec["accepted"]:
            check(k<len(accepted_shams) and np.array_equal(root,accepted_shams[k]),
                  "successful correction missing from selected native sham list")
            check(k not in seen_accept,"same sham corrected twice")
            seen_accept[k]=m
            n_rescued+=1
    check(n_rescued==arm["n_rescued_shams"]
          and n_rescued==plan.get("n_rescued_shams",0),
          "corrected-sham tally changed")
    return n_rescued


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(receipt,raw,out=None,*,expected_seed=61070):
    receipt,raw=Path(receipt),Path(raw)
    obj=json.loads(receipt.read_text(encoding="utf-8"))
    failures=[]
    def check(cond,msg):
        if not cond:
            failures.append(msg)
    check(obj.get("schema")=="LGA10G/61070-native-fp32-prospective-v1",
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
    repaired_donors=0
    selected_corrected_shams=0
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
                    R=np.asarray(cap[prefix+"common__natural_row_basis"],dtype=np.float64)
                    H=np.asarray(cap[prefix+"common__natural_hessian"],dtype=np.float64)
                    V=np.asarray(cap[prefix+"common__natural_complement"],dtype=np.float64)
                    check(H.shape==(96,96) and V.shape==(96,64)
                          and np.linalg.norm(H-H.T)<=1e-8*max(1.,np.linalg.norm(H)),
                          "recorded native-state Hessian invalid")
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
            if case["source_eligible"] and J is not None and (
                    prefix+"common__natural_hessian" in cap):
                repaired_donors+=check_levelset_donors(
                    case,cap,prefix,natural,target,h32,k,J,R,H,check)
            if case["source_eligible"] and prefix+"levelset_sham_repair__selected" in cap:
                selected_corrected_shams+=audit_sham_repairs(
                    case,cap,prefix,h32,head,check)
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
                    if name in ("frozen_baseline","fixed_donor_oracle_filter"):
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
                if name=="levelset_repair" and arm["native_fp32_ready"]:
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
    result={"schema":SCHEMA,"n_cases":64,
            "n_repaired_donor_candidates_admitted":repaired_donors,
            "n_local_corrected_shams_selected":selected_corrected_shams,
            "n_completed_arm_cases":len(rows),
            "status":"PASS" if not failures else "INTEGRITY_FAIL",
            "audited_counts":dict(totals),"audited_per_base":per_base,
            "n_failures":len(failures),"failures":failures,
            "torch_model_matcher_or_precision_oracle_imported":False,
            "preserved_raw_sha256":obj.get("raw_sha256"),
            "source_seed":expected_seed,
            "primary_repaired_endpoint":
                "PASS" if totals["levelset_sham_repair"]>=61 else "FAIL"}
    if out is not None:
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    if failures:
        raise AssertionError("Independent LGA-10G audit failures: "+
                             "; ".join(failures[:12]))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print("LGA10G_AUDIT",json.dumps(audit(a.receipt,a.raw,a.out),sort_keys=True))
