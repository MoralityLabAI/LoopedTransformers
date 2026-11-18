"""LGA-05 independent standard-library source/trajectory/native receipt audit.

Checks first LGA-02 teacher bytes, independent calibration and new cyclic
inputs, raw fixed-potential before/after energies, all 256 whole-sequence
bootstrap draws, native full 10-logit CE identities and curl witnesses.
Does not rerun the frozen neural forward or second-order potential training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct

SCHEMA="LGA-05/native-recurrent-energy-and-causal-fresh-test-v1"
CAL_SCHEMA="LGA-05/calibration-only-recurrent-energy-potentials-v1"
SOURCE_SHA="d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266"
SOURCE_META_SHA="7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d"
SEEDS=(103,211,307)
DEPTHS=(0,1,2,3)
CAUSAL_DEPTHS=(1,2)
N_TEST=256
CAUSAL_N=64
GOALS=(0,10,20,30,40,50,60,70)
ARMS=("shared_down","conditioned_down","initial_down","radial_down",
      "native","sae_active","sae_remainder","head_null","random")
CONTRASTS=(("shared_down","random"),("shared_down","conditioned_down"),
           ("shared_down","native"),("sae_active","sae_remainder"),
           ("head_null","random"))
BOOT=2000


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def protocol_sha():
    return digest((Path(__file__).resolve().parent/"PROTOCOL.md").read_bytes())


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(("LGA05|"+"|".join(map(str,parts))).encode()).digest()[:8],"big")&((1<<63)-1)


def bootstrap(vals,root):
    n=len(vals)
    if n<2 or any(not math.isfinite(float(x)) for x in vals):
        raise ValueError("source sequence cluster data nonfinite")
    rng=random.Random(root)
    draws=sorted(sum(vals[rng.randrange(n)] for _ in range(n))/n for _ in range(BOOT))
    return [draws[int(.025*BOOT)],draws[int(.975*BOOT)-1]]


def close(a,b,where,tol=2e-6):
    a=float(a);b=float(b)
    if not(math.isfinite(a) and math.isfinite(b)):
        raise ValueError("nonfinite original source result "+where)
    if abs(a-b)>tol*max(1.,abs(a),abs(b)):
        raise ValueError("independent original source arithmetic mismatch "+where)


def sha_tensor(rows):
    flat=[int(z) for row in rows for z in row]
    return digest(struct.pack("<"+"q"*len(flat),*flat))


def cyclic(roster,n):
    x,y=roster["x"],roster["y"]
    if len(x)!=n or len(y)!=n:
        raise ValueError("source sequence population changed")
    vals=[tuple(row) for row in x]
    if len(set(vals))!=n:
        raise ValueError("source 81-token sample duplicated")
    for a,b in zip(x,y):
        if (len(a)!=81 or len(b)!=81 or
            any(not 1<=int(z)<=9 for z in a) or b!=a[-1:]+a[:-1]):
            raise ValueError("source cyclic-copy answer changed")
    if sha_tensor(x)!=roster["x_sha256"] or sha_tensor(y)!=roster["y_sha256"]:
        raise ValueError("original ordered little-endian int64 source hash drift")
    return set(vals)


def validate_sources(source_raw,source_meta_raw,potential_raw,potential_meta_raw,test_raw):
    source=json.loads(source_meta_raw)
    calibration=json.loads(potential_meta_raw)
    result=json.loads(test_raw)
    if digest(source_raw)!=SOURCE_SHA or digest(source_meta_raw)!=SOURCE_META_SHA:
        raise ValueError("original first LGA-02 teacher/SAE weights changed")
    if (calibration["schema"]!=CAL_SCHEMA or result["schema"]!=SCHEMA or
        calibration["protocol_sha256"]!=protocol_sha() or
        result["protocol_sha256"]!=protocol_sha() or
        calibration["mode"]!="full" or result["mode"]!="full"):
        raise ValueError("registered LGA-05 source schema/protocol changed")
    if (calibration["original_teacher_source_bundle_sha256"]!=SOURCE_SHA or
        calibration["original_teacher_source_manifest_sha256"]!=SOURCE_META_SHA or
        calibration["source_potential_bundle_sha256"]!=digest(potential_raw) or
        result["original_source_bundle_sha256"]!=SOURCE_SHA or
        result["original_source_manifest_sha256"]!=SOURCE_META_SHA or
        result["frozen_potential_bundle_sha256"]!=digest(potential_raw) or
        result["frozen_potential_manifest_sha256"]!=digest(potential_meta_raw)):
        raise ValueError("source-frozen trainer or potential weights replaced")
    if (calibration["source_fit_steps"]!=200 or
        calibration["teacher_seeds"]!=list(SEEDS) or
        result["teacher_seeds"]!=list(SEEDS) or
        calibration["paired_arms"]!=["shared","conditioned"] or
        calibration["test_labels_read"] is not False or
        calibration["calibration_splits"]["new_test_generated"] is not False or
        calibration["calibration_splits"]["new_test_labels_available"] is not False or
        calibration["calibration_splits"]["new_test_seed_reserved"]!=31057):
        raise ValueError("new test leaked to source-only potential fitting")
    prior=set()
    split=calibration["calibration_splits"]
    for name,sd,n in (("train",31055,256),("validation",31056,64)):
        row=split[name]
        if row["seed"]!=sd or row["n"]!=n:
            raise ValueError("calibration registered seed/size changed")
        observed=cyclic(row,n)
        if observed&prior:
            raise ValueError("paired old train/validation source overlap")
        prior.update(observed)
    for sid in SEEDS:
        receipt=calibration["teacher_potential_receipts"][str(sid)]
        if (len(receipt["initial_sha256"])!=64 or
            len(receipt["normalizer_sha256"])!=64 or
            len(receipt["train_state_sha256"])!=64 or
            len(receipt["validation_state_sha256"])!=64 or
            len(receipt["native_field_train_sha256"])!=64 or
            receipt["native_RMS"]<=0):
            raise ValueError("train-only native field checkpoint/normalizer SHA missing")
        arms=receipt["arms"]
        if set(arms)!={"shared","conditioned"}:
            raise ValueError("missing source fitting control arm")
        if arms["shared"]["same_initial_weights_sha256"]!=arms["conditioned"]["same_initial_weights_sha256"]:
            raise ValueError("source potential arms began from different weights")
        if arms["shared"]["same_initial_weights_sha256"]!=receipt["initial_sha256"]:
            raise ValueError("source initial null potential changed")
        for name in ("shared","conditioned"):
            item=arms[name]
            if (item["optimizer_steps"]!=200 or
                [a["update"] for a in item["validation_grid"]]!=[0,100,200] or
                len(item["trained_weights_sha256"])!=64):
                raise ValueError("source training budget or fixed checkpoint list changed")
    test=result["new_test"]
    if (test["seed"]!=31057 or test["n_sequences"]!=N_TEST or
        result["n_independent_new_sequences"]!=N_TEST or
        test["independent_source_no_overlap"] is not True):
        raise ValueError("new independent 31057 sequence cohort differs")
    new={"x":result["test_source"]["x"],"y":result["test_source"]["y"],
         "x_sha256":test["x_sha256"],"y_sha256":test["y_sha256"]}
    seen=cyclic(new,N_TEST)
    if seen&prior:
        raise ValueError("new test reused calibrator source")
    if set(result["results"])!={str(k) for k in SEEDS}:
        raise ValueError("registered three original teacher seeds missing")
    return calibration,result


def validate_profiles(teacher,sid):
    rows=teacher["trajectory_by_depth"]
    if set(rows)!={str(k) for k in DEPTHS}:
        raise ValueError("original four native transition depths not present")
    passes=0
    for k in DEPTHS:
        item=rows[str(k)]
        cases=item["per_sequence"]
        if item["n_independent_sequences"]!=N_TEST or len(cases)!=N_TEST:
            raise ValueError("per-token pseudoreplication or missing whole sequences")
        means={field:sum(x[field] for x in cases)/N_TEST for field in cases[0]}
        for key,val in means.items():
            close(val,item["means"][key],f"trajectory depth{k} {key}",tol=1e-8)
        for source in cases:
            close(source["gain_shared_over_initial"],
                  source["shared_cosine"]-source["initial_cosine"],
                  "same-source cosine vs initial")
            close(source["gain_shared_over_radial"],
                  source["shared_cosine"]-source["radial_cosine"],
                  "same-source cosine vs radial")
            for name,stem in (("shared","shared"),("conditioned","conditioned"),
                              ("initial","initial"),("radial","radial")):
                close(source[name+"_energy_drop"],
                      (source[name+"_energy_before"]-
                       source[name+"_energy_after"])/81,
                      "source fixed scalar energy decrease")
            if source["shared_positive_drop"]!=float(source["shared_energy_drop"]>0):
                raise ValueError("reported source energy monotonicity changed")
            if source["conditioned_positive_drop"]!=float(source["conditioned_energy_drop"]>0):
                raise ValueError("reported conditioned same-horizon energy sign changed")
            for field in ("shared_cosine","conditioned_cosine","initial_cosine",
                          "radial_cosine","native_terminal_gradient_cosine",
                          "native_current_head_gradient_cosine"):
                if abs(source[field])>1.00001:
                    raise ValueError("non-normalized gradient/physical vector cosine")
        okay=True
        for key in ("gain_shared_over_initial","gain_shared_over_radial",
                    "shared_energy_drop","shared_positive_drop"):
            dist=[x[key] for x in cases]
            interval=bootstrap(dist,stable_seed(45057,sid,k,key))
            for left,right in zip(interval,item["bootstrap95"][key]):
                close(left,right,"registered 2000-draw source whole-sequence interval",tol=1e-9)
            okay=okay and interval[0]>(.50 if key=="shared_positive_drop" else 0.)
        if bool(okay)!=item["registered_depth_pass"]:
            raise ValueError("preregistered energy-gradient depth gate changed")
        passes+=int(okay)
    full=sum(int(all(rows[str(k)]["per_sequence"][i]["shared_positive_drop"]==1. for k in DEPTHS))
             for i in range(N_TEST))/N_TEST
    close(full,teacher["fraction_all_four_energy_decrease"],"all-depth same-potential monotonicity")
    if (passes!=teacher["n_positive_depths"] or
        bool(passes>=3)!=teacher["registered_teacher_pass"]):
        raise ValueError("source original learned-conservative-field classification changed")
    return passes


def logsumexp(z):
    m=max(z)
    return m+math.log(sum(math.exp(a-m) for a in z))


def validate_native(teacher,sid):
    original=teacher["native_causal"]
    if set(original)!={"1","2"}:
        raise ValueError("native intervention two depths missing")
    out={};tested=0
    for k in (1,2):
        cases=original[str(k)]["raw_cases"]
        cell=original[str(k)]
        if len(cases)!=64 or cell["n_independent_source_sequences"]!=64 or cell["n_original_goals"]!=512:
            raise ValueError("original 64-sequence native source sample changed")
        clusters=[];n_admitted=0
        for i,row in enumerate(cases):
            if row["test_sequence_index"]!=i or len(row["goals"])!=8:
                raise ValueError("original eight input-only native goal positions absent")
            admitted=[]
            for j,goal in enumerate(row["goals"]):
                if goal["slot"]!=j or goal["token"]!=GOALS[j] or goal["dose"]<0:
                    raise ValueError("original fixed source goal admission changed")
                if not goal["source_admitted"]:
                    if goal["arms"] or "reason" not in goal:
                        raise ValueError("ineligible source goal altered")
                    continue
                n_admitted+=1
                if not set(ARMS).issubset(set(goal["arms"])):
                    raise ValueError("original outcome-blind source edit arm missing")
                for name,z in goal["arms"].items():
                    tested+=1
                    if abs(z["edit_L2"]-goal["dose"])>1e-5 or len(z["edit_sha256"])!=64:
                        raise ValueError("native physical matched-L2-dose or edit SHA altered")
                    truth=z["truth_digit"]
                    c=z["clean_goal_logits"];logits=z["final_goal_logits"]
                    if not 1<=truth<=9 or len(c)!=10 or len(logits)!=10:
                        raise ValueError("native ten-digit outcome geometry changed")
                    rival=max((ix for ix in range(10) if ix!=truth),key=lambda ix:c[ix])
                    if rival!=z["clean_rival"]:
                        raise ValueError("clean rival was switched after native edit")
                    if len(z["all_final_token_CE"])!=81 or len(z["clean_all_final_token_CE"])!=81:
                        raise ValueError("native full-sequence CE loss not saved")
                    val=logsumexp(logits)-logits[truth]
                    clean=logsumexp(c)-c[truth]
                    close(val,z["all_final_token_CE"][goal["token"]],"native source goal CE",3e-5)
                    close(clean,z["clean_all_final_token_CE"][goal["token"]],"clean goal CE",3e-5)
                    close(val-clean,z["goal_CE_change"],"native goal loss change",3e-5)
                    close(z["goal_margin_change"],
                          logits[truth]-logits[rival]-c[truth]+c[rival],
                          "native goal clean rival margin change",3e-5)
                    global_ce=sum(z["all_final_token_CE"])/81
                    global_clean=sum(z["clean_all_final_token_CE"])/81
                    close(global_ce,z["full_sequence_CE"],"edited global task CE")
                    close(global_clean,z["clean_full_sequence_CE"],"clean global task CE")
                    close(global_ce-global_clean,z["global_CE_change"],"real remaining-loop CE")
                    close((z["shared_E_before"]-z["shared_E_after"])/81,
                          z["physical_edit_E_decrease"],"native edit same-source potential energy",2e-6)
                    if not (math.isfinite(z["true_terminal_linear_global_CE_prediction"]) and
                            math.isfinite(z["learned_shared_linear_E_prediction"])):
                        raise ValueError("source first-order sensitivity missing")
                admitted.append(goal)
            contrasts={}
            for a,b in CONTRASTS:
                tag=f"{a}_minus_{b}"
                values=[z["arms"][a]["global_CE_change"]-z["arms"][b]["global_CE_change"]
                        for z in admitted]
                contrasts[tag]=sum(values)/len(values) if values else 0.
            clusters.append({"test_sequence_index":i,
                             "n_original_goals":8,"n_admitted_goals":len(admitted),
                             "paired_contrasts":contrasts})
        if n_admitted!=cell["n_admitted_goal_cases"]:
            raise ValueError("physical admission denominator differs")
        if len(clusters)!=len(cell["per_sequence"]):
            raise ValueError("64 independent source-string causal clusters missing")
        for x,y in zip(clusters,cell["per_sequence"]):
            if x["n_admitted_goals"]!=y["n_admitted_goals"]:
                raise ValueError("source goal ineligible count changed")
            for key,val in x["paired_contrasts"].items():
                close(val,y["paired_contrasts"][key],"native paired whole-source string mean")
        for a,b in CONTRASTS:
            key=f"{a}_minus_{b}"
            vals=[r["paired_contrasts"][key] for r in clusters]
            confidence=bootstrap(vals,stable_seed("causal",45057,sid,k,key))
            registered=cell["contrasts"][key]
            close(sum(vals)/64,registered["mean"],"source native comparison mean")
            for p,q in zip(confidence,registered["bootstrap95"]):
                close(p,q,"original source-native 2000-draw paired CI",tol=1e-9)
            verdict=("POSITIVE" if confidence[0]>0 else
                     "NEGATIVE" if confidence[1]<0 else "INCONCLUSIVE")
            if verdict!=registered["nominal_direction"]:
                raise ValueError("source registered causal effect sign changed")
        out[str(k)]={"n_original_goal_cases":512,
                     "n_admitted":n_admitted,"edited_source_arms":tested}
    return out


def validate_curl(teacher):
    rows=teacher["integrability_descriptive"]
    if set(rows)!={"1","2"}:
        raise ValueError("missing two native vector-field curl planes")
    result={}
    for k in (1,2):
        cohort=rows[str(k)]
        cases=cohort["per_sequence"]
        if len(cases)!=64 or cohort["n_source_sequences"]!=64:
            raise ValueError("original source curl sample count drift")
        for i,record in enumerate(cases):
            if record["source_index"]!=i:raise ValueError("curl plane source identity changed")
            u=record["u_dot_J_w"];w=record["w_dot_J_u"]
            close(u-w,record["skew_difference"],"native Jacobian skew bilinear")
            close(abs(u-w)/(abs(u)+abs(w)+1e-8),record["skew_ratio"],"native curl ratio")
        mean=sum(z["skew_ratio"] for z in cases)/64
        close(mean,cohort["mean_abs_skew_ratio"],"native curl mean")
        result[str(k)]=mean
    return result


def audit(source_raw,source_meta_raw,potential_raw,potential_meta_raw,test_raw):
    calibration,test=validate_sources(
        source_raw,source_meta_raw,potential_raw,potential_meta_raw,test_raw)
    n_pass=0;teachers=[]
    for sid in SEEDS:
        row=test["results"][str(sid)]
        depths=validate_profiles(row,sid)
        native=validate_native(row,sid)
        curl=validate_curl(row)
        n_pass+=int(row["registered_teacher_pass"])
        teachers.append({
            "seed":sid,"n_positive_depths":depths,
            "registered_teacher_pass":row["registered_teacher_pass"],
            "whole_trajectory_monotonic_fraction":row["fraction_all_four_energy_decrease"],
            "per_depth_shared_cosines":[
                row["trajectory_by_depth"][str(k)]["means"]["shared_cosine"] for k in DEPTHS],
            "per_depth_shared_energy_drop":[
                row["trajectory_by_depth"][str(k)]["means"]["shared_energy_drop"] for k in DEPTHS],
            "native":native,"curl":curl
        })
    verdict="POSITIVE" if n_pass>=2 else "FAIL"
    if test["registered_scientific_status"]!=verdict or test["n_registered_teacher_passes"]!=n_pass:
        raise ValueError("registered study verdict replaced after test")
    return {
        "schema":"LGA-05/independent-stdlib-source-and-native-audit-v1",
        "audit_status":"PASS","scientific_status":verdict,
        "number_of_original_teacher_seeds_passed":n_pass,
        "original_teacher_sha256":digest(source_raw),
        "source_potential_bundle_sha256":digest(potential_raw),
        "original_test_json_sha256":digest(test_raw),
        "registered_protocol_sha256":protocol_sha(),
        "n_independent_new_sequences":256,
        "original_teacher_source_seeds":list(SEEDS),
        "teacher_results":teachers,
        "limitations":"Verifies original ordered sample identities, before/after scalar energies, full-logit native results, curl bilinear and 2000 whole-source bootstraps; does not independently rerun PyTorch trainer or teacher."
    }


def main():
    p=argparse.ArgumentParser()
    for name in ("source-bundle","source-meta","potential-bundle",
                 "potential-meta","test-receipt","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    result=audit(a.source_bundle.read_bytes(),a.source_meta.read_bytes(),
                 a.potential_bundle.read_bytes(),a.potential_meta.read_bytes(),
                 a.test_receipt.read_bytes())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("LGA05_INDEPENDENT_AUDIT_PASS",result["scientific_status"],
          result["number_of_original_teacher_seeds_passed"],flush=True)


if __name__=="__main__":
    main()
