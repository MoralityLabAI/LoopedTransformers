"""LGA-03: independent stdlib replay of original source/fit/fresh native receipts.

No Torch, model weights, learned value fit, new test-label fitting, or
hardware-specific numerical libraries are used by the auditor.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct

SCHEMA="LGA-03/fresh-horizon-value-gradient-and-native-causal-v1"
FIT_SCHEMA="LGA-03/calibration-horizon-value-potential-v1"
SEEDS=(103,211,307)
DEPTHS=(0,1,2,3)
N_TEST=256
N_CAUSAL=64
GOALS=(0,10,20,30,40,50,60,70)
ARMS=("native","sae_active","sae_remainder","head_null",
      "random","value_down","oracle_down")
CONTRASTS=(("value_down","random"),("value_down","native"),
           ("sae_active","sae_remainder"),("head_null","random"))
SOURCE_SHA="d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266"
SOURCE_META_SHA="7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d"
BOOT=2000


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def protocol_sha():
    return digest((Path(__file__).resolve().parent/"PROTOCOL.md").read_bytes())


def seed(*parts):
    v="LGA03|"+"|".join(str(t) for t in parts)
    return int.from_bytes(hashlib.sha256(v.encode()).digest()[:8],"big")&((1<<63)-1)


def p95(vals,root):
    rng=random.Random(root)
    n=len(vals)
    if n<2 or any(not math.isfinite(x) for x in vals):
        raise ValueError("whole-sequence bootstrap lacks finite independent samples")
    samples=sorted(sum(vals[rng.randrange(n)] for _ in range(n))/n
                   for __ in range(BOOT))
    return [samples[int(.025*BOOT)],samples[int(.975*BOOT)-1]]


def close(a,b,name="",tol=2e-6):
    x=float(a);y=float(b)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("nonfinite source receipt: "+name)
    if abs(x-y)>tol*max(1.,abs(x),abs(y)):
        raise ValueError("source receipt arithmetic mismatch: "+name)


def sha_int64(rows):
    vals=[v for row in rows for v in row]
    return digest(struct.pack("<"+"q"*len(vals),*vals))


def logsumexp(z):
    top=max(z)
    return top+math.log(sum(math.exp(x-top) for x in z))


def validate_sources(src_raw,src_meta_raw,fit_raw,fit_meta_raw,test_raw):
    original=json.loads(src_meta_raw)
    fit=json.loads(fit_meta_raw)
    test=json.loads(test_raw)
    protocol=protocol_sha()
    if (digest(src_raw)!=SOURCE_SHA or digest(src_meta_raw)!=SOURCE_META_SHA):
        raise ValueError("source original LGA02 teacher bundle or source manifest replaced")
    if (fit["schema"]!=FIT_SCHEMA or test["schema"]!=SCHEMA or
        fit["protocol_sha256"]!=protocol or test["protocol_sha256"]!=protocol):
        raise ValueError("source prospective registration mismatch")
    if fit["mode"]!="full" or test["mode"]!="full":
        raise ValueError("pilot cannot substitute for original science evaluation")
    if (fit["source_LGA02_bundle_sha256"]!=SOURCE_SHA or
        fit["source_LGA02_train_receipt_sha256"]!=SOURCE_META_SHA or
        fit["value_bundle_sha256"]!=digest(fit_raw) or
        test["source_frozen_value_bundle_sha256"]!=digest(fit_raw) or
        test["source_frozen_value_receipt_sha256"]!=digest(fit_meta_raw) or
        test["original_LGA02_source_bundle_sha256"]!=SOURCE_SHA or
        test["original_LGA02_train_receipt_sha256"]!=SOURCE_META_SHA):
        raise ValueError("frozen source value or original teacher artifact SHA mismatch")
    if fit["new_test_labels_read"] is not False or fit["value_training_source"]["new_test_generated"] is not False:
        raise ValueError("test data used to train horizon-conditioned value critic")
    if fit["teacher_seeds"]!=list(SEEDS) or fit["value_train_steps"]!=300:
        raise ValueError("registered value fit seeds/updates changed")
    if (fit["value_training_source"]["train"]["seed"]!=31048 or
        fit["value_training_source"]["validation"]["seed"]!=31049 or
        fit["value_training_source"]["train"]["n"]!=768 or
        fit["value_training_source"]["validation"]["n"]!=192):
        raise ValueError("value train/validation source sizes or seeds changed")
    if (test["n_independent_teacher_seeds"]!=3 or
        test["n_original_heldout_sequences"]!=N_TEST or
        set(test["results"])!={str(x) for x in SEEDS}):
        raise ValueError("original seed/test population differs")
    for sid in SEEDS:
        rec=fit["source_fits"][str(sid)]
        if not rec["selected_update"] in (0,50,100,150,200,250,300):
            raise ValueError("value checkpoint selected outside validation-only grid")
        history=rec["validation_grid"]
        if [r["update"] for r in history]!=[0,50,100,150,200,250,300]:
            raise ValueError("registered value validation checks altered")
        selected=min(history,key=lambda r:(r["val_huber_objective"],r["update"]))
        if rec["selected_update"]!=selected["update"]:
            raise ValueError("value model checkpoint chosen using test outcomes")
        for key in ("value_selected_state_sha256","value_initial_state_sha256",
                    "normalization_sha256"):
            if len(rec[key])!=64:
                raise ValueError("value state not canonical SHA stamped")
    return fit,test,protocol


def verify_new_split(fit,test):
    testm=test["heldout"]
    raw=test["new_test_source"]
    x,y=raw["x"],raw["y"]
    if testm["seed"]!=31050 or testm["n_sequences"]!=256 or len(x)!=256 or len(y)!=256:
        raise ValueError("registered fresh independent test seed/cases absent")
    if any(len(a)!=81 or len(b)!=81 or b!=a[-1:]+a[:-1] for a,b in zip(x,y)):
        raise ValueError("new cyclic supervised targets changed")
    if len({tuple(a) for a in x})!=256:
        raise ValueError("fresh source contains duplicate sequences")
    if sha_int64(x)!=testm["input_sha256"] or sha_int64(y)!=testm["target_sha256"]:
        raise ValueError("original fresh test input/target SHA differs")
    if (testm["input_sha256"]==fit["value_training_source"]["train"]["x_sha256"] or
        testm["input_sha256"]==fit["value_training_source"]["validation"]["x_sha256"]):
        raise ValueError("test source reused value calibration")
    return testm


def verify_profiles(receipt,sid):
    depth_passes=0
    profile=receipt["value_gradient_profiles"]
    if set(profile)!={str(k) for k in DEPTHS}:
        raise ValueError("test omitted registered horizon depths")
    for k in DEPTHS:
        entry=profile[str(k)]
        rows=entry["rows"]
        if len(rows)!=256 or entry["n_whole_sequences"]!=256:
            raise ValueError("per-token pseudo-replication or missing whole-sequence metrics")
        means={key:sum(row[key] for row in rows)/len(rows) for key in rows[0]}
        for key,value in means.items():
            close(value,entry["mean"][key],f"profile_{sid}_{k}_{key}",tol=2e-8)
        for row in rows:
            close(row["gain_value_over_initial"],
                  row["cos_value_future"]-row["cos_initial_future"],
                  "value-vs-init signed alignment",tol=2e-6)
            close(row["gain_value_over_local"],
                  row["cos_value_future"]-row["cos_local_future"],
                  "value-vs-local signed alignment",tol=2e-6)
            for name in ("cos_value_future","cos_initial_future","cos_local_future",
                         "cos_native_negative_future","cos_native_negative_value",
                         "cos_future_local"):
                if abs(row[name])>1.00001:
                    raise ValueError("unnormalized cosine in source result")
        validity=True
        for metric in ("gain_value_over_initial","gain_value_over_local"):
            vals=[row[metric] for row in rows]
            interval=p95(vals,seed(43050,sid,k,metric))
            for x,v in zip(interval,entry["bootstrap95"][metric]):
                close(x,v,"bootstrap95_"+metric,tol=2e-8)
            validity=validity and interval[0]>0
        expected=bool(validity and
                      means["value_pred_token_MSE"]<means["constant_pred_token_MSE"])
        if expected!=entry["depth_pass"]:
            raise ValueError("registered value-potential depth pass changed")
        depth_passes+=int(expected)
    if depth_passes!=receipt["n_positive_depths"] or bool(depth_passes>=2)!=receipt["registered_seed_pass"]:
        raise ValueError("source registered seed-level gate changed")
    return depth_passes


def verify_causal(receipt,sid):
    details=receipt["causal_interventions"]
    if set(details)!={"1","2"}:
        raise ValueError("original two intervention depths missing")
    checks=0
    n_original=0
    n_admitted=0
    for k in (1,2):
        cell=details[str(k)]
        original=cell["raw_cases"]
        if cell["source_sequences"]!=64 or len(original)!=64 or cell["original_goals"]!=512:
            raise ValueError("original causal sequence/goals missing")
        per_sequence=[]
        for i,row in enumerate(original):
            if row["new_test_sequence_index"]!=i:
                raise ValueError("wrong original first 64 test-source sample index")
            base=row["clean_token_CE"]
            if len(base)!=81:
                raise ValueError("original clean full-final token CE missing")
            clean_mean=sum(base)/81
            if len(row["original_goals"])!=8:
                raise ValueError("original eight fixed intervention goal positions missing")
            admitted=[]
            for j,g in enumerate(row["original_goals"]):
                n_original+=1
                if g["slot"]!=j or g["position"]!=GOALS[j] or g["radius"]<0:
                    raise ValueError("intervention selected a different goal")
                if not g["admitted"]:
                    if g["arms"] or "reason" not in g:
                        raise ValueError("source-degenerate goal admitted into scoring")
                    continue
                n_admitted+=1
                required=set(ARMS)-{"oracle_down"}
                if not required.issubset(set(g["arms"])):
                    raise ValueError("source label-free physical intervention arm missing")
                goal_classes=set()
                for arm,part in g["arms"].items():
                    checks+=1
                    if arm not in ARMS:
                        raise ValueError("unregistered physical direction")
                    close(part["dose"],g["radius"],"equal L2 dose",tol=1e-5)
                    if len(part["edit_sha256"])!=64:
                        raise ValueError("physical edit SHA not recorded")
                    if len(part["full_token_CE"])!=81:
                        raise ValueError("final sequence token CE not recorded")
                    mean=sum(part["full_token_CE"])/81
                    close(mean,part["native_final_sequence_CE"],"native sequence global CE",tol=2e-6)
                    close(clean_mean,part["clean_final_sequence_CE"],"paired global clean CE",tol=2e-6)
                    close(part["global_CE_change"],mean-clean_mean,"native global CE difference",tol=2e-6)
                    goal_target=part["truth_digit"]
                    if not 1<=goal_target<=9:
                        raise ValueError("actual goal label outside task alphabet")
                    logits=part["goal_logits"]
                    original_logits=part["clean_goal_logits"]
                    if len(logits)!=10 or len(original_logits)!=10:
                        raise ValueError("actual native goal full logits missing")
                    competitor=max((t for t in range(10) if t!=goal_target),
                                   key=lambda t:original_logits[t])
                    if competitor!=part["clean_rival"]:
                        raise ValueError("strongest clean rival changed after edit")
                    y_pred=logsumexp(logits)-logits[goal_target]
                    y_clean=logsumexp(original_logits)-original_logits[goal_target]
                    close(y_pred,part["full_token_CE"][g["position"]],"goal CE from native full logits",tol=3e-5)
                    close(y_clean,base[g["position"]],"clean goal CE from native logits",tol=3e-5)
                    close(part["goal_CE_change"],y_pred-y_clean,"goal native CE damage",tol=3e-5)
                    close(part["goal_margin_change"],
                          logits[goal_target]-logits[competitor]-
                          original_logits[goal_target]+original_logits[competitor],
                          "goal margin damage",tol=3e-5)
                    for key in ("learned_linear_global_CE_change",
                                "oracle_linear_global_CE_change"):
                        if not math.isfinite(part[key]):
                            raise ValueError("nonfinite terminal-linear prediction")
                    goal_classes.add(goal_target)
                if len(goal_classes)!=1:
                    raise ValueError("matched interventions used different true goal digit")
                admitted.append(g)
            contrasts={}
            for a,b in CONTRASTS:
                name=f"{a}_minus_{b}"
                contrasts[name]=(
                    sum(g["arms"][a]["global_CE_change"]-
                        g["arms"][b]["global_CE_change"] for g in admitted)/len(admitted)
                    if admitted else 0.)
            per_sequence.append({"source_sequence_index":i,
                                 "original_goals":8,
                                 "admitted_goals":len(admitted),
                                 "paired_contrasts":contrasts})
        if cell["admitted_goals"]!=sum(z["admitted_goals"] for z in per_sequence):
            raise ValueError("source causal geometry admission count changed")
        if len(cell["per_sequence"])!=len(per_sequence):
            raise ValueError("source causal whole-string cluster missing")
        for a,b in zip(cell["per_sequence"],per_sequence):
            if a["admitted_goals"]!=b["admitted_goals"]:
                raise ValueError("source per-sequence goal count mismatch")
            for key in b["paired_contrasts"]:
                close(a["paired_contrasts"][key],b["paired_contrasts"][key],
                      "per-sequence matched edit contrast")
        for a,b in CONTRASTS:
            name=f"{a}_minus_{b}"
            vals=[row["paired_contrasts"][name] for row in per_sequence]
            expected=cell["contrasts"][name]
            ci=p95(vals,seed("causal",43050,sid,k,name))
            close(sum(vals)/len(vals),expected["mean"],"native contrast mean")
            for x,v in zip(ci,expected["bootstrap95"]):
                close(x,v,"native paired bootstrap")
            sign=("POSITIVE" if ci[0]>0 else
                  "NEGATIVE" if ci[1]<0 else "INCONCLUSIVE")
            if expected["nominal_direction"]!=sign:
                raise ValueError("source native physical effect classification changed")
    return {"n_original_goal_cases":n_original,"n_admitted_goal_cases":n_admitted,
            "n_physical_arm_dose_checks":checks}


def logsumexp(values):
    top=max(values)
    return top+math.log(sum(math.exp(v-top) for v in values))


def audit(source_raw,source_meta_raw,value_raw,value_meta_raw,test_raw):
    fit,result,protocol=validate_sources(
        source_raw,source_meta_raw,value_raw,value_meta_raw,test_raw)
    tested=verify_new_split(fit,result)
    passed=0
    seeds=[]
    for sid in SEEDS:
        record=result["results"][str(sid)]
        positives=verify_profiles(record,sid)
        native=verify_causal(record,sid)
        passed+=int(record["registered_seed_pass"])
        seeds.append({
            "teacher_seed":sid,
            "positive_depths":positives,
            "registered_seed_pass":record["registered_seed_pass"],
            "future_gradient_cosines":[
                record["value_gradient_profiles"][str(k)]["mean"]["cos_value_future"]
                for k in range(4)],
            "native":native
        })
    verdict="POSITIVE" if passed>=2 else "FAIL"
    if passed!=result["n_teacher_seed_passes"] or result["registered_primary_status"]!=verdict:
        raise ValueError("reported registered primary verdict changed after evaluation")
    return {
        "schema":"LGA-03/independent-stdlib-full-science-replay-v1",
        "audit_status":"PASS","science":verdict,
        "source_protocol_sha256":protocol,
        "original_teacher_SAE_bundle_sha256":digest(source_raw),
        "old_train_only_value_bundle_sha256":digest(value_raw),
        "value_fit_calibration_receipt_sha256":digest(value_meta_raw),
        "fresh_test_receipt_sha256":digest(test_raw),
        "fresh_test":tested,"teacher_seed_pass_count":passed,
        "n_independent_new_test_sequences":256,
        "teacher_seeds":seeds,
        "native_model_and_autograd_recomputed":False,
        "limitations":"Replays true new-test cyclic task, original source file hashes, per-sequence gradient/causal arithmetic and original whole-sequence bootstrap, not model forward or autograd."
    }


def main():
    p=argparse.ArgumentParser()
    for name in ("source-bundle","source-meta","value-bundle","value-receipt",
                 "test-receipt","out"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    res=audit(a.source_bundle.read_bytes(),a.source_meta.read_bytes(),
              a.value_bundle.read_bytes(),a.value_receipt.read_bytes(),
              a.test_receipt.read_bytes())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(res,sort_keys=True,indent=2)+"\n")
    print("LGA03_INDEPENDENT_RECEIPT_PASS",res["science"],
          res["teacher_seed_pass_count"],flush=True)


if __name__=="__main__":
    main()
