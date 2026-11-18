"""LGA-02 source-independent stdlib audit of frozen teacher and native test receipts.

Reads the model+SAE bundle as immutable bytes, but intentionally never loads
PyTorch. Checks the original LGA-01 source train/validation/old-test hashes,
the fresh full-sequence cyclic-copy task and data SHA, all reported
per-sequence terminal/local gradient aggregates, 2000 sequence-bootstrap
draws, and matched causal final-CE/goal-logit results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct


SCHEMA="LGA-02/future-gradient-SAE-and-native-causal-receipt-v1"
TRAIN_SCHEMA="LGA-02/source-reproduced-trained-model-and-SAE-v1"
SEEDS=(103,211,307)
TEST_SEED=31044
BOOTS=2000
GOALS=(0,10,20,30,40,50,60,70)
ARMS=("full","sae_active","sae_remainder","head_null","random")
CONTRASTS=(("sae_active","sae_remainder"),("head_null","random"))
PREVIOUS_SPLITS={
    "train":"12deb8f0b37bc4709f395786f455ee147b4fcf16606432fde8250d04818ad818",
    "val":"f1506d79fdc7daac45b2624297d9abc0634ad9f0b52c0f8c04d33e5853d71bc5",
    "old_test":"e0980800dce2b03d5fe18afdaa0a71e873634826e7641c9ae24bb385ad592fac",
}
PREVIOUS_OLD_FINAL_ACC={103:.9873167506884784,
                        211:.984712584875524,
                        307:.9866415972355753}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def new_seed(*parts):
    x="LGA02|"+"|".join(map(str,parts))
    return int.from_bytes(hashlib.sha256(x.encode()).digest()[:8],"big") & ((1<<63)-1)


def p95(values,seed):
    n=len(values)
    gen=random.Random(seed)
    if n<2 or any(not math.isfinite(float(v)) for v in values):
        raise ValueError("nonfinite source sequence or too few clusters")
    samples=sorted(sum(values[gen.randrange(n)] for _ in range(n))/n
                   for __ in range(BOOTS))
    return [samples[int(.025*BOOTS)],samples[int(.975*BOOTS)-1]]


def close(a,b,name,limit=1e-8):
    if not math.isfinite(float(a)) or not math.isfinite(float(b)):
        raise ValueError("nonfinite reported result: "+name)
    if abs(float(a)-float(b))>limit*max(1.,abs(float(a)),abs(float(b))):
        raise ValueError("independent source arithmetic mismatch: "+name)


def int64_sha(values):
    flat=[x for row in values for x in row]
    return digest(struct.pack("<"+"q"*len(flat),*flat))


def verify_source(meta,raw_meta,raw_weights,result):
    protocol=digest((Path(__file__).parent/"PROTOCOL.md").read_bytes())
    if meta["schema"]!=TRAIN_SCHEMA or result["schema"]!=SCHEMA:
        raise ValueError("source-trained schema differs")
    if meta["protocol_sha256"]!=protocol or result["protocol_sha256"]!=protocol:
        raise ValueError("pre-data registration changed")
    if meta["mode"]!="full" or result["mode"]!="full":
        raise ValueError("source scientific test must exclude pilot")
    if (meta["source_bundle_sha256"]!=digest(raw_weights) or
        result["source_training_bundle_sha256"]!=digest(raw_weights) or
        result["source_training_manifest_sha256"]!=digest(raw_meta)):
        raise ValueError("source weights/SAE/training manifest replaced")
    if meta["teacher_init_seeds"]!=list(SEEDS) or result["primary_seeds"]!=list(SEEDS):
        raise ValueError("trained model seed population changed")
    if meta["teacher_training_steps"]!=200 or meta["sae_training_steps"]!=150:
        raise ValueError("source training budget changed")
    if meta["new_test_generated"] is not False or meta["new_test_labels_read"] is not False:
        raise ValueError("fresh test source available to train-only SAE fit")
    if (meta["source_training_inputs"]["x_sha256"]!=PREVIOUS_SPLITS["train"] or
        meta["source_validation_inputs"]["x_sha256"]!=PREVIOUS_SPLITS["val"] or
        meta["prior_test_used_only_for_reproduction"]["x_sha256"]!=PREVIOUS_SPLITS["old_test"]):
        raise ValueError("source train/validation/old-test input data changed")
    for sid in SEEDS:
        r=meta["teacher_receipts"][str(sid)]
        if r["old_training_update_step"]!=200:
            raise ValueError("source checkpoint optimizer steps changed")
        if abs(r["original_LGA01_accuracy"]-PREVIOUS_OLD_FINAL_ACC[sid])>1e-12:
            raise ValueError("old LGA01 teacher target changed")
        if abs(r["reproduced_accuracy"]-PREVIOUS_OLD_FINAL_ACC[sid])>.01:
            raise ValueError("reproduced model differs from original source checkpoint skill")
        if not r["original_source_reproduction_pass"] or set(r["saes"])!={"1","2"}:
            raise ValueError("source model checkpoint/SAE admission changed")
        for dep in ("1","2"):
            info=r["saes"][dep]
            if not len(info["sae_tensor_sha256"])==len(info["normalizer_sha256"])==64:
                raise ValueError("source SAE weights not SHA-pinned")
            if info["train_tokens"]!=512*81 or info["val_tokens"]!=128*81:
                raise ValueError("SAE fit used non-source train or validation states")
    return protocol


def verify_new_test(result,meta):
    m=result["fresh_test"]
    src=result["fresh_test_source"]
    x,y=src["x"],src["y"]
    if (m["seed"]!=TEST_SEED or m["n_sequences"]!=256 or
        len(x)!=256 or len(y)!=256 or m["no_exact_sequence_overlap"] is not True):
        raise ValueError("registered fresh 256 heldout source missing")
    for a,b in zip(x,y):
        if len(a)!=81 or len(b)!=81 or any(v not in range(1,10) for v in a):
            raise ValueError("wrong 81-cell independent cyclic source")
        if b!=a[-1:]+a[:-1]:
            raise ValueError("heldout target differs from registered cyclic-copy task")
    if len(set(tuple(row) for row in x))!=256:
        raise ValueError("new heldout source has repeated sequence")
    if int64_sha(x)!=m["x_sha256"] or int64_sha(y)!=m["y_sha256"]:
        raise ValueError("original fresh data SHA differs from archived source")
    if m["x_sha256"]==meta["prior_test_used_only_for_reproduction"]["x_sha256"]:
        raise ValueError("new source data duplicate original LGA01 test")
    return {"seed":m["seed"],"n_sequences":256,
            "test_x_sha256":m["x_sha256"],"test_y_sha256":m["y_sha256"]}


def check_profiles(item,teacher_seed):
    profiles=item["gradient_by_loop"]
    if set(profiles)!={str(k) for k in range(4)}:
        raise ValueError("source only registered 4 linked steps")
    count=0
    for k in range(4):
        p=profiles[str(k)]
        rows=p["per_sequence"]
        if len(rows)!=256 or p["n_whole_sequences"]!=256:
            raise ValueError("whole-string bootstrap numerator/denominator changed")
        for name,val in p["means"].items():
            values=[r[name] for r in rows]
            if any(not math.isfinite(float(x)) for x in values):
                raise ValueError("nonfinite hidden-state trajectory")
            close(sum(values)/len(values),val,f"profile_{name}_{k}")
        for name in ("future_cos","local_cos","future_minus_local_cos"):
            interval=p95([r[name] for r in rows],
                         new_seed(42044,teacher_seed,k,name))
            for x,z in zip(interval,p["bootstrap95"][name]):
                close(x,z,f"cos_ci_{name}_{k}")
        if p["bootstrap95"]["future_cos"][0]>0:
            count+=1
    if count!=item["positive_future_gradient_depths"]:
        raise ValueError("future gradient classification changed")
    if bool(item["old_model_source_parity_pass"] and count>=2)!=item["registered_seed_positive"]:
        raise ValueError("source positive model-seed gate changed")
    return count


def logsumexp(z):
    mx=max(z)
    return mx+math.log(sum(math.exp(t-mx) for t in z))


def audit_causal(item,sid):
    saved=item["native_causal"]
    if set(saved)!={"1","2"}:
        raise ValueError("missing native causal interventions at two source depths")
    output={}
    for ks in ("1","2"):
        k=int(ks)
        cohort=saved[ks]
        raw=cohort["raw_cases"]
        if cohort["n_source_sequences"]!=64 or len(raw)!=64:
            raise ValueError("original first-64 new test source sample list altered")
        if cohort["n_original_goals"]!=512:
            raise ValueError("source native causal goal count changed")
        summary=[]
        dose_checks=0
        all_original_goals=0
        total_admitted=0
        for sequence_index,row in enumerate(raw):
            if row["test_sequence_index"]!=sequence_index or len(row["goals"])!=8:
                raise ValueError("source native heldout sequence/goal order changed")
            enrolled=[g for g in row["goals"] if g["admitted"]]
            total_admitted+=len(enrolled)
            row_averages={}
            for a,b in CONTRASTS:
                name=f"{a}_minus_{b}"
                row_averages[name]=(
                    sum(g["arms"][a]["global_CE_change"]-
                        g["arms"][b]["global_CE_change"] for g in enrolled)/len(enrolled)
                    if enrolled else 0.)
            for goal_index,goal in enumerate(row["goals"]):
                all_original_goals+=1
                if goal["goal_index"]!=goal_index or goal["position"]!=GOALS[goal_index]:
                    raise ValueError("original fixed goal positions changed")
                if not goal["admitted"]:
                    if goal["arms"] or goal.get("reason")!="degenerate_source_sae_or_headnull_projection":
                        raise ValueError("ineligible source direction incorrectly scored")
                    continue
                if set(goal["arms"])!=set(ARMS):
                    raise ValueError("registered native edit arm missing")
                target=goal["truth_digit"];rival=goal["frozen_clean_rival"]
                if target not in range(1,10) or target==rival or rival not in range(10):
                    raise ValueError("native true-digit readout or clean rival invalid")
                for name in ARMS:
                    r=goal["arms"][name]
                    dose_checks+=1
                    if abs(r["dose"]-goal["dose"])>1e-5 or abs(r["edit_physical_l2"]-goal["dose"])>1e-5:
                        raise ValueError("physical native edit dose control differs")
                    logits=r["native_goal_full_logits"]
                    clean=r["clean_goal_full_logits"]
                    if len(logits)!=10 or len(clean)!=10:
                        raise ValueError("native full digit logits missing")
                    expected_rival=max((z for z in range(10) if z!=target),key=lambda z:clean[z])
                    if rival!=expected_rival:
                        raise ValueError("source clean readout rival changed")
                    loss=logsumexp(logits)-logits[target]
                    clean_loss=logsumexp(clean)-clean[target]
                    close(loss,r["native_goal_CE"],"native_goal_CE",limit=2e-5)
                    close(clean_loss,r["clean_goal_CE"],"clean_goal_CE",limit=2e-5)
                    close(r["goal_CE_change"],
                          r["native_goal_CE"]-r["clean_goal_CE"],"native_goal_loss_delta")
                    close(r["global_CE_change"],
                          r["native_global_mean_CE"]-r["clean_global_mean_CE"],
                          "native_global_CE_delta")
                    close(r["goal_margin_change"],
                          logits[target]-logits[rival]-clean[target]+clean[rival],
                          "native_goal_margin")
                    if not math.isfinite(r["linear_terminal_global_CE_change"]):
                        raise ValueError("nonfinite terminal tangent prediction")
            means={name:(
                sum(g["arms"][name]["global_CE_change"] for g in enrolled)/len(enrolled)
                if enrolled else 0.) for name in ARMS}
            summary.append({"eligible_goals":len(enrolled),
                            "original_goals":8,
                            "contrasts":row_averages,
                            "arms_mean":means})
        if all_original_goals!=512 or total_admitted!=cohort["n_admitted_goal_comparisons"]:
            raise ValueError("source paired SAE/headnull admission denominator changed")
        for i,(observed,recorded) in enumerate(zip(summary,cohort["source_sequence_rows"])):
            if observed["eligible_goals"]!=recorded["eligible_goals"]:
                raise ValueError("source paired causal case counts differ")
            for key in observed["contrasts"]:
                close(observed["contrasts"][key],recorded["contrasts"][key],f"cluster_{i}_{key}")
            for key in ARMS:
                close(observed["arms_mean"][key],recorded["arms_mean"][key],f"arm_{i}_{key}")
        for a,b in CONTRASTS:
            name=f"{a}_minus_{b}"
            data=[r["contrasts"][name] for r in summary]
            mean=sum(data)/len(data)
            actual=cohort["contrasts"][name]
            ci=p95(data,new_seed("causal",sid,k,name))
            close(mean,actual["mean"],name)
            for x,z in zip(ci,actual["bootstrap95"]):
                close(x,z,f"causal95_{name}")
            verdict=("POSITIVE" if ci[0]>0 else
                     "NEGATIVE" if ci[1]<0 else "INCONCLUSIVE")
            if verdict!=actual["signed_classification"]:
                raise ValueError("registered physical direction uncertainty changed")
        output[ks]={"n_original_goals":512,
                    "n_admitted":total_admitted,
                    "matched_arm_dose_checks":dose_checks,
                    "contrasts":cohort["contrasts"]}
    return output


def audit(meta_raw,weight_raw,test_raw):
    meta=json.loads(meta_raw);test=json.loads(test_raw)
    protocol=verify_source(meta,meta_raw,weight_raw,test)
    fresh=verify_new_test(test,meta)
    passed=0
    summaries=[]
    total_native_goal=0
    for sid in SEEDS:
        item=test["results"][str(sid)]
        n=check_profiles(item,sid)
        causal=audit_causal(item,sid)
        total_native_goal+=sum(x["n_original_goals"] for x in causal.values())
        passed+=int(item["registered_seed_positive"])
        summaries.append({"seed":sid,
                          "positive_future_depths":n,
                          "registered_seed_positive":item["registered_seed_positive"],
                          "cosine_means":[item["gradient_by_loop"][str(k)]["means"]["future_cos"]
                                          for k in range(4)],
                          "causal":causal})
    if (passed!=test["n_primary_seed_passes"] or
        test["registered_primary_status"]!=(
            "POSITIVE" if passed>=2 else "FAIL_OR_NOT_ADMITTED")):
        raise ValueError("study gradient classification changed after source test")
    return {
        "schema":"LGA-02/independent-stdlib-full-receipt-audit-v1",
        "audit":"PASS",
        "science":test["registered_primary_status"],
        "model_seeds_passed":passed,
        "protocol_sha256":protocol,
        "teacher_SAE_bundle_sha256":digest(weight_raw),
        "teacher_SAE_training_manifest_sha256":digest(meta_raw),
        "fresh_scored_sha256":digest(test_raw),
        "fresh_test":fresh,
        "n_independent_new_test_sequences":256,
        "n_original_native_causal_goal_sequences":total_native_goal,
        "teacher_results":summaries,
        "limitations":"Replays source and native logits/CE/margin arithmetic, plus whole-sequence bootstraps, without running the trained teacher or recomputing autograd."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--meta",type=Path,required=True)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    report=audit(a.meta.read_bytes(),a.bundle.read_bytes(),a.receipt.read_bytes())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("LGA02_INDEPENDENT_AUDIT_PASS",report["science"],report["model_seeds_passed"],flush=True)


if __name__=="__main__":
    main()
