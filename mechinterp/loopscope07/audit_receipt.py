"""LGA-04 independent standard-library validation of original source receipts.

Replays calibration/heldout cyclic source hashes, model/critic artifact SHA,
whole-sequence 2000-bootstrap gradient gates and native ten-logit CE/margins.
Original neural forward/gradients remain verified by immutable model-job bytes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct

SCHEMA="LGA-04/new-heldout-paired-sobolev-value-native-causal-v1"
FIT="LGA-04/calibration-only-matched-sobolev-critic-v1"
LOCKS=(
    "d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266",
    "7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d",
    "4d1cb63d301f6cc1b1c90fea999a82de0fb4f077b1f1f2e3bb6f0047aead937a",
    "267ffabb76e3a9fd7e42cf10b4eb31268d635f3bfe67a9c875ef461437ab189d",
)
SEEDS=(103,211,307)
GOALS=(0,10,20,30,40,50,60,70)
ARMS=("sobolev_down","valueonly_down","original_down","native",
      "random","sae_active","sae_remainder","head_null")
CONTRASTS=(("sobolev_down","valueonly_down"),("sobolev_down","random"),
           ("sobolev_down","original_down"),("sae_active","sae_remainder"),
           ("head_null","random"))


def sha(raw):return hashlib.sha256(raw).hexdigest()


def protocol_sha():
    return sha((Path(__file__).resolve().parent/"PROTOCOL.md").read_bytes())


def stable(*x):
    q="LGA04|"+"|".join(map(str,x))
    return int.from_bytes(hashlib.sha256(q.encode()).digest()[:8],"big")&((1<<63)-1)


def close(a,b,name,tol=2e-6):
    a=float(a);b=float(b)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("nonfinite numerical receipt "+name)
    if abs(a-b)>tol*max(1.,abs(a),abs(b)):
        raise ValueError("independent numerical replay mismatch "+name)


def bootstrap(vals,root):
    n=len(vals);rng=random.Random(root)
    if n<2 or any(not math.isfinite(x) for x in vals):
        raise ValueError("whole-sequence bootstrap input not admitted")
    v=sorted(sum(vals[rng.randrange(n)] for _ in range(n))/n
             for j in range(2000))
    return [v[50],v[1949]]


def sha_tensor(rows):
    flat=[int(z) for row in rows for z in row]
    return sha(struct.pack("<"+"q"*len(flat),*flat))


def cyclic_source(x,y,n):
    if len(x)!=n or len(y)!=n:
        raise ValueError("source independent sequence sample size changed")
    for a,b in zip(x,y):
        if (len(a)!=81 or len(b)!=81 or
            any(not 1<=z<=9 for z in a) or b!=a[-1:]+a[:-1]):
            raise ValueError("source cyclic-copy target changed")
    if len({tuple(row) for row in x})!=n:
        raise ValueError("duplicated source independent strings")


def logsumexp(z):
    m=max(z)
    return m+math.log(sum(math.exp(t-m) for t in z))


def check_calibration(meta):
    if (meta["schema"]!=FIT or meta["protocol_sha256"]!=protocol_sha() or
        meta["training_steps"]!=100 or meta["gradient_loss_lambda"]!=1.0 or
        meta["original_teacher_seeds"]!=list(SEEDS) or
        meta["training_arms"]!=["value_only_continue","value_plus_gradient"] or
        meta["test_data_opened"] is not False):
        raise ValueError("frozen calibration method/seed/registration changed")
    c=meta["calibration_source"]
    seen=set()
    for name,sd,n in (("train",31052,256),("validation",31053,64)):
        row=c[name]
        if row["generator_seed"]!=sd or row["n_sequences"]!=n:
            raise ValueError("calibration source split changed")
        cyclic_source(row["x"],row["y"],n)
        if sha_tensor(row["x"])!=row["x_sha256"] or sha_tensor(row["y"])!=row["y_sha256"]:
            raise ValueError("calibration source byte SHA corrupted")
        if any(tuple(z) in seen for z in row["x"]):
            raise ValueError("calibration train-validation source overlap")
        seen.update(tuple(z) for z in row["x"])
    if c["new_test_generated"] is not False or c["reserved_test_seed"]!=31054:
        raise ValueError("heldout source revealed before calibration")
    for sid in SEEDS:
        r=meta["paired_training_receipts"][str(sid)]
        paired=r["paired_training"]
        if ([z["arm"] for z in paired]!=["value_only_continue","value_plus_gradient"]
            or paired[0]["starting_weight_sha256"]!=paired[1]["starting_weight_sha256"]
            or paired[0]["starting_weight_sha256"]!=r["original_critic_initial_sha256"]
            or r["fit_grad_lambda"]!=1.0):
            raise ValueError("paired critic initialization or derivative objective changed")
        for arm in paired:
            if (arm["optimizer_updates"]!=100 or
                [z["updates"] for z in arm["validation_checkpoints"]]!=[0,50,100]
                or len(arm["weight_sha256"])!=64):
                raise ValueError("paired critic training budget or checkpoint list changed")
        if len(r["calibration_gradient_cache_sha256"])!=6 or r["train_gradient_RMS"]<=0:
            raise ValueError("terminal gradient calibration cache missing")
    return seen


def replay_gradients(entry,sid):
    source=entry["gradient_profiles"]
    if set(source)!={str(i) for i in range(4)}:
        raise ValueError("four registered recurrent horizons incomplete")
    passed=0
    for k in range(4):
        r=source[str(k)];rows=r["rows"]
        if len(rows)!=256 or r["n_whole_sequences"]!=256:
            raise ValueError("per-token pseudo-replication in gradient test")
        means={name:sum(v[name] for v in rows)/256 for name in rows[0]}
        for name,val in means.items():
            close(val,r["means"][name],"per-sequence mean "+name,tol=1e-8)
        for v in rows:
            close(v["gain_over_valueonly"],v["value_plus_gradient_cos"]-
                  v["value_only_continue_cos"],"paired value-only cosine",tol=1e-7)
            close(v["gain_over_original"],v["value_plus_gradient_cos"]-
                  v["original_cos"],"paired frozen prior cosine",tol=1e-7)
            for name in ("value_plus_gradient_cos","value_only_continue_cos",
                         "original_cos","local_cos","native_terminal_cos"):
                if abs(v[name])>1.00001:raise ValueError("reported tokenwise cosine not normalized")
        ok=True
        for field in ("gain_over_valueonly","gain_over_original"):
            ci=bootstrap([z[field] for z in rows],stable(44054,sid,k,field))
            for x,y in zip(ci,r["bootstrap95"][field]):
                close(x,y,"registered 2000-bootstrap",tol=1e-8)
            ok=ok and ci[0]>0
        ok=ok and (means["value_plus_gradient_grad_NMSE"] <
                   means["value_only_continue_grad_NMSE"])
        if bool(ok)!=r["registered_depth_pass"]:
            raise ValueError("original preregistered depth acceptance drifted")
        passed+=int(ok)
    if passed!=entry["positive_gradient_depths"] or bool(passed>=2)!=entry["registered_seed_pass"]:
        raise ValueError("original trained-seed classification changed")
    return passed


def replay_native(entry,sid):
    results={}
    for k in (1,2):
        cell=entry["native_effects"][str(k)]
        cases=cell["raw_cases"]
        if len(cases)!=64 or cell["n_original_sequences"]!=64 or cell["n_original_goal_slots"]!=512:
            raise ValueError("original native source cohort not 64 by eight")
        clusters=[];admitted=0;arm_count=0
        for i,seq in enumerate(cases):
            if seq["test_sequence_index"]!=i or len(seq["goal_plan"])!=8:
                raise ValueError("source intervention order/selected blank positions changed")
            good=[]
            for j,goal in enumerate(seq["goal_plan"]):
                if goal["slot"]!=j or goal["token"]!=GOALS[j] or goal["radius"]<0:
                    raise ValueError("fixed source goal changed")
                if not goal["preoutcome_admitted"]:
                    if goal["arms"] or "reason" not in goal:
                        raise ValueError("source-ineligible intervention was scored")
                    continue
                if not set(ARMS).issubset(set(goal["arms"])):
                    raise ValueError("outcome-blind matched physical arm missing")
                admitted+=1
                for arm,r in goal["arms"].items():
                    arm_count+=1
                    if abs(r["dose"]-goal["radius"])>1e-5 or len(r["edit_sha256"])!=64:
                        raise ValueError("edit direction changed physical dose")
                    native=r["goal_logits"];clean=r["clean_goal_logits"]
                    true=int(r["truth_digit"])
                    if len(native)!=10 or len(clean)!=10 or not 1<=true<=9:
                        raise ValueError("native ten-logit output is missing")
                    rival=max((c for c in range(10) if c!=true),key=lambda c:clean[c])
                    if rival!=r["clean_rival"]:raise ValueError("source clean competitor changed")
                    if len(r["all_final_token_CE"])!=81 or len(r["clean_all_final_token_CE"])!=81:
                        raise ValueError("global 81-token CE unavailable")
                    close(logsumexp(native)-native[true],
                          r["all_final_token_CE"][goal["token"]],"native goal CE",3e-5)
                    close(logsumexp(clean)-clean[true],
                          r["clean_all_final_token_CE"][goal["token"]],"clean goal CE",3e-5)
                    close(r["goal_CE_change"],
                          (logsumexp(native)-native[true])-(logsumexp(clean)-clean[true]),
                          "native goal CE delta",3e-5)
                    close(r["goal_margin_change"],native[true]-native[rival]-
                          clean[true]+clean[rival],"native goal margin",3e-5)
                    final=sum(r["all_final_token_CE"])/81
                    base=sum(r["clean_all_final_token_CE"])/81
                    close(final,r["final_global_CE"],"full native terminal CE")
                    close(base,r["clean_global_CE"],"full clean terminal CE")
                    close(final-base,r["global_CE_change"],"global CE change")
                    if not (math.isfinite(r["predicted_sobolev_global_CE_change"]) and
                            math.isfinite(r["predicted_terminal_global_CE_change"])):
                        raise ValueError("terminal gradient directional estimate nonfinite")
                good.append(goal)
            observed={}
            for a,b in CONTRASTS:
                name=f"{a}_minus_{b}"
                dif=[g["arms"][a]["global_CE_change"]-g["arms"][b]["global_CE_change"]
                     for g in good]
                observed[name]=sum(dif)/len(dif) if dif else 0.
            clusters.append({"n_admitted_goals":len(good),"paired_global_CE":observed})
        if admitted!=cell["n_admitted_goal_slots"]:
            raise ValueError("source physical edit admission denominator changed")
        for a,b in zip(clusters,cell["source_rows"]):
            if a["n_admitted_goals"]!=b["n_admitted_goals"]:
                raise ValueError("per-string native denominator changed")
            for name,value in a["paired_global_CE"].items():
                close(value,b["paired_global_CE"][name],"per-string native CE difference",1e-8)
        for a,b in CONTRASTS:
            name=f"{a}_minus_{b}"
            v=[x["paired_global_CE"][name] for x in clusters]
            ci=bootstrap(v,stable("causal",44054,sid,k,name))
            r=cell["comparisons"][name]
            close(sum(v)/64,r["mean"],"causal contrast mean")
            for x,y in zip(ci,r["bootstrap95"]):close(x,y,"causal cluster interval",1e-8)
            sign="POSITIVE" if ci[0]>0 else "NEGATIVE" if ci[1]<0 else "INCONCLUSIVE"
            if sign!=r["direction"]:raise ValueError("source native effect sign gate changed")
        results[str(k)]={"n_source_goals":512,"n_eligible":admitted,
                         "n_native_arms":arm_count,"contrasts":cell["comparisons"]}
    return results


def audit(raw_source,raw_meta,raw_prior,raw_prior_meta,raw_fit,raw_fit_meta,raw_test):
    if [sha(z) for z in (raw_source,raw_meta,raw_prior,raw_prior_meta)]!=list(LOCKS):
        raise ValueError("original source teacher/critic fixed SHA changed")
    m=json.loads(raw_fit_meta)
    t=json.loads(raw_test)
    seen=check_calibration(m)
    if (m["frozen_bundle_sha256"]!=sha(raw_fit) or
        t["schema"]!=SCHEMA or t["protocol_sha256"]!=protocol_sha() or
        t["mode"]!="full" or t["frozen_matched_critic_bundle_sha256"]!=sha(raw_fit) or
        t["frozen_matched_critic_meta_sha256"]!=sha(raw_fit_meta) or
        t["source_bundle_sha256"]!=LOCKS[0] or
        t["source_meta_sha256"]!=LOCKS[1] or
        t["original_critic_bundle_sha256"]!=LOCKS[2] or
        t["original_critic_meta_sha256"]!=LOCKS[3]):
        raise ValueError("source prospective artifact SHA/registration does not agree")
    new=t["new_test"]
    if new["seed"]!=31054 or new["n_sequences"]!=256 or t["n_test_sequences"]!=256:
        raise ValueError("fresh test source size/seed changed")
    x=t["fresh_source"]["x"];y=t["fresh_source"]["y"]
    cyclic_source(x,y,256)
    if sha_tensor(x)!=new["x_sha256"] or sha_tensor(y)!=new["y_sha256"]:
        raise ValueError("fresh input/target byte SHA changed")
    if any(tuple(z) in seen for z in x):
        raise ValueError("new test leaked calibration source")
    if t["teacher_seeds"]!=list(SEEDS):
        raise ValueError("original independent model seed population changed")
    model_results=[]
    n_pass=0
    for sid in SEEDS:
        r=t["results"][str(sid)]
        depths=replay_gradients(r,sid)
        native=replay_native(r,sid)
        n_pass+=int(r["registered_seed_pass"])
        model_results.append({"teacher_seed":sid,"positive_depths":depths,
                              "registered_pass":r["registered_seed_pass"],
                              "mean_cosines":[r["gradient_profiles"][str(k)]["means"][
                                  "value_plus_gradient_cos"] for k in range(4)],
                              "native":native})
    status="POSITIVE" if n_pass>=2 else "FAIL"
    if t["registered_scientific_verdict"]!=status or t["n_registered_seed_passes"]!=n_pass:
        raise ValueError("registered final verdict changed in stored source receipt")
    return {"schema":"LGA-04/independent-stdlib-source-native-audit-v1",
            "audit_status":"PASS","scientific_status":status,
            "number_of_teacher_seeds_passed":n_pass,
            "registered_protocol_sha256":protocol_sha(),
            "calibration_bundle_sha256":sha(raw_fit),
            "fresh_test_receipt_sha256":sha(raw_test),
            "n_new_independent_sequences":256,
            "models":model_results,
            "limitations":"Replays source SHA, gradient cosine bootstrap, and native CE/margins. Does not execute original frozen teacher or second-order critic autograd."}


def main():
    p=argparse.ArgumentParser()
    for name in ("source-bundle","source-meta","prior-bundle","prior-meta",
                 "paired-bundle","paired-meta","test-receipt","out"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    r=audit(a.source_bundle.read_bytes(),a.source_meta.read_bytes(),
            a.prior_bundle.read_bytes(),a.prior_meta.read_bytes(),
            a.paired_bundle.read_bytes(),a.paired_meta.read_bytes(),
            a.test_receipt.read_bytes())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    print("LGA04_INDEPENDENT_AUDIT_PASS",r["scientific_status"],
          r["number_of_teacher_seeds_passed"],flush=True)


if __name__=="__main__":
    main()
