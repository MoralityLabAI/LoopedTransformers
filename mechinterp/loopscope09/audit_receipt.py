"""LGA-06 independent stdlib audit of frozen OOD source, potential descent and native task effects.

This auditor never imports torch. Source bytes, original matched source
IDs, full energy drops, 2,000 whole-base-sequence bootstraps, task scores,
native ten-logit CE, physical edits and finite curl are independently
replayed from original immutable science artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct

SCHEMA="LGA-06/frozen-shifted-12-pass-native-energy-v1"
LOCK_SCHEMA="LGA-06/frozen-pretest-source-and-hypothesis-v1"
SOURCE_SHA="d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266"
SOURCE_META_SHA="7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d"
ENERGY_SHA="24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53"
ENERGY_META_SHA="69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d"
LOCKS=(SOURCE_SHA,SOURCE_META_SHA,ENERGY_SHA,ENERGY_META_SHA)
TEACHERS=(103,211,307)
ARMS=("iid","period9","block9","zero20")
N_TEST=256
N_LOOPS=12
N_NATIVE=64
GOALS=(0,10,20,30,40,50,60,70)
EDIT_ARMS=("shared_down","init_down","radial_down","native",
           "sae_active","sae_remainder","head_null","random")
CONTRASTS=(("shared_down","random"),("shared_down","native"),
           ("shared_down","init_down"),("sae_active","sae_remainder"),
           ("head_null","random"))
BOOT=2000


def digest(x):return hashlib.sha256(x).hexdigest()


def protocol_sha():
    return digest((Path(__file__).resolve().parent/"PROTOCOL.md").read_bytes())


def stable_seed(*parts):
    key="LGA06|"+"|".join(map(str,parts))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"big") & ((1<<63)-1)


def close(a,b,what="numerical receipt",tol=2e-6):
    x=float(a);y=float(b)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("source nonfinite result "+what)
    if abs(x-y)>tol*max(1.,abs(x),abs(y)):
        raise ValueError("independent replay disagrees on "+what)


def bootstrap(values,key):
    n=len(values)
    if n!=N_TEST and n!=N_NATIVE:
        raise ValueError("source cluster bootstrap denominator changed")
    if any(not math.isfinite(float(x)) for x in values):
        raise ValueError("source independent cluster values nonfinite")
    rng=random.Random(key)
    d=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n
             for _ in range(BOOT))
    return [d[int(.025*BOOT)],d[int(.975*BOOT)-1]]


def sha_int64(x):
    values=[int(v) for row in x for v in row]
    return digest(struct.pack("<"+"q"*len(values),*values))


def source_cyclic(x,y):
    if len(x)!=len(y):
        raise ValueError("original source x/y population differs")
    for a,b in zip(x,y):
        if len(a)!=81 or len(b)!=81 or any(not 0<=z<=9 for z in a):
            raise ValueError("source input alphabet or width changed")
        if b!=a[-1:]+a[:-1]:
            raise ValueError("cyclic label changed by input distribution shift")


def verify_sources(source_raw,meta_raw,energy_raw,energy_meta_raw,lock_raw,test_raw):
    found=tuple(digest(x) for x in (source_raw,meta_raw,energy_raw,energy_meta_raw))
    if found!=LOCKS:
        raise ValueError("original trained teacher/SAE/energy checkpoint byte SHA drift")
    lock=json.loads(lock_raw)
    receipt=json.loads(test_raw)
    if (lock["schema"]!=LOCK_SCHEMA or lock["protocol_sha256"]!=protocol_sha()
        or lock["new_test_generated"] is not False or lock["test_labels_read"] is not False
        or tuple(lock[k] for k in (
            "teacher_sha256","teacher_manifest_sha256",
            "potential_sha256","potential_manifest_sha256"))!=LOCKS
        or lock["test_seed_reserved"]!=31058 or
        lock["n_native_extra_loops"]!=12 or tuple(lock["test_arms"])!=ARMS or
        lock["teachers"]!=list(TEACHERS)):
        raise ValueError("preregistered pretest source lock changed")
    if (receipt["schema"]!=SCHEMA or receipt["mode"]!="full"
        or receipt["protocol_sha256"]!=protocol_sha()
        or receipt["source_lock_sha256"]!=digest(lock_raw)
        or tuple(receipt["source_artifact_sha256"])!=LOCKS
        or receipt["teacher_seeds"]!=list(TEACHERS)
        or receipt["distributions"]!=list(ARMS)
        or receipt["n_recurrent_steps"]!=12
        or receipt["n_base_independent_sequences"]!=N_TEST):
        raise ValueError("original source/test registration or artifact changed")
    source=receipt["new_test"]
    base=receipt["base_source_x"]
    if (source["seed"]!=31058 or source["n_base_independent_sequences"]!=N_TEST
        or len(base)!=N_TEST or len(set(tuple(z) for z in base))!=N_TEST):
        raise ValueError("new matched IID base original population changed")
    if sha_int64(base)!=source["base_x_sha256"]:
        raise ValueError("original base IID little-endian source SHA changed")
    families=source["families"]
    if set(families)!=set(ARMS):raise ValueError("shift family missing")
    inputs={}
    for name in ARMS:
        x=[]
        for b in base:
            if name=="iid":r=b[:]
            elif name=="period9":r=b[:9]*9
            elif name=="block9":r=[q for digit in b[::9] for q in [digit]*9]
            elif name=="zero20":r=[0 if j%5==0 else z for j,z in enumerate(b)]
            x.append(r)
        y=[a[-1:]+a[:-1] for a in x]
        source_cyclic(x,y)
        if len(set(tuple(z) for z in x))!=N_TEST:
            raise ValueError("source transformation collapsed matched independent base identities")
        m=families[name]
        if (m["n"]!=N_TEST or sha_int64(x)!=m["x_sha256"]
            or sha_int64(y)!=m["y_sha256"]
            or m["n_zero_input_tokens"]!=sum(sum(v==0 for v in z) for z in x)
            or m["reference_base_indices"]!=list(range(N_TEST))):
            raise ValueError("new source family transform or byte SHA changed")
        inputs[name]=x
    if set(receipt["results"])!={str(z) for z in TEACHERS}:
        raise ValueError("source teacher seed roster changed")
    return receipt


def replay_profile(cell,sid,arm):
    curves=cell["curves"]
    if set(curves)!={str(k) for k in range(N_LOOPS)}:
        raise ValueError("original 12 observed native transition depths missing")
    if cell["n_original_sequences"]!=N_TEST or cell["n_original_steps"]!=12:
        raise ValueError("original source sample or recurrence count changed")
    early=0;late=0
    for k in range(N_LOOPS):
        row=curves[str(k)]
        cases=row["per_sequence"]
        if len(cases)!=N_TEST or row["n_original_sequences"]!=N_TEST:
            raise ValueError("source per-token pseudo-replication or missing sample")
        keys=set(cases[0])
        means={name:sum(v[name] for v in cases)/N_TEST for name in keys}
        if set(means)!=set(row["means"]):
            raise ValueError("original source metric missing from independent replay")
        for name,val in means.items():
            close(val,row["means"][name],f"trajectory_{sid}_{arm}_{k}_{name}",tol=1e-8)
        for idx,v in enumerate(cases):
            close((v["shared_E_before"]-v["shared_E_after"])/81,
                  v["shared_energy_drop"],"native same-energy decrease",tol=1e-8)
            close((v["initial_E_before"]-v["initial_E_after"])/81,
                  v["initial_energy_drop"],"initial source energy decrease",tol=1e-8)
            close((v["radial_E_before"]-v["radial_E_after"])/81,
                  v["radial_energy_drop"],"radial source energy decrease",tol=1e-8)
            if k<4:
                close((v["conditioned_E_before"]-v["conditioned_E_after"])/81,
                      v["conditioned_energy_drop"],"original depth-conditioned energy",tol=1e-8)
            if v["shared_positive_drop"]!=float(v["shared_E_before"]>v["shared_E_after"]):
                raise ValueError("source strict energy sign changed")
            close(v["gain_vs_initial"],
                  v["shared_cosine"]-v["initial_cosine"],"matched local gradient advantage")
            close(v["gain_vs_radial"],
                  v["shared_cosine"]-v["radial_cosine"],"radial source gradient advantage")
            for c in ("shared_cosine","initial_cosine","radial_cosine"):
                if abs(v[c])>1.00001:raise ValueError("source cosine not normalized")
            if not 0<=v["accuracy_before"]<=1 or not 0<=v["accuracy_after"]<=1:
                raise ValueError("source 10-class accuracy invalid")
            if v["CE_before"]<0 or v["CE_after"]<0:
                raise ValueError("source 10-class CE invalid")
            if k>0:
                previous=curves[str(k-1)]["per_sequence"][idx]
                close(previous["shared_E_after"],v["shared_E_before"],
                      "state energy continuity across genuine tied-loop",tol=1e-7)
                close(previous["CE_after"],v["CE_before"],
                      "source 10-class task CE continuity",tol=2e-6)
        fields=("shared_energy_drop","shared_positive_drop")
        if k<4:fields+=("gain_vs_initial","gain_vs_radial")
        if set(row["bootstrap95"])!=set(fields):
            raise ValueError("source registered 12-loop bootstrap comparison list changed")
        positive=True
        for name in fields:
            data=[v[name] for v in cases]
            ci=bootstrap(data,stable_seed(46058,sid,arm,k,name))
            for x,y in zip(ci,row["bootstrap95"][name]):
                close(x,y,"original source-string 2000-bootstrap interval",tol=1e-8)
            minimum=.90 if name=="shared_positive_drop" else 0.
            positive=positive and ci[0]>minimum
        if bool(positive)!=row["pass"]:
            raise ValueError("preregistered source depth decision changed")
        if k<4:early+=int(positive)
        else:late+=int(positive)
    early_all=sum(int(all(curves[str(k)]["per_sequence"][i]["shared_positive_drop"]==1.
                          for k in range(4))) for i in range(N_TEST))/N_TEST
    late_all=sum(int(all(curves[str(k)]["per_sequence"][i]["shared_positive_drop"]==1.
                         for k in range(4,12))) for i in range(N_TEST))/N_TEST
    all12=sum(int(all(curves[str(k)]["per_sequence"][i]["shared_positive_drop"]==1.
                        for k in range(12))) for i in range(N_TEST))/N_TEST
    for a,b,name in (
        (early,cell["n_early_passes"],"early passes"),
        (late,cell["n_extra_passes"],"late passes")):
        if a!=b:raise ValueError("source registered depth coverage changed "+name)
    for a,b,name in (
        (early_all,cell["fraction_early_all_positive"],"early all-positive"),
        (late_all,cell["fraction_extra_all_positive"],"late all-positive"),
        (all12,cell["fraction_all12_positive"],"all12-positive")):
        close(a,b,name,tol=1e-9)
    read4=curves["3"]["means"]
    read12=curves["11"]["means"]
    for a,b,name in (
        (read4["accuracy_after"],cell["readout4_accuracy"],"accuracy4"),
        (read12["accuracy_after"],cell["readout12_accuracy"],"accuracy12"),
        (read4["CE_after"],cell["readout4_CE"],"CE4"),
        (read12["CE_after"],cell["readout12_CE"],"CE12")):
        close(a,b,name,tol=1e-9)
    geometry=(early>=3 and late>=6)
    task=(cell["readout4_accuracy"]>=.60 and
          cell["readout12_accuracy"]>=cell["readout4_accuracy"]-.05 and
          cell["readout12_CE"]<=cell["readout4_CE"]+.05)
    if (geometry!=cell["registered_geometric_pass"] or
        task!=cell["registered_task_preservation_pass"] or
        bool(geometry and task)!=cell["registered_cell_pass"]):
        raise ValueError("source preregistered geom/task conjunction changed")
    return {"early":early,"late":late,"geometry":geometry,"task":task,
            "all12_monotonic":all12,
            "acc4":cell["readout4_accuracy"],"acc12":cell["readout12_accuracy"],
            "CE4":cell["readout4_CE"],"CE12":cell["readout12_CE"]}


def logsumexp(z):
    peak=max(z)
    return peak+math.log(sum(math.exp(t-peak) for t in z))


def replay_native(cell,sid,arm,k):
    record=cell["native_interventions"][str(k)]
    cases=record["raw_cases"]
    if (record["source_sequences"]!=N_NATIVE or len(cases)!=N_NATIVE
        or record["original_goals"]!=N_NATIVE*len(GOALS)
        or record["native_target_readout"]!=(4 if k==2 else 12)):
        raise ValueError("original fixed native source-goal and horizon roster changed")
    rows=[]
    admitted_count=0
    edited=0
    for i,sample in enumerate(cases):
        if sample["source_index"]!=i or sample["native_endpoint"]!=(4 if k==2 else 12):
            raise ValueError("native source sample endpoint/horizon changed")
        goals=sample["original_goals"]
        if len(goals)!=len(GOALS):
            raise ValueError("source eight goal positions not fixed")
        admitted=[]
        for j,g in enumerate(goals):
            if (g["slot"]!=j or g["token"]!=GOALS[j] or g["dose"]<0):
                raise ValueError("source fixed native goal or physical dose changed")
            if not g["source_admitted"]:
                if g["arms"] or "reason" not in g:
                    raise ValueError("source-ineligible physical direction scored")
                continue
            admitted_count+=1
            if set(g["arms"])!=set(EDIT_ARMS):
                raise ValueError("source matched physical edit direction missing")
            digit_set=set()
            for name,r in g["arms"].items():
                edited+=1
                if (abs(r["edit_L2"]-g["dose"])>1e-5
                    or len(r["edit_sha256"])!=64):
                    raise ValueError("matched source physical edit dose/hash altered")
                truth=r["truth_digit"]
                clean=r["clean_goal_logits"];final=r["goal_logits"]
                if not 0<=truth<=9 or len(clean)!=10 or len(final)!=10:
                    raise ValueError("original full native 10-logit task head missing")
                rival=max((v for v in range(10) if v!=truth),key=lambda v:clean[v])
                if rival!=r["clean_rival"]:
                    raise ValueError("original strongest clean output competitor changed")
                if len(r["all_final_token_CE"])!=81 or len(r["clean_all_final_token_CE"])!=81:
                    raise ValueError("genuine 81-output task CE lost")
                ce=logsumexp(final)-final[truth]
                clean_ce=logsumexp(clean)-clean[truth]
                close(ce,r["all_final_token_CE"][g["token"]],"native goal CE",tol=3e-5)
                close(clean_ce,r["clean_all_final_token_CE"][g["token"]],"native clean goal CE",tol=3e-5)
                close(ce-clean_ce,r["goal_CE_change"],"native CE intervention",tol=3e-5)
                close(final[truth]-final[rival]-clean[truth]+clean[rival],
                      r["goal_margin_change"],"native strongest clean-rival margin",tol=3e-5)
                actual=sum(r["all_final_token_CE"])/81
                baseline=sum(r["clean_all_final_token_CE"])/81
                close(actual-baseline,r["global_CE_change"],"native final whole-task CE",tol=2e-6)
                close((r["shared_E_before"]-r["shared_E_after"])/81,
                      r["edit_shared_energy_drop"],"external edit measured scalar energy",tol=2e-6)
                if not math.isfinite(r["shared_firstorder_energy_drop"]):
                    raise ValueError("nonfinite external scalar predicted energy change")
                digit_set.add(truth)
            if len(digit_set)!=1:
                raise ValueError("original goal different true label among source arms")
            admitted.append(g)
        diffs={}
        for a,b in CONTRASTS:
            key=f"{a}_minus_{b}"
            vals=[r["arms"][a]["global_CE_change"]-
                  r["arms"][b]["global_CE_change"] for r in admitted]
            diffs[key]=sum(vals)/len(vals) if vals else 0.
        rows.append({"source_index":i,"n_original_goals":8,
                     "n_admitted_goals":len(admitted),"contrasts":diffs})
    if admitted_count!=record["admitted_goals"]:
        raise ValueError("original physical admission count changed")
    if len(record["per_sequence"])!=N_NATIVE:
        raise ValueError("source causal cluster receipt incomplete")
    for a,b in zip(rows,record["per_sequence"]):
        if a["n_admitted_goals"]!=b["n_admitted_goals"]:
            raise ValueError("source matched causal cluster admission changed")
        for key,value in a["contrasts"].items():
            close(value,b["contrasts"][key],"native source-cluster paired global CE",tol=1e-8)
    for a,b in CONTRASTS:
        key=f"{a}_minus_{b}"
        vals=[row["contrasts"][key] for row in rows]
        ci=bootstrap(vals,stable_seed("causal",sid,arm,k,key))
        result=record["contrasts"][key]
        close(sum(vals)/N_NATIVE,result["mean"],"native causal global CE mean",tol=1e-8)
        for x,y in zip(ci,result["bootstrap95"]):
            close(x,y,"native source 64-cluster 2000-bootstrap",tol=1e-8)
        sign="POSITIVE" if ci[0]>0 else "NEGATIVE" if ci[1]<0 else "INCONCLUSIVE"
        if sign!=result["direction"]:
            raise ValueError("predeclared native contrast sign changed")
    return {"original_goals":N_NATIVE*len(GOALS),
            "admitted":admitted_count,"native_edited_arms":edited,
            "causal_contrasts":record["contrasts"]}


def replay_curl(cell,k):
    cohort=cell["native_curl"][str(k)]
    cases=cohort["per_sequence"]
    if len(cases)!=N_NATIVE or cohort["n_source_sequences"]!=N_NATIVE:
        raise ValueError("original source curl sample denominator changed")
    vals=[]
    for i,p in enumerate(cases):
        if p["source_index"]!=i:
            raise ValueError("curl source sample selected posthoc")
        a=p["u_dot_J_w"];b=p["w_dot_J_u"]
        close(a-b,p["skew"],"original native F Jacobian skew")
        ratio=abs(a-b)/(abs(a)+abs(b)+1e-8)
        close(ratio,p["skew_ratio"],"normalized native field curl")
        vals.append(ratio)
    close(sum(vals)/N_NATIVE,cohort["mean_skew_ratio"],"source curl mean")
    return sum(vals)/N_NATIVE


def audit(source_raw,source_meta_raw,energy_raw,energy_meta_raw,
          lock_raw,test_raw):
    test=verify_sources(source_raw,source_meta_raw,energy_raw,
                        energy_meta_raw,lock_raw,test_raw)
    findings=[]
    for sid in TEACHERS:
        for arm in ARMS:
            cell=test["results"][str(sid)][arm]
            summary=replay_profile(cell,sid,arm)
            natives={}
            curls={}
            for k in (2,6):
                natives[str(k)]=replay_native(cell,sid,arm,k)
                curls[str(k)]=replay_curl(cell,k)
            findings.append({"teacher":sid,"arm":arm,"trajectory":summary,
                             "native":natives,"curl":curls})
    geometric=0
    robust=0
    for sid in TEACHERS:
        group=[x for x in findings if x["teacher"]==sid]
        geo=all(x["trajectory"]["geometry"] for x in group)
        task=all(x["trajectory"]["task"] for x in group)
        joint=geo and task
        expected=test["teacher_summary"][str(sid)]
        if expected!={"all_four_geometric":geo,"all_four_task":task,
                     "all_four_joint":joint}:
            raise ValueError("registered global teacher robustness classification changed")
        geometric+=int(geo)
        robust+=int(joint)
    if (geometric!=test["n_all_arm_geometric_teacher_passes"] or
        robust!=test["n_all_arm_robust_teacher_passes"] or
        test["registered_primary_verdict"]!=("PASS" if robust>=2 else "FAIL")):
        raise ValueError("source program-wide preregistered OOD verdict changed")
    return {
        "schema":"LGA-06/independent-stdlib-12loop-OOD-native-audit-v1",
        "audit_status":"PASS",
        "scientific_status":test["registered_primary_verdict"],
        "original_source_sha256":digest(source_raw),
        "original_energy_sha256":digest(energy_raw),
        "pretest_source_lock_sha256":digest(lock_raw),
        "new_science_receipt_sha256":digest(test_raw),
        "n_independent_base_sequences":N_TEST,
        "teacher_seeds":list(TEACHERS),"distributions":list(ARMS),
        "n_geometric_transport_seeds":geometric,
        "n_robust_task_and_geom_seeds":robust,
        "teacher_arm_results":findings,
        "limitations":"Replays original source identities, same-potential energies, task CE/accuracy, 2000 sequence bootstraps, native ten logits and curl algebra. Does not reexecute original trained forward or autograd.",
    }


def main():
    parser=argparse.ArgumentParser()
    for name in ("source-bundle","source-meta","energy-bundle","energy-meta",
                 "lock","test-receipt","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    a=parser.parse_args()
    result=audit(a.source_bundle.read_bytes(),a.source_meta.read_bytes(),
                 a.energy_bundle.read_bytes(),a.energy_meta.read_bytes(),
                 a.lock.read_bytes(),a.test_receipt.read_bytes())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("LGA06_INDEPENDENT_REPLAY_PASS",result["scientific_status"],
          result["n_geometric_transport_seeds"],
          result["n_robust_task_and_geom_seeds"],flush=True)


if __name__=="__main__":
    main()
