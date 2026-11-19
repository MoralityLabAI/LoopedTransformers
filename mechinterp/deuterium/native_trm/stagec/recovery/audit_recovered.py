# Post-result recovery audit only: corrected unbound `row` in y-shape guard.
# Science-generating source and original failed auditor remain byte-immutable.
"""DEUTERIUM-NATIVE-01 independent no-Torch first-heldout replay auditor.

Recomputes every 81-token ten-class CE and prediction from the immutable
float32 native logits, source hashes/targets, dose receipts, paired 256-string
bootstrap and original registered goal/off-goal endpoints. Never runs a model.
"""
from __future__ import annotations
import argparse
import array
import gzip
import hashlib
import json
import math
import random
import struct
import sys
from pathlib import Path

N=256
GOALS=(0,10,20,30,40,50,60,70)
ARMS=(
    "clean","sham","vpd","random","task_gradient",
    "vpd_half","vpd_double","random_half","random_double",
    "vpd_depth0","vpd_depth1","vpd_depth2","vpd_depth3",
    "random_depth0","random_depth1","random_depth2","random_depth3"
)
SCHEMA="DEUTERIUM-NATIVE-01/first-native-heldout-score-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_sha(rows):
    values=[int(v) for row in rows for v in row]
    return hashlib.sha256(struct.pack("<"+"q"*len(values),*values)).hexdigest()


def bootstrap(paired,seed,draws=2000):
    rng=random.Random(seed)
    n=len(paired)
    q=sorted(math.fsum(paired[rng.randrange(n)] for _ in range(n))/n
             for _ in range(draws))
    return [q[int(.025*draws)],q[int(.975*draws)-1]]


def bootstrap_seed(seed):
    s=f"DEUTERIUM-NATIVE-01|first-score|bootstrap|{seed}".encode()
    return int.from_bytes(hashlib.sha256(s).digest()[:8],"big")


def basic_scores(values,truth):
    largest=max(values)
    logz=largest+math.log(math.fsum(math.exp(v-largest) for v in values))
    ce=logz-values[truth]
    prediction=max(range(10),key=lambda k:values[k])
    return ce, int(prediction==truth)


def recompute(raw,ys):
    floats=array.array("f")
    floats.frombytes(raw)
    if sys.byteorder!="little":
        floats.byteswap()
    expected=len(ARMS)*N*81*10
    if len(floats)!=expected:
        raise ValueError(f"raw logits size {len(floats)} != {expected}")
    metrics={}
    mask=[i for i in range(81) if i not in GOALS]
    for arm_i,arm in enumerate(ARMS):
        result={k:[] for k in (
            "goal_nll","offgoal_nll","full_nll",
            "goal_accuracy","offgoal_accuracy","full_accuracy")}
        offset=arm_i*N*81*10
        for i in range(N):
            losses=[]
            rights=[]
            for token in range(81):
                left=offset+(i*81+token)*10
                values=floats[left:left+10]
                if any(not math.isfinite(z) for z in values):
                    raise ValueError("nonfinite native original model logits")
                ce,correct=basic_scores(values,ys[i][token])
                losses.append(ce)
                rights.append(correct)
            for k,subset in (("goal",GOALS),("offgoal",mask),("full",range(81))):
                result[k+"_nll"].append(math.fsum(losses[j] for j in subset)/len(subset))
                result[k+"_accuracy"].append(math.fsum(rights[j] for j in subset)/len(subset))
        metrics[arm]=result
    # Verify exact identity, including every class logit rather than just scores.
    n=N*81*10
    if floats[:n]!=floats[n:2*n]:
        raise ValueError("zero-edit sham changed native clean logits")
    return metrics


def audit(root,stageb_teacher,test_lock,out):
    root=Path(root)
    stageb_teacher=Path(stageb_teacher)
    scored=json.loads((root/"seed_result.json").read_text())
    sources=json.loads((root/"heldout_sources.json").read_text())
    original=json.loads((root/"scored_metrics.json").read_text())
    exposure=json.loads((root/"heldout_exposure.json").read_text())
    lock=json.loads(Path(test_lock).read_text())
    calibration=json.loads((stageb_teacher/"calibration_sources.json").read_text())
    stageb=json.loads((stageb_teacher/"seed_result.json").read_text())
    errs=[]
    def require(ok,label):
        if not ok:
            errs.append(label)
    require(scored.get("schema")==SCHEMA and
            scored.get("status") in ("SCORED_NATIVE_DEVELOPMENT","BLOCKED_CLEAN_NATIVE_TASK"),
            "score_schema_status")
    seed=scored.get("teacher_seed")
    require(seed in (103,211,307) and
            seed==exposure.get("teacher_seed"), "teacher_roster")
    require(scored.get("first_test_exposed") is True and
            exposure.get("status")=="GENERATION_STARTED" and
            exposure.get("seed")==731057, "original_heldout_exposure")
    require(lock.get("status")=="READY_FOR_FIRST_HELDOUT_RELEASE" and
            scored.get("test_lock_sha256")==sha(test_lock),
            "source_pretest_lock")
    require(scored.get("stageb_selected_edits_sha256")==stageb.get("selected_edits_sha256")
            ==sha(stageb_teacher/"selected_edits.pt"), "trained_selection_frozen_before_test")
    require(sources.get("seed")==731057 and sources.get("n")==N and
            sources.get("no_exact_source_overlap") is True and
            calibration.get("seed")==731055, "first_holdout_roster")
    x,y=sources.get("input",[]),sources.get("target",[])
    require(len(x)==N and len(y)==N, "source_count")
    if len(x)==N and len(y)==N:
        require(all(len(row)==81 and all(type(d) is int and 1<=d<=9 for d in row)
                    for row in x),"original_task_digits")
        require(all(len(y[i])==81 and all(y[i][j]==x[i][(j-1)%81]
                    for j in range(81)) for i in range(N)), "right_cyclic_copy_labels")
        require(len({tuple(row) for row in x})==N and
                not {tuple(row) for row in x}.intersection(
                    {tuple(row) for row in calibration["input"]}),"calibration_heldout_disjoint")
        require(input_sha(x)==sources.get("input_sha256")==
                scored.get("heldout",{}).get("input_sha256"),"input_sha256")
        require(input_sha(y)==sources.get("target_sha256")==
                scored.get("heldout",{}).get("target_sha256"),"target_sha256")
    require(scored.get("arm_order")==list(ARMS) and
            scored.get("n_arms")==len(ARMS) and
            scored.get("logits_shape")==[len(ARMS),N,81,10],
            "full_ten_class_logits_roster")
    require(scored.get("logits_gzip_sha256")==sha(root/"all_arm_logits_f32.gz"),
            "compressed_logits_bytes")
    with gzip.open(root/"all_arm_logits_f32.gz","rb") as f:
        raw=f.read()
    require(scored.get("logits_f32_raw_sha256")==
            hashlib.sha256(raw).hexdigest(),"raw_logits_sha256")
    results=recompute(raw,y)
    max_nll_error=0.
    max_acc_error=0.
    for arm in ARMS:
        for key in results[arm]:
            got=original.get(arm,{}).get(key,[])
            vals=results[arm][key]
            require(len(got)==N,"per_source_"+arm+"_"+key)
            if len(got)==N:
                error=max(abs(a-b) for a,b in zip(vals,got))
                if key.endswith("nll"):
                    max_nll_error=max(max_nll_error,error)
                else:
                    max_acc_error=max(max_acc_error,error)
                require(error<5e-6,"native_float32_scoring_"+arm+"_"+key)
    physical=scored.get("measured_native_edits",{})
    require(set(physical)==set(ARMS), "physical_arms")
    dose=stageb.get("dose",0)
    for arm in ARMS:
        rec=physical.get(arm,{})
        multiplier=0 if arm in ("clean","sham") else (
            .5 if arm.endswith("half") else 2 if arm.endswith("double") else 1)
        expected=dose*multiplier
        actual=rec.get("applied_frobenius")
        require(isinstance(actual,(int,float)) and math.isfinite(actual) and
                abs(actual-expected)<=max(1e-6,1e-4*expected),
                arm+":physical_dose")
        require(rec.get("native_weight_uses")==
                (1 if arm.startswith(("vpd_depth","random_depth")) else 4),
                arm+":native_recurrent_application_count")
        require(rec.get("restored_clean_max_logit_error",float("inf"))<=1e-4 and
                rec.get("native_manual_max_logit_error",float("inf"))<=1e-4,
                arm+":native_replay_restore")
    clean=sum(results["clean"]["full_accuracy"])/N
    passed=clean>=.95
    require(scored.get("clean_accuracy_gate_pass") is passed and
            abs(clean-scored.get("mean_clean_full_accuracy",float("nan")))<1e-6,
            "clean_task_accuracy")
    differences=[a-b for a,b in zip(results["random"]["goal_nll"],
                                     results["vpd"]["goal_nll"])]
    mean=math.fsum(differences)/N
    ci=bootstrap(differences,bootstrap_seed(seed))
    orig_primary=scored.get("primary_random_minus_vpd_goal_nll",{})
    require(abs(mean-orig_primary.get("estimate",float("inf")))<5e-6 and
            max(abs(a-b) for a,b in zip(ci,orig_primary.get("ci95",[float("inf")]*2)))<5e-6,
            "registered_cluster_bootstrap")
    offgap=sum(results["vpd"]["offgoal_accuracy"])/N - sum(
        results["clean"]["offgoal_accuracy"])/N
    require(abs(offgap-scored.get("vpd_offgoal_accuracy_minus_clean",float("inf")))<1e-6,
            "offgoal_collateral_accuracy")
    expected_status="SCORED_NATIVE_DEVELOPMENT" if passed else "BLOCKED_CLEAN_NATIVE_TASK"
    require(scored.get("status")==expected_status and
            scored.get("comparison_completed") is passed,"clean_task_gate")
    outp={
        "schema":"DEUTERIUM-NATIVE-01/first-stdlib-heldout-audit-v1",
        "status":"PASS" if not errs else "FAIL",
        "errors":errs,
        "torch_imported":"torch" in sys.modules,
        "teacher_seed":seed,
        "source_rows_rechecked":N,
        "native_arms_rechecked":len(ARMS),
        "logits_rechecked":len(ARMS)*N*81*10,
        "max_nll_arithmetic_error":max_nll_error,
        "max_accuracy_arithmetic_error":max_acc_error,
        "clean_accuracy":clean,
        "clean_task_gate_pass":passed,
        "random_minus_vpd_goal_nll":mean,
        "paired_256_source_ci95":ci,
        "vpd_offgoal_accuracy_minus_clean":offgap,
        "comparison_completed":passed,
        "evidence_class":"DEVELOPMENT_ONLY_OLD_TEACHER_ROOTS",
    }
    if outp["torch_imported"]:
        outp["status"]="FAIL"
        outp["errors"].append("torch_imported")
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(outp,sort_keys=True,indent=2)+"\n")
    if outp["status"]!="PASS":
        raise RuntimeError("independent native heldout arithmetic failure: "+", ".join(outp["errors"]))
    return outp


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--stageb-teacher",type=Path,required=True)
    p.add_argument("--test-lock",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    r=audit(a.root,a.stageb_teacher,a.test_lock,a.out)
    print("DEUTERIUM_NATIVE_FIRST_TEST_STDLIB_AUDIT",r["status"],flush=True)


if __name__=="__main__":
    main()
