"""DEUTERIUM-NATIVE-01: first native four-pass VPD heldout score.

All directions must arrive from independently audited first Stage B artifacts
and a separately uploaded Stage C source lock. The registered 731057 test is
generated precisely once per original trained root, only after writing an
irreversible exposure marker. Subsequent executions are reproductions.
"""
from __future__ import annotations
import argparse
import contextlib
import gzip
import hashlib
import json
import math
import os
import platform
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mechinterp.loopscope04.gradient_flow import (
    TinyRecursiveModel, fixed_data, tensor_int64_sha
)
from mechinterp.loopscope05.fit_and_freeze import load_frozen
from mechinterp.deuterium.native_trm.preflight import (
    MODEL_SHA, META_SHA, sha256, four_pass_manual
)
from mechinterp.deuterium.native_trm.stagec.source_lock import verify_lock, TEACHERS

SCHEMA="DEUTERIUM-NATIVE-01/first-native-heldout-score-v1"
TEST_SEED=731057
TEST_N=256
GOALS=(0,10,20,30,40,50,60,70)
ARMS=(
    "clean","sham","vpd","random","task_gradient",
    "vpd_half","vpd_double","random_half","random_double",
    "vpd_depth0","vpd_depth1","vpd_depth2","vpd_depth3",
    "random_depth0","random_depth1","random_depth2","random_depth3"
)
MAIN_ARMS=("clean","sham","vpd","random","task_gradient")
MAX_NATIVE_REPLAY_ERROR=1e-4
MAX_DOSE_RELATIVE_ERROR=1e-4
MIN_CLEAN_ACCURACY=0.95
OLD_SOURCES={
    31041:512,31042:128,31043:256,31044:256,
    31048:768,31049:192,31050:256,31052:256,31053:64,31054:256,
    31055:256,31056:64,31057:256,31058:256,
    31060:64,31061:64,31062:64,31063:64
}


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")


def seed_rng(*parts):
    s="DEUTERIUM-NATIVE-01|first-score|"+"|".join(map(str,parts))
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],"big")


def percentile_bootstrap(paired,seed,draws=2000):
    if len(paired)!=TEST_N:
        raise ValueError("source-cluster bootstrap expects exactly 256 independent test strings")
    rng=random.Random(seed)
    n=len(paired)
    distribution=sorted(sum(paired[rng.randrange(n)] for _ in range(n))/n
                        for _ in range(draws))
    return {"estimate":math.fsum(paired)/n,
            "ci95":[distribution[int(.025*draws)],distribution[int(.975*draws)-1]],
            "replicates":draws,"unit":"whole_source_string","n":n}


@contextlib.contextmanager
def edit_weight(model,delta):
    w=model.block.chan_mix[2].weight
    before=w.detach().clone()
    try:
        with torch.no_grad():
            w.copy_(before+delta)
            actual=w.detach().clone()-before
        yield actual
    finally:
        with torch.no_grad():
            w.copy_(before)
        if not torch.equal(w.detach(),before):
            raise RuntimeError("native recurrent parameter restoration failed")


def local_one_pass(model,x,delta,only_depth):
    with torch.no_grad():
        h=model.initial_state(x)
        for k in range(4):
            if k==only_depth:
                with edit_weight(model,delta):
                    h=model.block(h)
            else:
                h=model.block(h)
        return model.head(h)


def logits_for_arm(model,x,arm,edits,dose,chunk=32):
    goal=[]
    physical=[]
    delta=(
        torch.zeros_like(edits["vpd"]) if arm in ("clean","sham") else
        edits["vpd"] if arm=="vpd" else
        edits["random"] if arm=="random" else
        edits["task_gradient"] if arm=="task_gradient" else
        edits["vpd"]*0.5 if arm=="vpd_half" else
        edits["vpd"]*2 if arm=="vpd_double" else
        edits["random"]*0.5 if arm=="random_half" else
        edits["random"]*2 if arm=="random_double" else
        edits["vpd"] if arm.startswith("vpd_depth") else
        edits["random"] if arm.startswith("random_depth") else None
    )
    if delta is None:
        raise ValueError("unexpected predeclared arm")
    multiplier=0 if arm in ("clean","sham") else (
        0.5 if arm.endswith("half") else 2 if arm.endswith("double") else 1)
    expected=dose*multiplier
    max_native_error=0.
    max_restore_error=0.
    with torch.no_grad():
        original_weight=model.block.chan_mix[2].weight.detach().clone()
        for i in range(0,len(x),chunk):
            xx=x[i:i+chunk]
            baseline=model(xx)
            if arm.startswith(("vpd_depth","random_depth")):
                k=int(arm[-1])
                # Only the k-th tied block sees the edit in this exploratory arm.
                out=local_one_pass(model,xx,delta,k)
                actual=(original_weight+delta)-original_weight
            else:
                with edit_weight(model,delta) as actual:
                    out=model(xx)
                    check=four_pass_manual(model,xx[:1])
                    max_native_error=max(max_native_error,float((out[:1]-check).abs().max()))
            restored=model(xx)
            max_restore_error=max(max_restore_error,float((restored-baseline).abs().max()))
            if not torch.equal(model.block.chan_mix[2].weight,original_weight):
                raise RuntimeError("native recurrent parameter changed after an arm")
            if not bool(torch.isfinite(out).all()):
                raise RuntimeError("nonfinite four-pass native output")
            if tuple(out.shape)!=(len(xx),81,10):
                raise RuntimeError("wrong original native output shape")
            goal.append(out.detach().cpu().contiguous().float())
            physical.append(float(actual.norm()))
    logits=torch.cat(goal,0)
    actual=physical[0]
    rel=0 if expected==0 else abs(actual-expected)/expected
    if (max_native_error>MAX_NATIVE_REPLAY_ERROR or
        max_restore_error>MAX_NATIVE_REPLAY_ERROR or
        rel>MAX_DOSE_RELATIVE_ERROR or
        any(abs(p-actual)>1e-8 for p in physical)):
        raise RuntimeError("native tied-weight replay, restoration or applied dose failed")
    if arm=="sham" and not torch.equal(logits,logits_for_clean(model,x,chunk)):
        raise RuntimeError("sham/clean native identity failed")
    return logits,{
        "arm":arm,"expected_frobenius":expected,
        "applied_frobenius":actual,"applied_relative_dose_error":rel,
        "native_manual_max_logit_error":max_native_error,
        "restored_clean_max_logit_error":max_restore_error,
        "native_weight_uses":1 if arm.startswith(("vpd_depth","random_depth")) else 4,
    }


def logits_for_clean(model,x,chunk=32):
    with torch.no_grad():
        return torch.cat([model(x[i:i+chunk]).cpu().contiguous().float()
                          for i in range(0,len(x),chunk)],0)


def per_source_metrics(logits,y):
    with torch.no_grad():
        ce=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1),
                           reduction="none").reshape(len(y),81)
        right=logits.argmax(-1).eq(y).float()
        mask=torch.ones(81,dtype=torch.bool)
        mask[list(GOALS)]=False
        return {
            "goal_nll":[float(v) for v in ce[:,GOALS].mean(-1)],
            "offgoal_nll":[float(v) for v in ce[:,mask].mean(-1)],
            "full_nll":[float(v) for v in ce.mean(-1)],
            "goal_accuracy":[float(v) for v in right[:,GOALS].mean(-1)],
            "offgoal_accuracy":[float(v) for v in right[:,mask].mean(-1)],
            "full_accuracy":[float(v) for v in right.mean(-1)],
        }


def heldout_sources(calibration_path):
    x,y=fixed_data(TEST_N,TEST_SEED)
    cal=json.loads(Path(calibration_path).read_text())
    if cal.get("seed")!=731055 or len(cal.get("input",[]))!=256:
        raise RuntimeError("registered first calibration cohort missing")
    seen={tuple(row) for row in cal["input"]}
    for sd,n in OLD_SOURCES.items():
        older,_=fixed_data(n,sd)
        seen.update(map(tuple,older.tolist()))
    rows=list(map(tuple,x.tolist()))
    if len(set(rows))!=TEST_N or any(row in seen for row in rows):
        raise RuntimeError("protected first scored cohort reuses a source")
    return x,y,{
        "seed":TEST_SEED,"n":TEST_N,
        "input_sha256":tensor_int64_sha(x),"target_sha256":tensor_int64_sha(y),
        "old_generated_source_seeds":list(OLD_SOURCES),
        "excluded_reserved_source_seeds":[31059,31064,31065],
        "calibration_seed":731055,
        "no_exact_source_overlap":True,
    }


def run(seed,teacher_bundle,teacher_meta,stageb_root,audit_root,stageb_lock,
        test_lock,out):
    t0=time.perf_counter()
    torch.set_num_threads(2)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    record={
        "schema":SCHEMA,"status":"VERIFYING_SOURCE_LOCK",
        "teacher_seed":seed,"first_test_exposed":False,
        "comparison_completed":False,"heldout_source_seed":TEST_SEED,
        "original_teacher_bundle_sha256":MODEL_SHA,
        "original_teacher_meta_sha256":META_SHA,
        "torch":torch.__version__,"python":platform.python_version(),
        "github_sha":os.environ.get("GITHUB_SHA"),
    }
    write_json(out/"progress.json",record)
    lock=verify_lock(test_lock,teacher_bundle,teacher_meta,
                     stageb_root,audit_root,stageb_lock)
    if seed not in TEACHERS:
        raise RuntimeError("unregistered teacher root")
    sd=Path(stageb_root)/f"seed_{seed}"
    stageb=json.loads((sd/"seed_result.json").read_text())
    if sha256(sd/"selected_edits.pt")!=stageb["selected_edits_sha256"]:
        raise RuntimeError("selected edit bytes changed")
    _,bundle=load_frozen(teacher_bundle,teacher_meta)
    model=TinyRecursiveModel(48,4)
    model.load_state_dict(bundle["models"][str(seed)],strict=True)
    model.eval().requires_grad_(False)
    original=model.block.chan_mix[2].weight.detach().clone()
    edits=torch.load(sd/"selected_edits.pt",map_location="cpu",weights_only=True)
    if set(edits)!={"sham","vpd","random","task_gradient"}:
        raise RuntimeError("selected edit arm roster drift")
    dose=float(stageb["dose"])
    if abs(float(original.norm())*.01-dose)>1e-5:
        raise RuntimeError("calibration original teacher target weight dose changed")
    for arm in ("vpd","random","task_gradient"):
        d=edits[arm]
        if (tuple(d.shape)!=(48,128) or not bool(torch.isfinite(d).all()) or
            abs(float(d.norm())-dose)/dose>MAX_DOSE_RELATIVE_ERROR):
            raise RuntimeError("frozen selected original physical edit drift")
    if not torch.equal(edits["sham"],torch.zeros_like(original)):
        raise RuntimeError("zero-edit sham changed")
    record["stageb_selected_edits_sha256"]=stageb["selected_edits_sha256"]
    record["test_lock_sha256"]=sha256(test_lock)
    record["stage"]="DIRECTIONS_FROZEN_BEFORE_NEW_SOURCE"
    write_json(out/"progress.json",record)

    # This is the irreversible science exposure boundary. Mark it before calling
    # fixed_data on the registered first heldout source.
    write_json(out/"heldout_exposure.json",{
        "schema":SCHEMA,"teacher_seed":seed,
        "status":"GENERATION_STARTED","seed":TEST_SEED,
        "new_source_labels_now_considered_exposed":True,
        "test_lock_sha256":record["test_lock_sha256"],
        "selected_edits_sha256":record["stageb_selected_edits_sha256"],
    })
    record["first_test_exposed"]=True
    write_json(out/"progress.json",record)

    x,y,roster=heldout_sources(sd/"calibration_sources.json")
    write_json(out/"heldout_sources.json",{
        **roster,"input":x.tolist(),"target":y.tolist()
    })
    logits=[]
    metrics={}
    edits_physical={}
    for arm in ARMS:
        outputs,info=logits_for_arm(model,x,arm,edits,dose)
        logits.append(outputs)
        metrics[arm]=per_source_metrics(outputs,y)
        edits_physical[arm]=info
        record["stage"]="SCORED_"+arm.upper()
        write_json(out/"progress.json",record)
    all_logits=torch.stack(logits).detach().cpu().contiguous().numpy().astype("<f4",copy=False)
    raw=all_logits.tobytes(order="C")
    raw_hash=hashlib.sha256(raw).hexdigest()
    with (out/"all_arm_logits_f32.gz").open("wb") as f:
        with gzip.GzipFile(fileobj=f,mode="wb",compresslevel=6,mtime=0) as gz:
            gz.write(raw)
    clean_avg=sum(metrics["clean"]["full_accuracy"])/len(x)
    baseline_pass=clean_avg>=MIN_CLEAN_ACCURACY
    target=[a-b for a,b in zip(metrics["random"]["goal_nll"],
                                metrics["vpd"]["goal_nll"])]
    primary=percentile_bootstrap(target,seed_rng("bootstrap",seed))
    collateral=sum(metrics["vpd"]["offgoal_accuracy"])/len(x) - (
        sum(metrics["clean"]["offgoal_accuracy"])/len(x))
    record.update({
        "status":"SCORED_NATIVE_DEVELOPMENT" if baseline_pass else "BLOCKED_CLEAN_NATIVE_TASK",
        "comparison_completed":bool(baseline_pass),
        "heldout":roster,
        "n_arms":len(ARMS),"arm_order":list(ARMS),
        "logits_shape":list(all_logits.shape),
        "logits_f32_raw_sha256":raw_hash,
        "logits_gzip_sha256":sha256(out/"all_arm_logits_f32.gz"),
        "mean_clean_full_accuracy":clean_avg,
        "clean_accuracy_gate_pass":baseline_pass,
        "primary_random_minus_vpd_goal_nll":primary,
        "vpd_offgoal_accuracy_minus_clean":collateral,
        "measured_native_edits":edits_physical,
        "runtime_seconds":time.perf_counter()-t0,
        "scope":"DEVELOPMENT_ONLY_ALREADY_EXAMINED_THREE_TEACHER_ROOTS",
    })
    write_json(out/"scored_metrics.json",metrics)
    write_json(out/"seed_result.json",record)
    write_json(out/"progress.json",record)
    return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--teacher-seed",type=int,required=True)
    p.add_argument("--teacher-bundle",type=Path,required=True)
    p.add_argument("--teacher-meta",type=Path,required=True)
    p.add_argument("--stageb-root",type=Path,required=True)
    p.add_argument("--audit-root",type=Path,required=True)
    p.add_argument("--stageb-lock",type=Path,required=True)
    p.add_argument("--test-lock",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    try:
        r=run(a.teacher_seed,a.teacher_bundle,a.teacher_meta,a.stageb_root,
              a.audit_root,a.stageb_lock,a.test_lock,a.out)
    except Exception as exc:
        write_json(a.out/"failure.json",{
            "schema":SCHEMA,
            "status":"ENGINEERING_OR_PREREQUISITE_FAILURE",
            "test_seed_exposed":(a.out/"heldout_exposure.json").is_file(),
            "exception_type":type(exc).__name__,
            "error":str(exc),"traceback":traceback.format_exc()
        })
        raise
    print("DEUTERIUM_NATIVE_FIRST_HELDOUT",r["status"],flush=True)


if __name__=="__main__":
    main()
