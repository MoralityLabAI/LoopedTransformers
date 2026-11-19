"""Audit a completed source schedule using only the Python standard library.

Exit zero means the recorded measurement and its PASS or BLOCKED decision
agree. It does not promote a blocked trajectory to a successful parity gate.
Raw parameter/gradient differences are recomputed from portable FP32 vectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path

MODES = ("stock","last_repeat","native_pass")
TOL = 2e-5
UPSTREAM_SHA = "2df0092d5be4d8e4748b3c66ef8325dedef17549"
SCHEMA = "DEUTERIUM-NATIVE-02/source-schedule-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    def reject(value):
        raise ValueError("nonfinite JSON constant: "+value)
    result = json.loads(Path(path).read_text(),parse_constant=reject)
    def check(value):
        if isinstance(value,float):
            require(math.isfinite(value),"nonfinite JSON value")
        elif isinstance(value,dict):
            for child in value.values():
                check(child)
        elif isinstance(value,list):
            for child in value:
                check(child)
    check(result)
    return result


def sha(value):
    return hashlib.sha256(value).hexdigest()


def vectors(folder, trace):
    spec = trace["tensor_trace"]
    require(spec["file"]=="optimizer_trace.f32","unexpected tensor archive path")
    raw = (folder/spec["file"]).read_bytes()
    require(sha(raw)==spec["sha256"],"tensor archive digest mismatch")
    result = {}
    position = 0
    for entry in spec["entries"]:
        phase,step,field = entry["phase"],entry["step"],entry["field"]
        count = 2816 if phase=="warmup" else 3600
        require(entry["count"]==count,"optimizer vector dimension changed")
        require(entry["offset"]==position,"tensor archive overlap or gap")
        end = position+4*count
        data = raw[position:end]
        require(len(data)==4*count and sha(data)==entry["sha256"],"vector digest mismatch")
        value = struct.unpack("<"+str(count)+"f",data)
        require(all(math.isfinite(x) for x in value),"nonfinite optimizer vector")
        key = (phase,step,field)
        require(key not in result,"duplicate optimizer vector")
        result[key] = value
        position = end
    require(position==len(raw),"trailing optimizer trace bytes")
    expected = {(phase,step,field) for phase,n in (("warmup",32),("main",64))
                for step in range(n) for field in ("parameters","gradients")}
    expected.update({("terminal",64,"terminal_gradients"),("terminal",64,"terminal_parameters")})
    require(set(result)==expected,"incomplete optimizer trace")
    return result


def error(a,b):
    require(len(a)==len(b),"paired optimizer dimensions differ")
    # The measurement used Torch FP32 subtraction before taking the maximum.
    return max(abs(struct.unpack("<f",struct.pack("<f",x-y))[0]) for x,y in zip(a,b))


def classify(errors):
    require(set(errors)=={"warmup_parameters","warmup_gradients","main_parameters",
                         "main_gradients","terminal_gradients","losses"},"error family changed")
    require(all(math.isfinite(x) and x>=0 for x in errors.values()),"invalid parity error")
    return ("SOURCE_ONLY_FULL_SCHEDULE_PASS" if all(x<=TOL for x in errors.values())
            else "BLOCKED_UNMATCHED_OPTIMIZER")


def audit(out: Path):
    record = load(out/"schedule_receipt.json")
    require(record["schema"]==SCHEMA,"unexpected schedule schema")
    require(record["source_seed"]==61077 and record["source_n"]==512,"excluded source changed")
    require(record["trained_teacher_used"] is False,"trained teacher was used")
    require(record["scientific_evaluation"] is False,"scientific scoring claimed")
    require(record["reserved_teacher_roots_opened"]==[] and record["reserved_data_splits_opened"]==[],
            "reserved root exposure declared")
    require(record["upstream_commit"]==UPSTREAM_SHA and record["tolerance"]==TOL,"frozen source or tolerance changed")
    require(set(record["arms"])==set(MODES),"missing source-only arm")
    traces,values,configs = {},{},{}
    for mode in MODES:
        folder = out/mode
        trace = load(folder/"trace.json")
        require(trace["mode"]==mode and trace["status"]=="COMPLETED","arm did not complete")
        require(len(trace["losses"])==65,"expected65 loss evaluations including final non-update")
        require([row["step"] for row in trace["losses"]]==list(range(65)),"main loss order changed")
        require([(row["phase"],row["update"]) for row in trace["updates"]]
                ==[(p,i+1) for p,n in (("warmup",32),("main",64)) for i in range(n)],
                "warmup or main update count/order changed")
        require([row["after_loss_step"] for row in trace["evaluations"]]==[0,64],"evaluation schedule changed")
        require(len(trace["adversaries"])==67 and all(row["shape"]==[8,81,17]
                for row in trace["adversaries"]),"PGD draw count or dimensionality changed")
        reported = record["arms"][mode]
        require(reported["counts"]=={"warmup":32,"main":64} and reported["main_loss_evaluations"]==65
                and reported["adversary_draws"]==67 and reported["evaluation_steps"]==[0,64],
                "receipt schedule disagrees with traces")
        for update in trace["updates"]:
            require(update["lr"]==.001,"learning rate changed")
            require(update["weight_decay"]==(.1 if update["phase"]=="warmup" else 0),"optimizer decay changed")
        for key in ("teacher_sha256","initial_state_sha256","terminal_state_sha256"):
            require(re.fullmatch("[0-9a-f]{64}",trace[key]) is not None,"invalid model digest")
            require(trace[key]==record["arms"][mode][key],"receipt/model identity mismatch")
        values[mode] = vectors(folder,trace)
        traces[mode] = trace
        configs[mode] = load(folder/"config.json")
    cfg = configs["stock"]
    require(all(value==cfg for value in configs.values()),"paired configs differ")
    require((cfg["steps"],cfg["faithfulness_warmup_steps"],cfg["batch_size"],cfg["sampling"])
            ==(64,32,8,"continuous"),"original config schedule changed")
    require(cfg["faithfulness_warmup_weight_decay"]==.1,"original warmup decay changed")
    require(cfg["faithfulness_warmup_lr"]==.001 and cfg["seed"]==61077,"warmup rate or smoke seed changed")
    require(cfg["sigmoid_type"]=="leaky_hard" and cfg["use_delta_component"] is True,
            "original sigmoid or residual component changed")
    require(cfg["autocast_bf16"] is False and cfg["n_mask_samples"]==1,"precision or mask budget changed")
    schedule = cfg["lr_schedule"]
    require(schedule["start_val"]==.001 and schedule["fn_type"]=="constant" and schedule["warmup_pct"]==0,
            "original main learning rate schedule changed")
    require(len(cfg["module_info"])==1 and cfg["module_info"][0]["module_pattern"]=="block.chan_mix.2"
            and cfg["module_info"][0]["C"]==16 and cfg["identity_module_info"] is None,
            "target layer or component budget changed")
    require(cfg["ci_config"]["mode"]=="layerwise" and cfg["ci_config"]["fn_type"]=="mlp"
            and cfg["ci_config"]["hidden_dims"]==[16],"shared CI architecture changed")
    require(cfg["grad_clip_norm_components"] is None and cfg["grad_clip_norm_ci_fns"] is None,"clipping changed")
    expected_coeffs = {"FaithfulnessLoss":1.0,"ImportanceMinimalityLoss":.001,"PGDReconLoss":1.0}
    require(len(cfg["loss_metric_configs"])==len(expected_coeffs),"duplicate or missing loss configuration")
    require({x["classname"]:x["coeff"] for x in cfg["loss_metric_configs"]}==expected_coeffs,"loss coefficients changed")
    require(next(c for c in cfg["loss_metric_configs"] if c["classname"]=="ImportanceMinimalityLoss")["beta"]==0,
            "minimality normalization changed")
    pgd = next(c for c in cfg["loss_metric_configs"] if c["classname"]=="PGDReconLoss")
    require(pgd["n_steps"]==2 and pgd["step_size"]==.25 and pgd["init"]=="random"
            and pgd["mask_scope"]=="unique_per_datapoint","original PGD schedule changed")
    stock = traces["stock"]
    for trace in traces.values():
        require(trace["teacher_sha256"]==stock["teacher_sha256"] and
                trace["initial_state_sha256"]==stock["initial_state_sha256"],"paired initialization differs")
        require(trace["adversaries"]==stock["adversaries"],"PGD initialization schedule differs")
        for key in ("batch_sha256","rng_before","rng_after"):
            require([r[key] for r in trace["losses"]]==[r[key] for r in stock["losses"]],"source/RNG schedule differs")
    errors = {}
    for phase,n in (("warmup",32),("main",64)):
        for field in ("parameters","gradients"):
            errors[phase+"_"+field] = max(error(values["stock"][(phase,step,field)],
                values["last_repeat"][(phase,step,field)]) for step in range(n))
    key = ("terminal",64,"terminal_gradients")
    errors["terminal_gradients"] = error(values["stock"][key],values["last_repeat"][key])
    errors["losses"] = max(abs(a["losses"][name]-b["losses"][name])
        for a,b in zip(stock["losses"],traces["last_repeat"]["losses"])
        for name in expected_coeffs)
    require(errors==record["forced_equal_max_abs_errors"],"reported parity error does not match raw traces")
    require(record["status"]==classify(errors),"recorded status contradicts fixed tolerance")
    if record["status"]=="BLOCKED_UNMATCHED_OPTIMIZER":
        require(record["blocking_gate"]=="FULL_TRAJECTORY_IDENTITY_TOLERANCE","unrelated failure cannot pass measurement audit")
    comparison = load(out/"trajectory_comparison.json")
    require(sha((out/"trajectory_comparison.json").read_bytes())==record["trajectory_comparison_sha256"],"comparison digest mismatch")
    require(len(comparison["updates"])==96 and comparison["tolerance"]==TOL,"comparison schedule changed")
    expected_rows = []
    for phase,n in (("warmup",32),("main",64)):
        for step in range(n):
            row = {"phase":phase,"step":step}
            for field in ("parameters","gradients"):
                row[field] = error(values["stock"][(phase,step,field)],values["last_repeat"][(phase,step,field)])
            if phase=="main":
                row["raw_losses"] = {name:abs(stock["losses"][step]["losses"][name]-
                    traces["last_repeat"]["losses"][step]["losses"][name]) for name in expected_coeffs}
            expected_rows.append(row)
    require(comparison["updates"]==expected_rows and comparison["first_main_update"]==expected_rows[32],
            "per-update comparison does not match raw vectors")
    return {"schema":SCHEMA+"/audit","measurement_audit":"PASS","parity_status":record["status"],
        "recomputed_max_abs_errors":errors,"reserved_roots_opened":False,
        "source_receipt_sha256":sha((out/"schedule_receipt.json").read_bytes()),
        "scope":"raw recorded optimizer vectors, loss arithmetic and schedule receipts; source-only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.out)
    (args.out/"schedule_audit.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("SOURCE_SCHEDULE_MEASUREMENT_AUDIT PASS; PARITY",result["parity_status"],flush=True)


if __name__ == "__main__":
    main()
