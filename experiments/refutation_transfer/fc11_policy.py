"""FC-11: LR-01 selects sequential tactics on real imported MathCan Lean states.

Only a FRESH Lean4.35 compilation of an original theorem with the model's
selected action list can count as an accepted proof. Scratch successor-state
probes contain sorry, are explicitly diagnostic, and never count as proof.
The pinned upstream MathCan source supplies theorem declarations, while
actual state strings are extracted from the elaborator's emitted trace_state.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time

import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/"lean_reasoner"))
from train import TypedLoop
import fc11_scratch as scratch
from research.refutation_bench.fc11 import source_core as core

PREREG="b57e2500e829b12fdfc257077b8e410171c93b98"
TRAIN_REV="2e7e749d4c7f8610a4083f4f8551d084102ea55a"
SOURCE_MATHCAN="c84e2025334e9a967c1297425a490f98d5b3d189"
TRAIN_STATES_SHA="47fae96eac1f9b176519faaa0a64cbf016203b6f23ba56e40a2295e9fedaa97d"
TEST_REV="b821e0d85423606cf971b1f80db8092ede693d81"
TEST_STATES_SHA="a8e7bee63f80c03c640ccdbb9bada1f8d943e33fb403e37ad1fc7dbf536e11ad"

VOCAB=tuple(core.TACTICS)
SEEDS=(17,29)
D=24
MAX_TOKENS=112
VOCAB_BUCKETS=3072
TRAIN_STEPS=300
BATCH=16
MAX_SEARCH_ACTIONS=3


def digest(path:Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenize(s:str)->list[str]:
    if "⊢" not in s:
        raise AssertionError("model must receive actual Lean compiler goal-state text")
    words=re.findall(r"[\w']+|[^\w\s]",s,flags=re.UNICODE)
    if len(words)>MAX_TOKENS:
        half=MAX_TOKENS//2
        words=words[:half]+words[-half:]
    return words


def graph_batch(states:list[str])->dict:
    B=len(states)
    A=MAX_TOKENS
    R=len(VOCAB)
    tokens=torch.zeros((B,A),dtype=torch.long)
    present=torch.zeros((B,A),dtype=torch.float32)
    head=torch.zeros((B,R,A),dtype=torch.float32)
    for b,state in enumerate(states):
        words=tokenize(state)
        for j,word in enumerate(words):
            tokens[b,j]=int.from_bytes(hashlib.blake2b(
                word.encode("utf8"),digest_size=8).digest(),"big")%VOCAB_BUCKETS
            present[b,j]=1.
        idx=words.index("⊢")
        head[b,:,idx]=1.
    prem=present[:,None,:].expand(-1,R,-1).clone()
    return {"tokens":tokens,"present":present,"prem":prem,"head":head}


class ImportedLoop(nn.Module):
    """The ORIGINAL TypedLoop shared recurrent cell, with eight tactic nodes."""
    def __init__(self,loops:int=4):
        super().__init__()
        self.loops=loops
        self.atom=nn.Embedding(VOCAB_BUCKETS,D)
        self.rule=nn.Embedding(len(VOCAB),D)
        self.cell=TypedLoop(D)
        self.score=nn.Linear(3*D,1)

    def forward(self,b:dict):
        z=self.atom(b["tokens"])*b["present"][...,None]
        idx=torch.arange(len(VOCAB),device=z.device)[None,:].expand(z.shape[0],-1)
        r=self.rule(idx)
        for _ in range(self.loops):
            z,r=self.cell(z,r,b["prem"],b["head"])
        parent=b["prem"]@z/b["prem"].sum(-1,keepdim=True).clamp_min(1.)
        head=b["head"]@z
        return self.score(torch.cat((r,parent,head),-1)).squeeze(-1)


def _rows(f:Path):
    return [json.loads(line) for line in f.read_text(encoding="utf8").splitlines() if line.strip()]


def _read_train_original_and_states(source:Path,mathcan:Path,out:Path):
    """Only the train source module is used. Held-out selection is absent."""
    from research.refutation_bench.fc11 import train_data
    if set(train_data.TRAIN_SOURCE)&{"MathCan/Intervention.lean",
           "MathCan/HookState.lean","MathCan/GraphAuthorization.lean"}:
        raise AssertionError("train module selection contains test family")
    out.mkdir(parents=True,exist_ok=True)
    report=train_data.emit(mathcan,out)
    from subprocess import run
    for original in train_data.TRAIN_SOURCE:
        stem=Path(original).stem
        probe_file=out/(stem+".lean")
        p=run(["lean",str(probe_file)],capture_output=True,text=True,timeout=120)
        (out/(stem+".log")).write_text(p.stdout+p.stderr)
        if p.returncode!=0 or "sorryAx" in p.stdout:
            raise AssertionError(f"Lean rejected original training theorem {stem}")
    extraction=train_data.extract(out)
    file=out/"actual_goal_states.jsonl"
    if digest(file)!=TRAIN_STATES_SHA or extraction["real_compiler_states"]!=31:
        raise AssertionError("frozen training state SHA/count differs")
    return _rows(file),report,extraction


def _make_train_probe_matrix(states:list[dict],mathcan:Path,out:Path):
    """Eight REAL Lean tactic probes for each of the 31 real train states."""
    original_files={}
    scored=[]
    coverage=Counter()
    for i,row in enumerate(states):
        if row["goal_count"]!=1:
            raise AssertionError("multi-goal training record needs separate policy")
        path=row["source_file"]
        if path not in original_files:
            original_files[path]=(mathcan/path).read_text(encoding="utf8")
        for action in VOCAB:
            result=scratch.action_effect(
                original_files[path],row["theorem"],
                action,row["stage_index"],row["actual_lean_goal_state"])
            coverage[action]+=1
            scored.append({
              "stage_id":row["stage_id"],"action":action,
              "accepted":result["accepted"],"progress":result["progress"],
              "closed":result["no_goal_left_in_probe"],
              "actual_source_sha256":result["source_sha256"],
              "compiler_wall_ms":result["lean_wall_ms"],
            })
        print("FC11_ACTION_EXPOSURE",json.dumps({
          "state":row["stage_id"],"attempted":len(VOCAB),
          "accepted":sum(s["accepted"] for s in scored[-len(VOCAB):]),
          "progress":sum(s["progress"] for s in scored[-len(VOCAB):]),
          "closed":sum(s["closed"] for s in scored[-len(VOCAB):])
        },sort_keys=True),flush=True)
    if len(scored)!=31*len(VOCAB) or any(coverage[a]!=31 for a in VOCAB):
        raise AssertionError("one of eight candidate actions absent from training")
    file=out/"training_action_probes.jsonl"
    file.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in scored))
    return scored,digest(file)


def _labels(states,probes):
    by_id={}
    for r in probes:
        by_id.setdefault(r["stage_id"],{})[r["action"]]=r
    values=[]
    trainable=0
    for row in states:
        per=by_id[row["stage_id"]]
        y=[]
        for action in VOCAB:
            a=per[action]
            weight=1. if a["closed"] and a["accepted"] else (
                0.5 if a["progress"] and a["accepted"] else 0.)
            y.append(weight)
        if max(y)>0:
            trainable+=1
        values.append(y)
    return torch.tensor(values,dtype=torch.float32),trainable


def train_policy(states,targets,seed:int):
    torch.manual_seed(seed+11011)
    torch.use_deterministic_algorithms(True)
    model=ImportedLoop(loops=4)
    inputs=graph_batch([x["actual_lean_goal_state"] for x in states])
    gen=torch.Generator().manual_seed(6154+seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    log=[]
    start=time.perf_counter()
    for step in range(TRAIN_STEPS):
        batch_ids=torch.randint(len(states),(BATCH,),generator=gen)
        b={key:value[batch_ids] for key,value in inputs.items()}
        target=targets[batch_ids]
        logits=model(b)
        weights=1.+4.*target
        loss=(F.binary_cross_entropy_with_logits(logits,target,
              reduction="none")*weights).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step in (0,TRAIN_STEPS//2,TRAIN_STEPS-1):
            log.append([step+1,round(float(loss.detach()),6)])
    model.eval()
    return model,{
      "seed":seed,"model_parameters":sum(p.numel() for p in model.parameters()),
      "training_steps":TRAIN_STEPS,"batch":BATCH,"lr":.002,
      "train_wall_seconds":round(time.perf_counter()-start,4),
      "loss_samples":log,
    }


def training(source:Path,mathcan:Path,out:Path):
    torch.set_num_threads(2)
    core.source_lean_version(mathcan)
    states,source_manifest,extraction=_read_train_original_and_states(
        source,mathcan,out/"source")
    scored,probe_hash=_make_train_probe_matrix(states,mathcan,out)
    target,trained=_labels(states,scored)
    manifest={
       "experiment":"FC11_real_imported_lean_multistep",
       "phase":"training_from_source_only_before_heldout",
       "prereg_sha":PREREG,"train_source_rev":TRAIN_REV,
       "source_mathcan_sha":SOURCE_MATHCAN,
       "real_train_goal_sha256":TRAIN_STATES_SHA,
       "probe_matrix_sha256":probe_hash,
       "each_candidate_action_attempted_in_real_lean":True,
       "train_goal_states":len(states),
       "train_states_with_progress_or_closed_action":trained,
       "action_list":list(VOCAB),
       "test_selection_available_in_source_checkout":False,
       "weights":[],
    }
    for seed in SEEDS:
        model,meta=train_policy(states,target,seed)
        name=f"fc11_s{seed}"
        dest=out/(name+".pt")
        torch.save(model.state_dict(),dest)
        meta["name"]=name
        meta["checkpoint_sha256"]=digest(dest)
        manifest["weights"].append(meta)
        print("FC11_FROZEN_MODEL",json.dumps(meta,sort_keys=True),flush=True)
    (out/"train_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("FC11_TRAIN_FINAL",json.dumps({
        "train_states":len(states),
        "train_eligible":trained,
        "probed_actions":len(scored),
        "model_param_count":manifest["weights"][0]["model_parameters"],
    },sort_keys=True),flush=True)


def _prefix_apply(original:str,theorem:str,actions:list[str]):
    return scratch.compile_candidate(original,theorem,actions,probe=False,
                                     timeout_seconds=30)


def infer_closed_loop(model:ImportedLoop,original:str,theorem:str,
                      initial_goal:str)->dict:
    """Choose actions from exact ACTUAL Lean prefix goals; proof gate is real."""
    history=[]
    goal=initial_goal
    start=time.perf_counter()
    for _ in range(MAX_SEARCH_ACTIONS):
        if goal.count("⊢")!=1:
            return {"accepted":False,"actions":history,
                    "reason":"multigoal_unsupported",
                    "elapsed_ms":round((time.perf_counter()-start)*1000,5)}
        with torch.no_grad():
            scores=model(graph_batch([goal]))[0]
            ranked=scores.argsort(descending=True).tolist()
        chosen=VOCAB[ranked[0]]
        history.append(chosen)
        verdict=_prefix_apply(original,theorem,history)
        if verdict["accepted"]:
            return {"accepted":True,"actions":history,
                    "reason":"lean_kernel_accepted",
                    "final_compiler":verdict,
                    "elapsed_ms":round((time.perf_counter()-start)*1000,5)}
        if verdict["timed_out"]:
            break
        # Scratch prefix can use sorry, solely for goal observation.
        scratch_result=scratch.compile_candidate(original,theorem,
             history,probe=True,timeout_seconds=30)
        if not scratch_result["accepted"]:
            return {"accepted":False,"actions":history,
                    "reason":"candidate_tactic_failed",
                    "final_compiler":verdict,
                    "scratch_error":scratch_result["compiler_error_excerpt"],
                    "elapsed_ms":round((time.perf_counter()-start)*1000,5)}
        next_states=scratch_result["real_remaining_goal_states"]
        next_goal="\n".join(next_states)
        if not next_goal or next_goal==goal:
            return {"accepted":False,"actions":history,
                    "reason":"candidate_made_no_progress",
                    "final_compiler":verdict,
                    "elapsed_ms":round((time.perf_counter()-start)*1000,5)}
        goal=next_goal
    return {"accepted":False,"actions":history,
            "reason":"three_tactic_budget_exhausted",
            "elapsed_ms":round((time.perf_counter()-start)*1000,5)}


def native_first_script(original:str,theorem:str)->str:
    """Single Lean-native proof search, without a reference-tactic hint."""
    raw=original.splitlines()
    _,begin,end=core.theorem_bounds(raw,theorem)
    header=raw[begin].split(":= by",1)[0]+":= by"
    lines=[header]
    for step in range(MAX_SEARCH_ACTIONS):
        lines.append(("  first" if step==0 else "  all_goals first"))
        lines.extend("  | "+action for action in VOCAB)
    closers=[s for s in raw[end:] if re.fullmatch(r"end\s+[A-Za-z_0-9.]+",s.strip())]
    code=raw[:begin]+lines+([closers[-1]] if closers else [])
    return "import Lean\n"+"\n".join(code)+"\n"


def native_first_check(original:str,theorem:str,allowed_ms:float):
    source=native_first_script(original,theorem)
    with tempfile.TemporaryDirectory(prefix="fc11_native_") as directory:
        file=Path(directory)/"NativeFirst.lean"
        file.write_text(source,encoding="utf8")
        start=time.perf_counter()
        try:
            proc=subprocess.run(["lean",str(file)],
                capture_output=True,text=True,
                timeout=max(0.05,allowed_ms/1000))
            rc=proc.returncode
            out=proc.stdout+proc.stderr
            timed_out=False
        except subprocess.TimeoutExpired as e:
            rc=124
            out=""
            timed_out=True
        elapsed=(time.perf_counter()-start)*1000
    return {"accepted":rc==0 and not timed_out and
               "error:" not in out and "sorryAx" not in out,
            "timed_out":timed_out,"allowed_ms":round(allowed_ms,5),
            "elapsed_ms":round(elapsed,5),
            "overrun_ms":round(max(0.,elapsed-allowed_ms),5),
            "compiler_error":out[-500:] if rc else "",
            "source_sha256":hashlib.sha256(source.encode()).hexdigest()}


def _read_test(source:Path,mathcan:Path,out:Path):
    from research.refutation_bench.fc11 import test_data
    out.mkdir(parents=True,exist_ok=True)
    report=test_data.emit(mathcan,out)
    for original in test_data.TEST_SOURCE:
        stem=Path(original).stem
        p=subprocess.run(["lean",str(out/(stem+".lean"))],
             capture_output=True,text=True,timeout=120)
        (out/(stem+".log")).write_text(p.stdout+p.stderr)
        if p.returncode!=0 or "sorryAx" in p.stdout:
            raise AssertionError(f"original held-out proof failed under Lean: {stem}")
    extracted=test_data.extract(out)
    file=out/"actual_goal_states.jsonl"
    if digest(file)!=TEST_STATES_SHA:
        raise AssertionError("held-out state SHA differs from source-only CI")
    return _rows(file),report,extracted


def evaluate(source:Path,mathcan:Path,weights:Path,out:Path):
    torch.set_num_threads(2)
    manifest=json.loads((weights/"train_manifest.json").read_text())
    if (manifest["prereg_sha"]!=PREREG or
        manifest["real_train_goal_sha256"]!=TRAIN_STATES_SHA or
        manifest["test_selection_available_in_source_checkout"] is not False or
        manifest["action_list"]!=list(VOCAB)):
        raise AssertionError("training provenance, tactics or exclusion changed")
    rows,source_manifest,extraction=_read_test(source,mathcan,out/"test_source")
    from research.refutation_bench.fc11 import test_data
    expected=[(path,theorem) for path,names in test_data.TEST_SOURCE.items()
              for theorem in names]
    first={}
    for row in rows:
        key=(row["source_file"],row["theorem"])
        if row["stage_index"]==0:
            first[key]=row["actual_lean_goal_state"]
    if len(first)!=len(expected):
        raise AssertionError("heldout initial goal state was not printed")
    original_files={path:(mathcan/path).read_text(encoding="utf8")
                    for path in test_data.TEST_SOURCE}
    predictions={}
    model_reports=[]
    for meta in manifest["weights"]:
        file=weights/(meta["name"]+".pt")
        if digest(file)!=meta["checkpoint_sha256"]:
            raise AssertionError("checkpoint hash changed after training")
        model=ImportedLoop(loops=4)
        model.load_state_dict(torch.load(file,weights_only=True,map_location="cpu"))
        model.eval()
        res=[]
        for path,theorem in expected:
            ans=infer_closed_loop(model,original_files[path],theorem,
                                  first[(path,theorem)])
            ans.update(source_file=path,theorem=theorem,model=meta["name"])
            res.append(ans)
            print("FC11_KERNEL_ACTIONS",json.dumps({
                "model":meta["name"],"theorem":theorem,
                "accepted":ans["accepted"],"actions":ans["actions"],
                "reason":ans["reason"]},sort_keys=True),flush=True)
        predictions[meta["name"]]=res
        dest=out/(meta["name"]+"_predictions.jsonl")
        dest.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in res))
        model_reports.append({
           "model":meta["name"],"seed":meta["seed"],
           "source_theorems":len(res),
           "kernel_accepted":sum(x["accepted"] for x in res),
           "total_elapsed_ms":round(sum(x["elapsed_ms"] for x in res),5),
           "prediction_sha256":digest(dest),
           "failure_reasons":dict(Counter(x["reason"] for x in res)),
           "accepted_by_module":{path:sum(x["accepted"] for x in res
                         if x["source_file"]==path) for path in test_data.TEST_SOURCE},
        })
    # A genuine Lean native 'first' search is given the measured model
    # wall-time allowance on each original source theorem.
    native=[]
    for meta in manifest["weights"]:
        tag=meta["name"]
        for item in predictions[tag]:
            path=item["source_file"]
            t=item["theorem"]
            out_native=native_first_check(original_files[path],t,item["elapsed_ms"])
            out_native.update(model=tag,theorem=t,source_file=path)
            native.append(out_native)
    native_file=out/"lean_native_cpu_matched.jsonl"
    native_file.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in native))
    for report in model_reports:
        selected=[r for r in native if r["model"]==report["model"]]
        report["native_kernel_accepted"]=sum(x["accepted"] for x in selected)
        report["native_total_elapsed_ms"]=round(sum(x["elapsed_ms"] for x in selected),5)
        report["native_timeouts"]=sum(x["timed_out"] for x in selected)
        report["native_overrun_ms"]=round(sum(x["overrun_ms"] for x in selected),5)
        report["model_beats_native"]=report["kernel_accepted"]>report["native_kernel_accepted"]
        m_cost=report["total_elapsed_ms"]/max(report["kernel_accepted"],1)
        n_cost=report["native_total_elapsed_ms"]/max(report["native_kernel_accepted"],1)
        report["model_cheaper_per_proof"]=m_cost<n_cost
    response={
        "experiment":"FC11_real_imported_lean_multistep",
        "first_frozen_heldout_test":True,
        "prereg_sha":PREREG,
        "training_revision":TRAIN_REV,
        "source_mathcan_sha":SOURCE_MATHCAN,
        "test_source_revision":TEST_REV,
        "real_test_goal_sha256":TEST_STATES_SHA,
        "n_original_test_theorems":len(expected),
        "original_reference_theorems_compiled":True,
        "all_positive_claims_require_separate_sorry_free_lean_compilation":True,
        "scratch_probes_are_untrusted":True,
        "all_eight_tactics_attempted_on_all_training_states":manifest[
             "each_candidate_action_attempted_in_real_lean"],
        "training_action_probe_sha256":manifest["probe_matrix_sha256"],
        "model_results":model_reports,
        "registered_success":any(x["model_beats_native"] for x in model_reports),
        "registered_compute_success":any(x["model_cheaper_per_proof"] for x in model_reports),
        "native_predictions_sha256":digest(native_file),
        "scope":"actual imported research theorem declarations; exact pinned Lean4.35; small test",
    }
    (out/"FC11_report.json").write_text(json.dumps(response,indent=2,sort_keys=True)+"\n")
    print("FC11_CRITERIA",json.dumps({
        "neural_beats_native":response["registered_success"],
        "neural_compute_advantage":response["registered_compute_success"],
        "models":model_reports,
    },sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("train","eval"))
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--mathcan",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--weights",type=Path)
    a=p.parse_args()
    if a.phase=="train":
        if a.weights:
            raise ValueError("training cannot read held-out artifacts")
        training(a.source,a.mathcan,a.out)
    else:
        if not a.weights:
            raise ValueError("evaluation requires frozen weights")
        evaluate(a.source,a.mathcan,a.weights,a.out)


if __name__=="__main__":
    main()
