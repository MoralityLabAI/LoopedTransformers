"""FC-12 original MathCan Lean4.35 typed parameterized tactic selection.

The recurrent actor ranks twelve registered tactic FAMILIES using real Lean
compiler-emitted goal states; every concrete parameter is a local name from
that goal state. Both native and neural controllers apply the same scratch
and separate sorry-free proof gate. The test selector is mounted ONLY after
the training-action result and model checkpoints have been frozen.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import tempfile

import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"lean_reasoner"))
from train import TypedLoop

SOURCE_MATHCAN="c84e2025334e9a967c1297425a490f98d5b3d189"
REGISTERED_PREREG="d6fae0ddb9ce60036cff09e5ec93820ed3879af8"
EARLY_TRAIN_SOURCE_SHA="8d192f0b323eade0fc764611f420414d70a0d81b"
EARLY_TRAIN_GOAL_SHA="403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0"
EARLY_TRAIN_PROBE_SHA="bd713c7fd3d9d2e82f1304d7556f7a25bfa8b1a0e416dbd2e8d271dc4ad047ec"
TEST_SOURCE_SHA="__FC12_TEST_SOURCE_SHA__"
TEST_GOAL_SHA="__FC12_TEST_GOAL_SHA__"

SEEDS=(17,29)
NUM_ACTIONS=12
EMBED=24
TOKEN_BUCKETS=2048
MAX_TOKENS=112
STEPS=300
BATCH=16
LOOPS=4


def hash_file(file:Path):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def tokenize(state:str)->list[str]:
    if "⊢" not in state:
        raise AssertionError("real Lean turnstile required for inference")
    words=re.findall(r"[\w']+|[^\w\s]",state,flags=re.UNICODE)
    if len(words)>MAX_TOKENS:
        i=words.index("⊢")
        begin=max(0,min(len(words)-MAX_TOKENS,i-MAX_TOKENS//2))
        words=words[begin:begin+MAX_TOKENS]
    if "⊢" not in words:
        raise AssertionError("lost Lean goal turnstile")
    return words


def graph(states:list[str])->dict[str,torch.Tensor]:
    n=len(states)
    ids=torch.zeros((n,MAX_TOKENS),dtype=torch.long)
    present=torch.zeros((n,MAX_TOKENS),dtype=torch.float32)
    head=torch.zeros((n,NUM_ACTIONS,MAX_TOKENS),dtype=torch.float32)
    for i,state in enumerate(states):
        words=tokenize(state)
        for j,w in enumerate(words):
            ids[i,j]=int.from_bytes(hashlib.blake2b(w.encode(),digest_size=8).digest(),"big")%TOKEN_BUCKETS
            present[i,j]=1.
        head[i,:,words.index("⊢")]=1.
    prem=present[:,None,:].expand(-1,NUM_ACTIONS,-1).clone()
    return {"tokens":ids,"present":present,"head":head,"prem":prem}


class FC12Loop(nn.Module):
    """Exact original 62,401-parameter LR-01 TypedLoop tactic scorer."""
    def __init__(self):
        super().__init__()
        self.atom=nn.Embedding(TOKEN_BUCKETS,EMBED)
        self.rule=nn.Embedding(NUM_ACTIONS,EMBED)
        self.cell=TypedLoop(EMBED)
        self.head=nn.Sequential(nn.Linear(3*EMBED,EMBED),nn.GELU(),nn.Linear(EMBED,1))

    def forward(self,inp:dict)->torch.Tensor:
        atom=self.atom(inp["tokens"])*inp["present"][...,None]
        index=torch.arange(NUM_ACTIONS,device=atom.device)[None,:].expand(atom.shape[0],-1)
        rule=self.rule(index)
        for _ in range(LOOPS):
            atom,rule=self.cell(atom,rule,inp["prem"],inp["head"])
        parents=inp["prem"]@atom/inp["prem"].sum(-1,keepdim=True).clamp_min(1.)
        target=inp["head"]@atom
        return self.head(torch.cat((rule,parents,target),-1)).squeeze(-1)


def read_jsonl(path:Path)->list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def build_train_matrix(states,probes,families):
    by_stage={}
    for probe in probes:
        by_stage.setdefault(probe["stage_id"],{}).setdefault(probe["family"],[]).append(probe)
    y=torch.zeros((len(states),len(families)))
    active=torch.zeros_like(y)
    for i,state in enumerate(states):
        for j,family in enumerate(families):
            observed=by_stage.get(state["stage_id"],{}).get(family,[])
            if observed:
                active[i,j]=1.
                y[i,j]=max(float(x["progress"]) for x in observed)
    return y,active


def fit_one(states,probes,families,seed):
    torch.manual_seed(seed+12000)
    torch.use_deterministic_algorithms(True)
    model=FC12Loop()
    if sum(p.numel() for p in model.parameters())!=62401:
        raise AssertionError("original TypedLoop 62401 parameter contract broken")
    inputs=graph([x["actual_lean_goal_state"] for x in states])
    labels,active=build_train_matrix(states,probes,families)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    generator=torch.Generator().manual_seed(6154+seed)
    t0=time.perf_counter()
    samples=[]
    for update in range(STEPS):
        ids=torch.randint(len(states),(BATCH,),generator=generator)
        values=model({k:v[ids] for k,v in inputs.items()})
        target=labels[ids]
        relevant=active[ids]
        importance=1.+3.*target
        loss=(F.binary_cross_entropy_with_logits(values,target,reduction="none")*
              relevant*importance).sum()/relevant.sum().clamp_min(1.)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if update in (0,STEPS//2,STEPS-1):
            samples.append([update+1,round(float(loss.detach()),6)])
    model.eval()
    return model,{
        "seed":seed,"train_steps":STEPS,"batch":BATCH,
        "training_loss_samples":samples,"training_wall_s":round(time.perf_counter()-t0,4),
        "train_parameters":62401,
        "train_states":len(states),
        "state_action_compiler_labels":int(active.sum().item()),
        "positive_state_action_families":int((labels>0).sum().item()),
    }


def train(source:Path,mathcan:Path,out:Path):
    """Recreate original training source and exact action probes before test."""
    from research.refutation_bench.fc12 import core,train_data,probe_train
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    source_root=out/"train_source"
    train_data.emit(mathcan,source_root)
    for path in train_data.TRAIN_MODULES:
        stem=Path(path).stem
        r=subprocess.run(["lean",str(source_root/(stem+".lean"))],
             capture_output=True,text=True,timeout=120)
        (source_root/(stem+".log")).write_text(r.stdout+r.stderr)
        if r.returncode!=0 or "sorryAx" in r.stdout:
            raise AssertionError("original MathCan training source proof failed")
    extraction=train_data.extract(source_root)
    actual=source_root/"actual_goal_states.jsonl"
    if hash_file(actual)!=EARLY_TRAIN_GOAL_SHA:
        raise AssertionError("source train goal states changed")
    probe_root=out/"train_probes"
    probes=probe_train.run(mathcan,actual,probe_root,workers=4)
    if probes["semantic_outcomes_sha256"]!=EARLY_TRAIN_PROBE_SHA:
        # Tactic success, progress, candidate script and source digests
        # are deterministic; compiler wall times and temp paths are not.
        raise AssertionError("source action effect semantics changed")
    states=core.states_from_compiler(source_root)
    action_rows=read_jsonl(probe_root/"train_candidate_outcomes.jsonl")
    training_manifest={
      "experiment":"FC12_parameterized_real_lean_tactics",
      "preregistration_sha":REGISTERED_PREREG,
      "original_mathcan_source_sha":SOURCE_MATHCAN,
      "train_source_sha":EARLY_TRAIN_SOURCE_SHA,
      "train_actual_goal_sha":EARLY_TRAIN_GOAL_SHA,
      "train_probe_data_sha":EARLY_TRAIN_PROBE_SHA,
      "tactic_families":list(core.FAMILIES),
      "train_heldout_selector_not_available":True,
      "real_training_states":len(states),
      "all_required_tactic_positives":True,
      "models":[],
    }
    for seed in SEEDS:
        student,meta=fit_one(states,action_rows,core.FAMILIES,seed)
        file=out/(f"fc12_seed{seed}.pt")
        torch.save(student.state_dict(),file)
        meta.update(checkpoint_sha256=hash_file(file),filename=file.name)
        training_manifest["models"].append(meta)
        print("FC12_FROZEN",json.dumps(meta,sort_keys=True),flush=True)
    (out/"train_manifest.json").write_text(json.dumps(training_manifest,indent=2,sort_keys=True)+"\n")


def policy_candidates(state:str,family_order:list[str]):
    from research.refutation_bench.fc12 import core
    pool=core.generate(state)
    for family in family_order:
        for script in pool[family]:
            yield family,script


def run_closed_loop(original:str,theorem:str,initial_goal:str,
                    model:FC12Loop|None=None,*,allowed_ms:float|None=None)->dict:
    """Neural or source-only tactic-family ranking with identical Lean proof gate.

    Time-matched symbolic search receives ONLY the model's measured total
    runtime scalar. It does not receive the model action list, logits or
    per-theorem reference script.
    """
    from research.refutation_bench.fc12 import core,scratch
    start=time.perf_counter()
    attempts=[]
    chosen=[]
    state=initial_goal
    outcome="three_step_budget"
    final_accepted=False
    for step in range(core.MAX_TRANSITIONS):
        if state.count("⊢")!=1:
            outcome="multigoal_unsupported"
            break
        if allowed_ms is not None and (time.perf_counter()-start)*1000>=allowed_ms:
            outcome="symbolic_deadline"
            break
        if model is None:
            ranked=list(core.FAMILIES)
        else:
            with torch.no_grad():
                scores=model(graph([state]))[0]
                ranked=[core.FAMILIES[i] for i in scores.argsort(descending=True).tolist()]
        did_advance=False
        for family,concrete in policy_candidates(state,ranked):
            if len(attempts)>=core.MAX_TOTAL_LEAN_PROBES:
                outcome="compiler_probe_budget"
                break
            if allowed_ms is not None and (time.perf_counter()-start)*1000>=allowed_ms:
                outcome="symbolic_deadline"
                break
            prefix=chosen+[concrete]
            actual=scratch.compile_candidate(original,theorem,prefix,probe=False,timeout_seconds=18)
            row={
               "step":step,"family":family,"script":concrete,
               "exact_compiler_accepted":actual["accepted"],
               "exact_compiler_ms":actual["lean_wall_ms"],
               "exact_source_sha256":actual["source_sha256"],
               "timed_out":actual["timed_out"],
               "original_error_excerpt":actual["compiler_error_excerpt"][:180],
            }
            attempts.append(row)
            if actual["accepted"]:
                chosen=prefix
                outcome="kernel_accepted"
                final_accepted=True
                did_advance=True
                break
            preview=scratch.compile_candidate(original,theorem,prefix,probe=True,timeout_seconds=18)
            row["scratch_compiler_accepted"]=preview["accepted"]
            row["scratch_compiler_ms"]=preview["lean_wall_ms"]
            row["scratch_preview_only"]=True
            if preview["accepted"]:
                nxt="\n".join(preview["real_remaining_goal_states"])
                if nxt.strip() and nxt.strip()!=state.strip():
                    state=nxt
                    chosen=prefix
                    did_advance=True
                    outcome="advanced"
                    break
            if actual["timed_out"]:
                outcome="compiler_timeout"
        if final_accepted or outcome in ("symbolic_deadline","compiler_probe_budget","compiler_timeout"):
            break
        if not did_advance:
            outcome="no_legal_progress"
            break
    elapsed_ms=(time.perf_counter()-start)*1000
    return {
       "accepted":final_accepted,
       "chosen_actions":chosen,
       "reason":outcome,
       "candidate_probe_count":len(attempts),
       "full_compiler_attempts":len(attempts),
       "sorry_free_release":final_accepted,
       "tactic_attempts":attempts,
       "elapsed_ms":round(elapsed_ms,6),
       "allowed_ms":allowed_ms,
       "overrun_ms":round(max(0.,elapsed_ms-allowed_ms),6) if allowed_ms is not None else 0.,
    }


def evaluate(source:Path,mathcan:Path,weights:Path,out:Path):
    from research.refutation_bench.fc12 import core,test_data
    torch.set_num_threads(2)
    manifest=json.loads((weights/"train_manifest.json").read_text())
    if manifest["preregistration_sha"]!=REGISTERED_PREREG or (
        manifest["train_source_sha"]!=EARLY_TRAIN_SOURCE_SHA or
        manifest["train_actual_goal_sha"]!=EARLY_TRAIN_GOAL_SHA or
        manifest["train_heldout_selector_not_available"] is not True):
        raise AssertionError("train/test source isolation or prereg changed")
    out.mkdir(parents=True,exist_ok=True)
    source_out=out/"heldout_source"
    test_data.emit(mathcan,source_out)
    for path in test_data.TEST_MODULES:
        stem=Path(path).stem
        r=subprocess.run(["lean",str(source_out/(stem+".lean"))],
            capture_output=True,text=True,timeout=120)
        (source_out/(stem+".log")).write_text(r.stdout+r.stderr)
        if r.returncode!=0 or "sorryAx" in r.stdout:
            raise AssertionError("original heldout reference proof failed under Lean")
    test_data.extract(source_out)
    if hash_file(source_out/"actual_goal_states.jsonl")!=TEST_GOAL_SHA:
        raise AssertionError("frozen heldout actual Lean state SHA changed")
    rows=core.states_from_compiler(source_out)
    initial={}
    for row in rows:
        key=(row["source_file"],row["theorem"])
        if row["stage_index"]==0:
            initial[key]=row["actual_lean_goal_state"]
    tests=[(p,name) for p,names in test_data.TEST_MODULES.items() for name in names]
    if len(tests)!=16 or len(initial)!=16:
        raise AssertionError("registered heldout theorem set or initial-state count changed")
    originals={p:(mathcan/p).read_text(encoding="utf8") for p in test_data.TEST_MODULES}
    all_models=[]
    for meta in manifest["models"]:
        path=weights/meta["filename"]
        if hash_file(path)!=meta["checkpoint_sha256"]:
            raise AssertionError("frozen checkpoint hash changed")
        actor=FC12Loop()
        actor.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"))
        actor.eval()
        model_rows=[]
        native_rows=[]
        for file,name in tests:
            goal=initial[file,name]
            m=run_closed_loop(originals[file],name,goal,actor)
            s=run_closed_loop(originals[file],name,goal,None,allowed_ms=m["elapsed_ms"])
            m.update(theorem=name,source_file=file,model=meta["filename"])
            s.update(theorem=name,source_file=file,baseline="native_context_ordered",model=meta["filename"])
            model_rows.append(m)
            native_rows.append(s)
            print("FC12_PAIRED",json.dumps({
               "seed":meta["seed"],"theorem":name,
               "student_certified":m["accepted"],"native_certified":s["accepted"],
               "student_ms":m["elapsed_ms"],"native_ms":s["elapsed_ms"],
               "student_actions":m["chosen_actions"],
               "native_actions":s["chosen_actions"]},sort_keys=True),flush=True)
        outname=f"fc12_seed{meta['seed']}"
        f1=out/(outname+"_model.jsonl")
        f2=out/(outname+"_native.jsonl")
        core.canonical_rows(f1,model_rows)
        core.canonical_rows(f2,native_rows)
        positives=sum(x["accepted"] for x in model_rows)
        native=sum(x["accepted"] for x in native_rows)
        mm=sum(x["elapsed_ms"] for x in model_rows)
        sm=sum(x["elapsed_ms"] for x in native_rows)
        all_models.append({
            "seed":meta["seed"],"model_proofs":positives,"native_proofs":native,
            "model_only":sum(a["accepted"] and not b["accepted"] for a,b in zip(model_rows,native_rows)),
            "native_only":sum(b["accepted"] and not a["accepted"] for a,b in zip(model_rows,native_rows)),
            "model_wall_ms":round(mm,6),"native_wall_ms":round(sm,6),
            "model_ms_per_proof":round(mm/positives,5) if positives else None,
            "native_ms_per_proof":round(sm/native,5) if native else None,
            "model_output_sha256":hash_file(f1),"native_output_sha256":hash_file(f2),
            "total_model_tactic_trials":sum(x["candidate_probe_count"] for x in model_rows),
            "total_native_tactic_trials":sum(x["candidate_probe_count"] for x in native_rows),
            "native_overrun_ms_total":round(sum(x["overrun_ms"] for x in native_rows),6),
            "student_beats_symbolic":positives>native,
            "student_cheaper_per_proof":bool(positives and native and mm/positives<sm/native),
        })
    report={
      "experiment":"FC12_typed_parameterized_original_Lean_theorem_search",
      "preregistration_sha":REGISTERED_PREREG,
      "source_upstream_sha":SOURCE_MATHCAN,
      "train_source_sha":EARLY_TRAIN_SOURCE_SHA,
      "train_actual_state_sha256":EARLY_TRAIN_GOAL_SHA,
      "train_probe_effect_sha256":EARLY_TRAIN_PROBE_SHA,
      "test_source_sha":TEST_SOURCE_SHA,
      "test_actual_state_sha256":TEST_GOAL_SHA,
      "heldout_theorem_count":16,
      "original_source_oracle_proof_count":16,
      "max_model_selected_goal_transitions":3,
      "max_tactic_probes_per_theorem":12,
      "all_accepted_model_proofs_separately_compiled_without_sorry":True,
      "comparison_basis":"post-training heldout source-family, CPU time allowance",
      "results":all_models,
      "registered_learning_criterion":all(x["student_beats_symbolic"] for x in all_models),
      "registered_compute_criterion":all(x["student_cheaper_per_proof"] for x in all_models),
    }
    (out/"FC12_report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("FC12_CRITERIA",json.dumps(report,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("train","eval"))
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--mathcan",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--weights",type=Path)
    args=p.parse_args()
    if args.phase=="train":
        if args.weights is not None:
            raise ValueError("train cannot use heldout artifacts")
        train(args.source,args.mathcan,args.out)
    else:
        if args.weights is None:
            raise ValueError("heldout must use immutable model train artifact")
        evaluate(args.source,args.mathcan,args.weights,args.out)


if __name__=="__main__":
    main()
