"""RAM-04: first prospective original-MathCan family-heldout theorem evaluation.

Uses the unchanged RAM-03 candidate grammar, Lean persistent search and
sub-250k parameter networks. Only the twelve names frozen in RAM-04's
preregistration can enter evaluation, after checkpoint selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import torch

RAM03 = Path(__file__).resolve().parents[1] / "lean_ram03"
sys.path.insert(0, str(RAM03))
import ram03_core as c
from runtime import ResourceGuard, guarded_process
from search import search
from train_local import load_checkpoint, load_data
from evaluate import Frequency

EXACT_CASES = {
    "MathCan/OpcodeRefinement.lean": (
        "two_step_gross_budget", "two_step_output_aggregation",
        "boundary_liquidity_transition",
    ),
    "MathCan/FoundryProbe.lean": (
        "two_nonzero_after_swap", "two_currency_zero_at_lock",
        "erc20_custody_neutral",
    ),
    "MathCan/TOETwoFormTriad.lean": (
        "orthogonal_triad_isotropic", "two_axes_missing_third",
        "two_fixed_axes_not_positive_isotropic",
    ),
    "MathCan/TOETripletJordan.lean": (
        "exact_source_factor", "exact_lightcone_null",
        "exact_lightcone_det_zero",
    ),
}
REGISTERED_SHA = "ae7be99b6367434f0dd724ae232cfa218ed99342"
MATHCAN_SHA = "c84e2025334e9a967c1297425a490f98d5b3d189"
FC12_TRAIN_SHA = "403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0"
FC12_EFFECT_SHA = "bd713c7fd3d9d2e82f1304d7556f7a25bfa8b1a0e416dbd2e8d271dc4ad047ec"
SOURCE_TRAIN = frozenset({
    "MathCan/FiniteHorizonGap.lean",
    "MathCan/DeletionMemory.lean",
    "MathCan/TOEGraph.lean",
    "MathCan/TOECosmology.lean",
    "MathCan/Intervention.lean",
    "MathCan/HookState.lean",
    "MathCan/GraphAuthorization.lean",
})
OLD_TEST = frozenset({
    "MathCan/Hodge.lean", "MathCan/TOEDressedMatter.lean",
    "MathCan/TOELorentzBivector.lean", "MathCan/TOETensorCompletion.lean",
    "MathCan/HookMultiTick.lean", "MathCan/Settlement.lean",
    "MathCan/TOEJordan.lean", "MathCan/TOENonAbelian.lean",
    "MathCan/FlashAccounting.lean", "MathCan/V4Transient.lean",
    "MathCan/TOEFormSectors.lean", "MathCan/TOEMultipletKinetics.lean",
})

def source_contract(config:dict, mathcan:Path|None=None)->None:
    assert config["registration"] == REGISTERED_SHA
    actual = {k:tuple(v) for k,v in config["prospective"].items()}
    assert actual == EXACT_CASES and sum(map(len,actual.values())) == 12
    assert not (set(actual) & SOURCE_TRAIN)
    assert not (set(actual) & OLD_TEST)
    assert config["primary"] == {
        "seconds":30,"evaluations":48,"transitions":8,
        "request_seconds":3,"max_goals":8,
    }
    assert config["secondary"] == {
        "seconds":60,"evaluations":128,"transitions":16,
        "request_seconds":3,"max_goals":8,
    }
    assert [(x["name"],x["width"],x["loops"]) for x in config["models"]] == [
        ("loop32",32,4),("loop64",64,4),("flat64",64,0),
    ]
    assert config["seeds"] == [17,29,43]
    if mathcan:
        assert (mathcan/"lean-toolchain").read_text().strip()=="leanprover/lean4:v4.35.0"
        for filename,names in EXACT_CASES.items():
            text = (mathcan/filename).read_text(encoding="utf8")
            declarations = re.findall(r"^theorem\s+([A-Za-z_0-9']+)",text,re.M)
            assert tuple(declarations[:3])==names, ("preregistered theorem changed",filename)

def load_models(training_root:Path,config:dict):
    receipt = json.loads((training_root/"training_report.json").read_text())
    assert receipt["status"]=="complete"
    assert receipt["config_sha256"] == c.digest(config["_path"])
    assert receipt["data"]["train"]["multi_input_states"]>0
    assert receipt["data"]["development"]["multi_input_states"]>0
    available={(r["name"],r["seed"]):r for r in receipt["models"]}
    assert set(available)=={(v["name"],s) for v in config["models"] for s in config["seeds"]}
    models={};checkpoints={}
    for name in ("loop32","loop64","flat64"):
        chosen=receipt["selected"][name]
        # The source training runner wrote absolute checkpoint paths. GitHub
        # artifacts have a different root in the evaluation job; remap by
        # registered model and seed, preserving byte-for-byte checkpoint hashes.
        original=available[name,chosen["seed"]]
        assert chosen["checkpoint_sha256"]==original["checkpoint_sha256"]
        file=training_root/f"{name}_s{chosen['seed']}"/"best.pt"
        assert file.is_file() and c.digest(file)==chosen["checkpoint_sha256"]
        model=load_checkpoint(file)
        assert all(torch.isfinite(t).all() for t in model.parameters())
        models[name]=model
        checkpoints[name]=dict(seed=chosen["seed"],sha256=c.digest(file),
                               parameters=sum(p.numel() for p in model.parameters()))
    return receipt,models,checkpoints

def summarize(rows:dict[str,list[dict]]):
    out=[]
    for key,records in rows.items():
        phase,policy=key.split("__")
        solved={r["source_file"]+":"+r["theorem"] for r in records if r["accepted"]}
        comparison=rows.get(phase+"__native",[])
        native={r["source_file"]+":"+r["theorem"] for r in comparison if r["accepted"]}
        cost=sum(r["elapsed_seconds"] for r in records)
        out.append(dict(
          experiment=phase,suite="prospective_ram04",model=policy,
          cases=len(records),certified=len(solved),solved=sorted(solved),
          model_only_vs_native=sorted(solved-native),
          native_only_vs_model=sorted(native-solved),
          total_seconds=cost,
          seconds_per_independent_certificate=cost/len(solved) if solved else None,
          logical_evaluations=sum(r["logical_evaluations"] for r in records),
          actual_lean_processes=sum(r["actual_lean_processes"] for r in records),
          multi_goal_transitions=sum(r["diagnostics"].get("multi_goal_transitions",0)
                                     for r in records),
          reasons=dict(Counter(r["reason"] for r in records)),
        ))
    return out

def run(mathcan:Path,data:Path,training:Path,out:Path,lean:str,config_file:Path):
    config=json.loads(config_file.read_text(encoding="utf8"))
    config["_path"]=config_file
    source_contract(config,mathcan)
    assert subprocess.check_output(["git","-C",str(mathcan),"rev-parse","HEAD"],
                                   text=True).strip()==MATHCAN_SHA
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    training_report,models,checkpoint=load_models(training,config)
    assert training_report["data"]["dataset_sha256"]==c.digest(data/"transitions.jsonl")
    assert training_report["data"]["train"]["rows"]>0
    train,dev,audit=load_data(data,config,c.digest(config_file))
    assert audit["train"]["theorems"]==22 and audit["development"]["theorems"]==5
    policy_frequency=Frequency(train)
    # Delay reading selected original theorem proof source until every
    # trained checkpoint and the frequency baseline have been verified.
    original={};source_hashes={}
    out.mkdir(parents=True,exist_ok=False)
    report=dict(
       experiment="RAM04_unseen_family_original_Lean435_transfer",
       preregistration_commit=REGISTERED_SHA,
       source_original_commit=MATHCAN_SHA,
       config_sha256=c.digest(config_file),
       dataset_sha256=training_report["data"]["dataset_sha256"],
       checkpoints=checkpoint,
       original_oracle_proofs=12, status="running",
       original_source_hashes=source_hashes,
       summaries=[],primary_model_only=False,recurrence_specific=False,
       policies={
         "primary":["native","frequency","loop32","loop64","flat64"],
         "secondary":["native","loop64"],
       },
    )
    all_rows=defaultdict(list)
    try:
        with ResourceGuard(out/"resources.jsonl",torch) as guard:
            for filename in EXACT_CASES:
                value=(mathcan/filename).read_text(encoding="utf8")
                original[filename]=value
                source_hashes[filename]=dict(
                     normalized=c.sha(value),raw=c.digest(mathcan/filename))
                check=guarded_process(
                    [lean,str((mathcan/filename).resolve())],
                    out/"oracle"/(Path(filename).stem+".log"),guard,120.)
                if check["returncode"]!=0 or check["timed_out"] or (
                    "sorryAx" in check["log"] or "error:" in check["log"]):
                    raise AssertionError("original Lean source oracle failed: "+filename)
            c.write_json(out/"RAM04_report.json",report)
            for phase in ("primary","secondary"):
                policies=({"native":None,"frequency":policy_frequency,**models}
                          if phase=="primary" else
                          {"native":None,"loop64":models["loop64"]})
                cases=[(f,name) for f,names in EXACT_CASES.items() for name in names]
                for index,(filename,theorem) in enumerate(cases):
                    # The deterministic permutation removes consistent
                    # model-first or native-first scheduling within the
                    # same theorem across repeated sequential policies.
                    names=list(policies)
                    offset=index%len(names);names=names[offset:]+names[:offset]
                    for policy in names:
                        row=search(lean,original[filename],theorem,policies[policy],
                          out/"evidence"/phase/f"{index:02d}"/policy,
                          guard,config[phase])
                        row.update(experiment=phase,suite="prospective_ram04",
                          model=policy,source_file=filename,theorem=theorem,
                          checkpoint=checkpoint.get(policy))
                        key=phase+"__"+policy
                        all_rows[key].append(row)
                        c.append_json(out/(key+".jsonl"),row)
                        print("RAM04_PROOF",json.dumps({
                            "phase":phase,"policy":policy,"theorem":theorem,
                            "certified":row["accepted"],"reason":row["reason"],
                            "wall_s":round(row["elapsed_seconds"],3),
                            "evaluations":row["logical_evaluations"]}),
                            flush=True)
                report["summaries"]=summarize(all_rows)
                c.write_json(out/"RAM04_report.json",report)
            report["summaries"]=summarize(all_rows)
            primary={r["model"]:r for r in report["summaries"] if r["experiment"]=="primary"}
            native=set(primary["native"]["solved"])
            report["primary_model_only"]=any(set(primary[p]["solved"])-native
                                              for p in ("loop32","loop64"))
            flat=set(primary["flat64"]["solved"])
            report["recurrence_specific"]=any(set(primary[p]["solved"])-(native|flat)
                                               for p in ("loop32","loop64"))
            report["status"]="complete";report["resources"]=guard.receipt()
    except BaseException as exc:
        report.update(status="aborted",error=repr(exc),summaries=summarize(all_rows))
        c.write_json(out/"RAM04_report.json",report)
        raise
    c.write_json(out/"RAM04_report.json",report)
    print("RAM04_FROZEN_RESULT",json.dumps(report,sort_keys=True),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mathcan",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--training",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",required=True)
    p.add_argument("--config",type=Path,
                    default=Path(__file__).with_name("config.json"))
    args=p.parse_args()
    run(args.mathcan,args.data,args.training,args.out,args.lean,args.config)
if __name__=="__main__":main()
