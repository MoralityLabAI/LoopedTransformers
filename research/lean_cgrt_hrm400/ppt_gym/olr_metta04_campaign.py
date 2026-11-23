"""OLR-METTA-04: source-registered native Lean staged policy campaign.

FREEZE: register paired dependent statements and public Hyperon claims.
FIT: use only TRAIN native audited trajectories for three small model types.
SCORE: freeze all root actions before native query execution; later decisions
       may depend only on *observed* native outcomes and a recurrent carry.

No sealed or prior held-out result is accessed. This is a CPU development
comparison and cannot certify an architecture-level mechanism advantage.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import torch

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from .data import canonical,digest,sha256_file
from .live_lean import candidate_menu,public_state
from .olr_dependent import ROOT_MENU,compile_certified,native_proof,replay_proof
from .olr_metta04_world import collect,load_world
from .olr_metta04_policy import (
    TiedStagePolicy,DirectStagePolicy,act,fit as fit_stage,training_trace
)
from .olr_metta04_episode import run_episode,replay_episode

REGISTRY="ppt.olr-metta04-registered-v1"
FIT="ppt.olr-metta04-stage-fit-v1"
SCORE="ppt.olr-metta04-stage-score-v1"
SEEDS=(17,29,43)
ARMS=("loop3","loop1","direct")
SPLITS=("train","dev","ood")
CONTROLS=("exact","fixed_r0","majority","always_stop","work_first","no_metta")


def _rows(path:Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def _write(path:Path,rows):
    path.write_text("".join(canonical(r)+"\n" for r in rows),encoding="utf-8")


def freeze(out:Path,*,seed:int=20261130):
    if out.exists():
        raise ValueError("04C root registry output must be create-only")
    train=collect("train",seed=seed,count=24,low=17,high=30)
    used=frozenset((w.proof.a,w.proof.b,w.proof.delta) for w in train)
    dev=collect("dev",seed=seed+13,count=8,low=17,high=30,excluded=used)
    ood=collect("ood",seed=seed+41,count=8,low=31,high=44)
    groups={"train":train,"dev":dev,"ood":ood}
    seen=set()
    all_public={}
    with LeanRepl(timeout=60) as lean:
        version=lean.version
        if not str(version).startswith("4.19.0"):
            raise ValueError("requires pinned native Lean 4.19")
        for split,group in groups.items():
            packets=[]
            for w in group:
                packet=w.public()
                if packet["root_id"] in seen:
                    raise ValueError("duplicated root ID in registered cohort")
                seen.add(packet["root_id"])
                lean.reset()
                native=lean.init(w.proof.statement(),pp=True)
                if candidate_menu(lean,native)!=ROOT_MENU:
                    raise ValueError("dependent legal root menu changed")
                packet["root_public_sha256"]=digest(public_state(lean,native))
                packet["source_evidence"]="synthetic_claims_and_native_Lean_public_root"
                packets.append(packet)
            all_public[split]=packets
    out.mkdir(parents=True)
    for split in SPLITS:
        _write(out/(split+".jsonl"),all_public[split])
    report={
        "schema":REGISTRY,"seed":seed,
        "root_pairs":{"train":24,"dev":8,"ood":8},
        "root_worlds":{s:len(all_public[s]) for s in SPLITS},
        "lean_version":version,"lean_sha256":sha256_file(config.LEAN_EXE),
        "repl_sha256":sha256_file(config.REPL_EXE),
        "metta_runtime":"hyperon==0.2.10",
        "policy_input":"six_me_tta_curated_claims_plus_public_stage_budget_and_feedback",
        "source_only_training_split":"train",
        "independent_query_lineages":True,
        "repair_eligibility_public_independent_of_branch":True,
        "files":{s+".jsonl":sha256_file(out/(s+".jsonl")) for s in SPLITS},
        "heldout_status":"open_development_only",
    }
    (out/"manifest.json").write_text(canonical(report)+"\n",encoding="utf-8")
    return report


def read_bank(path:Path):
    info=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    if info.get("schema")!=REGISTRY or (
            info["lean_sha256"]!=sha256_file(config.LEAN_EXE) or
            info["repl_sha256"]!=sha256_file(config.REPL_EXE)):
        raise ValueError("registered native source toolchain changed")
    rows={}
    global_ids=set()
    lineages={}
    for split in SPLITS:
        shard=path/(split+".jsonl")
        if sha256_file(shard)!=info["files"][shard.name]:
            raise ValueError("source frozen shard bytes were modified")
        parts=_rows(shard)
        if len(parts)!=info["root_worlds"][split]:
            raise ValueError("lost root from 04C frozen cohort")
        grouped=defaultdict(list)
        for r in parts:
            if (r["split"]!=split or r["root_id"] in global_ids
                    or len(r["model_features"])!=6):
                raise ValueError("duplicate/malformed registered root")
            global_ids.add(r["root_id"])
            w=load_world(r)
            if w.proof.statement()!=r["statement"] or (
                    digest(r["statement"])!=r["theorem_sha256"]):
                raise ValueError("source/dependent theorem mismatch")
            grouped[r["pair_id"]].append(r)
        if len(grouped)!=info["root_pairs"][split] or any(
                sorted(x["world"] for x in pair)!=[0,1]
                for pair in grouped.values()):
            raise ValueError("registered dependent paired-world split changed")
        lineages[split]={(load_world(r).proof.a,
                          load_world(r).proof.b,
                          load_world(r).proof.delta)
                          for r in parts}
        rows[split]=parts
    if (lineages["train"]&lineages["dev"] or
            lineages["ood"]&(lineages["train"]|lineages["dev"])):
        raise ValueError("source/evaluation theorem lineages overlap")
    return info,rows


def _source_references(lean:LeanRepl,records:list[dict]):
    """Check both native roots for every source world before teaching stages."""
    references=[]
    positives=[]
    traces=[]
    for row in records:
        w=load_world(row)
        statuses=[]
        for first in (0,1):
            ep=native_proof(lean,w.proof,first=first,tail="manual")
            if not replay_proof(lean,w.proof,ep):
                raise ValueError("native source branch replay changed")
            statuses.append(ep)
        if sum(e["kernel_closed"] for e in statuses)!=1:
            raise ValueError("source must have one kernel-certified first branch")
        correct=next(i for i,ep in enumerate(statuses) if ep["kernel_closed"])
        if correct!=w.proof.reference_branch():
            raise ValueError("native source proof conflicts with registration")
        for wrong in (False,True):
            seq=training_trace(row["model_features"],winner=correct,
                               macro_allowed=row["macro_allowed"],
                               wrong_start=wrong)
            traces.append(seq)
        positives.append((w.proof,statuses[correct]))
        references.append({"root_id":w.proof.root_id,"pair_id":w.proof.pair_id,
                           "world":w.proof.world,"source_first_winner":correct,
                           "manual_proof_tactic_calls":statuses[correct]["calls"],
                           "rejected_branch_tactic_calls":statuses[1-correct]["calls"],
                           "native_statement_sha256":digest(w.proof.statement())})
    return traces,positives,references


def fit(bank:Path,out:Path,*,steps:int=48):
    if out.exists() or not 1<=steps<=80:
        raise ValueError("new source fit with registered bounded optimizer updates")
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)
    registry,rows=read_bank(bank)
    with LeanRepl(timeout=60) as lean:
        traces,source_proofs,references=_source_references(lean,rows["train"])
    out.mkdir(parents=True)
    cert=compile_certified(source_proofs,out/"source_standalone.lean")
    if cert["axiom_free"]!=len(rows["train"]):
        raise ValueError("not all TRAIN sources have an independent certificate")
    models={}; fitted={}
    for seed in SEEDS:
        for arm in ARMS:
            model,stats=fit_stage(traces,seed=seed,arm=arm,steps=steps,
                                  batch_size=6)
            models[seed,arm]=model
            fitted[f"{seed}:{arm}"]=stats
    replay,_=fit_stage(traces,seed=SEEDS[0],arm="loop3",steps=steps,batch_size=6)
    if any(not torch.equal(x,replay.state_dict()[key])
           for key,x in models[SEEDS[0],"loop3"].state_dict().items()):
        raise ValueError("same-worker staged recurrent fit failed bit-exact replay")
    with (out/"models.pt").open("xb") as handle:
        torch.save({f"{seed}:{arm}":model.state_dict()
                   for (seed,arm),model in models.items()},handle)
    _write(out/"source_native_labels.jsonl",references)
    receipt={
        "schema":FIT,"source_registration_sha256":sha256_file(bank/"manifest.json"),
        "source_worlds":len(rows["train"]),"training_traces":len(traces),
        "source_independent_certificates":cert,
        "seeds":SEEDS,"trained_arms":ARMS,"optimizer_updates":steps,
        "model_metadata":fitted,"same_worker_bit_exact_seed17_loop3":True,
        "training_split_only":"train",
        "cpu_torch":torch.__version__,
        "artifacts":{n:sha256_file(out/n) for n in (
            "models.pt","source_native_labels.jsonl","source_standalone.lean")},
    }
    (out/"manifest.json").write_text(canonical(receipt)+"\n",encoding="utf-8")
    return receipt


def load_fit(source_bank:Path,fit_path:Path):
    info=json.loads((fit_path/"manifest.json").read_text(encoding="utf-8"))
    if (info.get("schema")!=FIT or
            info["source_registration_sha256"]!=sha256_file(source_bank/"manifest.json")
            or tuple(info["seeds"])!=SEEDS or tuple(info["trained_arms"])!=ARMS):
        raise ValueError("trained source checkpoint does not match registered corpus")
    for name,sha in info["artifacts"].items():
        if sha256_file(fit_path/name)!=sha:
            raise ValueError("trained checkpoint or native labels changed")
    saved=torch.load(fit_path/"models.pt",map_location="cpu",weights_only=True)
    models={}
    for seed in SEEDS:
        for arm in ARMS:
            obj=DirectStagePolicy() if arm=="direct" else TiedStagePolicy()
            obj.load_state_dict(saved[f"{seed}:{arm}"],strict=True)
            obj.eval();models[seed,arm]=obj
    return info,models


def _majority(claims:list[list[float]])->int:
    return 0 if sum(x[0]*x[1] for x in claims)>=0 else 1


def _scripted(name:str,w,claims):
    truth=w.proof.reference_branch()
    guess=_majority(claims)
    if name=="exact":
        first=truth
    elif name in ("fixed_r0","work_first","always_stop"):
        first=0
    else:
        first=guess
    def policy(phase,remaining,branch):
        if phase=="root":
            return "root_r"+str(first),0
        if phase=="retry":
            return "root_r"+str(1-branch),0
        if phase=="open":
            return ("work_first" if name=="work_first" else
                    "stop" if name=="always_stop" else
                    "probe_guard"),0
        if phase=="failed":
            return ("restore" if w.macro_allowed and remaining>=4
                    else "stop"),0
        if phase=="guard_ok":
            if name=="no_metta":
                return "manual",0
            return ("metta_repair" if w.macro_allowed else "manual"),0
        if phase=="closed":
            return "stop",0
        raise ValueError("unknown scripted dependent policy phase")
    return policy


def _model_policy(model,arm,claims,macro,expected_root):
    carry=None
    root_invocations=0
    def policy(phase,remaining,branch):
        nonlocal carry,root_invocations
        action,new_carry,_=act(model,claims,phase=phase,
                               remaining=remaining,branch=branch,
                               macro_allowed=macro,carry=carry,arm=arm)
        carry=new_carry
        if phase=="root":
            root_invocations+=1
            if action!=expected_root:
                raise ValueError("frozen predecision root model choice differs")
        return action,(1 if arm in ("direct","loop1") else 3)
    return policy


def _score_rows(records:list[dict]):
    groups=defaultdict(list)
    for r in records:
        groups[f"{r['split']}:{r['seed']}:{r['policy']}"].append(r)
    out={}
    for key,items in groups.items():
        pairs=defaultdict(list)
        for row in items:
            pairs[row["pair_id"]].append(row)
        if any(sorted(x["world"] for x in v)!=[0,1] for v in pairs.values()):
            raise ValueError("evaluated source pair lost a world")
        out[key]={
            "worlds":len(items),"pairs":len(pairs),
            "kernel_certified":sum(int(x["kernel_closed"]) for x in items),
            "complete_pairs":sum(all(x["kernel_closed"] for x in v) for v in pairs.values()),
            "tactic_calls":sum(x["tactic_calls"] for x in items),
            "restores":sum(x["restores"] for x in items),
            "focus_ops":sum(x["focus_ops"] for x in items),
            "metta_queries":sum(x["metta_queries"] for x in items),
            "false_stops":sum(x["false_stop"] for x in items),
            "vetoed_restores":sum(x["vetoed_restores"] for x in items),
            "model_passes":sum(x["model_passes"] for x in items),
            "mean_utility":round(sum(x["utility"] for x in items)/len(items),6),
        }
    return out


def score(bank:Path,source_fit:Path,out:Path,*,max_seconds:float=600.):
    if out.exists():
        raise ValueError("new evaluation folder required")
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)
    stamp=time.perf_counter()
    registry,cohorts=read_bank(bank)
    source,models=load_fit(bank,source_fit)
    frozen=[]
    worlds=[]
    for split in ("dev","ood"):
        for row in cohorts[split]:
            w=load_world(row)
            worlds.append((w,row))
            for (seed,arm),model in models.items():
                choice,_,_=act(model,row["model_features"],phase="root",
                              remaining=6,branch=None,
                              macro_allowed=w.macro_allowed,carry=None,arm=arm)
                frozen.append({"root_id":w.proof.root_id,"pair_id":w.proof.pair_id,
                               "world":w.proof.world,"split":split,
                               "seed":seed,"policy":arm,"root_action":choice,
                               "root_packet_sha256":digest(row["model_features"])})
    choices={(r["root_id"],r["seed"],r["policy"]):r for r in frozen}
    out.mkdir(parents=True)
    _write(out/"frozen_root_choices.jsonl",frozen)
    results=[]
    episodes=[]
    successes=[]
    with LeanRepl(timeout=60) as lean:
        for w,row in worlds:
            arms=[(0,name,None) for name in CONTROLS]
            arms +=[(seed,arm,model) for (seed,arm),model in models.items()]
            for seed,name,model in arms:
                callback=(
                    _scripted(name,w,row["model_features"]) if model is None else
                    _model_policy(model,name,row["model_features"],w.macro_allowed,
                                  choices[w.proof.root_id,seed,name]["root_action"])
                )
                ep=run_episode(lean,w.proof,claims=row["model_features"],
                    macro_allowed=w.macro_allowed,policy=callback)
                if not replay_episode(lean,w.proof,ep,macro_allowed=w.macro_allowed):
                    raise ValueError("native policy event replay rejected")
                record={"root_id":w.proof.root_id,"pair_id":w.proof.pair_id,
                        "world":w.proof.world,"split":w.proof.split,
                        "seed":seed,"policy":name,"kernel_closed":ep["kernel_closed"],
                        "false_stop":ep["false_stop"],"tactic_calls":ep["tactic_calls"],
                        "restores":ep["restores"],"focus_ops":ep["focus_ops"],
                        "vetoed_restores":ep["vetoed_restores"],
                        "metta_queries":ep["metta_queries"],
                        "model_passes":ep["model_passes"],"utility":ep["utility"],
                        "episode_sha256":digest(ep)}
                results.append(record)
                episodes.append({"identity":record,"episode":ep})
                if ep["kernel_closed"]:
                    canonical_script=[f"apply r{ep['verified_branch']}","rfl"]
                    canonical_script.extend(("apply h0","exact h1 _ h0")
                        if ep["tail"]=="fast" else ("apply h0","apply h1","apply h0"))
                    successes.append((w.proof,{"root_id":w.proof.root_id,
                                               "kernel_closed":True,"script":canonical_script}))
                if time.perf_counter()-stamp>max_seconds:
                    raise TimeoutError("native OLR-METTA-04 CPU wall budget exhausted")
    cert=compile_certified(successes,out/"standalone.lean")
    verified_roots={world.root_id for world,ep in successes}
    total_worlds=sum(len(cohorts[s]) for s in ("dev","ood"))
    if len(verified_roots)!=total_worlds or cert["axiom_free"]<total_worlds:
        raise ValueError("not all development roots have a certified exact-oracle witness")
    _write(out/"score_rows.jsonl",results)
    _write(out/"native_episodes.jsonl",episodes)
    report={
        "schema":SCORE,
        "registry_sha256":sha256_file(bank/"manifest.json"),
        "source_fit_sha256":sha256_file(source_fit/"manifest.json"),
        "dev_worlds":len(cohorts["dev"]),"ood_worlds":len(cohorts["ood"]),
        "root_model_choices_frozen_before_native_query":True,
        "model_root_predecisions":len(frozen),
        "policy_native_episodes":len(episodes),
        "independent_native_replays":len(episodes),
        "independent_kernel_certificates":cert,
        "scorecard":_score_rows(results),
        "science":"exploratory_dependent_witness_stage_controller",
        "model_flo_ps_matched":False,
        "native_six_tactic_budget_matched":True,
        "architecture_promotion_eligible":False,
        "artifacts":{n:sha256_file(out/n) for n in (
            "frozen_root_choices.jsonl","score_rows.jsonl",
            "native_episodes.jsonl","standalone.lean")},
    }
    (out/"manifest.json").write_text(canonical(report)+"\n",encoding="utf-8")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="phase",required=True)
    a=sub.add_parser("freeze")
    a.add_argument("--out",type=Path,required=True)
    a.add_argument("--seed",type=int,default=20261130)
    b=sub.add_parser("fit")
    b.add_argument("--bank",type=Path,required=True)
    b.add_argument("--out",type=Path,required=True)
    b.add_argument("--steps",type=int,default=48)
    c=sub.add_parser("score")
    c.add_argument("--bank",type=Path,required=True)
    c.add_argument("--fit",type=Path,required=True)
    c.add_argument("--out",type=Path,required=True)
    c.add_argument("--max-seconds",type=float,default=600.)
    args=p.parse_args(argv)
    if args.phase=="freeze":
        receipt=freeze(args.out,seed=args.seed)
        keys=("schema","root_worlds","lean_version")
    elif args.phase=="fit":
        receipt=fit(args.bank,args.out,steps=args.steps)
        keys=("schema","source_worlds","source_independent_certificates","model_metadata")
    else:
        receipt=score(args.bank,args.fit,args.out,max_seconds=args.max_seconds)
        keys=("schema","policy_native_episodes","independent_native_replays",
              "independent_kernel_certificates","scorecard")
    print(canonical({key:receipt[key] for key in keys}))


if __name__=="__main__":
    main()
