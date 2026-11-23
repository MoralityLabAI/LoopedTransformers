"""OLR-METTA-03 preregistered native evidence-budget campaign.

Phases must run in order:
  freeze: Hyperon public evidence admission and Lean initial-goal identity
  fit:    TRAIN-only full native action table, replay, verified source fitting
  score:  freeze ALL query choices before native evaluation and proof replay

The restricted policy observation is six synthetic public witness claims.
A full-information arithmetic oracle remains a separate reference arm.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import platform

import torch

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from .data import canonical,digest,sha256_file
from .live_lean import candidate_menu, public_state
from .metta_budget_bridge import (
    SCHEMA as TASK, ACTIONS, generate, masked_packet,
    uncurated_features, native_episode, replay_episode, COST,
)
from .metta_budget_model import (
    EvidenceLoop, EvidenceDirect, fit as fit_model, select,
    majority_action, source_threshold,
)
from .metta_obligations import HYPERON_VERSION
from .olr_delayed import DelayedWorld, STMT, ROOT_MENU, compile_scripts

REGISTRY="ppt.olr-metta-budget-registry.v1"
FIT="ppt.olr-metta-budget-fit.v1"
SCORE="ppt.olr-metta-budget-score.v1"
SEEDS=(17,29,43)
ARMS=("loop3","loop1","direct")
SPLITS=("train","dev","ood")
BASE=("always_probe","fixed_r0","majority","majority_raw","arithmetic_oracle")


def _write_lines(path:Path,rows)->None:
    path.write_text("".join(canonical(r)+"\n" for r in rows),encoding="utf-8")


def _read_lines(path:Path)->list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def _native_world(row:dict)->DelayedWorld:
    # Execution-side reconstruction. This theorem is *not* a model feature.
    m=STMT.fullmatch(row["theorem_statement"])
    if m is None:
        raise ValueError("invalid registered Lean theorem grammar")
    a,b,c,d,e,f=(int(x) for x in m.groups())
    if a!=d or b!=e:
        raise ValueError("source operands changed across candidate implications")
    world=DelayedWorld(row["pair_id"],row["world"],a,b,(c,f))
    if world.statement()!=row["theorem_statement"]:
        raise ValueError("registered statement changed")
    return world


def _hashes(directory:Path,names)->dict[str,str]:
    return {n:sha256_file(directory/n) for n in names}


def freeze(out:Path, *, seed:int=20261129,train_pairs:int=24,
           dev_pairs:int=8,ood_pairs:int=8)->dict:
    if out.exists() or (train_pairs,dev_pairs,ood_pairs)!=(24,8,8):
        raise ValueError("registered source split sizes and create-only output required")
    train=generate("train",seed=seed,count=train_pairs,low=17,high=30)
    used=frozenset(w.lineage() for w in train)
    dev=generate("dev",seed=seed+11,count=dev_pairs,low=17,high=30,exclude=used)
    ood=generate("ood",seed=seed+31,count=ood_pairs,low=31,high=44)
    worlds={"train":train,"dev":dev,"ood":ood}
    seen=set()
    prepared={}
    with LeanRepl(timeout=60) as lean:
        version=lean.version
        if not str(version).startswith("4.19.0"):
            raise ValueError("pinned native Lean 4.19 required")
        for split,group in worlds.items():
            rows=[]
            for w in group:
                if (w.pair_id,w.world) in seen:
                    raise ValueError("duplicated paired world")
                seen.add((w.pair_id,w.world))
                packet=masked_packet(w)
                # Public syntactic Lean state is recorded for custody; the
                # model only receives packet['model_features'].
                lean.reset()
                state=lean.init(w.lean_world().statement(),pp=True)
                if candidate_menu(lean,state)!=ROOT_MENU:
                    raise ValueError("Lean root actions changed before scoring")
                packet["lean_public_sha256"]=digest(public_state(lean,state))
                packet["truth_mode_source_only"]="synthetic_external_testimony"
                rows.append(packet)
            prepared[split]=rows
    def lineage(rows):
        return {(w.a,w.b,w.delta) for w in rows if w.world==0}
    if lineage(train)&lineage(dev) or lineage(ood)&(lineage(train)|lineage(dev)):
        raise ValueError("source and query world origins overlap")
    out.mkdir(parents=True)
    for split,rows in prepared.items():
        _write_lines(out/(split+".jsonl"),rows)
    receipt={
        "schema":REGISTRY,"task_schema":TASK,"seed":seed,
        "split_pairs":{"train":train_pairs,"dev":dev_pairs,"ood":ood_pairs},
        "split_worlds":{s:len(prepared[s]) for s in SPLITS},
        "policy_features":"six_synthetic_claims_without_Lean_arithmetic_operands",
        "source_roles":{s:"training_only" if s=="train" else "evaluation_only"
                        for s in SPLITS},
        "metta_runtime":"hyperon=="+HYPERON_VERSION,
        "lean_version":version,
        "lean_sha256":sha256_file(config.LEAN_EXE),
        "repl_sha256":sha256_file(config.REPL_EXE),
        "source_truth_labels_excluded_from_model":True,
        "scientific_scope":"synthetic_testimony_over_native_Lean_arithmetic",
        "files":_hashes(out,[s+".jsonl" for s in SPLITS]),
        "model_training_eligible_from_split":"train",
    }
    (out/"manifest.json").write_text(canonical(receipt)+"\n",encoding="utf-8")
    return receipt


def read_bank(directory:Path):
    m=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if m.get("schema")!=REGISTRY or m.get("policy_features")!=(
            "six_synthetic_claims_without_Lean_arithmetic_operands"):
        raise ValueError("unexpected MeTTa budget registry")
    if (m["lean_sha256"]!=sha256_file(config.LEAN_EXE)
            or m["repl_sha256"]!=sha256_file(config.REPL_EXE)):
        raise ValueError("native Lean runtime no longer matches registration")
    rows={}
    roots=set()
    for split in SPLITS:
        path=directory/(split+".jsonl")
        if sha256_file(path)!=m["files"][path.name]:
            raise ValueError("registered evidence bytes changed")
        records=_read_lines(path)
        if len(records)!=m["split_worlds"][split]:
            raise ValueError("registered world count changed")
        pairs=defaultdict(list)
        for r in records:
            if r["split"]!=split or r["root_id"] in roots or len(r["model_features"])!=6:
                raise ValueError("duplicate or malformed source packet")
            roots.add(r["root_id"])
            pairs[r["pair_id"]].append(r)
            _native_world(r)
            if any(len(v)!=6 for v in r["model_features"]):
                raise ValueError("model-visible evidence has wrong width")
        if any(sorted(x["world"] for x in group)!=[0,1]
               for group in pairs.values()):
            raise ValueError("lost paired proof worlds")
        rows[split]=records
    return m,rows


def _native_table(lean:LeanRepl,rows:list[dict],*,replay:bool)->tuple[list[dict],list[tuple]]:
    """Verify all policy actions with real Lean; only real kernels give success."""
    output=[]
    successful=[]
    for row in rows:
        world=_native_world(row)
        attempts={}
        for action in ACTIONS:
            ep=native_episode(lean,world,action)
            if replay and replay_episode(lean,world,ep) is not True:
                raise ValueError("native verifier replay failed")
            attempts[action]=ep
            if ep["kernel_closed"]:
                successful.append((world,{
                    "kernel_closed":True,
                    "script":[ep["verified_branch"],"rfl","apply h0","apply h1"]
                }))
        # The source full-information witness is a structural ceiling and
        # is checked only against actual native episode results.
        if not attempts["probe_r0"]["kernel_closed"]:
            raise ValueError("native registered probe did not certify a proof")
        if sum(attempts[a]["kernel_closed"] for a in ACTIONS[:2])!=1:
            raise ValueError("two root tactics lost their unique native winner")
        output.append({
            "root_id":row["root_id"],"pair_id":row["pair_id"],
            "world":row["world"],
            "model_features":row["model_features"],
            "native_action_utilities":[attempts[a]["utility"] for a in ACTIONS],
            "native_action_receipts":attempts,
        })
    return output,successful


def fit(bank:Path,out:Path,*,steps:int=80):
    if out.exists():
        raise ValueError("source fit output must be new")
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)
    original,roots=read_bank(bank)
    with LeanRepl(timeout=60) as lean:
        source,proofs=_native_table(lean,roots["train"],replay=True)
    out.mkdir(parents=True)
    cert=compile_scripts(proofs,out/"source_standalone.lean")
    if cert["axiom_clean"]!=len(roots["train"]):
        raise ValueError("some native source roots lack independent certificates")
    selected=source_threshold(source)
    models={}
    metadata={}
    for seed in SEEDS:
        for arm in ARMS:
            model,info=fit_model(source,seed=seed,arm=arm,steps=steps)
            models[seed,arm]=model
            metadata[f"{seed}:{arm}"]=info
        # The same architecture is initialized identically for one- and
        # three-pass recurrence. Initial tensor identity is checked by test.
    replay,_=fit_model(source,seed=SEEDS[0],arm="loop3",steps=steps)
    if not all(torch.equal(tensor,replay.state_dict()[key])
               for key,tensor in models[SEEDS[0],"loop3"].state_dict().items()):
        raise ValueError("in-worker model replay changed tensor values")
    with (out/"models.pt").open("xb") as file:
        torch.save({f"{seed}:{arm}":model.state_dict()
                    for (seed,arm),model in models.items()},file)
    _write_lines(out/"source_rewards.jsonl",source)
    receipt={
        "schema":FIT,"registration_sha256":sha256_file(bank/"manifest.json"),
        "training_worlds":len(source),"verified_source_scripts":cert,
        "training_steps":steps,"seeds":SEEDS,"arms":ARMS,
        "trained_models":metadata,"source_majority_calibration":selected,
        "torch_version":torch.__version__,"deterministic":True,
        "cpu_threads":torch.get_num_threads(),
        "model_training_only_from_source":True,
        "same_worker_bit_exact_seed17_loop3":True,
        "artifacts":_hashes(out,["models.pt","source_rewards.jsonl","source_standalone.lean"]),
    }
    (out/"manifest.json").write_text(canonical(receipt)+"\n",encoding="utf-8")
    return receipt


def load_fit(bank:Path,path:Path):
    r=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    if (r.get("schema")!=FIT or
            r["registration_sha256"]!=sha256_file(bank/"manifest.json")
            or tuple(r["seeds"])!=SEEDS or tuple(r["arms"])!=ARMS):
        raise ValueError("source fit changed before query evaluation")
    for name,expected in r["artifacts"].items():
        if sha256_file(path/name)!=expected:
            raise ValueError("source model or source truth receipt changed")
    weights=torch.load(path/"models.pt",map_location="cpu",weights_only=True)
    models={}
    for seed in SEEDS:
        for arm in ARMS:
            model=EvidenceDirect() if arm=="direct" else EvidenceLoop()
            model.load_state_dict(weights[f"{seed}:{arm}"],strict=True)
            model.eval()
            models[seed,arm]=model
    return r,models


def _oracle(world:DelayedWorld)->str:
    if world.a+world.b==world.targets[0]:
        return "skip_r0"
    if world.a+world.b==world.targets[1]:
        return "skip_r1"
    raise ValueError("registered theorem has no native arithmetic branch")


def group_score(rows:list[dict])->dict:
    groups=defaultdict(list)
    for row in rows:
        groups[f"{row['split']}:{row['seed']}:{row['policy']}"].append(row)
    results={}
    for name,records in groups.items():
        pairs=defaultdict(list)
        for record in records:
            pairs[record["pair_id"]].append(record)
        if any(sorted(r["world"] for r in subset)!=[0,1]
               for subset in pairs.values()):
            raise ValueError("score lost paired source root")
        results[name]={
            "worlds":len(records),"pairs":len(pairs),
            "certified":sum(int(r["kernel_closed"]) for r in records),
            "both_worlds":sum(all(r["kernel_closed"] for r in group) for group in pairs.values()),
            "native_tactic_calls":sum(r["tactic_calls"] for r in records),
            "mean_utility":round(sum(r["utility"] for r in records)/len(records),6),
            "paid_probes":sum(r["action"]=="probe_r0" for r in records),
        }
    return results


def score(bank:Path,source_fit:Path,out:Path):
    if out.exists():
        raise ValueError("query result output must be new")
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)
    registration,roots=read_bank(bank)
    fit_receipt,models=load_fit(bank,source_fit)
    threshold=fit_receipt["source_majority_calibration"]["threshold"]
    choices=[]
    for split in ("dev","ood"):
        for row in roots[split]:
            world=_native_world(row)
            policies={
                (0,"always_probe"):"probe_r0",
                (0,"fixed_r0"):"skip_r0",
                (0,"arithmetic_oracle"):_oracle(world),
                (0,"majority") :majority_action(row["model_features"],threshold),
                (0,"majority_raw"):majority_action(uncurated_features(row),threshold),
            }
            for (seed,arm),model in models.items():
                policies[seed,arm]=select(model,row["model_features"],arm=arm)
                if arm=="loop3":
                    policies[seed,"loop3_uncurated"]=select(
                        model,uncurated_features(row),arm=arm)
                    no_admission=[[*features[:2],0.,*features[3:]]
                                  for features in row["model_features"]]
                    policies[seed,"loop3_no_admission"]=select(
                        model,no_admission,arm=arm)
            for (seed,policy),action in policies.items():
                choices.append({"root_id":row["root_id"],"pair_id":row["pair_id"],
                    "world":row["world"],"split":split,"seed":seed,
                    "policy":policy,"action":action,
                    "model_packet_sha256":digest(row["model_features"])})
    out.mkdir(parents=True)
    # All readouts are irrevocably decided before any eval-native outcome.
    _write_lines(out/"frozen_choices.jsonl",choices)
    actions={}
    native_receipts=[]
    proofs=[]
    with LeanRepl(timeout=60) as lean:
        for split in ("dev","ood"):
            table,compiled=_native_table(lean,roots[split],replay=True)
            proofs.extend(compiled)
            for row in table:
                for action in ACTIONS:
                    ep=row["native_action_receipts"][action]
                    actions[row["root_id"],action]=ep
                    native_receipts.append({"root_id":row["root_id"],
                        "action":action,"receipt":ep})
    cert=compile_scripts(proofs,out/"eval_standalone.lean")
    if cert["axiom_clean"]!=sum(len(roots[s]) for s in ("dev","ood")):
        raise ValueError("evaluation standalone proof count changed")
    score_rows=[]
    for row in choices:
        result=actions[row["root_id"],row["action"]]
        score_rows.append({**row,
            "kernel_closed":result["kernel_closed"],
            "tactic_calls":result["tactic_calls"],
            "utility":result["utility"],
            "native_receipt_sha256":digest(result)})
    _write_lines(out/"native_events.jsonl",native_receipts)
    _write_lines(out/"score_rows.jsonl",score_rows)
    receipt={
        "schema":SCORE,"registration_sha256":sha256_file(bank/"manifest.json"),
        "source_fit_sha256":sha256_file(source_fit/"manifest.json"),
        "model_choices_committed_before_native_scoring":True,
        "query_worlds":sum(len(roots[s]) for s in ("dev","ood")),
        "native_independent_episodes":len(native_receipts),
        "native_replays":len(native_receipts),
        "standalone":cert,"scorecard":group_score(score_rows),
        "root_experiment_unit":"paired_worlds",
        "all_comparators_same_six_call_cap":True,
        "full_knowledge_arithmetic_oracle_is_privileged":True,
        "model_inference_compute_matched":False,
        "scientific_scope":"synthetic_evidence_budget_pilot_not_native_hidden_Lean_theorem",
        "architecture_promotion_eligible":False,
        "artifacts":_hashes(out,["frozen_choices.jsonl","native_events.jsonl",
                                "score_rows.jsonl","eval_standalone.lean"]),
    }
    (out/"manifest.json").write_text(canonical(receipt)+"\n",encoding="utf-8")
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest="phase",required=True)
    a=sp.add_parser("freeze")
    a.add_argument("--out",type=Path,required=True)
    a.add_argument("--seed",type=int,default=20261129)
    b=sp.add_parser("fit")
    b.add_argument("--bank",type=Path,required=True)
    b.add_argument("--out",type=Path,required=True)
    b.add_argument("--steps",type=int,default=80)
    c=sp.add_parser("score")
    c.add_argument("--bank",type=Path,required=True)
    c.add_argument("--fit",type=Path,required=True)
    c.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    if args.phase=="freeze":
        result=freeze(args.out,seed=args.seed)
        fields=("schema","split_worlds","metta_runtime","lean_version")
    elif args.phase=="fit":
        result=fit(args.bank,args.out,steps=args.steps)
        fields=("schema","training_worlds","verified_source_scripts","trained_models")
    else:
        result=score(args.bank,args.fit,args.out)
        fields=("schema","query_worlds","native_independent_episodes","standalone","scorecard")
    print(canonical({name:result[name] for name in fields}))


if __name__=="__main__":
    main()
