"""OLR-01D: freeze multi-goal roots, then score held frozen 01C checkpoints.

Root decisions are computed from public Lean features before any 01D outcome
is read by a model. Native actions, budgeted rollback, independent event replay
and standalone Lean compilation are audited separately. This remains a
controlled search/representation pilot rather than an online-learning claim.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import time
from pathlib import Path

import torch

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from .data import canonical, digest, sha256_file
from .olr_delayed import DelayedWorld, ARMS, ROOT_MENU, observe_root, run_path, replay_path, compile_scripts
from .olr_guards import generate_pairs
from .olr_guard_fit import load_bank
from .olr_guard_model import GuardScorer, choose, exact_arithmetic_reference

REGISTRY="ppt.olr-delayed-registry.v1"
RESULTS="ppt.olr-delayed-score.v1"
BUDGETS=(4,6)
SEEDS=(17,29,43)
MODEL_ARMS=("cross","cross_onepass","direct","isolated","frozen_cross")
BASELINES=("fixed_r0","exact_arithmetic")


def existing_arithmetic(bank: Path) -> frozenset[tuple[int,int,int]]:
    _, splits=load_bank(bank)
    seen=set()
    for rows in splits.values():
        for row in rows:
            if row["world"]!=0:
                continue
            a,b,c=(int(float(v)*s) for v,s in zip(row["features"][0],(40,40,80)))
            # Stored floating point source features are rational scale
            # projections; use the fully parsed integer theorem to avoid
            # half-even roundoff and separate source from query.
            from .olr_guard_fit import reconstruct
            root=reconstruct(row)
            seen.add((root.a,root.b,root.targets[1]-root.targets[0]))
    return frozenset(seen)


def frozen_root_bank(source_bank: Path, seed: int, pairs: int, out: Path):
    if out.exists() or not 1<=pairs<=16:
        raise ValueError("new registry directory and 1..16 root pairs required")
    excluded=existing_arithmetic(source_bank)
    selected=generate_pairs("ood",seed,pairs,low=9,high=16,exclude=excluded)
    if any((p.a,p.b,p.delta) in excluded for p in selected):
        raise ValueError("01D reused source/dev/OOD arithmetic tuple")
    root_rows=[]
    private_rows=[]
    compiled=[]
    with LeanRepl(timeout=60) as lean:
        version=lean.version
        if not str(version).startswith("4.19.0"):
            raise ValueError("pinned Lean 4.19 required")
        for pair in selected:
            for bit in (0,1):
                source=pair.world(bit)
                world=DelayedWorld(pair.pair_id,bit,source.a,source.b,source.targets)
                visible=observe_root(lean,world)
                public={"pair":world.pair,"world":bit,"root":visible["root"],
                        "statement":world.statement(),
                        "public_sha256":visible["source_sha256"],
                        "menu":visible["menu"],"features":visible["features"]}
                if tuple(public["menu"])!=ROOT_MENU:
                    raise ValueError("registered menu changed before scoring")
                attempts=[run_path(lean,world,idx,"probe_early",4) for idx in (0,1)]
                labels=[int(t["kernel_closed"]) for t in attempts]
                if labels.count(1)!=1:
                    raise ValueError("no unique feasible native reference tactic")
                if labels.index(1)!=(0 if bit==0 else 1):
                    raise ValueError("generator witness disagrees with native Lean")
                if any(t["root_public_sha256"]!=public["public_sha256"]
                       for t in attempts):
                    raise ValueError("reference native public input differed")
                root_rows.append(public)
                private_rows.append({"pair":world.pair,"world":bit,"root":public["root"],
                    "witness_action":"apply r"+str(labels.index(1)),
                    "reference_status":[a["status"] for a in attempts],
                    "reference_native_checks":[a["tactic_calls"] for a in attempts]})
                compiled.append((world,attempts[labels.index(1)]))
        if len({r["pair"] for r in root_rows})!=pairs:
            raise ValueError("repeated registered proof-root lineage")
    out.mkdir(parents=True)
    (out/"public.jsonl").write_text(
        "".join(canonical(r)+"\n" for r in root_rows),encoding="utf-8")
    (out/"private_witnesses.jsonl").write_text(
        "".join(canonical(r)+"\n" for r in private_rows),encoding="utf-8")
    proofs=compile_scripts(compiled,out/"reference.lean")
    record={
        "schema":REGISTRY,"seed":seed,"pairs":pairs,"worlds":2*pairs,
        "source_manifest_sha256":sha256_file(source_bank/"manifest.json"),
        "source_bank_root_lineage_excluded":True,
        "lean_version":version, "lean_exe_sha256":sha256_file(config.LEAN_EXE),
        "repl_exe_sha256":sha256_file(config.REPL_EXE),
        "candidate_witnesses_native_checked":len(private_rows)*2,
        "reference_certificates":proofs,
        "model_training_eligible":False,"predecision_outcomes_hidden":True,
        "readout_status":"native_registration_before_model_scoring",
    }
    record["artifacts"]={n:sha256_file(out/n) for n in
                         ("public.jsonl","private_witnesses.jsonl","reference.lean")}
    (out/"manifest.json").write_text(canonical(record)+"\n",encoding="utf-8")
    return record


def load_registry(path: Path, bank: Path) -> tuple[dict,list[dict],list[dict]]:
    m=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    if (m.get("schema")!=REGISTRY
            or m["source_manifest_sha256"]!=sha256_file(bank/"manifest.json")
            or m["lean_exe_sha256"]!=sha256_file(config.LEAN_EXE)
            or m["repl_exe_sha256"]!=sha256_file(config.REPL_EXE)):
        raise ValueError("changed native/root/source registration")
    for name,digest_value in m["artifacts"].items():
        if sha256_file(path/name)!=digest_value:
            raise ValueError("changed registered public or private bytes")
    public=[json.loads(l) for l in (path/"public.jsonl").read_text().splitlines() if l]
    private=[json.loads(l) for l in (path/"private_witnesses.jsonl").read_text().splitlines() if l]
    if len(public)!=m["worlds"] or len(private)!=m["worlds"]:
        raise ValueError("registered world count changed")
    if len({x["root"] for x in public})!=len(public):
        raise ValueError("registered same theorem twice")
    return m,public,private


def read_models(source_fit: Path, bank: Path):
    source=json.loads((source_fit/"manifest.json").read_text(encoding="utf-8"))
    if source.get("schema")!="ppt.olr-guard-fit.v1" or source["source_manifest_sha256"]!=sha256_file(bank/"manifest.json"):
        raise ValueError("fitted checkpoint source does not match frozen 01C bank")
    for n in ("models.pt","decisions.jsonl"):
        if source["artifacts"][n]!=sha256_file(source_fit/n):
            raise ValueError("01C checkpoint or source decisions changed")
    if tuple(source["seed_roster"])!=SEEDS:
        raise ValueError("unexpected model seed roster")
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)
    states=torch.load(source_fit/"models.pt",map_location="cpu",weights_only=True)
    models={}
    for seed in SEEDS:
        for arm in MODEL_ARMS:
            name=f"{seed}:{arm}"
            meta=source["training"][name]
            if meta["parameters"]!=17985 or meta["width"]!=32:
                raise ValueError("frozen OLR checkpoint architecture changed")
            mode="cross" if arm in ("cross_onepass","frozen_cross") else arm
            model=GuardScorer(width=32,mode=mode)
            model.load_state_dict(states[name],strict=True)
            model.eval()
            models[(seed,arm)]=model
    return source,models


def score_groups(records: list[dict]) -> dict:
    grouped=defaultdict(list)
    for r in records:
        key=f"{r['budget']}:{r['schedule']}:{r['seed']}:{r['policy']}"
        grouped[key].append(r)
    output={}
    for key,rows in grouped.items():
        pairs=defaultdict(list)
        for row in rows:
            pairs[row["pair"]].append(row)
        if any(len(v)!=2 or sorted(x["world"] for x in v)!=[0,1]
               for v in pairs.values()):
            raise ValueError("lost two-world pair during scoring")
        output[key]={
            "worlds":len(rows),"pairs":len(pairs),
            "certified_proofs":sum(int(x["kernel_closed"]) for x in rows),
            "both_worlds_certified":sum(all(x["kernel_closed"] for x in v)
                                         for v in pairs.values()),
            "tactic_calls":sum(x["tactic_calls"] for x in rows),
            "focus_actions":sum(x["focus_actions"] for x in rows),
            "rollbacks":sum(x["rollbacks"] for x in rows),
        }
    return output


def score(registry: Path, source_bank: Path, source_fit: Path, out: Path,
          *, max_seconds: float=500.):
    if out.exists():
        raise ValueError("score output directory must be fresh")
    start=time.monotonic()
    registered,public,private=load_registry(registry,source_bank)
    source,models=read_models(source_fit,source_bank)
    choices=[]
    with LeanRepl(timeout=60) as lean:
        if lean.version!=registered["lean_version"]:
            raise ValueError("different native Lean version")
        for root in public:
            world=DelayedWorld.from_row(root)
            visible=observe_root(lean,world)
            if (visible["root"]!=root["root"] or
                    visible["source_sha256"]!=root["public_sha256"] or
                    visible["features"]!=root["features"] or
                    visible["menu"]!=root["menu"]):
                raise ValueError("registered public theorem/policy packet drifted")
            # Public features alone are forwarded to every learned readout.
            actions={(0,"fixed_r0"):0,
                     (0,"exact_arithmetic"):exact_arithmetic_reference(visible["features"])}
            for (seed,arm),model in models.items():
                depth=1 if arm=="cross_onepass" else 3
                actions[(seed,arm)]=choose(model,visible["features"],depth=depth)
            for (seed,arm),index in actions.items():
                choices.append({"root":root["root"],"pair":root["pair"],
                    "world":root["world"],"seed":seed,"policy":arm,
                    "initial":index, "public_packet_sha256":root["public_sha256"]})
    out.mkdir(parents=True)
    (out/"frozen_choices.jsonl").write_text(
        "".join(canonical(x)+"\n" for x in choices),encoding="utf-8")
    # None of the next calls can change the decisions above.
    outcomes={}
    compiled=[]
    replays=0
    with LeanRepl(timeout=60) as lean:
        for row in public:
            world=DelayedWorld.from_row(row)
            for budget in BUDGETS:
                for schedule in ARMS:
                    for initial in (0,1):
                        ep=run_path(lean,world,initial,schedule,budget)
                        if ep["root_public_sha256"]!=row["public_sha256"]:
                            raise ValueError("candidate action saw modified source")
                        key=(row["root"],budget,schedule,initial)
                        outcomes[key]=ep
                        if not replay_path(lean,world,ep)["replay_passed"]:
                            raise ValueError("native replay did not verify")
                        replays+=1
                        compiled.append((world,ep))
                        if time.monotonic()-start>max_seconds:
                            raise TimeoutError("OLR-01D bounded native CPU wall cap exceeded")
    # Structural checks on the registered world pairs and native resource costs.
    truths={r["root"]:r for r in private}
    for row in public:
        root=row["root"]
        winner=int(truths[root]["witness_action"][-1])
        for schedule in ARMS:
            for initial in (0,1):
                e4=outcomes[root,4,schedule,initial]
                e6=outcomes[root,6,schedule,initial]
                if initial==winner:
                    if not (e4["kernel_closed"] and e6["kernel_closed"]
                            and e4["tactic_calls"]==4 and e6["tactic_calls"]==4):
                        raise ValueError("registered native winner lost its four-step witness")
                else:
                    if (e4["kernel_closed"] or e4["rollbacks"]!=1 or
                            e4["tactic_calls"]!=(2 if schedule=="probe_early" else 4)):
                        raise ValueError("wrong-branch four-call witness changed")
                    if schedule=="probe_early":
                        if not (e6["kernel_closed"] and e6["tactic_calls"]==6 and e6["rollbacks"]==1):
                            raise ValueError("early native guard cannot recover a six-call proof")
                    elif e6["kernel_closed"] or e6["tactic_calls"]!=4:
                        raise ValueError("late guard wrongly recovered an over-budget proof")
    proof_receipt=compile_scripts(compiled,out/"standalone.lean")
    results=[]
    for choice in choices:
        for budget in BUDGETS:
            for schedule in ARMS:
                ep=outcomes[choice["root"],budget,schedule,choice["initial"]]
                results.append({
                    "root":choice["root"],"pair":choice["pair"],"world":choice["world"],
                    "seed":choice["seed"],"policy":choice["policy"],
                    "budget":budget,"schedule":schedule,"initial":choice["initial"],
                    "kernel_closed":ep["kernel_closed"],"tactic_calls":ep["tactic_calls"],
                    "focus_actions":ep["focus_actions"],"rollbacks":ep["rollbacks"],
                    "trajectory_sha256":digest(ep),
                })
    (out/"native_trajectories.jsonl").write_text(
        "".join(canonical(x)+"\n" for x in outcomes.values()),encoding="utf-8")
    (out/"score_rows.jsonl").write_text(
        "".join(canonical(x)+"\n" for x in results),encoding="utf-8")
    summary=score_groups(results)
    receipt={
        "schema":RESULTS,"source_registration_sha256":sha256_file(registry/"manifest.json"),
        "source_fit_sha256":sha256_file(source_fit/"models.pt"),
        "lean_version":registered["lean_version"],"model_seeds":SEEDS,
        "root_pairs":registered["pairs"],"root_worlds":registered["worlds"],
        "budgets":BUDGETS,"schedules":ARMS,
        "model_root_decisions":len(choices),
        "native_unique_trajectories":len(outcomes),
        "native_event_replays":replays,
        "source_witnesses_private_until_after_choices":True,
        "standalone":proof_receipt,
        "scorecard":summary,
        "model_parameters":17985,
        "model_compute_matched":False,
        "learned_feedback_update":False,
        "architecture_promotion_eligible":False,
        "scope":"frozen_pretrained_root_scoring_plus_deterministic_native_search",
    }
    receipt["artifacts"]={n:sha256_file(out/n) for n in (
        "frozen_choices.jsonl","native_trajectories.jsonl","score_rows.jsonl","standalone.lean")}
    (out/"manifest.json").write_text(canonical(receipt)+"\n",encoding="utf-8")
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest="op",required=True)
    a=sp.add_parser("freeze")
    a.add_argument("--source-bank",type=Path,required=True)
    a.add_argument("--seed",type=int,default=20261127)
    a.add_argument("--pairs",type=int,default=8)
    a.add_argument("--out",type=Path,required=True)
    b=sp.add_parser("score")
    b.add_argument("--registry",type=Path,required=True)
    b.add_argument("--source-bank",type=Path,required=True)
    b.add_argument("--source-fit",type=Path,required=True)
    b.add_argument("--out",type=Path,required=True)
    b.add_argument("--max-seconds",type=float,default=500.)
    args=p.parse_args(argv)
    if args.op=="freeze":
        receipt=frozen_root_bank(args.source_bank,args.seed,args.pairs,args.out)
    else:
        receipt=score(args.registry,args.source_bank,args.source_fit,args.out,max_seconds=args.max_seconds)
    if args.op=="score":
        main_keys=("schema","root_pairs","native_unique_trajectories","native_event_replays",
                   "standalone","scorecard")
    else:
        main_keys=("schema","worlds","candidate_witnesses_native_checked","reference_certificates")
    print(canonical({key:receipt[key] for key in main_keys}))


if __name__=="__main__":
    main()
