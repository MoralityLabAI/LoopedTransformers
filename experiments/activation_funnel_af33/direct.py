"""AF-33 source-pinned original AF23 Direct on identical heldout cycle graphs.

Native STOP+Direct ranking, forced Direct continuation, and equal public-Horn
negative certificate with original Direct legal ordering share the unchanged
6036-weight original checkpoint. No original optimizer updates occur.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch

from experiments.activation_funnel_af25 import evaluate as original
from experiments.activation_funnel_af26 import factor
from . import gym, evaluate, transfer_planner

ARMS=("AF23_DIRECT_NATIVE","AF23_DIRECT_FORCED","AF23_DIRECT_CERT")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rollout(model,cases,seed,budget_mode="strict"):
    cases=tuple(cases)
    lengths=[gym.shortest_steps(c) for c in cases]
    if budget_mode not in evaluate.BUDGET_MODES:
        raise ValueError("AF33 original Direct budget mode invalid")
    budgets=[(d if budget_mode=="strict" else d+1) if d is not None else 2
             for d in lengths]
    state={arm:[gym.initial_facts(c) for c in cases] for arm in ARMS}
    done={arm:[False]*len(cases) for arm in ARMS}
    history={arm:[[] for _ in cases] for arm in ARMS}
    graphs={c.id:gym.public_features(c) for c in cases}
    certificates={}
    work={}
    for c in cases:
        closure,trace=gym.public_fixed_point(graphs[c.id])
        certificates[c.id]=graphs[c.id]["goal_slot"] in closure
        work[c.id]=trace
    for step in range(max(budgets)):
        jobs=[(arm,i,c,state[arm][i])
              for arm in ARMS for i,c in enumerate(cases)
              if not done[arm][i] and step<budgets[i]]
        active=[(arm,i,c,s) for arm,i,c,s in jobs
                if gym.enabled(c,s) and not (
                    arm=="AF23_DIRECT_CERT" and not certificates[c.id]
                )]
        plans=[]
        for begin in range(0,len(active),32):
            sub=active[begin:begin+32]
            plans.extend(transfer_planner.decisions(
                model,[c for arm,i,c,s in sub],
                [s for arm,i,c,s in sub]))
        if len(plans)!=len(active):
            raise ValueError("source original inference trace count mismatch")
        chosen_plans={(arm,i):plan for (arm,i,c,s),plan in zip(active,plans)}
        for arm,i,c,s in jobs:
            legal=gym.enabled(c,s)
            plan=chosen_plans.get((arm,i))
            if arm=="AF23_DIRECT_CERT" and not certificates[c.id]:
                action=gym.STOP
            elif not legal:
                action=gym.STOP
            elif plan is None:
                raise AssertionError("missing original learned policy forward")
            elif arm=="AF23_DIRECT_NATIVE":
                receipt=factor.choose(plan["scores"],legal,"dd",plan["selected"])
                action=receipt["chosen"]
            else:
                action=max(legal,key=lambda a:plan["scores"]["direct"][a])
            next_state,proof=gym.sound_fire(c,s,action)
            before=gym.to_physical(c,s)
            after=gym.to_physical(c,next_state)
            rule_work=work[c.id]["rule_examinations"] if (
                arm=="AF23_DIRECT_CERT" and step==0) else 0
            h={
                "step":step,
                "before_physical":sorted(before),
                "enabled_physical":list(legal),
                "action":action,"after_physical":sorted(after),
                "proof":proof,
                "source_scores":(plan["scores"]["direct"] if plan else None),
                "source_selected":(plan["selected"] if plan else None),
                "plan_trace":(plan["trace"] if plan else None),
                "forward_nodes":(plan["trace"]["nodes_forward_count"]
                                 if plan else 0),
                "certificate_examinations":rule_work,
            }
            history[arm][i].append(h)
            state[arm][i]=next_state
            if action==gym.STOP or gym.GOAL in next_state:
                done[arm][i]=True
    rows=[]
    for arm in ARMS:
        for i,c in enumerate(cases):
            acts=history[arm][i]
            solved=gym.GOAL in state[arm][i]
            row={
                "experiment":"AF-33","arm":arm,"seed":seed,
                "budget_mode":budget_mode,
                "split":c.split,"family":c.family,
                "case_id":c.id,"pair_id":c.pair_id,
                "cycle_length":c.cycle_length,
                "swap_position":c.swap_position,
                "designed_reachable":c.reachable,
                "public_graph_sha256":gym.digest(graphs[c.id]),
                "shortest_steps_oracle":lengths[i],
                "budget_oracle":budgets[i],
                "solved":solved,
                "initial_stop":bool(acts and acts[0]["action"]==gym.STOP),
                "false_accepted_proof":bool(not c.reachable and solved),
                "root_first_off_optimal":bool(
                    c.reachable and acts and (
                        acts[0]["action"]==gym.STOP or
                        gym.public_bitset_bfs(graphs[c.id],acts[0]["after_physical"])!=lengths[i]-1
                    )
                ),
                "certificate_reachable":(certificates[c.id] if
                    arm=="AF23_DIRECT_CERT" else None),
                "certificate_work":{
                    "rule_examinations":(work[c.id]["rule_examinations"] if
                        arm=="AF23_DIRECT_CERT" else 0),
                    "closure_passes":(work[c.id]["closure_passes"] if
                        arm=="AF23_DIRECT_CERT" else 0),
                },
                "history":acts,
                "total_rule_examinations":sum(
                    h["certificate_examinations"] for h in acts),
                "total_neural_forward_nodes":sum(
                    h["forward_nodes"] for h in acts),
                "total_symbolic_expansions":0,
                "total_linear_scores":0,
                "wall_ns":0,
            }
            rows.append(row)
    return rows


def summarize(rows):
    # Native neural policy does not score a shared nonlearned ranker.
    a=evaluate.summarize(rows)
    a["original_neural_forward_nodes"]=sum(r["total_neural_forward_nodes"] for r in rows)
    return a


def evaluate_frozen(seed,zip_dir,out_dir,smoke=False):
    if seed not in (17,29,43,61,73,89,103,131):
        raise ValueError("unregistered original AF23 checkpoint seed")
    out_dir=Path(out_dir)
    if out_dir.exists():
        raise FileExistsError("original-model evaluation output must be immutable")
    out_dir.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model,provenance=original.load_frozen_original(zip_dir,seed)
    start=time.perf_counter()
    rows=[];scores={}
    for budget_mode,split in evaluate.COHORTS:
        roster=gym.cases_for(split)
        if smoke:roster=roster[:8]
        r=rollout(model,roster,seed,budget_mode)
        rows.extend(r)
        by_arm={arm:[x for x in r if x["arm"]==arm] for arm in ARMS}
        key=evaluate.cohort_key(budget_mode,split)
        if smoke:
            scores[key]={arm:{
                "completed":sum(x["solved"] for x in by_arm[arm]),
                "correct_negative_STOP":sum(x["initial_stop"] for x in
                        by_arm[arm] if not x["designed_reachable"])
            } for arm in ARMS}
        else:
            scores[key]={arm:summarize(by_arm[arm]) for arm in ARMS}
    raw=("".join(json.dumps(r,sort_keys=True,allow_nan=False)+"\n"
                 for r in rows)).encode()
    (out_dir/"direct_episodes.jsonl").write_bytes(raw)
    receipt={
        "experiment":"AF-33","seed":seed,"smoke":smoke,
        "original_frozen_AF23_checkpoint":provenance,
        "original_checkpoint_parameter_updates":0,
        "original_model_parameters":6036,
        "original_model_thread_count":torch.get_num_threads(),
        "source_manifest_sha256":gym.manifest()["manifest_sha256"],
        "protocol_sha256":sha((Path(__file__).parent/"PROTOCOL.json").read_bytes()),
        "direct_episodes_sha256":sha(raw),
        "direct_episode_count":len(rows),
        "budget_modes":list(evaluate.BUDGET_MODES),
        "scores":scores,"modes":list(ARMS),
        "original_model_forward_nodes":sum(
            r["total_neural_forward_nodes"] for r in rows),
        "elapsed_seconds":time.perf_counter()-start,
        "verdict":"SMOKE_ONLY" if smoke else
            "AWAITING_INDEPENDENT_SAME_SOURCE_DIRECT_AUDIT",
    }
    (out_dir/"direct_evaluation.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--original-zip-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    r=evaluate_frozen(args.seed,args.original_zip_dir,args.out,args.smoke)
    print(json.dumps({"seed":args.seed,"cases":r["direct_episode_count"],
                      "model_forward_nodes":r["original_model_forward_nodes"],
                      "smoke":args.smoke}))
if __name__=="__main__":
    main()
