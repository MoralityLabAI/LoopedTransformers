"""AF-33 held-out complete host-legal proof episodes, public-only rankers.

Every controller sees a public PHYSICAL graph and current physical fact set.
The scorer alone sees Case, true BFS distance and the episode budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

from experiments.activation_funnel_af24 import symbolic
from . import features, gym, ranker, stop

SPLITS = ("held_depth5", "held_depth6")
BUDGET_MODES = ("strict", "slack")
COHORTS = tuple((mode, split) for mode in BUDGET_MODES for split in SPLITS)

def cohort_key(mode, split):
    if mode not in BUDGET_MODES or split not in SPLITS:
        raise ValueError("unregistered AF33 budget or held cohort")
    return mode + "_" + split.removeprefix("held_")
CERT_ARMS = ("RANK_CERT", "UNTRAINED_CERT", "PATH_CERT", "ANCESTRY_CERT",
             "FIXED_CERT", "RANDOM_CERT")
SYMBOLIC_ARMS = ("CLOSURE", "BFS16", "BFS64")
LEARNED_STOP_ARMS = ("BAL_STOP", "J12_STOP")
ARMS = CERT_ARMS + LEARNED_STOP_ARMS + SYMBOLIC_ARMS
SYM_MAP = {"CLOSURE":"closure_direct",
           "BFS16":"bfs_cap16","BFS64":"bfs_cap64"}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _ranked_action(public, state, mode, model, seed, step):
    """No access to split, pair, truth, private logical atom or budget."""
    vectors, work = features.extract(public, state)
    if not vectors:
        return gym.STOP, work
    if mode == "RANK_CERT":
        selected = ranker.choose(model, vectors)
        work["linear_scores_computed"] = len(vectors)
    elif mode == "UNTRAINED_CERT":
        selected = ranker.choose(model, vectors, untrained=True)
        work["linear_scores_computed"] = len(vectors)
    elif mode == "PATH_CERT":
        selected = min(vectors, key=lambda a: (-vectors[a][1], a))
        work["linear_scores_computed"] = 0
    elif mode == "ANCESTRY_CERT":
        selected = min(vectors, key=lambda a: (-vectors[a][0], a))
        work["linear_scores_computed"] = 0
    elif mode == "FIXED_CERT":
        selected = min(vectors)
        work["linear_scores_computed"] = 0
    elif mode == "RANDOM_CERT":
        rng = random.Random(int(gym.digest(
            ["AF33_RANDOM_ACTION", public, sorted(state), seed, step]
        )[:16],16))
        selected = rng.choice(sorted(vectors))
        work["linear_scores_computed"] = 0
    else:
        raise ValueError("AF33 unregistered ranker arm")
    if selected not in vectors:
        raise ValueError("ranker emitted illegal action")
    return selected, work


def episode(case, mode, model, seed, learned_stop=None, budget_mode="strict"):
    if mode not in ARMS or model["seed"] != seed:
        raise ValueError("unregistered AF33 evaluation arm/checkpoint seed")
    public = gym.public_features(case)
    graph = symbolic.PublicHorn.from_public_features(public)
    state = graph.initial
    host = gym.initial_facts(case)
    d = gym.shortest_steps(case)
    if budget_mode not in BUDGET_MODES:
        raise ValueError("unregistered AF33 source-only budget")
    budget = (d if budget_mode=="strict" else d+1) if d is not None else 2
    history = []
    certificate = None
    certificate_work = {"closure_passes":0,"rule_examinations":0}
    start_ns = time.perf_counter_ns()
    if mode in CERT_ARMS:
        certificate, certificate_work = gym.public_fixed_point(public, state)
    if mode in LEARNED_STOP_ARMS and (
        not isinstance(learned_stop,dict) or mode not in learned_stop or
        learned_stop[mode]["seed"]!=seed):
        raise ValueError("AF33 learned-only STOP model missing or foreign")
    for step in range(budget):
        if graph.goal in state:
            break
        now = time.perf_counter_ns()
        if mode in CERT_ARMS:
            if graph.goal not in certificate:
                # Public exact fixed-point STOP; cost charged once per episode.
                action, work = gym.STOP, {
                    "feature_rule_examinations":0,
                    "candidate_feature_rows":0, "linear_scores_computed":0
                }
            else:
                action, work = _ranked_action(public,state,mode,model,seed,step)
        elif mode in LEARNED_STOP_ARMS:
            stop_model=learned_stop[mode]
            stopping,p_stop,stop_work=stop.choose(stop_model,public,state,model)
            if stopping:
                action=gym.STOP
                work={"feature_rule_examinations":stop_work["feature_rule_examinations"],
                      "candidate_feature_rows":stop_work["candidate_feature_rows"],
                      "linear_scores_computed":stop_work["candidate_feature_rows"]}
            else:
                action,work=_ranked_action(public,state,"RANK_CERT",model,seed,step)
                work["feature_rule_examinations"]+=stop_work["feature_rule_examinations"]
                work["candidate_feature_rows"]+=stop_work["candidate_feature_rows"]
                work["linear_scores_computed"]+=stop_work["candidate_feature_rows"]
            work["learned_STOP_probability"]=p_stop
            work["learned_STOP_threshold"]=stop_model["dev_threshold"]
            work["learned_STOP_variant"]=stop_model["variant"]
        else:
            result = symbolic.choose(graph,state,SYM_MAP[mode],
                                     seed,gym.digest(public),step)
            action = gym.STOP if result["action"] is None else result["action"]
            work = {
                "feature_rule_examinations":0,
                "candidate_feature_rows":0,
                "linear_scores_computed":0,
                "symbolic_rule_examinations":result["rule_examinations"],
                "symbolic_state_expansions":result["expanded"],
                "symbolic_closure_passes":result["closure_passes"],
            }
        enabled = tuple(slot for slot, _ in graph.enabled(state))
        if action != gym.STOP and action not in enabled:
            raise ValueError("controller proposed public-illegal rule")
        if tuple(enabled) != gym.enabled(case,host):
            raise AssertionError("source host/public action mask mismatch")
        if action == gym.STOP:
            next_host, proof = host, None
            next_state = state
        else:
            next_host, proof = gym.sound_fire(case,host,action)
            next_state = graph.fire(state,action)
            if next_state != gym.to_physical(case,next_host):
                raise AssertionError("host/public step transition mismatch")
        history.append({
            "step":step,"before_physical":sorted(state),
            "enabled_physical":list(enabled),"action":action,
            "after_physical":sorted(next_state),
            "proof":proof, "work":work,
            "step_wall_ns":time.perf_counter_ns()-now,
        })
        host, state = next_host, next_state
        if action == gym.STOP:
            break
    solved = graph.goal in state
    if solved != (gym.GOAL in host):
        raise AssertionError("physical/host GOAL mismatch")
    root_off_optimal = bool(
        case.reachable and history and (
            history[0]["action"]==gym.STOP or
            gym.public_bitset_bfs(public, history[0]["after_physical"])!=d-1
        )
    )
    return {
        "experiment":"AF-33","seed":seed,
        "budget_mode":budget_mode,
        "root_first_off_optimal":root_off_optimal,
        "split":case.split,"family":case.family,
        "case_id":case.id,"pair_id":case.pair_id,
        "swap_position":case.swap_position,
        "cycle_length":case.cycle_length,
        "designed_reachable":case.reachable,
        "public_graph_sha256":gym.digest(public),
        "arm":mode,"shortest_steps_oracle":d,
        "budget_oracle":budget,
        "solved":solved,
        "initial_stop":bool(history and history[0]["action"]==gym.STOP),
        "false_accepted_proof":bool(not case.reachable and solved),
        "certificate_work":certificate_work,
        "certificate_reachable":(graph.goal in certificate
                                  if certificate is not None else None),
        "history":history,
        "total_rule_examinations":(
            certificate_work["rule_examinations"] +
            sum(h["work"].get("feature_rule_examinations",0) +
                h["work"].get("symbolic_rule_examinations",0)
                for h in history)
        ),
        "total_symbolic_expansions":sum(
            h["work"].get("symbolic_state_expansions",0) for h in history),
        "total_linear_scores":sum(
            h["work"].get("linear_scores_computed",0) for h in history),
        "wall_ns":time.perf_counter_ns()-start_ns,
    }


def summarize(rows):
    pos = [r for r in rows if r["designed_reachable"]]
    neg = [r for r in rows if not r["designed_reachable"]]
    if (len(pos),len(neg)) != (96,96):
        raise ValueError("AF33 heldout label counts changed")
    return {
        "positive_proofs":sum(r["solved"] for r in pos),
        "negative_correct_initial_stop":sum(r["initial_stop"] for r in neg),
        "positive_premature_initial_stop":sum(r["initial_stop"] for r in pos),
        "false_accepted_proofs":sum(r["false_accepted_proof"] for r in rows),
        "positive_root_first_off_optimal":sum(r["root_first_off_optimal"] for r in pos),
        "rule_examinations":sum(r["total_rule_examinations"] for r in rows),
        "symbolic_expansions":sum(r["total_symbolic_expansions"] for r in rows),
        "linear_scores":sum(r["total_linear_scores"] for r in rows),
        "wall_ns":sum(r["wall_ns"] for r in rows),
        "by_cycle_length":{
            str(length):{
                "reachable_cases":sum(r["cycle_length"]==length for r in pos),
                "completed_proofs":sum(r["solved"] and r["cycle_length"]==length for r in pos),
                "correct_negative_stop":sum(r["initial_stop"] and r["cycle_length"]==length for r in neg),
            } for length in (2,3)
        },
        "by_swap_position":{
            str(pos):{
                "reachable_cases":sum(r["swap_position"]==pos for r in
                                      [r for r in rows if r["designed_reachable"]]),
                "completed_proofs":sum(r["solved"] and r["swap_position"]==pos
                                        for r in rows if r["designed_reachable"]),
                "correct_negative_stop":sum(r["initial_stop"] and
                                              r["swap_position"]==pos for r in neg),
            } for pos in (1,2,3)
        },
    }


def gate(scores):
    results = {}
    for split in SPLITS:
        s = scores[cohort_key("strict",split)]
        base = max(s[b]["positive_proofs"] for b in
                   ("UNTRAINED_CERT","PATH_CERT","ANCESTRY_CERT",
                    "FIXED_CERT","RANDOM_CERT"))
        target = s["RANK_CERT"]
        results[split] = {
            "learned_completed":target["positive_proofs"],
            "strongest_certificate_control_completed":base,
            "completion_gain":target["positive_proofs"]-base,
            "negative_STOP":target["negative_correct_initial_stop"],
            "no_false_accepted_proofs":target["false_accepted_proofs"]==0,
            "pass":(
                target["positive_proofs"]>=60 and
                target["positive_proofs"]-base>=10 and
                target["negative_correct_initial_stop"]>=80 and
                target["false_accepted_proofs"]==0
            ),
        }
    return {"per_depth":results,
            "both_depths_pass":all(s["pass"] for s in results.values())}


def balanced_stop_gate(scores):
    """Preregistered balanced calibration primary comparative gate, strict only."""
    per_depth={}
    for split in SPLITS:
        row=scores[cohort_key("strict",split)]
        bal,j12=row["BAL_STOP"],row["J12_STOP"]
        gain=bal["negative_correct_initial_stop"]-j12["negative_correct_initial_stop"]
        per_depth[split]={
            "bal_completed_proofs":bal["positive_proofs"],
            "bal_correct_negative_STOP":bal["negative_correct_initial_stop"],
            "j12_correct_negative_STOP":j12["negative_correct_initial_stop"],
            "bal_minus_j12_negative_STOP":gain,
            "bal_positive_false_initial_STOP":bal["positive_premature_initial_stop"],
            "false_accepted_proofs":bal["false_accepted_proofs"],
            "pass":bool(
                bal["positive_proofs"]>=60 and
                bal["negative_correct_initial_stop"]>=60 and
                gain>=10 and
                bal["positive_premature_initial_stop"]<=10 and
                bal["false_accepted_proofs"]==0
            ),
        }
    return {"per_depth":per_depth,
            "both_depths_pass":all(x["pass"] for x in per_depth.values())}


def absolute_bal_stop_gate(scores):
    """Registered independent absolute learned STOP endpoint."""
    per_depth={}
    for split in SPLITS:
        bal=scores[cohort_key("strict",split)]["BAL_STOP"]
        per_depth[split]={
            "bal_completed_proofs":bal["positive_proofs"],
            "bal_correct_negative_STOP":bal["negative_correct_initial_stop"],
            "bal_positive_false_initial_STOP":bal["positive_premature_initial_stop"],
            "false_accepted_proofs":bal["false_accepted_proofs"],
            "pass":bool(
                bal["positive_proofs"]>=60 and
                bal["negative_correct_initial_stop"]>=60 and
                bal["positive_premature_initial_stop"]<=10 and
                bal["false_accepted_proofs"]==0
            ),
        }
    return {"per_depth":per_depth,
            "both_depths_pass":all(x["pass"] for x in per_depth.values())}


def evaluate(seed, output):
    if seed not in ranker.SEEDS:
        raise ValueError("unregistered AF33 evaluation seed")
    output = Path(output)
    if output.exists():
        raise FileExistsError("source-locked AF33 output directory already exists")
    output.mkdir(parents=True)
    manifest = gym.manifest()
    started = time.perf_counter()
    model = ranker.fit(seed)
    model_raw = (json.dumps(model,sort_keys=True,allow_nan=False)+"\n").encode()
    (output/"model.json").write_bytes(model_raw)
    learned_stop={
        "BAL_STOP":stop.fit(seed,model,variant="BAL"),
        "J12_STOP":stop.fit(seed,model,variant="J12"),
    }
    stop_raw={
        arm:(json.dumps(sub,sort_keys=True,allow_nan=False)+"\n").encode()
        for arm,sub in learned_stop.items()
    }
    for arm,data in stop_raw.items():
        (output/("stop_"+arm.lower()+".json")).write_bytes(data)
    all_rows, score = [], {}
    for budget_mode, split in COHORTS:
        roster = gym.cases_for(split)
        per_arm = {}
        for arm in ARMS:
            rows = [episode(c,arm,model,seed,learned_stop,budget_mode) for c in roster]
            all_rows.extend(rows)
            per_arm[arm] = summarize(rows)
        score[cohort_key(budget_mode,split)]=per_arm
    episodes_raw = ("".join(json.dumps(row,sort_keys=True,allow_nan=False)+"\n"
                            for row in all_rows)).encode()
    (output/"episodes.jsonl").write_bytes(episodes_raw)
    verdict = gate(score)
    receipt = {
        "experiment":"AF-33","seed":seed,
        "manifest_sha256":manifest["manifest_sha256"],
        "protocol_sha256":sha_bytes((
            Path(__file__).parent/"PROTOCOL.json").read_bytes()),
        "model_sha256":sha_bytes(model_raw),
        "stop_models_sha256":{arm:sha_bytes(raw) for arm,raw in stop_raw.items()},
        "stop_train_rows":{arm:m["train_rows"] for arm,m in learned_stop.items()},
        "stop_dev_thresholds":{arm:m["dev_threshold"] for arm,m in learned_stop.items()},
        "episodes_sha256":sha_bytes(episodes_raw),
        "episodes":len(all_rows),"all_arms":list(ARMS),
        "budget_modes":list(BUDGET_MODES),
        "cohort_keys":[cohort_key(b,s) for b,s in COHORTS],
        "scores":score,"registered_seed_gate":verdict,
        "balanced_stop_seed_gate":balanced_stop_gate(score),
        "absolute_bal_stop_seed_gate":absolute_bal_stop_gate(score),
        "original_af23_model_optimizer_updates":0,
        "original_af23_checkpoint_loaded":False,
        "elapsed_seconds":time.perf_counter()-started,
        "scientific_verdict":"AWAITING_EIGHT_SEED_AGGREGATION",
    }
    (output/"evaluation.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,required=True,choices=ranker.SEEDS)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    r=evaluate(a.seed,a.out)
    print(json.dumps({"seed":a.seed,"episodes":r["episodes"],
                      "seed_gate":r["registered_seed_gate"]}))


if __name__=="__main__":
    main()
