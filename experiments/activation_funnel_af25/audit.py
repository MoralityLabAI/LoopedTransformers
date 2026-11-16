"""AF-25 independent physical-Horn proof and numeric Bellman audit.

The auditor rebuilds Horn incidence from the exact model-visible tensors,
uses independent bitset BFS, replays every selected action and the numeric
VI-1/VI-2 minima, and samples genuine original Torch checkpoint forwards.
An aggregate CI success is not proof that a trained model transferred.
"""
from __future__ import annotations

import argparse
from collections import deque
from functools import lru_cache
import json
import math
from pathlib import Path

import torch

from . import gym, evaluate as ex, transfer_planner
from experiments.activation_funnel_af24 import symbolic


@lru_cache(maxsize=512)
def public_graph(c):
    """Physical graph reconstructed solely from model-visible incidence."""
    encoded = gym.encode(c)
    pre = encoded.prem_up + encoded.prem_down
    head = encoded.head_up + encoded.head_down
    rules = {}
    for slot in range(gym.RULES):
        if encoded.rule_features[slot, 2] < .5:
            continue
        premise = tuple(torch.where(pre[slot] > .5)[0].tolist())
        target = tuple(torch.where(head[slot] > .5)[0].tolist())
        if not premise or len(premise) > 2 or len(target) != 1:
            raise ValueError("corrupt model-visible Horn rule")
        rules[slot] = (premise, target[0])
    initial = frozenset(torch.where(encoded.atom_features[:,0] > .5)[0].tolist())
    return rules, initial, int(encoded.read_slots[2])


def shortest_public(rules, initial, goal):
    start = sum(1 << x for x in initial)
    if start & (1 << goal):
        return 0
    mask = [(sum(1 << a for a in p), 1 << h)
            for p,h in rules.values()]
    q = deque([(start,0)])
    seen = {start}
    while q:
        state, depth = q.popleft()
        for pre, head in mask:
            if pre & state != pre or head & state:
                continue
            after = state | head
            if after & (1 << goal):
                return depth + 1
            if after not in seen:
                seen.add(after)
                q.append((after,depth+1))
    return None


def physical_to_logical(c):
    apos,_ = gym.physical_maps(c)
    return {physical:logical for logical,physical in enumerate(apos)}


@lru_cache(maxsize=512)
def independent_case(c):
    rules, initial, goal = public_graph(c)
    d = shortest_public(rules, initial, goal)
    if d != gym.independent_bitset_bfs(c) or d != gym.shortest_steps(c):
        raise ValueError("public and independently encoded shortest proof differ")
    return d


def _softplus(x):
    return max(x,0.) + math.log1p(math.exp(-abs(x)))


def _sigmoid(x):
    if x >= 0:
        return 1. / (1. + math.exp(-x))
    z = math.exp(x)
    return z / (1.+z)


def independent_neural_trace(c, root_state, trace, observed):
    """Recompute the complete two-step public ledger without model/label oracle."""
    start = frozenset(root_state)
    rules, _, goal_phys = public_graph(c)
    logical_to_phys,_ = gym.physical_maps(c)
    ptol = physical_to_logical(c)
    goal = ptol[goal_phys]
    if goal != gym.GOAL:
        raise ValueError("wrong public goal role")
    q = deque([(start,0)])
    levels = {start:0}
    expected = {}
    while q:
        logical, depth = q.popleft()
        expected[logical] = {}
        phys = frozenset(logical_to_phys[x] for x in logical)
        if gym.GOAL in logical or depth == 2:
            continue
        for slot, (pre, head) in sorted(rules.items()):
            if head in phys or any(p not in phys for p in pre):
                continue
            future = logical | {ptol[head]}
            expected[logical][str(slot)] = sorted(future)
            if future not in levels:
                levels[future] = depth+1
                q.append((future,depth+1))
    saved = {frozenset(n["state"]):n for n in trace["nodes"]}
    if set(saved) != set(expected) or trace["nodes_forward_count"] != len(expected):
        raise ValueError("missing/extra neural host successor state")
    value = {}
    stop = {}
    for s, raw in saved.items():
        if raw["edges"] != expected[s]:
            raise ValueError("forged public Horn Bellman transitions")
        p = _sigmoid(raw["reach_logit"])
        finite = min(16., max(0., _softplus(raw["distance_logit"])))
        v = p*finite + (1.-p)*16.
        h = 16.*p
        if gym.GOAL in s:
            v, h = 0., 0.
        if abs(v-raw["value0"]) > 1e-5 or abs(h-raw["stop_cost"]) > 1e-5:
            raise ValueError("incorrect neural leaf or STOP cost")
        value[s], stop[s] = v,h
    backup = {
        s:min([stop[s]] + [1. + value[frozenset(t)]
                          for t in expected[s].values()])
        for s in expected
    }
    scores = {m:[-1e4]*(gym.STOP+1) for m in ex.NEURAL}
    root_edges = expected[start]
    if len(trace["root_raw"]) != gym.STOP + 1:
        raise ValueError("wrong original logit count")
    for name, future in root_edges.items():
        a = int(name)
        succ = frozenset(future)
        scores["direct"][a] = trace["root_raw"][a]
        scores["vi1"][a] = -(1.+value[succ])
        scores["vi2"][a] = -(1.+backup[succ])
    scores["direct"][gym.STOP] = trace["root_raw"][gym.STOP]
    for m in ("vi1","vi2"):
        scores[m][gym.STOP] = -stop[start]
    for m, actual in observed.items():
        if (len(actual) != gym.STOP+1 or
                any(abs(float(x)-float(y)) > 1e-5 for x,y in zip(scores[m],actual))):
            raise ValueError("independent numeric Bellman/logit replay mismatch")
    return {m:(max(range(gym.STOP+1),key=lambda a:scores[m][a])
               if root_edges else gym.STOP) for m in observed}


def audit_episode(c, row):
    rules, initial, goal = public_graph(c)
    ptol = physical_to_logical(c)
    logical_to_phys,_ = gym.physical_maps(c)
    d = independent_case(c)
    if (row["experiment"],row["split"],row["case_id"],row["family"]) != (
        "AF-25",c.split,c.id,c.family):
        raise ValueError("missing/forged original test case")
    if (row["goal_truth"] != int(d is not None) or
            row["optimal_steps"] != d or
            row["budget"] != (d+1 if d is not None else 2)):
        raise ValueError("incorrect independent shortest proof/budget")
    if len(row["actions"]) > row["budget"]:
        raise ValueError("neural controller exceeded proof action budget")
    translate = lambda atoms: sorted(ptol[x] for x in atoms)
    if row["initial"] != translate(initial):
        raise ValueError("incorrect encoded initial known facts")
    state = set(initial)
    finished = False
    forward_nodes = exam = expanded = 0
    for index, step in enumerate(row["actions"]):
        if finished or step["before"] != translate(state):
            raise ValueError("continued after termination or forged physical state")
        legal = sorted(slot for slot, (prem,head) in rules.items()
                       if head not in state and all(p in state for p in prem))
        if step["enabled"] != legal:
            raise ValueError("wrong true public rule mask")
        action = step["action"]
        if row["mode"] in ex.NEURAL:
            if "work" in step or not step.get("plan_trace") or not step.get("decision_scores"):
                raise ValueError("missing original neural action receipt")
            trace = step["plan_trace"]
            chosen = independent_neural_trace(
                c, step["before"], trace,
                {row["mode"]:step["decision_scores"]}
            )
            if chosen[row["mode"]] != action:
                raise ValueError("neural selected action differs from signed logits")
            if step["host_graph_sha256"] != gym.digest(gym.public_features(c)):
                raise ValueError("wrong source public graph")
            forward_nodes += trace["nodes_forward_count"]
        else:
            if step["plan_trace"] is not None or step["decision_scores"] is not None:
                raise ValueError("symbolic policy used private neural trace")
            work = step["work"]
            if row["mode"] == "shortest":
                current = frozenset(ptol[x] for x in state)
                selected = gym.optimal_actions(c,current)[0]
                expected = gym.STOP if selected == gym.STOP else selected
            else:
                graph = symbolic.PublicHorn(
                    rules=tuple(sorted((slot,tuple(pre),head) for slot,(pre,head) in rules.items())),
                    initial=frozenset(state), goal=goal,
                )
                scored = symbolic.choose(graph,frozenset(state),
                    row["mode"],17,c.id,index)
                expected = gym.STOP if scored["action"] is None else scored["action"]
                for key in ("expanded","rule_examinations",
                            "closure_passes","search_exhausted"):
                    if work[key] != scored[key]:
                        raise ValueError("symbolic controller metered work receipt changed")
            if expected != action:
                raise ValueError("symbolic action replay mismatch")
            exam += work["rule_examinations"]
            expanded += work["expanded"]
        if action == gym.STOP:
            if step["proof"] is not None:
                raise ValueError("STOP carries forged proof")
            finished = True
        else:
            if action not in legal:
                raise ValueError("illegal physical rule")
            prem, head = rules[action]
            evidence = step["proof"]
            if (not isinstance(evidence,dict) or
                    evidence.get("rule_slot") != action or
                    evidence.get("head") != ptol[head] or
                    sorted(evidence.get("premises",[])) != sorted(ptol[p] for p in prem)):
                raise ValueError("forged Horn premises/head proof")
            state.add(head)
            if goal in state:
                finished = True
        if step["after"] != translate(state):
            raise ValueError("incorrect rule-firing poststate")
    solved = goal in state
    if row["final"] != translate(state) or row["solved"] != solved:
        raise ValueError("false scored goal or changed output state")
    if row["false_accepted_proof"] or (solved and d is None):
        raise ValueError("false accepted proof")
    is_stop = bool(row["actions"] and row["actions"][-1]["action"]==gym.STOP)
    if row["early_stop"] != bool(d is not None and not solved and is_stop):
        raise ValueError("incorrect early STOP accounting")
    if row["initial_stop"] != bool(row["actions"] and row["actions"][0]["action"]==gym.STOP):
        raise ValueError("incorrect first STOP accounting")
    if (forward_nodes != row["forward_nodes"] or exam != row["rule_examinations"]
            or expanded != row["bfs_expansions"]):
        raise ValueError("incorrect inference work accounting")
    return forward_nodes


def audit(seed,original_zip_dir,output_dir,smoke=False,replays_per_mode=2):
    folder = Path(output_dir)
    receipt_path = folder/"evaluation.json"
    rows_path = folder/"episodes.jsonl"
    receipt = json.loads(receipt_path.read_text())
    if (receipt["experiment"],receipt["seed"],receipt["smoke"]) != (
        "AF-25",seed,smoke):
        raise ValueError("wrong AF25 scored source identity")
    if (receipt["registered_manifest_sha256"] != gym.manifest()["manifest_sha256"] or
            receipt["registered_protocol_sha256"] != ex.sha256(
                (ex.HERE/"PROTOCOL.json").read_bytes()) or
            receipt["source_hashes"] != ex.hashes() or
            receipt["episodes_sha256"] != ex.sha256(rows_path.read_bytes()) or
            receipt["all_arms"] != list(ex.ALL) or
            receipt["training_updates"] != 0):
        raise ValueError("source-locked result checksum mismatch")
    rows = [json.loads(x) for x in rows_path.read_text().splitlines()]
    index = {}
    for row in rows:
        key = (row["split"],row["mode"],row["case_id"])
        if key in index:
            raise ValueError("duplicate saved proof episode")
        index[key] = row
    expected = sum((4 if smoke else 192) for _ in ex.SPLITS)*len(ex.ALL)
    if len(rows) != expected or receipt["episode_count"] != expected:
        raise ValueError("scored episode count incomplete")
    total_nodes = 0
    for split in ex.SPLITS:
        cases = gym.cases_for(split)
        if smoke:
            cases = [c for c in cases if c.designed_positive][:2] + [
                c for c in cases if not c.designed_positive][:2]
        for c in cases:
            for mode in ex.ALL:
                row = index.get((split,mode,c.id))
                if row is None:
                    raise ValueError("missing scored case/arm")
                total_nodes += audit_episode(c,row)
        for mode in ex.ALL:
            group=[index[(split,mode,c.id)] for c in cases]
            recomputed=ex.summarize(group)
            for key in ("cases","provable","unprovable","solved",
                        "premature_stop","first_premature_stop",
                        "correct_initial_stop","false_accepted_proofs",
                        "neural_forward_nodes","symbolic_rule_examinations",
                        "symbolic_bfs_expansions"):
                if recomputed[key] != receipt["scores"][split][mode][key]:
                    raise ValueError("saved aggregate per-arm totals differ from episodes")
    model, provenance = ex.load_frozen_original(original_zip_dir,seed)
    if provenance != receipt["original_af23_checkpoint"]:
        raise ValueError("original immutable AF23 model provenance mismatch")
    replayed = 0
    for split in ex.SPLITS:
        cases=gym.cases_for(split)
        if smoke:
            cases=[c for c in cases if c.designed_positive][:2] + [
                c for c in cases if not c.designed_positive][:2]
        for c in cases[:replays_per_mode]:
            for mode in ex.NEURAL:
                row=index[(split,mode,c.id)]
                if not row["actions"]:
                    continue
                saved=row["actions"][0]
                fresh=transfer_planner.decisions(
                    model,[c],[frozenset(saved["before"])]
                )[0]
                if (max(abs(x-y) for x,y in zip(
                        fresh["scores"][mode],saved["decision_scores"]))>1e-5
                        or fresh["selected"][mode] != saved["action"]):
                    raise ValueError("frozen source checkpoint Torch forward replay failed")
                replayed += 1
    result = {
        "experiment":"AF-25","seed":seed,"smoke":smoke,"verified":True,
        "episodes_audited":len(rows),
        "planner_nodes_reconstructed":total_nodes,
        "original_checkpoint_replays":replayed,
        "false_accepted_proofs":0,
        "evaluation_sha256":ex.sha256(receipt_path.read_bytes()),
        "episodes_sha256":ex.sha256(rows_path.read_bytes()),
        "source_hashes":ex.hashes(),
        "original_af23_checkpoint":provenance,
        "registered_manifest_sha256":gym.manifest()["manifest_sha256"],
    }
    (folder/"independent_audit.json").write_text(
        json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
    )
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,choices=gym.SEEDS,required=True)
    p.add_argument("--original-zip-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    result=audit(a.seed,a.original_zip_dir,a.output_dir,a.smoke)
    print(json.dumps({"seed":a.seed,"verified":result["verified"],
                      "episodes":result["episodes_audited"],
                      "forward_replays":result["original_checkpoint_replays"]}))


if __name__=="__main__":
    main()
