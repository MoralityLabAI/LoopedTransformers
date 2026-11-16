"""AF-26 independent physical-rule, native-score factorial and Torch audit.

The physical proof oracle is reconstructed from the model-visible incidence,
as in AF-25. The auditor independently rederives Direct and VI-2 neural
scores from stored node values, replays the *discrete* STOP/rank swap, checks
every proof action and budget, and samples original frozen Torch forwards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.activation_funnel_af25 import audit as physical
from experiments.activation_funnel_af25 import gym, transfer_planner
from experiments.activation_funnel_af25 import evaluate as original
from . import factor, evaluate as ev


def verify_episode(c,row):
    rules,initial,goal = physical.public_graph(c)
    tological = physical.physical_to_logical(c)
    distance = physical.independent_case(c)
    if (row["experiment"],row["split"],row["case_id"],row["family"]) != (
            "AF-26",c.split,c.id,c.family):
        raise ValueError("AF26 graph/episode identity mismatch")
    budget=distance+1 if distance is not None else 2
    if (row["goal_truth"] != int(distance is not None) or
            row["optimal_steps"] != distance or row["budget"] != budget or
            len(row["actions"]) > budget):
        raise ValueError("falsified independent Horn truth/shortest budget")
    if row["mode"] not in factor.ARMS:
        raise ValueError("unknown or extra factorial arm")
    translate=lambda physical_atoms:sorted(tological[p] for p in physical_atoms)
    if row["initial"] != translate(initial):
        raise ValueError("incorrect physical initial known facts")
    state=set(initial)
    ended=False
    forward_nodes=0
    action_count=0
    for step in row["actions"]:
        if ended or step["before"] != translate(state):
            raise ValueError("continued after finished/STOP or incorrect state")
        enabled=tuple(sorted(
            slot for slot,(prem,head) in rules.items()
            if head not in state and all(p in state for p in prem)
        ))
        if step["enabled"] != list(enabled):
            raise ValueError("host physical legal rule mask changed")
        if step.get("host_graph_sha256") != gym.digest(gym.public_features(c)):
            raise ValueError("wrong public graph hash")
        trace=step["plan_trace"]
        if trace is None or not isinstance(step.get("original_scores"),dict):
            raise ValueError("missing source-score and true model trace")
        orig_selected=physical.independent_neural_trace(
            c,step["before"],trace,step["original_scores"]
        )
        computed=factor.choose(step["original_scores"],enabled,
                               row["mode"],orig_selected)
        if computed != step["factor"] or step["action"] != computed["chosen"]:
            raise ValueError("AF26 factor action or label was altered")
        action=step["action"]
        if row["mode"] in ("nd","nv") and enabled and action==gym.STOP:
            raise ValueError("forced-continuation policy stopped with legal actions")
        forward_nodes+=trace["nodes_forward_count"]
        if action==gym.STOP:
            if step["proof"] is not None:
                raise ValueError("STOP contains fictitious proof")
            ended=True
        else:
            if action not in enabled:
                raise ValueError("selected unsupported Horn rule")
            prem,head=rules[action]
            evidence=step["proof"]
            if (not isinstance(evidence,dict) or
                evidence.get("rule_slot")!=action or
                evidence.get("head")!=tological[head] or
                sorted(evidence.get("premises",[]))!=sorted(tological[p] for p in prem)):
                raise ValueError("forged physical Horn proof evidence")
            state.add(head)
            action_count+=1
            if goal in state:
                ended=True
        if step["after"] != translate(state):
            raise ValueError("invalid physical Horn post-state")
    solved=goal in state
    if row["final"]!=translate(state) or row["solved"]!=solved:
        raise ValueError("wrong completed proof status")
    if row["false_accepted_proof"] or (solved and distance is None):
        raise ValueError("false accepted unprovable goal")
    stopped=bool(row["actions"] and row["actions"][-1]["action"]==gym.STOP)
    if row["initial_stop"] != bool(row["actions"] and
                                    row["actions"][0]["action"]==gym.STOP):
        raise ValueError("initial STOP counterfactual record mismatch")
    if row["early_stop"] != bool(distance is not None and not solved and stopped):
        raise ValueError("later premature STOP mismatch")
    if row["forward_nodes"] != forward_nodes:
        raise ValueError("neural forward work counter inaccurate")
    return forward_nodes,action_count


def audit(seed,zip_dir,original_file,out_dir,smoke=False,replays_per_split=2):
    folder=Path(out_dir)
    scores_file=folder/"evaluation.json"
    episode_file=folder/"episodes.jsonl"
    receipt=json.loads(scores_file.read_text())
    official=ev.original_aggregate(original_file)
    if (receipt["experiment"],receipt["seed"],receipt["smoke"]) != (
            "AF-26",seed,smoke):
        raise ValueError("scored seed or smoke identifier differs")
    if (receipt["source_hashes"]!=ev.hashes() or
        receipt["protocol_sha256"]!=ev.sha((ev.HERE/"PROTOCOL.json").read_bytes()) or
        receipt["af25_aggregate_sha256"]!=ev.sha(Path(original_file).read_bytes()) or
        receipt["af25_structural_manifest_sha256"]!=gym.manifest()["manifest_sha256"] or
        receipt["episodes_sha256"]!=ev.sha(episode_file.read_bytes()) or
        receipt["training_updates"]!=0 or
        receipt["all_arms"]!=list(factor.ARMS)):
        raise ValueError("scoring and source registration checksum mismatch")
    rows=[json.loads(line) for line in episode_file.read_text().splitlines()]
    expected=(4 if smoke else 192)*len(gym.SPLITS)*len(factor.ARMS)
    if len(rows)!=expected or receipt["episode_count"]!=expected:
        raise ValueError("incomplete complete-episode factorial ledger")
    keys={}
    for row in rows:
        key=(row["split"],row["mode"],row["case_id"])
        if key in keys:
            raise ValueError("duplicated factorial proof episode")
        keys[key]=row
    count_nodes=count_rules=0
    for split in gym.SPLITS:
        cases=gym.cases_for(split)
        if smoke:
            cases=([c for c in cases if c.designed_positive][:2]+
                   [c for c in cases if not c.designed_positive][:2])
        for c in cases:
            for arm in factor.ARMS:
                row=keys.get((split,arm,c.id))
                if row is None:
                    raise ValueError("missing registered factorial case")
                n,r=verify_episode(c,row)
                count_nodes+=n
                count_rules+=r
        for arm in factor.ARMS:
            original_rows=[keys[(split,arm,c.id)] for c in cases]
            summary=ev.summarize(original_rows)
            for key in ("cases","provable","unprovable","solved",
                        "first_stop_positive","later_premature_stop",
                        "early_stop_positive","first_stop_negative",
                        "false_accepted_proofs","neural_forward_nodes",
                        "rule_actions"):
                if summary[key] != receipt["scores"][split][arm][key]:
                    raise ValueError("tampered factorial aggregate metric "+key)
    if not smoke:
        ev.check_original_parity(receipt["scores"],official,seed)
    model,provenance=original.load_frozen_original(zip_dir,seed)
    if provenance!=receipt["original_af23_checkpoint"]:
        raise ValueError("original AF23 training identity mismatch")
    replayed=0
    for split in gym.SPLITS:
        cases=gym.cases_for(split)
        if smoke:
            cases=([c for c in cases if c.designed_positive][:2]+
                   [c for c in cases if not c.designed_positive][:2])
        for c in cases[:replays_per_split]:
            for arm in factor.ARMS:
                row=keys[(split,arm,c.id)]
                if not row["actions"]:
                    continue
                saved=row["actions"][0]
                plan=transfer_planner.decisions(
                    model,[c],[frozenset(saved["before"])]
                )[0]
                for mode in ("direct","vi2"):
                    a,b=plan["scores"][mode],saved["original_scores"][mode]
                    if len(a)!=len(b) or any(abs(x-y)>1e-5 for x,y in zip(a,b)):
                        raise ValueError("frozen model Torch forward score replay differs")
                if factor.choose(plan["scores"],
                    tuple(saved["enabled"]),arm,plan["selected"])["chosen"] != saved["action"]:
                    raise ValueError("source-bound Torch action replay mismatch")
                replayed+=1
    result={
        "experiment":"AF-26","seed":seed,"smoke":smoke,"verified":True,
        "episodes_audited":len(rows),
        "planner_nodes_reconstructed":count_nodes,
        "physical_rule_firings_audited":count_rules,
        "torch_forward_replays":replayed,
        "false_accepted_proofs":0,
        "evaluation_sha256":ev.sha(scores_file.read_bytes()),
        "episodes_sha256":ev.sha(episode_file.read_bytes()),
        "source_hashes":ev.hashes(),
        "original_af23_checkpoint":provenance,
        "original_af25_aggregate_sha256":ev.PARENT_AGG_SHA,
        "original_af25_direct_vi2_parity":not smoke,
    }
    (folder/"independent_audit.json").write_text(
        json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
    )
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",required=True,type=int,choices=gym.SEEDS)
    p.add_argument("--original-zip-dir",required=True,type=Path)
    p.add_argument("--af25-aggregate",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    x=audit(a.seed,a.original_zip_dir,a.af25_aggregate,a.output_dir,a.smoke)
    print(json.dumps({k:x[k] for k in (
        "seed","verified","episodes_audited",
        "torch_forward_replays","original_af25_direct_vi2_parity"
    )}))


if __name__=="__main__":
    main()
