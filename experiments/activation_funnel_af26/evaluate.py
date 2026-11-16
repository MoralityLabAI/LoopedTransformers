"""AF-26 six frozen-factorial complete proof rollouts with source provenance.

The original AF-23 6036-weight checkpoint is loaded via AF-25's independent
SHA-pinned original-artifact loader. At every newly reached state compute the
unchanged source AF-25 two-step successor graph once and reuse native Direct
and VI-2 scores, separating the STOP gate and enabled rule ranking. Historical
AF-25 episodes are not counterfactual training or policy inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from experiments.activation_funnel_af25 import gym, transfer_planner
from experiments.activation_funnel_af25 import evaluate as original
from . import factor

HERE = Path(__file__).resolve().parent
SEEDS = gym.SEEDS
SPLITS = gym.SPLITS
ARMS = factor.ARMS
SOURCE_FILES = (
    "PROTOCOL.json","factor.py","evaluate.py","audit.py","aggregate.py",
    "tests/test_factor.py","README.md",
    "../../.github/workflows/activation-funnel-af26-preflight.yml",
    "../../.github/workflows/activation-funnel-af26.yml",
)
PARENT_AGG_SHA = "5624d2bfe913c870dada6be00a722eceb8157be26824a875d36d0723eec99163"
PARENT_AGG_ZIP_SHA = "5ca614b06cee3e2e3fdccb64bb842cda9533938d3dfb001fb3c4861effbc1f5c"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hashes():
    return {p: sha((HERE / p).read_bytes()) for p in SOURCE_FILES}


def original_aggregate(path):
    data = Path(path).read_bytes()
    if sha(data) != PARENT_AGG_SHA:
        raise ValueError("official AF25 aggregate member SHA-256 mismatch")
    value = json.loads(data)
    if (value.get("experiment") != "AF-25" or
            value.get("scientific_verdict") != "NEURAL_ADVANTAGE_NOT_SHOWN"
            or value.get("eligible_seeds") != 8 or
            value.get("structural_manifest_sha256") != gym.manifest()["manifest_sha256"]
            or value.get("source_hashes") != original.hashes()):
        raise ValueError("original AF25 source-bound science receipt mismatch")
    for split in SPLITS:
        for arm in ("direct","vi2"):
            if value["pooled"][split][arm]["provable"] != 640:
                raise ValueError("original AF25 held-out positive roster mismatch")
    if (value["per_family"]["held_depth5"]["decoy_and_shared_overcount"]
             ["direct"]["solved"] != 240 or
        value["per_family"]["held_depth5"]["decoy_and_shared_overcount"]
             ["vi2"]["solved"] != 131):
        raise ValueError("AF26 registered 109-proof original hard gap changed")
    return value


def make_row(c, mode, distance, budget, history, final):
    solved = gym.GOAL in final
    stopped = bool(history and history[-1]["action"] == gym.STOP)
    return {
        "experiment":"AF-26","case_id":c.id,"split":c.split,
        "family":c.family,"mode":mode,
        "initial":sorted(gym.initial_facts(c)),"final":sorted(final),
        "goal_truth":int(distance is not None),
        "optimal_steps":distance,"budget":budget,
        "actions":history,"solved":solved,
        "early_stop":bool(distance is not None and not solved and stopped),
        "initial_stop":bool(history and history[0]["action"] == gym.STOP),
        "false_accepted_proof":bool(distance is None and solved),
        "forward_nodes":sum(step["plan_trace"]["nodes_forward_count"]
                            for step in history),
    }


def rollout(model, cases):
    cases = tuple(cases)
    if not cases:
        raise ValueError("nonempty AF26 scoring cases required")
    distances = [gym.shortest_steps(c) for c in cases]
    budgets = [d+1 if d is not None else 2 for d in distances]
    states = {arm:[gym.initial_facts(c) for c in cases] for arm in ARMS}
    done = {arm:[False]*len(cases) for arm in ARMS}
    history = {arm:[[] for _ in cases] for arm in ARMS}
    for t in range(max(budgets)):
        jobs = [(arm,i,c,states[arm][i])
                for arm in ARMS for i,c in enumerate(cases)
                if not done[arm][i] and t < budgets[i]]
        plans = []
        for start in range(0,len(jobs),32):
            part=jobs[start:start+32]
            plans.extend(transfer_planner.decisions(
                model, [c for _,_,c,_ in part],
                [s for _,_,_,s in part]
            ))
        if len(plans)!=len(jobs):
            raise ValueError("source policy forward count differs from host jobs")
        for (arm,i,c,known), plan in zip(jobs,plans):
            legal = gym.enabled(c,known)
            selected = factor.choose(plan["scores"],legal,arm,plan["selected"])
            action = selected["chosen"]
            after,proof = gym.sound_fire(c,known,action)
            if action != gym.STOP and proof is None:
                raise ValueError("model action missing host Horn rule evidence")
            trace = plan["trace"]
            history[arm][i].append({
                "before":sorted(known),"enabled":list(legal),
                "action":action,"after":sorted(after),"proof":proof,
                "original_scores":{
                    "direct":plan["scores"]["direct"],
                    "vi2":plan["scores"]["vi2"],
                },
                "factor":selected,"plan_trace":trace,
                "host_graph_sha256":plan["host_graph_sha256"],
            })
            states[arm][i]=after
            if action==gym.STOP or gym.GOAL in after:
                done[arm][i]=True
    return {
        arm:[
            make_row(c,arm,distances[i],budgets[i],
                     history[arm][i],states[arm][i])
            for i,c in enumerate(cases)
        ]
        for arm in ARMS
    }


def summarize(rows):
    yes=[r for r in rows if r["goal_truth"]]
    no=[r for r in rows if not r["goal_truth"]]
    if not yes or not no:
        raise ValueError("missing positive or negative case controls")
    solved=sum(bool(r["solved"]) for r in yes)
    first_pos=sum(bool(r["initial_stop"]) for r in yes)
    early=sum(bool(r["early_stop"]) for r in yes)
    return {
        "cases":len(rows),"provable":len(yes),"unprovable":len(no),
        "solved":solved,"proof_success":solved/len(yes),
        "first_stop_positive":first_pos,
        "later_premature_stop":sum(
            bool(r["early_stop"] and not r["initial_stop"]) for r in yes
        ),
        "early_stop_positive":early,
        "first_stop_negative":sum(bool(r["initial_stop"]) for r in no),
        "false_accepted_proofs":sum(bool(r["false_accepted_proof"]) for r in rows),
        "neural_forward_nodes":sum(r["forward_nodes"] for r in rows),
        "rule_actions":sum(
            sum(step["action"]!=gym.STOP for step in r["actions"])
            for r in rows
        ),
    }


def check_original_parity(results,original_results,seed):
    original_seed=original_results["per_seed"][str(seed)]
    for split in SPLITS:
        original_depth=original_seed["depths"][split]
        for mode,original_key in (("dd","direct_solved"),("vv","vi2_solved")):
            if results[split][mode]["solved"] != original_depth[original_key]:
                raise ValueError(
                    f"original AF25 {mode} exact completion parity failed"
                    f" at seed={seed} split={split}"
                )
    return True


def evaluate(seed, original_zips, af25_aggregate, out, smoke=False):
    if seed not in SEEDS:
        raise ValueError("AF26 seed outside frozen registration")
    out=Path(out)
    if out.exists():
        raise FileExistsError("AF26 output must be a fresh directory")
    out.mkdir(parents=True)
    manifest=gym.manifest()
    official=original_aggregate(af25_aggregate)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model,provenance=original.load_frozen_original(original_zips,seed)
    if (official["per_seed"][str(seed)]["original_zip_sha256"] !=
            provenance["zip_sha256"] or
            official["per_seed"][str(seed)]["original_training_sha256"] !=
            provenance["training_sha256"]):
        raise ValueError("source checkpoint and prior science receipts disagree")
    scores,all_rows={},[]
    started=time.perf_counter()
    for split in SPLITS:
        cases=gym.cases_for(split)
        if smoke:
            cases=([c for c in cases if c.designed_positive][:2]+
                   [c for c in cases if not c.designed_positive][:2])
        branches=rollout(model,cases)
        scores[split]={arm:summarize(rows) for arm,rows in branches.items()}
        for rows in branches.values():
            all_rows.extend(rows)
    if not smoke:
        check_original_parity(scores,official,seed)
    data=("".join(json.dumps(r,sort_keys=True,allow_nan=False)+"\n"
                  for r in all_rows)).encode()
    (out/"episodes.jsonl").write_bytes(data)
    result={
        "experiment":"AF-26","seed":seed,"smoke":smoke,
        "source_hashes":hashes(),
        "protocol_sha256":sha((HERE/"PROTOCOL.json").read_bytes()),
        "af25_aggregate_sha256":sha(Path(af25_aggregate).read_bytes()),
        "af25_structural_manifest_sha256":manifest["manifest_sha256"],
        "original_af23_checkpoint":provenance,
        "all_arms":list(ARMS),
        "scores":scores,"episode_count":len(all_rows),
        "episodes_sha256":sha(data),
        "training_updates":0,
        "replayed_original_policy_controls":not smoke,
        "elapsed_seconds":time.perf_counter()-started,
        "scientific_verdict":"SMOKE_ONLY" if smoke else "AWAITING_INDEPENDENT_ALL_SEED_AGGREGATE",
    }
    (out/"evaluation.json").write_text(
        json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
    )
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,required=True,choices=SEEDS)
    p.add_argument("--original-zip-dir",type=Path,required=True)
    p.add_argument("--af25-aggregate",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    r=evaluate(a.seed,a.original_zip_dir,a.af25_aggregate,a.out,a.smoke)
    print(json.dumps({
        "seed":a.seed,"smoke":a.smoke,
        "episodes":r["episode_count"],
        "original_parity":r["replayed_original_policy_controls"],
        "status":r["scientific_verdict"]
    }))


if __name__=="__main__":
    main()
