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
import zipfile

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
# Immutable original AF-25 per-seed proof-episode archive ZIP digests.
# Each comes from the original successful hosted Actions run #43829157457.
AF25_EPISODE_ZIPS = {
    17:"20e5698c30be9b23648ba17a5cae5238952e1ddeed5af0cb74b1a4a426ae24db",
    29:"28d82740b0c9b72d304fb236a4ca71919d839dc1deb2a1e787e489f93b282618",
    43:"4f99518f16bc31d2f38cab26a1f05a70e682c711d40da0c9e9f1ee09f4880e2a",
    61:"14c760aaf3e6d9c9267aca1ca72c88021c07545607b5c3dab7c75433f4227740",
    73:"2173553e03aac7dd657c3ad7dcb1ab744d3c98785c0adedd3cbce3469d45542a",
    89:"a33ad7dfba493e22d965d36f08d4418601bb66c3b0bb7a8cf4758c83451af792",
    103:"b07f2c4156ed3c626dadf188e7566c597a04f7957e771f8b8467674c81faeb27",
    131:"4c9411ad6cb9eea47ee1e5ae439ecfb3efb4e134bcef05aeced75081eebbe5e0",
}


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



def original_seed_episodes(zip_dir, seed, official):
    """Exact AF-25 original DD/VV controls from immutable audited proof traces.

    The historical source control is validated against two independent
    receipts: GitHub's SHA-pinned ZIP, and the separately SHA-pinned eight-seed
    aggregate. This preserves original numerical batch tie-breaking. Crossed
    arms always generate *fresh* closed-loop counterfactual states.
    """
    if seed not in SEEDS:
        raise ValueError("unregistered original AF25 seed")
    path=Path(zip_dir)/f"af25-seed-{seed}.zip"
    if not path.is_file():
        raise FileNotFoundError(f"original AF25 source episode ZIP absent for {seed}")
    payload=path.read_bytes()
    expected=AF25_EPISODE_ZIPS[seed]
    if sha(payload)!=expected:
        raise ValueError("original AF25 archived per-seed ZIP SHA-256 mismatch")
    with zipfile.ZipFile(path) as archive:
        required={"evaluation.json","episodes.jsonl","independent_audit.json"}
        if not required <= set(archive.namelist()):
            raise ValueError("original AF25 proof/audit artifact members missing")
        ev_bytes=archive.read("evaluation.json")
        episode_bytes=archive.read("episodes.jsonl")
        audit_bytes=archive.read("independent_audit.json")
    ev=json.loads(ev_bytes)
    independent=json.loads(audit_bytes)
    if (ev["experiment"]!="AF-25" or ev["seed"]!=seed or ev["smoke"]
            or ev["source_hashes"]!=original.hashes()
            or ev["registered_manifest_sha256"]!=
                official["structural_manifest_sha256"]
            or ev["registered_protocol_sha256"]!=
                official["registered_protocol_sha256"]
            or ev["episodes_sha256"]!=sha(episode_bytes)
            or ev["episode_count"]!=192*2*len(original.ALL)
            or independent["experiment"]!="AF-25"
            or independent["seed"]!=seed or not independent["verified"]
            or independent["false_accepted_proofs"]!=0
            or independent["source_hashes"]!=original.hashes()
            or independent["evaluation_sha256"]!=sha(ev_bytes)
            or independent["episodes_sha256"]!=sha(episode_bytes)
            or independent["original_af23_checkpoint"]!=
               ev["original_af23_checkpoint"]):
        raise ValueError("AF25 original per-seed proof source or audit invalid")
    before=official["per_seed"][str(seed)]
    if (ev["original_af23_checkpoint"]["zip_sha256"]!=
            before["original_zip_sha256"] or
            ev["original_af23_checkpoint"]["training_sha256"]!=
            before["original_training_sha256"]):
        raise ValueError("AF25 source episode weights differ from official aggregate")
    for split in SPLITS:
        for mode in ("direct","vi2"):
            if ev["scores"][split][mode]["solved"]!=before["depths"][split][
               "direct_solved" if mode=="direct" else "vi2_solved"]:
                raise ValueError("original AF25 per-seed control score differs")
    rows=[json.loads(line) for line in episode_bytes.splitlines()]
    if len(rows)!=ev["episode_count"]:
        raise ValueError("AF25 original per-seed episode count incomplete")
    index={}
    for row in rows:
        key=(row["split"],row["mode"],row["case_id"])
        if key in index:
            raise ValueError("duplicated original AF25 proof episode")
        index[key]=row
    for split in SPLITS:
        for mode in ("direct","vi2"):
            for case in gym.cases_for(split):
                if (split,mode,case.id) not in index:
                    raise ValueError("missing exact AF25 reference control episode")
    return index,{
        "source":"independent original AF25 hosted complete proof records",
        "artifact_run":43829157457,
        "artifact_zip_sha256":expected,
        "evaluation_sha256":sha(ev_bytes),
        "episode_member_sha256":sha(episode_bytes),
        "independent_audit_sha256":sha(audit_bytes),
        "source_model_sha256":ev["original_af23_checkpoint"]["zip_sha256"],
        "source_original_science":"NEURAL_ADVANTAGE_NOT_SHOWN",
    }

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


def _source_plan_from_ledger(step, original_mode):
    """Rebuild all native scores from a *freshly computed* original-source trace.

    AF-25's Direct/VI-1/VI-2 mode batching is retained exactly for DD and VV.
    The full saved per-state model outputs suffice to recompute both source
    score vectors without touching oracle labels or changing model weights.
    """
    from experiments.activation_funnel_af23 import branching_planner as scorer
    trace=step["plan_trace"]
    if not trace or not trace.get("nodes"):
        raise ValueError("original policy trace has no neural node ledger")
    root=frozenset(step["before"])
    edges={}
    neural={}
    for node in trace["nodes"]:
        state=frozenset(node["state"])
        edges[state]={int(a):frozenset(target)
                      for a,target in node["edges"].items()}
        neural[state]=(trace["root_raw"] if state==root else
                        [0.0]*(gym.STOP+1),
                       float(node["distance_logit"]),
                       float(node["reach_logit"]))
    if root not in edges:
        raise ValueError("original root absent from source trace")
    scores,selected,_=scorer.score_from_nodes(root,edges,neural)
    saved=step["decision_scores"]
    if (len(scores[original_mode])!=len(saved) or
        any(abs(a-b)>1e-5 for a,b in zip(scores[original_mode],saved))):
        raise ValueError("original AF25 on-policy numerical scores differ >1e-5")
    # Original archived float score vectors and selected actions are the
    # authoritative CPU-batched source baseline. Recompute the *other* source
    # from the same frozen node ledger, while retaining original source-native
    # score values so near-tie batch effects cannot silently alter DD/VV.
    original_choice=max(range(len(saved)),key=lambda a:saved[a])
    if original_choice!=step["action"]:
        raise ValueError("original signed source action contradicts native logits")
    scores[original_mode]=saved
    selected[original_mode]=original_choice
    return scores,selected


def _original_as_factor(c,old_row,arm,original_mode):
    """Fresh, source-identical DD/VV episode converted to the six-arm receipt."""
    if old_row["mode"]!=original_mode or old_row["case_id"]!=c.id:
        raise ValueError("AF25 original source episode identity changed")
    hist=[]
    for step in old_row["actions"]:
        scores,source_selected=_source_plan_from_ledger(step,original_mode)
        selected=factor.choose(scores,tuple(step["enabled"]),arm,source_selected)
        if selected["chosen"]!=step["action"]:
            raise ValueError("source original and factor native action disagree")
        hist.append({
            "before":step["before"],"enabled":step["enabled"],
            "action":step["action"],"after":step["after"],
            "proof":step["proof"],
            "original_scores":{
                "direct":scores["direct"],"vi2":scores["vi2"],
            },
            "factor":selected,
            "plan_trace":step["plan_trace"],
            "host_graph_sha256":gym.digest(gym.public_features(c)),
        })
    return make_row(c,arm,old_row["optimal_steps"],old_row["budget"],
                    hist,frozenset(old_row["final"]))


def rollout(model, cases, original_index):
    """Archived, source-verified original controls plus new mixed trajectories.

    DD/VV use the original audited AF25 host episodes and native score vectors
    so their exact source-batch tie breaking remains intact. These are SOURCE
    CONTROLS, not newly collected model trajectories. The four mixed-factor
    interventions make independent fresh model forwards at each chosen state.
    """
    cases=tuple(cases)
    if not cases:
        raise ValueError("nonempty AF26 scoring cases required")
    original_rows={
        mode:[original_index[(c.split,mode,c.id)] for c in cases]
        for mode in ("direct","vi2")
    }
    distances=[gym.shortest_steps(c) for c in cases]
    budgets=[d+1 if d is not None else 2 for d in distances]
    mixed=("dv","vd","nd","nv")
    states={arm:[gym.initial_facts(c) for c in cases] for arm in mixed}
    done={arm:[False]*len(cases) for arm in mixed}
    history={arm:[[] for _ in cases] for arm in mixed}
    for t in range(max(budgets)):
        jobs=[(arm,i,c,states[arm][i])
              for arm in mixed for i,c in enumerate(cases)
              if not done[arm][i] and t < budgets[i]]
        plans=[]
        for start in range(0,len(jobs),32):
            part=jobs[start:start+32]
            plans.extend(transfer_planner.decisions(
                model,[c for _,_,c,_ in part],
                [s for _,_,_,s in part]
            ))
        if len(plans)!=len(jobs):
            raise ValueError("counterfactual model forward count differs from jobs")
        for (arm,i,c,known),plan in zip(jobs,plans):
            legal=gym.enabled(c,known)
            selected=factor.choose(plan["scores"],legal,arm,plan["selected"])
            action=selected["chosen"]
            after,proof=gym.sound_fire(c,known,action)
            if action!=gym.STOP and proof is None:
                raise ValueError("counterfactual model fired illegal Horn rule")
            history[arm][i].append({
                "before":sorted(known),"enabled":list(legal),
                "action":action,"after":sorted(after),"proof":proof,
                "original_scores":{
                    "direct":plan["scores"]["direct"],
                    "vi2":plan["scores"]["vi2"],
                },
                "factor":selected,"plan_trace":plan["trace"],
                "host_graph_sha256":plan["host_graph_sha256"],
            })
            states[arm][i]=after
            if action==gym.STOP or gym.GOAL in after:
                done[arm][i]=True
    output={
        "dd":[_original_as_factor(c,r,"dd","direct")
              for c,r in zip(cases,original_rows["direct"])],
        "vv":[_original_as_factor(c,r,"vv","vi2")
              for c,r in zip(cases,original_rows["vi2"])],
    }
    output.update({
        arm:[make_row(c,arm,distances[i],budgets[i],
                      history[arm][i],states[arm][i])
             for i,c in enumerate(cases)]
        for arm in mixed
    })
    return {arm:output[arm] for arm in ARMS}


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


def evaluate(seed, original_zips, af25_seed_zips, af25_aggregate, out, smoke=False):
    if seed not in SEEDS:
        raise ValueError("AF26 seed outside frozen registration")
    out=Path(out)
    if out.exists():
        raise FileExistsError("AF26 output must be a fresh directory")
    out.mkdir(parents=True)
    manifest=gym.manifest()
    official=original_aggregate(af25_aggregate)
    original_index, original_receipt=original_seed_episodes(
        af25_seed_zips,seed,official)
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
        branches=rollout(model,cases,original_index)
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
        "original_af25_control_receipt":original_receipt,
        "dd_vv_controls_source":"archived immutable AF25 proof episodes; not counterfactual rerolls",
        "crossed_arms_source":"fresh full closed-loop checkpoint forwards",
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
    p.add_argument("--af25-seed-zip-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    r=evaluate(a.seed,a.original_zip_dir,a.af25_seed_zip_dir,a.af25_aggregate,a.out,a.smoke)
    print(json.dumps({
        "seed":a.seed,"smoke":a.smoke,
        "episodes":r["episode_count"],
        "original_parity":r["replayed_original_policy_controls"],
        "status":r["scientific_verdict"]
    }))


if __name__=="__main__":
    main()
