"""AF-27 source-locked frozen policy STOP transfer with rule-gap receipts.

Original AF-23 model is loaded from immutable SHA-pinned source artifacts,
without training. All policies act on newly constructed AF-27 public Horn
graphs; each policy has its own fresh full closed-loop neural decisions.
The logged posthoc root optimal-rule diagnostics use independent shortest
proof costs AFTER all chosen policy actions; no oracle data enter inference.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import torch

from experiments.activation_funnel_af25 import evaluate as original_models
from experiments.activation_funnel_af24 import symbolic
from experiments.activation_funnel_af26 import factor
from . import gym,transfer_planner,gap

HERE=Path(__file__).resolve().parent
SPLITS=gym.SPLITS
SEEDS=gym.SEEDS
NEURAL=("dd","vd","vv","nd")
SYMBOLIC=("bfs_node_matched","bfs64","closure_direct","shortest")
ARMS=(*NEURAL,*SYMBOLIC)
SOURCE_FILES=(
    "PROTOCOL.json","gym.py","transfer_planner.py","gap.py",
    "evaluate.py","audit.py","aggregate.py",
    "tests/test_source.py","tests/test_policy.py","README.md",
    "../../.github/workflows/activation-funnel-af27-preflight.yml",
    "../../.github/workflows/activation-funnel-af27.yml",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source_hashes():
    return {name:sha((HERE/name).read_bytes()) for name in SOURCE_FILES}


def make_row(c,mode,d,budget,actions,final):
    solved=gym.GOAL in final
    stopped=bool(actions and actions[-1]["action"]==gym.STOP)
    return {
        "experiment":"AF-27",
        "split":c.split,"family":c.family,"case_id":c.id,"mode":mode,
        "initial":sorted(gym.initial_facts(c)),"final":sorted(final),
        "goal_truth":int(d is not None),"optimal_steps":d,"budget":budget,
        "actions":actions,"solved":solved,
        "initial_stop":bool(actions and actions[0]["action"]==gym.STOP),
        "premature_stop":bool(d is not None and not solved and stopped),
        "false_accepted_proof":bool(d is None and solved),
        "model_nodes":sum(
            int((step.get("plan_trace") or {}).get("nodes_forward_count",0))
            for step in actions
        ),
        "symbolic_expansions":sum(
            int((step.get("symbolic_work") or {}).get("expanded",0))
            for step in actions
        ),
        "symbolic_rule_examinations":sum(
            int((step.get("symbolic_work") or {}).get("rule_examinations",0))
            for step in actions
        ),
    }


def neural_rollout(model,cases):
    cases=tuple(cases)
    if not cases:
        raise ValueError("AF27 nonempty heldout case roster required")
    starts=[gym.initial_facts(c) for c in cases]
    distances=[gym.shortest_steps(c) for c in cases]
    budgets=[d+1 if d is not None else 2 for d in distances]
    states={arm:list(starts) for arm in NEURAL}
    done={arm:[False]*len(cases) for arm in NEURAL}
    histories={arm:[[] for _ in cases] for arm in NEURAL}
    for t in range(max(budgets)):
        jobs=[
            (arm,i,c,states[arm][i])
            for arm in NEURAL for i,c in enumerate(cases)
            if not done[arm][i] and t<budgets[i]
        ]
        plans=[]
        for start in range(0,len(jobs),32):
            part=jobs[start:start+32]
            plans.extend(transfer_planner.decisions(
                model,[c for _,_,c,_ in part],
                [s for _,_,_,s in part],
            ))
        if len(plans)!=len(jobs):
            raise ValueError("source-model complete forward count differs")
        for (arm,i,c,state),plan in zip(jobs,plans):
            legal=gym.enabled(c,state)
            selection=factor.choose(plan["scores"],legal,arm,
                                    plan["selected"])
            action=selection["chosen"]
            after,proof=gym.sound_fire(c,state,action)
            histories[arm][i].append({
                "before":sorted(state),"after":sorted(after),
                "enabled":list(legal),"action":action,"proof":proof,
                "source_scores":{
                    "direct":plan["scores"]["direct"],
                    "vi2":plan["scores"]["vi2"],
                },
                "selection":selection,
                "rule_gap":gap.gap_receipt(plan["scores"],legal) if legal else None,
                "plan_trace":plan["trace"],
                "public_graph_sha256":plan["host_graph_sha256"],
            })
            states[arm][i]=after
            if action==gym.STOP or gym.GOAL in after:
                done[arm][i]=True
    return {
        arm:[make_row(c,arm,distances[i],budgets[i],
                      histories[arm][i],states[arm][i])
             for i,c in enumerate(cases)]
        for arm in NEURAL
    }


def symbolic_episode(c,mode,seed):
    if mode not in SYMBOLIC:
        raise ValueError("unregistered AF27 symbolic policy")
    public=symbolic.PublicHorn.from_public_features(gym.public_features(c))
    amap,_=gym.physical_maps(c)
    ptol={phys:logical for logical,phys in enumerate(amap)}
    state=public.initial
    distance=gym.shortest_steps(c)
    budget=distance+1 if distance is not None else 2
    actions=[]
    for t in range(budget):
        if public.goal in state:
            break
        if mode=="shortest":
            known=frozenset(ptol[x] for x in state)
            action=gym.optimal_actions(c,known)[0]
            action=None if action==gym.STOP else action
            work={"arm":mode,"action":action,"expanded":0,
                  "rule_examinations":0,"closure_passes":0,
                  "search_exhausted":True,"wall_ns":0,
                  "oracle_only":True}
        elif mode=="bfs_node_matched":
            known=frozenset(ptol[x] for x in state)
            nodes,_=transfer_planner.state_graph(c,known)
            cap=len(nodes)
            started=time.perf_counter_ns()
            action,work=symbolic._bfs_move(public,state,cap)
            work={"arm":mode,"action":action,**work,
                  "wall_ns":time.perf_counter_ns()-started,
                  "registered_node_cap":cap}
        else:
            source="bfs_cap64" if mode=="bfs64" else "closure_direct"
            work=symbolic.choose(public,state,source,seed,c.id,t)
            action=work["action"]
        before=frozenset(ptol[x] for x in state)
        if action is None:
            after=state
            proof=None
        else:
            logical_after,proof=gym.sound_fire(c,before,action)
            after=public.fire(state,action)
            if after!=frozenset(amap[x] for x in logical_after):
                raise ValueError("original public symbolic/host transition disagree")
        actions.append({
            "before":sorted(before),
            "enabled":list(gym.enabled(c,before)),
            "action":gym.STOP if action is None else action,
            "after":sorted(ptol[x] for x in after),
            "proof":proof,"symbolic_work":work,
            "plan_trace":None,"source_scores":None,"rule_gap":None,
            "public_graph_sha256":gym.digest(gym.public_features(c)),
        })
        state=after
        if action is None or public.goal in state:
            break
    return make_row(c,mode,distance,budget,actions,
                    frozenset(ptol[x] for x in state))


def summarize(rows):
    yes=[r for r in rows if r["goal_truth"]]
    no=[r for r in rows if not r["goal_truth"]]
    if not yes or not no:
        raise ValueError("missing independent positive or negative cohort")
    solved=sum(int(r["solved"]) for r in yes)
    return {
        "cases":len(rows),"provable":len(yes),"unprovable":len(no),
        "solved":solved,"success_rate":solved/len(yes),
        "initial_stop_provable":sum(int(r["initial_stop"]) for r in yes),
        "premature_stop_provable":sum(int(r["premature_stop"]) for r in yes),
        "correct_initial_stop_unprovable":sum(int(r["initial_stop"]) for r in no),
        "false_accepted_proofs":sum(int(r["false_accepted_proof"]) for r in rows),
        "neural_model_node_evaluations":sum(r["model_nodes"] for r in rows),
        "symbolic_node_expansions":sum(r["symbolic_expansions"] for r in rows),
        "symbolic_rule_examinations":sum(r["symbolic_rule_examinations"] for r in rows),
    }


def root_gap_rows(cases,dd):
    out=[]
    for c,row in zip(cases,dd):
        if not c.designed_positive:
            continue
        if row["mode"]!="dd" or row["case_id"]!=c.id or not row["actions"]:
            raise ValueError("missing independent Direct root model ledger")
        step=row["actions"][0]
        optimal=gym.optimal_actions(c,gym.initial_facts(c))
        diagnostic=gap.evaluate_root_gap(
            c,step["source_scores"],tuple(step["enabled"]),
            step["plan_trace"],optimal
        )
        out.append({
            "case_id":c.id,"split":c.split,"family":c.family,
            "root_known":step["before"],
            "root_rule_gap":diagnostic,
            "model_node_count":step["plan_trace"]["nodes_forward_count"],
            "truth_used_at_inference":False,
        })
    return out


def evaluate(seed,zip_dir,out_dir,smoke=False):
    if seed not in SEEDS:
        raise ValueError("AF27 unregistered frozen checkpoint seed")
    out=Path(out_dir)
    if out.exists():
        raise FileExistsError("AF27 output must use a fresh directory")
    out.mkdir(parents=True)
    manifest=gym.manifest()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model,provenance=original_models.load_frozen_original(zip_dir,seed)
    scores={}
    episodes=[]
    gaprows=[]
    started=time.perf_counter()
    for split in SPLITS:
        cases=gym.cases_for(split)
        if smoke:
            cases=([c for c in cases if c.designed_positive][:2]+
                   [c for c in cases if not c.designed_positive][:2])
        results=neural_rollout(model,cases)
        for mode in SYMBOLIC:
            results[mode]=[symbolic_episode(c,mode,seed) for c in cases]
        if tuple(results)!=ARMS:
            raise ValueError("registered AF27 arm roster changed")
        scores[split]={mode:summarize(rows) for mode,rows in results.items()}
        episodes.extend(r for rows in results.values() for r in rows)
        gaprows.extend(root_gap_rows(cases,results["dd"]))
    payload=("".join(json.dumps(x,sort_keys=True,allow_nan=False)+"\n"
                     for x in episodes)).encode()
    gapdata=("".join(json.dumps(x,sort_keys=True,allow_nan=False)+"\n"
                     for x in gaprows)).encode()
    (out/"episodes.jsonl").write_bytes(payload)
    (out/"root_gaps.jsonl").write_bytes(gapdata)
    receipt={
        "experiment":"AF-27","seed":seed,"smoke":smoke,
        "protocol_sha256":sha((HERE/"PROTOCOL.json").read_bytes()),
        "structural_manifest_sha256":manifest["manifest_sha256"],
        "source_hashes":source_hashes(),
        "original_af23_checkpoint":provenance,
        "training_updates":0,"all_arms":list(ARMS),
        "scores":scores,
        "episode_count":len(episodes),"episodes_sha256":sha(payload),
        "root_gap_count":len(gaprows),"root_gaps_sha256":sha(gapdata),
        "elapsed_seconds":time.perf_counter()-started,
        "scientific_verdict":"SMOKE_ONLY" if smoke
                             else "AWAITING_EIGHT_SEED_INDEPENDENT_AGGREGATE",
    }
    (out/"evaluation.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n"
    )
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",required=True,type=int,choices=SEEDS)
    p.add_argument("--original-zip-dir",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    r=evaluate(a.seed,a.original_zip_dir,a.out,a.smoke)
    print(json.dumps({"seed":a.seed,"episodes":r["episode_count"],
                      "root_gaps":r["root_gap_count"],
                      "status":r["scientific_verdict"]}))


if __name__=="__main__":
    main()
