"""AF-28 sealed heldout full-episode evaluation after train/dev-only fit.

The fit archive is an input and NEVER altered. CAL and THR use train-fitted
logistic probabilities with development-only thresholds; all neural modes
use the original 6036-weight frozen recurrent model and Direct legal ranking.
Each mode follows its own fresh host-certified complete trajectory.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import torch

from experiments.activation_funnel_af25 import evaluate as original
from experiments.activation_funnel_af26 import factor
from experiments.activation_funnel_af24 import symbolic
from . import gym,transfer_planner,calibration,calibrate

HERE=Path(__file__).resolve().parent
SEEDS=gym.SEEDS
SPLITS=("held_depth5","held_depth6")
NEURAL=("DD","VD","CAL","THR","ND")
BASELINES=("CLOSURE","BFS16","BFS64","SHORTEST")
ARMS=(*NEURAL,*BASELINES)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def make_row(c,mode,d,budget,history,final):
    solved=gym.GOAL in final
    stopped=bool(history and history[-1]["action"]==gym.STOP)
    return {
        "experiment":"AF-28","split":c.split,"family":c.family,
        "case_id":c.id,"mode":mode,
        "initial":sorted(gym.initial_facts(c)),
        "final":sorted(final),"goal_truth":int(d is not None),
        "optimal_steps":d,"budget":budget,"actions":history,
        "solved":bool(solved),
        "initial_stop":bool(history and history[0]["action"]==gym.STOP),
        "premature_stop":bool(d is not None and not solved and stopped),
        "false_accepted_proof":bool(d is None and solved),
        "neural_node_forward_count":sum(
            int((a.get("plan_trace") or {}).get("nodes_forward_count",0))
            for a in history
        ),
        "symbolic_expansions":sum(int((a.get("work") or {}).get("expanded",0))
                                   for a in history),
        "symbolic_rule_examinations":sum(
            int((a.get("work") or {}).get("rule_examinations",0))
            for a in history
        ),
    }


def neural_rollout(model,learned,cases):
    cases=tuple(cases)
    distances=[gym.shortest_steps(c) for c in cases]
    budgets=[(d+1) if d is not None else 2 for d in distances]
    states={arm:[gym.initial_facts(c) for c in cases] for arm in NEURAL}
    finished={arm:[False]*len(cases) for arm in NEURAL}
    history={arm:[[] for _ in cases] for arm in NEURAL}
    for step_index in range(max(budgets)):
        jobs=[(arm,i,c,states[arm][i])
              for arm in NEURAL for i,c in enumerate(cases)
              if not finished[arm][i] and step_index < budgets[i]]
        active=[(arm,i,c,s) for arm,i,c,s in jobs if gym.enabled(c,s)]
        plans=[]
        for start in range(0,len(active),32):
            part=active[start:start+32]
            plans.extend(transfer_planner.decisions(
                model,[c for arm,i,c,s in part],[s for arm,i,c,s in part]
            ))
        if len(plans)!=len(active):
            raise ValueError("AF28 missing original-model inference forward")
        by_job={(arm,i):plan for (arm,i,c,s),plan in zip(active,plans)}
        for arm,i,c,state in jobs:
            legal=gym.enabled(c,state)
            trace=source=None
            features=None
            prob=None
            stopped_by_gate=False
            if not legal:
                action=gym.STOP
            else:
                plan=by_job[(arm,i)]
                trace=plan["trace"]
                source={
                    "direct":plan["scores"]["direct"],
                    "vi2":plan["scores"]["vi2"],
                }
                if arm in ("DD","VD","ND"):
                    source_arm={"DD":"dd","VD":"vd","ND":"nd"}[arm]
                    chosen=factor.choose(
                        plan["scores"],legal,source_arm,plan["selected"]
                    )
                    action=chosen["chosen"]
                    stopped_by_gate=chosen["gate_selected_stop"]
                else:
                    features=calibration.feature_row(plan,state,legal)
                    submodel=learned["classifiers"][arm]
                    prob=calibration.predict(submodel,features)
                    stopped_by_gate=calibration.choose(
                        prob,submodel["dev"]["chosen_threshold"]
                    )
                    action=(gym.STOP if stopped_by_gate else
                            max(legal,key=lambda a:source["direct"][a]))
            after,proof=gym.sound_fire(c,state,action)
            history[arm][i].append({
                "before":sorted(state),"enabled":list(legal),
                "action":action,"after":sorted(after),"proof":proof,
                "plan_trace":trace,"source_scores":source,
                "calibration_features":features,
                "calibrated_STOP_probability":prob,
                "calibration_threshold":(
                    learned["classifiers"][arm]["dev"]["chosen_threshold"]
                    if arm in ("CAL","THR") else None
                ),
                "gate_selected_stop":stopped_by_gate,
                "public_graph_sha256":gym.digest(gym.public_features(c)),
            })
            states[arm][i]=after
            if action==gym.STOP or gym.GOAL in after:
                finished[arm][i]=True
    return {
        arm:[make_row(c,arm,distances[i],budgets[i],
                     history[arm][i],states[arm][i])
             for i,c in enumerate(cases)]
        for arm in NEURAL
    }


def symbolic_episode(c,mode,seed):
    if mode not in BASELINES:
        raise ValueError("unregistered AF28 host control")
    graph=symbolic.PublicHorn.from_public_features(gym.public_features(c))
    perm,_=gym.physical_maps(c)
    ptol={phys:logical for logical,phys in enumerate(perm)}
    state=graph.initial
    d=gym.shortest_steps(c)
    budget=(d+1) if d is not None else 2
    history=[]
    for t in range(budget):
        if graph.goal in state:break
        if mode=="SHORTEST":
            current=frozenset(ptol[p] for p in state)
            a=gym.optimal_actions(c,current)[0]
            action=None if a==gym.STOP else a
            work={"action":action,"arm":"shortest",
                  "expanded":0,"rule_examinations":0,
                  "closure_passes":0,"search_exhausted":True,"wall_ns":0,
                  "oracle_only":True}
        else:
            registered={
                "CLOSURE":"closure_direct",
                "BFS16":"bfs_cap16",
                "BFS64":"bfs_cap64"
            }[mode]
            work=symbolic.choose(graph,state,registered,seed,c.id,t)
            action=work["action"]
        before=frozenset(ptol[p] for p in state)
        if action is None:
            after=state;proof=None
        else:
            host_after,proof=gym.sound_fire(c,before,action)
            after=graph.fire(state,action)
            if after!=frozenset(perm[a] for a in host_after):
                raise ValueError("public incidence and host action mismatch")
        history.append({
            "before":sorted(before),"enabled":list(gym.enabled(c,before)),
            "action":gym.STOP if action is None else action,
            "after":sorted(ptol[p] for p in after),"proof":proof,
            "work":work,"plan_trace":None,"source_scores":None,
            "calibration_features":None,
            "calibrated_STOP_probability":None,
            "calibration_threshold":None,
            "gate_selected_stop":None,
            "public_graph_sha256":gym.digest(gym.public_features(c)),
        })
        state=after
        if action is None or graph.goal in state:
            break
    return make_row(c,mode,d,budget,history,
                    frozenset(ptol[p] for p in state))


def summarize(rows):
    positives=[r for r in rows if r["goal_truth"]]
    negatives=[r for r in rows if not r["goal_truth"]]
    if not positives or not negatives:
        raise ValueError("AF28 positive and negative evidence required")
    solved=sum(r["solved"] for r in positives)
    return {
        "cases":len(rows),"provable":len(positives),
        "unprovable":len(negatives),
        "solved":solved,"proof_success":solved/len(positives),
        "first_stop_positive":sum(r["initial_stop"] for r in positives),
        "premature_stop_positive":sum(r["premature_stop"] for r in positives),
        "correct_initial_stop_negative":sum(r["initial_stop"] for r in negatives),
        "false_accepted_proofs":sum(r["false_accepted_proof"] for r in rows),
        "neural_model_nodes":sum(r["neural_node_forward_count"] for r in rows),
        "symbolic_expansions":sum(r["symbolic_expansions"] for r in rows),
        "symbolic_rule_examinations":sum(r["symbolic_rule_examinations"] for r in rows),
    }


def evaluate(seed,original_zip_dir,calibrator_dir,out_dir,smoke=False):
    if seed not in SEEDS:
        raise ValueError("AF28 frozen checkpoint seed missing")
    out=Path(out_dir)
    if out.exists():
        raise FileExistsError("AF28 heldout outputs must be created in new folder")
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model,provenance=original.load_frozen_original(original_zip_dir,seed)
    learned=calibrate.load_calibrator(calibrator_dir,seed,provenance)
    manifest=gym.manifest()
    rows=[];scores={};rootprob=[]
    started=time.perf_counter()
    for split in SPLITS:
        cases=gym.cases_for(split)
        if smoke:
            cases=[c for c in cases if c.designed_positive][:2]+[
                c for c in cases if not c.designed_positive][:2]
        results=neural_rollout(model,learned,cases)
        for arm in BASELINES:
            results[arm]=[symbolic_episode(c,arm,seed) for c in cases]
        if tuple(results)!=ARMS:
            raise ValueError("AF28 source arm roster changed")
        for arm in ARMS:
            rows.extend(results[arm])
        scores[split]={arm:summarize(episodes) for arm,episodes in results.items()}
        for c,r_cal,r_thr in zip(cases,results["CAL"],results["THR"]):
            pc=(r_cal["actions"][0]["calibrated_STOP_probability"]
                if r_cal["actions"] else None)
            pt=(r_thr["actions"][0]["calibrated_STOP_probability"]
                if r_thr["actions"] else None)
            if pc is None or pt is None:
                raise ValueError("missing source heldout root calibrated probability")
            rootprob.append({
                "experiment":"AF-28",
                "case_id":c.id,"split":split,"family":c.family,
                "actual_STOP_label":int(not c.designed_positive),
                "CAL_probability":pc,"THR_probability":pt,
                "chosen_CAL_threshold":learned["classifiers"]["CAL"]["dev"]["chosen_threshold"],
                "chosen_THR_threshold":learned["classifiers"]["THR"]["dev"]["chosen_threshold"],
                "used_as_calibration_training":False,
                "used_for_dev_selection":False,
            })
    raw=("".join(json.dumps(r,sort_keys=True,allow_nan=False)+"\n"
                 for r in rows)).encode()
    root_raw=("".join(json.dumps(r,sort_keys=True,allow_nan=False)+"\n"
                      for r in rootprob)).encode()
    (out/"episodes.jsonl").write_bytes(raw)
    (out/"root_probabilities.jsonl").write_bytes(root_raw)
    cal_sha=sha((Path(calibrator_dir)/"calibrator.json").read_bytes())
    receipt={
        "experiment":"AF-28","seed":seed,"smoke":smoke,
        "source_hashes":calibrate.source_hashes(),
        "protocol_sha256":sha((HERE/"PROTOCOL.json").read_bytes()),
        "manifest_sha256":manifest["manifest_sha256"],
        "original_af23_checkpoint":provenance,
        "calibrator_receipt_sha256":cal_sha,
        "calibration_dev_thresholds":{
            a:learned["classifiers"][a]["dev"]["chosen_threshold"]
            for a in ("CAL","THR")
        },
        "training_updates_source_model":0,
        "all_arms":list(ARMS),
        "scores":scores,
        "episodes_sha256":sha(raw),"episode_count":len(rows),
        "held_root_sha256":sha(root_raw),"held_root_count":len(rootprob),
        "elapsed_seconds":time.perf_counter()-started,
        "original_frozen_model_training":False,
        "scientific_verdict":"SMOKE_ONLY" if smoke
            else "AWAITING_INDEPENDENT_EIGHT_SEED_VERDICT",
    }
    calibrate.write_json(out/"evaluation.json",receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,required=True,choices=SEEDS)
    p.add_argument("--original-zip-dir",type=Path,required=True)
    p.add_argument("--calibrator-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    receipt=evaluate(a.seed,a.original_zip_dir,a.calibrator_dir,a.out,a.smoke)
    print(json.dumps({
        "seed":a.seed,"smoke":a.smoke,"episodes":receipt["episode_count"],
        "thresholds":receipt["calibration_dev_thresholds"],
        "status":receipt["scientific_verdict"]
    }))


if __name__=="__main__":
    main()
