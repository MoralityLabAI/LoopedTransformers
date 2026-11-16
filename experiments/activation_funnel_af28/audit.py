"""AF-28 independent physical-incidence proof and native-gap audit.

Rebuild Horn incidence from model-visible GraphBatch tensors. Derive true
shortest proof length using a separate bitset BFS, verify every applied rule
and STOP, recompute the complete original neural Bellman score ledger, check
the source-specific STOP/rank factor and all symbolic work counters, then
sample genuine Torch forwards from the original frozen checkpoint archive.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from collections import deque
from functools import lru_cache
from pathlib import Path
import torch

from experiments.activation_funnel_af24 import symbolic
from experiments.activation_funnel_af26 import factor
from experiments.activation_funnel_af25 import evaluate as original
from . import gym,transfer_planner,calibration,calibrate,evaluate as ex


@lru_cache(maxsize=512)
def public_graph(c):
    x=gym.encode(c)
    pre=x.prem_up+x.prem_down
    head=x.head_up+x.head_down
    rules={}
    for slot in range(gym.RULES):
        if x.rule_features[slot,2] <.5:
            continue
        ps=tuple(torch.where(pre[slot]>.5)[0].tolist())
        hs=tuple(torch.where(head[slot]>.5)[0].tolist())
        if not ps or len(ps)>2 or len(hs)!=1:
            raise ValueError("invalid model-visible physical Horn rule")
        rules[slot]=(ps,hs[0])
    initial=frozenset(torch.where(x.atom_features[:,0]>.5)[0].tolist())
    return rules,initial,int(x.read_slots[2])


def shortest_bfs(rules,known,goal):
    root=sum(1<<p for p in known)
    if root & (1<<goal):
        return 0
    masks=tuple((sum(1<<p for p in pre),1<<head)
                for pre,head in rules.values())
    queue=deque([(root,0)]);seen={root}
    while queue:
        state,d=queue.popleft()
        for pre,head in masks:
            if pre&state!=pre or head&state:
                continue
            nxt=state|head
            if nxt&(1<<goal):
                return d+1
            if nxt not in seen:
                seen.add(nxt);queue.append((nxt,d+1))
    return None


@lru_cache(maxsize=512)
def oracle(c):
    rules,initial,goal=public_graph(c)
    d=shortest_bfs(rules,initial,goal)
    if d!=gym.independent_bfs(c) or d!=gym.shortest_steps(c):
        raise ValueError("model-visible public proof BFS differs from separate oracle")
    return d


def sigmoid(z):
    if z>=0:
        return 1.0/(1.+math.exp(-z))
    x=math.exp(z)
    return x/(1.+x)


def softplus(z):
    return max(z,0.)+math.log1p(math.exp(-abs(z)))


def numeric_trace(c,root_state,trace,scores):
    """Independent physical graph and original AF23 min-plus action score."""
    rules,_,goal=public_graph(c)
    apos,_=gym.physical_maps(c)
    ptol={p:l for l,p in enumerate(apos)}
    root=frozenset(root_state)
    todo=deque([(root,0)])
    visited={root:0}
    physical_edges={}
    while todo:
        logical,depth=todo.popleft()
        phys=frozenset(apos[a] for a in logical)
        ed={}
        if goal not in phys and depth <2:
            for slot,(pre,head) in sorted(rules.items()):
                if head in phys or not all(p in phys for p in pre):
                    continue
                after=frozenset(set(logical)|{ptol[head]})
                ed[str(slot)]=sorted(after)
                if after not in visited:
                    visited[after]=depth+1
                    todo.append((after,depth+1))
        physical_edges[logical]=ed
    saved={frozenset(n["state"]):n for n in trace["nodes"]}
    if (set(saved)!=set(physical_edges) or
        trace["nodes_forward_count"]!=len(saved)):
        raise ValueError("original model node ledger changed")
    V={};S={}
    for state,node in saved.items():
        if node["edges"]!=physical_edges[state]:
            raise ValueError("forged public successor edge in model node ledger")
        p=sigmoid(float(node["reach_logit"]))
        d=min(16.,max(0.,softplus(float(node["distance_logit"]))))
        leaf=p*d+(1-p)*16.
        halt=16.*p
        if gym.GOAL in state:
            leaf=halt=0.
        if abs(leaf-node["value0"])>1e-5 or abs(halt-node["stop_cost"])>1e-5:
            raise ValueError("original learned leaf/STOP cost did not replay")
        V[state],S[state]=leaf,halt
    backups={
        state:min([S[state]]+[1.+V[frozenset(t)]
                        for t in edges.values()])
        for state,edges in physical_edges.items()
    }
    original={name:[-1e4]*15 for name in ("direct","vi2")}
    if len(trace["root_raw"])!=15:
        raise ValueError("original root neural logits not present")
    for slot,future in physical_edges[root].items():
        a=int(slot);s=frozenset(future)
        original["direct"][a]=float(trace["root_raw"][a])
        original["vi2"][a]=float(-1.-backups[s])
    original["direct"][gym.STOP]=float(trace["root_raw"][gym.STOP])
    original["vi2"][gym.STOP]=-S[root]
    for name,vector in original.items():
        observed=scores[name]
        if len(observed)!=15 or any(abs(a-b)>1e-5 for a,b in zip(vector,observed)):
            raise ValueError("independently recomputed native source score mismatch")
    return original


def validate_episode(c,row,learned):
    rules,initial,goal=public_graph(c)
    perm,_=gym.physical_maps(c)
    ptol={physical:logical for logical,physical in enumerate(perm)}
    logical=lambda physical:sorted(ptol[p] for p in physical)
    shortest=oracle(c)
    if (row["experiment"],row["case_id"],row["split"],row["family"])!=(
        "AF-28",c.id,c.split,c.family):
        raise ValueError("forged AF28 physical case identity")
    if (row["initial"]!=logical(initial) or
        row["goal_truth"]!=int(shortest is not None) or
        row["optimal_steps"]!=shortest or
        row["budget"]!=(shortest+1 if shortest is not None else 2)):
        raise ValueError("source truth/proof budget differs from independent BFS")
    if len(row["actions"])>row["budget"] or row["mode"] not in ex.ARMS:
        raise ValueError("unknown or overbudget AF28 proof episode")
    state=set(initial)
    ended=False;nodes=expanded=examined=0
    for t,step in enumerate(row["actions"]):
        if ended or step["before"]!=logical(state):
            raise ValueError("AF28 action after termination or from wrong state")
        legal=tuple(sorted(slot for slot,(prem,head) in rules.items()
            if head not in state and all(p in state for p in prem)))
        if step["enabled"]!=list(legal):
            raise ValueError("physical Horn legal mask differs")
        if step["public_graph_sha256"]!=gym.digest(gym.public_features(c)):
            raise ValueError("source public graph identity mismatch")
        if row["mode"] in ex.NEURAL:
            if "work" in step or step["source_scores"] is None and legal:
                raise ValueError("incomplete frozen neural policy node receipt")
            if legal:
                original=numeric_trace(c,step["before"],
                                      step["plan_trace"],step["source_scores"])
                if row["mode"] in ("DD","VD","ND"):
                    factored=factor.choose(
                        step["source_scores"],legal,row["mode"].lower()
                    )
                    expected=factored["chosen"]
                    if step["gate_selected_stop"]!=factored["gate_selected_stop"]:
                        raise ValueError("native original STOP gate was forged")
                    if (step["calibration_features"] is not None or
                        step["calibrated_STOP_probability"] is not None or
                        step["calibration_threshold"] is not None):
                        raise ValueError("original native STOP arm included fitted feature")
                else:
                    arm=row["mode"]
                    features=calibration.feature_row(
                        {"scores":step["source_scores"],
                         "trace":step["plan_trace"]},
                        step["before"],legal
                    )
                    if step["calibration_features"]!=features:
                        raise ValueError("source-public calibrated feature differs")
                    params=learned["classifiers"][arm]
                    p=calibration.predict(params,features)
                    t0=params["dev"]["chosen_threshold"]
                    if (step["calibrated_STOP_probability"] is None or
                        abs(p-step["calibrated_STOP_probability"])>1e-10 or
                        step["calibration_threshold"]!=t0):
                        raise ValueError("frozen STOP probability/threshold changed")
                    stop=calibration.choose(p,t0)
                    expected=(gym.STOP if stop else
                              max(legal,key=lambda a:step["source_scores"]["direct"][a]))
                    if step["gate_selected_stop"]!=stop:
                        raise ValueError("calibrated STOP threshold action differs")
                if step["action"]!=expected:
                    raise ValueError("registered neural STOP/rule action differs")
                nodes+=step["plan_trace"]["nodes_forward_count"]
            else:
                if (step["action"]!=gym.STOP or
                    step["plan_trace"] is not None or
                    step["source_scores"] is not None or
                    step["calibration_features"] is not None):
                    raise ValueError("STOP-only state claims a neural policy forward")
        else:
            if (step["plan_trace"] is not None or
                step["source_scores"] is not None or
                step["calibration_features"] is not None or
                step["calibrated_STOP_probability"] is not None):
                raise ValueError("symbolic controller has private neural features")
            name={
                "CLOSURE":"closure_direct","BFS16":"bfs_cap16",
                "BFS64":"bfs_cap64","SHORTEST":"shortest"
            }[row["mode"]]
            graph=symbolic.PublicHorn(
                rules=tuple(sorted((slot,tuple(p),h)
                                   for slot,(p,h) in rules.items())),
                initial=frozenset(state),goal=goal
            )
            if name=="shortest":
                current=frozenset(ptol[p] for p in state)
                a=gym.optimal_actions(c,current)[0]
                expected=a
            else:
                scored=symbolic.choose(graph,frozenset(state),name,17,c.id,t)
                expected=gym.STOP if scored["action"] is None else scored["action"]
                for field in ("expanded","rule_examinations",
                              "closure_passes","search_exhausted"):
                    if scored[field]!=step["work"][field]:
                        raise ValueError("AF28 symbolic controller work counter differs")
            if expected!=step["action"]:
                raise ValueError("source original symbolic action replay differs")
            expanded+=step["work"]["expanded"]
            examined+=step["work"]["rule_examinations"]
        if step["action"]==gym.STOP:
            if step["proof"] is not None:
                raise ValueError("STOP cannot claim physical Horn proof")
            ended=True
        else:
            if step["action"] not in legal:
                raise ValueError("model fired nonenabled physical rule")
            pre,head=rules[step["action"]]
            proof=step["proof"]
            if (not isinstance(proof,dict) or
                proof.get("rule_slot")!=step["action"] or
                proof.get("head")!=ptol[head] or
                sorted(proof.get("premises",[]))!=sorted(ptol[p] for p in pre)):
                raise ValueError("forged physical Horn proof evidence")
            state.add(head)
            if goal in state:ended=True
        if step["after"]!=logical(state):
            raise ValueError("original source physical Horn successor mismatch")
    solved=goal in state
    is_stop=bool(row["actions"] and row["actions"][-1]["action"]==gym.STOP)
    if (row["final"]!=logical(state) or row["solved"]!=solved or
        row["false_accepted_proof"] or (shortest is None and solved) or
        row["initial_stop"]!=bool(row["actions"] and
                                     row["actions"][0]["action"]==gym.STOP) or
        row["premature_stop"]!=bool(shortest is not None and not solved and is_stop)
        or row["neural_node_forward_count"]!=nodes
        or row["symbolic_expansions"]!=expanded
        or row["symbolic_rule_examinations"]!=examined):
        raise ValueError("AF28 source proof acceptance or inference accounting inconsistent")
    return nodes


def audit(seed,original_zip_dir,calibrator_dir,output_dir,smoke=False):
    output=Path(output_dir)
    rec=json.loads((output/"evaluation.json").read_text())
    episodes_bytes=(output/"episodes.jsonl").read_bytes()
    root_bytes=(output/"root_probabilities.jsonl").read_bytes()
    model,original_provenance=original.load_frozen_original(original_zip_dir,seed)
    calibration_data=calibrate.load_calibrator(
        calibrator_dir,seed,original_provenance
    )
    if (rec["experiment"]!="AF-28" or rec["seed"]!=seed or
        rec["smoke"]!=smoke or rec["all_arms"]!=list(ex.ARMS) or
        rec["source_hashes"]!=calibrate.source_hashes() or
        rec["protocol_sha256"]!=ex.sha((ex.HERE/"PROTOCOL.json").read_bytes()) or
        rec["manifest_sha256"]!=gym.manifest()["manifest_sha256"] or
        rec["original_af23_checkpoint"]!=original_provenance or
        rec["calibrator_receipt_sha256"]!=ex.sha(
            (Path(calibrator_dir)/"calibrator.json").read_bytes()) or
        rec["training_updates_source_model"]!=0 or
        rec["episodes_sha256"]!=ex.sha(episodes_bytes) or
        rec["held_root_sha256"]!=ex.sha(root_bytes)):
        raise ValueError("AF28 source, original-model or calibrator hash invalid")
    rows=[json.loads(line) for line in episodes_bytes.splitlines()]
    roots=[json.loads(line) for line in root_bytes.splitlines()]
    expected_per_split=4 if smoke else 192
    expected=expected_per_split*len(ex.SPLITS)*len(ex.ARMS)
    if len(rows)!=expected or rec["episode_count"]!=expected:
        raise ValueError("incomplete heldout AF28 full proof episode roster")
    index={}
    for row in rows:
        key=(row["split"],row["mode"],row["case_id"])
        if key in index:raise ValueError("duplicate scored Horn policy episode")
        index[key]=row
    nodes=0
    for split in ex.SPLITS:
        cases=gym.cases_for(split)
        if smoke:
            cases=[c for c in cases if c.designed_positive][:2]+[
                c for c in cases if not c.designed_positive][:2]
        for c in cases:
            for arm in ex.ARMS:
                item=index.get((split,arm,c.id))
                if item is None:raise ValueError("missing scored source case")
                nodes+=validate_episode(c,item,calibration_data)
        for arm in ex.ARMS:
            selected=[index[(split,arm,c.id)] for c in cases]
            current=ex.summarize(selected)
            for k in ("cases","provable","unprovable","solved",
                      "first_stop_positive","premature_stop_positive",
                      "correct_initial_stop_negative","false_accepted_proofs",
                      "neural_model_nodes","symbolic_expansions",
                      "symbolic_rule_examinations"):
                if current[k]!=rec["scores"][split][arm][k]:
                    raise ValueError("summarized proof/STOP counts differ")
    expected_roots=expected_per_split*len(ex.SPLITS)
    if len(roots)!=rec["held_root_count"] or len(roots)!=expected_roots:
        raise ValueError("heldout initial root probability roster missing")
    seen_roots=set()
    for row in roots:
        key=(row["split"],row["case_id"])
        if key in seen_roots:raise ValueError("duplicate probability root")
        seen_roots.add(key)
        for arm,field in (("CAL","CAL_probability"),("THR","THR_probability")):
            item=index[(row["split"],arm,row["case_id"])]
            if (not item["actions"] or
                row[field]!=item["actions"][0]["calibrated_STOP_probability"] or
                row["actual_STOP_label"]!=int(not bool(item["goal_truth"])) or
                row["used_as_calibration_training"] or row["used_for_dev_selection"]):
                raise ValueError("calibration heldout label leakage or wrong probability")
    # Randomly selected cases would create a posthoc audit budget. Use fixed
    # first two cases per split and fixed native policies instead.
    replays=0
    for split in ex.SPLITS:
        cases=gym.cases_for(split)
        if smoke:
            cases=[c for c in cases if c.designed_positive][:2]+[
                c for c in cases if not c.designed_positive][:2]
        for c in cases[:2]:
            for arm in ex.NEURAL:
                step=index[(split,arm,c.id)]["actions"][0]
                if not step["enabled"]:continue
                fresh=transfer_planner.decisions(
                    model,[c],[frozenset(step["before"])]
                )[0]
                for name in ("direct","vi2"):
                    a=fresh["scores"][name];b=step["source_scores"][name]
                    if len(a)!=len(b) or any(abs(x-y)>1e-5
                                            for x,y in zip(a,b)):
                        raise ValueError("actual original Torch forward score changed")
                replays+=1
    # Reconstruct ALL training and development rows in the exact original
    # 32-item forward batches, preserving source-neural FP32 tie behavior.
    # This is stronger than singleton source-feature sampling: every fitted
    # input, target label, source case and graph digest is checked against
    # an actual original frozen-model forward from the source schedule.
    source_feature_replays=0
    for split in ("train","dev"):
        saved_path=Path(calibrator_dir)/f"{split}_features.jsonl"
        records=[json.loads(line) for line in saved_path.read_text().splitlines()]
        source_rows=calibrate.generate_features(
            model,split,include_successors=(split=="train")
        )
        if len(records)!=len(source_rows):
            raise ValueError("original calibration train/dev batch roster drift")
        for saved,fresh in zip(records,source_rows):
            for key in ("split","case_id","family","state",
                        "public_graph_sha256","stop_target"):
                if saved[key]!=fresh[key]:
                    raise ValueError("original batched train/dev sample identity drift")
            if any(abs(a-b)>1e-5 for a,b in
                   zip(saved["features"],fresh["features"])):
                raise ValueError(
                    f"AF28 original batch-locked calibration feature drift "
                    f"{split}/{saved['case_id']}/{saved['state']}"
                )
            source_feature_replays+=1
    audit_result={
        "experiment":"AF-28","seed":seed,"smoke":smoke,
        "verified":True,
        "audited_episodes":len(rows),
        "independent_neural_node_receipts":nodes,
        "heldout_initial_probability_receipts":len(roots),
        "actual_frozen_checkpoint_forward_replays":replays,
        "train_dev_source_feature_replays":source_feature_replays,
        "false_accepted_proofs":0,
        "original_checkpoint":original_provenance,
        "source_hashes":calibrate.source_hashes(),
        "protocol_sha256":rec["protocol_sha256"],
        "manifest_sha256":rec["manifest_sha256"],
        "calibrator_receipt_sha256":rec["calibrator_receipt_sha256"],
        "evaluation_sha256":ex.sha((output/"evaluation.json").read_bytes()),
        "episodes_sha256":ex.sha(episodes_bytes),
        "held_root_sha256":ex.sha(root_bytes),
    }
    calibrate.write_json(output/"independent_audit.json",audit_result)
    return audit_result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,required=True,choices=gym.SEEDS)
    p.add_argument("--original-zip-dir",type=Path,required=True)
    p.add_argument("--calibrator-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    result=audit(a.seed,a.original_zip_dir,a.calibrator_dir,
                 a.output_dir,a.smoke)
    print(json.dumps({
        "seed":a.seed,"episodes":result["audited_episodes"],
        "nodes":result["independent_neural_node_receipts"],
        "frozen_model_replays":result["actual_frozen_checkpoint_forward_replays"],
        "train_dev_feature_replays":result["train_dev_source_feature_replays"]
    }))


if __name__=="__main__":
    main()
