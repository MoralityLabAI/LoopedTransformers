"""AF-27 independent physical-incidence proof and native-gap audit.

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
from . import gym,transfer_planner,gap,evaluate as ex


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


def validate_episode(c,row):
    rules,initial,goal=public_graph(c)
    apos,_=gym.physical_maps(c)
    ptol={p:l for l,p in enumerate(apos)}
    logical=lambda phys:sorted(ptol[p] for p in phys)
    d=oracle(c)
    if (row["experiment"],row["case_id"],row["split"],row["family"])!=(
            "AF-27",c.id,c.split,c.family):
        raise ValueError("missing or altered source Horn case")
    if (row["initial"]!=logical(initial)
        or row["goal_truth"]!=int(d is not None)
        or row["optimal_steps"]!=d
        or row["budget"]!=(d+1 if d is not None else 2)):
        raise ValueError("independent source root/truth/budget mismatch")
    if len(row["actions"])>row["budget"]:
        raise ValueError("proof episode exceeded budget")
    state=set(initial)
    ended=False
    nodes=expansions=examinations=0
    for step_idx,step in enumerate(row["actions"]):
        if ended or step["before"]!=logical(state):
            raise ValueError("continued after terminal or altered state")
        legal=tuple(sorted(slot for slot,(pre,head) in rules.items()
                           if head not in state and all(p in state for p in pre)))
        if step["enabled"]!=list(legal):
            raise ValueError("public Horn legal action mask differs")
        if step["public_graph_sha256"]!=gym.digest(gym.public_features(c)):
            raise ValueError("root graph source identity changed")
        if row["mode"] in ex.NEURAL:
            if step.get("symbolic_work") is not None or (bool(legal) != (step.get("rule_gap") is not None)):
                raise ValueError("neural policy missing native gap source")
            original=numeric_trace(c,step["before"],step["plan_trace"],
                                   step["source_scores"])
            selected=factor.choose(step["source_scores"],legal,row["mode"])
            if (selected!=step["selection"] or step["action"]!=selected["chosen"]
                or gap.gap_receipt(step["source_scores"],legal)!=step["rule_gap"]):
                if legal:
                    raise ValueError("native STOP/rule factor or score-gap record changed")
                if selected!=step["selection"] or step["action"]!=selected["chosen"]:
                    raise ValueError("action selection differs without legal rules")
            nodes+=step["plan_trace"]["nodes_forward_count"]
        else:
            if step["source_scores"] is not None or step["plan_trace"] is not None:
                raise ValueError("symbolic policy accessed neural scores")
            work=step["symbolic_work"]
            graph=symbolic.PublicHorn(
                rules=tuple(sorted((slot,tuple(pre),head)
                                   for slot,(pre,head) in rules.items())),
                initial=frozenset(state),goal=goal,
            )
            if row["mode"]=="shortest":
                logical_state=frozenset(ptol[p] for p in state)
                a=gym.optimal_actions(c,logical_state)[0]
                expected=gym.STOP if a==gym.STOP else a
            elif row["mode"]=="bfs_node_matched":
                logical_state=frozenset(ptol[p] for p in state)
                cap=len(transfer_planner.state_graph(c,logical_state)[0])
                a,proofwork=symbolic._bfs_move(graph,frozenset(state),cap)
                expected=gym.STOP if a is None else a
                for key in ("expanded","rule_examinations","closure_passes",
                            "search_exhausted"):
                    if proofwork[key]!=work[key]:
                        raise ValueError("node-matched BFS work metric changed")
                if work["registered_node_cap"]!=cap:
                    raise ValueError("symbolic BFS node cap not matched to model frontier")
            else:
                name="bfs_cap64" if row["mode"]=="bfs64" else "closure_direct"
                replay=symbolic.choose(graph,frozenset(state),name,
                                       17,c.id,step_idx)
                expected=gym.STOP if replay["action"] is None else replay["action"]
                for key in ("expanded","rule_examinations","closure_passes",
                            "search_exhausted"):
                    if replay[key]!=work[key]:
                        raise ValueError("symbolic controller arithmetic/work changed")
            if step["action"]!=expected:
                raise ValueError("symbolic host action did not replay")
            expansions+=work["expanded"]
            examinations+=work["rule_examinations"]
        action=step["action"]
        if action==gym.STOP:
            if step["proof"] is not None:
                raise ValueError("STOP cannot claim physical proof")
            ended=True
        else:
            if action not in legal:
                raise ValueError("forged unsupported physical rule")
            pre,head=rules[action]
            proof=step["proof"]
            if (not isinstance(proof,dict)
                or proof.get("rule_slot")!=action
                or proof.get("head")!=ptol[head]
                or sorted(proof.get("premises",[]))!=sorted(ptol[p] for p in pre)):
                raise ValueError("host-verified Horn proof evidence changed")
            state.add(head)
            if goal in state:
                ended=True
        if step["after"]!=logical(state):
            raise ValueError("physical Horn post-state changed")
    solved=goal in state
    if (row["final"]!=logical(state) or row["solved"]!=solved
        or row["false_accepted_proof"] or (d is None and solved)):
        raise ValueError("false accepted Horn proof or terminal state")
    is_stop=bool(row["actions"] and row["actions"][-1]["action"]==gym.STOP)
    if (row["initial_stop"]!=bool(row["actions"] and
            row["actions"][0]["action"]==gym.STOP)
        or row["premature_stop"]!=bool(d is not None and not solved and is_stop)
        or row["model_nodes"]!=nodes
        or row["symbolic_expansions"]!=expansions
        or row["symbolic_rule_examinations"]!=examinations):
        raise ValueError("STOP or model/symbolic inference work accounting mismatch")
    return nodes


def audit(seed,zip_dir,folder,smoke=False,replays=2):
    folder=Path(folder)
    rec=json.loads((folder/"evaluation.json").read_text())
    raw=(folder/"episodes.jsonl").read_bytes()
    rootraw=(folder/"root_gaps.jsonl").read_bytes()
    if (rec["experiment"]!="AF-27" or rec["seed"]!=seed or rec["smoke"]!=smoke
        or rec["all_arms"]!=list(ex.ARMS) or rec["training_updates"]!=0
        or rec["source_hashes"]!=ex.source_hashes()
        or rec["protocol_sha256"]!=ex.sha((ex.HERE/"PROTOCOL.json").read_bytes())
        or rec["structural_manifest_sha256"]!=gym.manifest()["manifest_sha256"]
        or rec["episodes_sha256"]!=ex.sha(raw)
        or rec["root_gaps_sha256"]!=ex.sha(rootraw)):
        raise ValueError("registered source/episode root-gap hash mismatch")
    rows=[json.loads(s) for s in raw.splitlines()]
    gaps=[json.loads(s) for s in rootraw.splitlines()]
    roster=4 if smoke else 192
    expected=roster*len(gym.SPLITS)*len(ex.ARMS)
    if rec["episode_count"]!=expected or len(rows)!=expected:
        raise ValueError("incomplete new-grammar proof episodes")
    index={}
    for row in rows:
        key=(row["split"],row["mode"],row["case_id"])
        if key in index:
            raise ValueError("duplicate new-grammar episode")
        index[key]=row
    verified_nodes=0
    for split in gym.SPLITS:
        cases=gym.cases_for(split)
        if smoke:
            cases=([c for c in cases if c.designed_positive][:2]+
                   [c for c in cases if not c.designed_positive][:2])
        for c in cases:
            for mode in ex.ARMS:
                row=index.get((split,mode,c.id))
                if row is None:
                    raise ValueError("missing scored AF27 case")
                verified_nodes+=validate_episode(c,row)
        for mode in ex.ARMS:
            group=[index[(split,mode,c.id)] for c in cases]
            computed=ex.summarize(group)
            for key in ("cases","provable","unprovable","solved",
                        "initial_stop_provable","premature_stop_provable",
                        "correct_initial_stop_unprovable",
                        "false_accepted_proofs","neural_model_node_evaluations",
                        "symbolic_node_expansions","symbolic_rule_examinations"):
                if computed[key]!=rec["scores"][split][mode][key]:
                    raise ValueError("tampered original scored metric "+key)
    root_index={(row["split"],row["case_id"]):row for row in gaps}
    if len(root_index)!=len(gaps) or len(gaps)!=rec["root_gap_count"] or (
            len(gaps)!=(4 if smoke else 80)*len(gym.SPLITS)):
        raise ValueError("incorrect unique provable root-gap roster")
    for split in gym.SPLITS:
        cases=gym.cases_for(split)
        if smoke:
            cases=([c for c in cases if c.designed_positive][:2]+
                   [c for c in cases if not c.designed_positive][:2])
        for c in cases:
            if not c.designed_positive:
                continue
            target=root_index.get((split,c.id))
            control=index[(split,"dd",c.id)]
            if not target or not control["actions"]:
                raise ValueError("missing Direct root for independent gap analysis")
            first=control["actions"][0]
            regap=gap.evaluate_root_gap(
                c,first["source_scores"],tuple(first["enabled"]),
                first["plan_trace"],gym.optimal_actions(c,gym.initial_facts(c))
            )
            if (target["root_rule_gap"]!=regap or
                    target["model_node_count"]!=
                    first["plan_trace"]["nodes_forward_count"] or
                    target["truth_used_at_inference"]):
                raise ValueError("root-gap oracle separation or min-plus replay failed")
    model,provenance=original.load_frozen_original(zip_dir,seed)
    if provenance!=rec["original_af23_checkpoint"]:
        raise ValueError("exact original checkpoint provenance mismatch")
    replayed=0
    near=0
    for split in gym.SPLITS:
        cases=gym.cases_for(split)
        if smoke:
            cases=([c for c in cases if c.designed_positive][:2]+
                   [c for c in cases if not c.designed_positive][:2])
        for c in cases[:replays]:
            for arm in ex.NEURAL:
                episode=index[(split,arm,c.id)]
                if not episode["actions"]:
                    continue
                step=episode["actions"][0]
                fresh=transfer_planner.decisions(
                    model,[c],[frozenset(step["before"])]
                )[0]
                for source in ("direct","vi2"):
                    saved=step["source_scores"][source]
                    trial=fresh["scores"][source]
                    if len(saved)!=len(trial) or any(
                            abs(a-b)>1e-5 for a,b in zip(saved,trial)):
                        raise ValueError("actual checkpoint forward score differs from receipt")
                chosen=factor.choose(
                    fresh["scores"],tuple(step["enabled"]),arm,fresh["selected"]
                )["chosen"]
                if chosen!=step["action"]:
                    native=factor.GATE[arm]
                    rank=factor.RANK[arm]
                    scores=fresh["scores"]
                    legal=tuple(step["enabled"])
                    if not legal:
                        raise ValueError("empty legal mask must have STOP")
                    if chosen==gym.STOP or step["action"]==gym.STOP:
                        chosen_score=scores[native] if native!="never" else scores[rank]
                        distance=abs(chosen_score[gym.STOP]-
                                     max(chosen_score[a] for a in legal))
                    else:
                        chosen_score=scores[rank]
                        distance=max(chosen_score[a] for a in legal) - (
                            chosen_score[step["action"]] if step["action"] in legal
                            else float("-inf"))
                    if distance>2e-5:
                        raise ValueError("frozen checkpoint action replay differs beyond near-tie tolerance")
                    near+=1
                replayed+=1
    audit_result={
        "experiment":"AF-27","seed":seed,"smoke":smoke,
        "verified":True,
        "audited_episodes":len(rows),
        "audited_neural_nodes":verified_nodes,
        "root_gap_rows_checked":len(gaps),
        "sampled_original_torch_forwards":replayed,
        "sampled_near_tie_action_replays":near,
        "false_accepted_proofs":0,
        "source_hashes":ex.source_hashes(),
        "original_af23_checkpoint":provenance,
        "evaluation_sha256":ex.sha((folder/"evaluation.json").read_bytes()),
        "episodes_sha256":ex.sha(raw),
        "root_gaps_sha256":ex.sha(rootraw),
        "manifest_sha256":gym.manifest()["manifest_sha256"],
    }
    (folder/"independent_audit.json").write_text(
        json.dumps(audit_result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    )
    return audit_result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",required=True,type=int,choices=gym.SEEDS)
    p.add_argument("--original-zip-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    result=audit(a.seed,a.original_zip_dir,a.output_dir,a.smoke)
    print(json.dumps({"seed":a.seed,"audited":result["audited_episodes"],
                      "root_gap_rows":result["root_gap_rows_checked"],
                      "checkpoint_replays":result["sampled_original_torch_forwards"]}))


if __name__=="__main__":
    main()
