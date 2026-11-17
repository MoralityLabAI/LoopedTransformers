"""AF-33 independent public-physical replay of original AF23 Direct checkpoints.

All 1,152 source trajectories per checkpoint get exact bitset legality, STOP
source and proof audits; stratified on-policy decisions undergo genuine new
original-checkpoint Torch forwards under original deterministic settings.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import torch

from experiments.activation_funnel_af25 import evaluate as original
from . import direct, evaluate, gym, transfer_planner


def checksum(raw):
    return hashlib.sha256(raw).hexdigest()


def legal_bitset(rules,bits):
    return tuple(sorted(slot for slot,(prem,head) in rules.items()
                        if bits&prem==prem and not bits&head))


def public_bitset_closure(rules,bits):
    while True:
        future=bits
        for premise,head in rules.values():
            if premise&bits==premise:
                future |= head
        if future==bits:return bits
        bits=future


def verify_episode(case,row,seed):
    public=gym.public_features(case)
    if (row["seed"]!=seed or row["case_id"]!=case.id or
        row["pair_id"]!=case.pair_id or row["cycle_length"]!=case.cycle_length or
        row["swap_position"]!=case.swap_position or
        row["public_graph_sha256"]!=gym.digest(public) or
        row["arm"] not in direct.ARMS):
        raise AssertionError("original Direct public case/source mismatch")
    start=public["initial_physical"]
    bits=sum(1<<a for a in start)
    rules={slot:(sum(1<<a for a in prem),1<<head)
           for slot,prem,head in public["rules_physical"]}
    physical_d=gym.public_bitset_bfs(public)
    if physical_d!=row["shortest_steps_oracle"] or \
       (physical_d is not None)!=row["designed_reachable"]:
        raise AssertionError("independent original graph proof distance mismatch")
    if row["budget_mode"] not in evaluate.BUDGET_MODES:
        raise AssertionError("unregistered original Direct proof budget")
    budget=(physical_d if row["budget_mode"]=="strict" else physical_d+1) if physical_d is not None else 2
    if budget!=row["budget_oracle"] or len(row["history"])>budget:
        raise AssertionError("original source budget contradiction")
    exact_reachable=bool(public_bitset_closure(rules,bits)&(1<<public["goal_slot"]))
    if row["arm"]=="AF23_DIRECT_CERT":
        if row["certificate_reachable"]!=exact_reachable:
            raise AssertionError("source public certificate drift")
    elif row["certificate_reachable"] is not None:
        raise AssertionError("original neural-only arms leaked certificate")
    for index,h in enumerate(row["history"]):
        if index!=h["step"]:
            raise AssertionError("source Direct step numbering changed")
        before=sorted(i for i in range(gym.ATOMS) if bits&(1<<i))
        if before!=h["before_physical"]:
            raise AssertionError("original Direct physical root mismatch")
        legal=legal_bitset(rules,bits)
        if list(legal)!=h["enabled_physical"]:
            raise AssertionError("original Direct legal mask mismatch")
        action=h["action"]
        certstop=row["arm"]=="AF23_DIRECT_CERT" and not exact_reachable
        if certstop or not legal:
            expected=gym.STOP
            if h["source_scores"] is not None or h["plan_trace"] is not None:
                raise AssertionError("unneeded original checkpoint forward made")
        else:
            scores=h["source_scores"]
            if scores is None or len(scores)!=gym.RULES+1:
                raise AssertionError("original 15-score source decision absent")
            if h["plan_trace"]["nodes_forward_count"]!=h["forward_nodes"]:
                raise AssertionError("original recurrent forward work mismatch")
            source=h["source_selected"]
            if source is None or source["direct"]!=max(range(gym.RULES+1),key=lambda i:scores[i]):
                raise AssertionError("original source Direct max-selection inconsistent")
            for a in range(gym.RULES):
                if a not in legal and scores[a]!=-1e4:
                    raise AssertionError("unmasked illegal original action logit")
            if row["arm"]=="AF23_DIRECT_NATIVE":
                expected=source["direct"]
            else:
                expected=max(legal,key=lambda a:scores[a])
            trace=h["plan_trace"]
            root_raw=trace["root_raw"]
            if any(abs(root_raw[i]-scores[i])>1e-5
                   for i in (*legal,gym.STOP)):
                raise AssertionError("original neural score/root raw mismatch")
        if expected!=action:
            raise AssertionError("original Direct full-policy chosen action drift")
        if action!=gym.STOP:
            if action not in legal:
                raise AssertionError("original Direct chose illegal Horn rule")
            prem,head=rules[action]
            if prem&bits!=prem or bits&head:
                raise AssertionError("original Direct physical proof invalid")
            if h["proof"] is None or h["proof"]["slot"]!=action:
                raise AssertionError("source original fired rule lacks host witness")
            bits |= head
        elif h["proof"] is not None:
            raise AssertionError("source original STOP contains forged proof")
        after=sorted(i for i in range(gym.ATOMS) if bits&(1<<i))
        if after!=h["after_physical"]:
            raise AssertionError("source original proof step-after state drift")
        if action==gym.STOP or bits&(1<<public["goal_slot"]):
            if index+1!=len(row["history"]):
                raise AssertionError("original Direct episode continued after terminal state")
    solved=bool(bits&(1<<public["goal_slot"]))
    if solved!=row["solved"] or \
       (not case.reachable and solved)!=row["false_accepted_proof"]:
        raise AssertionError("original Direct false proof/positive count drift")
    root_off = bool(case.reachable and row["history"] and (
        row["history"][0]["action"]==gym.STOP or
        gym.public_bitset_bfs(public,row["history"][0]["after_physical"])!=physical_d-1
    ))
    if root_off!=row["root_first_off_optimal"]:
        raise AssertionError("independent original Direct first-off-optimal diagnostic drift")
    if bool(row["history"] and row["history"][0]["action"]==gym.STOP)!=row["initial_stop"]:
        raise AssertionError("source original initial STOP drift")
    total=sum(h["forward_nodes"] for h in row["history"])
    if total!=row["total_neural_forward_nodes"]:
        raise AssertionError("original checkpoint trace forward accounting drift")
    return total


def genuine_forward_samples(model, rows, case_by_id, max_groups=48):
    """Independent actual frozen-checkpoint forward vs serialized source logits."""
    samples=[]
    covered=set()
    for row in rows:
        case=case_by_id[row["case_id"]]
        for h in row["history"]:
            if h["source_scores"] is None:
                continue
            key=(row["split"],row["arm"],case.cycle_length,case.reachable,
                 h["step"]>0)
            if key in covered:
                continue
            covered.add(key)
            samples.append((case,h))
            if len(samples)>=max_groups:
                break
        if len(samples)>=max_groups:
            break
    if not samples:
        raise AssertionError("no genuine heldout forwards sampled")
    max_delta=0.
    for case,h in samples:
        perm,_=gym.physical_maps(case)
        inverse={a:i for i,a in enumerate(perm)}
        state=frozenset(inverse[x] for x in h["before_physical"])
        actual=transfer_planner.decisions(model,[case],[state])[0]
        source=h["source_scores"]
        observed=actual["scores"]["direct"]
        delta=max(abs(a-b) for a,b in zip(source,observed))
        max_delta=max(max_delta,delta)
        if delta>1e-4 or actual["selected"]!=h["source_selected"]:
            raise AssertionError(f"AF33 genuine original checkpoint forward drift {delta}")
    return len(samples),max_delta


def audit(seed,original_zip_dir,out):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out=Path(out)
    receipt=json.loads((out/"direct_evaluation.json").read_text())
    source=(out/"direct_episodes.jsonl").read_bytes()
    if receipt["smoke"] or receipt["seed"]!=seed or \
       checksum(source)!=receipt["direct_episodes_sha256"] or \
       receipt["original_checkpoint_parameter_updates"]!=0:
        raise AssertionError("original Direct source/integrity mismatch")
    rows=[json.loads(line) for line in source.decode().splitlines()]
    expected={c.id:c for split in evaluate.SPLITS
              for c in gym.cases_for(split)}
    if len(rows)!=len(expected)*len(direct.ARMS)*len(evaluate.BUDGET_MODES):
        raise AssertionError("incomplete AF33 original checkpoint source roster")
    seen=set(); work=0; per=defaultdict(list)
    for row in rows:
        identity=(row["case_id"],row["arm"],row["budget_mode"])
        if identity in seen or row["case_id"] not in expected or row["budget_mode"] not in evaluate.BUDGET_MODES:
            raise AssertionError("duplicated or foreign original Direct source case")
        seen.add(identity)
        c=expected[row["case_id"]]
        work+=verify_episode(c,row,seed)
        per[(evaluate.cohort_key(row["budget_mode"],row["split"]),row["arm"])].append(row)
    scores={evaluate.cohort_key(mode,split):{
        arm:direct.summarize(per[(evaluate.cohort_key(mode,split),arm)])
        for arm in direct.ARMS} for mode,split in evaluate.COHORTS}
    if scores!=receipt["scores"] or work!=receipt["original_model_forward_nodes"]:
        raise AssertionError("original model independent score/work aggregation drift")
    model,provenance=original.load_frozen_original(original_zip_dir,seed)
    if provenance!=receipt["original_frozen_AF23_checkpoint"]:
        raise AssertionError("AF23 original immutable checkpoint provenance drift")
    count,max_delta=genuine_forward_samples(model,rows,expected)
    result={
        "experiment":"AF-33","seed":seed,
        "source_manifest_sha256":gym.manifest()["manifest_sha256"],
        "original_checkpoint_provenance":provenance,
        "direct_episodes_sha256":checksum(source),
        "independent_physical_proof_episodes":len(rows),
        "independently_replayed_budget_modes":list(evaluate.BUDGET_MODES),
        "original_neural_nodes_from_receipt":work,
        "sampled_genuine_original_torch_forward_decisions":count,
        "maximum_original_score_absolute_drift":max_delta,
        "all_original_direct_actions_reconstructed":True,
        "all_original_host_transitions_replayed":True,
        "all_original_scores_reaggregated":True,
        "no_false_accepted_proofs":all(not r["false_accepted_proof"] for r in rows),
    }
    if not result["no_false_accepted_proofs"]:
        raise AssertionError("frozen original source accepted false Horn proof")
    if receipt["source_manifest_sha256"]!=result["source_manifest_sha256"]:
        raise AssertionError("original Direct AF33 source manifest mismatch")
    (out/"direct_audit.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--original-zip-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    r=audit(a.seed,a.original_zip_dir,a.out)
    print(json.dumps({
        "original_seed":a.seed,
        "independent_episodes":r["independent_physical_proof_episodes"],
        "genuine_checkpoint_replay":r["sampled_genuine_original_torch_forward_decisions"],
        "max_delta":r["maximum_original_score_absolute_drift"],
    }))


if __name__=="__main__":
    main()
