"""AF-30 independent per-step audit from physical incidence bitsets.

Freshly recomputes TRAIN linear weights and every HELD episode action against
the public physical graph. A FAIL is not a missing or inconclusive audit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from experiments.activation_funnel_af24 import symbolic
from . import evaluate, features, gym, ranker


def bytes_sha(raw):
    return hashlib.sha256(raw).hexdigest()


def public_rules(pub):
    return {
        int(slot):(sum(1<<int(p) for p in prem), 1<<int(head))
        for slot, prem, head in pub["rules_physical"]
    }


def bitset_legal(rules, state):
    return tuple(sorted(
        s for s,(prem,head) in rules.items()
        if prem & state == prem and not (head & state)
    ))


def bitset_fixedpoint(rules, state):
    while True:
        nxt=state
        for prem, head in rules.values():
            if prem & state == prem:
                nxt|=head
        if nxt == state:
            return state
        state = nxt


def _independent_action(pub, state, arm, model, seed, step):
    vectors, _ = features.extract(pub,state)
    if not vectors:
        return gym.STOP
    if arm == "RANK_CERT" or arm == "UNTRAINED_CERT":
        weight_key = "weights" if arm == "RANK_CERT" else "initial_weights"
        w = model[weight_key]
        def raw(slot):
            vec = vectors[slot]
            return sum(w[j]*max(-8.,min(8.,(
                vec[j]-model["mean"][j])/model["std"][j]))
                       for j in range(8))
        return min(vectors,key=lambda a:(-raw(a),a))
    if arm == "PATH_CERT":
        return min(vectors,key=lambda a:(-vectors[a][1],a))
    if arm == "ANCESTRY_CERT":
        return min(vectors,key=lambda a:(-vectors[a][0],a))
    if arm == "FIXED_CERT":
        return min(vectors)
    if arm == "RANDOM_CERT":
        rng=random.Random(int(gym.digest(
            ["AF30_RANDOM_ACTION",pub,sorted(state),seed,step]
        )[:16],16))
        return rng.choice(sorted(vectors))
    raise ValueError("unregistered independent audit policy")


def audit_one(case,row,model,seed):
    pub = gym.public_features(case)
    if row["case_id"] != case.id or row["pair_id"] != case.pair_id:
        raise AssertionError("case identity/source mismatch")
    if row["swap_position"] != case.swap_position or row["split"] != case.split:
        raise AssertionError("source swap provenance mismatch")
    if gym.digest(pub) != row["public_graph_sha256"]:
        raise AssertionError("source public graph SHA mismatch")
    physical_d = gym.public_bitset_bfs(pub)
    if physical_d != row["shortest_steps_oracle"]:
        raise AssertionError("independent bitset shortest result mismatch")
    if (physical_d is not None) != row["designed_reachable"]:
        raise AssertionError("source ground truth mislabeled")
    budget = physical_d+1 if physical_d is not None else 2
    if row["budget_oracle"] != budget:
        raise AssertionError("source proof budget mismatch")
    rules=public_rules(pub)
    state=sum(1<<p for p in pub["initial_physical"])
    goalbit=1<<pub["goal_slot"]
    exact_reach=bool(bitset_fixedpoint(rules,state)&goalbit)
    certificate=row["arm"] in evaluate.CERT_ARMS
    if certificate and row["certificate_reachable"] != exact_reach:
        raise AssertionError("source public negative certificate incorrect")
    if not certificate and row["certificate_reachable"] is not None:
        raise AssertionError("symbolic arm received a certificate")
    if len(row["history"]) > budget:
        raise AssertionError("policy exceeded scorer-only proof budget")
    for step,h in enumerate(row["history"]):
        if step != h["step"]:
            raise AssertionError("episode action numbering changed")
        logical_public=frozenset(i for i in range(gym.ATOMS) if state & (1<<i))
        if sorted(logical_public) != h["before_physical"]:
            raise AssertionError("source before-state physical mismatch")
        legal=bitset_legal(rules,state)
        if h["enabled_physical"] != list(legal):
            raise AssertionError("source enabled physical actions mismatch")
        action=h["action"]
        if certificate:
            if not exact_reach:
                expected=gym.STOP
            else:
                expected=_independent_action(pub,logical_public,row["arm"],
                                             model,seed,step)
        else:
            result=symbolic.choose(
                symbolic.PublicHorn.from_public_features(pub),
                logical_public, evaluate.SYM_MAP[row["arm"]],seed,
                gym.digest(pub),step)
            expected=gym.STOP if result["action"] is None else result["action"]
        if action != expected:
            raise AssertionError(f"source controller action drift: {action} != {expected}")
        if action == gym.STOP:
            if h["proof"] is not None:
                raise AssertionError("STOP supplied a forged proof")
        else:
            if action not in legal:
                raise AssertionError("source policy chose illegal Horn rule")
            pre,head=rules[action]
            if pre & state != pre or head & state:
                raise AssertionError("invalid independent physical transition")
            state|=head
            if h["proof"] is None or h["proof"]["slot"] != action:
                raise AssertionError("host proof witness missing or wrong")
        after=sorted(i for i in range(gym.ATOMS) if state & (1<<i))
        if h["after_physical"] != after:
            raise AssertionError("physical consequence inconsistent")
        if action == gym.STOP or state & goalbit:
            if step+1 != len(row["history"]):
                raise AssertionError("actions after terminal state")
            break
    solved=bool(state&goalbit)
    if solved != row["solved"] or (
        not case.reachable and solved
    ) != row["false_accepted_proof"]:
        raise AssertionError("false acceptance or completed proof mismatch")
    if bool(row["history"] and row["history"][0]["action"]==gym.STOP) != row["initial_stop"]:
        raise AssertionError("root STOP mismatch")
    total=row["certificate_work"]["rule_examinations"]+sum(
        h["work"].get("feature_rule_examinations",0)+
        h["work"].get("symbolic_rule_examinations",0)
        for h in row["history"])
    if total != row["total_rule_examinations"]:
        raise AssertionError("model/symbolic work ledger mismatch")
    return solved


def audit(seed,out):
    if seed not in ranker.SEEDS:
        raise ValueError("AF30 audit seed not registered")
    out=Path(out)
    receipt=json.loads((out/"evaluation.json").read_text())
    model_raw=(out/"model.json").read_bytes()
    raw=(out/"episodes.jsonl").read_bytes()
    if (receipt["model_sha256"]!=bytes_sha(model_raw) or
        receipt["episodes_sha256"]!=bytes_sha(raw)):
        raise AssertionError("source artifact SHA-256 integrity failure")
    model=json.loads(model_raw)
    fresh=ranker.fit(seed)
    if model != fresh or model["seed"] != seed or model["updates"] != 500:
        raise AssertionError("TRAIN-only deterministic source refit failure")
    expected={c.id:c for split in evaluate.SPLITS for c in gym.cases_for(split)}
    rows=[json.loads(s) for s in raw.decode().splitlines()]
    if len(rows)!=len(expected)*len(evaluate.ARMS):
        raise AssertionError("AF30 heldout complete-case audit missing episodes")
    keyed=set()
    by_arm=defaultdict(list)
    for r in rows:
        key=(r["case_id"],r["arm"])
        if key in keyed or r["case_id"] not in expected or r["arm"] not in evaluate.ARMS:
            raise AssertionError("duplicate or unseen source episode")
        keyed.add(key)
        if r["seed"]!=seed or r["experiment"]!="AF-30":
            raise AssertionError("foreign experiment source row")
        audit_one(expected[r["case_id"]],r,model,seed)
        by_arm[(r["split"],r["arm"])].append(r)
    rebuilt={}
    for split in evaluate.SPLITS:
        rebuilt[split]={arm:evaluate.summarize(by_arm[(split,arm)])
                        for arm in evaluate.ARMS}
        for arm in evaluate.ARMS:
            # Wall-clock is immutable receipt, recompute all counts exactly.
            check=rebuilt[split][arm]
            stored=receipt["scores"][split][arm]
            for k in check:
                if k!="wall_ns" and check[k]!=stored[k]:
                    raise AssertionError(f"score audit mismatch {split}/{arm}/{k}")
    gate=evaluate.gate(rebuilt)
    if gate!=receipt["registered_seed_gate"]:
        raise AssertionError("independent registered gate aggregation mismatch")
    result={
        "experiment":"AF-30","seed":seed,
        "source_manifest_sha256":gym.manifest()["manifest_sha256"],
        "model_sha256":bytes_sha(model_raw),
        "episodes_sha256":bytes_sha(raw),
        "independently_replayed_episodes":len(rows),
        "fresh_train_refit":True,
        "all_public_physical_transitions_valid":True,
        "all_action_decisions_recomputed":True,
        "all_scores_independently_aggregated":True,
        "no_false_accepted_proofs":all(
            r["false_accepted_proof"]==False for r in rows),
        "registered_seed_gate":gate,
        "science_verdict":"SOURCE_VERIFIED_AWAITING_EIGHT_SEEDS",
    }
    if not result["no_false_accepted_proofs"]:
        raise AssertionError("AF30 policy accepted a nonexistent Horn proof")
    if result["source_manifest_sha256"]!=receipt["manifest_sha256"]:
        raise AssertionError("source manifest drift")
    (out/"audit.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,required=True,choices=ranker.SEEDS)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    r=audit(args.seed,args.out)
    print(json.dumps({"seed":args.seed,
                      "verified_episodes":r["independently_replayed_episodes"],
                      "seed_gate":r["registered_seed_gate"]}))


if __name__=="__main__":
    main()
