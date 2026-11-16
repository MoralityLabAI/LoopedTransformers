"""AF-23 source-registered STOP ablations on AF-22's saved plan ledgers.

Pure Python scorer: no Torch, training, BFS oracle or hidden labels.
Full rollout must call this scorer on a *fresh model forward at each visited
state*. Applying it only to stored AF-22 traces is retrospective first-step
analysis, never a complete counterfactual proof episode.
"""
from __future__ import annotations
import math

CAP = 16.0
STOP = 14
GOAL = 14
ARMS = ("root_tied_reach", "no_interior_stop",
        "root_tied_no_interior_stop")
AUDIT_ONLY = ("vi2_exact",)


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softplus(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def _state(seq):
    return tuple(seq)


def score(trace, root_state, arm):
    """Return action, full rule/STOP scores and interior STOP binding receipt.

    The physical rule-slot tie breaker is identical to the original AF-22
    planner: lowest numeric slot wins on equal costs. Only model outputs and
    public host-state/edge traces enter this computation.
    """
    if arm not in ARMS + AUDIT_ONLY:
        raise ValueError(f"unregistered AF23 arm: {arm}")
    if not isinstance(trace.get("nodes"), list) or not trace["nodes"]:
        raise ValueError("missing host successor nodes")
    root = _state(root_state)
    nodes = {}
    for raw in trace["nodes"]:
        s = _state(raw["state"])
        if s in nodes:
            raise ValueError("duplicate physical host-state ledger entry")
        if not set(("value0", "stop_cost", "reach_logit", "distance_logit", "edges")) <= raw.keys():
            raise ValueError("incomplete neural value receipt")
        nodes[s] = raw
    if root not in nodes:
        raise ValueError("root state absent from host graph")
    for s, raw in nodes.items():
        if any(_state(t) not in nodes for t in raw["edges"].values()):
            raise ValueError("dangling successor target in receipt")
        if any(not str(k).isdigit() or not 0 <= int(k) < STOP
               for k in raw["edges"]):
            raise ValueError("invalid physical rule-slot index")
    if len(trace.get("root_raw", [])) != STOP + 1:
        raise ValueError("malformed root rule/STOP logits")

    tied = arm in ("root_tied_reach", "root_tied_no_interior_stop")
    no_interior = arm in ("no_interior_stop", "root_tied_no_interior_stop")
    root_p = sigmoid(nodes[root]["reach_logit"])
    v0, cstop = {}, {}
    for s, n in nodes.items():
        if GOAL in s:
            v0[s] = 0.0
            cstop[s] = 0.0
        elif tied:
            p = root_p
            d = min(CAP, max(0.0, softplus(n["distance_logit"])))
            v0[s] = p * d + (1.0 - p) * CAP
            cstop[s] = CAP * p
        else:
            v0[s] = float(n["value0"])
            cstop[s] = float(n["stop_cost"])
        if not (math.isfinite(v0[s]) and math.isfinite(cstop[s])):
            raise ValueError("nonfinite learned value")

    root_nodes = nodes[root]["edges"]
    action_cost = {STOP: float(nodes[root]["stop_cost"])}
    binds = 0
    comparisons = 0
    for a, target in root_nodes.items():
        s = _state(target)
        onward = [1.0 + v0[_state(t)] for t in nodes[s]["edges"].values()]
        if GOAL in s:
            b1 = 0.0
        elif no_interior:
            # If no legal continuation remains, retain the learned leaf
            # boundary. Never fabricate a free STOP or an infinite edge.
            b1 = min(onward) if onward else v0[s]
        else:
            b1 = min(cstop[s], *onward)
            if onward:
                comparisons += 1
                if cstop[s] <= min(onward):
                    binds += 1
        action_cost[int(a)] = 1.0 + b1
    if len(action_cost) == 1:  # no legal rule, STOP is the only action
        chosen = STOP
    else:
        chosen = min(action_cost, key=lambda a: (action_cost[a], a))
    return {
        "arm": arm, "chosen": chosen,
        "scores": {str(a): -v for a, v in sorted(action_cost.items())},
        "node_count": len(nodes),
        "interior_stop_bind": binds,
        "interior_stop_compared": comparisons,
    }


def compare_original_vi2(trace, root_state, original_scores, tolerance=1e-5):
    """Tamper-aware independent numerical replay of the original VI-2 scores."""
    rec = score(trace, root_state, "vi2_exact")
    for slot, saved in enumerate(original_scores):
        if str(slot) in rec["scores"]:
            if abs(rec["scores"][str(slot)] - float(saved)) > tolerance:
                raise ValueError("original VI-2 action score failed replay")
        elif float(saved) > -1e3:
            raise ValueError("illegal action had an unexpectedly large score")
    return rec
