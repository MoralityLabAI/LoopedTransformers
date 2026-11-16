"""AF-23 Track B host-gated two-step Bellman scorer on branch-rich Horn cases.

The source proof oracle is absent from inference. All three neural arms use
one frozen ReadoutPolicy checkpoint per seed and the same public, host-certified
lookahead nodes. Their actual algorithmic work differs and is recorded.
"""
from __future__ import annotations
from collections import deque
import math
import torch
from . import branching_gym as gym

MODES = ("direct", "vi1", "vi2")
CAP = 16.
MAX_DEPTH = 2


def state_graph(c, start):
    root = frozenset(start)
    levels = {root: 0}
    todo = deque([root])
    edges = {}
    while todo:
        s = todo.popleft()
        depth = levels[s]
        edges[s] = {}
        if gym.GOAL in s or depth >= MAX_DEPTH:
            continue
        for rule in gym.enabled(c, s):
            successor, proof = gym.sound_fire(c, s, rule)
            if proof is None:
                raise ValueError("host-legal proof missing")
            edges[s][rule] = successor
            if successor not in levels:
                levels[successor] = depth + 1
                todo.append(successor)
    if any(depth > MAX_DEPTH for depth in levels.values()):
        raise AssertionError("two-step search overflow")
    return tuple(levels), edges


def score_from_nodes(root, edges, neural, cap=CAP):
    raw, distance, reach = neural[root]
    def values(s):
        if gym.GOAL in s:
            return 0., 0.
        _, d, r = neural[s]
        p = 1. / (1. + math.exp(-r)) if r >= 0 else math.exp(r) / (1. + math.exp(r))
        v0 = p * min(cap, max(0., max(d, 0.) + math.log1p(math.exp(-abs(d))))) + (1.-p) * cap
        stop = cap * p
        if not (math.isfinite(v0) and math.isfinite(stop)):
            raise ValueError("nonfinite neural Bellman boundary")
        return v0, stop
    leaf = {s: values(s)[0] for s in edges}
    halt = {s: values(s)[1] for s in edges}
    backup = {
        s: min([halt[s]] + [1. + leaf[t] for t in successors.values()])
        for s, successors in edges.items()
    }
    scores = {m: [-1e4] * (gym.STOP + 1) for m in MODES}
    for action, next_state in edges[root].items():
        scores["direct"][action] = float(raw[action])
        scores["vi1"][action] = -(1. + leaf[next_state])
        scores["vi2"][action] = -(1. + backup[next_state])
    scores["direct"][gym.STOP] = float(raw[gym.STOP])
    scores["vi1"][gym.STOP] = -halt[root]
    scores["vi2"][gym.STOP] = -halt[root]
    selected = {
        m: max(range(gym.STOP + 1), key=lambda a: scores[m][a])
        if edges[root] else gym.STOP
        for m in MODES
    }
    return scores, selected, {
        "nodes": [
            {"state": sorted(s),
             "edges": {str(a): sorted(t) for a, t in edges[s].items()},
             "value0": leaf[s], "stop_cost": halt[s],
             "distance_logit": neural[s][1],
             "reach_logit": neural[s][2]}
            for s in edges
        ],
        "root_raw": list(map(float, raw)),
        "nodes_forward_count": len(edges),
    }


def decisions(model, cases, states, batch_size=128):
    if not cases or len(cases) != len(states):
        raise ValueError("nonempty matched inference input required")
    all_cases, all_states, groups = [], [], []
    for c, known in zip(cases, states):
        nodes, edges = state_graph(c, known)
        i = len(all_cases)
        all_cases.extend([c] * len(nodes))
        all_states.extend(nodes)
        groups.append((nodes, edges, i))
    outputs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_cases), batch_size):
            cs = all_cases[i:i+batch_size]
            ss = all_states[i:i+batch_size]
            x, bits, _ = gym.observations(cs, ss)
            raw, distance, reach = model.full(x, bits)
            outputs.extend((raw[j].tolist(), float(distance[j]), float(reach[j]))
                           for j in range(len(cs)))
    result = []
    for c, (nodes, edges, start) in zip(cases, groups):
        neural = {s: outputs[start + i] for i, s in enumerate(nodes)}
        scores, selected, receipt = score_from_nodes(nodes[0], edges, neural)
        result.append({"scores": scores, "selected": selected, "trace": receipt})
    return result


def check_trace(c, start, receipt, scores):
    """Numerical/physical replay from logged public nodes, independent of Torch."""
    nodes, edges = state_graph(c, start)
    got = {tuple(n["state"]): n for n in receipt["nodes"]}
    if set(got) != {tuple(sorted(s)) for s in nodes} or receipt["nodes_forward_count"] != len(nodes):
        raise ValueError("missing or extra successor values")
    for s in nodes:
        saved = got[tuple(sorted(s))]
        if (saved["edges"] != {str(a): sorted(t) for a, t in edges[s].items()}):
            raise ValueError("public graph receipt mismatch")
    neural = {}
    for s in nodes:
        saved = got[tuple(sorted(s))]
        neural[s] = (None, saved["distance_logit"], saved["reach_logit"])
    root = nodes[0]
    neural[root] = (receipt["root_raw"], neural[root][1], neural[root][2])
    recomputed, chosen, _ = score_from_nodes(root, edges, neural)
    for mode, vector in scores.items():
        if mode not in MODES:
            raise ValueError("invalid scored mode")
        if any(abs(float(a) - float(b)) > 1e-5
               for a, b in zip(recomputed[mode], vector)):
            raise ValueError("Bellman numeric receipt mismatch")
    return chosen
