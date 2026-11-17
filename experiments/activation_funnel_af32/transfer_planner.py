"""AF-32 original AF-23 Direct/VI1/VI2, on new host-legal Horn states.

The exact original AF-23 min-plus score_from_nodes is imported unchanged.
Only the source of public Horn graph/state data changes. All three policies
share a single frozen 6036-parameter source-pinned checkpoint per seed.
"""
from __future__ import annotations

from collections import deque
import torch

from experiments.activation_funnel_af23 import branching_planner as parent
from . import gym

MODES = parent.MODES


def state_graph(c, start):
    root = frozenset(start)
    levels = {root: 0}
    todo = deque([root])
    edges = {}
    while todo:
        state = todo.popleft()
        depth = levels[state]
        edges[state] = {}
        if gym.GOAL in state or depth == 2:
            continue
        for rule in gym.enabled(c, state):
            next_state, proof = gym.sound_fire(c, state, rule)
            if proof is None:
                raise AssertionError("non-STOP host action lacks a rule proof")
            edges[state][rule] = next_state
            if next_state not in levels:
                levels[next_state] = depth + 1
                todo.append(next_state)
    return tuple(levels), edges


def decisions(model, cases, states, batch_size=128):
    if not cases or len(cases) != len(states):
        raise ValueError("nonempty matched AF32 inference roster required")
    all_cases, all_states, groups = [], [], []
    for c, known in zip(cases, states):
        nodes, edges = state_graph(c, known)
        start = len(all_cases)
        all_cases.extend([c] * len(nodes))
        all_states.extend(nodes)
        groups.append((nodes, edges, start))
    neural = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(all_cases), batch_size):
            group_cases = all_cases[begin:begin+batch_size]
            group_states = all_states[begin:begin+batch_size]
            x, bits, _ = gym.observations(group_cases, group_states)
            raw, distance, reach = model.full(x, bits)
            neural.extend(
                (raw[i].tolist(), float(distance[i]), float(reach[i]))
                for i in range(len(group_cases))
            )
    results = []
    for c, (nodes, edges, start) in zip(cases, groups):
        values = {s: neural[start+i] for i,s in enumerate(nodes)}
        scores, chosen, receipt = parent.score_from_nodes(nodes[0], edges, values)
        results.append({
            "scores": scores, "selected": chosen, "trace": receipt,
            "host_graph_sha256": gym.digest(gym.public_features(c)),
        })
    return results
