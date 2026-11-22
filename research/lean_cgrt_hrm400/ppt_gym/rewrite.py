"""Bounded left-hand-side rewrites using public local Nat equalities."""

import re


BPD_REWRITE_FAMILY = "bpd-local-equality-rewrite"
NAME = r"[A-Za-z_][A-Za-z_0-9]*"
EQUALITY = re.compile(rf"({NAME}) = ({NAME})")
HYPOTHESIS = re.compile(r"[hr][0-9]{1,4}")
REWRITE = re.compile(r"rw \[(?:← )?([hr][0-9]{1,4})\]")


def rewrite_problem(goal):
    """Return the public goal and both directions of every local equality."""
    local_nat = {hyp["name"] for hyp in goal["hypotheses"] if hyp["type"] == {"raw": "Nat"}}
    target = EQUALITY.fullmatch(goal["target"].get("raw", ""))
    if not target or not set(target.groups()) <= local_nat:
        return None
    edges = []
    for hyp in goal["hypotheses"]:
        name = hyp["name"]
        equality = EQUALITY.fullmatch(hyp["type"].get("raw", ""))
        if not HYPOTHESIS.fullmatch(name) or not equality or not set(equality.groups()) <= local_nat:
            continue
        left, right = equality.groups()
        if left != right:
            edges.extend(((f"rw [{name}]", left, right), (f"rw [← {name}]", right, left)))
    return *target.groups(), edges


def rewrite_menu(goal):
    problem = rewrite_problem(goal)
    if problem is None:
        return ()
    left, right, edges = problem
    return ("rfl",) if left == right else tuple(action for action, start, _ in edges if start == left)


def rewrite_hypothesis(action):
    match = REWRITE.fullmatch(action)
    return match[1] if match else None


def remember_rewrite_target(state, visited):
    if state["goals"]:
        problem = rewrite_problem(state["goals"][0])
        if problem is None:
            raise ValueError("the rewrite proof must stay in the local equality grammar")
        visited.add(problem[:2])


def unvisited_rewrites(goal, menu, visited):
    if len(menu) < 2:
        return None
    _, target, edges = rewrite_problem(goal)
    endpoints = {action: right for action, _, right in edges}
    choices = [i for i, action in enumerate(menu) if (endpoints[action], target) not in visited]
    return choices if 0 < len(choices) < len(menu) else None


def restrict_rewrite_scores(logits, choices):
    if choices is None:
        return logits
    keep = logits.new_tensor([i in choices for i in range(len(logits))]).bool()
    return logits.masked_fill(~keep, float("-inf"))
