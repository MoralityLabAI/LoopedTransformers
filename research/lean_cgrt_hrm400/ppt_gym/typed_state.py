"""Public Lean Horn state encodings for matched LSA-02C CPU comparisons.

Only the public state packet and public candidate menu can influence the text.
No Lean search oracle, theorem generator metadata, labels or proof outcomes
enter this transformation. The typed view uses a bounded dependency walk over
declared hypothesis types, prioritizing immediate candidate prerequisites.
"""
from __future__ import annotations

from collections import deque
import json
import re
from typing import Sequence

SCHEMA = "lsa02c.typed_public_state.v1"
VIEWS = ("raw", "typed")
_ATOM = re.compile(r"P[0-9]+$")
_HYP = re.compile(r"[hr][0-9]{1,4}$")


def _formula(value: dict) -> str:
    if not isinstance(value, dict):
        raise ValueError("public formula must be a Lean type object")
    if set(value) == {"a"} and isinstance(value["a"], str) and _ATOM.fullmatch(value["a"]):
        return value["a"]
    if set(value) == {"and"}:
        args = value["and"]
        if isinstance(args, list) and len(args) == 2:
            return "(" + "&".join(_formula(item) for item in args) + ")"
    if set(value) == {"imp"}:
        pair = value["imp"]
        if isinstance(pair, list) and len(pair) == 2:
            return _formula(pair[0]) + ">" + _formula(pair[1])
    raise ValueError("unsupported Lean public formula type")


def _rule(ty: dict) -> tuple[tuple[str, ...], str]:
    antecedents = []
    while isinstance(ty, dict) and set(ty) == {"imp"}:
        parts = ty["imp"]
        if not isinstance(parts, list) or len(parts) != 2:
            raise ValueError("invalid public hypothesis implication")
        antecedents.append(_formula(parts[0]))
        ty = parts[1]
    return tuple(antecedents), _formula(ty)


def _public(packet: str) -> tuple[list[dict], int, tuple[str, ...]]:
    try:
        row = json.loads(packet)
    except (ValueError, TypeError) as exc:
        raise ValueError("public Lean packet is not JSON") from exc
    if not isinstance(row, dict) or set(row) != {
        "state", "failed_choices", "checks_remaining",
    }:
        raise ValueError("public Lean packet shape changed")
    state = row["state"]
    if not isinstance(state, dict) or set(state) != {"goals"}:
        raise ValueError("public Lean state shape changed")
    goals = state["goals"]
    if not isinstance(goals, list) or not goals:
        raise ValueError("public Lean goal list is empty")
    for goal in goals:
        if not isinstance(goal, dict) or not {"target", "hypotheses"} <= set(goal):
            raise ValueError("Lean goal lacks target or local context")
        if not isinstance(goal["hypotheses"], list):
            raise ValueError("Lean hypothesis context is malformed")
        _formula(goal["target"])
    remaining = row["checks_remaining"]
    if type(remaining) is not int or remaining not in range(1, 9):
        raise ValueError("public Lean remaining-check budget is invalid")
    failures = row["failed_choices"]
    if not isinstance(failures, list) or any(
        not isinstance(s, str) for s in failures
    ):
        raise ValueError("public Lean failed-tactic list is invalid")
    return goals, remaining, tuple(failures)


def _type_entries(goal: dict) -> list[tuple[str, tuple[str, ...], str]]:
    entries = []
    names = set()
    for hyp in goal["hypotheses"]:
        if not isinstance(hyp, dict) or set(hyp) != {"name", "type"}:
            raise ValueError("malformed public Lean hypothesis")
        name, typ = hyp["name"], hyp["type"]
        if not isinstance(name, str) or not isinstance(typ, dict) or name in names:
            raise ValueError("malformed or duplicate public Lean hypothesis")
        names.add(name)
        if set(typ) == {"sort"} and typ["sort"] == "Prop" and _ATOM.fullmatch(name):
            continue
        if not _HYP.fullmatch(name):
            raise ValueError("unexpected non-Horn declaration")
        pre, head = _rule(typ)
        entries.append((name, pre, head))
    return entries


def _dependencies(entries: list[tuple[str, tuple[str, ...], str]],
                  target: str) -> dict[str, int]:
    dist = {target: 0}
    queue = deque([target])
    by_head: dict[str, list[tuple[str, ...]]] = {}
    for _, inputs, output in entries:
        by_head.setdefault(output, []).append(inputs)
    while queue:
        atom = queue.popleft()
        for premises in by_head.get(atom, []):
            for premise in premises:
                if premise not in dist:
                    dist[premise] = dist[atom] + 1
                    queue.append(premise)
    return dist


def encode_public_packet(
    packet: str,
    candidates: Sequence[str],
    *,
    view: str = "raw",
    max_bytes: int = 384,
) -> str:
    """Encode identical public evidence under a declared byte budget.

    Raw keeps the original JSON string; the downstream byte encoder applies
    its historical trailing max_bytes crop. Typed tries to fit the most
    target-relevant public rules into max_bytes without a byte-level crop.
    """
    if view not in VIEWS or type(max_bytes) is not int or not 32 <= max_bytes <= 384:
        raise ValueError("unknown public-state representation or byte budget")
    if not 1 <= len(candidates) <= 64 or len(set(candidates)) != len(candidates):
        raise ValueError("public action menu is empty, duplicated or oversized")
    if view == "raw":
        return packet
    goals, remaining, failed = _public(packet)
    goal = goals[0]
    goal_type = _formula(goal["target"])
    entries = _type_entries(goal)
    names = {name: (pre, head) for name, pre, head in entries}
    candidate_names = []
    for action in candidates:
        if action == "constructor":
            if set(goal["target"]) != {"and"}:
                raise ValueError("constructor is absent from the public goal")
            continue
        if not action.startswith("apply ") or action[6:] not in names:
            raise ValueError("candidate missing from public Lean hypothesis context")
        candidate_names.append(action[6:])
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("repeated public candidate hypothesis")

    # The complete current goal, candidate interface and remaining budget are
    # always retained; background rules are selected by public graph distance.
    tasks = ",".join(_formula(g["target"]) for g in goals)
    header = f"B{remaining}|G:{tasks}|"
    if failed:
        failed_names = ",".join(
            item[6:] if item.startswith("apply ") else item for item in failed
        )
        header += f"F:{failed_names}|"
    if "constructor" in candidates:
        header += "A:constructor|"
    if len(header.encode("utf-8")) > max_bytes:
        raise ValueError("public goal header alone exceeds the declared encoding budget")

    distance = _dependencies(entries, goal_type)
    by_name = {name: (pre, head) for name, pre, head in entries}
    priorities = list(dict.fromkeys(candidate_names))
    priorities += [
        name for name, pre, head in sorted(
            entries,
            key=lambda item: (distance.get(item[2], 100000), item[0]),
        )
        if name not in priorities
    ]
    output = header
    for name in priorities:
        pre, head = by_name[name]
        if pre:
            entry = name + ":" + "+".join(pre) + ">" + head + ";"
        else:
            entry = name + ":" + head + ";"
        if len((output + entry).encode("utf-8")) > max_bytes:
            continue
        output += entry
    if len(output.encode("utf-8")) > max_bytes:
        raise AssertionError("typed public proof packet exceeded byte budget")
    return output
