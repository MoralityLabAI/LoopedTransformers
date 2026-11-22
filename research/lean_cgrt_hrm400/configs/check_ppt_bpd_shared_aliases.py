"""Verify paired native proof budgets whose branch identity depends on a local definition."""

import argparse
from collections import deque
from dataclasses import asdict
import gzip
from itertools import product
import json
from pathlib import Path
import random

from configs.check_ppt_bpd_dependent_rewrites import BUDGET, SEED
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rewrite import BPD_REWRITE_FAMILY


SCHEMA = "ppt.dependent-rewrite-shared-alias-witness.v1"


def layouts():
    rng = random.Random(SEED + 9)
    flags = list(product((0, 1), (0, 1), (0, 1), (0, 1), range(9)))
    picked = rng.sample(flags, 36)
    return {"train": picked[:24], "dev": picked[24:]}


def world(split, shape, swapped):
    rng = random.Random(digest([SCHEMA, shape]))
    nats = [f"n{rng.randrange(1 << 48):012x}" for _ in range(10)]
    aliases = [f"u{rng.randrange(1 << 48):012x}" for _ in range(2)]
    f, g, l, r = [f"f{rng.randrange(1 << 48):012x}" for _ in range(4)]
    if len(set([*nats, *aliases, f, g, l, r])) != 16:
        raise ValueError("the shared-alias fixture collided on a variable name")
    a, b, c, d, s, t, p, q, tail, spur = nats
    swap_l, swap_r, reverse_decoy, reverse_alias, attach = shape
    short, long = (d, c) if reverse_decoy else (c, d)
    edges = [(a, s), (s, t), (t, c), (t, d), (b, p), (p, q), (q, short), (q, tail), (tail, long), (nats[attach], spur)]
    rng.shuffle(edges)
    rewrites = {f"h{i}": edge for i, edge in enumerate(edges, 4)}
    premises = []
    for middle, swap in ((l, swap_l), (r, swap_r)):
        output = "y x" if swap else "x y"
        premises.append(f"∀ (x y : Nat), {f} x y = {middle} {output}")
    for middle, swap, side in ((l, swap_l, 0), (r, swap_r, 1)):
        subject = ("x", "y")[side ^ swap]
        output = "y x" if swap else "x y"
        premises.append(f"∀ (x y : Nat), ({subject} = {c} ∧ {subject} = {d}) → {middle} x y = {g} {output}")
    premises.extend(f"{left} = {right}" for left, right in edges)
    order = aliases[::-1] if reverse_alias else aliases
    values = (b, a) if swapped else (a, b)
    definitions = dict(zip(aliases, values, strict=True))
    lets = "".join(f"let {name} : Nat := {definitions[name]}; " for name in order)
    args = " ".join(aliases)
    target = f"{lets}{f} {args} = {g} {args}"
    binders = [f"({' '.join(sorted(nats))} : Nat)", f"({f} {g} {l} {r} : Nat → Nat → Nat)",
               *(f"(h{i} : {phrase})" for i, phrase in enumerate(premises))]
    cluster = digest([SCHEMA, shape])
    case = SearchCase("loopt_dependent_shared_alias", split, BPD_REWRITE_FAMILY, digest([cluster, swapped]), cluster,
                      f"shared_alias_{cluster[:12]}_{int(swapped)}", " ".join(binders), target)
    return case, definitions, order, rewrites, (c, d), aliases


def suffix(subject, targets, rewrites, aliases):
    """Enumerate the registered local rewrite, normalization, split, and reflexivity states."""
    initial = tuple((subject, target) for target in targets)
    queue = deque([(True, initial, [])])
    visited = {(True, initial)}
    moves = [(f"rw [{name}]", *edge) for name, edge in rewrites.items()]
    moves += [(f"rw [← {name}]", *edge[::-1]) for name, edge in rewrites.items()]
    moves += [(f"dsimp only [{name}]", name, value) for name, value in aliases.items()]

    def reflexive(goal):
        a, b = goal
        return aliases.get(a, a) == aliases.get(b, b)

    while queue:
        joined, goals, path = queue.popleft()
        if not goals:
            return path
        successors = []
        if joined:
            successors.append(("constructor", False, goals))
        elif reflexive(goals[0]):
            successors.append(("rfl", False, goals[1:]))
        considered = goals if joined else goals[:1]
        for tactic, source, dest in moves:
            changed = tuple(tuple(dest if item == source else item for item in goal) for goal in considered)
            if changed == considered:
                continue
            if not joined:
                changed = (() if reflexive(changed[0]) else changed) + goals[1:]
            successors.append((tactic, joined, changed))
        for tactic, coupled, state in successors:
            key = coupled, state
            if key not in visited:
                queue.append((coupled, state, [*path, tactic]))
                visited.add(key)
    raise ValueError("the projected alias guard has no completion")


def scripts(definitions, rewrites, targets, aliases):
    result = {}
    for side, subject in enumerate(aliases):
        route = [f"rw [h{side}]", f"rw [h{side + 2}]"]
        result[route[0]] = route + suffix(subject, targets, rewrites, definitions)
    for alias in aliases:
        normalization = f"dsimp only [{alias}]"
        candidates = []
        for side, subject in enumerate(aliases):
            start = definitions[subject] if alias == subject else subject
            candidates.append([normalization, f"rw [h{side}]", f"rw [h{side + 2}]",
                               *suffix(start, targets, rewrites, definitions)])
        result[normalization] = min(candidates, key=len)
    return result


def replay(repl, case, script, *, capture):
    from ppt_gym.alias_rewrite import alias_menu
    from ppt_gym.live_lean import public_state, state_input

    repl.reset()
    state = repl.init(case.statement, pp=True)
    events = []
    for attempt, tactic in enumerate(script):
        packet, before = state_input(repl, state, (), max(0, BUDGET - attempt))
        menu = alias_menu(json.loads(packet)["state"])
        if tactic not in menu:
            raise ValueError(f"the alias reference tactic is absent from the public menu: {case.tid}, {tactic}")
        reply = repl.run(state["sid"], tactic)
        if not reply["ok"]:
            raise ValueError(f"Lean rejected an alias trial for {case.tid}: {tactic}: {reply.get('error', '')}")
        state = reply["state"]
        after = public_state(repl, state)
        event = {"tactic": tactic, "menu": menu, "before": before, "after": digest(after),
                 "targets": [goal["target"] for goal in after["goals"]]}
        if capture or attempt == 0:
            event["packet"] = packet
        events.append(event)
    kernel = repl.kernel(state["sid"])
    if events[-1]["targets"] or not kernel["ok"] or kernel["axioms"]:
        raise ValueError("the alias proof failed the native kernel check")
    return events


def audit(split, tree):
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import certify
    from ppt_gym.ram_campaign import configure

    configure(tree.threads, 103)
    rows = {}
    with LeanRepl(timeout=15) as repl:
        if repl.version != "4.19.0":
            raise ValueError("the alias cohort requires Lean 4.19.0")
        for shape in layouts()[split]:
            raw_states, feasible = [], []
            for swapped in (False, True):
                case, definitions, order, rewrites, targets, aliases = world(split, shape, swapped)
                plans = scripts(definitions, rewrites, targets, aliases)
                viable = {action for action, script in plans.items() if len(script) <= BUDGET}
                expected = {f"rw [h{int(swapped)}]", f"dsimp only [{aliases[int(swapped)]}]"}
                if viable != expected:
                    raise ValueError(f"the alias role lost its unique in-budget branch: {case.tid}, {[(a, len(s)) for a, s in plans.items()]}")
                paths = {}
                for action, script in plans.items():
                    start = tree.cpu_seconds()
                    events = replay(repl, case, script, capture=split == "train")
                    if tree.cpu_seconds() - start > 6 or events != replay(repl, case, script, capture=split == "train"):
                        raise ValueError("an alias proof changed on replay or exceeded its CPU budget")
                    if bool(events[BUDGET - 1]["targets"]) == (action in viable):
                        raise ValueError("Lean disagreed with the registered eight-check alias outcome")
                    cert = certify(case, {"live_kernel": True, "accepted_script": script}, aliases=order)
                    if not cert or not cert["passed"] or cert["axioms"]:
                        raise ValueError("the alias trial failed independent certification")
                    paths[action] = {"checks": len(script), "events": events, "certificate": cert}
                packet = paths["rw [h0]"]["events"][0]["packet"]
                current = json.loads(packet)["state"]
                raw_states.append(digest({"goals": [{key: value for key, value in goal.items() if key != "pp"}
                                                   for goal in current["goals"]],
                                          "menu": paths["rw [h0]"]["events"][0]["menu"]}))
                feasible.append(viable)
                rows[case.root] = {"case": asdict(case), "shape": shape, "swapped": swapped, "packet": packet,
                                   "raw_state_sha256": raw_states[-1], "viable_openings": sorted(viable), "paths": paths}
            if raw_states[0] != raw_states[1] or not feasible[0].isdisjoint(feasible[1]):
                raise ValueError("the swapped local definitions leaked into raw proof types or share an eight-check action")
    return {"schema": SCHEMA, "split": split, "lean": "4.19.0", "tactic_checks": BUDGET,
            "source_sha256": sha256_file(Path(__file__)), "rows": rows, "memory": tree.stats()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("split", choices=("train", "dev"))
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    tree = TreeBudget(2, 1)
    try:
        result = audit(args.split, tree)
        with args.output.open("xb") as handle:
            handle.write(gzip.compress(canonical(result).encode("utf-8"), mtime=0))
    finally:
        tree.close()
    print(f"Verified four live opening plans in {len(result['rows'])} {args.split} alias worlds.", flush=True)


if __name__ == "__main__":
    main()
