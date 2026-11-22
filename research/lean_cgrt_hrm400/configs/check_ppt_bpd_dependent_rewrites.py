"""Certify BPD-style rewrite branches whose guards depend on argument binding."""

import argparse
from dataclasses import asdict, replace
import gzip
from itertools import combinations
from math import inf
from pathlib import Path
import random

from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rewrite import BPD_REWRITE_FAMILY, EQUALITY, rewrite_menu


SEED = 119421817
SCHEMA = "ppt.dependent-rewrite-witness.v1"
SPUR_SCHEMA = "ppt.dependent-rewrite-spur-witness.v1"
ROUTING_SCHEMA = "ppt.dependent-rewrite-routing-witness.v1"
ROUTING_CONFIRMATION_SCHEMA = "ppt.dependent-rewrite-routing-confirmation-witness.v1"
EFFORT_SCHEMA = "ppt.dependent-rewrite-serial-effort-witness.v1"
BUDGET = 8


def shape_pool():
    choices = {}
    for length in range(3, 7):
        masks = [mask for mask in range(1, 1 << length) if mask.bit_count() % 2 == 0]
        choices[length] = [(length, first, second, left, right)
                           for first, second in combinations(range(1, length), 2)
                           for left in masks for right in masks if left != right]
    return choices


def layouts():
    rng = random.Random(SEED)
    choices = shape_pool()
    dev = [*rng.sample(choices[5], 4), *rng.sample(choices[6], 4)]
    train = rng.sample([shape for length in (3, 4, 5) for shape in choices[length] if shape not in dev], 24)
    return {"train": train, "dev": dev}


def spur_layouts():
    rng = random.Random(SEED + 1)
    old = {shape for split in layouts().values() for shape in split}
    choices = {length: [shape for shape in group if shape not in old] for length, group in shape_pool().items()}
    dev = [*rng.sample(choices[5], 4), *rng.sample(choices[6], 4)]
    train = rng.sample([shape for length in (3, 4, 5) for shape in choices[length] if shape not in dev], 24)
    return {"train": train, "dev": dev}


def routing_layouts(*, confirmation=False):
    rng = random.Random(SEED + (3 if confirmation else 2))
    used = {shape for study in (layouts(), spur_layouts()) for split in study.values() for shape in split}
    if confirmation:
        used.update(routing_layouts())
    count = 6 if confirmation else 4
    choices = shape_pool()
    return [shape for length in (5, 6) for shape in rng.sample([s for s in choices[length] if s not in used], count)]


def effort_layouts():
    rng = random.Random(SEED + 4)
    used = {shape for study in (layouts(), spur_layouts()) for split in study.values() for shape in split}
    used.update(routing_layouts())
    used.update(routing_layouts(confirmation=True))
    choices = shape_pool()
    chosen = [rng.sample([shape for shape in choices[length] if shape not in used], 18) for length in (4, 5)]
    return {"train": [shape for group in chosen for shape in group[:12]],
            "dev": [shape for group in chosen for shape in group[12:]]}


def world(split, shape, winner, *, serial_effort=False):
    length, first_guard, second_guard, *masks = shape
    rng = random.Random(digest([SEED, shape]))
    used = {"n": [], "f": []}
    rules, routes = [], {}

    def fresh(kind):
        value = f"{kind}{rng.randrange(1 << 40):010x}"
        if value in used[kind]:
            raise ValueError("dependent proof symbols collided")
        used[kind].append(value)
        return value

    def premise(formula):
        name = f"h{len(rules)}"
        rules.append(formula)
        return f"rw [{name}]"

    a, b, c, d = [fresh("n") for _ in range(4)]
    start, target = fresh("f"), fresh("f")
    lanes = [[start, *(fresh("f") for _ in range(length - 1)), target] for _ in range(2)]
    demands = [a, b]
    if rng.randrange(2):
        demands.reverse()
    scripts, permutations = [[], []], [[a, b], [a, b]]
    for step in range(length):
        for side, lane in enumerate(lanes):
            permutation = permutations[side]
            guard = ""
            if step in (first_guard, second_guard):
                variable = ("x", "y")[permutation.index(demands[side])]
                anchor = c if step == first_guard else d
                guard = f"{variable} = {anchor} → "
            swapped = bool(masks[side] & (1 << step))
            output = "y x" if swapped else "x y"
            scripts[side].append(premise(f"∀ (x y : Nat), {guard}{lane[step]} x y = {lane[step + 1]} {output}"))
            if swapped:
                permutation.reverse()
    if permutations != [[a, b], [a, b]]:
        raise ValueError("the two term routes must finish at the same arguments")
    short = ((BUDGET - length) // 2, (BUDGET - length + 1) // 2)
    long = (1, BUDGET - length)
    if serial_effort:
        longer_guard = (BUDGET - length) // 2 + 1
        shorter_guard = BUDGET - length - longer_guard
        short, long = (shorter_guard, longer_guard), (shorter_guard + 1, longer_guard)
    for side, term in enumerate(demands):
        for anchor, cost in zip((c, d), short if side == winner else long, strict=True):
            nodes = [term, *(fresh("n") for _ in range(cost - 1)), anchor]
            for before, after in zip(nodes, nodes[1:]):
                routes[(before, anchor)] = premise(f"{before} = {after}")

    bindings = [f"({' '.join(sorted(used['n']))} : Nat)", f"({' '.join(sorted(used['f']))} : Nat → Nat → Nat)"]
    bindings.extend(f"(h{i} : {rule})" for i, rule in enumerate(rules))
    cluster = digest([SCHEMA, shape])
    case = SearchCase("loopt_dependent_rewrite", split, BPD_REWRITE_FAMILY, digest([cluster, winner]), cluster,
                      f"dependent_{cluster[:12]}_{winner}", " ".join(bindings), f"{start} {a} {b} = {target} {a} {b}")
    return case, scripts, routes


def spurred_world(split, shape, winner, *, serial_effort=False):
    case, scripts, routes = world(split, shape, winner, serial_effort=serial_effort)
    cluster = digest([EFFORT_SCHEMA if serial_effort else SPUR_SCHEMA, shape])
    argument = case.target.split()[1 + int(cluster[-1], 16) % 2]
    stem, separator, tail = case.binders.partition(" : Nat)")
    source = "loopt_dependent_rewrite_effort" if serial_effort else "loopt_dependent_rewrite_spur"
    case = replace(case, source=source, lineage=digest([cluster, winner]), cluster=cluster,
                   tid=f"dependent_spur_{cluster[:12]}_{winner}",
                   binders=f"{stem} zspur{separator}{tail} (h9000 : {argument} = zspur)")
    return case, scripts, routes


def effort_world(split, shape, winner):
    return spurred_world(split, shape, winner, serial_effort=True)


def parallel_depth(graph):
    """Report the earliest synchronous arrival of each public candidate."""
    depths = [0 if feature[0] else inf for feature in graph.features]
    for _ in range(BUDGET):
        steps = [1 + max((depths[source - 1] for source in row if source), default=0) for row in graph.premises]
        remaining = depths.copy()
        for head, depth in zip(graph.heads, steps, strict=True):
            remaining[head] = min(remaining[head], depth)
        depths = remaining
    return [steps[action - 1] for action in graph.actions]


def run_branch(repl, case, term_script, guards):
    from ppt_gym.dependent_rewrite import term_rewrites
    from ppt_gym.live_lean import public_state, state_input

    repl.reset()
    state = repl.init(case.statement, pp=True)
    packet, _ = state_input(repl, state, (), BUDGET)
    root, = public_state(repl, state)["goals"]
    root_menu = term_rewrites(root)
    if root_menu != ("rw [h0]", "rw [h1]"):
        raise ValueError("each dependent proof starts with the two registered term rewrites")
    events, term_steps = [], 0
    for attempt in range(1, BUDGET + 1):
        before = public_state(repl, state)
        if not before["goals"]:
            break
        active = before["goals"][0]
        menu = term_rewrites(active)
        if menu is None:
            equality = EQUALITY.fullmatch(active["target"].get("raw", ""))
            if equality is None:
                raise ValueError("Lean exposed an unexpected proof obligation")
            menu = rewrite_menu(active)
            tactic = guards[equality.groups()]
        else:
            tactic = term_script[term_steps]
            term_steps += 1
        if tactic not in menu:
            raise ValueError("a dependent rewrite or guard tactic was absent from the public menu")
        observation = state_input(repl, state, (), BUDGET - attempt + 1)[0] if case.split == "train" else None
        reply = repl.run(state["sid"], tactic)
        if not reply["ok"]:
            raise ValueError("Lean rejected a dependent rewrite or its instantiated guard")
        state = reply["state"]
        after = public_state(repl, state)
        event = {"tactic": tactic, "before": digest(before), "after": digest(after),
                 "menu": list(menu), "targets": [goal["target"] for goal in after["goals"]]}
        if observation is not None:
            event["packet"] = observation
        events.append(event)
    closed = not events[-1]["targets"]
    if closed:
        kernel = repl.kernel(state["sid"])
        if not kernel["ok"] or kernel["axioms"] != []:
            raise ValueError("Lean's kernel rejected a dependent rewrite proof")
    return {"packet": packet, "menu": list(root_menu), "events": events, "closed": closed, "term_rewrites": term_steps}


def audit(split, tree, *, spurs=False, routing=False, confirmation=False, serial_effort=False):
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.dependent_rewrite import dependent_graph
    from ppt_gym.hybrid_rl.__main__ import certify

    schema, shapes, make_world = (SPUR_SCHEMA, spur_layouts(), spurred_world) if spurs else (SCHEMA, layouts(), world)
    if routing or confirmation:
        if split != "dev" or spurs or (routing and confirmation):
            raise ValueError("the routing witness is a separate held-out cohort")
        schema = ROUTING_CONFIRMATION_SCHEMA if confirmation else ROUTING_SCHEMA
        shapes, make_world = {"dev": routing_layouts(confirmation=confirmation)}, spurred_world
    if serial_effort:
        if spurs or routing or confirmation:
            raise ValueError("serial effort uses its own training and held-out layouts")
        schema, shapes, make_world = EFFORT_SCHEMA, effort_layouts(), effort_world
    rows = {}
    with LeanRepl(timeout=15) as repl:
        if repl.version != "4.19.0":
            raise ValueError("the dependent rewrite witness requires Lean 4.19.0")
        for shape in shapes[split]:
            for winner in (0, 1):
                case, scripts, routes = make_world(split, shape, winner)
                branches = []
                for choice, script in enumerate(scripts):
                    start = tree.cpu_seconds()
                    trace = run_branch(repl, case, script, routes)
                    within = tree.cpu_seconds() - start <= 6
                    if (not within or trace != run_branch(repl, case, script, routes)
                            or trace["closed"] != (choice == winner) or len(trace["events"]) != BUDGET
                            or trace["term_rewrites"] != shape[0]):
                        raise ValueError(f"dependent rewrites failed native replay or their budget in {case.tid}")
                    graph = dependent_graph(trace["packet"], trace["menu"])
                    if graph is None or not all(graph.actions):
                        raise ValueError("the argument-binding graph exceeded its public limits")
                    cert = certify(case, {"live_kernel": trace["closed"],
                                          "accepted_script": [step["tactic"] for step in trace["events"]]})
                    if trace["closed"] and (not cert or not cert["passed"] or cert["axioms"]):
                        raise ValueError("the dependent branch lacks an independent Lean proof")
                    branches.append({**trace, "choice": choice, "certificate": cert, "replay_passed": True})
                rows[case.root] = {"case": asdict(case), "shape": shape, "winner": winner,
                                   "packet": branches[0]["packet"], "menu": branches[0]["menu"], "branches": branches}
                if serial_effort:
                    span = parallel_depth(dependent_graph(branches[0]["packet"], branches[0]["menu"]))
                    if span[0] != span[1] or span[0] == inf:
                        raise ValueError("the serial effort alternatives must have tied public arrival depths")
                    rows[case.root]["parallel_depth"] = span[0]
    source = Path(__file__).resolve()
    return {"schema": schema, "source_sha256": sha256_file(source),
            "policy_sha256": sha256_file(source.parents[1] / "ppt_gym/dependent_rewrite.py"),
            "split": split, "tactic_checks": BUDGET, "lean": "4.19.0", "rows": rows, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("train", "dev"))
    parser.add_argument("output", type=Path)
    study = parser.add_mutually_exclusive_group()
    study.add_argument("--spurs", action="store_true")
    study.add_argument("--routing", action="store_true")
    study.add_argument("--routing-confirmation", action="store_true")
    study.add_argument("--serial-effort", action="store_true")
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = audit(args.split, tree, spurs=args.spurs, routing=args.routing,
                       confirmation=args.routing_confirmation, serial_effort=args.serial_effort)
        with args.output.open("xb") as output:
            output.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()
    print(f"Lean replayed both dependent rewrite openings in {len(report['rows'])} {args.split} worlds.", flush=True)


if __name__ == "__main__":
    main()
