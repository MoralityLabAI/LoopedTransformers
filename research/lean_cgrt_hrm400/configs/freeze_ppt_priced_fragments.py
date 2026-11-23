"""Collect sparse native proof fragments and seal paired public tactic prices."""

import argparse
from collections import Counter
from pathlib import Path
import random

from configs.freeze_ppt_native_progress import advance, observe
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt import fragment
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.hybrid_rl.graph_model import MAX_ATOMS, MAX_PREMISES, MAX_RULES
from ppt_gym.live_lean import candidate_menu
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.priced-native-fragments.v1"
FAMILIES = ("alt-dead", "alt-deep")
SOURCE_ROOTS = 48
HELD_ROOTS = 40
FRAGMENTS = 28
LIMIT = 8


def weighted_sizes(theorem, prices):
    sizes = {atom: fragment.INF for atom in range(theorem.n_atoms)}
    for index, atom in enumerate(theorem.facts):
        sizes[atom] = min(sizes[atom], prices[f"apply h{index}"])
    for _ in range(theorem.n_atoms):
        for index, (premises, head) in enumerate(theorem.rules):
            cost = prices[f"apply r{index}"] + sum(sizes[atom] for atom in premises)
            sizes[head] = min(sizes[head], cost)
    return sizes


def reference(theorem, prices):
    sizes = weighted_sizes(theorem, prices)
    producers = {}
    for index, atom in enumerate(theorem.facts):
        producers.setdefault(atom, []).append((f"apply h{index}", ()))
    for index, (premises, atom) in enumerate(theorem.rules):
        producers.setdefault(atom, []).append((f"apply r{index}", premises))

    def cost(producer):
        action, premises = producer
        return prices[action] + sum(sizes[atom] for atom in premises)

    def script(atom):
        action, parents = min(producers[atom], key=cost)
        return [action, *(step for parent in parents for step in script(parent))]

    active, *pending = theorem.goal
    first = [action for action, parents in producers[active] if cost((action, parents)) == sizes[active]]
    split = ["constructor"] if pending else []
    total = sum(sizes[atom] for atom in theorem.goal) + (prices["constructor"] if pending else 0)
    if total > LIMIT or len(first) != 1:
        return None
    return total, first[0], split + [action for atom in theorem.goal for action in script(atom)]


def price_pair(theorem, seed):
    actions = ["constructor", *(f"apply h{i}" for i in range(len(theorem.facts))),
               *(f"apply r{i}" for i in range(len(theorem.rules)))]
    randomizer = random.Random(seed)
    found = {}
    for _ in range(18):
        prices = {action: 1 if action == "constructor" else randomizer.choice((1, 1, 1, 2, 2, 3))
                  for action in actions}
        proof = reference(theorem, prices)
        if proof is None or proof[0] < 4:
            continue
        budget, first, script = proof
        found.setdefault(first, {"prices": prices, "budget": budget, "reference": script, "producer": first})
        if len(found) == 2:
            return tuple(found.values())
    return ()


def roots(previous):
    seen = {row["module"] for row in previous["source"] + previous["held"]}
    for partition, count in (("source", SOURCE_ROOTS), ("held", HELD_ROOTS)):
        cases = []
        for index in range(14000):
            family = FAMILIES[len(cases) % len(FAMILIES)]
            seed = int(digest([SCHEMA, partition, index])[:15], 16)
            theorem = fragment.generate(seed, 2 + index % 3, family, mirror_p=.6, conj_p=.4,
                                        tid=f"priced_{partition}_{len(cases)}")
            signature = fragment.canonical_hash(theorem)
            if signature in seen or theorem.n_atoms > MAX_ATOMS or len(theorem.rules) + len(theorem.facts) > MAX_RULES:
                continue
            if any(len(premises) > MAX_PREMISES for premises, _ in theorem.rules):
                continue
            options = fragment.reference_sizes(theorem)
            live = [1 + sum(options[p] for p in parents) for parents, head in theorem.rules
                    if head == theorem.goal[0] and all(options[p] < fragment.INF for p in parents)]
            if len(live) < 2:
                continue
            pair = price_pair(theorem, seed)
            if pair:
                seen.add(signature)
                cases.append({"module": signature, "source": "LoopedTransformers Horn", "theorem": theorem.tid,
                              "statement": fragment.statement(theorem), "family": family, "graph": theorem.to_json(),
                              "schedules": pair})
            if len(cases) == count:
                break
        if len(cases) != count:
            raise ValueError("insufficient paired price layouts")
        yield cases


def checked_root(repl, case):
    repl.reset()
    state = repl.init(case["statement"], pp=True)
    return state, observe(repl, state)


def certify(repl, case):
    state, initial = checked_root(repl, case)
    for variant in case["schedules"]:
        current, public, spent = state, initial, 0
        for index, move in enumerate(variant["reference"], start=1):
            if move not in candidate_menu(repl, current):
                raise ValueError("a reference move left the public grammar")
            status, current, public = advance(repl, current, public, move)
            expected = "closed" if index == len(variant["reference"]) else "progress"
            if status != expected:
                raise ValueError("the kernel disagreed with the weighted reference")
            spent += variant["prices"][move]
        if spent != variant["budget"]:
            raise ValueError("a source reference exceeded the registered price")
    return {**case, "visible_sha256": digest(initial),
            "intro_names": [hyp["name"] for hyp in initial["goals"][0]["hypotheses"]]}


def collect_schedule(repl, root, initial, prices, seed):
    """Restart only at native proof snapshots reached by an earlier fragment."""
    rng = random.Random(seed)
    root_key = digest(initial)
    cache = {root_key: (root, initial, ())}
    counts = Counter()
    logs, public_states = [], {}
    for episode in range(FRAGMENTS):
        start = root_key
        if episode % 4:
            start = rng.choice(list(cache))
        key = start
        allowance = rng.choice((2, 3, 4))
        left, transitions = allowance, []
        while left:
            state, public, history = cache[key]
            menu = candidate_menu(repl, state)
            if not menu:
                break
            public_states.setdefault(key, {"public": public, "menu": menu, "history": history})
            legal = [move for move in menu if prices[move] <= left]
            if not legal:
                break
            minimum = min(counts[key, move] for move in legal)
            action = rng.choice([move for move in legal if counts[key, move] == minimum])
            effect, child, after = advance(repl, state, public, action)
            charge = prices[action]
            left -= charge
            counts[key, action] += 1
            successor = digest(after) if effect == "progress" else None
            transitions.append({"before": key, "action": action, "cost": charge, "status": effect, "after": successor})
            if successor is None:
                break
            cache.setdefault(successor, (child, after, (*history, action)))
            key = successor
        logs.append({"start": start, "allowance": allowance, "steps": transitions})
    return {"prices": prices, "root": root_key, "public_states": public_states, "fragments": logs}


def collect(repl, case):
    state, public = checked_root(repl, case)
    logs = [collect_schedule(repl, state, public, prices["prices"],
                             int(digest([SCHEMA, case["module"], index])[:15], 16))
            for index, prices in enumerate(case["schedules"])]
    # The source policy never receives the symbolic route or its return.
    return {key: val for key, val in case.items() if key != "schedules"} | {"logs": logs}


def run(args, tree):
    prior = read_receipt(args.previous)
    source, held = roots(prior)
    with LeanRepl() as repl:
        if repl.version != LEAN_VERSION:
            raise ValueError("unexpected native Lean version")
        logged = [collect(repl, case) for case in source]
        sealed = [certify(repl, case) for case in held]
    counts = Counter(step["status"] for case in logged for log in case["logs"]
                     for episode in log["fragments"] for step in episode["steps"])
    print("trained roots", len(logged), "scoring roots", len(sealed), "native sparse checks", dict(counts), flush=True)
    write_receipt(args.output, {"schema": SCHEMA, "source": logged, "held": sealed, "evidence": counts,
                                "generator_sha256": sha256_file(Path(fragment.__file__)),
                                "prior_registry_sha256": sha256_file(args.previous),
                                "lean_sha256": sha256_file(LEAN_EXE), "repl_sha256": sha256_file(REPL_EXE),
                                "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        run(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
