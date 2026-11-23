"""Seal new native Horn branch tasks and bounded source transition graphs."""

import argparse
from collections import Counter, deque
import json
from pathlib import Path

from configs.freeze_ppt_native_progress import advance, observe
from configs.freeze_ppt_successor_curriculum import proof_script
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt import fragment
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.hybrid_rl.corpus import REPO
from ppt_gym.hybrid_rl.graph_model import MAX_ATOMS, MAX_PREMISES, MAX_RULES
from ppt_gym.live_lean import candidate_menu
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.premise-budget.v1"
FAMILIES = ("chain-deep", "tree-dead", "alt-deep")
SOURCE_ROOTS = 60
SCORE_ROOTS = 48
EXPANSIONS = 24


def first_costs(theorem, sizes):
    active, *pending = theorem.goal
    shared = len(pending) + sum(sizes[goal] for goal in pending)
    options = [1 + shared for fact in theorem.facts if fact == active]
    options += [1 + shared + sum(sizes[atom] for atom in deps) for deps, head in theorem.rules if head == active]
    return options


def roster(past):
    cohort = REPO / "research/lean_cgrt_hrm400/results/ppt_hybrid_rl/development-cohort.json"
    seen = {row["lineage"] for row in json.loads(cohort.read_bytes())["cases"]}
    seen.update(row["module"] for row in past["training"] + past["cases"] if row["source"] == "LoopedTransformers Horn")
    for split, count in (("source", SOURCE_ROOTS), ("score", SCORE_ROOTS)):
        entries = []
        for index in range(12000):
            family = FAMILIES[len(entries) % len(FAMILIES)]
            seed = int(digest([SCHEMA, split, index])[:15], 16)
            theorem = fragment.generate(seed, 2 + index % 4, family, arity3=index % 3 == 0,
                                        mirror_p=.8, conj_p=.4, tid=f"premise_{split}_{len(entries)}")
            if theorem.n_atoms > MAX_ATOMS or len(theorem.facts) + len(theorem.rules) > MAX_RULES:
                continue
            if any(len(pre) > MAX_PREMISES for pre, _ in theorem.rules):
                continue
            signature = fragment.canonical_hash(theorem)
            sizes = fragment.reference_sizes(theorem)
            costs = first_costs(theorem, sizes)
            minimum = min(costs)
            if signature in seen or not 3 <= minimum <= 8 or max(costs) == minimum:
                continue
            seen.add(signature)
            entries.append({"module": signature, "source": "LoopedTransformers Horn",
                            "theorem": theorem.tid, "statement": fragment.statement(theorem),
                            "reference": proof_script(theorem, sizes), "checks": minimum,
                            "family": family, "graph": theorem.to_json(), "first_costs": costs})
            if len(entries) == count:
                break
        if len(entries) != count:
            raise ValueError("insufficient disjoint Horn roots with a meaningful branch choice")
        yield entries


def initialize(repl, case):
    repl.reset()
    root = repl.init(case["statement"], pp=True)
    public = observe(repl, root)
    return root, public


def certify(repl, case):
    state, initial = initialize(repl, case)
    visible = initial
    if len(case["reference"]) != case["checks"]:
        raise ValueError("the registered Horn proof has inconsistent check counts")
    for index, action in enumerate(case["reference"], start=1):
        if action not in candidate_menu(repl, state):
            raise ValueError("the public menu hides a legal reference producer")
        status, state, visible = advance(repl, state, visible, action)
        if status != ("closed" if index == case["checks"] else "progress"):
            raise ValueError("native Lean did not certify the structural reference")
    return {**case, "visible_sha256": digest(initial),
            "intro_names": [hyp["name"] for hyp in initial["goals"][0]["hypotheses"]]}


def collect(repl, case):
    initial, visible = initialize(repl, case)
    queue = deque()
    nodes = {}

    def expand(prefix, state, public):
        choices, children = {}, {}
        for action in candidate_menu(repl, state):
            status, child, after = advance(repl, state, public, action)
            choices[action] = {"status": status, "after_sha256": digest(after)}
            if status == "progress" and len(prefix) + 1 < 8:
                children[action] = child, after
                queue.append(((*prefix, action), child, after))
        nodes[prefix] = {"prefix": prefix, "public": public, "choices": choices}
        return children

    prefix, state, public = (), initial, visible
    for action in case["reference"]:
        children = expand(prefix, state, public)
        if action in children:
            state, public = children[action]
            prefix = (*prefix, action)
    while queue and len(nodes) < EXPANSIONS:
        prefix, state, public = queue.popleft()
        if prefix not in nodes:
            expand(prefix, state, public)
    return {**case, "nodes": list(nodes.values()), "unexpanded": len(queue)}


def run(args, tree):
    prior = read_receipt(args.previous)
    source, score = roster(prior)
    training, held = [], []
    with LeanRepl() as repl:
        if repl.version != LEAN_VERSION:
            raise ValueError("unexpected native Lean version")
        for case in source:
            training.append(collect(repl, certify(repl, case)))
        for case in score:
            held.append(certify(repl, case))
    if {row["visible_sha256"] for row in training} & {row["visible_sha256"] for row in held}:
        raise ValueError("a scoring observation entered source training")
    print("source roots", len(training), "native prefixes", sum(len(case["nodes"]) for case in training),
          "held roots", len(held), "reference effort", dict(Counter(case["checks"] for case in held)), flush=True)
    write_receipt(args.output, {"schema": SCHEMA, "source": training, "held": held,
                                "generator_sha256": sha256_file(Path(fragment.__file__)),
                                "previous_sha256": sha256_file(args.previous), "lean_sha256": sha256_file(LEAN_EXE),
                                "repl_sha256": sha256_file(REPL_EXE), "memory": tree.stats()})


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
