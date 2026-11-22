"""Collect witnessed training branches on held-in conditional rewrite layouts."""

import argparse
from dataclasses import asdict, replace
import gzip
from itertools import combinations
import json
from pathlib import Path

from configs.check_ppt_bpd_conditional_rewrites import BUDGET, LAYOUTS, paired_world, play
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.ram_limits import TreeBudget


def training_shapes():
    return sorted((length, first, second, cost)
                  for length in range(3, 7)
                  for first, second in combinations(range(1, length), 2)
                  for cost in range(1, BUDGET - length)
                  if (length, first, second, cost) not in LAYOUTS)


def collect(witness_path, tree):
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.conditional_rewrite import conditional_graph, project_equalities
    from ppt_gym.hybrid_rl.__main__ import certify
    from ppt_gym.live_lean import public_state, state_input
    from ppt_gym.rewrite import rewrite_menu

    witness = json.loads(gzip.decompress(witness_path.read_bytes()))
    if witness["tactic_checks"] != BUDGET or len(witness["rows"]) != 2 * len(LAYOUTS):
        raise ValueError("the conditional training data needs the fixed development witness")
    development = {row["case"]["cluster"] for row in witness["rows"].values()}
    rows = {}
    with LeanRepl(timeout=15) as repl:
        if repl.version != witness["lean"]:
            raise ValueError("the conditional training and evaluation verifiers differ")
        for layout, shape in enumerate(training_shapes(), 100):
            for winner in (0, 1):
                case, scripts, horn = paired_world(layout, shape, winner)
                case = replace(case, split="train")
                if case.root in witness["rows"] or case.cluster in development:
                    raise ValueError("conditional training and development layouts overlap")
                repl.reset()
                initial = repl.init(case.statement, pp=True)
                actual = public_state(repl, initial)
                packet, identity = state_input(repl, initial, (), BUDGET)
                menu = rewrite_menu(project_equalities(actual["goals"][0])[0])
                graph = conditional_graph(packet, menu)
                if graph is None or not all(graph.actions):
                    raise ValueError("the typed model cannot represent a generated training declaration")
                branches = []
                for choice, script in enumerate(scripts):
                    start = tree.cpu_seconds()
                    record = play(repl, case, script, horn)
                    valid = (record["closed"] == (choice == winner) and len(record["events"]) == BUDGET
                             and record["events"][0]["before"] == identity and tree.cpu_seconds() - start <= 6)
                    if not valid or record != play(repl, case, script, horn):
                        raise ValueError("a conditional training branch exceeded its budget or failed replay")
                    cert = certify(case, {"live_kernel": record["closed"],
                                          "accepted_script": [step["tactic"] for step in record["events"]]})
                    if record["closed"] and (not cert or not cert["passed"] or cert["axioms"]
                                             or digest(cert["source"]) != cert["source_digest"]):
                        raise ValueError("a winning training branch lacks an independent Lean certificate")
                    branches.append({**record, "choice": choice, "certificate": cert, "replay_passed": True})
                rows[case.root] = {"case": asdict(case), "packet": packet, "menu": menu, "branches": branches}
    return {"witness_sha256": sha256_file(witness_path), "tactic_checks": BUDGET, "lean": witness["lean"],
            "scope": "conditional rewrite training on layouts disjoint from development", "rows": rows, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("witness", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = collect(args.witness, tree)
        with args.output.open("xb") as output:
            output.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()
    print(f"Verified both conditional branches in {len(report['rows'])} training worlds.", flush=True)


if __name__ == "__main__":
    main()
