"""Verify all witness and cache decisions for disjoint Lean layouts."""

import argparse
import gzip
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE, lean_env
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.dependent_witness import FORK, PREFIX, proof_view
from ppt_gym.live_lean import public_state
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.witness_reuse import OPTIONS, cohort

from configs.check_ppt_dependent_witness import action_costs
from configs.run_ppt_two_slot import request_state


SCHEMA = "ppt.witness-reuse-plans.v1"
CACHE_INDEX = len(PREFIX) + 2


def certificate_source(tid, statement, binders, tactics):
    return (f"theorem {tid} : {statement} := by\n  intro {' '.join(binders)}\n"
            + "".join(f"  {action}\n" for action in tactics) + f"#print axioms {tid}\n")


def certify_batch(roots):
    sources = []
    for root in roots:
        for plans in root["receipts"]:
            for row in plans:
                source = certificate_source(row["tid"], root["statement"], root["binders"], row["actions"])
                row["source_digest"] = digest(source)
                sources.append(source)
    with TemporaryDirectory() as scratch:
        path = Path(scratch) / "ReusePlans.lean"
        path.write_text("\n".join(sources), encoding="utf-8")
        proc = subprocess.run([str(LEAN_EXE), str(path)], check=True, capture_output=True,
                              text=True, encoding="utf-8", env=lean_env(), timeout=120)
    axioms = {line for line in proc.stdout.splitlines() if line.endswith(" does not depend on any axioms")}
    for root in roots:
        for plans in root["receipts"]:
            for row in plans:
                receipt = f"'{row['tid']}' does not depend on any axioms"
                if receipt not in axioms:
                    raise ValueError(f"missing independent certificate for {row['tid']}")
                row["axioms"] = []


def verify_root(repl, case):
    repl.reset()
    initial = repl.init(case["statement"])
    states = {(): initial}

    def observed(prefix):
        return public_state(repl, states[prefix])["goals"]

    receipts, cache_states = [], []
    for witness, (paths, expected) in enumerate(zip(case["paths"], case["costs"], strict=True)):
        options = []
        for mode, actions, cost in zip(OPTIONS, paths, expected, strict=True):
            for index, action in enumerate(actions):
                prefix = actions[:index]
                next_prefix = actions[:index + 1]
                if next_prefix not in states:
                    states[next_prefix] = request_state(repl.run(states[prefix]["sid"], action))
                if index == case["checks"] - 1 and states[next_prefix]["goals"] and cost <= case["checks"]:
                    raise ValueError("an in-budget plan unexpectedly left goals")
            kernel = repl.kernel(states[actions]["sid"])
            if len(actions) != cost or observed(actions) or not kernel["ok"] or kernel["axioms"]:
                raise ValueError("the native plan disagreed with the registered cost or closure")
            options.append({"tid": f"reuse_{case['cluster'][:12]}_{case['world']}_{witness}_{mode}",
                            "actions": actions, "checks": cost, "closed_in_budget": cost <= case["checks"],
                            "budget_goals": observed(actions[:min(cost, case["checks"])])})
        cache_prefix = paths[0][:CACHE_INDEX]
        if any(path[:CACHE_INDEX] != cache_prefix for path in paths):
            raise ValueError("a cache choice has a different native prefix")
        cache_states.append({"goals": observed(cache_prefix), "menu": [path[CACHE_INDEX] for path in paths]})
        receipts.append(options)

    fork = proof_view(observed(PREFIX))
    if action_costs(fork, FORK, len(PREFIX)) != [row[0]["checks"] for row in receipts]:
        raise ValueError("the separate proof-tree reference disagreed with the direct plans")
    wins = [(i, j) for i, options in enumerate(receipts) for j, row in enumerate(options)
            if row["closed_in_budget"]]
    if len(wins) != 1 or wins[0][1] == 0:
        raise ValueError("expected a unique in-budget cached proof")
    public_case = {key: value for key, value in case.items() if key not in {"paths", "costs"}}
    return {**public_case, "fork": fork, "cache_states": cache_states, "receipts": receipts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        with LeanRepl() as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("the reuse comparison requires the pinned Lean verifier")
            rows = [verify_root(repl, case) for case in cohort()]
        for start in range(0, len(rows), 10):
            certify_batch(rows[start:start + 10])
        report = {"schema": SCHEMA, "lean": LEAN_VERSION, "server_sha256": sha256_file(REPL_EXE),
                  "verifier_sha256": sha256_file(LEAN_EXE), "rows": rows, "memory": tree.stats()}
        with args.output.open("xb") as out:
            out.write(gzip.compress((canonical(report) + "\n").encode(), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
