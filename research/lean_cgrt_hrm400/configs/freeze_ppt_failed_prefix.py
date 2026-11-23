"""Validate retained and abandoned Lean recovery routes on native snapshots."""

import argparse
from collections import Counter
from pathlib import Path

from configs.freeze_ppt_computational_guards import start
from configs.freeze_ppt_guard_recovery import certify_routes
from configs.freeze_ppt_native_progress import advance, native_sources
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import sha256_file
from ppt_gym.failed_prefix import PREFIX, REJECTION, SCHEMA, SETUP, cases
from ppt_gym.ram_limits import TreeBudget


ROOTS = {"source": 48, "development": 40}


def advance_script(repl, snapshot, script, terminal):
    state, public = snapshot
    trace = []
    for index, tactic in enumerate(script):
        status, state, public = advance(repl, state, public, tactic)
        trace.append({"tactic": tactic, "status": status})
        expected = terminal if index == len(script) - 1 else "progress"
        if status != expected:
            raise ValueError(f"unexpected Lean recovery step {tactic}: {status}, expected {expected}")
    return (state, public), trace


def setup(repl, case):
    base = start(repl, case)
    fork, _ = advance_script(repl, base, PREFIX, "progress")
    saved, _ = advance_script(repl, fork, SETUP[len(PREFIX):], "progress")
    if len(saved[1]["goals"]) != 1 or len(fork[1]["goals"]) < 2:
        raise ValueError("the saved Lean snapshot did not retain the first conjunct")
    _, failure = advance_script(repl, saved, REJECTION, "rejected")
    return fork, saved, failure


def route_attempt(repl, route, snapshot, passed):
    script = route["attempt"] + (route["suffix"] if passed else [])
    _, trace = advance_script(repl, snapshot, script, "closed" if passed else "rejected")
    if len(trace) != route["pass_cost" if passed else "fail_cost"]:
        raise ValueError("the native recovery check count changed")
    return trace


def verify_case(repl, case):
    fork, saved, failure = setup(repl, case)
    routes = []
    for arm, (route, passed) in enumerate(zip(case["routes"], case["expected"], strict=True)):
        trace = route_attempt(repl, route, saved if arm == 0 else fork, passed)
        prefix = SETUP if arm == 0 else PREFIX
        proof = [*prefix, *(step["tactic"] for step in trace)] if passed else []
        routes.append({**route, "proof": proof, "trace": trace})
    introductions = [hyp["name"] for hyp in fork[1]["goals"][0]["hypotheses"]]
    return {**{key: value for key, value in case.items() if key != "expected"}, "routes": routes,
            "intro_names": introductions,
            "truth": case["expected"], "fork": fork[1], "saved": saved[1], "initial_rejection": failure}


def replay_case(repl, case, row):
    fork, saved, failure = setup(repl, case)
    if (fork[1], saved[1], failure) != (case["fork"], case["saved"], case["initial_rejection"]):
        raise ValueError("native recovery snapshots changed")
    checks = len(SETUP) + len(REJECTION)
    for event in row["trace"]:
        arm = event["arm"]
        again = route_attempt(repl, case["routes"][arm], saved if arm == 0 else fork, event["passed"])
        if again != case["routes"][arm]["trace"] or len(again) != event["charge"]:
            raise ValueError("a paid recovery route changed on replay")
        checks += len(again)
    if checks != row["checks"]:
        raise ValueError("the paid recovery trace cost changed")


def freeze(args, tree):
    paths = (args.bpd, args.mathcan)
    bank, certificates = {}, []
    with native_sources(paths) as (directory, env, imports):
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected native Lean version")
            for partition, count in ROOTS.items():
                bank[partition] = [verify_case(repl, case) for case in cases(partition, count)]
                if Counter(Counter(case["pair"] for case in bank[partition]).values()) != {2: count}:
                    raise ValueError("the recovery bank lost its paired theorem structure")
                certificates.extend((case, route["proof"]) for case in bank[partition]
                                    for route in case["routes"] if route["proof"])
                print("sealed native", partition, "recovery worlds", len(bank[partition]), flush=True)
        independent = certify_routes(directory, env, imports, certificates)
    write_receipt(args.results, {"schema": SCHEMA, **bank, "independent": independent, "memory": tree.stats(),
                                 "lean": sha256_file(LEAN_EXE), "repl": sha256_file(REPL_EXE),
                                 "imports": {path.name: sha256_file(path) for path in paths}})
    print("independently compiled recovery proofs", independent["certificates"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bpd", "mathcan", "results"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        freeze(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
