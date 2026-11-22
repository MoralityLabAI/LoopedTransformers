"""Verify training-only continuations of the eight-check BPD failures."""

import argparse
from collections import defaultdict, deque
import gzip
from itertools import combinations
import json
from pathlib import Path

from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rewrite import rewrite_problem


EVALUATION_CHECKS = 8
EXTRA_TRAINING_CHECKS = 4


def shortest_suffix(goal):
    left, right, edges = rewrite_problem(goal)
    neighbors = defaultdict(list)
    for tactic, source, target in edges:
        neighbors[source].append((tactic, target))
    queue, seen = deque([(left, [])]), {left}
    while queue:
        vertex, script = queue.popleft()
        if vertex == right:
            return script
        if len(script) == EXTRA_TRAINING_CHECKS:
            continue
        for tactic, successor in neighbors[vertex]:
            if successor not in seen:
                seen.add(successor)
                queue.append((successor, [*script, tactic]))
    return None


def planned_branches(traces):
    results = []
    for trace in traces:
        ep = trace["episode"]
        if (ep["verified_closed"] or ep["tactic_checks"] != EVALUATION_CHECKS
                or len(ep["accepted_script"]) != EVALUATION_CHECKS or not trace["native_replay"]["replay_passed"]):
            continue
        final = ep["events"][-1]["state_after"]
        goal, = final["goals"]
        script = shortest_suffix(goal)
        if script:
            results.append((trace, script))
    return results


def verify_extension(repl, case, trace, suffix):
    from ppt_gym.hybrid_rl.__main__ import certify
    from ppt_gym.live_lean import candidate_menu, public_state

    repl.reset()
    state = repl.init(case.statement, pp=True)
    ep = trace["episode"]
    script = [*ep["accepted_script"], *suffix]
    extension = []
    for offset, tactic in enumerate(script):
        old = public_state(repl, state)
        if tactic not in candidate_menu(repl, state, rewrite=True):
            raise ValueError("the proposed continuation departs from Lean's public grammar")
        reply = repl.run(state["sid"], tactic)
        if not reply["ok"]:
            raise ValueError("Lean rejected a training continuation")
        state = reply["state"]
        observed = public_state(repl, state)
        if offset < EVALUATION_CHECKS:
            recorded = ep["events"][offset]
            if (recorded["kind"] != "tactic" or recorded["tactic"] != tactic
                    or recorded["state_before_sha256"] != digest(old) or recorded["state_after_sha256"] != digest(observed)):
                raise ValueError("the eight-check training prefix changed during replay")
        else:
            extension.append({"tactic": tactic, "before": digest(old), "after": digest(observed)})
    kernel = repl.kernel(state["sid"])
    if not kernel["ok"] or kernel["axioms"] != []:
        raise ValueError("the extended trajectory did not close in the Lean kernel")
    certificate = certify(case, {"live_kernel": True, "accepted_script": script})
    if not certificate or not certificate["passed"] or certificate["axioms"]:
        raise ValueError("the extended trajectory lacks its standalone Lean certificate")
    return {"choice": trace["choice"], "checks_after_cutoff": len(suffix), "extension": extension,
            "training_check_count": len(script), "certificate": certificate}


def audit(cohort, evidence, tree):
    from lcgrt.lean_repl import LeanRepl

    data = json.loads(gzip.decompress(evidence.read_bytes()))
    if data["cohort_sha256"] != sha256_file(cohort):
        raise ValueError("expected the original verified rewrite training cohort")
    cases = {case.root: case for case in load(cohort)["train"]}
    rows = {}
    with LeanRepl(timeout=15) as repl:
        if repl.version != "4.19.0":
            raise ValueError("the rewrite suffix study requires Lean 4.19.0")
        for root, item in data["interventions"]["rows"].items():
            planned = planned_branches(item["traces"])
            if len({len(script) for _, script in planned}) < 2:
                continue
            verified = [verify_extension(repl, cases[root], trace, script) for trace, script in planned]
            pairs = [[a["choice"], b["choice"]] for a, b in combinations(verified, 2)
                     if a["checks_after_cutoff"] != b["checks_after_cutoff"]]
            rows[root] = {"fork": item["fork"], "verified": verified, "unequal_cost_pairs": pairs}
    return {"cohort_sha256": sha256_file(cohort), "fork_evidence_sha256": sha256_file(evidence),
            "scope": "training only, native proof extensions after the eight-check cutoff",
            "evaluation_checks": EVALUATION_CHECKS, "extra_training_checks": EXTRA_TRAINING_CHECKS,
            "rows": rows, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "evidence", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = audit(args.cohort, args.evidence, tree)
        with args.output.open("xb") as output:
            output.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()
    print(len(report["rows"]), "failed forks with distinct recertified training costs", flush=True)


if __name__ == "__main__":
    main()
