"""Replay and independently certify Lean proofs with a shared witness."""

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import json
from math import inf
from pathlib import Path

from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.dependent_witness import (
    BUDGETS, DEEP_FAMILY, FORK, FORK_INDEX, action_projection, cohorts, deep_cohorts,
    examples, horn_problem, proof_view, script,
)
from ppt_gym.hybrid_rl.__main__ import certify
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.live_lean import public_state
from ppt_gym.ram_limits import TreeBudget
from configs.run_ppt_two_slot import trajectory


def verify_pair(repl):
    rows = []
    for case, winner in examples():
        branches = []
        for branch in range(2):
            repl.reset()
            state = repl.init(case.statement)
            events = []
            tactics = script(winner, branch)
            for tactic in tactics:
                reply = repl.run(state["sid"], tactic)
                if not reply["ok"]:
                    raise ValueError(f"Lean rejected a shared-witness tactic: {reply.get('error')}")
                state = reply["state"]
                observation = public_state(repl, state)
                events.append({"tactic": tactic, "state": observation, "state_sha256": digest(observation)})
            before = events[1]["state"]["goals"]
            after = events[2]["state"]["goals"]
            if (len(before) != 3 or before[2]["target"] != {"raw": "Nat"}
                    or before[0]["target"].get("raw", "")[:2] != "P "
                    or before[1]["target"].get("raw", "")[:2] != "Q "
                    or len(after) != 1 or after[0]["target"] != {"raw": f"Q {branch}"}):
                raise ValueError("the first proof did not assign the witness and specialize its sibling")
            metaterm = before[0]["target"]["raw"][2:]
            if not metaterm.startswith("?") or before[1]["target"]["raw"][2:] != metaterm:
                raise ValueError("the pending goals did not share one unresolved Lean metavariable")
            if bool(events[3]["state"]["goals"]) == (branch == winner):
                raise ValueError("the dependent branch disagreed with its four-check deadline")
            kernel = repl.kernel(state["sid"])
            if not kernel["ok"] or kernel["axioms"]:
                raise ValueError("the dependent witness proof failed its live kernel check")
            certificate = certify(case, {"live_kernel": True, "accepted_script": tactics}, timeout=60)
            if not certificate or not certificate["passed"] or certificate["axioms"]:
                raise ValueError("the dependent witness proof failed standalone Lean certification")
            branches.append({"choice": branch, "checks": len(tactics), "events": events, "certificate": certificate})
        if branches[0]["events"][:2] != branches[1]["events"][:2]:
            raise ValueError("dependent branches lost their shared native prefix")
        rows.append({"case": asdict(case), "winner": winner, "branches": branches})
    return rows


def verify_cohort(repl, tree, cases):
    rows = []
    for case, winner, scripts in cases:
        checks = BUDGETS[case.family]
        trials = [trajectory(repl, tree, case, forced_prefix=actions) for actions in scripts]
        if any(len(t["events"]) != checks or not t["within"] or t["success"] != (i == winner)
               for i, t in enumerate(trials)):
            raise ValueError(f"the shared witness disagreed with its {checks}-check deadline for {case.tid}")
        if trials[0]["events"][:FORK_INDEX] != trials[1]["events"][:FORK_INDEX]:
            raise ValueError(f"the shared witness branches lost their common prefix for {case.tid}")
        for i, trial in enumerate(trials):
            events = trial["events"]
            before = events[FORK_INDEX]["native_goals"]
            after = events[FORK_INDEX + 1]["native_goals"]
            if len(before) != 3 or len(proof_view(before)) != 2 or len(after) != 1:
                raise ValueError(f"the native shared witness did not specialize its sibling for {case.tid}")
            fact = next(h["type"]["raw"] for h in before[0]["hypotheses"] if h["name"] == f"h{i}")
            predicate = before[1]["target"]["raw"].split()[0]
            if after[0]["target"] != {"raw": f"{predicate} {fact.split()[1]}"}:
                raise ValueError(f"the sibling goal received the wrong shared witness for {case.tid}")
        losing = 1 - winner
        late = trajectory(repl, tree, case, forced_prefix=scripts[losing], extra_checks=1)
        if (not late["success"] or len(late["events"]) != checks + 1
                or late["events"][:checks] != trials[losing]["events"]):
            raise ValueError(f"the late dependent continuation could not be certified for {case.tid}")
        rows.append({"case": asdict(case), "winner": FORK[winner], "trials": trials, "late": late})
    return rows


def projected_cost(goal, target, hops=None):
    if target is None:
        return 0
    terms, facts, edges, active = horn_problem(goal, target)
    costs = {term: 1 if term in facts else inf for term in terms}
    # Each tactic has positive cost. A cheapest derivation has no repeated
    # target down a branch, so at most len(terms) rounds suffice.
    for _ in range(len(terms) if hops is None else hops):
        updated = costs.copy()
        for head, premise, partner in edges:
            cost = 1 + costs[premise] + (costs[partner] if partner else 0)
            updated[head] = min(updated[head], cost)
        if updated == costs:
            break
        costs = updated
    return costs[active]


def action_costs(goals, actions, checks_before, *, hops=None):
    """Count tactics after each projected exact or apply choice."""
    return [
        checks_before + 1 + sum(projected_cost(goal, target, hops)
                                for goal, target in zip(goals, action_projection(goals, action), strict=True))
        for action in actions
    ]


def deadline_actions(actions, costs, checks):
    return [action for action, cost in zip(actions, costs, strict=True) if cost is not None and cost <= checks]


def read_receipt(receipt):
    packet = receipt.read_bytes()
    witness = json.loads(gzip.decompress(packet) if receipt.suffix == ".gz" else packet)
    if witness["schema"] != "ppt.shared-existential-continuation.v1" or witness["lean"] != LEAN_VERSION:
        raise ValueError("expected a frozen shared-witness continuation receipt")
    return witness


def audit_public_cost(receipt):
    """Read the public fork snapshot and use the certificates only to verify the answer."""
    witness = read_receipt(receipt)
    families = {row["case"]["family"] for row in witness["rows"]}
    if len(families) != 1 or not families <= BUDGETS.keys():
        raise ValueError("expected a receipt for one shared-witness family")
    family, = families
    budget = BUDGETS[family]
    hop_counts = range(8 if family == DEEP_FAMILY else 5)
    rows = []
    for row in witness["rows"]:
        fork = row["trials"][0]["events"][FORK_INDEX]
        actions = tuple(fork["menu"])
        if actions != FORK:
            raise ValueError("the witness receipt has an unexpected fork menu")
        winner = actions.index(row["winner"])
        exact = action_costs(fork["goals"], actions, FORK_INDEX)
        certified = [len(row["late"]["events"])] * len(actions)
        certified[winner] = len(row["trials"][winner]["events"])
        if exact != certified:
            raise ValueError(f"public tactic costs disagree with the native receipt for {row['case']['tid']}")
        trace = row["trials"][winner]
        choices = [(i, e) for i, e in enumerate(trace["events"]) if len(e["menu"]) > 1]
        by_depth, scripts = [], []
        for hops in hop_counts:
            by_depth.append([cost if cost < inf else None for cost in
                             action_costs(fork["goals"], actions, FORK_INDEX, hops=hops)])
            scripts.append(all(deadline_actions(e["menu"], action_costs(e["goals"], e["menu"], i, hops=hops), budget)
                               == [e["action"]] for i, e in choices))
        if family == DEEP_FAMILY and by_depth[4] != [None, None]:
            raise ValueError(f"four public rule rounds already reach a witness in {row['case']['tid']}")
        case = SearchCase(**row["case"])
        rows.append({"root": case.root, "cluster": case.cluster, "winner": row["winner"],
                     "full_costs": exact, "costs_by_hop": by_depth, "certified_script_by_hop": scripts,
                     "choices": len(choices)})

    clusters = {row["cluster"] for row in rows}
    sides = Counter((row["cluster"], row["winner"]) for row in rows)
    if not clusters or any(sides[cluster, action] != 1 for cluster in clusters for action in FORK):
        raise ValueError("the witness receipt does not contain one pair per layout")
    counts = []
    for hops in hop_counts:
        correct = Counter(row["cluster"] for row in rows
                          if deadline_actions(FORK, row["costs_by_hop"][hops], budget) == [row["winner"]])
        counts.append({"hops": hops, "correct_worlds": correct.total(),
                       "complete_pairs": sum(n == 2 for n in correct.values()),
                       "certified_scripts": sum(row["certified_script_by_hop"][hops] for row in rows)})
    return {"schema": "ppt.shared-witness-public-cost.v1", "receipt_sha256": sha256_file(receipt),
            "worlds": len(rows), "pairs": len(clusters), "budget": budget, "counts": counts, "rows": rows}


def audit_workspace(receipt):
    """Compare retained and reset support using the same fixed gates and native decisions."""
    import torch

    from ppt_gym.slot_witness_bank import SlotWitnessBank, WitnessWorkspace

    bank = SlotWitnessBank(4)
    cases = read_receipt(receipt)["rows"]
    results = {}
    with torch.no_grad():
        for name in ("reset", "persistent"):
            rows, max_nodes = [], 0
            for row in cases:
                trial = row["trials"][FORK.index(row["winner"])]
                if not trial["success"] or not trial["certificate"]["passed"] or trial["certificate"]["axioms"]:
                    raise ValueError("the workspace audit requires an independently certified winning script")
                matches, remembered = [], None
                for index, event in enumerate(trial["events"]):
                    menu = event["menu"]
                    view, values, gates, keys = bank.resume(event["goals"], menu, remembered)
                    for _ in range(2):
                        values, _ = bank.advance(view, values, gates)
                    max_nodes = max(max_nodes, len(values))
                    if name == "persistent":
                        remembered = WitnessWorkspace(keys, values)
                    if len(menu) > 1:
                        scores = values[view.read_indices].reshape(view.shape).prod(1)
                        peak = scores.max()
                        best = [action for action, score in zip(menu, scores, strict=True) if score == peak and score > 0]
                        matches.append({"index": index, "action": event["action"], "best": best})
                case = SearchCase(**row["case"])
                rows.append({"root": case.root, "cluster": case.cluster, "decisions": matches,
                             "script_matched": all(m["best"] == [m["action"]] for m in matches)})
            pairs = Counter(row["cluster"] for row in rows if row["script_matched"])
            results[name] = {"scripts": sum(pairs.values()), "complete_pairs": sum(count == 2 for count in pairs.values()),
                             "max_workspace_nodes": max_nodes, "rows": rows}
    return {"schema": "ppt.shared-witness-workspace-audit.v1", "receipt_sha256": sha256_file(receipt),
            "hops_per_tactic": 4, "results": results}


def verify_late_credit(repl, tree, receipt):
    """Verify the two proof costs and the dead ends using the fit's credit protocol."""
    from configs.run_ppt_slot_continuations import training_return

    rows = []
    for row in read_receipt(receipt)["rows"]:
        case = SearchCase(**row["case"])
        if case.family != DEEP_FAMILY or case.split != "train":
            raise ValueError("late credit verification expects the deep training receipt")
        winner = FORK.index(row["winner"])
        on_time = row["trials"][winner]
        late = row["late"]
        prefix = tuple(event["action"] for event in late["events"][:-1])
        credited = trajectory(repl, tree, case, forced_prefix=prefix, late_proof_credit=True)
        if (not credited["success"] or not credited["events"][-1].get("certificate_only")
                or canonical(credited["events"][:-1]) != canonical(late["events"][:-1])
                or credited["certificate"]["source_digest"] != late["certificate"]["source_digest"]):
            raise ValueError(f"the forced closure did not reproduce the registered late proof for {case.tid}")
        decoys = []
        for trace in (on_time, late):
            events = trace["events"]
            for i, event in enumerate(events):
                if i <= FORK_INDEX or len(event["menu"]) == 1:
                    continue
                before = tuple(e["action"] for e in events[:i])
                for choice in event["menu"]:
                    if choice == event["action"]:
                        continue
                    attempt = trajectory(repl, tree, case, forced_prefix=(*before, choice), late_proof_credit=True)
                    if (not attempt["within"] or attempt["success"] or len(attempt["events"]) != i + 1
                            or canonical(attempt["events"][:i]) != canonical(events[:i])):
                        raise ValueError(f"the decoy failed to reproduce its registered dead end for {case.tid}")
                    decoys.append({"action": choice, "tactics": i + 1, "state_after": attempt["events"][-1]["native_after"],
                                   "reward": training_return(attempt, True)})
        rows.append({"root": case.root, "cluster": case.cluster, "faster": training_return(on_time, True),
                     "slower": training_return(credited, True), "decoys": decoys,
                     "certificate_sha256": credited["certificate"]["source_digest"]})
    return {"schema": "ppt.deep-witness-late-credit.v1", "receipt_sha256": sha256_file(receipt),
            "lean": repl.version, "server_sha256": sha256_file(REPL_EXE),
            "verifier_sha256": sha256_file(LEAN_EXE), "rows": rows}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_mutually_exclusive_group()
    modes.add_argument("--cohort", choices=("train", "dev"))
    modes.add_argument("--deep-cohort", choices=("train", "dev", "transfer", "invariant"))
    modes.add_argument("--audit-receipt", type=Path)
    modes.add_argument("--audit-workspace", type=Path)
    modes.add_argument("--verify-late-credit", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.audit_receipt is not None or args.audit_workspace is not None:
        if args.audit_workspace is not None:
            from ppt_gym.ram_campaign import configure

            configure(1, 2201)
            result = audit_workspace(args.audit_workspace)
        else:
            result = audit_public_cost(args.audit_receipt)
        with args.output.open("x", encoding="utf-8") as target:
            target.write(canonical(result) + "\n")
        return
    tree = TreeBudget(2, 1)
    try:
        with LeanRepl() as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("the dependent witness requires the pinned Lean verifier")
            if args.verify_late_credit is not None:
                result = verify_late_credit(repl, tree, args.verify_late_credit)
            else:
                if args.cohort is None and args.deep_cohort is None:
                    schema, rows = "ppt.shared-existential-witness.v1", verify_pair(repl)
                else:
                    schema = "ppt.shared-existential-continuation.v1"
                    cases = deep_cohorts(args.deep_cohort) if args.deep_cohort else cohorts(args.cohort)
                    rows = verify_cohort(repl, tree, cases)
                result = {"schema": schema, "lean": repl.version, "rows": rows,
                          "server_sha256": sha256_file(REPL_EXE), "verifier_sha256": sha256_file(LEAN_EXE)}
        with args.output.open("x", encoding="utf-8") as target:
            target.write(canonical(result) + "\n")
    finally:
        tree.close()


if __name__ == "__main__":
    main()
