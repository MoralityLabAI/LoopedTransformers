"""Seal distinct rule-bound proof plans and collect source-only quote fragments."""

import argparse
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from configs.freeze_ppt_priced_fragments import certify, roots
from configs.freeze_ppt_private_quotes import eligible
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_native_progress import verify_independent
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt import fragment
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE, lean_env
from lcgrt.lean_repl import LeanRepl
from ppt_gym.bound_quotes import REGIMES, bind, outcomes, public_view, shared_steps
from ppt_gym.data import sha256_file
from ppt_gym.private_quotes import COMMITS, exact_policy, metered_rollout
from ppt_gym.quote_fragments import QuoteFragmentGraph, taped_fragments
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.rule-bound-priced-plans.v1"


def admissible(schedules, *, shared):
    return eligible(schedules) and (not shared or bool(shared_steps([item["reference"] for item in schedules])))


def seal(repl, case, *, shared):
    checked = certify(repl, case)
    schedules = checked.pop("schedules")
    return bind({**checked, "plans": [entry["reference"] for entry in schedules]}, shared=shared)


def audit(held):
    records = defaultdict(list)
    for case in held:
        scenarios = outcomes(case)
        swapped = {**case, "plans": case["plans"][::-1]}
        if outcomes(swapped) != scenarios:
            raise ValueError("a presentation swap changed the physical prices")
        for regime in REGIMES:
            view = public_view(case, regime)
            counterfactual = public_view(swapped, regime)
            for name, incidence in view["links"].items():
                changed = counterfactual["links"][name]
                if incidence["direct"] != changed["direct"][::-1] or incidence["inverse"] != changed["inverse"][::-1]:
                    raise ValueError("quote occurrence counts failed to follow the literal tactics")
            arms = {"exact": exact_policy(view, scenarios),
                    "one_step": exact_policy(view, scenarios, one_step=True),
                    "commit_first": lambda menu, *_: menu[0]}
            mirror = exact_policy(counterfactual, scenarios)
            for world, (prices, bits) in enumerate(scenarios):
                for arm, select in arms.items():
                    row = metered_rollout(view, prices, bits, select)
                    records[arm].append({"root": case["module"], "family": case["family"],
                                         "regime": regime, "world": world, **row})
                    if arm == "exact":
                        def choose(menu, *args):
                            return mirror((menu[1], menu[0], *menu[2:]), *args)

                        mirrored = metered_rollout(counterfactual, prices, bits, choose)
                        if mirrored != {**row, "plan": COMMITS[1 - COMMITS.index(row["plan"])]}:
                            raise ValueError("the exact policy lost physical identity after the slot swap")
    return dict(records)


def collect(cases):
    records, summary = [], Counter()
    for case in cases:
        scenarios = outcomes(case)
        for regime in REGIMES:
            view = public_view(case, regime)
            fragments = taped_fragments(case["module"], regime, view, scenarios)
            graph = QuoteFragmentGraph(view, fragments)
            for _, action, (backup, empirical) in graph.rows():
                summary.update(actions=1, observed_suffix=empirical is not None, stitched=backup is not None)
                if action not in COMMITS:
                    summary.update(quote_actions=1, stitched_only=backup is not None and empirical is None)
            summary.update(fragments=len(fragments), decisions=len(graph.states),
                           recorded_events=sum(len(row["steps"]) for row in fragments))
            records.append({"module": case["module"], "regime": regime, "view": view, "fragments": fragments})
    return records, summary


def run(args, tree):
    previous = {"source": [], "held": []}
    for path in args.exclude:
        ledger = read_receipt(path)
        for part in previous:
            previous[part].extend(ledger[part])
    raw_source, raw_held = roots(previous, schema=SCHEMA,
                                 eligible={"source": partial(admissible, shared=False),
                                           "held": partial(admissible, shared=True)})
    with LeanRepl() as repl:
        if repl.version != LEAN_VERSION:
            raise ValueError("unexpected native Lean version")
        source = [seal(repl, case, shared=False) for case in raw_source]
        held = [seal(repl, case, shared=True) for case in raw_held]
    with TemporaryDirectory(prefix="ppt-bound-quote-certs-") as scratch:
        independent = verify_independent(Path(scratch), lean_env(), (),
                                         [(case, plan) for case in source + held for plan in case["plans"]])
    bank = {"schema": SCHEMA, "source": source, "held": held, "independent": independent,
            "generator_sha256": sha256_file(Path(fragment.__file__)),
            "lean_sha256": sha256_file(LEAN_EXE), "repl_sha256": sha256_file(REPL_EXE),
            "exclusions": {path.name: sha256_file(path) for path in args.exclude}, "memory": tree.stats()}
    write_receipt(args.bank, bank)
    records, summary = collect(source)
    bank_sha = sha256_file(args.bank)
    write_receipt(args.fragments, {"schema": SCHEMA, "bank_sha256": bank_sha, "source": records,
                                   "source_summary": summary, "memory": tree.stats()})
    print("source", dict(summary), flush=True)
    controls = audit(held)
    for name, rows in controls.items():
        print(name, sum(row["proof"] for row in rows), "/", len(rows),
              sum(row["utility"] for row in rows) / len(rows), flush=True)
    write_receipt(args.audit, {"schema": SCHEMA, "bank_sha256": bank_sha,
                               "controls": controls, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "fragments", "audit"):
        parser.add_argument(name, type=Path)
    parser.add_argument("exclude", nargs="+", type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        run(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
