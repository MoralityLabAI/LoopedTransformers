"""Seal a privately priced plan menu with independently checked Lean certificates."""

import argparse
from collections import defaultdict
from pathlib import Path
import random
from tempfile import TemporaryDirectory

from configs.freeze_ppt_priced_fragments import certify, roots
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_native_progress import verify_independent
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt import fragment
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE, lean_env
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.private_quotes import COMMITS, QUOTES, REGIMES, private_steps, rollout, scenario, structure, tariff_layout
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.private-priced-proof-plans.v1"


def eligible(schedules):
    plans = tuple(schedule["reference"] for schedule in schedules)
    first, second = private_steps(plans)
    return len(plans[0]) == len(plans[1]) and min(len(first), len(second)) >= 3


def register(repl, case):
    case = certify(repl, case)
    schedules = case.pop("schedules")
    plans = [schedule["reference"] for schedule in schedules]
    rng = random.Random(int(digest([SCHEMA, case["module"]])[:15], 16))
    rng.shuffle(plans)
    quote_order = list(QUOTES)
    rng.shuffle(quote_order)
    ledger, other = tariff_layout(plans)
    return {**case, "plans": plans, "ledger": ledger, "other": other, "quote_order": quote_order}


def audit(held):
    arms = {"exact": {}, "one_step": {"one_step": True}, "always_quote": {"always_quote": True},
            "always_affordable_quote": {"always_quote": True, "reserve_proof_effort": True},
            "commit_first": None, "swap_reconciliation": {}}
    rows = defaultdict(list)
    for case in held:
        for name, regime in REGIMES.items():
            for world in range(4):
                prices = scenario(case["plans"], case["ledger"], case["other"], regime, world)
                costs = [sum(prices[tactic] for tactic in plan) for plan in case["plans"]]
                base = len(case["plans"][0]) + 2
                expected = [base + regime.surcharge * int(index != ((world // 2) ^ (world % 2)))
                            for index in range(2)]
                if costs != expected:
                    raise ValueError("a proof plan's actual prices violated the sealed budget design")
                for arm, kwargs in arms.items():
                    if kwargs is None:
                        def select(menu, history, left, fees):
                            return COMMITS[0]
                    else:
                        select = structure(case, name, **kwargs)
                    patch = flip_reconciliation if arm == "swap_reconciliation" else None
                    row = rollout(case, name, world, select, patch=patch)
                    rows[arm].append({"root": case["module"], "regime": name, "world": world, **row})
    return dict(rows)


def flip_reconciliation(action, bit):
    return 1 - bit if action == "reconcile" else bit


def run(args, tree):
    previous = {"source": [], "held": []}
    for path in args.exclude:
        ledger = read_receipt(path)
        for partition in previous:
            previous[partition].extend(ledger[partition])
    source, held = roots(previous, schema=SCHEMA, eligible=eligible)
    with LeanRepl() as repl:
        if repl.version != LEAN_VERSION:
            raise ValueError("unexpected native Lean version")
        source = [register(repl, case) for case in source]
        held = [register(repl, case) for case in held]
    with TemporaryDirectory(prefix="ppt-quote-certs-") as scratch:
        independent = verify_independent(Path(scratch), lean_env(), (),
                                         [(case, plan) for case in source + held for plan in case["plans"]])
    bank = {"schema": SCHEMA, "source": source, "held": held, "independent": independent,
            "generator_sha256": sha256_file(Path(fragment.__file__)),
            "lean_sha256": sha256_file(LEAN_EXE), "repl_sha256": sha256_file(REPL_EXE),
            "exclusions": {path.name: sha256_file(path) for path in args.exclude}, "memory": tree.stats()}
    write_receipt(args.output, bank)
    rows = audit(held)
    for arm, outcomes in rows.items():
        print(arm, "proofs", sum(row["proof"] for row in outcomes), "/", len(outcomes),
              "charges", sum(row["quote_effort"] for row in outcomes),
              "mean return", sum(row["utility"] for row in outcomes) / len(outcomes), flush=True)
    write_receipt(args.audit, {"schema": SCHEMA, "registry_sha256": sha256_file(args.output),
                               "controls": rows, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("exclude", nargs="+", type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        run(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
