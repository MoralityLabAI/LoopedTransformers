"""Collect short native Lean rollouts for paid evidence inside proof search."""

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from functools import partial
from pathlib import Path
import random
from tempfile import TemporaryDirectory

from configs.freeze_ppt_bound_quotes import admissible, seal
from configs.freeze_ppt_native_progress import advance
from configs.freeze_ppt_priced_fragments import checked_root, roots
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_native_progress import verify_independent
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt import fragment
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE, lean_env
from lcgrt.lean_repl import LeanRepl
from ppt_gym.bound_quotes import REGIMES, outcomes, public_view
from ppt_gym.data import digest, sha256_file
from ppt_gym.live_lean import candidate_menu
from ppt_gym.live_quote_fragments import NativeFragments, NativeQuoteBackup
from ppt_gym.private_quotes import exact_policy, metered_rollout
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.live-rule-quote-fragments.v1"
DETOUR_SCHEMA = "ppt.live-rule-quote-late-detour.v1"
SOURCE_TAPES = 28
DIAGNOSTIC_TAPES = 64


def collect_native(repl, case, limit, *, menu_for=candidate_menu):
    rng = random.Random(int(digest([SCHEMA, case["module"]])[:15], 16))
    state, public = checked_root(repl, case)
    root = digest(public)
    if root != case["visible_sha256"]:
        raise ValueError("the Lean root differs from the certified reference")
    snapshots = {root: (state, 0)}
    states = {root: {"public": public, "menu": menu_for(repl, state)}}
    untried = {root: list(states[root]["menu"])}
    fragments = []
    for number in range(limit):
        active = [key for key, moves in untried.items() if moves]
        if not active:
            break
        # Alternate newly discovered suffix anchors with oldest incomplete anchors.
        start = (active[0] if number % 3 == 0 else max(active, key=lambda key: snapshots[key][1]))
        key, steps = start, []
        for _ in range(2):
            moves = untried[key]
            if not moves:
                break
            action = moves.pop(rng.randrange(len(moves)))
            snapshot, depth = snapshots[key]
            status, child, after = advance(repl, snapshot, states[key]["public"], action)
            new = digest(after) if status == "progress" else None
            steps.append({"before": key, "action": action, "status": status, "after": new})
            if new is None:
                break
            if new not in states:
                states[new] = {"public": after, "menu": menu_for(repl, child)}
                snapshots[new] = child, depth + 1
                untried[new] = list(states[new]["menu"])
            key = new
        fragments.append({"start": start, "steps": steps})
    log = {"root": root, "states": states, "fragments": fragments}
    NativeFragments(log)
    return log


def meter(case, native):
    checked = {action for edges in native.edges.values() for action in edges}
    # The price meter is observed for a tactic only where Lean ran that tactic.
    return [{"prices": {action: prices.get(action, 1) for action in sorted(checked)}, "bits": bits}
            for prices, bits in outcomes(case)]


def source_entry(repl, case):
    log = collect_native(repl, case, SOURCE_TAPES)
    billing = meter(case, NativeFragments(log))
    # The fit can use typed links and names, but receives neither certified reference scripts nor their outcomes.
    views = {regime: {key: value for key, value in public_view(case, regime).items()
                     if key not in ("plans", "links", "min_proof_effort")}
             for regime in REGIMES}
    return {**{key: value for key, value in case.items() if key != "plans"},
            "native": log, "billing": billing, "views": views}


def has_detour_site(schedules):
    if not admissible(schedules, shared=True):
        return False
    first, second = (schedule["reference"] for schedule in schedules)
    return first[-2:] == second[-2:] and first[-2].startswith("apply r") and first[-1].startswith("apply h")


def with_late_detour(repl, case):
    """Give the shared final obligation a distinct two-tactic native derivation."""
    theorem = fragment.Theorem.from_json(case["graph"])
    leading, witness = case["schedules"][0]["reference"][-2:]
    premises, atom = theorem.rules[int(leading.removeprefix("apply r"))]
    if premises != [theorem.facts[int(witness.removeprefix("apply h"))]]:
        raise ValueError("the late tariff site is not a shared two-tactic unary derivation")
    fresh = theorem.n_atoms
    new_rule = f"apply r{len(theorem.rules)}"
    modified = replace(theorem, n_atoms=fresh + 1, rules=[*theorem.rules, ([fresh], atom)],
                       facts=[*theorem.facts, fresh])
    changed = {**case, "ancestor": case["module"], "module": fragment.canonical_hash(modified),
               "statement": fragment.statement(modified), "graph": modified.to_json()}
    result = seal(repl, changed, shared=True)
    shared = next(name for name, channel in result["channels"].items()
                  if all(channel["rule"] in plan for plan in result["plans"]))
    result["channels"][shared]["rule"] = leading
    result["late_link"] = {"quote": shared, "counterpart": new_rule}
    return result


def audited_source(source):
    rows, certificates = [], {}
    for case in source:
        native = NativeFragments(case["native"])
        if any(row["start"] == native.root and row["steps"][-1]["status"] == "closed" for row in case["native"]["fragments"]):
            raise ValueError("a source tape provided a root-to-proof label")
        details = []
        for name, view in case["views"].items():
            model = NativeQuoteBackup(native, view["fees"], case["billing"])
            budget = view["budget"]
            lower, upper = model.value(native.root, budget, 8, model.initial_belief, 3, 0)
            proofs = []
            for world in model.initial_belief:
                row = model.solve(world, budget, 3)
                if row["proof"]:
                    proofs.append(row)
                    certificates[case["module"], tuple(row["script"])] = case
            details.append({"regime": name, "verified": proofs, "utility_bounds": [lower, upper]})
            model.action.cache_clear()
            model.value.cache_clear()
        rows.append({"root": case["module"], "family": case["family"], "regimes": details})
    return rows, certificates


def audited_development(held, logs):
    records, certificates = defaultdict(list), {}
    for case, log in zip(held, logs, strict=True):
        native = NativeFragments(log)
        scenarios = outcomes(case)
        billing = meter(case, native)
        for regime in REGIMES:
            view = public_view(case, regime)
            budget = view["budget"]
            model = NativeQuoteBackup(native, view["fees"], billing)
            fixed = exact_policy(view, scenarios)
            for world, (prices, bits) in enumerate(scenarios):
                results = {"live_continuation": model.solve(world, budget, 3),
                           "live_single_quote": model.solve(world, budget, 1),
                           "live_no_quotes": model.solve(world, budget, 0),
                           "fixed_reference_plans": metered_rollout(view, prices, bits, fixed)}
                for arm, row in results.items():
                    if row["proof"]:
                        certificates[case["module"], tuple(row["script"])] = case
                    if len(row["script"]) > 8:
                        raise ValueError("the native tactic cap was exceeded")
                    records[arm].append({"root": case["module"], "family": case["family"], "regime": regime, "world": world,
                                         "reference": row["script"] in case["plans"], **row})
            model.action.cache_clear()
            model.value.cache_clear()
    return dict(records), certificates


def run(args, tree):
    exclusions = {"source": [], "held": []}
    for path in args.exclude:
        ledger = read_receipt(path)
        for part in exclusions:
            exclusions[part].extend(ledger[part])
    schema = DETOUR_SCHEMA if args.late_detour else SCHEMA
    scope = has_detour_site if args.late_detour else partial(admissible, shared=True)
    raw_source, raw_held = roots(exclusions, schema=schema, eligible=scope)
    with LeanRepl() as repl:
        if repl.version != LEAN_VERSION:
            raise ValueError("unexpected native Lean version")
        if args.late_detour:
            source_cases = [with_late_detour(repl, case) for case in raw_source]
            held = [with_late_detour(repl, case) for case in raw_held]
            observed_ids = {case["module"] for case in source_cases + held}
            earlier_ids = {case["module"] for partition in exclusions.values() for case in partition}
            if len(observed_ids) != len(source_cases) + len(held) or observed_ids & earlier_ids:
                raise ValueError("a late-detour theorem duplicates an earlier root")
        else:
            source_cases = [seal(repl, case, shared=False) for case in raw_source]
            held = [seal(repl, case, shared=True) for case in raw_held]
        source = [source_entry(repl, case) for case in source_cases]
        logs = [collect_native(repl, case, DIAGNOSTIC_TAPES) for case in held]
    source_audit, source_proofs = audited_source(source)
    development, development_proofs = audited_development(held, logs)
    all_proofs = {**source_proofs, **development_proofs}
    with TemporaryDirectory(prefix="ppt-joint-lean-certificates-") as directory:
        verified = verify_independent(Path(directory), lean_env(), (), [(case, key[1]) for key, case in all_proofs.items()])
    stats = Counter(step["status"] for case in source for tape in case["native"]["fragments"] for step in tape["steps"])
    write_receipt(args.bank, {"schema": schema, "source": source, "held": held,
                             "exclusions": {path.name: sha256_file(path) for path in args.exclude},
                             "generator": sha256_file(Path(fragment.__file__)), "lean": sha256_file(LEAN_EXE),
                             "repl": sha256_file(REPL_EXE), "source_checks": stats, "memory": tree.stats()})
    write_receipt(args.audit, {"schema": schema, "bank_sha256": sha256_file(args.bank), "source": source_audit,
                              "held_graphs": logs, "development": development, "independent": verified,
                              "source_certificates": len(source_proofs), "development_certificates": len(development_proofs),
                              "memory": tree.stats()})
    print("source checks", dict(stats), "stitched scripts", len(source_proofs), flush=True)
    for arm, rows in development.items():
        print(arm, "proofs", sum(row["proof"] for row in rows), "/", len(rows),
              "unregistered", sum(row["proof"] and not row["reference"] for row in rows),
              "utility", sum(row["utility"] for row in rows) / len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("exclude", type=Path, nargs="+")
    parser.add_argument("--late-detour", action="store_true")
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        run(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
