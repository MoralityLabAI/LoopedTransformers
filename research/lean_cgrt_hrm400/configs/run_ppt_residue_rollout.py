"""Measure headroom in a native Lean task with adaptive checks and proof attempts."""

import argparse
from collections import defaultdict
from math import fsum
from pathlib import Path

import torch

from configs.freeze_ppt_failed_prefix import route_attempt, setup
from configs.freeze_ppt_guard_recovery import certify_routes
from configs.freeze_ppt_native_progress import advance, native_sources
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_residue_acquisition import CALIBRATORS, experience, inputs, restore
from configs.run_ppt_residue_recovery import BATCH, UPDATES, prepare, train
from configs.run_ppt_residue_recovery import SCHEMA as RESIDUE_SCHEMA
from configs.run_ppt_residue_recovery import curves, distribution, models_for, public_tokens
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_VERSION
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest
from ppt_gym.failed_prefix import ALLOWANCES, PREFIX_COST
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.residue_rollout import FIRST_QUERY, FIRST_ROUTE, Recovery, State, greedy, mask


SCHEMA = "ppt.residue-rollout-preflight.v1"
SEED = 1157
DEPTHS = (0, 1, 2)
FLOOR = .025
HEADROOM = .075


def frozen(args, pins):
    checkpoint = torch.load(args.calibration, map_location="cpu", weights_only=True)
    if (checkpoint["schema"] != RESIDUE_SCHEMA
            or checkpoint["pins"] != {name: pins[name] for name in ("bank", "warm", "prior")}):
        raise ValueError("the frozen residue calibrators have different input pins")
    readers = models_for()
    return {name: restore(readers[name], checkpoint, name) for name in CALIBRATORS}


def read_priors(pair, experience_row, prior, calibrators):
    choice, observed, _, support, warm, uniform = experience_row
    public, _ = public_tokens(pair, choice, observed, warm, curves(pair["second"], prior))
    features, allowed = torch.tensor(public), torch.tensor(support)
    return {"uniform": uniform, **{name: distribution(model(features).flatten(), allowed)
                                  for name, model in calibrators.items()}}


def folded_calibrators(pairs, prior):
    source = [pair for pair, _ in pairs]
    expressions = [pair["certificate"].rsplit(" = ", 1)[0] for pair in source]
    ordered = sorted(set(expressions), key=lambda expression: digest([SEED, expression]))
    group = {expression: index % 4 for index, expression in enumerate(ordered)}
    data = prepare(source, prior)
    readers_by_pair, folds = {}, []
    with torch.enable_grad():
        for fold in range(4):
            train_ids = [i for i, expression in enumerate(expressions) if group[expression] != fold]
            test_ids = [i for i, expression in enumerate(expressions) if group[expression] == fold]
            sampler = torch.Generator().manual_seed(SEED + fold)
            batches = torch.randint(len(train_ids), (UPDATES, BATCH), generator=sampler)
            training = tuple(piece[train_ids] for piece in data)
            previous = models_for()
            calibrators = {name: previous[name] for name in CALIBRATORS}
            for name, model in calibrators.items():
                train(model, name, training, batches)
            for i in test_ids:
                readers_by_pair[source[i]["pair"]] = calibrators
            folds.append({"train": sorted({expressions[i] for i in train_ids}),
                          "test": sorted({expressions[i] for i in test_ids}),
                          "sampler_sha256": tensor_hash({"batches": batches}),
                          "weights": {name: tensor_hash(model.state_dict()) for name, model in calibrators.items()}})
            print("fitted expression fold", fold, len(train_ids), len(test_ids), flush=True)
    return readers_by_pair, folds


def collect(pairs, prior, readers):
    records = defaultdict(list)
    for pair, native in pairs:
        observed = experience(pair, prior)
        _, _, _, allowed, _, _ = observed
        support = mask(i for i, supported in enumerate(allowed) if supported)
        priors = read_priors(pair, observed, prior, readers[pair["pair"]])
        priors["actual_residue"] = [float(i == pair["residue"]) for i in range(len(priors["uniform"]))]
        target = pair["second"]
        env = Recovery(target, native["menu"])
        expression = pair["certificate"].rsplit(" = ", 1)[0]
        for name, weights in priors.items():
            scores = env.planner(weights)

            def decide(state, _history):
                return greedy(scores(state))

            for budget in ALLOWANCES:
                for depth in (0,) if name == "actual_residue" else DEPTHS:
                    start = State(budget, support, depth)
                    row = env.rollout(start, target["truth"], native["outcomes"], decide)
                    records[f"{name}_{depth}"].append({**row, "pair": pair["pair"], "expression": expression,
                                                       "depth": depth, "forecast": max(scores(start).values())})
    return records


def summary(rows):
    traces = [[step["action"] for step in row["trace"]] for row in rows]
    purchases = [sum(action >= FIRST_QUERY for action in trace) for trace in traces]
    return {"cases": len(rows), "proofs": sum(row["proved"] for row in rows),
            "checks": sum(row["checks"] for row in rows), "queries": sum(purchases),
            "two_queries": sum(count == 2 for count in purchases),
            "query_after_route": sum(any(a < FIRST_QUERY <= b for a, b in zip(trace, trace[1:]))
                                     for trace in traces),
            "utility": fsum(row["utility"] for row in rows) / len(rows)}


def sample_paths(records):
    for name, rows in records.items():
        for depth in range(3):
            candidates = [row for row in rows if
                          sum(event["action"] >= FIRST_QUERY for event in row["trace"]) == depth]
            if candidates:
                yield name, max(candidates, key=lambda row: (row["proved"], len(row["trace"])))
        interleaved = next((row for row in rows if any(a["action"] < FIRST_QUERY <= b["action"]
                            for a, b in zip(row["trace"], row["trace"][1:]))), None)
        if interleaved is not None:
            yield name, interleaved


def replay(repl, pair, native, row):
    target = pair["second"]
    fork, saved, rejection = setup(repl, target)
    if (fork[1], saved[1], rejection) != (target["fork"], target["saved"], target["initial_rejection"]):
        raise ValueError("a rollout restored a different native proof state")
    calls = PREFIX_COST
    for event in row["trace"]:
        action, passed = event["action"], event["passed"]
        if action >= FIRST_QUERY:
            status, _, _ = advance(repl, *saved, native["menu"][action - FIRST_QUERY]["tactic"])
            if status != ("progress" if passed else "rejected") or event["charge"] != 1:
                raise ValueError("a paid query changed in its native rollout")
            calls += 1
        else:
            index = action - FIRST_ROUTE
            route = target["routes"][index]
            trace = route_attempt(repl, route, saved if index == 0 else fork, passed)
            if trace != route["trace"] or len(trace) != event["charge"]:
                raise ValueError("a paid route changed in its native rollout")
            calls += len(trace)
    if calls != row["checks"]:
        raise ValueError("the native rollout check count changed")


def audit(args, pairs, records):
    bank = {pair["pair"]: (pair, native) for pair, native in pairs}
    used = {(row["pair"], row["trace"][-1]["action"] - FIRST_ROUTE)
            for rows in records.values() for row in rows if row["proved"]}
    scripts = []
    for key, index in sorted(used):
        target = bank[key][0]["second"]
        scripts.append((target, target["routes"][index]["proof"]))
    samples = []
    with native_sources((args.bpd, args.mathcan)) as (directory, env, imports):
        independent = certify_routes(directory, env, imports, scripts)
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("the native rollout toolchain changed")
            for name, row in sample_paths(records):
                replay(repl, *bank[row["pair"]], row)
                samples.append({"policy": name, **row})
    return independent, samples


@torch.no_grad()
def run(args, tree):
    pairs, pins = inputs(args, args.partition)
    prior = MixedSourcePrior(args)
    folds = []
    if args.folded:
        if args.partition != "source":
            raise ValueError("the expression folds must come from the source bank")
        readers, folds = folded_calibrators(pairs, prior)
    else:
        calibrators = frozen(args, pins)
        readers = {pair["pair"]: calibrators for pair, _ in pairs}
    records = collect(pairs, prior, readers)
    summaries = {name: summary(rows) for name, rows in records.items()}
    for name, result in summaries.items():
        print(name, result, flush=True)
    best = max((key for key in summaries if key != "actual_residue_0"),
               key=lambda key: summaries[key]["utility"])
    gap = summaries["actual_residue_0"]["utility"] - summaries[best]["utility"]
    checks, samples = audit(args, pairs, records)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "partition": args.partition, "folds": folds,
                                 "floor": FLOOR, "training_headroom": HEADROOM, "oracle_gap": gap,
                                 "strongest": best, "summary": summaries, "records": records,
                                 "independent": checks, "live_replays": samples, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partition", choices=("source", "development"))
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "native", "calibration", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--folded", action="store_true", help="train comparators with expression-out source folds")
    tree = TreeBudget(2, 1)
    try:
        configure(1, SEED)
        run(parser.parse_args(), tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
