"""Freeze and score a single unseen-expression Lean probe cohort."""

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from math import fsum
from pathlib import Path
import random

import torch

from configs.freeze_ppt_guard_recovery import certify_routes, verify_case
from configs.freeze_ppt_native_progress import native_sources
from configs.run_ppt_guard_belief import audit_routes, root_contrast
from configs.run_ppt_guard_probe import INTERVENTIONS, corrections, score_case
from configs.run_ppt_guard_recovery import encode_case, summary
from configs.run_ppt_guard_residual import scaled
from configs.run_ppt_guard_warm_probe import ADAPTERS, SCHEMA as FIT_SCHEMA
from configs.run_ppt_guard_warm_probe import checked_inputs, load_weights, make_anchors
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_VERSION
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import sha256_file
from ppt_gym.guard_probe import ProbeGraph
from ppt_gym.guard_recovery import ALLOWANCES, feasible_menu
from ppt_gym.guard_recovery_learning import baked_scores
from ppt_gym.guard_transfer import expressions, public_digest, unseen_pairs
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_limits import TreeBudget


LOCK = Path(__file__).with_name("ppt_guard_probe_transfer_lock.json")
SCHEMA = "ppt.guard-probe-transfer.v1"
SEED = 1127


def locked(args):
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    hashes = {key: sha256_file(getattr(args, key)) for key in lock["inputs"]}
    if lock["schema"] != "ppt.guard-probe-transfer-lock.v1" or hashes != lock["inputs"]:
        raise ValueError("the registered probe transfer inputs changed")
    bank, deep, pins, reader, _ = checked_inputs(args)
    return lock, bank["source"] + bank["development"] + deep, pins, reader


def freeze(args, tree):
    lock, previous, pins, _ = locked(args)
    plan, audit = unseen_pairs(previous)
    if public_digest(plan) != lock["public_plan_sha256"]:
        raise ValueError("the scheduled public theorem pairs changed")
    with native_sources((args.bpd, args.mathcan)) as (directory, env, imports):
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected transfer Lean version")
            cohort = [verify_case(repl, case) for case in plan]
        certificates = [(case, route["proof"]) for case in cohort for route in case["routes"].values() if route["proof"]]
        independent = certify_routes(directory, env, imports, certificates)
    if public_digest(cohort) != lock["public_plan_sha256"]:
        raise ValueError("native certification changed the public transfer plan")
    write_receipt(args.cohort, {"schema": SCHEMA, "lock_sha256": sha256_file(LOCK), "pins": pins,
                                "cohort": cohort, "audit": audit, "independent": independent, "memory": tree.stats()})
    print("sealed unseen-expression roots", audit["roots"], "native scripts", independent["certificates"], flush=True)


def stratified_interval(contrast, rows, draws=4000):
    """Resample whole pairs within family for a descriptive interval."""
    families = defaultdict(set)
    for row in rows:
        families[row["family"]].add(row["pair"])
    groups = [[contrast["effects"][root] for root in sorted(roots)] for _, roots in sorted(families.items())]
    rng = random.Random(SEED)
    count = sum(map(len, groups))
    means = sorted(fsum(rng.choice(group) for group in groups for _ in group) / count for _ in range(draws))
    return [means[draws // 40], means[draws - draws // 40 - 1]]


def probe_models(args, pins, lock):
    fit = torch.load(args.fit, map_location="cpu", weights_only=True)
    if (fit["schema"] != FIT_SCHEMA or fit["pins"] != pins
            or any(fit["data_sha"][name] != lock["inputs"][name] for name in ("curriculum", "warm_probe"))):
        raise ValueError("the locked probe checkpoint disagrees with its source")
    anchors = make_anchors()
    for name, model in anchors.items():
        load_weights(model, fit["anchors"][name], fit["anchor_sha"][name])
    models = {}
    for readout, (arm, blend) in lock["adapters"].items():
        name = ADAPTERS[arm][0]
        model = deepcopy(anchors[name])
        load_weights(model, fit["weights"][arm], fit["weight_sha"][arm])
        models[readout] = model, name, blend
    return anchors, models


def score_world(graph, case, descriptors, probability, warm, lock, anchors, models):
    encoded = graph.encode(descriptors, probability, warm)
    tokens = torch.tensor([encoded[0]])
    anchor_deltas = {name: corrections(model, tokens)[0] for name, model in anchors.items()}
    updates = {}
    for arm, (model, name, blend) in models.items():
        anchor = anchor_deltas[name]
        proposal = corrections(model, tokens)[0]
        updates[arm] = (proposal if blend == 1. else anchor + blend * (proposal - anchor)).tolist()
    anchor_deltas = {name: values.tolist() for name, values in anchor_deltas.items()}
    base = baked_scores(probability, action_space=feasible_menu)
    temperature = baked_scores(scaled(probability, lock["frozen_temperature"]), action_space=feasible_menu)
    for budget in ALLOWANCES:
        results = {"frozen_planner": score_case(case, budget, graph, base, encoded),
                   "temperature_calibration": score_case(case, budget, graph, temperature, encoded)}
        for name, delta in anchor_deltas.items():
            results[f"{name}_anchor"] = score_case(case, budget, graph, base, encoded, delta)
        for arm, delta in updates.items():
            name = models[arm][1]
            results[arm] = score_case(case, budget, graph, base, encoded, delta, anchor_deltas=anchor_deltas[name])
        primary = lock["primary"]
        natural = results[primary]["trace"]
        for arm in INTERVENTIONS:
            row = score_case(case, budget, graph, base, encoded, updates[primary], arm,
                             anchor_deltas=anchor_deltas[models[primary][1]])
            patch = row["patch"]
            step = patch["step"] if patch else len(natural)
            if row["trace"][:step] != natural[:step] or patch is None and row["trace"] != natural:
                raise ValueError("the transfer intervention changed its paid prefix")
            results[arm] = row
        a, b = (results[arm]["patch"] for arm in INTERVENTIONS)
        if (a is None) != (b is None) or a and (a["step"], a["l2_dose"]) != (b["step"], b["l2_dose"]):
            raise ValueError("the transfer interventions changed eligibility or dose")
        if budget == 6 and any(row != results["frozen_planner"] for arm, row in results.items()
                               if arm != "temperature_calibration"):
            raise ValueError("a learned probe policy changed a six-call trace")
        yield results
    base.cache_clear()
    temperature.cache_clear()


@torch.no_grad()
def evaluate(args, tree):
    lock, previous, pins, reader = locked(args)
    cohort_receipt = read_receipt(args.cohort)
    cohort = cohort_receipt["cohort"]
    if (cohort_receipt["schema"] != SCHEMA or cohort_receipt["lock_sha256"] != sha256_file(LOCK)
            or cohort_receipt["pins"] != pins or public_digest(cohort) != lock["public_plan_sha256"]
            or expressions(cohort) & expressions(previous)
            or cohort_receipt["independent"]["certificates"] != 2 * len(cohort)):
        raise ValueError("the certified unseen-expression cohort changed")
    anchors, models = probe_models(args, pins, lock)
    prior = MixedSourcePrior(args)
    graph = ProbeGraph()
    records, certificates = defaultdict(list), {}
    for case in cohort:
        tags = {name: case[name] for name in ("module", "pair", "family")}
        for results in score_world(graph, case, *encode_case(case, reader, prior), lock, anchors, models):
            for arm, row in results.items():
                records[arm].append({**tags, **row})
                if row["proved"]:
                    certificates[case["module"], tuple(row["proof"])] = case
    independent, replays = audit_routes(args, cohort, records, certificates,
                                        (lock["primary"], "two_slot_anchor", "trm_deep_return_full", *INTERVENTIONS))
    spare = {arm: [row for row in rows if row["allowance"] in lock["budgets"]] for arm, rows in records.items()}
    candidate = spare[lock["primary"]]
    contrasts = {}
    for arm, rows in spare.items():
        if arm != lock["primary"]:
            contrast = root_contrast(candidate, rows)
            contrasts[arm] = {**contrast, "descriptive_interval": stratified_interval(contrast, candidate)}
    mechanism = root_contrast(spare["chosen_swap"], spare["bystander_swap"])
    mechanism["descriptive_interval"] = stratified_interval(mechanism, spare["chosen_swap"])
    summaries = {arm: {**summary(rows), "spare": summary(spare[arm]),
                       "by_budget": {budget: summary([row for row in rows if row["allowance"] == budget])
                                     for budget in ALLOWANCES}} for arm, rows in records.items()}
    for arm, row in summaries.items():
        print(arm, row["proofs"], row["spare"]["proofs"], row["spare"]["utility"], flush=True)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit),
                                 "lock_sha256": sha256_file(LOCK), "cohort_sha256": sha256_file(args.cohort),
                                 "audit": cohort_receipt["audit"], "floor": lock["comparison_floor"],
                                 "summary": summaries, "spare_root_contrasts": contrasts, "mechanism": mechanism,
                                 "results": records, "independent": independent, "live_replays": replays,
                                 "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "eval"))
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "planner", "curriculum", "warm_probe", "fit",
                 "cohort", "results"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        configure(1, SEED)
        (freeze if args.phase == "freeze" else evaluate)(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
