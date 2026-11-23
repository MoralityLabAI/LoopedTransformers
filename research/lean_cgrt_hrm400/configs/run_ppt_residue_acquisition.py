"""Fit and replay a paid diagnostic policy on native Lean proof pairs."""

import argparse
from collections import defaultdict
from copy import deepcopy
from math import dist, fsum, isclose
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from configs.freeze_ppt_native_progress import native_sources
from configs.freeze_ppt_residue_acquisition import SCHEMA as NATIVE_SCHEMA, witness
from configs.run_ppt_guard_belief import root_contrast
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_paired_recovery import manipulate, replays, summarize
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_residue_recovery import SCHEMA as RESIDUE_SCHEMA
from configs.run_ppt_residue_recovery import curves, distribution, locked, models_for, public_tokens, source_trial
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import sha256_file
from ppt_gym.failed_prefix import ALLOWANCES
from ppt_gym.live_quote_policy import FEATURES, LiveQuoteTRM
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.residue_acquisition import COMMIT, execute, forecast, queries
from ppt_gym.residue_acquisition_policy import FeedbackSlotActor, tokens
from ppt_gym.residue_recovery import compatibility


SCHEMA = "ppt.residue-acquisition.v1"
SEED = 1141
BATCH = 16
UPDATES = 400
TEMPERATURE = .20
FLOOR = .025
INTERVENTIONS = ("action_score_swap", "orthogonal_sham")
CALIBRATORS = ("affine_supervised", "affine_return")


def inputs(args, partition):
    bank, pins = locked(args)
    native = read_receipt(args.native)
    if native["schema"] != NATIVE_SCHEMA or native["bank"] != pins["bank"]:
        raise ValueError("the paid query bank changed")
    records = list(zip(bank[partition], native[partition], strict=True))
    for pair, record in records:
        if (record["pair"] != pair["pair"] or record["menu"] != queries(pair)
                or len(record["menu"]) != len(record["outcomes"])):
            raise ValueError("a proof pair does not match its native query bank")
    return records, {**pins, **{name: sha256_file(getattr(args, name)) for name in ("native", "calibration")}}


def models():
    candidate = FeedbackSlotActor()
    trm = LiveQuoteTRM()
    affine = nn.Sequential(nn.Flatten(1), nn.Linear(3 * FEATURES, 1))
    for head in (candidate.head, trm.head, affine[-1]):
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    return {"two_slot_return": candidate, "two_slot_value": deepcopy(candidate),
            "trm_return": trm, "affine_return": affine}


def experience(pair, prior):
    choice, observed, first = source_trial(pair["first"])
    support = compatibility(pair["first"], choice, observed)
    warm = curves(pair["first"], prior)
    uniform = [float(ok) / sum(support) for ok in support]
    return choice, observed, first, support, warm, uniform


def choices(pair, record, episode, allowance, prior):
    choice, observed, _, support, warm, _ = episode
    target, menu = pair["second"], record["menu"]
    anchor = forecast(target, allowance, menu, prior, support)
    available = {action: execute(target, allowance, menu, prior, action, record["outcomes"])
                 for action in anchor}
    features = tokens(pair, choice, observed, support, warm, allowance, menu, anchor)
    scores = [anchor.get(action, -torch.inf) for action in range(len(features))]
    return torch.tensor(features), torch.tensor(scores), available


def training_rows(records, prior):
    packed = []
    for pair, record in records:
        episode = experience(pair, prior)
        for allowance in ALLOWANCES:
            features, anchor, outcomes = choices(pair, record, episode, allowance, episode[-1])
            rewards = torch.zeros_like(anchor)
            for action, outcome in outcomes.items():
                rewards[action] = outcome["utility"]
            packed.append((features, anchor, rewards))
    return tuple(torch.stack(values) for values in zip(*packed, strict=True))


def train(model, name, packed, batches):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for batch in batches:
        features, anchor, rewards = (piece[batch] for piece in packed)
        scores = anchor + model(features.flatten(0, 1)).reshape_as(anchor)
        if name == "two_slot_value":
            legal = anchor.isfinite()
            loss = F.mse_loss(scores[legal], rewards[legal])
        else:
            loss = -((scores / TEMPERATURE).softmax(-1) * rewards).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval()


def fit(args, tree):
    records, pins = inputs(args, "source")
    packed = training_rows(records, MixedSourcePrior(args))
    rng = torch.Generator().manual_seed(SEED)
    batches = torch.randint(len(packed[0]), (UPDATES, BATCH), generator=rng)
    readers = models()
    initial = {name: tensor_hash(model.state_dict()) for name, model in readers.items()}
    if initial["two_slot_return"] != initial["two_slot_value"]:
        raise ValueError("matched acquisition arms changed their initialization")
    for name, model in readers.items():
        train(model, name, packed, batches)
        print("fitted acquisition", name, flush=True)
    weights = {name: model.state_dict() for name, model in readers.items()}
    checkpoint = {"schema": SCHEMA, "pins": pins, "weights": weights, "initial": initial,
                  "weight_sha": {name: tensor_hash(state) for name, state in weights.items()},
                  "sampler_sha": tensor_hash({"batches": batches}), "source_rows": len(packed[0]),
                  "memory": tree.stats()}
    with args.fit.open("xb") as handle:
        torch.save(checkpoint, handle)
    print("sealed acquisition fit", sha256_file(args.fit), flush=True)


def restore(model, fit, name):
    weights = fit["weights"][name]
    if tensor_hash(weights) != fit["weight_sha"][name]:
        raise ValueError("a frozen acquisition model changed")
    model.load_state_dict(weights)
    return model.eval()


def best(values):
    return max(values, key=values.__getitem__)


def interventions(pair, base, scores):
    candidate, anchor = best(scores), int(base.argmax())
    if candidate == anchor or len(scores) < 3:
        return {name: (candidate, None) for name in INTERVENTIONS}
    patched = {}
    for name in INTERVENTIONS:
        sign = None if name == INTERVENTIONS[0] else (1 if int(pair["pair"][-1], 16) % 2 else -1)
        changed, dose = manipulate(scores, candidate, anchor, sham_sign=sign)
        if not isclose(dist(scores.values(), changed.values()), dose, abs_tol=1e-7):
            raise ValueError("an acquisition score edit changed physical dose")
        patched[name] = best(changed), {"dose": dose, "candidate": candidate, "anchor": anchor,
                                       "scores": [changed[action] for action in changed]}
    return patched


def observable(trace, pair, first, allowance, patch=None):
    target = trace["second"]
    return {**trace, "first": first, "allowance": allowance, "patch": patch,
            "pair": pair["pair"], "module": pair["first"]["module"], "family": pair["family"],
            "expression": pair["certificate"].rsplit(" = ", 1)[0],
            "first_module": pair["first"]["module"], "second_module": pair["second"]["module"],
            "proofs": int(first["proved"]) + int(target["proved"]),
            "checks": first["checks"] + target["checks"] + (trace["query"] or {}).get("charge", 0)}


def extended_summary(rows):
    bought = [row for row in rows if row["query"] is not None]
    return {**summarize(rows), "purchases": len(bought), "passed_checks": sum(row["query"]["passed"] for row in bought)}


def expression_contrast(candidate, control):
    grouped = defaultdict(list)
    for left, right in zip(candidate, control, strict=True):
        if left["expression"] != right["expression"]:
            raise ValueError("unpaired expression contrast")
        grouped[left["expression"]].append(left["utility"] - right["utility"])
    return fsum(fsum(values) / len(values) for values in grouped.values()) / len(grouped)


def replay_paid(args, pairs, records, cases):
    audits = ("two_slot_return", "two_slot_value", "trm_return", "affine_return", "slot_reset",
              "uniform_acquire", *(name + "_acquire" for name in CALIBRATORS), *INTERVENTIONS)
    compiled, live = replays(args, cases, records, audits, patched_arms=INTERVENTIONS)
    used = defaultdict(dict)
    for rows in records.values():
        for row in rows:
            if paid := row["query"]:
                used[row["pair"]][paid["action"]] = paid["passed"]
    replayed = 0
    with native_sources((args.bpd, args.mathcan)) as (directory, env, imports):
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            for pair, record in pairs:
                purchased = used[pair["pair"]]
                if purchased:
                    actions = sorted(purchased)
                    actual = witness(repl, pair, [record["menu"][action - 1] for action in actions])
                    if actual != [purchased[action] for action in actions]:
                        raise ValueError("a purchased native check changed on replay")
                    replayed += len(actions)
    return compiled, live, replayed


@torch.no_grad()
def evaluate(args, tree):
    pairs, pins = inputs(args, "development")
    checkpoint = torch.load(args.fit, map_location="cpu", weights_only=True)
    calibration = torch.load(args.calibration, map_location="cpu", weights_only=True)
    if (checkpoint["schema"] != SCHEMA or checkpoint["pins"] != pins
            or calibration["schema"] != RESIDUE_SCHEMA
            or calibration["pins"] != {name: pins[name] for name in ("bank", "warm", "prior")}):
        raise ValueError("the acquisition or calibration fit has different input pins")
    readers = {name: restore(model, checkpoint, name) for name, model in models().items()}
    previous = models_for()
    calibrators = {name: restore(previous[name], calibration, name) for name in CALIBRATORS}
    prior = MixedSourcePrior(args)
    records, cases = defaultdict(list), []
    for pair, record in pairs:
        cases.extend((pair["first"], pair["second"]))
        episode = experience(pair, prior)
        choice, observed, first, support, warm, uniform = episode
        public, _ = public_tokens(pair, choice, observed, warm, curves(pair["second"], prior))
        public = torch.tensor(public)
        probabilities = {name: distribution(model(public).flatten(), torch.tensor(support))
                         for name, model in calibrators.items()}
        for allowance in ALLOWANCES:
            features, anchor, rows = choices(pair, record, episode, allowance, uniform)
            scores = {name: (anchor + model(features).flatten()) for name, model in readers.items()}
            scores["slot_reset"] = anchor + readers["two_slot_return"](features, reset=True).flatten()
            policies = {name: (int(values.argmax()), None) for name, values in scores.items()}
            policies.update({"uniform_commit": (COMMIT, None), "uniform_acquire": (int(anchor.argmax()), None),
                             "realised_oracle": (max(rows, key=lambda action: rows[action]["utility"]), None)})
            candidate = {action: scores["two_slot_return"][action].item() for action in rows}
            policies.update(interventions(pair, anchor, candidate))
            for name, (action, patch) in policies.items():
                outcome = observable(rows[action], pair, first, allowance, patch)
                if name in ("two_slot_return", *INTERVENTIONS):
                    outcome["available_actions"] = list(candidate)
                    outcome["original_scores"] = list(candidate.values())
                records[name].append(outcome)
            for name, masses in probabilities.items():
                exact = forecast(pair["second"], allowance, record["menu"], masses, support)
                for suffix, action in (("_commit", COMMIT), ("_acquire", best(exact))):
                    trace = execute(pair["second"], allowance, record["menu"], masses, action, record["outcomes"])
                    records[name + suffix].append(observable(trace, pair, first, allowance))
    summary = {name: {**extended_summary(rows), "budgets": {
        budget: extended_summary([row for row in rows if row["allowance"] == budget]) for budget in ALLOWANCES}}
        for name, rows in records.items()}
    for name, stats in summary.items():
        print(name, stats["second_proofs"], stats["purchases"], stats["utility"], flush=True)
    main = records["two_slot_return"]
    contrasts = {name: {**root_contrast(main, rows), "expression_macro": expression_contrast(main, rows)}
                 for name, rows in records.items() if rows is not main}
    independent, live, queries_replayed = replay_paid(args, pairs, records, cases)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit),
                                 "floor": FLOOR, "summary": summary, "results": records, "contrasts": contrasts,
                                 "independent": independent, "live_replays": live, "queries_replayed": queries_replayed,
                                 "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "native", "calibration", "fit", "results"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        configure(1, SEED)
        (fit if args.phase == "fit" else evaluate)(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
