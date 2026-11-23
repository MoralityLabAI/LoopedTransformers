"""Train cross-proof credit on paired, native-checked Lean recovery routes."""

import argparse
from collections import defaultdict
from copy import deepcopy
from math import fsum, sqrt
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from configs.freeze_ppt_failed_prefix import replay_case
from configs.freeze_ppt_guard_recovery import certify_routes
from configs.freeze_ppt_native_progress import native_sources
from configs.run_ppt_failed_prefix import pack, public_data
from configs.run_ppt_failed_prefix_residual import inputs, out_of_fold
from configs.run_ppt_guard_belief import root_contrast
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_VERSION
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import sha256_file
from ppt_gym.failed_prefix import CHARGE, PREFIX_COST, WORLDS
from ppt_gym.failed_prefix_learning import STARTS
from ppt_gym.failed_prefix_residual import BOUND, RecoveryCorrection
from ppt_gym.live_quote_policy import FEATURES, LiveQuoteTRM, TwoSlotQuoteActor, pad_features
from ppt_gym.paired_recovery import BUDGETS, STOP, order, prospect, run_pair
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.paired-recovery.v1"
SEED = 1131
UPDATES = 480
BATCH = 32
TEMPERATURE = .12
FLOOR = .025
INTERVENTIONS = ("chosen_transfer_swap", "orthogonal_transfer_sham")


def pairs(rows):
    groups = defaultdict(list)
    for index, case in enumerate(rows):
        groups[case["pair"]].append(index)
    for group in groups.values():
        if len(group) != 2:
            raise ValueError("the paired recovery bank lost a theorem")
        first, second = sorted(group, key=lambda index: rows[index]["world"])
        left, right = rows[first], rows[second]
        if len({route["guard"] for route in left["routes"]}) != 3:
            continue
        alias = order(left, right)
        if [left["truth"][index] for index in alias] != right["truth"]:
            raise ValueError("paid truth does not transfer across the public guard mapping")
        yield first, second


def combined_prior(weights, target_weights, alias):
    inverse = [alias.index(i) for i in range(3)]
    projected = {tuple(bits[index] for index in inverse): weight
                 for bits, weight in zip(WORLDS, target_weights, strict=True)}
    return [(first + projected[bits]) / 2 for bits, first in zip(WORLDS, weights, strict=True)]


def pack_world(case, prior):
    tokens, indices, _, _ = public_data(case, prior)
    return torch.tensor([tokens[indices.index(3 * STARTS[-1] + arm)] for arm in range(3)])


def public_choices(source, target, budgets, weights, first_tokens, target_tokens):
    joint, myopic = prospect(source, target, budgets, weights)
    shifts = order(target, source)
    first, second = budgets
    rows = []
    for arm in joint:
        if arm == STOP:
            from_slot, to_slot = [0.] * FEATURES, [0.] * FEATURES
            warm = [0.] * 16
        else:
            from_slot = first_tokens[arm, 1].tolist()
            to_slot = target_tokens[shifts[arm], 1].tolist()
            from_slot[24], to_slot[24] = first / 7, second / 7
            warm = [*first_tokens[arm, 2, 9:17].tolist(), *target_tokens[shifts[arm], 2, 9:17].tolist()]
        context = [first / 7, second / 7, arm == STOP, joint[arm], myopic[arm], joint[STOP],
                   joint[arm] - myopic[arm] - joint[STOP], *warm]
        rows.append([from_slot, to_slot, pad_features(context)])
    return list(joint), torch.tensor(rows), joint, myopic


def models_for():
    loop = RecoveryCorrection(TwoSlotQuoteActor())
    return {"two_slot_return": loop, "two_slot_q": deepcopy(loop), "trm_return": RecoveryCorrection(LiveQuoteTRM()),
            "affine_return": RecoveryCorrection(nn.Sequential(nn.Flatten(1), nn.Linear(3 * FEATURES, 1)))}


def pack_pairs(source, packed, priors):
    examples = []
    for left, right in pairs(source):
        for first, second in ((left, right), (right, left)):
            a, b = source[first], source[second]
            weights = combined_prior(priors[first], priors[second], order(a, b))
            for budgets in BUDGETS:
                arms, rows, joint, _ = public_choices(a, b, budgets, weights, packed[2][first], packed[2][second])
                padded = torch.zeros(4, 3, FEATURES)
                scores = torch.full((4,), -torch.inf)
                rewards = torch.zeros(4)
                for row, arm in enumerate(arms):
                    padded[arm] = rows[row]
                    scores[arm] = joint[arm]
                    rewards[arm] = run_pair(a, b, budgets, weights, arm)["utility"]
                examples.append((padded, scores, rewards))
    return tuple(torch.stack(values) for values in zip(*examples, strict=True))


def train(model, arm, data, batches):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for index in batches:
        rows, base, reward = (value[index] for value in data)
        scores = base + model(rows.flatten(0, 1)).reshape_as(base)
        if arm == "two_slot_q":
            mask = base.isfinite()
            # The two six-call setups are paid for every available route.
            loss = F.mse_loss(scores[mask], (reward + 2 * CHARGE * PREFIX_COST)[mask])
        else:
            loss = -((scores / TEMPERATURE).softmax(-1) * reward).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval()


def fit(args, tree):
    bank, pins, _ = inputs(args)
    rng = torch.Generator().manual_seed(SEED)
    source = bank["source"]
    packed = pack(source, MixedSourcePrior(args))
    prior, folds = out_of_fold(source, packed, rng)
    data = pack_pairs(source, packed, prior)
    samples = torch.randint(len(data[0]), (UPDATES, BATCH), generator=rng)
    models = models_for()
    initial = {arm: tensor_hash(model.state_dict()) for arm, model in models.items()}
    if initial["two_slot_return"] != initial["two_slot_q"]:
        raise ValueError("the matched recurrent models changed initialization")
    for arm, model in models.items():
        train(model, arm, data, samples)
        print("fitted", arm, flush=True)
    weights = {arm: model.state_dict() for arm, model in models.items()}
    result = {"schema": SCHEMA, "pins": pins, "weights": weights, "initial": initial, "folds": folds,
              "weight_sha": {arm: tensor_hash(value) for arm, value in weights.items()}, "source_rows": len(data[0]),
              "sampler_sha": tensor_hash({"batches": samples}), "memory": tree.stats()}
    with args.fit.open("xb") as handle:
        torch.save(result, handle)
    print("sealed paired recovery fit", sha256_file(args.fit), flush=True)


def manipulate(delta, candidate, anchor, *, sham_sign=None):
    adjusted = dict(delta)
    difference = adjusted[anchor] - adjusted[candidate]
    if sham_sign is None:
        adjusted[candidate], adjusted[anchor] = adjusted[anchor], adjusted[candidate]
    else:
        bystanders = set(adjusted) - {candidate, anchor}
        count = len(bystanders)
        shift = sham_sign * difference * sqrt(count / (count + 2))
        adjusted[candidate] += shift
        adjusted[anchor] += shift
        for arm in bystanders:
            adjusted[arm] -= 2 * shift / count
    return adjusted, sqrt(2) * abs(difference)


def replays(args, cases, records, arms=("two_slot_return", "trm_return", "joint_planner", *INTERVENTIONS),
            *, patched_arms=INTERVENTIONS):
    case_by_key = {row["module"]: row for row in cases}
    used = set()
    for rows in records.values():
        for row in rows:
            for stage in ("first", "second"):
                if row[stage]["proved"]:
                    used.add((row[stage + "_module"], row[stage]["trace"][-1]["arm"]))
    live = []
    with native_sources((args.bpd, args.mathcan)) as (directory, env, imports):
        scripts = [(case_by_key[key], case_by_key[key]["routes"][arm]["proof"]) for key, arm in sorted(used)]
        certificates = certify_routes(directory, env, imports, scripts)
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected native paired-recovery toolchain")
            for arm in arms:
                candidates = [row for row in records[arm]
                              if row["first"]["trace"] and (arm not in patched_arms or row["patch"])]
                for passed in (True, False):
                    row = next((x for x in candidates if x["first"]["proved"] == passed), None)
                    if row is not None:
                        for stage in ("first", "second"):
                            replay_case(repl, case_by_key[row[stage + "_module"]], row[stage])
                        live.append({"arm": arm, "module": row["module"], "first_proved": passed,
                                     "second_proved": row["second"]["proved"]})
    return certificates, live


def summarize(rows):
    return {"proofs": sum(row["proofs"] for row in rows),
            "first_proofs": sum(row["first"]["proved"] for row in rows),
            "second_proofs": sum(row["second"]["proved"] for row in rows),
            "checks": sum(row["checks"] for row in rows),
            "count": len(rows), "utility": fsum(row["utility"] for row in rows) / len(rows)}


@torch.no_grad()
def evaluate(args, tree):
    bank, pins, reader = inputs(args)
    fit = torch.load(args.fit, map_location="cpu", weights_only=True)
    if fit["schema"] != SCHEMA or fit["pins"] != pins:
        raise ValueError("the locked paired recovery policy changed")
    models = models_for()
    for arm, model in models.items():
        weights = fit["weights"][arm]
        if tensor_hash(weights) != fit["weight_sha"][arm]:
            raise ValueError("a frozen cross-proof readout changed")
        model.load_state_dict(weights)
        model.eval()
    development = bank["development"]
    prior = MixedSourcePrior(args)
    tokens = [pack_world(case, prior) for case in development]
    forecasts = [reader(row.reshape(1, -1)).softmax(-1)[0].tolist() for row in tokens]
    records = defaultdict(list)
    for left, right in pairs(development):
        for first, second in ((left, right), (right, left)):
            source, target = development[first], development[second]
            weights = combined_prior(forecasts[first], forecasts[second], order(source, target))
            tags = {"module": source["module"], "pair": source["pair"], "family": source["family"],
                    "first_module": source["module"], "second_module": target["module"]}
            for budgets in BUDGETS:
                arms, action_tokens, joint, immediate = public_choices(source, target, budgets, weights,
                                                                        tokens[first], tokens[second])
                anchor = max(joint, key=joint.__getitem__)
                separate = max(immediate, key=immediate.__getitem__)
                choices = {"reset_planner": (separate, None), "myopic_memory": (separate, None),
                           "joint_planner": (anchor, None)}
                for arm, model in models.items():
                    correction = dict(zip(arms, model(action_tokens).flatten().tolist(), strict=True))
                    picked = max(arms, key=lambda move: joint[move] + correction[move])
                    choices[arm] = picked, None
                    if arm == "two_slot_return":
                        for mode in INTERVENTIONS:
                            patch, altered = None, picked
                            if len(arms) >= 3 and picked != anchor:
                                sign = None if mode == INTERVENTIONS[0] else (1 if int(source["module"][-1], 16) % 2 else -1)
                                updated, dose = manipulate(correction, picked, anchor, sham_sign=sign)
                                altered = max(arms, key=lambda move: joint[move] + updated[move])
                                patch = {"before": picked, "after": altered, "affine": anchor, "l2_dose": dose}
                            choices[mode] = altered, patch
                allowed = [arm for arm in arms if joint[arm] + BOUND >= max(joint.values()) - BOUND]
                oracle = max(allowed, key=lambda arm: run_pair(source, target, budgets, weights, arm)["utility"])
                choices["bounded_oracle"] = oracle, None
                for arm, (choice, patch) in choices.items():
                    row = run_pair(source, target, budgets, weights, choice, remember=arm != "reset_planner")
                    records[arm].append({**tags, **row, "patch": patch, "changed_choice": choice != anchor,
                                         "immediate_opportunity_cost": immediate[separate] - immediate[choice]})
    primary = records["two_slot_return"]
    if any(a["utility"] > b["utility"] + 1e-9 for a, b in zip(primary, records["bounded_oracle"], strict=True)):
        raise ValueError("the learned cross-proof choice exceeded its action envelope")
    for a, b in zip(*(records[arm] for arm in INTERVENTIONS), strict=True):
        if (a["patch"] is None) != (b["patch"] is None) or a["patch"] and a["patch"]["l2_dose"] != b["patch"]["l2_dose"]:
            raise ValueError("cross-proof interventions disagree on eligibility or dose")
    independent, live = replays(args, development, records)
    contrasts = {arm: root_contrast(primary, rows) for arm, rows in records.items() if rows is not primary}
    summary = {arm: {**summarize(rows), "by_budget": {10 * a + b: summarize([x for x in rows if x["allowance"] == 10 * a + b])
                                                    for a, b in BUDGETS}} for arm, rows in records.items()}
    for arm, row in summary.items():
        print(arm, row["first_proofs"], row["second_proofs"], row["checks"], row["utility"], flush=True)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit),
                                 "floor": FLOOR, "source_rows": fit["source_rows"], "summary": summary,
                                 "root_contrasts": contrasts, "results": records, "independent": independent,
                                 "live_replays": live, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "anchor", "fit", "results"):
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
