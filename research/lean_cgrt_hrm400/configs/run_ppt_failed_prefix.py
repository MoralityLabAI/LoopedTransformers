"""Compare reward, supervised and calibration policies after native rejection."""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from configs.freeze_ppt_failed_prefix import ROOTS, replay_case
from configs.freeze_ppt_guard_recovery import certify_routes
from configs.freeze_ppt_native_progress import native_sources
from configs.run_ppt_guard_belief import root_contrast
from configs.run_ppt_guard_recovery import summary
from configs.run_ppt_live_quote_fit import warm_values
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import sha256_file
from ppt_gym.failed_prefix import ALLOWANCES, SCHEMA as BANK_SCHEMA, WORLDS
from ppt_gym.failed_prefix import planned_values, probability_weights, trace_case
from ppt_gym.failed_prefix_learning import COUNT, STARTS, STATES, dense_scores, observations, policy_distribution, returns
from ppt_gym.live_quote_policy import FEATURES, LiveQuoteTRM, TwoSlotQuoteActor
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.failed-prefix-recovery-fit.v1"
SEED = 1128
UPDATES = 360
BATCH = 16
FLOOR = .025


def locked(args):
    bank = read_receipt(args.bank)
    pairs = {name: {row["pair"] for row in bank[name]} for name in ROOTS}
    if (bank["schema"] != BANK_SCHEMA or any(len(pairs[name]) != count for name, count in ROOTS.items())
            or pairs["source"] & pairs["development"] or bank["lean"] != sha256_file(LEAN_EXE)
            or bank["repl"] != sha256_file(REPL_EXE)
            or bank["imports"] != {path.name: sha256_file(path) for path in (args.bpd, args.mathcan)}):
        raise ValueError("the failed-prefix native bank or toolchain changed")
    return bank, {name: sha256_file(getattr(args, name)) for name in ("bank", "warm", "prior")}


def models_for():
    actor = TwoSlotQuoteActor()
    return {"two_slot_return": actor, "two_slot_q": deepcopy(actor), "two_slot_truth": deepcopy(actor),
            "trm_return": LiveQuoteTRM(), "affine_joint": nn.Linear(9 * FEATURES, len(WORLDS))}


def public_data(case, prior):
    kept = warm_values(prior, case["saved"], ("apply hr0",))
    switches = warm_values(prior, case["fork"], ("apply hr1", "apply hr2"))
    warm = [kept["apply hr0"], switches["apply hr1"], switches["apply hr2"]]
    return observations(case["routes"], warm)


def pack(cases, prior):
    rows = [public_data(case, prior) for case in cases]
    width = max(len(row[0]) for row in rows)
    tokens = torch.zeros(len(rows), width, 3, FEATURES)
    indices = torch.full((len(rows), width), 3 * COUNT)
    for case, (words, ids, _, _) in enumerate(rows):
        tokens[case, :len(words)] = torch.tensor(words)
        indices[case, :len(ids)] = torch.tensor(ids)
    root = torch.stack([torch.tensor([words[ids.index(STARTS[-1] * 3 + arm)] for arm in range(3)])
                        for words, ids, _, _ in rows])
    menu = torch.tensor([row[2] for row in rows])
    following = torch.tensor([row[3] for row in rows])
    pass_cost = torch.tensor([[route["pass_cost"] for route in case["routes"]] for case in cases], dtype=torch.float)
    fail_cost = torch.tensor([[route["fail_cost"] for route in case["routes"]] for case in cases], dtype=torch.float)
    truth = torch.tensor([case["truth"] for case in cases])
    worlds = torch.tensor([WORLDS.index(tuple(case["truth"])) for case in cases])
    return tokens, indices, root, menu, following, pass_cost, fail_cost, truth, worlds


def train(model, arm, data, batches):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    compatible = torch.tensor([[not bool(failed & (1 << index)) for index in range(3)] for _, failed in STATES])
    for indices in batches:
        tokens, code, roots, menu, links, pass_cost, fail_cost, truth, worlds = [row[indices] for row in data]
        if arm == "affine_joint":
            loss = F.cross_entropy(model(roots.flatten(1)), worlds)
        elif arm == "two_slot_truth":
            predicted = model(roots.flatten(0, 1)).reshape_as(truth)
            loss = F.binary_cross_entropy_with_logits(predicted, truth.float())
        else:
            scores = dense_scores(model, tokens, code, menu)
            if arm == "two_slot_q":
                target = returns(links, menu, pass_cost, fail_cost, truth)
                reachable = ~(truth[:, None] & ~compatible).any(-1)
                mask = menu & reachable.unsqueeze(-1)
                loss = F.mse_loss(scores[mask], target[mask])
            else:
                policy = policy_distribution(scores)
                loss = -returns(links, menu, pass_cost, fail_cost, truth, policy).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval()


def fit(args, tree):
    bank, pins = locked(args)
    data = pack(bank["source"], MixedSourcePrior(args))
    batches = torch.randint(len(data[0]), (UPDATES, BATCH), generator=torch.Generator().manual_seed(SEED))
    models = models_for()
    initial = {arm: tensor_hash(model.state_dict()) for arm, model in models.items()}
    if len({initial[arm] for arm in ("two_slot_return", "two_slot_q", "two_slot_truth")}) != 1:
        raise ValueError("the matched candidate arms changed initial weights")
    for arm, model in models.items():
        train(model, arm, data, batches)
        print("fitted", arm, flush=True)
    weights = {arm: model.state_dict() for arm, model in models.items()}
    receipt = {"schema": SCHEMA, "pins": pins, "weights": weights, "initial": initial,
               "weight_sha": {arm: tensor_hash(value) for arm, value in weights.items()},
               "sampler_sha": tensor_hash({"batches": batches}), "memory": tree.stats()}
    with args.fit.open("xb") as handle:
        torch.save(receipt, handle)
    print("sealed saved-prefix fit", sha256_file(args.fit), flush=True)


def choose(values):
    selected = max(values, key=values.__getitem__, default=None)
    return selected if selected is not None and values[selected] > 0. else None


def audit(args, cases, records):
    index = {case["module"]: case for case in cases}
    used = {(row["module"], row["trace"][-1]["arm"]) for rows in records.values() for row in rows if row["proved"]}
    replays = []
    with native_sources((args.bpd, args.mathcan)) as (directory, env, imports):
        scripts = [(index[key], index[key]["routes"][arm]["proof"]) for key, arm in sorted(used)]
        certificates = certify_routes(directory, env, imports, scripts)
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected native evaluation toolchain")
            for arm in ("two_slot_return", "trm_return", "two_slot_truth", "slots_swapped"):
                for outcome in (True, False):
                    row = next((r for r in records[arm] if r["proved"] == outcome and len(r["trace"]) > 1), None)
                    if row is None:
                        row = next((r for r in records[arm] if r["proved"] == outcome and r["trace"]), None)
                    if row is not None:
                        replay_case(repl, index[row["module"]], row)
                        replays.append({"arm": arm, "module": row["module"], "allowance": row["allowance"],
                                        "proved": outcome})
    return certificates, replays


@torch.no_grad()
def evaluate(args, tree):
    bank, pins = locked(args)
    fitted = torch.load(args.fit, map_location="cpu", weights_only=True)
    if fitted["schema"] != SCHEMA or fitted["pins"] != pins:
        raise ValueError("the locked failed-prefix fit changed")
    models = models_for()
    for arm, model in models.items():
        weights = fitted["weights"][arm]
        if tensor_hash(weights) != fitted["weight_sha"][arm]:
            raise ValueError("a frozen recovery policy changed")
        model.load_state_dict(weights)
        model.eval()
    prior = MixedSourcePrior(args)
    records = defaultdict(list)
    for case in bank["development"]:
        tokens, indices, _, _ = public_data(case, prior)
        tokens = torch.tensor(tokens)
        positions = {code: row for row, code in enumerate(indices)}
        root = tokens[[positions[STARTS[-1] * 3 + arm] for arm in range(3)]]
        rows = {arm: model(tokens).tolist() for arm, model in models.items()
                if arm in ("two_slot_return", "two_slot_q", "trm_return")}
        swapped = models["two_slot_return"](tokens[:, [1, 0, 2]]).tolist()
        probability = models["two_slot_truth"](root).sigmoid().clamp(.02, .98).tolist()
        joint = models["affine_joint"](root.reshape(1, -1)).softmax(-1)[0].tolist()
        frozen = {"two_slot_truth": probability_weights(probability), "affine_joint": joint,
                  "uniform_planner": [1.] * len(WORLDS)}
        forecasts = {arm: planned_values(case["routes"], weights) for arm, weights in frozen.items()}
        oracle = planned_values(case["routes"], [float(bits == tuple(case["truth"])) for bits in WORLDS])
        tags = {name: case[name] for name in ("pair", "module", "family")}
        for allowance in ALLOWANCES:
            natural = None
            for arm in (*rows, *forecasts, "slots_swapped", "oracle"):
                patch = None

                def select(remaining, failed, menu):
                    nonlocal patch
                    if arm == "oracle":
                        return choose(oracle(remaining, failed))
                    if arm in forecasts:
                        return choose(forecasts[arm](remaining, failed))
                    scores = rows["two_slot_return" if arm == "slots_swapped" else arm]
                    lookup = {move: positions[(remaining * 8 + failed) * 3 + move] for move in menu}
                    selected = choose({move: scores[row] for move, row in lookup.items()})
                    if arm == "slots_swapped" and patch is None and len(menu) >= 2:
                        swapped_pick = choose({move: swapped[row] for move, row in lookup.items()})
                        patch = {"spent": allowance - remaining, "before": selected, "after": swapped_pick}
                        return swapped_pick
                    return selected

                trace = trace_case(case, allowance, select)
                if arm == "two_slot_return":
                    natural = trace
                if patch is not None:
                    left = [event for event in natural["trace"] if allowance - event["before"] < patch["spent"]]
                    right = [event for event in trace["trace"] if allowance - event["before"] < patch["spent"]]
                    if left != right:
                        raise ValueError("the recovery slot swap changed its paid prefix")
                records[arm].append({**tags, **trace, "patch": patch})
    certificates, replays = audit(args, bank["development"], records)
    primary = records["two_slot_return"]
    contrasts = {arm: root_contrast(primary, rows) for arm, rows in records.items() if rows is not primary}
    results = {arm: {**summary(rows), "by_budget": {budget: summary([row for row in rows if row["allowance"] == budget])
                                                   for budget in ALLOWANCES}} for arm, rows in records.items()}
    for arm, row in results.items():
        print(arm, row["proofs"], row["checks"], row["utility"], flush=True)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit),
                                 "floor": FLOOR, "summary": results, "root_contrasts": contrasts, "results": records,
                                 "independent": certificates, "live_replays": replays, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "fit", "results"):
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
