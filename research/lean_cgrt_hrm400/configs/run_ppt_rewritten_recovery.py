"""Learn to route a paid Lean guard observation across arithmetic rewrites."""

import argparse
from collections import defaultdict
from copy import deepcopy
from math import fsum
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from configs.run_ppt_failed_prefix import pack
from configs.run_ppt_failed_prefix_residual import inputs, out_of_fold
from configs.run_ppt_guard_belief import root_contrast
from configs.run_ppt_live_quote_fit import warm_values
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_paired_recovery import pack_world, replays, summarize
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_rollback_diagnostics import write_receipt
from ppt_gym.data import sha256_file
from ppt_gym.failed_prefix import ALLOWANCES, CHARGE, WORLDS, legal
from ppt_gym.failed_prefix_learning import STARTS, policy_distribution, recovery_graph, returns
from ppt_gym.failed_prefix_residual import redistribute
from ppt_gym.live_quote_policy import FEATURES, LiveQuoteTRM, TwoSlotQuoteActor
from ppt_gym.paired_recovery import STOP, one_trial, solve_target
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rewritten_recovery import ALIASES, SCHEMA as BANK_SCHEMA, assignment_logits, mix_table, pair_tokens, planned_batch


SCHEMA = "ppt.rewritten-recovery-fit.v1"
SEED = 1132
UPDATES = 400
BATCH = 16
FLOOR = .025
FIRST_BUDGET = 5
INTERVENTIONS = ("chosen_binding_swap", "orthogonal_binding_sham")


def locked(args):
    bank, pins, reader = inputs(args)
    rewritten = read_receipt(args.rewritten)
    if (rewritten["schema"] != BANK_SCHEMA or rewritten["bank_sha256"] != pins["bank"]
            or any(rewritten[name] != bank[name] for name in ("lean", "repl", "imports"))):
        raise ValueError("the sealed rewrite bank changed")
    return bank, rewritten, {**pins, "rewritten": sha256_file(args.rewritten)}, reader


def models_for():
    actor = TwoSlotQuoteActor()
    linear = nn.Sequential(nn.Flatten(1), nn.Linear(3 * FEATURES, 1))
    return {"two_slot_return": actor, "two_slot_supervised": deepcopy(actor), "trm_return": LiveQuoteTRM(),
            "affine_return": linear, "affine_supervised": deepcopy(linear)}


def fixed_first(source, weights):
    values = {STOP: 0.}
    for arm in legal(source["routes"], FIRST_BUDGET, 0):
        route = source["routes"][arm]
        passed = fsum(weight for world, weight in zip(WORLDS, weights, strict=True) if world[arm])
        values[arm] = passed * (1 - CHARGE * route["pass_cost"]) - (1 - passed) * CHARGE * route["fail_cost"]
    return max(values, key=values.__getitem__)


def existing_curves(case, prior):
    saved = warm_values(prior, case["saved"], ("apply hr0",))
    restart = warm_values(prior, case["fork"], ("apply hr1", "apply hr2"))
    return [saved["apply hr0"], restart["apply hr1"], restart["apply hr2"]]


def episode(source, target, weights, prior):
    choice = fixed_first(source, weights)
    first = one_trial(source, FIRST_BUDGET, choice)
    observed = first["proved"] if choice != STOP else None
    rows, lexical = pair_tokens(source, target, choice, observed, existing_curves(source, prior),
                                existing_curves(target, prior))
    projections = mix_table(weights, choice, observed)
    _, legal_menu, links = recovery_graph(target["routes"])
    pass_cost = [float(route["pass_cost"]) for route in target["routes"]]
    fail_cost = [float(route["fail_cost"]) for route in target["routes"]]
    return rows, lexical, projections, first, choice, observed, (legal_menu, links, pass_cost, fail_cost)


def pack_rewrites(bank, entries, probabilities, prior):
    index = {case["module"]: (case, weights) for case, weights in zip(bank, probabilities, strict=True)}
    examples = []
    for pair in entries:
        source, p = index[pair["source"]]
        target = pair["target"]
        rows, _, priors, _, _, _, ordinary = episode(source, target, p, prior)
        examples.append((rows, priors, *ordinary, target["truth"], ALIASES.index(tuple(pair["alignment"]))))
    return tuple(torch.tensor(values) for values in zip(*examples, strict=True))


def train(model, arm, data, samples):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for indices in samples:
        rows, priors, menu, links, pass_cost, fail_cost, truth, target = (item[indices] for item in data)
        edge = model(rows.flatten(0, 1)).reshape(-1, 9)
        logits = assignment_logits(edge)
        if arm.endswith("supervised"):
            loss = F.cross_entropy(logits, target)
        else:
            belief = (logits.softmax(-1).unsqueeze(-1) * priors).sum(1)
            scores = planned_batch(links, menu, pass_cost, fail_cost, belief)
            loss = -returns(links, menu, pass_cost, fail_cost, truth, policy_distribution(scores)).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval()


def fit(args, tree):
    bank, rewritten, pins, _ = locked(args)
    if rewritten.get("form") == "unfolded-development":
        raise ValueError("the recurrence-unfolding bank is evaluation-only")
    rng = torch.Generator().manual_seed(SEED)
    prior = MixedSourcePrior(args)
    source = bank["source"]
    packed = pack(source, prior)
    probabilities, folds = out_of_fold(source, packed, rng)
    rows = pack_rewrites(source, rewritten["source"], probabilities, prior)
    samples = torch.randint(len(rows[0]), (UPDATES, BATCH), generator=rng)
    models = models_for()
    start = {arm: tensor_hash(model.state_dict()) for arm, model in models.items()}
    for a, b in (("two_slot_return", "two_slot_supervised"), ("affine_return", "affine_supervised")):
        if start[a] != start[b]:
            raise ValueError("the matched correspondence heads changed initialization")
    for arm, model in models.items():
        train(model, arm, rows, samples)
        print("fitted", arm, flush=True)
    weights = {arm: model.state_dict() for arm, model in models.items()}
    result = {"schema": SCHEMA, "pins": pins, "weights": weights, "initial": start, "folds": folds,
              "weight_sha": {arm: tensor_hash(state) for arm, state in weights.items()},
              "source_pairs": len(rows[0]), "sampler_sha": tensor_hash({"batches": samples}), "memory": tree.stats()}
    with args.fit.open("xb") as handle:
        torch.save(result, handle)
    print("sealed arithmetic binding fit", sha256_file(args.fit), flush=True)


def mixture(edge, table):
    logits = assignment_logits(edge)[0]
    probabilities = logits.softmax(-1)
    return (probabilities @ table).tolist(), int(probabilities.argmax())


def posteriors(models, rows, lexical, table, reset, observed_arm, observed, truth_alignment, identity):
    results = {"reset": (reset.mean(0).tolist(), None), "certified_binding": (table[truth_alignment].tolist(), truth_alignment)}
    results["lexical_binding"] = mixture(20 * torch.tensor(lexical), table)
    primary = None
    for arm, model in models.items():
        edge = model(rows).flatten()
        results[arm] = mixture(edge, table)
        if arm == "two_slot_return":
            primary = edge
    patches = {}
    for arm in INTERVENTIONS:
        if observed is None:
            results[arm] = results["two_slot_return"]
            continue
        edge = primary.view(3, 3).clone()
        column = {dest: value for dest, value in enumerate(edge[:, observed_arm].tolist())}
        a, b, _ = sorted(column, key=column.__getitem__, reverse=True)
        sign = None if arm == INTERVENTIONS[0] else (1 if int(identity[-1], 16) % 2 else -1)
        shifted, dose = redistribute(column, a, b, sham_sign=sign)
        edge[:, observed_arm] = edge.new_tensor([shifted[dest] for dest in range(3)])
        results[arm] = mixture(edge, table)
        patches[arm] = {"l2_dose": dose, "bound_before": ALIASES[results["two_slot_return"][1]].index(observed_arm),
                        "bound_after": ALIASES[results[arm][1]].index(observed_arm)}
    if patches and patches[INTERVENTIONS[0]]["l2_dose"] != patches[INTERVENTIONS[1]]["l2_dose"]:
        raise ValueError("the binding interventions changed physical dose")
    return results, patches


@torch.no_grad()
def evaluate(args, tree):
    bank, rewritten, pins, reader = locked(args)
    trained = torch.load(args.fit, map_location="cpu", weights_only=True)
    current = {name: value for name, value in pins.items() if name != "rewritten"}
    sealed = {name: value for name, value in trained["pins"].items() if name != "rewritten"}
    stress = rewritten.get("form") == "unfolded-development"
    if (trained["schema"] != SCHEMA or current != sealed
            or not stress and trained["pins"]["rewritten"] != pins["rewritten"]):
        raise ValueError("the locked rewrite fit changed")
    models = models_for()
    for arm, model in models.items():
        state = trained["weights"][arm]
        if tensor_hash(state) != trained["weight_sha"][arm]:
            raise ValueError("a frozen rewrite model changed")
        model.load_state_dict(state)
        model.eval()
    prior = MixedSourcePrior(args)
    old = {case["module"]: case for case in bank["development"]}
    records, all_cases = defaultdict(list), []
    for pair in rewritten["development"]:
        source, target = old[pair["source"]], pair["target"]
        all_cases.extend((source, target))
        source_shape = pack_world(source, prior)
        weights = reader(source_shape.reshape(1, -1)).softmax(-1)[0].tolist()
        rows, lexical, projection, first, paid, observed, _ = episode(source, target, weights, prior)
        table = torch.tensor(projection)
        reset = torch.tensor(mix_table(weights, paid, None))
        actual = ALIASES.index(tuple(pair["alignment"]))
        results, patches = posteriors(models, torch.tensor(rows), lexical, table, reset, paid, observed, actual, source["module"])
        tags = {"module": source["module"], "pair": source["pair"], "family": source["family"],
                "first_module": source["module"], "second_module": target["module"]}
        for budget in ALLOWANCES:
            for arm, (belief, binding) in results.items():
                second = solve_target(target, budget, belief)
                ordered = ALIASES[binding] if binding is not None else None
                records[arm].append({**tags, "allowance": budget, "first": first, "second": second,
                                     "checks": first["checks"] + second["checks"], "choice": paid,
                                     "proofs": int(first["proved"]) + int(second["proved"]), "utility": second["utility"],
                                     "patch": patches.get(arm), "binding_correct": None if binding is None else binding == actual,
                                     "observed_binding_correct": None if binding is None or observed is None
                                     else ordered.index(paid) == tuple(pair["alignment"]).index(paid)})
    certificates, live_replays = replays(args, all_cases, records,
                                          ("two_slot_return", "trm_return", "two_slot_supervised", *INTERVENTIONS),
                                          patched_arms=INTERVENTIONS)
    summaries = {name: summarize(rows) for name, rows in records.items()}
    comparisons = {name: root_contrast(records["two_slot_return"], rows) for name, rows in records.items()
                   if name != "two_slot_return"}
    for name, row in summaries.items():
        print(name, row["second_proofs"], row["checks"], row["utility"], flush=True)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit),
                                 "floor": FLOOR, "source_pairs": trained["source_pairs"], "summary": summaries,
                                 "posthoc_unfolding": stress, "training_bank_sha256": trained["pins"]["rewritten"],
                                 "root_contrasts": comparisons, "results": records, "independent": certificates,
                                 "live_replays": live_replays, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "anchor", "rewritten", "fit", "results"):
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
