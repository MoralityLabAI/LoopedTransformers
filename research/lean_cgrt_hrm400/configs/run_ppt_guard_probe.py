"""Fit observation-conditioned probe selection with fixed witness commitment."""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn import functional as F

from configs.run_ppt_guard_belief import SCHEMA as BELIEF_SCHEMA
from configs.run_ppt_guard_belief import audit_routes, root_contrast
from configs.run_ppt_guard_recovery import encode_case, summary
from configs.run_ppt_guard_residual import cross_fit, load_inputs
from configs.run_ppt_guard_return import SCHEMA as RETURN_SCHEMA
from configs.run_ppt_guard_return import checked_archive
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_rollback_diagnostics import write_receipt
from ppt_gym.data import sha256_file
from ppt_gym.guard_belief import LOGIT_BOUND, BeliefCorrection
from ppt_gym.guard_probe import PROBE_TEMPERATURE, ProbeGraph, best_action, revised_scores
from ppt_gym.guard_recovery import ALLOWANCES, PROBE, feasible_menu
from ppt_gym.guard_recovery_learning import GUARD_WORLDS, baked_scores, trace_policy
from ppt_gym.live_quote_policy import LiveQuoteTRM, TwoSlotQuoteActor
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.guard-probe.v1"
SEED = 1125
UPDATES = 480
BATCH = 16
FLOOR = .025
INTERVENTIONS = ("chosen_swap", "bystander_swap")


def models_for():
    actor = BeliefCorrection(TwoSlotQuoteActor())
    return {"two_slot_return": actor, "two_slot_supervised": deepcopy(actor),
            "trm_return": BeliefCorrection(LiveQuoteTRM()), "affine_return": BeliefCorrection(),
            "affine_supervised": BeliefCorrection()}


def prepare(cases, cache, graph):
    encodings = [graph.encode(*cache[case["module"]]) for case in cases]
    world = torch.tensor([GUARD_WORLDS.index(tuple(case["truth"])) for case in cases])
    return (world, *(torch.tensor(rows) for rows in zip(*encodings, strict=True)))


def corrections(model, tokens):
    return model.correction(tokens.flatten(0, 1)).reshape(*tokens.shape[:2], 4)


def train(model, arm, graph, data, batches):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for batch in batches:
        world, tokens, prior, enabled, offsets, payoffs, edges, rewards, baseline = [value[batch] for value in data]
        correction = corrections(model, tokens)
        if arm.endswith("supervised"):
            eligible = enabled & graph.support.T[world]
            logits = (prior[:, None, :] + correction).masked_fill(~graph.support, -torch.inf)
            labels = world[:, None].expand_as(enabled)
            loss = F.cross_entropy(logits[eligible], labels[eligible])
        else:
            values = graph.candidate_values(prior, correction, offsets, payoffs)
            policy = (values / PROBE_TEMPERATURE).softmax(dim=-1)
            loss = -graph.rollout(policy, world, enabled, edges, rewards, baseline)[:, 1:].mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        world, tokens, prior, enabled, offsets, payoffs, edges, rewards, baseline = data
        values = graph.candidate_values(prior, corrections(model, tokens), offsets, payoffs)
        policy = (values / PROBE_TEMPERATURE).softmax(dim=-1)
        source = graph.rollout(policy, world, enabled, edges, rewards, baseline)
    return source.mean(dim=0).tolist()


def fit(args, tree):
    bank, pins, _ = load_inputs(args)
    rng = torch.Generator().manual_seed(SEED)
    cache, folds, _, _ = cross_fit(bank["source"], MixedSourcePrior(args), rng)
    graph = ProbeGraph()
    data = prepare(bank["source"], cache, graph)
    batches = torch.randint(len(data[0]), (UPDATES, BATCH), generator=rng)
    models = models_for()
    init = {arm: tensor_hash(model.state_dict()) for arm, model in models.items()}
    if init["two_slot_return"] != init["two_slot_supervised"] or init["affine_return"] != init["affine_supervised"]:
        raise ValueError("the matched probe fits changed initialization")
    utility = {}
    for arm, model in models.items():
        utility[arm] = train(model, arm, graph, data, batches)
        print("fitted", arm, utility[arm], flush=True)
    weights = {arm: model.state_dict() for arm, model in models.items()}
    receipt = {"schema": SCHEMA, "pins": pins, "weights": weights, "initial": init,
               "weight_sha": {arm: tensor_hash(value) for arm, value in weights.items()},
               "sampler_sha": tensor_hash({"batches": batches}), "source_soft_utility": utility,
               "folds": folds, "states": len(graph.states), "bound": LOGIT_BOUND,
               "softmax": PROBE_TEMPERATURE, "memory": tree.stats()}
    with args.fit.open("xb") as handle:
        torch.save(receipt, handle)
    print("sealed probe fit", sha256_file(args.fit), flush=True)


def paired_intervention(correction, selected, native, sham):
    first, second = PROBE.index(selected), PROBE.index(native)
    difference = correction[second] - correction[first]
    if sham:
        first, second = (i for i in range(4) if i not in (first, second))
    patched = list(correction)
    patched[first] += difference
    patched[second] -= difference
    return patched, 2 ** .5 * abs(difference)


def score_case(case, allowance, graph, base, encoded, deltas=None, intervention=None, *, anchor_deltas=None):
    _, prior, enabled, _, payoffs, _, _, _ = encoded
    changes, patch = [], None

    def select(state, _menu):
        nonlocal patch
        scores = base(state)
        native = best_action(scores)
        index = graph.index.get(state, 0) - 1
        if deltas is None or index < 0 or not enabled[index]:
            return native
        local = {move: value for move, value in scores.items() if move in PROBE}
        support = graph.support[index].tolist()
        anchor = anchor_deltas[index] if anchor_deltas is not None else [0.] * len(PROBE)
        if anchor_deltas is not None:
            anchor_values = revised_scores(local, payoffs[index], prior, anchor, support)
            native = max(anchor_values, key=anchor_values.__getitem__)
        changed = revised_scores(local, payoffs[index], prior, deltas[index], support)
        selected = max(changed, key=changed.__getitem__)
        if selected != native:
            step = allowance - 2 - state.remaining
            changes.append({"step": step, "native": native, "learned": selected})
            if intervention and patch is None and len(local) == 4:
                update = [value - initial_value for value, initial_value in zip(deltas[index], anchor, strict=True)]
                revised, dose = paired_intervention(update, selected, native, intervention == "bystander_swap")
                revised = [initial_value + value for initial_value, value in zip(anchor, revised, strict=True)]
                patched = revised_scores(local, payoffs[index], prior, revised, support)
                altered = max(patched, key=patched.__getitem__)
                patch = {"step": step, "before": selected, "after": altered, "l2_dose": dose}
                return altered
        return selected

    return {**trace_policy(case, allowance, select, action_space=feasible_menu), "changes": changes, "patch": patch}


@torch.no_grad()
def evaluate(args, tree):
    bank, pins, reader = load_inputs(args)
    belief = checked_archive(args.belief_control, BELIEF_SCHEMA, pins)
    previous = checked_archive(args.return_control, RETURN_SCHEMA, pins)
    fitted = torch.load(args.fit, map_location="cpu", weights_only=True)
    if (fitted["schema"] != SCHEMA or fitted["pins"] != pins or fitted["bound"] != LOGIT_BOUND
            or fitted["softmax"] != PROBE_TEMPERATURE):
        raise ValueError("the locked probe fit changed")
    models = models_for()
    for arm, model in models.items():
        weights = fitted["weights"][arm]
        if tensor_hash(weights) != fitted["weight_sha"][arm]:
            raise ValueError("a frozen probe fit changed")
        model.load_state_dict(weights)
        model.eval()
    graph = ProbeGraph()
    prior_model = MixedSourcePrior(args)
    records, certificates, heads, soft_utility = defaultdict(list), {}, {}, defaultdict(list)
    for case in bank["development"]:
        descriptors, probability, warm = encode_case(case, reader, prior_model)
        encoded = graph.encode(descriptors, probability, warm)
        base = baked_scores(probability, action_space=feasible_menu)
        tokens = torch.tensor([encoded[0]])
        model_output = {arm: corrections(model, tokens) for arm, model in models.items()}
        deltas = {arm: output[0].tolist() for arm, output in model_output.items()}
        heads[case["module"]] = deltas
        tags = {key: case[key] for key in ("module", "pair", "family")}
        for budget in ALLOWANCES:
            natural = None
            for arm in (*models, "frozen_planner", *INTERVENTIONS):
                retained = "two_slot_return" if arm in INTERVENTIONS else arm
                row = score_case(case, budget, graph, base, encoded, deltas.get(retained),
                                 arm if arm in INTERVENTIONS else None)
                if arm == "two_slot_return":
                    natural = row
                if arm in INTERVENTIONS:
                    patch = row["patch"]
                    step = patch["step"] if patch else len(natural["trace"])
                    if (row["trace"][:step] != natural["trace"][:step]
                            or patch is None and row["trace"] != natural["trace"]):
                        raise ValueError("the probe intervention changed the paid prefix")
                records[arm].append({**tags, **row})
                if row["proved"]:
                    certificates[case["module"], tuple(row["proof"])] = case
        base.cache_clear()
        # This readout uses the training temperature and is separate from greedy scoring.
        _, prior, enabled, offsets, payoffs, edges, rewards, baseline = (torch.tensor([part]) for part in encoded)
        world = torch.tensor([GUARD_WORLDS.index(tuple(case["truth"]))])
        for arm, output in model_output.items():
            values = graph.candidate_values(prior, output, offsets, payoffs)
            policy = (values / PROBE_TEMPERATURE).softmax(dim=-1)
            reward = graph.rollout(policy, world, enabled, edges, rewards, baseline)
            soft_utility[arm].append(reward[0].tolist())
    if records["frozen_planner"] != belief["results"]["frozen_planner"]:
        raise ValueError("the frozen reference traces changed")
    for a, b in zip(*(records[arm] for arm in INTERVENTIONS), strict=True):
        if (a["patch"] is None) != (b["patch"] is None):
            raise ValueError("the probe interventions lost matching eligibility")
        if a["patch"] and (a["patch"]["step"], a["patch"]["l2_dose"]) != (b["patch"]["step"], b["patch"]["l2_dose"]):
            raise ValueError("the probe interventions lost matching dose or paid prefix")
    independent, replays = audit_routes(args, bank["development"], records, certificates,
                                        ("two_slot_return", "two_slot_supervised", "trm_return", *INTERVENTIONS))
    controls = {**records, "archived_supervised_belief": belief["results"]["supervised_belief"],
                "archived_checks_only": previous["results"]["check_only"]}
    spare = {arm: [row for row in rows if row["allowance"] >= 7] for arm, rows in controls.items()}
    contrasts = {arm: root_contrast(spare["two_slot_return"], rows) for arm, rows in spare.items()
                 if arm != "two_slot_return"}
    results = {arm: {**summary(rows), "spare": summary(spare[arm]),
                     "by_budget": {budget: summary([row for row in rows if row["allowance"] == budget])
                                   for budget in ALLOWANCES}} for arm, rows in controls.items()}
    for arm, row in results.items():
        print(arm, row["proofs"], row["spare"]["proofs"], row["spare"]["utility"], flush=True)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit),
                                 "archives": {k: sha256_file(getattr(args, k)) for k in ("belief_control", "return_control")},
                                 "fitting": {k: fitted[k] for k in ("initial", "weight_sha", "sampler_sha", "states",
                                                                   "source_soft_utility", "memory")},
                                 "development_soft_utility": {arm: torch.tensor(rows).mean(dim=0).tolist()
                                                              for arm, rows in soft_utility.items()},
                                 "floor": FLOOR, "summary": results, "spare_root_contrasts": contrasts, "heads": heads,
                                 "results": records, "independent": independent, "live_replays": replays,
                                 "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "planner", "fit", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--belief-control", type=Path)
    parser.add_argument("--return-control", type=Path)
    args = parser.parse_args()
    if args.phase == "eval" and (args.belief_control is None or args.return_control is None):
        parser.error("eval requires the audited --belief-control and --return-control receipts")
    tree = TreeBudget(2, 1)
    try:
        configure(1, SEED)
        (fit if args.phase == "fit" else evaluate)(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
