"""Fit rule-aligned quote composition and match it against full-history controls."""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch

from configs.freeze_ppt_bound_quotes import SCHEMA as BANK_SCHEMA
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_quote_policy import (
    affine_features, audit_native, choice, fit_model, inputs_for, plan_features, prepare, summarize,
)
from configs.run_ppt_reuse_plans import apply_ridge, ridge
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, REPL_EXE
from ppt_gym.bound_quotes import REGIMES, outcomes, public_view
from ppt_gym.data import sha256_file
from ppt_gym.private_quotes import COMMITS, QUOTES, metered_rollout
from ppt_gym.quote_policy import ACTIONS, HISTORIES, FullHistoryQuoteTRM, RuleQuotePolicy, decision_mask
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.bound-quote-policy.v1"
SEED = 1063
UPDATES = 640
BATCH = 16


def registered(args):
    bank, fragments = read_receipt(args.bank), read_receipt(args.fragments)
    pins = {name: sha256_file(getattr(args, name)) for name in ("bank", "fragments", "warm", "prior")}
    source = {case["module"] for case in bank["source"]}
    held = {case["module"] for case in bank["held"]}
    rows = fragments["source"]
    if (bank["schema"] != BANK_SCHEMA or fragments["schema"] != BANK_SCHEMA or source & held
            or fragments["bank_sha256"] != pins["bank"] or bank["lean_sha256"] != sha256_file(LEAN_EXE)
            or bank["repl_sha256"] != sha256_file(REPL_EXE)
            or {(row["module"], row["regime"]) for row in rows} != {(root, regime) for root in source for regime in REGIMES}
            or len(rows) != len(source) * len(REGIMES)):
        raise ValueError("a frozen native or source-only feedback input changed")
    return bank, rows, pins


def encode(view, plan, index):
    links = [view["links"][name] for name in QUOTES]
    tensor = plan.new_tensor([[entry["arity"] / 4, entry["banded"], *entry["direct"], *entry["inverse"]] for entry in links])
    return (*inputs_for(view, plan, index), tensor)


def calibration_inputs(inputs):
    links = inputs[-1]
    history = torch.tensor(HISTORIES)[inputs[4]]
    bought_high = (history == 1).unsqueeze(-1) * links[..., 2:4]
    bought_low = (history == 0).unsqueeze(-1) * links[..., 4:]
    return torch.cat((affine_features(inputs[:-1]), links.flatten(1), bought_high.flatten(1), bought_low.flatten(1)), -1)


def models_for(width):
    policy = RuleQuotePolicy(width)
    return {"rule_bellman": policy, "rule_empirical": deepcopy(policy), "terminal_fit": deepcopy(policy),
            "unfitted": deepcopy(policy), "full_history_trm": FullHistoryQuoteTRM(width, bound=True)}


def fit(args, tree):
    bank, fragments, pins = registered(args)
    prior = MixedSourcePrior(args)
    features = plan_features(bank["source"], prior)
    source = prepare(bank["source"], fragments, features, view_for=public_view, encode=encode)
    width = source.inputs[0].shape[-1]
    models = models_for(width)
    initial = {arm: tensor_hash(model.state_dict()) for arm, model in models.items()}
    if len({initial[arm] for arm in ("rule_bellman", "rule_empirical", "terminal_fit", "unfitted")}) != 1:
        raise ValueError("the matched rule models have different initializations")
    batches = torch.randint(len(source.groups), (UPDATES, BATCH), generator=torch.Generator().manual_seed(SEED))
    commit_only = source.bellman.clone()
    commit_only[:, len(COMMITS):] = torch.nan
    for arm, model in models.items():
        if arm == "unfitted":
            continue
        labels = {"rule_empirical": source.empirical, "terminal_fit": commit_only}.get(arm, source.bellman)
        fit_model(model, source, labels, batches, .0025 if arm == "full_history_trm" else .006)
        print("fitted", arm, sum(p.numel() for p in model.parameters()), flush=True)
    x = calibration_inputs(source.inputs)
    frozen = [ridge(x[label.isfinite()], label[label.isfinite()], source.groups[label.isfinite()])
              for label in source.bellman.T]
    states = {arm: model.state_dict() for arm, model in models.items()}
    with args.fit.open("xb") as handle:
        torch.save({"schema": SCHEMA, "pins": pins, "weights": states, "plan_width": width, "frozen": frozen,
                    "initial_hashes": initial, "hashes": {arm: tensor_hash(state) for arm, state in states.items()},
                    "sampler_sha256": tensor_hash({"batches": batches}), "source_states": len(source.groups),
                    "mixed_sources": prior.counts, "memory": tree.stats()}, handle)
    print("sealed weights", sha256_file(args.fit), flush=True)


def score_table(view, plans, model, checkpoint):
    inputs = tuple(torch.stack(items) for items in zip(
        *(encode(view, plans, index) for index in range(len(HISTORIES))), strict=True))
    if model is not None:
        scores, _ = model(*inputs)
    else:
        x = calibration_inputs(inputs)
        scores = torch.stack([apply_ridge(x, fit) for fit in checkpoint["frozen"]], -1)
    mask, _ = decision_mask(inputs[2], inputs[3], torch.tensor(HISTORIES))
    return scores.masked_fill(~mask, -torch.inf)


def replace_paid_bit(view, scenarios, world, natural, selector):
    observed = natural["trace"]
    for index in reversed(range(len(observed))):
        target = observed[index]
        action = target["action"]
        compatible = [donor for donor, (_, bits) in enumerate(scenarios)
                      if bits[action] != target["actual"]
                      and all(bits[step["action"]] == step["actual"] for step in observed[:index])]
        if not compatible:
            continue
        replacement = scenarios[compatible[0]][1][action]
        prices, bits = scenarios[world]
        patched = metered_rollout(view, prices, bits, selector,
                                  patch=lambda name, bit: replacement if name == action else bit)
        sham = metered_rollout(view, prices, bits, selector, patch=lambda _name, bit: bit)
        if (sham != natural or patched["trace"][:index] != observed[:index]
                or patched["trace"][index] != {**target, "seen": replacement}):
            raise ValueError("a donor changed the prefix before the charged observation")
        return {"target": action, "donor": compatible[0], "replacement": replacement,
                "position": index, "edited": patched, "identity_sham": True}
    return None


def permute_links(view):
    return {**view, "links": {name: {**channel, "direct": channel["direct"][::-1], "inverse": channel["inverse"][::-1]}
                              for name, channel in view["links"].items()}}


def replay(view, prices, bits, scores, *, flip_priority=False):
    def select(menu, history, *_):
        if flip_priority:
            menu = (menu[1], menu[0], *menu[2:])
        return choice(scores, menu, history)

    return metered_rollout(view, prices, bits, select), select


@torch.no_grad()
def evaluate(args, tree):
    bank, _, pins = registered(args)
    checkpoint = torch.load(args.fit, map_location="cpu", weights_only=True)
    if checkpoint["schema"] != SCHEMA or pins != checkpoint["pins"]:
        raise ValueError("the frozen checkpoint no longer matches the source inputs")
    models = models_for(checkpoint["plan_width"])
    for arm, model in models.items():
        state = checkpoint["weights"][arm]
        if tensor_hash(state) != checkpoint["hashes"][arm]:
            raise ValueError("frozen quote weights changed")
        model.load_state_dict(state)
        model.eval()
    held = bank["held"]
    features = plan_features(held, MixedSourcePrior(args))
    records, interventions = defaultdict(list), defaultdict(list)
    for arm in (*models, "frozen_calibration"):
        model = models.get(arm)
        for case in held:
            plans = features[case["module"]]
            scenarios = outcomes(case)
            swapped_case = {**case, "plans": case["plans"][::-1]}
            for regime in REGIMES:
                view, swapped = public_view(case, regime), public_view(swapped_case, regime)
                bound = permute_links(view)
                if bound["links"] != swapped["links"]:
                    raise ValueError("the model's relabeled graph disagrees with literal Lean incidence")
                scores = score_table(view, plans, model, checkpoint)
                mirror = score_table(swapped, plans.flip(0), model, checkpoint)
                wrong_link = score_table(bound, plans, model, checkpoint)
                if isinstance(model, RuleQuotePolicy):
                    order = [1, 0, *range(len(COMMITS), len(ACTIONS))]
                    torch.testing.assert_close(mirror[:, order], scores, rtol=1e-6, atol=1e-6)
                    # Roundoff on equivalent plans should not turn an implementation tie into a policy change.
                    mirror = scores[:, order]
                for world, (prices, bits) in enumerate(scenarios):
                    natural, select = replay(view, prices, bits, scores)
                    bpd = replace_paid_bit(view, scenarios, world, natural, select)
                    moved, _ = replay(swapped, prices, bits, mirror, flip_priority=True)
                    misbound, _ = replay(view, prices, bits, wrong_link)
                    tags = {"root": case["module"], "family": case["family"], "regime": regime, "world": world}
                    records[arm].append({**tags, **natural, "bpd": bpd})
                    interventions[f"{arm}_presentation"].append({**tags, **moved, "bpd": None})
                    interventions[f"{arm}_binding_swap"].append({**tags, **misbound, "bpd": None})
        print(arm, summarize(records[arm]), flush=True)
    all_rows = {**records, **interventions}
    audit = audit_native(held, all_rows)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit),
                                 "results": all_rows, "summary": {arm: summarize(rows) for arm, rows in all_rows.items()},
                                 "native_audit": audit, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "fragments", "warm", "prior", "fit", "results"):
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
