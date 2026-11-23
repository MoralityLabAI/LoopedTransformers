"""Adapt a frozen supervised probe selector on native proof return."""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn import functional as F

from configs.freeze_ppt_guard_curriculum import NAMESPACE, SCHEMA as CURRICULUM_SCHEMA, audit_cohort
from configs.run_ppt_guard_belief import audit_routes, root_contrast
from configs.run_ppt_guard_probe import BATCH, INTERVENTIONS, SCHEMA as PROBE_SCHEMA, UPDATES as ANCHOR_UPDATES
from configs.run_ppt_guard_probe import corrections, prepare, score_case, train
from configs.run_ppt_guard_recovery import encode_case, summary
from configs.run_ppt_guard_residual import cross_fit, load_inputs
from configs.run_ppt_guard_return import checked_archive
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_rollback_diagnostics import write_receipt
from ppt_gym.data import sha256_file
from ppt_gym.guard_belief import BeliefCorrection
from ppt_gym.guard_probe import PROBE_TEMPERATURE, ProbeGraph
from ppt_gym.guard_recovery import ALLOWANCES, feasible_menu
from ppt_gym.guard_recovery_learning import GUARD_WORLDS, baked_scores
from ppt_gym.live_quote_policy import LiveQuoteTRM, TwoSlotQuoteActor
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.guard-warm-probe.v1"
SEED = 1126
UPDATES = 240
BLEND = .35
RATE = .0004
KL_WEIGHT = .025
FLOOR = .025
PRIMARY = "two_slot_deep_return"
ADAPTERS = {
    "two_slot_shallow_return": ("two_slot", "shallow", "return"),
    PRIMARY: ("two_slot", "deep", "return"),
    "two_slot_deep_supervised": ("two_slot", "deep", "supervised"),
    "trm_deep_return": ("trm", "deep", "return"),
    "trm_deep_supervised": ("trm", "deep", "supervised"),
    "affine_deep_return": ("affine", "deep", "return"),
}


def make_anchors():
    return {"two_slot": BeliefCorrection(TwoSlotQuoteActor()), "trm": BeliefCorrection(LiveQuoteTRM()),
            "affine": BeliefCorrection()}


def checked_inputs(args):
    bank, pins, reader = load_inputs(args)
    curriculum = read_receipt(args.curriculum)
    bank_pins = {"bank": pins["bank"], **{key: bank[key] for key in ("lean", "repl", "imports")}}
    if (curriculum["schema"] != CURRICULUM_SCHEMA or curriculum["namespace"] != NAMESPACE
            or curriculum["pins"] != bank_pins
            or curriculum["overlap"] != audit_cohort(curriculum["source"], bank)
            or curriculum["independent"]["certificates"] != 2 * len(curriculum["source"])):
        raise ValueError("the sealed deep source changed")
    warm = torch.load(args.warm_probe, map_location="cpu", weights_only=True)
    if warm["schema"] != PROBE_SCHEMA or warm["pins"] != pins:
        raise ValueError("the supervised probe anchor uses different inputs")
    return bank, curriculum["source"], pins, reader, warm


def load_weights(model, weights, sha):
    if tensor_hash(weights) != sha:
        raise ValueError("sealed probe weights changed")
    model.load_state_dict(weights)
    return model.eval()


def blended(anchor, proposal):
    return anchor + BLEND * (proposal - anchor)


def probe_policy(graph, prior, adjustment, offsets, payoffs):
    scores = graph.candidate_values(prior, adjustment, offsets, payoffs) / PROBE_TEMPERATURE
    log_prob = F.log_softmax(scores, dim=-1)
    return log_prob.exp(), log_prob.masked_fill(~graph.available, 0.)


def adapt(model, anchor, graph, objective, data, batches):
    with torch.no_grad():
        original = corrections(anchor, data[1])
        reference, reference_log = probe_policy(graph, data[2], original, data[4], data[5])
    optimizer = torch.optim.AdamW(model.parameters(), lr=RATE)
    for indices in batches:
        world, tokens, prior, enabled, offsets, payoffs, edges, rewards, baseline = [value[indices] for value in data]
        candidate = blended(original[indices], corrections(model, tokens))
        policy, log_prob = probe_policy(graph, prior, candidate, offsets, payoffs)
        relative = (policy * (log_prob - reference_log[indices])).sum(dim=-1)
        penalty = relative[enabled].mean()
        if objective == "supervised":
            eligible = enabled & graph.support.T[world]
            logits = (prior[:, None, :] + candidate).masked_fill(~graph.support, -torch.inf)
            labels = world[:, None].expand_as(enabled)
            loss = F.cross_entropy(logits[eligible], labels[eligible])
        else:
            loss = -graph.rollout(policy, world, enabled, edges, rewards, baseline)[:, 1:].mean()
        optimizer.zero_grad()
        (loss + KL_WEIGHT * penalty).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        world, tokens, prior, enabled, offsets, payoffs, edges, rewards, baseline = data
        reference_utility = graph.rollout(reference, world, enabled, edges, rewards, baseline)
        policy, _ = probe_policy(graph, prior, blended(original, corrections(model, tokens)), offsets, payoffs)
        adapted_utility = graph.rollout(policy, world, enabled, edges, rewards, baseline)
    return {"anchor": reference_utility.mean(dim=0).tolist(), "adapted": adapted_utility.mean(dim=0).tolist()}


def fit(args, tree):
    bank, deep, pins, _, warm = checked_inputs(args)
    rng = torch.Generator().manual_seed(SEED)
    cache, folds, _, _ = cross_fit(bank["source"], MixedSourcePrior(args), rng, transfer=deep)
    graph = ProbeGraph()
    data = {"shallow": prepare(bank["source"], cache, graph), "deep": prepare(deep, cache, graph)}
    if len(data["shallow"][0]) != len(data["deep"][0]):
        raise ValueError("the curricula need matched sampling counts")
    samples = torch.randint(len(data["shallow"][0]), (UPDATES, BATCH), generator=rng)
    anchor_samples = torch.randint(len(data["shallow"][0]), (ANCHOR_UPDATES, BATCH), generator=rng)
    anchors = make_anchors()
    for name in ("two_slot", "affine"):
        key = f"{name}_supervised"
        load_weights(anchors[name], warm["weights"][key], warm["weight_sha"][key])
    train(anchors["trm"], "trm_supervised", graph, data["shallow"], anchor_samples)
    print("sealed supervised TRM anchor", flush=True)
    models, source_metrics = {}, {}
    for arm, (name, source, objective) in ADAPTERS.items():
        models[arm] = deepcopy(anchors[name])
        source_metrics[arm] = adapt(models[arm], anchors[name], graph, objective, data[source], samples)
        print("fitted", arm, source_metrics[arm], flush=True)
    weights = {arm: model.state_dict() for arm, model in models.items()}
    anchor_weights = {name: model.state_dict() for name, model in anchors.items()}
    receipt = {"schema": SCHEMA, "pins": pins, "weights": weights, "anchors": anchor_weights,
               "weight_sha": {arm: tensor_hash(value) for arm, value in weights.items()},
               "anchor_sha": {name: tensor_hash(value) for name, value in anchor_weights.items()},
               "data_sha": {"curriculum": sha256_file(args.curriculum), "warm_probe": sha256_file(args.warm_probe)},
               "adapter_sampler_sha": tensor_hash({"batches": samples}),
               "trm_anchor_sampler_sha": tensor_hash({"batches": anchor_samples}),
               "source_metrics": source_metrics, "blend": BLEND, "kl": KL_WEIGHT,
               "folds": folds, "memory": tree.stats()}
    with args.fit.open("xb") as handle:
        torch.save(receipt, handle)
    print("sealed warm probe fit", sha256_file(args.fit), flush=True)


@torch.no_grad()
def evaluate(args, tree):
    bank, _, pins, reader, warm = checked_inputs(args)
    archived = checked_archive(args.archive, PROBE_SCHEMA, pins)
    if archived["fit_sha256"] != sha256_file(args.warm_probe):
        raise ValueError("the archived probe scores use different weights")
    fitted = torch.load(args.fit, map_location="cpu", weights_only=True)
    if (fitted["schema"] != SCHEMA or fitted["pins"] != pins or fitted["blend"] != BLEND or fitted["kl"] != KL_WEIGHT
            or fitted["data_sha"] != {"curriculum": sha256_file(args.curriculum), "warm_probe": sha256_file(args.warm_probe)}):
        raise ValueError("the locked warm probe inputs changed")
    anchors = make_anchors()
    for name, model in anchors.items():
        sha = fitted["anchor_sha"][name]
        if name != "trm" and sha != warm["weight_sha"][f"{name}_supervised"]:
            raise ValueError("an inherited supervised anchor changed")
        load_weights(model, fitted["anchors"][name], sha)
    models = {arm: load_weights(deepcopy(anchors[name]), fitted["weights"][arm], fitted["weight_sha"][arm])
              for arm, (name, _, _) in ADAPTERS.items()}
    graph = ProbeGraph()
    prior_model = MixedSourcePrior(args)
    records, certificates, heads, soft = defaultdict(list), {}, {}, defaultdict(list)
    intervention_parents = {PRIMARY: INTERVENTIONS}
    if args.proposal_diagnostic:
        intervention_parents[f"{PRIMARY}_proposal"] = tuple(f"proposal_{arm}" for arm in INTERVENTIONS)
    for case in bank["development"]:
        descriptors, probability, warm_values = encode_case(case, reader, prior_model)
        encoded = graph.encode(descriptors, probability, warm_values)
        base = baked_scores(probability, action_space=feasible_menu)
        tokens = torch.tensor([encoded[0]])
        original = {name: corrections(model, tokens) for name, model in anchors.items()}
        proposals = {arm: corrections(model, tokens) for arm, model in models.items()}
        adapted = {arm: blended(original[ADAPTERS[arm][0]], proposal) for arm, proposal in proposals.items()}
        if args.proposal_diagnostic:
            adapted.update({f"{arm}_proposal": proposal for arm, proposal in proposals.items()})
        deltas = {arm: output[0].tolist() for arm, output in adapted.items()}
        anchor_deltas = {name: output[0].tolist() for name, output in original.items()}
        heads[case["module"]] = {"adapted": deltas, "anchors": anchor_deltas}
        tags = {name: case[name] for name in ("module", "pair", "family")}
        for budget in ALLOWANCES:
            choices = {}
            for arm, delta in deltas.items():
                anchor = anchor_deltas[ADAPTERS[arm.removesuffix("_proposal")][0]]
                choices[arm] = score_case(case, budget, graph, base, encoded, delta, anchor_deltas=anchor)
            for name, delta in anchor_deltas.items():
                choices[f"{name}_anchor"] = score_case(case, budget, graph, base, encoded, delta)
            for parent, arms in intervention_parents.items():
                natural = choices[parent]
                for arm, mode in zip(arms, INTERVENTIONS, strict=True):
                    row = score_case(case, budget, graph, base, encoded, deltas[parent], mode,
                                     anchor_deltas=anchor_deltas["two_slot"])
                    patch = row["patch"]
                    step = patch["step"] if patch else len(natural["trace"])
                    if (row["trace"][:step] != natural["trace"][:step]
                            or patch is None and row["trace"] != natural["trace"]):
                        raise ValueError("the probe intervention changed the paid prefix")
                    choices[arm] = row
                first, second = (choices[arm]["patch"] for arm in arms)
                if (first is None) != (second is None):
                    raise ValueError("the probe interventions disagree on eligibility")
                if first and (first["step"], first["l2_dose"]) != (second["step"], second["l2_dose"]):
                    raise ValueError("the probe interventions disagree on physical dose")
            for arm, row in choices.items():
                records[arm].append({**tags, **row})
                if row["proved"]:
                    certificates[case["module"], tuple(row["proof"])] = case
        base.cache_clear()
        if args.proposal_diagnostic:
            _, prior, enabled, offsets, payoffs, edges, rewards, baseline = (torch.tensor([part]) for part in encoded)
            world = torch.tensor([GUARD_WORLDS.index(tuple(case["truth"]))])
            candidates = {**adapted, **{f"{name}_anchor": output for name, output in original.items()}}
            for arm, output in candidates.items():
                policy, _ = probe_policy(graph, prior, output, offsets, payoffs)
                utility = graph.rollout(policy, world, enabled, edges, rewards, baseline)
                soft[arm].append(utility[0].tolist())
    archived_results = archived["results"]
    if records["two_slot_anchor"] != archived_results["two_slot_supervised"]:
        raise ValueError("the frozen two-slot supervised traces changed")
    if records["affine_anchor"] != archived_results["affine_supervised"]:
        raise ValueError("the frozen affine supervised traces changed")
    frozen_six = [row for row in archived_results["frozen_planner"] if row["allowance"] == 6]
    if any([row for row in rows if row["allowance"] == 6] != frozen_six for rows in records.values()):
        raise ValueError("a warm adapter changed a six-call trace")
    intervention_arms = tuple(arm for group in intervention_parents.values() for arm in group)
    independent, replays = audit_routes(args, bank["development"], records, certificates,
                                        (PRIMARY, "two_slot_deep_supervised", "trm_deep_return", *intervention_arms))
    controls = {**records, "frozen_planner": archived_results["frozen_planner"],
                "archived_cold_return": archived_results["two_slot_return"]}
    spare = {arm: [row for row in rows if row["allowance"] >= 7] for arm, rows in controls.items()}
    contrasts = {arm: root_contrast(spare[PRIMARY], rows) for arm, rows in spare.items() if arm != PRIMARY}
    results = {arm: {**summary(rows), "spare": summary(spare[arm]),
                     "by_budget": {budget: summary([row for row in rows if row["allowance"] == budget])
                                   for budget in ALLOWANCES}} for arm, rows in controls.items()}
    for arm, row in results.items():
        print(arm, row["proofs"], row["spare"]["proofs"], row["spare"]["utility"], flush=True)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit),
                                 "archive_sha256": sha256_file(args.archive), "floor": FLOOR, "summary": results,
                                 "fit": {key: fitted[key] for key in ("anchor_sha", "weight_sha", "data_sha",
                                                                     "adapter_sampler_sha", "trm_anchor_sampler_sha",
                                                                     "source_metrics", "memory")},
                                 "posthoc_soft_utility": {arm: torch.tensor(rows).mean(dim=0).tolist()
                                                          for arm, rows in soft.items()},
                                 "spare_root_contrasts": contrasts, "results": records, "heads": heads,
                                 "independent": independent, "live_replays": replays, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "planner", "curriculum", "warm_probe", "fit", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--proposal-diagnostic", action="store_true",
                        help="score unblended proposals and fixed-temperature utility as post-hoc diagnostics")
    args = parser.parse_args()
    if args.phase == "eval" and args.archive is None:
        parser.error("eval requires the audited --archive")
    tree = TreeBudget(2, 1)
    try:
        configure(1, SEED)
        (fit if args.phase == "fit" else evaluate)(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
