"""Fit and audit paid-quote control on SHA-bound native Lean plan fragments."""

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from math import fsum
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from configs.freeze_ppt_native_progress import advance
from configs.freeze_ppt_priced_fragments import checked_root
from configs.freeze_ppt_private_quotes import SCHEMA as BANK_SCHEMA
from configs.freeze_ppt_quote_fragments import SCHEMA as FRAGMENT_SCHEMA
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_native_progress import verify_independent
from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_reuse_plans import apply_ridge, ridge
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE, lean_env
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.live_lean import candidate_menu
from ppt_gym.private_quotes import COMMITS, QUOTES, REGIMES, public_view, quote_bit, rollout, scenario
from ppt_gym.quote_fragments import QuoteFragmentGraph
from ppt_gym.quote_policy import ACTIONS, HISTORIES, BeliefQuotePolicy, FullHistoryQuoteTRM, decision_mask, history_index
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.fitted-private-quote-policy.v1"
SEED = 1051
UPDATES = 480
BATCH = 16


@dataclass(frozen=True)
class Source:
    inputs: tuple
    groups: torch.Tensor
    bellman: torch.Tensor
    empirical: torch.Tensor
    forecast: torch.Tensor
    keys: tuple


def load_registered(args):
    bank = read_receipt(args.bank)
    fragments = read_receipt(args.fragments)
    if (bank["schema"] != BANK_SCHEMA or fragments["schema"] != FRAGMENT_SCHEMA
            or fragments["bank_sha256"] != sha256_file(args.bank) or bank["lean_sha256"] != sha256_file(LEAN_EXE)
            or bank["repl_sha256"] != sha256_file(REPL_EXE)):
        raise ValueError("the registered feedback, bank or Lean toolchain changed")
    source = {case["module"]: case for case in bank["source"]}
    held = {case["module"] for case in bank["held"]}
    observed = {(row["module"], row["regime"]) for row in fragments["source"]}
    if (held & source.keys() or observed != {(module, regime) for module in source for regime in REGIMES}
            or len(observed) != len(fragments["source"])):
        raise ValueError("the source fragment cohorts changed or overlap development")
    return bank, fragments["source"]


@torch.no_grad()
def plan_features(cases, prior):
    features = {}
    with LeanRepl() as repl:
        if repl.version != LEAN_VERSION:
            raise ValueError("unexpected native Lean version")
        for case in cases:
            state, public = checked_root(repl, case)
            if digest(public) != case["visible_sha256"]:
                raise ValueError("a frozen Lean observation changed")
            plans = case["plans"]
            history = []
            if plans[0][0] == "constructor":
                if plans[1][0] != "constructor":
                    raise ValueError("the two plans disagree on the public constructor")
                status, state, public = advance(repl, state, public, "constructor")
                if status != "progress":
                    raise ValueError("the certified constructor no longer makes progress")
                history.append(("constructor", 1.))
            opening = tuple(plan[len(history)] for plan in plans)
            if not set(opening) <= set(candidate_menu(repl, state)):
                raise ValueError("a certified plan changed its opening action")
            typed = prior.state(public, opening, history)
            graph = typed.graph
            public_types = [len(premises) / 8 for premises in typed.projected]
            relevant_facts = [sum(graph.features[atom - 1][0] for atom in premises) / 8 for premises in typed.projected]
            features[case["module"]] = torch.cat((typed.prior, torch.tensor([public_types, relevant_facts]).T), -1)
    return features


def inputs_for(view, plan, index):
    base = view["min_proof_effort"]
    fees = torch.tensor(tuple(view["fees"][action] for action in QUOTES), dtype=plan.dtype)
    slack = torch.tensor([view["budget"] - base], dtype=plan.dtype)
    context = torch.tensor([base / 12, view["budget"] / 12, view["max_surcharge"] / 6], dtype=plan.dtype)
    return plan, context, fees, slack, torch.tensor(index)


def table_key(view, index):
    return (view["budget"] - view["min_proof_effort"], view["max_surcharge"],
            *(view["fees"][action] for action in QUOTES), index)


def prepare(cases, fragments, features, *, view_for=public_view, encode=inputs_for):
    rows, keys, groups, bellman, empirical, forecast = [], [], [], [], [], []
    lookup = {case["module"]: case for case in cases}
    folds = {module: idx % 6 for idx, module in enumerate(sorted(lookup))}
    for row in fragments:
        module, regime = row["module"], row["regime"]
        view = view_for(lookup[module], regime)
        if view != row["view"]:
            raise ValueError("logged source prices and the public plan view disagree")
        graph = QuoteFragmentGraph(view, row["fragments"])
        for history in sorted(graph.states):
            index = history_index(history)
            rows.append(encode(view, features[module], index))
            keys.append(table_key(view, index))
            groups.append(folds[module])
            for arm, dest in ((0, bellman), (1, empirical)):
                dest.append([value_or_missing(graph.labels(history, action)[arm]) for action in ACTIONS])
            rates = []
            for action in QUOTES:
                branches = graph.transitions.get((history, action))
                rates.append(sum(dict(child)[action] * count for child, count in branches.items()) / branches.total()
                             if branches else torch.nan)
            forecast.append(rates)
    return Source(tuple(torch.stack(items) for items in zip(*rows, strict=True)), torch.tensor(groups),
                  torch.tensor(bellman), torch.tensor(empirical), torch.tensor(forecast), tuple(keys))


def value_or_missing(value):
    return torch.nan if value is None else value


def fit_model(model, source, labels, batches, rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate)
    for indices in batches:
        scores, forecast = model(*(item[indices] for item in source.inputs))
        observed = labels[indices]
        mask = observed.isfinite()
        target = source.forecast[indices]
        known = target.isfinite()
        if not mask.any() or not scores[mask].isfinite().all():
            raise ValueError("the model masked logged source actions")
        loss = F.mse_loss(scores[mask], observed[mask])
        if known.any():
            loss = loss + .25 * F.binary_cross_entropy(forecast[known].clamp(1e-6, 1 - 1e-6), target[known])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    return model.eval()


def affine_features(inputs):
    plans, context, fees, slack, ids = inputs
    history = torch.tensor(HISTORIES)[ids]
    _, left = decision_mask(fees, slack, history)
    symbols = F.one_hot(history + 1, 3).flatten(1)
    return torch.cat((plans.flatten(1), context, fees / 3, slack / 3, left / 3, symbols), -1)


def calibration(source):
    inputs = affine_features(source.inputs)
    frozen = []
    for observed in source.bellman.T:
        known = observed.isfinite()
        frozen.append(ridge(inputs[known], observed[known], source.groups[known]))
    tables = {}
    for kind, values in (("lookup_bellman", source.bellman), ("lookup_empirical", source.empirical)):
        groups = defaultdict(list)
        for key, row in zip(source.keys, values, strict=True):
            groups[key].append(row)
        tables[kind] = {key: torch.nanmean(torch.stack(rows), dim=0).nan_to_num(nan=-torch.inf)
                        for key, rows in groups.items()}
    return frozen, tables


def new_models(plan_width):
    model = BeliefQuotePolicy(plan_width)
    return {"belief_bellman": model, "belief_empirical": deepcopy(model), "full_history_trm": FullHistoryQuoteTRM(plan_width)}


def fit(args, tree):
    bank, fragments = load_registered(args)
    prior = MixedSourcePrior(args)
    features = plan_features(bank["source"], prior)
    source = prepare(bank["source"], fragments, features)
    batches = torch.randint(len(source.groups), (UPDATES, BATCH), generator=torch.Generator().manual_seed(SEED))
    width = source.inputs[0].shape[-1]
    models = new_models(width)
    initial = {name: tensor_hash(model.state_dict()) for name, model in models.items()}
    if initial["belief_bellman"] != initial["belief_empirical"]:
        raise ValueError("matched heads did not start at identical parameters")
    for name, model in models.items():
        labels = source.empirical if name == "belief_empirical" else source.bellman
        fit_model(model, source, labels, batches, .0025 if name == "full_history_trm" else .004)
        print("fitted", name, "parameters", sum(p.numel() for p in model.parameters()), flush=True)
    frozen, tables = calibration(source)
    states = {name: model.state_dict() for name, model in models.items()}
    with args.fit.open("xb") as handle:
        torch.save({"schema": SCHEMA, "plan_width": width, "bank_sha256": sha256_file(args.bank),
                    "fragment_sha256": sha256_file(args.fragments), "warm_sha256": sha256_file(args.warm),
                    "prior_sha256": sha256_file(args.prior), "mixed_sources": prior.counts,
                    "weights": states, "hashes": {name: tensor_hash(state) for name, state in states.items()},
                    "initial_hashes": initial, "sampler_sha256": tensor_hash({"batches": batches}),
                    "frozen": frozen, "tables": tables, "source_groups": len(source.groups),
                    "threads": tree.stats()["cpu_threads"]}, handle)
    print("frozen fit", sha256_file(args.fit), flush=True)


def policy_table(model, view, plan, checkpoint, arm):
    inputs = tuple(torch.stack(items) for items in zip(
        *(inputs_for(view, plan, index) for index in range(len(HISTORIES))), strict=True))
    if model is not None:
        scores, _ = model(*inputs)
    elif arm == "frozen_calibration":
        scores = torch.stack([apply_ridge(affine_features(inputs), readout) for readout in checkpoint["frozen"]], -1)
    else:
        tabulated = checkpoint["tables"][arm]
        missing = plan.new_full((len(ACTIONS),), -torch.inf)
        scores = torch.stack([tabulated.get(table_key(view, index), missing) for index in range(len(HISTORIES))])
    mask, _ = decision_mask(inputs[2], inputs[3], torch.tensor(HISTORIES))
    return scores.masked_fill(~mask, -torch.inf)


def choice(scores, menu, history):
    ranking = scores[history_index(history)]
    return max(menu, key=lambda action: float(ranking[ACTIONS.index(action)]))


def intervention(case, regime, world, natural, select):
    trace = natural["trace"]
    target = next((event for event in trace if event["action"] == "reconcile"), None)
    if target is None and len(trace) == 2:
        target = trace[-1]
    if target is None:
        return None
    action = target["action"]
    donor = world ^ (2 if action == "ledger_u" else 1)
    prices = scenario(case["plans"], case["ledger"], case["other"], REGIMES[regime], donor)
    bit = quote_bit(action, case["ledger"], prices)
    if bit == target["actual"]:
        return None

    def patch(name, original):
        return bit if name == action else original

    altered = rollout(case, regime, world, select, patch=patch)
    sham = rollout(case, regime, world, select, patch=lambda _name, original: original)
    if sham != natural:
        raise ValueError("an equal-observation sham changed the natural outcome")
    position = trace.index(target)
    if (altered["trace"][:position] != trace[:position] or len(altered["trace"]) <= position
            or altered["trace"][position] != {**target, "seen": bit}):
        raise ValueError("the paid observation patch changed a pre-intervention decision or charge")
    return {"donor": donor, "action": action, "original": target["actual"], "replacement": bit,
            "edited": altered, "equal_observation_sham": True,
            "plan_changed": altered["plan"] != natural["plan"], "proof_changed": altered["proof"] != natural["proof"]}


def audit_native(held, records):
    cases = {case["module"]: case for case in held}
    expected = {}
    for rows in records.values():
        for row in rows:
            outcomes = [row]
            if row["bpd"] is not None:
                outcomes.append(row["bpd"]["edited"])
            for outcome in outcomes:
                key = row["root"], tuple(outcome["script"])
                previous = expected.setdefault(key, outcome["proof"])
                if previous != outcome["proof"]:
                    raise ValueError("a paid prefix represented both failure and proof")
    with LeanRepl() as repl:
        if repl.version != LEAN_VERSION:
            raise ValueError("unexpected native Lean version")
        for (module, script), proof in expected.items():
            state, public = checked_root(repl, cases[module])
            for index, action in enumerate(script):
                if action not in candidate_menu(repl, state):
                    raise ValueError("a paid replay action left the native menu")
                status, state, public = advance(repl, state, public, action)
                want = "closed" if proof and index == len(script) - 1 else "progress"
                if status != want:
                    raise ValueError("the paid rollout diverged from Lean")
    certificates = [(cases[module], script) for (module, script), proof in expected.items() if proof]
    with TemporaryDirectory(prefix="ppt-paid-quotes-") as scratch:
        result = verify_independent(Path(scratch), lean_env(), (), certificates)
    return {**result, "native_prefixes": len(expected), "native_steps": sum(len(script) for _, script in expected)}


def summarize(rows):
    roots = defaultdict(list)
    for row in rows:
        roots[row["root"]].append(row["utility"])
    return {"proofs": sum(row["proof"] for row in rows), "count": len(rows),
            "quote_effort": sum(row["quote_effort"] for row in rows),
            "proof_effort": sum(row["proof_effort"] for row in rows),
            "utility": fsum(fsum(root) / len(root) for root in roots.values()) / len(roots)}


def restore_fit(args):
    checkpoint = torch.load(args.fit, map_location="cpu", weights_only=True)
    if (checkpoint["schema"] != SCHEMA or checkpoint["bank_sha256"] != sha256_file(args.bank)
            or checkpoint["fragment_sha256"] != sha256_file(args.fragments)
            or checkpoint["warm_sha256"] != sha256_file(args.warm) or checkpoint["prior_sha256"] != sha256_file(args.prior)):
        raise ValueError("the fit no longer matches its registered inputs")
    models = new_models(checkpoint["plan_width"])
    for name, model in models.items():
        state = checkpoint["weights"][name]
        if tensor_hash(state) != checkpoint["hashes"][name]:
            raise ValueError("a frozen model changed")
        model.load_state_dict(state)
        model.eval()
    return checkpoint, models


@torch.no_grad()
def evaluate(args, tree):
    bank, _ = load_registered(args)
    checkpoint, models = restore_fit(args)
    held = bank["held"]
    features = plan_features(held, MixedSourcePrior(args))
    records = defaultdict(list)
    arms = (*models, "frozen_calibration", *checkpoint["tables"])
    for arm in arms:
        for case in held:
            for regime in REGIMES:
                view = public_view(case, regime)
                scores = policy_table(models.get(arm), view, features[case["module"]], checkpoint, arm)

                def select(menu, history, _left, _fees):
                    return choice(scores, menu, history)

                for world in range(4):
                    row = rollout(case, regime, world, select)
                    bpd = intervention(case, regime, world, row, select)
                    records[arm].append({"root": case["module"], "family": case["family"],
                                         "regime": regime, "world": world, **row, "bpd": bpd})
        print(arm, summarize(records[arm]), flush=True)
    native = audit_native(held, records)
    write_receipt(args.results, {"schema": SCHEMA, "bank_sha256": sha256_file(args.bank),
                                "fit_sha256": sha256_file(args.fit), "summary": {
                                    arm: summarize(rows) for arm, rows in records.items()},
                                "results": records, "native_audit": native, "memory": tree.stats()})


@torch.no_grad()
def presentation(args, tree):
    """Swap plan presentation while keeping theorem, billed actions and observed prices fixed."""
    bank, _ = load_registered(args)
    checkpoint, models = restore_fit(args)
    held = bank["held"]
    features = plan_features(held, MixedSourcePrior(args))
    arms = (*models, "frozen_calibration", *checkpoint["tables"])
    records = defaultdict(list)
    natural = defaultdict(list)
    for arm in arms:
        for case in held:
            swapped = {**case, "plans": case["plans"][::-1]}
            plan = features[case["module"]]
            for regime, regime_spec in REGIMES.items():
                view = public_view(case, regime)
                scores = policy_table(models.get(arm), view, plan, checkpoint, arm)
                naive = policy_table(models.get(arm), public_view(swapped, regime), plan.flip(0), checkpoint, arm)
                aligned = scores[:, [1, 0, *range(len(COMMITS), len(ACTIONS))]]
                for world in range(4):
                    inputs = case["ledger"], case["other"], regime_spec, world
                    if scenario(case["plans"], *inputs) != scenario(swapped["plans"], *inputs):
                        raise ValueError("relabeling presentation changed the actual tariff")

                    def replay(layout, table, *, aligned_priority=False):
                        def select(menu, history, *_):
                            priority = (menu[1], menu[0], *menu[2:]) if aligned_priority else menu
                            return choice(table, priority, history)

                        return rollout(layout, regime, world, select)

                    base = replay(case, scores)
                    row = replay(swapped, naive)
                    remapped = replay(swapped, aligned, aligned_priority=True)
                    wanted = {**base, "plan": COMMITS[1 - COMMITS.index(base["plan"])]}
                    if remapped != wanted:
                        raise ValueError("the identity-aligned plan swap changed realized tactics or charges")
                    tags = {"root": case["module"], "family": case["family"], "regime": regime, "world": world}
                    natural[arm].append({**tags, **base})
                    records[arm].append({**tags, **row, "bpd": None,
                                         "physical_plan_changed": row["plan"] != wanted["plan"],
                                         "identity_aligned_plan": remapped["plan"]})
        print(arm, "swap", summarize(records[arm]), flush=True)
    native = audit_native(held, records)
    write_receipt(args.results, {"schema": SCHEMA, "intervention": "posthoc_plan_presentation",
                                "fit_sha256": sha256_file(args.fit), "bank_sha256": sha256_file(args.bank),
                                "results": records, "natural_summary": {arm: summarize(rows) for arm, rows in natural.items()},
                                "native_audit": native, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval", "presentation"))
    for name in ("bank", "fragments", "warm", "prior", "fit", "results"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        configure(1, SEED)
        {"fit": fit, "eval": evaluate, "presentation": presentation}[args.phase](args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
