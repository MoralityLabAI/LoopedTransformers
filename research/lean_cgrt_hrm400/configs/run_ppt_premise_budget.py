"""Fit budget composition and graph TRM on the same native search outcomes."""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import random
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from configs.freeze_ppt_premise_budget import SCHEMA as CURRICULUM
from configs.run_ppt_mathcan_progress import fixed_clock, read_receipt
from configs.run_ppt_mathcan_sparse import load_warm, summarize
from configs.run_ppt_native_feedback import typed_batch
from configs.run_ppt_native_progress import NativeInputs, WARM_SHA, replay, verify_independent
from configs.run_ppt_reuse_plans import apply_ridge, ridge
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE, lean_env
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.mathcan_deadline import deadline_utility
from ppt_gym.mathcan_sparse import CHECKS
from ppt_gym.native_feedback import NativeFeedbackRoute, PrefixGraph, behavior_choice, closure_curve, horizon_labels
from ppt_gym.premise_budget import BudgetRecurrence, BudgetTRM, budget_state
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.premise-budget-fit.v1"
SEED = 1021
UPDATES = 176
BATCH = 8
LEGACY_SHA = "608e6b258dbabc5b8af348ae13fb217443a56f9aa6d7d33290aa498a73c3ce21"


@dataclass(frozen=True)
class Example:
    state: object
    family: str
    bellman: tuple
    empirical: tuple


class MixedSourcePrior:
    def __init__(self, args):
        if sha256_file(args.warm) != WARM_SHA or sha256_file(args.prior) != LEGACY_SHA:
            raise ValueError("the mixed-source prior changed")
        hybrid, trm, counts = load_warm(args.warm)
        self.observations = NativeInputs(hybrid.text)
        self.model = NativeFeedbackRoute(trm, trm=True).eval()
        saved = torch.load(args.prior, weights_only=True, map_location="cpu")
        weights = saved["weights"]["trm_bellman"]
        if tensor_hash(weights) != saved["weight_hashes"]["trm_bellman"]:
            raise ValueError("the mixed-source prior state changed")
        self.model.load_state_dict(weights)
        self.counts = counts

    @torch.no_grad()
    def curves(self, public, menu, history):
        batch = fixed_clock(typed_batch(self.observations, public, menu, history, CHECKS))
        logits, delayed = self.model.factors(self.model.encode(*batch))
        return closure_curve(logits.softmax(-1), delayed)[0]

    def state(self, public, menu, history, *, prices=None):
        return budget_state(public, menu, self.curves(public, menu, history), prices=prices)


def prepare_rows(source, prior):
    rows = []
    for theorem in source:
        graph = PrefixGraph(theorem)
        reference = tuple(theorem["reference"])
        for prefix, node in graph.nodes.items():
            if not node["choices"]:
                continue
            options = tuple(node["choices"])
            bellman, empirical = [], []
            for action in options:
                status = node["choices"][action]["status"]
                if status == "closed":
                    known, sampled = (1,) * CHECKS, (1,) * CHECKS
                elif status == "progress":
                    child = (*prefix, action)
                    known = horizon_labels(*(count + 1 for count in graph.backed_up(child)))
                    sampled = horizon_labels(*(count + 1 for count in graph.empirical(child)))
                    if reference[:len(child)] == child:
                        remaining = len(reference) - len(prefix)
                        sampled = tuple(max(label, int(i >= remaining)) if i >= remaining else label
                                        for i, label in enumerate(sampled, start=1))
                else:
                    known, sampled = (-1,) * CHECKS, (-1,) * CHECKS
                bellman.append(known)
                empirical.append(sampled)
            history = [(action, 1.) for action in prefix]
            rows.append(Example(prior.state(node["public"], options, history), theorem["family"],
                                tuple(bellman), tuple(empirical)))
    PrefixGraph.backed_up.cache_clear()
    return rows


def train(model, rows, *, empirical=False, learning_rate=.0015):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    rng = random.Random(SEED)
    for _ in range(UPDATES):
        losses = []
        for row in rng.choices(rows, k=BATCH):
            predicted = model(row.state)
            observed = torch.tensor(row.empirical if empirical else row.bellman)
            known = observed != -1
            loss = F.binary_cross_entropy(predicted.clamp(1e-6, 1. - 1e-6), observed.clamp_min(0).float(), reduction="none")
            losses.append((loss * known).sum() / known.sum().clamp_min(1))
        optimizer.zero_grad()
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()


def calibration_features(frozen, state):
    graph = state.graph
    features = state.prior.new_tensor(graph.features)
    direct = torch.cat((features.new_zeros(1), features[:, 0]))
    summaries = [(len(premises) / CHECKS, float(direct[list(premises)].sum()) / CHECKS)
                 for premises in state.projected]
    info = state.prior.new_tensor(summaries)
    if state.action_prices:
        prices = state.prior.new_tensor(state.action_prices).unsqueeze(-1) / CHECKS
        info = torch.cat((info, prices), dim=-1)
    raw = torch.cat((state.prior, info), dim=-1)
    return {"prior": raw, "frozen_budget": torch.cat((raw, frozen(state)), dim=-1)}


@torch.no_grad()
def calibrate(frozen, rows):
    subsets = {}
    labels = torch.tensor([target for row in rows for target in row.bellman])
    families = {name: i for i, name in enumerate(sorted({row.family for row in rows}))}
    groups = torch.tensor([families[row.family] for row in rows for _ in row.bellman])
    for row in rows:
        for kind, features in calibration_features(frozen, row.state).items():
            subsets.setdefault(kind, []).append(features)
    heads = {}
    for kind, values in subsets.items():
        matrix = torch.cat(values)
        heads[kind] = []
        for horizon in labels.T:
            known = horizon >= 0
            heads[kind].append(ridge(matrix[known], horizon[known].float(), groups[known]))
    chosen = min(heads, key=lambda kind: sum(min(head["cv_error"].values()) for head in heads[kind]))
    return chosen, heads[chosen]


@torch.no_grad()
def evaluate(args, models, prior, calibration, held, tree, fit_path):
    arms = (*models, "calibrated", "shifted_budget", "mixed_source_prior", "public_behavior")
    results = {}
    proofs = {(case["module"], case["theorem"], tuple(case["reference"])): case for case in held}
    with LeanRepl() as repl:
        if repl.version != LEAN_VERSION:
            raise ValueError("unexpected native Lean toolchain")
        for arm in arms:
            def choose(public, menu, history, remaining):
                if arm == "public_behavior":
                    return behavior_choice(menu)
                state = prior.state(public, menu, history)
                if arm == "calibrated":
                    name, heads = calibration
                    features = calibration_features(models["unfitted_budget"], state)[name]
                    curve = torch.stack([apply_ridge(features, head) for head in heads], -1).clamp(0, 1).cummax(-1).values
                elif arm == "shifted_budget":
                    curve = models["budget_bellman"](state, foreign_budget=True)
                elif arm == "mixed_source_prior":
                    curve = state.prior
                else:
                    curve = models[arm](state)
                scores = deadline_utility(curve.unsqueeze(0), torch.tensor([remaining]))
                return menu[int(scores[0].argmax())]

            rows = []
            for case in held:
                row = replay(repl, case, case["checks"], choose)
                row["family"] = case["family"]
                rows.append(row)
                if row["proof"]:
                    proofs[case["module"], case["theorem"], tuple(row["script"])] = case
            results[arm] = rows
            print(arm, "closed", sum(row["proof"] for row in rows), len(rows), flush=True)
    with TemporaryDirectory(prefix="ppt-premise-certs-") as scratch:
        independent = verify_independent(Path(scratch), lean_env(), (), [(case, key[-1]) for key, case in proofs.items()])
    write_receipt(args.output / "results.json.gz",
                  {"schema": SCHEMA, "fit_sha256": sha256_file(fit_path), "curriculum_sha256": sha256_file(args.curriculum),
                   "results": results, "summary": {arm: summarize(rows) for arm, rows in results.items()},
                   "independent": independent, "memory": tree.stats()})


def run(args, tree):
    curriculum = read_receipt(args.curriculum)
    if (curriculum["schema"] != CURRICULUM or curriculum["lean_sha256"] != sha256_file(LEAN_EXE)
            or curriculum["repl_sha256"] != sha256_file(REPL_EXE)):
        raise ValueError("the registered source curriculum changed toolchains")
    held = {case["visible_sha256"] for case in curriculum["held"]}
    if any(digest(node["public"]) in held for case in curriculum["source"] for node in case["nodes"]):
        raise ValueError("a scoring theorem entered source learning")
    args.output.mkdir()
    prior = MixedSourcePrior(args)
    rows = prepare_rows(curriculum["source"], prior)
    print("source decisions", len(rows), "source tactics", sum(len(row.bellman) for row in rows),
          "available proof labels", sum(1 in target for row in rows for target in row.bellman),
          "behavior/demo proof labels", sum(1 in target for row in rows for target in row.empirical), flush=True)
    base = BudgetRecurrence()
    models = {"budget_bellman": base, "budget_empirical": deepcopy(base), "graph_trm": BudgetTRM(),
              "unfitted_budget": deepcopy(base).eval()}
    for arm, model in models.items():
        if arm != "unfitted_budget":
            train(model, rows, empirical=arm == "budget_empirical",
                  learning_rate=.0015 if arm == "graph_trm" else .025)
            model.eval()
            print("fitted", arm, flush=True)
    calibration = calibrate(models["unfitted_budget"], rows)
    state = {arm: model.state_dict() for arm, model in models.items()}
    fit_path = args.output / "fit.pt"
    with fit_path.open("xb") as handle:
        torch.save({"schema": SCHEMA, "weights": state, "calibration": calibration, "mixed_sources": prior.counts,
                    "prior_sha256": LEGACY_SHA, "curriculum_sha256": sha256_file(args.curriculum),
                    "weight_hashes": {arm: tensor_hash(weights) for arm, weights in state.items()}}, handle)
    checkpoint = torch.load(fit_path, weights_only=True, map_location="cpu")
    for arm, weights in checkpoint["weights"].items():
        if tensor_hash(weights) != checkpoint["weight_hashes"][arm]:
            raise ValueError("a sealed premise-budget fit changed")
        models[arm].load_state_dict(weights)
    evaluate(args, models, prior, checkpoint["calibration"], curriculum["held"], tree, fit_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("curriculum", "warm", "prior", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    configure(1, SEED)
    tree = TreeBudget(2, 1)
    try:
        run(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
