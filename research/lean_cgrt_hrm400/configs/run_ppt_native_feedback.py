"""Fit and compare a disposition-and-continuation policy on live Lean feedback."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from configs.collect_ppt_native_feedback import EVAL_REGISTRY_SHA, SCHEMA as TRAIN_SCHEMA, catalogue
from configs.freeze_ppt_native_progress import add_sources, candidate_menu, inventory, native_sources, support_paths
from configs.run_ppt_mathcan_progress import fixed_clock, read_receipt
from configs.run_ppt_mathcan_sparse import calibration_features, load_warm, summarize
from configs.run_ppt_native_progress import NativeInputs, WARM_SHA, replay, verify_independent
from configs.run_ppt_reuse_plans import apply_ridge, ridge
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.mathcan_deadline import deadline_utility
from ppt_gym.mathcan_sparse import CHECKS
from ppt_gym.native_feedback import (
    DISPOSITIONS, NativeFeedbackRoute, PrefixGraph, SuccessorFeedbackRoute, behavior_choice, closure_curve, horizon_labels,
)
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.native-feedback-fit.v1"
SEED = 991
STEPS = 320
BATCH = 24


@dataclass(frozen=True)
class Example:
    module: str
    public: dict
    prefix: tuple
    action: str
    disposition: int
    bellman: tuple
    empirical: tuple
    successor: dict | None


def examples(tape):
    rows = []
    for theorem in tape["rows"]:
        graph = PrefixGraph(theorem)
        group = "Horn:" + theorem["family"] if theorem.get("source") == "LoopedTransformers Horn" else theorem["module"]
        for prefix, node in graph.nodes.items():
            if set(candidate_menu(node["public"])) != set(node["choices"]):
                raise ValueError("a native training state changed its public action menu")
            for action, edge in node["choices"].items():
                status = edge["status"]
                if status == "resource_limit":
                    continue
                blank = (-1,) * CHECKS
                child = (*prefix, action)
                bellman = horizon_labels(*graph.backed_up(child)) if status == "progress" else blank
                empirical = horizon_labels(*graph.empirical(child)) if status == "progress" else blank
                successor = graph.nodes.get(child)
                rows.append(Example(group, node["public"], prefix, action,
                                    DISPOSITIONS.index(status), bellman, empirical, successor["public"] if successor else None))
    PrefixGraph.backed_up.cache_clear()
    return rows


def typed_batch(observations, public, menu, history, remaining):
    types = {hyp["name"]: hyp["type"].get("raw", canonical(hyp["type"]))
             for hyp in public["goals"][0]["hypotheses"]}
    described = []
    for action in menu:
        kind, _, local = action.partition(" ")
        described.append(f"{action}\n{types[local]}" if kind in {"apply", "cases"} and local in types else action)
    return observations.batch(public, described, history, remaining)


def retain_training_examples(rows):
    """Retain all productive actions and a matched public-state failure sample."""
    productive = Counter(row.module for row in rows if row.disposition <= DISPOSITIONS.index("progress"))
    failures = defaultdict(list)
    kept = []
    for row in rows:
        if row.disposition <= DISPOSITIONS.index("progress"):
            kept.append(row)
        else:
            failures[row.module].append(row)
    for module in sorted(failures):
        ordered = sorted(failures[module], key=lambda row: digest([row.public, row.action, row.prefix]))
        kept.extend(ordered[:2 * max(1, productive[module])])
    return kept


@torch.no_grad()
def batches(observations, items):
    per_action = [typed_batch(observations, item.public, (item.action,),
                              [(action, 1.) for action in item.prefix], CHECKS) for item in items]
    return tuple(torch.cat(column) for column in zip(*per_action, strict=True))


def train(model, observations, data, prepared, *, empirical=False):
    rng = random.Random(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015)
    for _ in range(STEPS):
        indices = rng.choices(range(len(data)), k=BATCH)
        items = [data[index] for index in indices]
        inputs = tuple(column[indices] for column in prepared)
        tokens = model.encode(*inputs)
        transition_model = isinstance(model, SuccessorFeedbackRoute)
        if transition_model:
            logits, prediction = model.successor(tokens, inputs[0])
            delayed = model.continuation(prediction)
        else:
            logits, delayed = model.factors(tokens)
        actual = torch.tensor([row.disposition for row in items])
        targets = torch.tensor([row.empirical if empirical else row.bellman for row in items])
        known = targets != -1
        completion_loss = F.binary_cross_entropy(delayed[:, 0], targets.clamp_min(0).float(), reduction="none")
        loss = F.cross_entropy(logits[:, 0], actual) + (completion_loss * known).sum() / known.sum().clamp_min(1)
        if transition_model:
            observed = [index for index, item in enumerate(items) if item.successor is not None]
            if observed:
                with torch.no_grad():
                    destinations = torch.stack([observations.goal_features(items[index].successor) for index in observed])
                error = F.smooth_l1_loss(prediction[observed, 0], destinations)
                true_state = model.continuation(destinations)
                true_targets = targets[observed]
                true_mask = known[observed]
                value_loss = F.binary_cross_entropy(true_state, true_targets.clamp_min(0).float(), reduction="none")
                loss = loss + error + (value_loss * true_mask).sum() / true_mask.sum().clamp_min(1)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()


@torch.no_grad()
def calibrate(model, data, prepared):
    features = defaultdict(list)
    for offset in range(0, len(data), BATCH):
        inputs = tuple(column[offset:offset + BATCH] for column in prepared)
        embedding = model.features(model.encode(*inputs))
        for name, tensor in calibration_features(inputs, embedding).items():
            features[name].append(tensor[:, 0])
    indices = {name: i for i, name in enumerate(sorted({row.module for row in data}))}
    groups = torch.tensor([indices[row.module] for row in data])
    disposition = torch.tensor([row.disposition for row in data])
    labels = torch.tensor([row.bellman for row in data]).T
    fitted = {}
    for name, shards in features.items():
        matrix = torch.cat(shards)
        immediate = [ridge(matrix, (disposition == i).float(), groups) for i in range(len(DISPOSITIONS))]
        delayed = []
        for horizon in labels:
            known = horizon >= 0
            delayed.append(ridge(matrix[known], horizon[known].float(), groups[known])
                           if groups[known].unique().numel() >= 2 else None)
        groups_of_heads = (immediate, [head for head in delayed if head])
        error = sum(sum(min(head["cv_error"].values()) for head in heads) / max(1, len(heads))
                    for heads in groups_of_heads)
        fitted[name] = (immediate, delayed, error)
    best = min(fitted, key=lambda name: fitted[name][-1])
    return best, fitted[best][:2]


def load_sources(args):
    if sha256_file(args.warm) != WARM_SHA or sha256_file(args.held) != EVAL_REGISTRY_SHA:
        raise ValueError("the source warm checkpoint or comparison bank changed")
    tape, held = read_receipt(args.training), read_receipt(args.held)
    modules, _ = catalogue(args.bpd)
    bench_modules, _ = inventory(args)
    if (tape["schema"] != TRAIN_SCHEMA or tape["held_sha256"] != EVAL_REGISTRY_SHA or tape["modules"] != modules
            or held["modules"] != bench_modules or held["lean_sha256"] != sha256_file(LEAN_EXE)
            or held["repl_sha256"] != sha256_file(REPL_EXE)
            or any(tape[name] != held[name] for name in ("lean_sha256", "repl_sha256"))):
        raise ValueError("the native training and scoring lineage changed")
    statements = {" ".join(row["statement"].split()) for row in held["cases"] + held["unresolved"]}
    goals = {row["visible_sha256"] for row in held["cases"]}
    if any(" ".join(row["statement"].split()) in statements or digest(row["nodes"][0]["public"]) in goals
           for row in tape["rows"]):
        raise ValueError("a scoring statement or initial state entered the native training graph")
    return tape, held


@torch.no_grad()
def score(args, models, frozen, calibration, observations, cases, tree, fit_path):
    intervention_baselines = {"shifted_disposition": "affine_bellman", "immediate_only": "affine_bellman"}
    if "affine_successor" in models:
        intervention_baselines["shifted_successor"] = "affine_successor"
    arms = (*models, "calibrated", *intervention_baselines, "public_behavior")
    results = {}
    proofs = {(case["module"], case["theorem"], tuple(case["reference"])): case for case in cases}
    name, (immediate, conditional) = calibration
    with native_sources(support_paths(args)) as (directory, env, imports):
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected live Lean version")
            for case in cases:
                if case["source"] == "LoopedTransformers Horn":
                    script = iter(case["reference"])

                    def reference(_public, menu, _history, _remaining):
                        action = next(script)
                        if action not in menu:
                            raise ValueError("the checked Horn reference left the public producer menu")
                        return action

                    if not replay(repl, case, case["checks"], reference)["proof"]:
                        raise ValueError("the sealed reference failed under the common scoring menu")
            for arm in arms:
                model = models.get(intervention_baselines.get(arm, arm), frozen)

                def choose(public, menu, history, remaining):
                    if arm == "public_behavior":
                        return behavior_choice(menu)
                    batch = typed_batch(observations, public, menu, history, remaining)
                    state = fixed_clock(batch)
                    tokens = model.encode(*state)
                    if arm == "calibrated":
                        features = calibration_features(state, model.features(tokens))[name]
                        probabilities = torch.stack([apply_ridge(features, head) for head in immediate], dim=-1).clamp_min(1e-6)
                        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
                        values = [apply_ridge(features, head) if head else features.new_zeros(features.shape[:2])
                                  for head in conditional]
                        delayed = torch.stack(values, dim=-1).clamp(0, 1).cummax(-1).values
                    else:
                        if isinstance(model, SuccessorFeedbackRoute):
                            logits, prediction = model.successor(tokens, state[0])
                            if arm == "shifted_successor":
                                prediction = prediction.roll(1, dims=-3)
                            delayed = model.continuation(prediction)
                        else:
                            logits, delayed = model.factors(tokens)
                        probabilities = logits.softmax(dim=-1)
                    if arm == "shifted_disposition":
                        probabilities = probabilities.roll(1, dims=-2)
                    if arm == "immediate_only":
                        delayed = torch.zeros_like(delayed)
                    values = deadline_utility(closure_curve(probabilities, delayed), batch[-1])
                    return menu[int(values[0].argmax())]

                rows = []
                for case in cases:
                    for slack in ((0, 1) if case["checks"] > 1 else (0,)):
                        row = replay(repl, case, case["checks"] + slack, choose)
                        row.update(minimum=case["checks"], slack=slack, cohort=case.get("cohort", "prior"))
                        rows.append(row)
                        if row["proof"]:
                            proofs[case["module"], case["theorem"], tuple(row["script"])] = case
                results[arm] = rows
                print(arm, "closed", sum(row["proof"] for row in rows), len(rows), flush=True)
        independent = verify_independent(directory, env, imports, [(case, key[-1]) for key, case in proofs.items()])
    grouped = {"exact": lambda row: row["minimum"] > 1 and row["slack"] == 0,
               "extra": lambda row: row["minimum"] > 1 and row["slack"] == 1,
               "single": lambda row: row["minimum"] == 1}
    if args.curriculum is not None:
        grouped.update({"fresh_exact": lambda row: row["cohort"] == "held" and row["slack"] == 0,
                        "fresh_extra": lambda row: row["cohort"] == "held" and row["slack"] == 1,
                        "prior_exact": lambda row: row["cohort"] == "prior" and row["minimum"] > 1 and row["slack"] == 0})
    summary = {arm: {name: summarize([row for row in rows if select(row)]) for name, select in grouped.items()}
               for arm, rows in results.items()}
    write_receipt(args.output / "results.json.gz", {"schema": SCHEMA, "fit_sha256": sha256_file(fit_path),
                  "held_sha256": EVAL_REGISTRY_SHA, "results": results, "summary": summary, "independent": independent,
                  "memory": tree.stats()})


def run(args, tree):
    tape, held = load_sources(args)
    if args.curriculum is not None:
        from configs.freeze_ppt_successor_curriculum import SCHEMA as CURRICULUM_SCHEMA

        curriculum = read_receipt(args.curriculum)
        if curriculum["schema"] != CURRICULUM_SCHEMA or curriculum["previous_registry_sha256"] != EVAL_REGISTRY_SHA:
            raise ValueError("the successor curriculum changed its registered ancestry")
        held["cases"] = [*held["cases"], *curriculum["cases"]]
        tape["rows"] = [*tape["rows"], *curriculum["training"]]
        seen = {row["visible_sha256"] for row in held["cases"]}
        statements = {" ".join(row["statement"].split()) for row in held["cases"]}
        if any(" ".join(row["statement"].split()) in statements
               or any(digest(node["public"]) in seen for node in row["nodes"]) for row in tape["rows"]):
            raise ValueError("a fresh scoring state entered source learning")
    all_examples = examples(tape)
    data = retain_training_examples(all_examples) if args.curriculum is not None else all_examples
    if not any(1 in row.bellman for row in data) or not any(1 in row.empirical for row in data):
        raise ValueError("the native source graph needs successful delayed outcomes for both fit objectives")
    print("training", len(data), "of", len(all_examples), "examples",
          dict(Counter(DISPOSITIONS[row.disposition] for row in data)),
          "delayed Bellman positives", sum(1 in row.bellman for row in data),
          "empirical positives", sum(1 in row.empirical for row in data), flush=True)
    args.output.mkdir()
    hybrid, trm, warm_sources = load_warm(args.warm)
    observations = NativeInputs(hybrid.text)
    prepared = batches(observations, data)
    base = NativeFeedbackRoute(hybrid)
    frozen = deepcopy(base).eval()
    trm_direct = NativeFeedbackRoute(trm, trm=True)
    models = {"affine_bellman": base, "affine_empirical": deepcopy(base), "trm_bellman": trm_direct}
    if args.curriculum is not None:
        predicted = SuccessorFeedbackRoute(hybrid)
        trm_predicted = SuccessorFeedbackRoute(trm, trm=True)
        for candidate, direct in ((predicted, base), (trm_predicted, trm_direct)):
            initial = direct.state_dict()
            candidate.load_state_dict({key: initial[key] if key in initial and initial[key].shape == val.shape else val
                                       for key, val in candidate.state_dict().items()})
        models = {"affine_successor": predicted, "trm_successor": trm_predicted,
                  "empirical_successor": deepcopy(predicted), "affine_bellman": base,
                  "trm_bellman": trm_direct}
    for arm, model in models.items():
        train(model, observations, data, prepared, empirical=arm in {"affine_empirical", "empirical_successor"})
        model.eval()
        print("fitted", arm, flush=True)
    calibration = calibrate(frozen, data, prepared)
    state = {arm: model.state_dict() for arm, model in {**models, "frozen": frozen}.items()}
    fit_path = args.output / "fit.pt"
    with fit_path.open("xb") as handle:
        torch.save({"schema": SCHEMA, "weights": state, "calibration": calibration, "warm_sources": warm_sources,
                    "training_sha256": sha256_file(args.training), "held_sha256": EVAL_REGISTRY_SHA,
                    "curriculum_sha256": sha256_file(args.curriculum) if args.curriculum else None,
                    "training_examples": len(data), "optimizer_seed": SEED,
                    "weight_hashes": {arm: tensor_hash(weights) for arm, weights in state.items()}}, handle)
    checkpoint = torch.load(fit_path, weights_only=True, map_location="cpu")
    for arm, weights in checkpoint["weights"].items():
        if tensor_hash(weights) != checkpoint["weight_hashes"][arm]:
            raise ValueError("a sealed native fit changed")
        (frozen if arm == "frozen" else models[arm]).load_state_dict(weights)
    score(args, models, frozen, checkpoint["calibration"], observations, held["cases"], tree, fit_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_sources(parser)
    for name in ("training", "held", "warm", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--curriculum", type=Path)
    args = parser.parse_args()
    configure(1, SEED)
    tree = TreeBudget(2, 1)
    try:
        run(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
