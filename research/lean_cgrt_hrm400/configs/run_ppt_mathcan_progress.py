"""Cross-fit disposition-factored proof search on registered MathCan tapes."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from configs.run_ppt_mathcan_deadline import MINIBATCH, UPDATES, key, sampled_tape
from configs.run_ppt_mathcan_sparse import Observations, calibration_features, load_warm, play, summarize
from configs.run_ppt_reuse_plans import apply_ridge, ridge
from configs.run_ppt_rollback_diagnostics import write_receipt
from ppt_gym.data import digest, sha256_file
from ppt_gym.hybrid_rl.mathcan import MathCanReplay
from ppt_gym.mathcan_deadline import deadline_utility, supported_distances
from ppt_gym.mathcan_progress import ProgressRoute, progress_curve
from ppt_gym.mathcan_sparse import CHECKS, Frame, RecordedSearch
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.mathcan-progress-pilot.v1"
SEED = 971
PINS = {
    "registry.json.gz": "8c4cd061fb7dcdd50a6748b6f25dc2e435f0fdf0d36ad3ae84e4f9fd55bfd6c9",
    "summary.json.gz": "f05bea94b498c39cc928fb5340fbf6377d83c59b244007ec1b7099be40165f6d",
    "clock-diagnostic.json.gz": "adadabbe7d3ac253b0a2e13971723ef02961b4f9592089add1f491fd011b0873",
}
SPARSE_BPD_SHA = "cbbb813aba88f0fa57e96eb27066615994991bd2222923c242cd9b22343354e9"


@dataclass(frozen=True)
class Sample:
    frame: Frame
    action: str
    accepted: bool
    backed_up: tuple[int, int]
    empirical: tuple[int, int]


def factor_targets(tape):
    distances = iter(supported_distances(tape))
    rows = []
    for frame, action, status, _, (empirical, observed) in tape:
        bound = next(distances) if status != "censored" else None
        accepted = status != "rejected"
        # Censored archive edges are accepted tactics with unobserved continuations.
        backed_up = (bound, CHECKS) if accepted and bound is not None else (CHECKS + 1, 0)
        empirical = (empirical, observed) if accepted and bound is not None and empirical is not None else (CHECKS + 1, 0)
        rows.append(Sample(frame, action, accepted, backed_up, empirical))
    return rows


def fixed_clock(batch):
    return (*batch[:-1], torch.full_like(batch[-1], CHECKS))


def train(model, inputs, rows, fold, *, empirical=False):
    rng = random.Random(SEED + fold)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015)
    horizons = torch.arange(1, CHECKS + 1)
    for _ in range(UPDATES):
        minibatch = rng.choices(rows, k=MINIBATCH)
        batch, menus, _ = inputs.batch([item.frame for item in minibatch])
        indices = torch.tensor([menu.index(item.action) for item, menu in zip(minibatch, menus, strict=True)])
        logits, curves = model.factors(model.encode(*fixed_clock(batch)))
        index = (torch.arange(MINIBATCH), indices)
        bounds, known = torch.tensor([item.empirical if empirical else item.backed_up for item in minibatch]).unbind(-1)
        targets = (bounds.unsqueeze(-1) <= horizons).float()
        mask = known.unsqueeze(-1) >= horizons
        acceptance = logits.new_tensor([item.accepted for item in minibatch])
        loss = F.binary_cross_entropy_with_logits(logits[index], acceptance)
        curve_loss = F.binary_cross_entropy(curves[index], targets, reduction="none")
        loss = loss + (curve_loss * mask).sum() / mask.sum().clamp_min(1)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()


@torch.no_grad()
def calibrate(model, inputs, rows):
    values = defaultdict(list)
    for offset in range(0, len(rows), MINIBATCH):
        items = rows[offset:offset + MINIBATCH]
        batch, menus, _ = inputs.batch([item.frame for item in items])
        batch = fixed_clock(batch)
        for name, vectors in calibration_features(batch, model.features(model.encode(*batch))).items():
            values[name].extend(vectors[i, menu.index(item.action)] for i, (item, menu) in enumerate(zip(items, menus, strict=True)))
    modules = {module: i for i, module in enumerate(sorted({item.frame.key[0] for item in rows}))}
    groups = torch.tensor([modules[item.frame.key[0]] for item in rows])
    acceptance = torch.tensor([item.accepted for item in rows]).float()
    bounds, known = torch.tensor([item.backed_up for item in rows]).unbind(-1)
    valid = known != 0
    fitted = {}
    for name, features in values.items():
        features = torch.stack(features)
        accept = ridge(features, acceptance, groups)
        conditional = [ridge(features[valid], (bounds[valid] <= horizon).float(), groups[valid])
                       for horizon in range(1, CHECKS + 1)]
        fitted[name] = {"accept": accept, "conditional": conditional}
    # Give acceptance and the full conditional curve equal weight in source selection.
    selected = min(fitted, key=lambda name: min(fitted[name]["accept"]["cv_error"].values()) +
                   sum(min(head["cv_error"].values()) for head in fitted[name]["conditional"]) / CHECKS)
    return selected, fitted[selected]


def consumed_states(search, previous):
    seen = set()
    for row in previous:
        frame = Frame(key(row["start"]), row["budget"])
        for step in row["steps"]:
            seen.add(frame.key)
            transition = search.step(frame, step["action"])
            if transition["status"] != step["status"]:
                raise ValueError("a previously scored trajectory changed")
            frame = transition["after"]
    return seen


def extra_prefixes(search, modules, seen):
    by_module = defaultdict(list)
    for state in search.archive.states:
        if state[2] and state[0] in modules and state not in seen and search.shortest(state, 1) == 1:
            by_module[state[0]].append(state)
    return [state for module in sorted(by_module)
            for state in sorted(by_module[module], key=lambda item: digest([SCHEMA, item]))[:3]]


@torch.no_grad()
def evaluate(search, inputs, models, frozen, calibration, cases):
    results = {}
    name, readouts = calibration
    switches = ("accept_only", "no_retry", "shuffled_acceptance")
    for arm in (*models, "frozen_progress", *switches):
        model = models["affine_bellman"] if arm in switches else models.get(arm, frozen)

        def choose(frame):
            batch, (menu,), _ = inputs.batch([frame])
            observation = fixed_clock(batch)
            tokens = model.encode(*observation)
            if arm == "frozen_progress":
                features = calibration_features(observation, model.features(tokens))[name]
                acceptance = apply_ridge(features, readouts["accept"]).clamp(0, 1)
                conditional = torch.stack([apply_ridge(features, head) for head in readouts["conditional"]],
                                          dim=-1).clamp(0, 1).cummax(-1).values
            else:
                logits, conditional = model.factors(tokens)
                acceptance = logits.sigmoid()
            if arm == "shuffled_acceptance":
                acceptance = acceptance.roll(1, dims=-1)
            if arm == "accept_only":
                selected = acceptance
            else:
                curve = progress_curve(acceptance, conditional, retry=arm != "no_retry")
                selected = deadline_utility(curve, batch[-1])
            return menu[int(selected[0].argmax())]

        results[arm] = [{**play(search, state, budget, choose), "condition": condition}
                        for state, budget, condition in cases]
    return results


def read_receipt(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def run(args, tree):
    previous = args.previous
    for filename, sha in PINS.items():
        if sha256_file(previous / filename) != sha:
            raise ValueError(f"previous cross-fit receipt changed: {filename}")
    original = read_receipt(previous / "registry.json.gz")
    summary = read_receipt(previous / "summary.json.gz")
    for name in ("archive", "warm", "prior"):
        if sha256_file(getattr(args, name)) != original[f"{name}_sha256"]:
            raise ValueError(f"previous source changed: {name}")
    search = RecordedSearch(MathCanReplay(args.archive))
    prior = read_receipt(args.prior)
    old = [row for rows in prior["results"].values() for row in rows]
    if sha256_file(args.sparse_bpd) != SPARSE_BPD_SHA:
        raise ValueError("the sparse breakpoint receipt changed")
    old.extend(pair[arm] for pair in read_receipt(args.sparse_bpd)["pairs"] for arm in ("clean", "reset", "sham"))
    old.extend(pair["source_clock"] for pair in read_receipt(previous / "clock-diagnostic.json.gz")["pairs"])
    for index, stats in enumerate(summary["folds"]):
        tape = previous / f"fold-{index}-tape.json.gz"
        replay = previous / f"fold-{index}-replay.json.gz"
        if sha256_file(tape) != stats["tape"] or sha256_file(replay) != stats["replay"]:
            raise ValueError("previous fold receipt changed")
        old.extend(row for rows in read_receipt(replay)["results"].values() for row in rows)
    seen = consumed_states(search, old)
    cases = []
    for fold in original["folds"]:
        multi = [(key(state), minimum + slack, f"multi_{slack}") for state, minimum in fold["held"] for slack in (0, 1)]
        single = [(state, 1, "single") for state in extra_prefixes(search, set(fold["held_modules"]), seen)]
        cases.append(multi + single)
    args.output.mkdir()
    registry = args.output / "registry.json.gz"
    write_receipt(registry, {"schema": SCHEMA, "prior_pins": PINS, "sparse_bpd_sha256": SPARSE_BPD_SHA, "folds": cases})
    for index, (fold, fold_cases) in enumerate(zip(original["folds"], cases, strict=True)):
        source = list(map(key, fold["source"]))
        episode_path = previous / f"fold-{index}-tape.json.gz"
        tape = sampled_tape(search, read_receipt(episode_path)["episodes"], source, [state for state, _, _ in fold_cases])
        rows = factor_targets(tape)
        hybrid, trm, counts = load_warm(args.warm)
        inputs = Observations(search, hybrid.text)
        base = ProgressRoute(hybrid)
        frozen = deepcopy(base).eval()
        models = {"affine_bellman": base, "affine_empirical": deepcopy(base), "trm_bellman": ProgressRoute(trm, trm=True)}
        print("fold", index, "cases", dict(Counter(condition for _, _, condition in fold_cases)),
              "source", len(rows), "accepted", sum(item.accepted for item in rows),
              "accept-only censored", sum(status == "censored" for _, _, status, _, _ in tape), flush=True)
        for arm, model in models.items():
            train(model, inputs, rows, index, empirical=arm == "affine_empirical")
            model.eval()
            print("fold", index, "fit", arm, flush=True)
        calibration = calibrate(frozen, inputs, rows)
        fit_path = args.output / f"fold-{index}-fit.pt"
        weights = {arm: model.state_dict() for arm, model in {**models, "frozen": frozen}.items()}
        with fit_path.open("xb") as handle:
            torch.save({"schema": SCHEMA, "weights": weights, "calibration": calibration, "warm_sources": counts,
                        "weight_hashes": {arm: tensor_hash(state) for arm, state in weights.items()},
                        "tape_sha256": sha256_file(episode_path), "registry_sha256": sha256_file(registry)}, handle)
        restored = torch.load(fit_path, weights_only=True, map_location="cpu")
        for arm, state in restored["weights"].items():
            if tensor_hash(state) != restored["weight_hashes"][arm]:
                raise ValueError("a fitted progress checkpoint changed")
            (frozen if arm == "frozen" else models[arm]).load_state_dict(state)
        results = evaluate(search, inputs, models, frozen, restored["calibration"], fold_cases)
        summary_rows = {arm: {condition: summarize([row for row in items if row["condition"] == condition])
                             for condition in sorted({row["condition"] for row in items})}
                        for arm, items in results.items()}
        write_receipt(args.output / f"fold-{index}-replay.json.gz",
                      {"schema": SCHEMA, "checkpoint_sha256": sha256_file(fit_path), "results": results,
                       "calibration": calibration[0], "summary": summary_rows, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for part in ("archive", "warm", "prior", "sparse_bpd", "previous", "output"):
        parser.add_argument(part, type=Path)
    args = parser.parse_args()
    configure(1, SEED)
    tree = TreeBudget(2, 1)
    try:
        run(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
