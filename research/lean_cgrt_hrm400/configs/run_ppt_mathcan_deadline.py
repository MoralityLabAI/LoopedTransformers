"""Cross-fit deadline-aware proof search on exact MathCan prefixes."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from configs.run_ppt_mathcan_sparse import Observations, calibration_features, load_warm, play, summarize
from configs.run_ppt_reuse_plans import apply_ridge, ridge
from configs.run_ppt_rollback_diagnostics import write_receipt
from ppt_gym.data import digest, sha256_file
from ppt_gym.hybrid_rl.mathcan import MathCanReplay
from ppt_gym.mathcan_deadline import deadline_utility, supported_distances
from ppt_gym.mathcan_route_policy import RoutePolicy, utility
from ppt_gym.mathcan_sparse import CHECKS, Frame, RecordedSearch
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.mathcan-deadline-pilot.v1"
SEED = 947
ROLLOUTS = 4
UPDATES = 192
MINIBATCH = 24
PER_MODULE = 18
WARM_MODULES = {"MathCan/TOENonAbelian.lean", "MathCan/TOETensorConstraints.lean", "MathCan/TOETripletJordan.lean"}
WARM_SHA = "ca60c3a2e29509a5599ef2d27a5781fa9c579c4558d4751d1757998e87567745"
PRIOR_SHA = "306de25f0c8a4dbc173d3ee5dec9900e97267bafad8ccdbfde5b1ff3d7f609dd"


def key(raw):
    return *raw[:2], tuple(raw[2])


def registry(search, prior):
    seen = set()
    for rows in prior["results"].values():
        for row in rows:
            frame = Frame(key(row["start"]), row["budget"])
            for step in row["steps"]:
                seen.add(frame.key)
                edge = search.step(frame, step["action"])
                if edge["status"] != step["status"]:
                    raise ValueError("the old scoring replay changed")
                frame = edge["after"]
    modules = sorted({state[0] for state in search.archive.states} - WARM_MODULES,
                     key=lambda name: digest([SCHEMA, name]))
    folds = []
    for side in range(2):
        held_modules = set(modules[side::2])
        source = defaultdict(list)
        held = []
        for state in search.archive.states:
            if state[0] not in held_modules:
                if len(search.menu(Frame(state, CHECKS))) > 1:
                    source[state[0]].append(state)
            elif state[2] and state not in seen:
                shortest = search.shortest(state, CHECKS)
                if shortest is not None and 2 <= shortest <= 5:
                    held.append((state, shortest))
        source = [state for module in sorted(source)
                  for state in sorted(source[module], key=lambda item: digest([SCHEMA, item]))[:PER_MODULE]]
        folds.append({"source": source, "held": sorted(held), "held_modules": sorted(held_modules)})
    return folds


@torch.no_grad()
def collect(search, inputs, warm, starts, randomizer):
    counted = Counter()

    def choose(frame):
        batch, (menu,), _ = inputs.batch([frame])
        scores = utility(warm(*batch))[0].tolist()
        attempts = [counted[frame.key, action] for action in menu]
        if randomizer.random() < .5:
            minimum = min(attempts)
            index = randomizer.choice([i for i, hits in enumerate(attempts) if hits == minimum])
        else:
            index = max(range(len(menu)), key=lambda i: scores[i] - .08 * attempts[i])
        action = menu[index]
        counted[frame.key, action] += 1
        return action

    rows = []
    starts = list(starts)
    for _ in range(ROLLOUTS):
        randomizer.shuffle(starts)
        rows.extend(play(search, start, CHECKS, choose) for start in starts)
    return rows


def training_rows(search, episodes, starts, held):
    if not set(starts).isdisjoint(held):
        raise ValueError("a scoring prefix entered source training")
    source_modules = {start[0] for start in starts}
    if source_modules.intersection(state[0] for state in held):
        raise ValueError("a scoring module entered source training")
    counts = Counter(key(row["start"]) for row in episodes)
    if counts != Counter({start: ROLLOUTS for start in starts}):
        raise ValueError("the registered behavior budget changed")
    tape = []
    for row in episodes:
        frame = Frame(key(row["start"]), CHECKS)
        for i, step in enumerate(row["steps"]):
            transition = search.step(frame, step["action"])
            if {name: value for name, value in transition.items() if name != "after"} != {
                    name: value for name, value in step.items() if name != "action"}:
                raise ValueError("the sampled tactic changed its archived verdict")
            status, after = search.archive.step(frame.key, step["action"])
            empirical = len(row["steps"]) - i if row["proof"] else (CHECKS + 1 if not row["censored"] else None)
            # Failure of a shorter remaining-budget rollout says nothing about a longer deadline.
            available = CHECKS if row["proof"] else frame.left
            tape.append((frame, step["action"], status, after, (empirical, available)))
            frame = transition["after"]
        if frame is not None:
            raise ValueError("the sampled rollout ended prematurely")
    distances = supported_distances(tape)
    usable = [sample for sample in tape if sample[2] != "censored"]
    observed = [(before, action, bound, CHECKS) for (before, action, _, _, _), bound in
                zip(usable, distances, strict=True)]
    supervised = [(before, action, empirical, available) for before, action, _, _, (empirical, available) in usable
                  if empirical is not None]
    return observed, supervised


def train(model, inputs, data, salt):
    randomizer = random.Random(SEED + salt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015)
    horizons = torch.arange(1, CHECKS + 1)
    if not data or not any(distance <= CHECKS for _, _, distance, _ in data):
        raise ValueError("the sampled source graph has no usable closure")
    for _ in range(UPDATES):
        items = randomizer.choices(data, k=MINIBATCH)
        batch, menus, _ = inputs.batch([item[0] for item in items])
        actions = torch.tensor([menu.index(item[1]) for menu, item in zip(menus, items, strict=True)])
        curves = model(*batch)[torch.arange(MINIBATCH), actions]
        distances = torch.tensor([item[2] for item in items])
        available = torch.tensor([item[3] for item in items])
        target = horizons >= distances[:, None]
        valid = horizons <= available[:, None]
        loss = F.binary_cross_entropy(curves[valid], target[valid].float())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()


@torch.no_grad()
def calibrate(base, inputs, data):
    values = defaultdict(list)
    roots, distances = [], []
    for offset in range(0, len(data), MINIBATCH):
        items = data[offset:offset + MINIBATCH]
        batch, menus, _ = inputs.batch([item[0] for item in items])
        features = base.read(base.encode(*batch), batch[-1])[1]
        for name, vectors in calibration_features(batch, features).items():
            values[name].extend(vectors[i, menu.index(item[1])] for i, (menu, item) in enumerate(zip(menus, items, strict=True)))
        roots.extend(item[0].key[0] for item in items)
        distances.extend(item[2] for item in items)
    indices = {module: i for i, module in enumerate(sorted(set(roots)))}
    groups = torch.tensor([indices[module] for module in roots])
    bounds = torch.tensor(distances)
    readouts = {name: [ridge(torch.stack(vectors), (bounds <= horizon).float(), groups)
                       for horizon in range(1, CHECKS + 1)] for name, vectors in values.items()}
    selected = min(readouts, key=lambda name: sum(min(record["cv_error"].values()) for record in readouts[name]))
    return selected, readouts


@torch.no_grad()
def evaluate(search, inputs, models, base, calibration, held):
    results = {}
    name, readouts = calibration
    for arm in (*models, "calibrated", "uncapped"):
        model = models["affine_rl"] if arm == "uncapped" else models.get(arm, base)

        def choose(frame):
            batch, (menu,), _ = inputs.batch([frame])
            curve, embedding = model.read(model.encode(*batch), batch[-1])
            if arm == "calibrated":
                obs = calibration_features(batch, embedding)[name]
                curve = torch.stack([apply_ridge(obs, head) for head in readouts[name]], dim=-1).clamp(0, 1).cummax(-1).values
            cutoff = torch.full_like(batch[-1], CHECKS) if arm == "uncapped" else batch[-1]
            values = deadline_utility(curve, cutoff)
            return menu[int(values[0].argmax())]

        results[arm] = [{**play(search, start, minimum + slack, choose), "slack": slack}
                        for start, minimum in held for slack in (0, 1)]
    return results


def run(args, tree):
    if sha256_file(args.warm) != WARM_SHA or sha256_file(args.prior) != PRIOR_SHA:
        raise ValueError("the pilot requires its original warm checkpoint and prior scoring receipt")
    prior = json.loads(gzip.decompress(args.prior.read_bytes()))
    search = RecordedSearch(MathCanReplay(args.archive))
    folds = registry(search, prior)
    args.output.mkdir()
    write_receipt(args.output / "registry.json.gz", {"schema": SCHEMA, "folds": folds, "prior_sha256": PRIOR_SHA,
                                                    "archive_sha256": sha256_file(args.archive), "warm_sha256": WARM_SHA})
    for index, fold in enumerate(folds):
        print("fold", index, "source prefixes", len(fold["source"]), "held prefixes", len(fold["held"]), flush=True)
        hybrid, trm, counts = load_warm(args.warm)
        inputs = Observations(search, hybrid.text)
        warm = RoutePolicy(hybrid).eval()
        episodes = collect(search, inputs, warm, fold["source"], random.Random(SEED + index))
        tape_path = args.output / f"fold-{index}-tape.json.gz"
        write_receipt(tape_path, {"schema": SCHEMA, "episodes": episodes})
        observed, supervised = training_rows(search, episodes, fold["source"], [start for start, _ in fold["held"]])
        print("fold", index, "episodes", len(episodes), "closures", sum(row["proof"] for row in episodes),
              "censored", sum(row["censored"] for row in episodes), "sampled transitions", len(observed),
              "resolved empirical targets", len(supervised), flush=True)
        base = RoutePolicy(hybrid, deadline=True)
        models = {"affine_rl": base, "affine_supervised": deepcopy(base), "trm_rl": RoutePolicy(trm, trm=True, deadline=True)}
        frozen = deepcopy(base).eval()
        for arm, model in models.items():
            train(model, inputs, supervised if arm == "affine_supervised" else observed, index)
            model.eval()
            print("fold", index, "fit", arm, flush=True)
        calibration = calibrate(frozen, inputs, observed)
        fit_path = args.output / f"fold-{index}-fit.pt"
        states = {arm: model.state_dict() for arm, model in {**models, "frozen": frozen}.items()}
        with fit_path.open("xb") as handle:
            torch.save({"schema": SCHEMA, "weights": states, "calibration": calibration, "warm_sources": counts,
                        "weight_hashes": {arm: tensor_hash(state) for arm, state in states.items()},
                        "tape_sha256": sha256_file(tape_path), "registry_sha256": sha256_file(args.output / "registry.json.gz")}, handle)
        checkpoint = torch.load(fit_path, weights_only=True, map_location="cpu")
        for arm, state in checkpoint["weights"].items():
            if tensor_hash(state) != checkpoint["weight_hashes"][arm]:
                raise ValueError("a fitted deadline model changed")
            (frozen if arm == "frozen" else models[arm]).load_state_dict(state)
        results = evaluate(search, inputs, models, frozen, checkpoint["calibration"], fold["held"])
        write_receipt(args.output / f"fold-{index}-replay.json.gz",
                      {"schema": SCHEMA, "results": results, "memory": tree.stats(),
                       "checkpoint_sha256": sha256_file(fit_path), "calibration": calibration[0],
                       "summary": {arm: {str(slack): summarize([row for row in rows if row["slack"] == slack])
                                         for slack in (0, 1)} for arm, rows in results.items()}})


@torch.no_grad()
def clock_diagnostic(args):
    """Hold the model's input clock at its source opening value, with the real deadline readout."""
    def read(name):
        return json.loads(gzip.decompress((args.output / name).read_bytes()))

    registry_path = args.output / "registry.json.gz"
    frozen = read(registry_path.name)
    for name in ("archive", "warm", "prior"):
        if frozen[f"{name}_sha256"] != sha256_file(getattr(args, name)):
            raise ValueError("the clock diagnostic requires the registered artifacts")
    search = RecordedSearch(MathCanReplay(args.archive))
    hybrid, _, _ = load_warm(args.warm)
    inputs = Observations(search, hybrid.text)
    model = RoutePolicy(hybrid, deadline=True).eval()
    records = []
    for i, fold in enumerate(frozen["folds"]):
        prefix = f"fold-{i}-"
        fit_path = args.output / f"{prefix}fit.pt"
        fit = torch.load(fit_path, weights_only=True, map_location="cpu")
        replay = read(f"{prefix}replay.json.gz")
        if (fit["registry_sha256"] != sha256_file(registry_path)
                or replay["checkpoint_sha256"] != sha256_file(fit_path)
                or tensor_hash(fit["weights"]["affine_rl"]) != fit["weight_hashes"]["affine_rl"]):
            raise ValueError("the replay no longer matches its fitted model")
        model.load_state_dict(fit["weights"]["affine_rl"])

        def simulate(start, budget, source_clock):
            def choose(frame):
                batch, (menu,), _ = inputs.batch([frame])
                left = batch[-1]
                policy_input = (*batch[:-1], torch.full_like(left, CHECKS)) if source_clock else batch
                scores = deadline_utility(model(*policy_input), left)
                return menu[int(scores[0].argmax())]
            return play(search, start, budget, choose)

        baseline = replay["results"]["affine_rl"]
        expected = [(key(state), minimum + slack, slack) for state, minimum in fold["held"] for slack in (0, 1)]
        for (start, budget, slack), row in zip(expected, baseline, strict=True):
            actual = simulate(start, budget, False)
            if digest(actual) != digest({name: value for name, value in row.items() if name != "slack"}):
                raise ValueError("the control did not reproduce the frozen replay")
            records.append({"slack": slack, "actual": actual, "source_clock": simulate(start, budget, True)})
    write_receipt(args.output / "clock-diagnostic.json.gz",
                  {"schema": SCHEMA, "registry_sha256": sha256_file(registry_path), "pairs": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for part in ("archive", "warm", "prior", "output"):
        parser.add_argument(part, type=Path)
    parser.add_argument("--clock-diagnostic", action="store_true")
    args = parser.parse_args()
    configure(1, SEED)
    tree = TreeBudget(2, 1)
    try:
        if args.clock_diagnostic:
            clock_diagnostic(args)
        else:
            run(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
