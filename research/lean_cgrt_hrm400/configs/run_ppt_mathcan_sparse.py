"""Learn from sampled MathCan transitions and evaluate by held-out module."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import gzip
import json
from math import fsum
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from ppt_gym.data import digest, sha256_file
from ppt_gym.hybrid_rl.mathcan import MathCanReplay
from ppt_gym.mathcan_route_policy import RoutePolicy, utility
from ppt_gym.mathcan_sparse import CHECKS, CHECK_COST, SCHEMA, Frame, RecordedSearch, outcome, split_starts
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.two_slot import TwoSlotPolicy, TwoSlotTRM, text_chunks

from configs.pretrain_ppt_slot_mixed import SCHEMA as WARM_SCHEMA
from configs.run_ppt_reuse_plans import apply_ridge, ridge
from configs.run_ppt_rollback_diagnostics import write_receipt


SEED = 831
ROLLOUTS = 6
UPDATES = 300
MINIBATCH = 24


@dataclass(frozen=True)
class Transition:
    before: Frame
    action: str
    after: Frame | None
    status: str
    terminal_proof: bool
    terminal_checks: int | None
    episode_proof: bool


def load_warm(path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if checkpoint["schema"] != WARM_SCHEMA or not checkpoint["counts"]:
        raise ValueError("expected the audited four-source mixed warm checkpoint")
    hybrid, trm = TwoSlotPolicy(bindings=True), TwoSlotTRM(bindings=True)
    for key, model in (("hybrid", hybrid), ("trm", trm)):
        state = checkpoint["weights"][key]
        if tensor_hash(state) != checkpoint["weight_hashes"][key]:
            raise ValueError("the source warm parameters changed")
        model.load_state_dict(state)
    return hybrid.eval(), trm.eval(), checkpoint["counts"]


class Observations:
    def __init__(self, search, encoder):
        self.search = search
        self.encoder = encoder
        self.action_vectors = {}

    @lru_cache(maxsize=None)
    def goals(self, key):
        state = self.search.archive.observations[key]
        active, *pending = state["goals"] or [""]
        texts = (active, "\n".join([*pending, *map(str, state["definitions"])]))
        with torch.no_grad():
            return torch.stack([self.encoder(text_chunks(text, self.encoder.max_bytes) or [""]).mean(0)
                                for text in texts])

    def action_text(self, actions):
        missing = list(dict.fromkeys(name for name in actions if name not in self.action_vectors))
        if missing:
            with torch.no_grad():
                self.action_vectors.update(zip(missing, self.encoder(missing), strict=True))
        return torch.stack([self.action_vectors[name] for name in actions])

    def batch(self, frames):
        menus = [self.search.menu(frame) for frame in frames]
        size = max(map(len, menus))
        goals = torch.stack([self.goals(frame.key) for frame in frames])
        actions = goals.new_zeros((len(frames), size, goals.shape[-1]))
        past = goals.new_zeros((len(frames), CHECKS, goals.shape[-1]))
        feedback = goals.new_zeros((len(frames), CHECKS))
        legal = torch.zeros((len(frames), size), dtype=torch.bool)
        for i, (frame, menu) in enumerate(zip(frames, menus, strict=True)):
            actions[i, :len(menu)] = self.action_text(menu)
            legal[i, :len(menu)] = True
            if frame.history:
                past[i, :len(frame.history)] = self.action_text([action for _, action, _ in frame.history])
                feedback[i, :len(frame.history)] = goals.new_tensor(
                    [-1. if status == "rejected" else 1. for _, _, status in frame.history])
        return (goals, actions, past, feedback, goals.new_tensor([frame.left for frame in frames])), menus, legal


def play(search, start, budget, select):
    frame = Frame(start, budget)
    steps = []
    while frame is not None:
        action = select(frame)
        transition = search.step(frame, action)
        steps.append({"action": action, **{k: v for k, v in transition.items() if k != "after"}})
        frame = transition["after"]
    return {"start": start, "budget": budget, "steps": steps, **outcome(steps)}


def all_models(hybrid, trm):
    base = RoutePolicy(hybrid)
    return {"affine_rl": deepcopy(base), "return_regression": deepcopy(base), "imitation": deepcopy(base),
            "warm": base, "trm_rl": RoutePolicy(trm, trm=True)}


def collect(args, search):
    sides = split_starts(search.archive)
    hybrid, trm, sources = load_warm(args.warm)
    observation = Observations(search, hybrid.text)
    base = all_models(hybrid, trm)["warm"].eval()
    counts, rejections = Counter(), set()
    randomizer = random.Random(SEED)

    def choose(frame):
        data, (menu,), _ = observation.batch([frame])
        # Prefer unexplored actions without inspecting their recorded verdicts.
        with torch.no_grad():
            score = utility(base(*data))[0].tolist()
        visits = [counts[frame.key, action] for action in menu]
        estimates = [value - .04 * hits - .35 * ((frame.key, action) in rejections)
                     for action, value, hits in zip(menu, score, visits, strict=True)]
        if randomizer.random() < .35:
            best = min(visits)
            chosen = randomizer.choice([i for i, n in enumerate(visits) if n == best])
        else:
            chosen = max(range(len(menu)), key=estimates.__getitem__)
        counts[frame.key, menu[chosen]] += 1
        return menu[chosen]

    recorded = []
    starts = list(sides["train"])
    for _ in range(ROLLOUTS):
        randomizer.shuffle(starts)
        for start in starts:
            episode = play(search, start, CHECKS, choose)
            frame = Frame(start, CHECKS)
            for step in episode["steps"]:
                if step["status"] == "rejected":
                    rejections.add((frame.key, step["action"]))
                frame = search.step(frame, step["action"])["after"]
            recorded.append(episode)
    print("train episodes", len(recorded), "proofs", sum(ep["proof"] for ep in recorded),
          "censored", sum(ep["censored"] for ep in recorded), flush=True)
    write_receipt(args.output, {"schema": SCHEMA, "stage": "collect", "seed": SEED, "rollouts": ROLLOUTS,
                                "archive_sha256": sha256_file(args.archive), "warm_sha256": sha256_file(args.warm),
                                "sources": sources, "splits": sides, "episodes": recorded})


def read_experience(args, search):
    record = json.loads(gzip.decompress(args.experience.read_bytes()))
    splits = split_starts(search.archive)
    if (record["schema"] != SCHEMA or record["stage"] != "collect"
            or record["seed"] != SEED or record["rollouts"] != ROLLOUTS
            or record["archive_sha256"] != sha256_file(args.archive)
            or record["warm_sha256"] != sha256_file(args.warm)
            or digest(record["splits"]) != digest(splits)):
        raise ValueError("recorded training requires the original archive, weights and module split")
    starts = set(splits["train"])
    counts = Counter()
    tapes = []
    for episode in record["episodes"]:
        module, theorem, prefix = episode["start"]
        start = module, theorem, tuple(prefix)
        steps = episode["steps"]
        if start not in starts or episode["budget"] != CHECKS or not 1 <= len(steps) <= CHECKS:
            raise ValueError("the sparse feedback tape violated its registered episode budget")
        recorded_outcome = outcome(steps)
        if recorded_outcome != {key: episode[key] for key in recorded_outcome}:
            raise ValueError("the sparse feedback tape exposed an invalid source or outcome")
        counts[start] += 1
        frame = Frame(start, CHECKS)
        for i, step in enumerate(steps):
            if frame is None:
                raise ValueError("a feedback tape continued past an archive stopping point")
            result = search.step(frame, step["action"])
            if {k: v for k, v in result.items() if k != "after"} != {k: v for k, v in step.items() if k != "action"}:
                raise ValueError("a feedback tape transition differs from its audited archive")
            proof = step["status"] == "archived_closure"
            if step["status"] != "censored":
                checks = None if episode["censored"] else len(steps) - i
                tapes.append(Transition(frame, step["action"], result["after"], step["status"],
                                        proof, checks, episode["proof"]))
            frame = result["after"]
        if frame is not None:
            raise ValueError("a feedback tape ended before its registered stopping point")
    if counts != Counter({start: ROLLOUTS for start in starts}):
        raise ValueError("the sparse feedback tape changed the number of rollouts per training theorem")
    return tapes


def observed_support(search, tape):
    by_prefix = defaultdict(set)
    for step in tape:
        by_prefix[step.before.key].add(step.action)
    return {frame: by_prefix[frame.key].intersection(search.menu(frame))
            for frame in {step.after for step in tape if step.after is not None}}


def calibration_features(batch, embedding):
    goals, actions, past, feedback, remaining = batch
    shape = actions.shape[:2]
    history = (past * feedback.unsqueeze(-1)).sum(1)
    shared = torch.cat((goals.flatten(1), history, remaining[:, None] / CHECKS), dim=-1)
    raw = torch.cat((actions, shared[:, None].expand(*shape, -1)), dim=-1)
    return {"frozen_slot": embedding, "raw": raw}


def optimize(model, observations, tapes, objective):
    supports = observed_support(observations.search, tapes) if objective == "rl" else {}
    data = [step for step in tapes if step.after is None or supports[step.after]] if objective == "rl" else [
        step for step in tapes if step.terminal_checks is not None]
    if objective == "imitation":
        best = {}
        for step in data:
            if step.episode_proof and step.status != "rejected":
                prior = best.get(step.before)
                if prior is None or step.terminal_checks < prior.terminal_checks:
                    best[step.before] = step
        data = list(best.values())
    if not data:
        raise ValueError(f"the source tape contains no eligible transitions for {objective}")
    randomizer = random.Random(SEED)
    target = deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015)
    for iteration in range(UPDATES):
        steps = randomizer.choices(data, k=MINIBATCH)
        features, menus, legal = observations.batch([step.before for step in steps])
        indices = torch.tensor([menu.index(step.action) for menu, step in zip(menus, steps, strict=True)])
        predictions = model(*features)
        selected = predictions[torch.arange(len(steps)), indices]
        if objective == "imitation":
            loss = F.cross_entropy(utility(predictions).masked_fill(~legal, -torch.inf), indices)
        else:
            if objective == "rl":
                correct = selected.new_tensor([float(step.terminal_proof) for step in steps])
                checks = torch.ones_like(correct)
                continued = [i for i, step in enumerate(steps) if step.after is not None]
                if continued:
                    frames = [steps[i].after for i in continued]
                    inputs, next_actions, _ = observations.batch(frames)
                    with torch.no_grad():
                        next_factors = target(*inputs)
                        support = torch.zeros(next_factors.shape[:2], dtype=torch.bool)
                        for i, (frame, menu) in enumerate(zip(frames, next_actions, strict=True)):
                            for choice in supports[frame]:
                                support[i, menu.index(choice)] = True
                        best = utility(next_factors).masked_fill(~support, -torch.inf).argmax(-1)
                        future = next_factors[torch.arange(len(best)), best]
                        correct[continued] = future[:, 0]
                        checks[continued] += future[:, 1]
            else:
                correct = selected.new_tensor([float(step.episode_proof) for step in steps])
                checks = selected.new_tensor([step.terminal_checks for step in steps])
            loss = F.mse_loss(selected[:, 0], correct) + F.mse_loss(selected[:, 1] / CHECKS, checks / CHECKS)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (iteration + 1) % 20 == 0 and objective == "rl":
            target.load_state_dict(model.state_dict())
    return len(data)


def fit(args, search):
    tape = read_experience(args, search)
    hybrid, trm, counts = load_warm(args.warm)
    observations = Observations(search, hybrid.text)
    models = all_models(hybrid, trm)
    sizes = {}
    for arm, objective in (("affine_rl", "rl"), ("return_regression", "return"),
                           ("imitation", "imitation"), ("trm_rl", "rl")):
        sizes[arm] = optimize(models[arm], observations, tape, objective)
        print(arm, "observed training transitions", sizes[arm], flush=True)
    values, labels, roots = defaultdict(list), [], []
    usable = [step for step in tape if step.terminal_checks is not None]
    with torch.no_grad():
        for start in range(0, len(usable), MINIBATCH):
            steps = usable[start:start + MINIBATCH]
            batch, menus, _ = observations.batch([step.before for step in steps])
            _, features = models["warm"].read(models["warm"].encode(*batch), batch[-1])
            for name, candidates in calibration_features(batch, features).items():
                values[name].extend(candidates[i, menu.index(step.action)]
                                    for i, (menu, step) in enumerate(zip(menus, steps, strict=True)))
            labels.extend(float(step.episode_proof) - CHECK_COST * step.terminal_checks for step in steps)
            roots.extend(step.before.key[0] for step in steps)
    indices = {root: i for i, root in enumerate(sorted(set(roots)))}
    groups = torch.tensor([indices[root] for root in roots])
    targets = torch.tensor(labels)
    readouts = {name: ridge(torch.stack(rows), targets, groups) for name, rows in values.items()}
    calibrated = min(readouts, key=lambda name: min(readouts[name]["cv_error"].values()))
    print("frozen calibration", calibrated, flush=True)
    weights = {arm: model.state_dict() for arm, model in models.items()}
    with args.output.open("xb") as out:
        torch.save({"schema": SCHEMA, "stage": "fit", "archive_sha256": sha256_file(args.archive),
                    "warm_sha256": sha256_file(args.warm), "experience_sha256": sha256_file(args.experience),
                    "sources": counts, "steps": UPDATES, "sizes": sizes, "weights": weights,
                    "calibrated": calibrated, "readouts": readouts,
                    "weight_hashes": {arm: tensor_hash(state) for arm, state in weights.items()}}, out)


def restore_fit(args):
    fit = torch.load(args.fit, weights_only=True, map_location="cpu")
    if (fit["schema"] != SCHEMA or fit["stage"] != "fit"
            or any(fit[f"{key}_sha256"] != sha256_file(getattr(args, key)) for key in ("archive", "warm", "experience"))):
        raise ValueError("the acquired policy changed its recorded feedback lineage")
    hybrid, trm, _ = load_warm(args.warm)
    models = all_models(hybrid, trm)
    for arm, weights in fit["weights"].items():
        if tensor_hash(weights) != fit["weight_hashes"][arm]:
            raise ValueError("a registered sparse policy changed")
        models[arm].load_state_dict(weights)
        models[arm].eval()
    return fit, models, hybrid.text


def summarize(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["start"][0]].append(row)
    rates = {}
    for module, entries in sorted(buckets.items()):
        rates[module] = {
            "n": len(entries),
            "closed": sum(row["proof"] for row in entries),
            "censored": sum(row["censored"] for row in entries),
            "checks": sum(row["checks"] for row in entries),
            "pessimistic_return": fsum(row["observed_return"] if not row["censored"]
                                       else -CHECK_COST * row["budget"] for row in entries) / len(entries),
            "optimistic_return": fsum(row["observed_return"] if not row["censored"]
                                      else 1 - CHECK_COST * row["checks"] for row in entries) / len(entries),
        }
    macro = {name: fsum(row[name] / row["n"] for row in rates.values()) / len(rates)
             for name in ("closed", "censored", "checks")}
    macro.update({name: fsum(row[name] for row in rates.values()) / len(rates)
                  for name in ("pessimistic_return", "optimistic_return")})
    return {"module_macro": macro, "modules": rates}


def evaluate(args, search, tree):
    fit, models, encoder = restore_fit(args)
    observations = Observations(search, encoder)
    models["calibrated"] = models["warm"]
    starts = split_starts(search.archive)["dev"]
    results = {}
    with torch.no_grad():
        for arm, model in models.items():
            def choose(frame):
                batch, (menu,), _ = observations.batch([frame])
                outcomes, embedding = model.read(model.encode(*batch), batch[-1])
                values = utility(outcomes)
                if arm == "calibrated":
                    name = fit["calibrated"]
                    values = apply_ridge(calibration_features(batch, embedding)[name], fit["readouts"][name])
                return menu[int(values[0].argmax())]
            rows = [play(search, start, budget, choose) for budget in (4, CHECKS) for start in starts]
            results[arm] = rows
            print(arm, "proofs", sum(ep["proof"] for ep in rows),
                  "censored", sum(ep["censored"] for ep in rows), flush=True)
    reference = [{"start": start, "budget": budget, "minimum_recorded_checks": search.shortest(start, budget)}
                 for budget in (4, CHECKS) for start in starts]
    write_receipt(args.output, {"schema": SCHEMA, "stage": "evaluate", "fit_sha256": sha256_file(args.fit),
                                "archive_sha256": sha256_file(args.archive), "experience_sha256": sha256_file(args.experience),
                                "available_reference": reference, "results": results, "memory": tree.stats(),
                                "summary": {arm: {str(budget): summarize([row for row in rows if row["budget"] == budget])
                                                  for budget in (4, CHECKS)} for arm, rows in results.items()}})


@torch.no_grad()
def bpd(args, search):
    fit, models, encoder = restore_fit(args)
    evaluation = json.loads(gzip.decompress(args.evaluation.read_bytes()))
    if (evaluation["schema"] != SCHEMA or evaluation["stage"] != "evaluate"
            or evaluation["fit_sha256"] != sha256_file(args.fit)
            or evaluation["archive_sha256"] != fit["archive_sha256"]):
        raise ValueError("BPD requires the frozen checkpoint and matching replay")
    model = models["affine_rl"]
    observations = Observations(search, encoder)
    pairs = []
    for clean in evaluation["results"]["affine_rl"]:
        if clean["budget"] != CHECKS:
            continue
        module, theorem, prefix = clean["start"]
        start = module, theorem, tuple(prefix)

        def simulate(arm):
            patch = {}

            def choose(frame):
                inputs, (menu,), _ = observations.batch([frame])
                tokens = model.encode(*inputs)
                original = int(utility(model.read(tokens, inputs[-1])[0])[0].argmax())
                if frame.history and not patch:
                    goals, actions, past, feedback, left = inputs
                    blank = model.encode(goals, actions, torch.zeros_like(past), torch.zeros_like(feedback), left)
                    delta = tokens[:, :, 1] - blank[:, :, 1]
                    if arm == "reset":
                        tokens = blank
                    elif arm == "sham":
                        tokens[:, :, 1] -= delta.roll(1, dims=-1)
                    selected = int(utility(model.read(tokens, left)[0])[0].argmax())
                    patch.update(depth=len(frame.history), kind=frame.history[-1][2], norm=float(delta[0, 0].norm()),
                                 clean=menu[original], selected=menu[selected],
                                 clean_status=search.step(frame, menu[original])["status"],
                                 selected_status=search.step(frame, menu[selected])["status"])
                    return menu[selected]
                return menu[original]

            return play(search, start, CHECKS, choose), patch

        replay, site = simulate("clean")
        if digest(replay) != digest(clean):
            raise ValueError("BPD replay differs from the frozen affine policy")
        if site:
            reset, reset_site = simulate("reset")
            sham, sham_site = simulate("sham")
            if not reset_site or not sham_site or any(reset_site[key] != sham_site[key]
                    for key in ("depth", "kind", "norm", "clean")):
                raise ValueError("BPD patches failed to share a breakpoint and perturbation norm")
            pairs.append({"clean": replay, "reset": reset, "sham": sham, "reset_site": reset_site, "sham_site": sham_site})
    write_receipt(args.output, {"schema": SCHEMA, "stage": "bpd", "fit_sha256": evaluation["fit_sha256"],
                                "evaluation_sha256": sha256_file(args.evaluation), "pairs": pairs})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    collect_parser, fit_parser, eval_parser, bpd_parser = (
        commands.add_parser(name) for name in ("collect", "fit", "evaluate", "bpd"))
    for command in (collect_parser, fit_parser, eval_parser, bpd_parser):
        for file in ("archive", "warm", "output"):
            command.add_argument(file, type=Path)
    for command in (fit_parser, eval_parser, bpd_parser):
        command.add_argument("--experience", type=Path, required=True)
    for command in (eval_parser, bpd_parser):
        command.add_argument("--fit", type=Path, required=True)
    bpd_parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    configure(1, SEED)
    tree = TreeBudget(2, 1)
    try:
        search = RecordedSearch(MathCanReplay(args.archive))
        if args.stage == "collect":
            collect(args, search)
        elif args.stage == "fit":
            fit(args, search)
        elif args.stage == "evaluate":
            evaluate(args, search, tree)
        else:
            bpd(args, search)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
