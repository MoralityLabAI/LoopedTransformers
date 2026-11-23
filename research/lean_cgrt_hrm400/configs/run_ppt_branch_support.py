"""Fit budgeted support policies and replay selected Lean branch choices."""

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from itertools import product
from math import fsum
from pathlib import Path

import torch
from torch.nn import functional as F

from configs.freeze_ppt_branch_support import SCHEMA as SUPPORT_SCHEMA, witness
from configs.freeze_ppt_scope_recovery import PREFIX_MODULE, native_scope
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_rollback_diagnostics import write_receipt
from configs.run_ppt_scope_recovery import checked_bank, improve, replay
from lcgrt.config import LEAN_EXE, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.branch_support import ACTIONS, BRANCHES, DEVELOPMENT, FEE, PROBES, SOURCE
from ppt_gym.branch_support import State, Support, legal, play
from ppt_gym.branch_support_policy import raw_features, tokens
from ppt_gym.data import digest, sha256_file
from ppt_gym.live_lean import public_state
from ppt_gym.live_quote_policy import LiveQuoteTRM, TwoSlotQuoteActor
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.scope_recovery import EARLY, OPEN as SCOPE_OPEN, RESOLVE, rollout


SCHEMA = "ppt.branch-support.v1"
SEED = 1187
UPDATES = 144
BATCH = 64
TEMPERATURE = .10
FLOOR = .025
CALIBRATORS = ("quadratic", "frozen_slots", "decoder")


def banks(args, split):
    scoped, prelude, scoped_sha = checked_bank(args, split)
    support = read_receipt(args.support)
    if (support["schema"] != SUPPORT_SCHEMA or support["lean"] != sha256_file(LEAN_EXE)
            or support["repl"] != sha256_file(REPL_EXE) or support["prelude"] != digest(prelude)):
        raise ValueError("the support witnesses changed their native pins")
    return scoped, support, prelude, {"bank": scoped_sha, "support": sha256_file(args.support)}


def query_results(rows):
    by_root = defaultdict(lambda: defaultdict(list))
    for row in rows:
        left, right, _, kept = (event["value"] for event in row["observations"])
        witness = left ^ right
        actions = iter((*EARLY, SCOPE_OPEN, RESOLVE, witness))
        proof = rollout(row, 2, lambda _: next(actions))
        by_root[row["payload_root"]][kept].append((row, proof))
    for outcomes in by_root.values():
        for kept, recipes in outcomes.items():
            if len(recipes) != 8 or sum(proof["proved"] for _, proof in recipes) != (4 if kept else 8):
                raise ValueError("the native branch return disagrees with the shared witness decoder")
    return by_root


def pack(environment, decks):
    rows, numbers = [], {}

    def visit(state):
        if state is None:
            return 0
        if state in numbers:
            return numbers[state]
        rewards, edges, probs = [0.] * ACTIONS, [[0, 0] for _ in range(ACTIONS)], [[0., 0.] for _ in range(ACTIONS)]
        available = legal(state)
        for action in available:
            rewards[action], children = environment.transition(state, action)
            for branch, (weight, next_state) in enumerate(children):
                edges[action][branch], probs[action][branch] = visit(next_state), weight
        height = 1 + max((rows[i - 1][-1] for pair in edges for i in pair if i), default=0)
        exact = environment.exact(state)
        rows.append((tokens(state), [action in available for action in range(ACTIONS)], rewards, edges, probs,
                     [exact.get(action, 0.) for action in range(ACTIONS)], height))
        numbers[state] = len(rows)
        return len(rows)

    for deck, limit in product(decks, (1, 2)):
        visit(State(deck, limit))
    return tuple(torch.tensor(column) for column in zip(*rows, strict=True)), numbers


def models():
    actor = TwoSlotQuoteActor()
    return {"support_reward": actor, "support_imitation": deepcopy(actor),
            "trm_reward": LiveQuoteTRM(), "frozen_encoder": deepcopy(actor)}


def scores(model, features, mask):
    return features.new_full(mask.shape, -torch.inf).masked_scatter(mask, model(features[mask]))


def train(model, name, graph, batches):
    features, mask, rewards, edges, probs, optimal, height = graph
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    target = optimal
    for iteration, batch in enumerate(batches):
        if name != "support_imitation" and iteration % 8 == 0:
            with torch.no_grad():
                target = improve(scores(model, features, mask), rewards, edges, probs, height)
        support = mask[batch]
        prediction = scores(model, features[batch], support)
        teacher = (target[batch].masked_fill(~support, -torch.inf) / TEMPERATURE).softmax(-1)
        log_policy = (prediction / TEMPERATURE).log_softmax(-1).masked_fill(~support, 0.)
        loss = -(teacher * log_policy).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval()


def calibration_features(name, encoder, data):
    base = raw_features(data)
    if name == "decoder":
        return base
    if name == "frozen_slots":
        cue = encoder.embed(data) + encoder.roles
        return torch.cat((base, cue.flatten(-2), encoder.read(cue)[..., None]), dim=-1)
    context = data[..., 2, :9]
    pair = torch.triu_indices(context.shape[-1], context.shape[-1])
    return torch.cat((base, context[..., pair[0]] * context[..., pair[1]]), dim=-1)


@torch.no_grad()
def ridge(model, features, mask, optimal):
    weights = {}
    for name in CALIBRATORS:
        selected = mask.clone()
        if name == "decoder":
            selected[:, :BRANCHES[0]] = False
        x = calibration_features(name, model, features[selected]).double()
        y = optimal[selected].double()
        system = x.T @ x + .01 * torch.eye(x.shape[1], dtype=x.dtype)
        weights[name] = torch.linalg.solve(system, x.T @ y)
    return weights


def fit(args, tree):
    scoped, bank, _, pins = banks(args, "train")
    query_results(scoped)
    graph, index = pack(Support(bank["worlds"]), SOURCE)
    readers = models()
    anchors = {name: tensor_hash(model.state_dict()) for name, model in readers.items()}
    calibration = ridge(readers["frozen_encoder"], graph[0], graph[1], graph[-2])
    batches = torch.randint(len(index), (UPDATES, BATCH), generator=torch.Generator().manual_seed(SEED))
    for name in ("support_reward", "support_imitation", "trm_reward"):
        train(readers[name], name, graph, batches)
        print("fitted support", name, flush=True)
    weights = {name: model.state_dict() for name, model in readers.items()}
    with args.fit.open("xb") as handle:
        torch.save({"schema": SCHEMA, "pins": pins, "initial": anchors, "source_states": len(index),
                    "weights": weights, "weight_sha256": {name: tensor_hash(state) for name, state in weights.items()},
                    "calibration": calibration, "calibration_sha256": tensor_hash(calibration),
                    "sampler_sha256": tensor_hash({"batches": batches}), "memory": tree.stats()}, handle)
    print("sealed support fit", sha256_file(args.fit), flush=True)


def choose(values):
    return max(values, key=lambda action: round(float(values[action]), 9))


def planned_decoder(environment, index, predictions):
    @lru_cache(maxsize=None)
    def q(state):
        if state.key:
            return {action: float(predictions[index[state] - 1, action].clamp(.5, 1.)) for action in BRANCHES}
        values = {}
        for action in legal(state):
            reward, branches = environment.transition(state, action)
            values[action] = reward + sum(prob * max(q(child).values()) for prob, child in branches)
        return values

    return q


def branch_patch(model, bank):
    rows = []
    for i, world in enumerate(bank["worlds"]):
        bits = tuple(world["observations"][key - 1]["value"] if key in (3, 4) else -1 for key in PROBES)
        state = State((1, 7), 0, bits, 7)
        donor_bits = tuple(1 - bit if key == 3 else bit for key, bit in zip(PROBES, bits, strict=True))
        alternate = replace(state, observed=donor_bits)
        history = torch.tensor(tokens(state))[list(BRANCHES)]
        reference = torch.tensor(tokens(alternate))[list(BRANCHES)]
        natural = model.embed(history) + model.roles
        donor = model.embed(reference) + model.roles
        delta = donor[:, 1] - natural[:, 1]
        dose = delta.norm()
        unit = F.normalize(delta.flatten(), dim=0)
        axis = torch.zeros_like(unit)
        axis[int(unit.abs().argmin())] = -1. if i % 2 else 1.
        sham = (F.normalize(axis - (axis @ unit) * unit, dim=0) * dose).reshape_as(delta)
        output = {}
        for name, shift in (("natural", torch.zeros_like(delta)), ("donor", delta), ("sham", sham)):
            cue = natural.clone()
            cue[:, 1] += shift
            if name != "natural" and not torch.isclose((cue - natural).norm(), dose, atol=1e-6):
                raise ValueError("branch slot intervention dose changed")
            output[name] = int(model.read(cue).argmax())
        rows.append({"world": i, "query": 7, "probes": [(key, bit) for key, bit in zip(PROBES, bits, strict=True)
                                                       if bit != -1],
                     "dose": float(dose), "truth": world["observations"][6]["value"], **output})
    return rows


def audit(prelude, bank, outcomes, records, bpd):
    def cases():
        for arm, rows in records.items():
            yield arm, max(rows, key=lambda row: (len(row["probes"]), row["query"] == 7))
        for arm in ("natural", "donor", "sham"):
            for patch in bpd[:2]:
                branch = patch[arm]
                yield "bpd_" + arm, {**patch, "branch": branch, "kept": branch ^ patch["truth"]}

    def verify(directory, env):
        receipts = []
        with LeanRepl(imports=(PREFIX_MODULE,), env=env, cwd=directory) as repl:
            for arm, choice in cases():
                world = bank["worlds"][choice["world"]]
                repl.reset()
                state = repl.init(world["source"])
                if public_state(repl, state) != bank["public"]:
                    raise ValueError("the support transcript changed its visible root")
                for key, bit in choice["probes"]:
                    state, receipt = witness(repl, state, key)
                    if receipt != world["observations"][key - 1] or receipt["value"] != bit:
                        raise ValueError("a chosen support action changed in native Lean")
                expected = world["observations"][choice["query"] - 1]["value"] ^ choice["branch"]
                if expected != choice["kept"]:
                    raise ValueError("the branch maps to a different verified scope")
                paths = next(iter(outcomes.values()))[expected]
                row, proof = min(paths, key=lambda pair: pair[1]["proved"])
                replay(repl, row, proof)
                receipts.append({"arm": arm, "tid": row["tid"], "world": choice["world"],
                                 "kept": expected, "proof": proof, "support": choice["probes"]})
        return receipts

    return native_scope(prelude, verify)


@torch.no_grad()
def evaluate(args, tree):
    scoped, bank, prelude, pins = banks(args, "dev")
    fit = torch.load(args.fit, map_location="cpu", weights_only=True)
    if (fit["schema"] != SCHEMA or fit["pins"] != pins
            or fit["calibration_sha256"] != tensor_hash(fit["calibration"])):
        raise ValueError("the support fit or frozen calibration changed")
    readers = models()
    for name, reader in readers.items():
        weights = fit["weights"][name]
        if tensor_hash(weights) != fit["weight_sha256"][name]:
            raise ValueError("a frozen support model changed")
        reader.load_state_dict(weights)
        reader.eval()
    tested = query_results(scoped)
    environment = Support(bank["worlds"])
    graph, index = pack(environment, DEVELOPMENT)
    features, mask = graph[:2]
    frozen = readers.pop("frozen_encoder")
    forecasts = {name: scores(reader, features, mask) for name, reader in readers.items()}
    for name, weights in fit["calibration"].items():
        predictions = calibration_features(name, frozen, features[mask]).double() @ weights
        forecasts[name] = predictions.new_full(mask.shape, -torch.inf).masked_scatter(mask, predictions)
    decoder = planned_decoder(environment, index, forecasts.pop("decoder"))

    def q(values, state):
        logits = values[index[state] - 1]
        return {action: logits[action] for action in legal(state)}

    arms = {name: (lambda state, value=values: q(value, state)) for name, values in forecasts.items()}
    arms.update({"decoder_planner": decoder, "native_exact": environment.exact})
    if args.stage_swaps:
        reward = arms["support_reward"]
        logits = forecasts["support_reward"][:, BRANCHES[0]:]
        relative = torch.where(mask[:, BRANCHES[0]:], logits, 0.)
        centered = .75 + relative - relative.mean(dim=1, keepdim=True)
        neural_return = torch.zeros_like(forecasts["support_reward"])
        neural_return[:, BRANCHES[0]:] = centered
        arms.update({
            "reward_support_calibrated_branch": lambda state: decoder(state) if state.key else reward(state),
            "calibrated_support_reward_branch": lambda state: reward(state) if state.key else decoder(state),
            "reward_decoder_planner": planned_decoder(environment, index, neural_return),
        })
    records = defaultdict(list)
    for name, values in arms.items():
        for i, world in enumerate(bank["worlds"]):
            for deck, limit in product(DEVELOPMENT, (1, 2)):
                for key in deck:
                    choice = play(world, State(deck, limit), key, lambda state: choose(values(state)))
                    kept = world["observations"][key - 1]["value"] ^ choice["branch"]
                    records[name].append({"world": i, "deck": deck, "limit": limit, "kept": kept, **choice})
    summaries = {}
    for name, rows in records.items():
        support_count = sum(len(row["probes"]) for row in rows)
        native_proofs = sum(proof["proved"] for row in rows for root in tested.values() for _, proof in root[row["kept"]])
        checks = sum(proof["checks"] for row in rows for root in tested.values() for _, proof in root[row["kept"]])
        n = len(rows) * len(tested) * 8
        summaries[name] = {"proofs": native_proofs, "count": n, "support_checks": support_count,
                           "total_checks": checks + support_count * len(tested) * 8,
                           "utility": native_proofs / n - FEE * support_count / len(rows)}
        print(name, summaries[name], flush=True)
    lesions = branch_patch(readers["support_reward"], bank)
    verified = audit(prelude, bank, tested, records, lesions)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit),
                                 "stage_swaps": args.stage_swaps,
                                 "summary": summaries, "records": records, "bpd": lesions, "replays": verified,
                                 "native_payloads": sorted(tested), "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "support", "mathcan", "fit", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--stage-swaps", action="store_true", help="score post-hoc fixed-weight stage interventions")
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        configure(1, SEED)
        (fit if args.phase == "fit" else evaluate)(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
