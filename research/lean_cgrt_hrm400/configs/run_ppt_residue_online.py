"""Train a persistent Lean policy from sampled native-witnessed rollouts."""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn import functional as F

from configs.run_ppt_premise_budget import MixedSourcePrior
from configs.run_ppt_residue_acquisition import experience, inputs
from configs.run_ppt_residue_rollout import audit, frozen, read_priors, summary
from configs.run_ppt_rollback_diagnostics import write_receipt
from ppt_gym.data import sha256_file
from ppt_gym.failed_prefix import ALLOWANCES, CHARGE
from ppt_gym.live_quote_policy import FEATURES, pad_features
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.residue_recovery import MODULUS
from ppt_gym.residue_rollout import FIRST_QUERY, FIRST_ROUTE, STOP, Recovery, State, greedy, mask
from ppt_gym.residue_rollout_policy import PersistentRecoverySlots, PersistentRecoveryTRM


SCHEMA = "ppt.residue-online.v1"
SEED = 1163
UPDATES = 240
BATCH = 16
TEMPERATURE = .16
ACTIONS = 9
EDITS = ("feedback_swap", "feedback_sham")


def membership(support):
    return [bool(support & (1 << index)) for index in range(MODULUS)]


class Task:
    def __init__(self, pair, native, prior):
        self.pair = pair
        self.target = pair["second"]
        self.outcomes = native["outcomes"]
        self.recovery = Recovery(self.target, native["menu"])
        self.experience = experience(pair, prior)
        choice, observed, _, support, warm, uniform = self.experience
        self.support = mask(index for index, ok in enumerate(support) if ok)
        self.anchor = self.recovery.planner(uniform)
        self.source = pad_features([pair["base"] / 8, pair["depth"] / 64, pair["family"] == "BPD.geom",
                                    observed is not None, bool(observed), *(choice == i for i in range(3)),
                                    *support, *(warm[choice] if choice < 3 else (0.,) * 8)])
        self.costs = [cost / 7 for route in self.target["routes"] for cost in (route["pass_cost"], route["fail_cost"])]

    def start(self, budget):
        return State(budget, self.support, 2)

    def groups(self):
        return (0, *self.recovery.groups, *self.recovery.queries)

    def public(self, state):
        anchor = self.anchor(state)
        target = pad_features([state.remaining / 7, state.queries_left / 2, *self.costs,
                               *((state.support & group).bit_count() / MODULUS for group in self.recovery.groups)])
        rows = []
        for action, group in enumerate(self.groups()):
            info = [action == STOP, FIRST_ROUTE <= action < FIRST_QUERY, action >= FIRST_QUERY,
                    action in anchor, anchor.get(action, 0.), *membership(state.support), *membership(group)]
            rows.append([self.source, target, pad_features(info)])
        return rows, [anchor.get(index, -torch.inf) for index in range(ACTIONS)]

    def transition(self, state, action):
        group = self.groups()[action]
        if action == STOP:
            return .0, None, [0.] * FEATURES
        query = action >= FIRST_QUERY
        bit = self.outcomes[action - FIRST_QUERY] if query else self.target["truth"][action - FIRST_ROUTE]
        charge, proved, following = self.recovery.transition(state, action, bit)
        if following is not None and not following.support:
            raise ValueError("the witnessed successor contradicted the policy state")
        event = pad_features([bit, query, charge / 7, state.remaining / 7,
                              state.queries_left / 2, *membership(group)])
        return float(proved) - CHARGE * charge, following, event


def pack(tasks):
    only_stop = [0., *([-torch.inf] * (ACTIONS - 1))]
    rows = [([[[0.] * FEATURES for _ in range(3)] for _ in range(ACTIONS)], only_stop,
             [0] * ACTIONS, [0.] * ACTIONS, [[0.] * FEATURES for _ in range(ACTIONS)], [0.] * ACTIONS)]
    starts = []
    for task in tasks:
        seen = {}

        def explore(state):
            if state is None:
                return 0
            if state in seen:
                return seen[state]
            index = len(rows)
            seen[state] = index
            rows.append(None)
            features, anchor = task.public(state)
            following, rewards, oracle = [0] * ACTIONS, [0.] * ACTIONS, [0.] * ACTIONS
            events = [[0.] * FEATURES for _ in range(ACTIONS)]
            for action in task.recovery.allowed(state):
                reward, child, event = task.transition(state, action)
                next_index = explore(child)
                following[action], rewards[action], events[action] = next_index, reward, event
                oracle[action] = reward + max(rows[next_index][-1])
            rows[index] = features, anchor, following, rewards, events, oracle
            return index

        starts.extend(explore(task.start(budget)) for budget in ALLOWANCES)
    return torch.tensor(starts), tuple(torch.tensor(column) for column in zip(*rows, strict=True))


def models():
    actor = PersistentRecoverySlots()
    return {"two_slot_online": actor, "two_slot_value": deepcopy(actor), "trm_online": PersistentRecoveryTRM()}


def train(model, name, starts, packed, schedule, draws):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    features, anchors, transitions, rewards, events, oracle = packed
    batch_ids = torch.arange(BATCH)
    for batch, sample in zip(schedule, draws, strict=True):
        state = starts[batch]
        memory = model.initialize(features[state, 0])
        path, errors = [], []
        for uniform in sample:
            public, base = features[state], anchors[state]
            scores = model.score(memory, public) + base
            log_policy = (scores / TEMPERATURE).log_softmax(-1)
            with torch.no_grad():
                cdf = log_policy.exp().cumsum(-1).clamp(max=1.)
                action = (cdf <= uniform[:, None]).sum(-1).clamp(max=ACTIONS - 1)
                if not torch.isfinite(base[batch_ids, action]).all():
                    raise ValueError("the rollout sampler selected a masked action")
            if name == "two_slot_value":
                legal = base.isfinite() & (state[:, None] != 0)
                errors.append(F.mse_loss(scores[legal], oracle[state][legal], reduction="sum"))
            else:
                path.append((log_policy[batch_ids, action], rewards[state, action], base.max(-1).values))
            selected_event = events[state, action]
            state = transitions[state, action]
            if not state.any():
                break
            memory = model.observe(memory, selected_event)
        if state.any():
            raise ValueError("a sampled episode exceeded its tactic horizon")
        if name == "two_slot_value":
            loss = torch.stack(errors).sum() / BATCH
        else:
            future = torch.zeros(BATCH)
            terms = []
            for log_probability, reward, baseline in reversed(path):
                future = reward + future
                terms.append(-log_probability * (future - baseline))
            loss = torch.stack(terms).sum() / BATCH
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval()


def fit(args, tree):
    pairs, pins = inputs(args, "source")
    prior = MixedSourcePrior(args)
    starts, rows = pack([Task(pair, native, prior) for pair, native in pairs])
    generator = torch.Generator().manual_seed(SEED)
    schedule = torch.randint(len(starts), (UPDATES, BATCH), generator=generator)
    draws = torch.rand((UPDATES, max(ALLOWANCES) + 1, BATCH), generator=generator).clamp(max=1 - 1e-6)
    readers = models()
    initial = {name: tensor_hash(model.state_dict()) for name, model in readers.items()}
    if initial["two_slot_online"] != initial["two_slot_value"]:
        raise ValueError("the matched reward and regression arms changed initialization")
    for name, model in readers.items():
        train(model, name, starts, rows, schedule, draws)
        print("fitted sampled policy", name, flush=True)
    weights = {name: model.state_dict() for name, model in readers.items()}
    checkpoint = {"schema": SCHEMA, "pins": pins, "weights": weights, "initial": initial,
                  "weight_sha": {name: tensor_hash(state) for name, state in weights.items()},
                  "sampler_sha": tensor_hash({"schedule": schedule, "draws": draws}),
                  "source_states": len(rows[0]) - 1, "memory": tree.stats()}
    with args.fit.open("xb") as handle:
        torch.save(checkpoint, handle)
    print("sealed sampled fit", sha256_file(args.fit), flush=True)


def perturb(model, old, memory, event, arm, sign):
    flipped = event.clone()
    flipped[:, 0] = 1. - flipped[:, 0]
    vector = model.observe(old, flipped)[0, 0] - memory[0, 0]
    norm = vector.norm()
    if not norm.item():
        return memory, None
    if arm == EDITS[1]:
        unit = vector / norm
        orthogonal = torch.zeros_like(vector)
        orthogonal[int(unit.abs().argmin())] = 1.
        orthogonal = orthogonal - (orthogonal @ unit) * unit
        vector = sign * norm * F.normalize(orthogonal, dim=0)
    altered = memory.clone()
    altered[0, 0] += vector
    if not torch.isclose((altered - memory).norm(), norm, rtol=1e-5, atol=1e-7):
        raise ValueError("the feedback intervention did not preserve its physical dose")
    return altered, float(norm)


def learned(task, budget, model, arm):
    start = task.start(budget)
    initial = model.initialize(torch.tensor(task.public(start)[0])[None, 0])
    memory, patch, last_state = initial, None, start

    def decide(state, history):
        nonlocal memory, patch, last_state
        if history:
            action = history[-1]["action"]
            _, _, event = task.transition(last_state, action)
            event = torch.tensor([event])
            updated = model.observe(memory, event)
            if arm == "slot_reset":
                updated = initial
            if arm in EDITS and patch is None and action >= FIRST_QUERY and len(task.recovery.allowed(state)) > 1:
                sign = 1 if int(task.pair["pair"][-1], 16) % 2 else -1
                updated, dose = perturb(model, memory, updated, event, arm, sign)
                if dose is not None:
                    patch = {"step": len(history), "action": action, "dose": dose}
            memory = updated
        last_state = state
        rows, anchor = task.public(state)
        values = model.score(memory, torch.tensor([rows]))[0] + torch.tensor(anchor)
        return greedy({action: values[action].item() for action in task.recovery.allowed(state)})

    row = task.recovery.rollout(start, task.target["truth"], task.outcomes, decide)
    return {**row, "patch": patch}


@torch.no_grad()
def evaluate(args, tree):
    pairs, pins = inputs(args, "development")
    checkpoint = torch.load(args.fit, map_location="cpu", weights_only=True)
    if checkpoint["schema"] != SCHEMA or checkpoint["pins"] != pins:
        raise ValueError("the sampled policy used different pinned inputs")
    readers = models()
    for name, model in readers.items():
        weights = checkpoint["weights"][name]
        if tensor_hash(weights) != checkpoint["weight_sha"][name]:
            raise ValueError("the sampled policy weights changed")
        model.load_state_dict(weights)
        model.eval()
    prior, calibrators = MixedSourcePrior(args), frozen(args, pins)
    online = readers["two_slot_online"]
    readers.update({name: online for name in ("slot_reset", *EDITS)})
    records = defaultdict(list)
    for pair, native in pairs:
        task = Task(pair, native, prior)
        priors = read_priors(pair, task.experience, prior, calibrators)
        planners = {name: task.recovery.planner(weights) for name, weights in priors.items()}
        for budget in ALLOWANCES:
            tags = {"pair": pair["pair"], "expression": pair["certificate"].rsplit(" = ", 1)[0]}
            trials = {name: learned(task, budget, model, name) for name, model in readers.items()}
            if trials[EDITS[0]]["patch"] != trials[EDITS[1]]["patch"]:
                raise ValueError("feedback edit eligibility or dose changed between intervention arms")
            for name, planner in planners.items():
                trials[name] = task.recovery.rollout(task.start(budget), task.target["truth"], task.outcomes,
                                                     lambda state, _: greedy(planner(state)))
            for name, row in trials.items():
                records[name].append({**row, **tags})
    summaries = {name: summary(rows) for name, rows in records.items()}
    for name, data in summaries.items():
        print(name, data, flush=True)
    independent, replays = audit(args, pairs, records)
    write_receipt(args.results, {"schema": SCHEMA, "pins": pins, "fit_sha256": sha256_file(args.fit),
                                 "summary": summaries, "records": records, "independent": independent,
                                 "live_replays": replays, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "eval"))
    for name in ("bank", "bpd", "mathcan", "warm", "prior", "native", "calibration", "fit", "results"):
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
