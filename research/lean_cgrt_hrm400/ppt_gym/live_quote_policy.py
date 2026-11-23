"""Two public obligation slots and gated cross-pass messages for live Lean."""

from math import fsum

import torch
from torch import nn

from lcgrt import fragment
from lcgrt.tactic_lm import PostBlock

from .live_quote_meter import grammar, predict_billing, successors
from .private_quotes import QUOTES


WIDTH = 24
FEATURES = 30


def public_agenda(public):
    def convert(formula):
        if "a" in formula:
            return int(formula["a"].removeprefix("P"))
        return tuple(convert(piece) for piece in formula["and"])

    return tuple(convert(goal["target"]) for goal in public["goals"])


def distances(theorem, prices):
    sizes = [1000] * theorem.n_atoms
    for index, atom in enumerate(theorem.facts):
        sizes[atom] = min(sizes[atom], prices.get(f"apply h{index}", 1))
    for _ in range(theorem.n_atoms):
        for index, (premises, head) in enumerate(theorem.rules):
            sizes[head] = min(sizes[head], prices.get(f"apply r{index}", 1) + sum(sizes[atom] for atom in premises))
    return sizes


def effort(goals, sizes):
    return sum(1 + effort(goal, sizes) if isinstance(goal, tuple) else sizes[goal] for goal in goals)


class PublicQuoteFeatures:
    """Shared public features for the candidate, TRM, and frozen calibration."""

    def __init__(self, case, estimated_meter):
        theorem = fragment.Theorem.from_json(case["graph"])
        self.producers = grammar(theorem)
        self.worlds = predict_billing(case, estimated_meter)
        self.distances = [distances(theorem, record["prices"]) for record in self.worlds]
        self.unit = distances(theorem, {})

    def posterior(self, quotes, charges):
        support = [index for index, record in enumerate(self.worlds)
                   if all(record["bits"][quote] == bit for quote, bit in quotes)
                   and all(record["prices"].get(move, 1) == cost for move, cost in charges)]
        return support or list(range(len(self.worlds)))

    def choice_forecast(self, guesses, support, left, ticks):
        eligible = [move for move, after in guesses.items() if effort(after, self.unit) + 1 <= ticks]
        return {move: {world: self.worlds[world]["prices"].get(move, 1) + effort(guesses[move], self.distances[world])
                       for world in support} for move in eligible}

    def quote_forecast(self, forecast, support, left, quote, fee):
        total = 0
        for bit in (0, 1):
            members = [world for world in support if self.worlds[world]["bits"][quote] == bit]
            total += max((0, *(sum(cost[world] <= left - fee for world in members) for cost in forecast.values())))
        return total / len(support)

    def row(self, agenda, action, after, support, view, left, ticks, quotes, warm, forecast):
        if action in QUOTES:
            active, pending = agenda[:1], agenda[1:]
            unit = 0
            step = [view["fees"][action]] * len(support)
        else:
            if action == "constructor":
                children = len(agenda[0])
            else:
                children = len(self.producers[agenda[0]][action])
            active, pending = after[:children], after[children:]
            unit = 1
            step = [self.worlds[i]["prices"].get(action, 1) for i in support]
        slots = []
        for goals in (active, pending):
            costs = [min(32, effort(goals, self.distances[i])) for i in support]
            slots.append((costs, [
                len(goals) / 4, bool(goals), min(effort(goals, self.unit), 16) / 8,
                min(costs) / 16, max(costs) / 16, fsum(costs) / len(costs) / 16,
                sum(cost <= left for cost in costs) / len(costs),
            ]))
        possible = [a + p + charge <= left and effort(active, self.unit) + effort(pending, self.unit) + unit <= ticks
                    for a, p, charge in zip(slots[0][0], slots[1][0], step, strict=True)]
        known = dict(quotes)
        info = sum(self.worlds[i]["bits"][action] for i in support) / len(support) if not unit else 0.
        conditional = self.quote_forecast(forecast, support, left, action, view["fees"][action]) if not unit else 0.
        unconditioned = max((0, *(sum(cost[world] <= left for world in support) for cost in forecast.values()))) / len(support)
        public = [left / 16, ticks / 8, view["budget"] / 16, len(support) / 4,
                  fsum(step) / len(step) / 8, max(step) / 8, min(step) / 8, sum(possible) / len(possible),
                  float(unit), float(action == "constructor"), info, view["fees"].get(action, 0) / 8,
                  conditional, unconditioned, conditional - unconditioned if not unit else 0.,
                  *((known.get(query, -1) + 1) / 2 for query in QUOTES)]
        return tuple(pad_features(item) for item in (slots[0][1] + public[:9], slots[1][1] + public[:9], public + list(warm)))

    def encode(self, public, tactics, view, left, ticks, quotes, charges, warm):
        agenda = public_agenda(public)
        guesses = successors(self.producers, agenda)
        support = self.posterior(quotes, charges)
        forecasts = self.choice_forecast(guesses, support, left, ticks)
        bought = {move for move, _ in quotes}
        menu = (*tactics, *(action for action in QUOTES if action not in bought and view["fees"][action] <= left))
        if not set(tactics) <= guesses.keys():
            raise ValueError("a live tactic disagrees with the public Horn projection")
        rows = [self.row(agenda, action, guesses.get(action, ()), support, view, left, ticks, quotes,
                         warm.get(action, (0.,) * 8), forecasts) for action in menu]
        return menu, torch.tensor(rows, dtype=torch.float32)


def pad_features(values):
    if len(values) > FEATURES:
        raise ValueError("the live feature schema changed")
    return [*values, *([0.] * (FEATURES - len(values)))]


class TwoSlotQuoteActor(nn.Module):
    """A tied three-token actor with gated messages between its two obligation slots."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(FEATURES, WIDTH)
        self.roles = nn.Parameter(torch.randn(3, WIDTH) * .02)
        self.loop = PostBlock(WIDTH, 4, 2)
        self.message = nn.Linear(WIDTH, WIDTH, bias=False)
        self.gate = nn.Linear(2 * WIDTH, WIDTH)
        self.head = nn.Linear(WIDTH, 1)

    def forward(self, features):
        cue = self.embed(features) + self.roles
        return self.read(cue)

    def read(self, cue):
        return self.head(self.representation(cue)).squeeze(-1)

    def representation(self, cue):
        slots = cue
        for _ in range(3):
            opposite = slots[:, [1, 0, 2]]
            update = self.message(opposite) * self.gate(torch.cat((cue, slots), -1)).sigmoid()
            slots = self.loop(cue + slots + update)
        return slots[:, -1]


class LiveQuoteTRM(nn.Module):
    """A tied answer and latent TRM control on the same public input tokens."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(FEATURES, WIDTH)
        self.roles = nn.Parameter(torch.randn(3, WIDTH) * .02)
        self.loop = nn.Sequential(PostBlock(WIDTH, 4, 2), PostBlock(WIDTH, 4, 2))
        self.head = nn.Linear(WIDTH, 1)
        self.register_buffer("answer_init", torch.randn(WIDTH) * .02)
        self.register_buffer("latent_init", torch.randn(WIDTH) * .02)

    def forward(self, features):
        cue = self.embed(features) + self.roles
        answer, latent = self.answer_init.expand_as(cue), self.latent_init.expand_as(cue)
        for _ in range(2):
            latent = self.loop(cue + answer + latent)
        answer = self.loop(answer + latent)
        return self.head(answer[:, -1]).squeeze(-1)
