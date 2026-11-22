"""Two-slot proposal scoring with a recurrent cache decision."""

from collections import defaultdict, deque
from copy import deepcopy
from math import inf
import re

import torch
from torch import nn

from .data import canonical
from .dependent_witness import FORK, action_projection, horn_graph
from .witness_reuse import CACHE


CACHE_TYPE = re.compile(rf"have {CACHE} : ([A-Za-z]+ [0-9]+) := \?_")
CHANNELS = 6


def plan_features(goal, target, cache_menu, checks):
    """Describe each obligation and its relationship to the proposed cache."""
    _, facts, edges, active = horn_graph(goal, [target])
    rules = defaultdict(list)
    for head, left, right in edges:
        rules[head].append((left, right))
    root, = active
    if "and" not in target:
        (root, partner), = rules[root]
        if partner:
            raise ValueError("the reuse proposal expected an explicit conjunction")
    (lhs, rhs), = rules[root]
    if not rhs:
        raise ValueError("the reuse menu requires two distinct proof obligations")

    def distances(start):
        distance = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for child, sibling in rules.get(current, ()):
                if not sibling and child not in distance:
                    distance[child] = distance[current] + 1
                    queue.append(child)
        return distance

    goal_distances = [distances(lhs), distances(rhs)]

    def fact_distance(distance):
        return min((distance[fact] + 1 for fact in facts if fact in distance), default=inf)

    if cache_menu[0] != "constructor":
        raise ValueError("the direct plan must be available")
    cache_targets = [None]
    for action in cache_menu[1:]:
        match = CACHE_TYPE.fullmatch(action)
        if not match:
            raise ValueError("the cache proposal must name a public ground atom")
        cache_targets.append(canonical({"raw": match[1]}))
    features = []
    for cached in cache_targets:
        build = fact_distance(distances(cached)) if cached is not None else 0
        slots = []
        for distance in goal_distances:
            to_fact = fact_distance(distance)
            reachable = cached in distance
            if build == inf or to_fact == inf:
                raise ValueError("the registered direct and cache facts must be provable")
            slots.append([checks / 20., float(cached is not None), float(reachable), to_fact / 8.,
                          distance[cached] / 8. if reachable else 0., build / 8.])
        features.append(slots)
    return features


def world_features(row):
    """Project candidate witnesses before the fork and verify the later observation."""
    output = []
    for action, cache in zip(FORK, row["cache_states"], strict=True):
        closed, target = action_projection(row["fork"], action)
        if closed is not None:
            raise ValueError("the witness projection must close the first obligation")
        planned = plan_features(row["fork"][1], target, cache["menu"], row["checks"])
        goal, = cache["goals"]
        observed = plan_features(goal, goal["target"], cache["menu"], row["checks"])
        if planned != observed:
            raise ValueError("the witness projection did not match the native cache state")
        output.append(planned)
    return torch.tensor(output, dtype=torch.float32)


class ReusePolicy(nn.Module):
    """Carry both goal tokens of the selected witness into its cache choice."""

    def __init__(self, warm, *, trm=False):
        super().__init__()
        self.trm = trm
        self.input = nn.Linear(CHANNELS, warm.width)
        self.roles = nn.Parameter(warm.roles.detach().clone())
        self.cell = deepcopy(warm.trm if trm else warm.cell)
        self.preview = deepcopy(warm.actor)
        self.actor = deepcopy(warm.actor)
        self.cache_actor = deepcopy(warm.actor)
        if trm:
            self.register_buffer("initial_answer", warm.answer_init.clone())
            self.register_buffer("initial_latent", warm.latent_init.clone())

    def loop(self, cues):
        flat = cues.flatten(0, -3)
        if self.trm:
            answer, latent = self.initial_answer, self.initial_latent
            for _ in range(2):
                latent = self.cell(flat + answer + latent)
            return self.cell(answer + latent).reshape_as(cues)
        return self.cell(self.cell(flat)).reshape_as(cues)

    def encode(self, features, *, reset=False, donor=None):
        cues = self.input(features) + self.roles
        previews = self.loop(cues)
        pooling = self.preview(previews.mean(-2)).softmax(dim=-2)
        memory = (pooling.unsqueeze(-1) * previews).sum(dim=-3)
        context = memory.mean(-2)
        if reset or self.trm:
            carry = 0.
        else:
            carry = (memory if donor is None else donor).unsqueeze(-3)
        plans = self.loop(cues + carry).mean(-2)
        return context, plans, memory

    def forward(self, features, *, reset=False, donor=None):
        context, plans, memory = self.encode(features, reset=reset, donor=donor)
        return self.actor(context).squeeze(-1), self.cache_actor(plans).squeeze(-1), memory
