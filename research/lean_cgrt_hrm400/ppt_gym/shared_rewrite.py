"""Recurrent proof values for joint rewrites and splitting a live Lean guard."""

from dataclasses import dataclass
from functools import lru_cache
import json

import torch
from torch import nn
from torch.nn import functional as F

from .conjunctive_rewrite import nat_equalities, project_rules
from .dependent_rewrite import DependentRewritePolicy, term_problem
from .gated_dependent_rewrite import RoutedDependentRewritePolicy
from .hybrid_rl.graph_model import MAX_RULES, indexed_graph, visible_graph
from .hybrid_rl.model import ProofPolicy
from .rewrite import EQUALITY


MAX_NAMES = 12
MAX_EDGES = 32
DISCOUNT = 0.8


def index(values):
    return torch.tensor(values, dtype=torch.long)


@dataclass(frozen=True)
class LocalTopology:
    names: dict
    rewrites: dict
    reflexive: torch.Tensor
    single_heads: torch.Tensor
    single_next: torch.Tensor
    single_unfold: torch.Tensor
    joint_heads: torch.Tensor
    joint_next: torch.Tensor
    joint_unfold: torch.Tensor
    first: torch.Tensor
    second: torch.Tensor


@lru_cache(maxsize=128)
def local_topology(names, edges):
    size = len(names)
    ids = {name: pos for pos, name in enumerate(names)}
    axis = torch.arange(size)
    endpoint, first, second = torch.meshgrid(axis, axis, axis, indexing="ij")
    flattened = axis[:, None] * size + axis[None, :]
    joint_indices = torch.arange(size ** 3).reshape(size, size, size)
    joint_heads, joint_next = [], []
    single_heads, single_next = [], []
    single_unfold, joint_unfold = [], []
    for action, source, dest in edges:
        a, b = ids[source], ids[dest]
        unfolds = action.startswith("dsimp only [")
        single_heads.append(flattened[a])
        single_next.append(flattened[b])
        single_unfold.append(torch.full((size,), unfolds))
        if unfolds:
            single_heads.append(flattened[:, a])
            single_next.append(flattened[:, b])
            single_unfold.append(torch.ones(size, dtype=torch.bool))
        changed = (endpoint == a) | (first == a) | (second == a)
        joint_heads.append(joint_indices[changed])
        replacement = (torch.where(endpoint == a, b, endpoint) * size ** 2
                       + torch.where(first == a, b, first) * size + torch.where(second == a, b, second))
        joint_next.append(replacement[changed])
        joint_unfold.append(torch.full((joint_heads[-1].numel(),), unfolds))
    return LocalTopology(ids, {action: (ids[src], ids[dst]) for action, src, dst in edges},
                         torch.eye(size).flatten(), torch.cat(single_heads), torch.cat(single_next), torch.cat(single_unfold),
                         torch.cat(joint_heads), torch.cat(joint_next), torch.cat(joint_unfold),
                         (endpoint * size + first).flatten(), (endpoint * size + second).flatten())


@dataclass(frozen=True)
class TermTopology:
    structural: object
    seed: torch.Tensor
    heads: torch.Tensor
    following: torch.Tensor
    guards: torch.Tensor
    actions: torch.Tensor


def term_topology(problem, conjuncts, local, edges, actions):
    left, right, rules = problem
    arguments = list(dict.fromkeys((left[1:], left[1:][::-1])))
    known = {left, right}
    facts = {right}
    structural, transitions, anchoring = [], [], set()
    nat_count = len(local.names)
    for rule in rules:
        for reverse in (False, True):
            head = rule.destination if reverse else rule.source
            for args in arguments:
                current = (head[0], *args)
                tactic, following, guard = rule.apply(current, reverse=reverse)
                known.update((current, following))
                sources = [following]
                pair = 0
                if guard:
                    a, b = guard
                    if rule.name not in conjuncts:
                        return None
                    variable, c = conjuncts[rule.name]
                    if variable != rule.guard[0] or not {a, b, c} <= local.names.keys():
                        return None
                    pair = 1 + local.names[a] * nat_count ** 2 + local.names[b] * nat_count + local.names[c]
                    sources.extend(((a, b), (a, c)))
                    anchoring.update((b, c))
                key = (tactic, current)
                transitions.append((key, current, following, pair))
                structural.append((key, sources, current))
    nodes = {name: pos for pos, name in enumerate(sorted(known))}
    positions = {key: pos + 1 for pos, (key, _, _, _) in enumerate(transitions)}
    active_actions = [(action, left) for action in actions]
    discovered = known | {(name, anchor) for name in local.names for anchor in anchoring}
    for anchor in anchoring:
        facts.add((anchor, anchor))
        for tactic, a, b in edges:
            if not tactic.startswith("rw [← "):
                structural.append(((tactic, anchor), [(b, anchor)], (a, anchor)))
    graph = indexed_graph({left}, set(), discovered, facts, structural, active_actions)
    seed = torch.zeros(len(nodes))
    seed[nodes[right]] = 1.
    return TermTopology(graph, seed, index([nodes[head] for _, head, _, _ in transitions]),
                        index([nodes[following] for _, _, following, _ in transitions]),
                        index([guard for _, _, _, guard in transitions]),
                        index([positions.get(key, 0) for key in active_actions]))


@dataclass(frozen=True)
class View:
    local: LocalTopology
    term: TermTopology | None
    targets: tuple


@lru_cache(maxsize=192)
def observation(packet, actions, definition_edges=()):
    goal = json.loads(packet)["state"]["goals"][0]
    names, edges = nat_equalities(goal)
    edges += definition_edges
    if not 1 <= len(names) <= MAX_NAMES or not 1 <= len(edges) <= MAX_EDGES:
        return None
    local = local_topology(names, edges)
    projected, conjuncts = project_rules(goal)
    term = term_problem(projected)
    if term is not None and conjuncts:
        topology = term_topology(term, conjuncts, local, edges, actions)
        return None if topology is None else View(local, topology, ())
    target = goal["target"]
    formulas = target.get("and", [target])
    matches = [EQUALITY.fullmatch(child.get("raw", "")) for child in formulas]
    if not matches or len(matches) > 2 or any(match is None or not set(match.groups()) <= local.names.keys() for match in matches):
        return None
    targets = tuple(tuple(local.names[name] for name in match.groups()) for match in matches)
    if len(targets) == 2 and targets[0][0] != targets[1][0]:
        return None
    return View(local, None, targets)


class SharedRewriteTransport(nn.Module):
    """Keep a collective state until constructor separates its obligations."""

    def __init__(self):
        super().__init__()
        self.selection_logit = nn.Parameter(torch.zeros(()))
        self.split_log_weight = nn.Parameter(torch.zeros(()))
        self.readout_weight = nn.Parameter(torch.tensor(0.15))

    @staticmethod
    def pool(bank, heads, candidates, power, *, hard, seed=None, prior=1.):
        if hard:
            initial = torch.zeros_like(bank) if seed is None else seed
            return initial.scatter_reduce(0, heads, candidates, reduce="amax", include_self=True)
        weights = candidates * candidates.clamp_min(1e-8).pow(power - 1)
        numerator = torch.zeros_like(bank)
        denominator = torch.zeros_like(bank)
        if seed is not None:
            denominator = prior * seed * seed.clamp_min(1e-8).pow(power - 1)
            numerator = seed * denominator
        numerator = numerator.index_add(0, heads, weights * candidates)
        denominator = denominator.index_add(0, heads, weights)
        return numerator / denominator.clamp_min(1e-12)

    def local_values(self, topology, hops, *, hard=False, factorized=False, normalization=DISCOUNT):
        power = 1 + F.softplus(self.selection_logit)
        split_prior = self.split_log_weight.clamp(-6, 6).exp()
        single = topology.reflexive
        single_discount = torch.where(topology.single_unfold, normalization, DISCOUNT)
        joint_discount = torch.where(topology.joint_unfold, normalization, DISCOUNT)
        for _ in range(hops):
            candidates = single_discount * single[topology.single_next]
            visited = self.pool(single, topology.single_heads, candidates, power, hard=hard)
            single = torch.maximum(single, visited)
        # A reflexive target created by constructor still costs its own rfl call.
        reified = single * (1 - (1 - DISCOUNT) * topology.reflexive)
        split = DISCOUNT * reified[topology.first] * reified[topology.second]
        joint = split
        if not factorized:
            for _ in range(hops):
                candidates = joint_discount * joint[topology.joint_next]
                visited = self.pool(joint, topology.joint_heads, candidates, power, hard=hard, seed=split, prior=split_prior)
                joint = torch.maximum(joint, visited)
        return single, split, joint

    def forward(self, view, actions, hops, *, hard=False, factorized=False, normalization=DISCOUNT, local_values=None):
        topology = view.local
        size = len(topology.names)
        single, split, joint = (self.local_values(topology, hops, hard=hard, factorized=factorized, normalization=normalization)
                                if local_values is None else local_values)
        if view.term is not None:
            term = view.term
            bank = term.seed
            power = 1 + F.softplus(self.selection_logit)
            multiplier = torch.cat((joint.new_ones(1), joint))[term.guards]
            for _ in range(hops):
                candidates = DISCOUNT * bank[term.following] * multiplier
                visited = self.pool(bank, term.heads, candidates, power, hard=hard)
                bank = torch.maximum(bank, visited)
            return torch.cat((bank.new_ones(1), candidates))[term.actions]
        targets = view.targets
        scores = []
        for action in actions:
            if action == "rfl":
                scores.append(single.new_tensor(DISCOUNT))
            elif action == "constructor":
                a, b = targets[0]
                d = targets[1][1]
                scores.append(split[(a * size + b) * size + d])
            else:
                a, b = topology.rewrites[action]
                changed = tuple(tuple(b if atom == a else atom for atom in goal) for goal in targets)
                if len(changed) == 2:
                    head, c = changed[0]
                    d = changed[1][1]
                    continuation = joint[(head * size + c) * size + d]
                else:
                    head, c = changed[0]
                    continuation = single[head * size + c]
                discount = normalization if action.startswith("dsimp only [") else DISCOUNT
                scores.append(discount * continuation)
        return torch.stack(scores)


class SharedRewritePolicy(RoutedDependentRewritePolicy):
    def __init__(self, architecture, width, *, residual_transport, premise_interaction):
        super().__init__(architecture, width, residual_transport=residual_transport, premise_interaction=premise_interaction)
        self.shared_transport = SharedRewriteTransport()
        self.hard_shared = False
        self.factorized = False

    def macs(self, candidates, passes):
        return super().macs(candidates, passes) + 2 * passes * 8 * (MAX_NAMES ** 3 * MAX_EDGES + MAX_RULES)

    def replay_inputs(self, packet, actions, passes):
        view = observation(packet, tuple(actions))
        if view is None:
            raise ValueError("the joint return channel needs a supported public rewrite state")
        graph = view.term.structural if view.term is not None else visible_graph(packet, actions)
        if graph is None:
            base, value = ProofPolicy.forward(self, packet, actions, passes)
            base = self.recurrent_log_gain.exp() * base
        else:
            base, value = self.routed_scores(packet, actions, passes, graph)
        return base, value, view

    def decision_scores(self, base, view, actions, passes):
        values = self.shared_transport(view, actions, 2 * passes, hard=self.hard_shared, factorized=self.factorized)
        scores = values.clamp_min(1e-8).log()
        return scores if self.hard_shared else base + self.shared_transport.readout_weight * scores

    def forward(self, packet, actions, passes, **kwargs):
        if observation(packet, tuple(actions)) is None:
            return super().forward(packet, actions, passes, **kwargs)
        base, value, view = self.replay_inputs(packet, actions, passes)
        return self.decision_scores(base, view, actions, passes), value


class ConjunctiveGraphPolicy(DependentRewritePolicy):
    """Matched graph control with the same public term and equality incidences."""

    def forward(self, packet, actions, passes, **kwargs):
        view = observation(packet, tuple(actions))
        if view is None or view.term is None or view.term.structural is None:
            return super().forward(packet, actions, passes, **kwargs)
        scores, value = ProofPolicy.forward(self, packet, actions, passes, **kwargs)
        evidence, graph_value = self.graph(view.term.structural, passes)
        return scores + evidence, value + graph_value
