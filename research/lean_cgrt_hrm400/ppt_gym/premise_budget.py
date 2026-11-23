"""Recurrent proof-effort distributions over typed public Horn premises."""

from dataclasses import dataclass

import torch
from torch import nn

from .data import canonical
from .hybrid_rl.graph_model import GRAPH_WIDTH, MAX_PREMISES, Graph, visible_graph, GraphEvidence
from .mathcan_sparse import CHECKS


ROUNDS = 4
TRM_PASSES = 2


@dataclass(frozen=True)
class BudgetState:
    graph: Graph
    projected: tuple
    prior: torch.Tensor
    rule_prices: tuple = ()
    action_prices: tuple = ()


def atoms(formula):
    if "a" in formula:
        return [formula["a"]]
    if "and" in formula:
        return [item for subgoal in formula["and"] for item in atoms(subgoal)]
    raise ValueError("the premise budget requires an atomic or conjunctive public goal")


def shift_budget(curve, prices, *, start):
    horizons = torch.arange(start, CHECKS + 1, device=curve.device)
    offsets = horizons - prices[:, None]
    return curve.gather(-1, offsets.clamp_min(0)) * (offsets >= 0)


def budget_state(public, menu, prior, *, prices=None):
    graph = visible_graph(canonical({"state": public}), menu)
    if graph is None:
        raise ValueError("the public state is outside the registered Horn budget grammar")
    positions = {atom: index + 1 for index, atom in enumerate(graph.atoms)}
    target, *pending = [goal["target"] for goal in public["goals"]]
    shared = [positions[name] for formula in pending for name in atoms(formula)]
    projected = []
    for action, index in zip(menu, graph.actions, strict=True):
        if action == "constructor":
            if "and" not in target:
                raise ValueError("a nonconjunctive target offered a split")
            projected.append(tuple([positions[name] for name in atoms(target)] + shared))
        elif index:
            projected.append(tuple([atom for atom in graph.premises[index - 1] if atom] + shared))
        else:
            raise ValueError("a public Horn producer lost its premises")
    if prices is None:
        return BudgetState(graph, tuple(projected), prior)
    rule_prices = tuple(prices[f"apply {name}"] for name in graph.rule_names)
    action_prices = tuple(prices[action] for action in menu)
    aligned = shift_budget(torch.nn.functional.pad(prior, (1, 0)), torch.tensor(action_prices, device=prior.device) - 1, start=1)
    return BudgetState(graph, tuple(projected), aligned, rule_prices, action_prices)


class BudgetRecurrence(nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf = nn.Linear(4, 1)
        self.rule = nn.Linear(5, 1)
        self.mix = nn.Parameter(torch.tensor(-2.5))
        indices = torch.arange(CHECKS + 1)
        differences = indices[:, None] - indices
        self.register_buffer("offsets", differences.clamp_min(0))
        self.register_buffer("valid", differences >= 0)
        with torch.no_grad():
            self.leaf.weight.zero_()
            self.leaf.weight[0, 0] = 2.5
            self.leaf.bias.fill_(-4.)
            self.rule.weight.zero_()
            self.rule.bias.fill_(1.5)

    def convolve(self, left, right):
        return (left.unsqueeze(-2) * right[..., self.offsets] * self.valid).sum(dim=-1)

    def joint(self, table, options):
        result = table.new_zeros((*options.shape[:-1], CHECKS + 1))
        result[..., 0] = 1.
        for slot in options.unbind(-1):
            result = self.convolve(result, table[slot])
        return result

    def forward(self, state, *, foreign_budget=False):
        graph = state.graph
        features = self.leaf.weight.new_tensor(graph.features)
        heads = torch.tensor(graph.heads, dtype=torch.long, device=features.device)
        premises = torch.tensor(graph.premises, dtype=torch.long, device=features.device)
        degree = torch.bincount(heads, minlength=len(features)).to(features).clamp_max(8) / 8
        arity = (premises > 0).sum(-1)
        rule_prices = torch.tensor(state.rule_prices or (1,) * len(heads), dtype=torch.long, device=features.device)
        fact = torch.cat((features.new_zeros(1), features[:, 0]))
        direct = fact[premises].sum(-1) / arity.clamp_min(1)
        edge_features = torch.stack((arity / MAX_PREMISES, direct, features[heads, 1],
                                     features[heads, 2], degree[heads]), dim=-1)
        gates = self.rule(edge_features).sigmoid()
        leaf = self.leaf(torch.cat((features, degree[:, None]), dim=-1)).sigmoid()
        fact_prices = torch.full((len(features),), CHECKS, dtype=torch.long, device=features.device)
        fact_prices = fact_prices.scatter_reduce(0, heads, torch.where(arity == 0, rule_prices, CHECKS), reduce="amin")
        positions = torch.where(features[:, 0] > 0, fact_prices, 2).unsqueeze(-1)
        prior = features.new_zeros((len(features), CHECKS + 1)).scatter(-1, positions, leaf).cumsum(-1)
        current = prior
        identity = torch.zeros_like(prior[:1])
        identity[0, 0] = 1.
        for _ in range(ROUNDS):
            table = torch.cat((identity, torch.diff(current, dim=-1, prepend=torch.zeros_like(current[:, :1]))), dim=0)
            mass = self.joint(table, premises)
            routed = shift_budget(mass.cumsum(-1), rule_prices, start=0) * gates
            routed = torch.where(arity[:, None] > 0, routed, torch.zeros_like(routed))
            current = prior.scatter_reduce(0, heads[:, None].expand_as(routed), routed, reduce="amax")
        table = torch.cat((identity, torch.diff(current, dim=-1, prepend=torch.zeros_like(current[:, :1]))), dim=0)
        width = max(map(len, state.projected), default=0)
        candidates = torch.tensor([list(premises) + [0] * (width - len(premises)) for premises in state.projected],
                                  dtype=torch.long, device=features.device)
        action_prices = torch.tensor(state.action_prices or (1,) * len(candidates), dtype=torch.long, device=features.device)
        composed = shift_budget(self.joint(table, candidates).cumsum(-1), action_prices, start=1)
        if foreign_budget:
            composed = composed.roll(1, dims=0)
        return composed.lerp(state.prior, self.mix.sigmoid())


class BudgetTRM(nn.Module):
    def __init__(self, *, priced=False):
        super().__init__()
        self.graph = GraphEvidence("trm")
        self.head = nn.Linear(4 * GRAPH_WIDTH + 1, CHECKS + 1)
        self.mix = nn.Parameter(torch.tensor(-2.5))
        self.price = nn.Linear(1, GRAPH_WIDTH, bias=False) if priced else None

    def forward(self, state):
        context = None
        if self.price is not None:
            if not state.rule_prices:
                raise ValueError("a priced graph TRM needs all public rule prices")
            context = self.price(self.head.weight.new_tensor(state.rule_prices).unsqueeze(-1) / CHECKS)
        nodes, edges, features, _, _, ids = self.graph.graph_states(state.graph, TRM_PASSES, rule_context=context)
        candidates = torch.cat((edges.new_zeros(1, GRAPH_WIDTH), edges))[ids]
        pooled = []
        for mask in features[:, 1:].T:
            pooled.append((nodes * mask[:, None]).sum(0).expand_as(candidates) / mask.sum().clamp_min(1))
        padded = torch.cat((nodes.new_zeros(1, GRAPH_WIDTH), nodes))
        obligations = torch.stack([padded[list(ids)].sum(0) for ids in state.projected])
        lengths = nodes.new_tensor([len(ids) / CHECKS for ids in state.projected])
        logits = self.head(torch.cat((candidates, *pooled, obligations, lengths[:, None]), dim=-1))
        return logits.softmax(dim=-1).cumsum(dim=-1)[..., :-1].lerp(state.prior, self.mix.sigmoid())
