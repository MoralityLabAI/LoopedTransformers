"""Recurrently transport discounted proof returns along public obligations."""

import torch
from torch import nn
from torch.nn import functional as F

from .gated_dependent_rewrite import RoutedDependentRewritePolicy
from .hybrid_rl.graph_model import MAX_ATOMS, MAX_PREMISES, MAX_RULES


class DiscountedProofTransport(nn.Module):
    """Learn how concurrent evidence and serial tactic work contribute to proof value."""

    def __init__(self):
        super().__init__()
        self.serial_logit = nn.Parameter(torch.zeros(()))
        self.selection_logit = nn.Parameter(torch.zeros(()))
        self.readout_weight = nn.Parameter(torch.tensor(0.04))

    def forward(self, facts, premises, heads, hops, *, seriality=None):
        mix = self.serial_logit.sigmoid() if seriality is None else seriality
        power = 1 + F.softplus(self.selection_logit)
        bank = facts
        for _ in range(hops):
            support = torch.cat((bank.new_ones(1), bank))[premises]
            # The minimum models concurrent arrival. Multiplication discounts every required tactic.
            candidate = 0.8 * torch.lerp(support.amin(dim=-1), support.prod(dim=-1), mix)
            weights = candidate * candidate.clamp_min(1e-8).pow(power - 1)
            numerator = torch.zeros_like(bank).index_add(0, heads, weights * candidate)
            denominator = torch.zeros_like(bank).index_add(0, heads, weights)
            bank = torch.maximum(bank, numerator / denominator.clamp_min(1e-8))
        scores = candidate.clamp_min(1e-8).log()
        return self.readout_weight * scores


class DiscountedRewritePolicy(RoutedDependentRewritePolicy):
    """Combine recurrent workspace scores, fact routing, and a learned proof-return channel."""

    def __init__(self, architecture, width, *, residual_transport, premise_interaction):
        super().__init__(architecture, width, residual_transport=residual_transport,
                         premise_interaction=premise_interaction)
        self.proof_transport = DiscountedProofTransport()
        self.seriality_override = None

    def macs(self, candidates, passes):
        return (super().macs(candidates, passes)
                + 2 * passes * (3 * MAX_ATOMS + (3 * MAX_PREMISES + 9) * MAX_RULES) + candidates)

    def forward(self, packet, actions, passes, **kwargs):
        graph = self.observation_graph(packet, actions)
        if graph is None:
            return super().forward(packet, actions, passes, **kwargs)
        logits, value = self.routed_scores(packet, actions, passes, graph, **kwargs)
        facts = logits.new_tensor(graph.features)[:, 0]
        premises = torch.as_tensor(graph.premises, dtype=torch.long, device=logits.device)
        heads = torch.as_tensor(graph.heads, dtype=torch.long, device=logits.device)
        selected = torch.as_tensor(graph.actions, dtype=torch.long, device=logits.device)
        scores = self.proof_transport(facts, premises, heads, 2 * passes, seriality=self.seriality_override)
        return logits + torch.cat((logits.new_zeros(1), scores))[selected], value
