"""Optional learned gating of fact transport in dependent rewrites."""

import torch
from torch import nn

from .dependent_rewrite import DependentRewritePolicy, dependent_graph
from .hybrid_rl.graph_model import visible_graph
from .hybrid_rl.model import ProofPolicy


class GatedDependentRewritePolicy(DependentRewritePolicy):
    """Let binding-aware transport weight messages by the evidence they carry."""

    def __init__(self, architecture, width, *, residual_transport, premise_interaction):
        super().__init__(architecture, width, residual_transport=residual_transport,
                         premise_interaction=premise_interaction)
        if self.graph.transport is not None:
            gate = nn.Linear(2, 1, bias=False)
            nn.init.zeros_(gate.weight)
            self.graph.transport.evidence_gate = gate


class RoutedDependentRewritePolicy(GatedDependentRewritePolicy):
    """Route the recurrent workspace and successively transported guard evidence separately."""

    def __init__(self, architecture, width, *, residual_transport, premise_interaction):
        super().__init__(architecture, width, residual_transport=residual_transport,
                         premise_interaction=premise_interaction)
        if self.graph.transport is None:
            raise ValueError("dependent evidence routing requires hybrid fact transport")
        self.recurrent_log_gain = nn.Parameter(torch.zeros(()))

    def workspace_scores(self, packet, actions, passes, graph, **kwargs):
        logits, value = ProofPolicy.forward(self, packet, actions, passes, **kwargs)
        graph_logits, graph_value = self.graph(graph, passes, include_transport=False)
        return logits + graph_logits, value + graph_value

    def macs(self, candidates, passes):
        return super().macs(candidates, passes) + 2 * candidates + 1

    @staticmethod
    def observation_graph(packet, actions):
        graph = dependent_graph(packet, actions)
        if graph is None and all(tactic.startswith("rw [") or tactic == "rfl" for tactic in actions):
            graph = visible_graph(packet, actions)
        return graph

    def forward(self, packet, actions, passes, **kwargs):
        graph = self.observation_graph(packet, actions)
        if graph is None:
            return super().forward(packet, actions, passes, **kwargs)
        logits, value = self.workspace_scores(packet, actions, passes, graph, **kwargs)
        features = logits.new_tensor(graph.features)[:, :1]
        premises = torch.as_tensor(graph.premises, dtype=torch.long, device=logits.device)
        heads = torch.as_tensor(graph.heads, dtype=torch.long, device=logits.device)
        selected = torch.as_tensor(graph.actions, dtype=torch.long, device=logits.device)
        support = self.graph.transport(features, premises, heads, 2 * passes)
        support = torch.cat((logits.new_zeros(1), support))[selected]
        return self.recurrent_log_gain.exp() * logits + support, value
