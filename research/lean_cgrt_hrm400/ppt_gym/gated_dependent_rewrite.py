"""Optional learned gating of fact transport in dependent rewrites."""

from torch import nn

from .dependent_rewrite import DependentRewritePolicy


class GatedDependentRewritePolicy(DependentRewritePolicy):
    """Let binding-aware transport weight messages by the evidence they carry."""

    def __init__(self, architecture, width, *, residual_transport, premise_interaction):
        super().__init__(architecture, width, residual_transport=residual_transport,
                         premise_interaction=premise_interaction)
        if self.graph.transport is not None:
            gate = nn.Linear(2, 1, bias=False)
            nn.init.zeros_(gate.weight)
            self.graph.transport.evidence_gate = gate
