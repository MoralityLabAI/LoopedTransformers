"""Value single-name normalization as a paid transition in the recurrent proof state."""

from dataclasses import dataclass
from functools import lru_cache
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from .alias_rewrite import TOKEN, alias_menu, local_definitions
from .data import canonical
from .hybrid_rl.graph_model import MAX_RULES, visible_graph
from .hybrid_rl.model import ProofPolicy
from .shared_rewrite import ConjunctiveGraphPolicy, SharedRewritePolicy, View, observation


@dataclass(frozen=True)
class AliasObservation:
    structured_packet: str
    view: View
    previews: tuple[tuple[int, tuple[str, ...], View], ...]


@lru_cache(maxsize=192)
def bind_observation(packet, actions):
    obs = json.loads(packet)
    goal = obs["state"]["goals"][0]
    definitions = local_definitions(goal)
    transitions = tuple((f"dsimp only [{name}]", name, value) for name, value in definitions.items())
    view = observation(packet, actions, transitions)
    if view is None:
        return None
    previews = []
    if view.term is not None:
        for action, name, value in transitions:
            if action not in actions:
                continue
            # Normalize the focused target. The local definitions are still in context.
            target = goal["target"]["raw"]
            projected = TOKEN.sub(lambda token: value if token.group(0) == name else token.group(0), target)
            changed = {**goal, "target": {"raw": projected}}
            state = {"goals": [changed, *obs["state"]["goals"][1:]]}
            candidates = tuple(tactic for tactic in alias_menu(state) if not tactic.startswith("dsimp only ["))
            if not candidates:
                continue
            successor = observation(canonical({**obs, "state": state}), candidates, transitions)
            if successor is not None:
                previews.append((actions.index(action), candidates, successor))
    encoded = {**goal, "target": {**goal["target"], "local_definitions": definitions}}
    structured = canonical({**obs, "state": {"goals": [encoded, *obs["state"]["goals"][1:]]}})
    return AliasObservation(structured, view, tuple(previews))


class AliasAwarePolicy(SharedRewritePolicy):
    """Use typed local bindings to value normalization now or within a later guard."""

    def __init__(self, architecture, width, *, residual_transport, premise_interaction):
        super().__init__(architecture, width, residual_transport=residual_transport, premise_interaction=premise_interaction)
        self.normalization_logit = nn.Parameter(torch.tensor(math.log(4.)))
        self.preview_logit = nn.Parameter(torch.zeros(()))

    def macs(self, candidates, passes):
        return super().macs(candidates, passes) + 2 * passes * 16 * MAX_RULES

    def replay_inputs(self, packet, actions, passes):
        view = bind_observation(packet, tuple(actions))
        if view is None:
            raise ValueError("the alias policy needs a supported public rewrite state")
        graph = view.view.term.structural if view.view.term is not None else visible_graph(packet, actions)
        if graph is None:
            logits, value = ProofPolicy.forward(self, view.structured_packet, actions, passes)
            return self.recurrent_log_gain.exp() * logits, value, view
        logits, value = self.routed_scores(view.structured_packet, actions, passes, graph)
        return logits, value, view

    def return_scores(self, view, actions, passes):
        transport = self.shared_transport
        discount = self.normalization_logit.clamp(-6., 6.).sigmoid()
        hops = 2 * passes
        state = view.view
        banks = transport.local_values(state.local, hops, hard=self.hard_shared, normalization=discount)
        values = transport(state, actions, hops, hard=self.hard_shared, normalization=discount, local_values=banks)
        if not view.previews:
            return values.clamp_min(1e-8).log()
        temperature = 1 + F.softplus(self.preview_logit)
        estimates = {}
        for index, candidates, successor in view.previews:
            downstream = transport(successor, candidates, hops, hard=self.hard_shared,
                                   normalization=discount, local_values=banks)
            if self.hard_shared:
                return_value = downstream.max()
            else:
                weights = downstream.clamp_min(1e-8).pow(temperature)
                return_value = (weights * downstream).sum() / weights.sum().clamp_min(1e-12)
            estimates[index] = discount * return_value
        return torch.stack([estimates.get(i, value) for i, value in enumerate(values)]).clamp_min(1e-8).log()

    def decision_scores(self, base, view, actions, passes):
        returns = self.return_scores(view, actions, passes)
        return returns if self.hard_shared else base + self.shared_transport.readout_weight * returns

    def forward(self, packet, actions, passes, **kwargs):
        if bind_observation(packet, tuple(actions)) is None:
            return super().forward(packet, actions, passes, **kwargs)
        base, value, view = self.replay_inputs(packet, actions, passes)
        return self.decision_scores(base, view, actions, passes), value


class AliasGraphPolicy(ConjunctiveGraphPolicy):
    """TRM control using the same public definition records and typed incidences."""

    def forward(self, packet, actions, passes, **kwargs):
        view = bind_observation(packet, tuple(actions))
        if view is None:
            return super().forward(packet, actions, passes, **kwargs)
        if view.view.term is None or view.view.term.structural is None:
            return super().forward(view.structured_packet, actions, passes, **kwargs)
        scores, value = ProofPolicy.forward(self, view.structured_packet, actions, passes, **kwargs)
        evidence, graph_value = self.graph(view.view.term.structural, passes)
        return scores + evidence, value + graph_value
