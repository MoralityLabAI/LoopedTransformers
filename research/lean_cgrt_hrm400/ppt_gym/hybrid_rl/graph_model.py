"""Give both recurrent policies the full public Horn premise graph."""
from dataclasses import dataclass
import json
import math

import torch
from torch import nn

from lcgrt.tactic_lm import PostBlock
from .model import ProofPolicy
from ..data import canonical

GRAPH_WIDTH = 32
MAX_ATOMS = 48
MAX_RULES = 64
MAX_PREMISES = 4
FACT_CHANNELS = 8
INTERACTION_LOGIT_SCALE = 128


def atom(formula):
    return formula.get("a")


def target_atoms(formula):
    if "a" in formula:
        return {formula["a"]}
    return set().union(*(target_atoms(child) for child in formula.get("and", [])))


@dataclass(frozen=True)
class Graph:
    features: list
    premises: list
    heads: list
    actions: list


def visible_graph(packet, actions):
    """Parse forward implications only. This evaluates no reachability labels."""
    goals = json.loads(packet)["state"]["goals"]
    active = target_atoms(goals[0]["target"])
    if not active:
        return None
    pending = set().union(*(target_atoms(goal["target"]) for goal in goals[1:]))
    discovered = active | pending
    facts, rules = set(), []
    for hyp in goals[0]["hypotheses"]:
        tail, sources = hyp["type"], []
        while "imp" in tail:
            premise, tail = tail["imp"]
            sources.append(atom(premise))
        head = atom(tail)
        if head is None or None in sources or len(sources) > MAX_PREMISES:
            continue
        discovered.update((*sources, head))
        if not sources:
            facts.add(head)
        rules.append((hyp["name"], sources, head))
    if not rules or len(rules) > MAX_RULES or len(discovered) > MAX_ATOMS:
        return None
    nodes = sorted(discovered)
    index = {name: i for i, name in enumerate(nodes)}
    by_name = {name: i + 1 for i, (name, _, _) in enumerate(rules)}
    source_ids = [[index[name] + 1 for name in source] + [0] * (MAX_PREMISES - len(source))
                  for _, source, _ in rules]
    heads = [index[head] for _, _, head in rules]
    return Graph([[float(name in facts), float(name in active), float(name in pending)]
                  for name in nodes], source_ids, heads,
                 [by_name.get(action.removeprefix("apply "), 0) for action in actions])


class GraphEvidence(nn.Module):
    def __init__(self, architecture, *, residual_transport=False, premise_interaction=False, root_fact_adapter=False,
                 arrival_transport=False):
        super().__init__()
        d = GRAPH_WIDTH
        self.architecture = architecture
        self.node = nn.Linear(3, d)
        self.atom_id = nn.Embedding(MAX_ATOMS, d)
        self.edge = nn.Linear(2 * d + 1, d)
        self.action = nn.Linear(d, 1, bias=False)
        self.value = nn.Linear(d, 1, bias=False)
        if architecture == "hybrid":
            self.route = nn.Linear(2 * d + 1, d)
            self.write = nn.GRUCell(2 * d, d)
        else:
            self.recursion = nn.Sequential(PostBlock(d, 4, 1), PostBlock(d, 4, 1))
        self.transport = ResidualFactTransport(premise_interaction=premise_interaction) if residual_transport else None
        self.root_transport = (ResidualFactTransport(premise_interaction=True, arrival_transport=arrival_transport)
                               if root_fact_adapter else None)
        self.root_gain = nn.Parameter(torch.tensor(0.0)) if root_fact_adapter else None

    @staticmethod
    def combine(nodes, premises, heads):
        padded = torch.cat((nodes.new_zeros(1, GRAPH_WIDTH), nodes), dim=0)
        arity = (premises > 0).sum(dim=1, keepdim=True)
        source = padded[premises].sum(dim=1) / arity.clamp_min(1)
        return torch.cat((source, nodes[heads], arity / MAX_PREMISES), dim=-1)

    def forward(self, graph, passes, *, use_adapter=False):
        features = self.node.weight.new_tensor(graph.features)
        premises = torch.as_tensor(graph.premises, dtype=torch.long, device=features.device)
        heads = torch.as_tensor(graph.heads, dtype=torch.long, device=features.device)
        selected = torch.as_tensor(graph.actions, dtype=torch.long, device=features.device)
        ids = torch.arange(features.shape[0], device=features.device)
        original = self.node(features) + self.atom_id(ids)
        edges = self.edge(self.combine(original, premises, heads))
        if self.architecture == "hybrid":
            bank = original
            degree = torch.bincount(heads, minlength=features.shape[0]).clamp_min(1).unsqueeze(1)
            # Two synchronous evidence updates in each tied workspace pass.
            for _ in range(2 * passes):
                votes = self.route(self.combine(bank, premises, heads)).tanh()
                messages = torch.zeros_like(bank).index_add(0, heads, votes) / degree
                bank = self.write(torch.cat((original, messages), dim=-1), bank)
            edges = edges + self.edge(self.combine(bank, premises, heads))
            nodes = bank
        else:
            cue = torch.cat((original, edges), dim=0).unsqueeze(0)
            answer, latent = torch.zeros_like(cue), torch.zeros_like(cue)
            for _ in range(passes):
                for _ in range(2):
                    latent = self.recursion(cue + answer + latent)
                answer = self.recursion(answer + latent)
            nodes, edges = answer[0].split([original.shape[0], edges.shape[0]])
        scores = self.action(edges).squeeze(-1)
        if use_adapter and self.root_transport is not None:
            gain = (math.log(8) * (12 * self.root_gain).tanh()).exp()
            scores = scores + self.root_transport(features[:, :1], premises, heads, 2 * passes, gain=gain)
        elif self.transport is not None:
            scores = scores + self.transport(features[:, :1], premises, heads, 2 * passes)
        active = features[:, 1:2]
        return torch.cat((scores.new_zeros(1), scores))[selected], self.value((nodes * active).sum(0)).squeeze()

    def upper_macs(self, passes):
        d, n, e = GRAPH_WIDTH, MAX_ATOMS, MAX_RULES
        edge = e * (2 * d + 1) * d
        if self.architecture == "hybrid":
            transport = (
                (2 * n + e * MAX_PREMISES + 2 * passes * (n + e * (MAX_PREMISES + 2))) * FACT_CHANNELS
                if self.transport is not None else 0
            )
            if self.transport is not None and self.transport.interaction is not None:
                transport += (n + 2 * passes * (3 * n + e * (6 * MAX_PREMISES + 8))) * FACT_CHANNELS
            if self.root_transport is not None:
                transport += e + 8
                if self.root_transport.arrival is not None:
                    transport += ((2 * passes + 1) * n + e * MAX_PREMISES) * FACT_CHANNELS
            return 2 * edge + 2 * passes * (edge + 9 * n * d * d) + (4 * n + e) * d + transport
        length = n + e
        attention = 6 * length * d * d + 2 * length * length * d
        return edge + 6 * passes * attention + (4 * n + e) * d


class ResidualFactTransport(nn.Module):
    """Carry seeded evidence through public incidence without repeated nonlinear contraction."""

    def __init__(self, *, premise_interaction=False, arrival_transport=False):
        super().__init__()
        if arrival_transport and not premise_interaction:
            raise ValueError("arrival transport requires joint-premise evidence")
        self.source = nn.Linear(1, FACT_CHANNELS, bias=False)
        self.retention = nn.Parameter(torch.full((FACT_CHANNELS,), 3.4))
        self.action = nn.Linear(FACT_CHANNELS, 1, bias=False)
        self.interaction = nn.Parameter(torch.zeros(FACT_CHANNELS)) if premise_interaction else None
        self.arrival = nn.Parameter(torch.zeros(FACT_CHANNELS)) if arrival_transport else None
        self.interaction_readout = nn.Identity()
        self.arrival_readout = nn.Identity()
        # Preserve the untrained policy. The actor learns whether and how much
        # transported evidence should influence its action preference.
        nn.init.zeros_(self.action.weight)

    def forward(self, facts, premises, heads, hops, *, gain=None):
        seed = self.source(facts)
        bank = seed
        arity = (premises > 0).sum(dim=1, keepdim=True)
        divisor = arity.clamp_min(1)
        degree = seed.new_zeros((len(seed), 1)).index_add(0, heads, (arity > 0).to(seed)).clamp_min(1)
        retention = self.retention.sigmoid()
        correction = torch.zeros_like(seed) if self.interaction is not None else None
        strength = self.interaction.tanh() if self.interaction is not None else None
        integrated = torch.zeros_like(seed) if self.arrival is not None else None
        for _ in range(hops):
            extended = torch.cat((bank.new_zeros(1, FACT_CHANNELS), bank), dim=0)
            incoming = extended[premises]
            votes = incoming.sum(dim=1) / divisor
            delivered = torch.zeros_like(bank).index_add(0, heads, votes) / degree
            bank = seed + retention * delivered
            if correction is not None:
                padded = torch.cat((correction.new_zeros(1, FACT_CHANNELS), correction), dim=0)
                residual = padded[premises]
                support = incoming + residual
                variance = (support.square().sum(dim=1) / divisor -
                            (support.sum(dim=1) / divisor).square()).clamp_min(0)
                delta = residual.sum(dim=1) / divisor + strength * variance.tanh()
                correction = retention * torch.zeros_like(correction).index_add(0, heads, delta) / degree
            if integrated is not None:
                integrated = integrated + bank + correction
        # Project first, then gather the scalar coordinate for each premise.
        # This is the same linear readout as projecting each premise mean.
        scalar = torch.cat((bank.new_zeros(1), self.action(bank).flatten()))
        scores = 8 * scalar[premises].sum(dim=1) / divisor.flatten()
        if correction is not None:
            residual = torch.cat((bank.new_zeros(1), correction.mean(dim=1)))
            interaction_scores = INTERACTION_LOGIT_SCALE * residual[premises].sum(dim=1) / divisor.flatten()
            if gain is not None:
                interaction_scores = gain * interaction_scores
            scores = scores + self.interaction_readout(interaction_scores)
        if integrated is not None:
            # Earlier support contributes over more recurrent passes. The learned
            # projection decides whether that history is relevant to the action.
            integrated = integrated @ self.arrival.tanh() / hops
            padded = torch.cat((bank.new_zeros(1), integrated))
            arrival_scores = 32 * padded[premises].sum(dim=1) / divisor.flatten()
            scores = scores + self.arrival_readout(arrival_scores)
        return scores


class RecoveryReadout(nn.Module):
    """Compare the saved branch point with the current recurrent proof state."""

    def __init__(self, architecture):
        super().__init__()
        self.readout = nn.Linear(2, 1)
        self.transport = (ResidualFactTransport(premise_interaction=True, arrival_transport=True)
                          if architecture == "hybrid" else None)
        self.history = nn.Identity()
        with torch.no_grad():
            self.readout.weight.copy_(torch.tensor([[0.5, 0.5]]))
            self.readout.bias.zero_()

    def forward(self, policy, observation, passes):
        parent, current = canonical(observation["parent"]), canonical(observation["current"])
        menu = observation["parent_menu"]
        root, _ = policy(parent, menu, passes)
        later, value = policy(current, menu, passes)
        chosen = observation["explored"]
        other = 1 - chosen
        evidence = torch.stack((root[chosen] - root[other], later[chosen] - later[other]))
        margin = self.readout(evidence).squeeze()
        if self.transport is not None:
            graph = visible_graph(parent, menu)
            if graph is not None:
                features = root.new_tensor(graph.features)
                premises = torch.as_tensor(graph.premises, dtype=torch.long, device=root.device)
                heads = torch.as_tensor(graph.heads, dtype=torch.long, device=root.device)
                raw = self.transport(features[:, :1], premises, heads, 2 * passes)
                candidates = self.history(torch.cat((raw.new_zeros(1), raw))[graph.actions])
                margin = margin + candidates[chosen] - candidates[other]
        return torch.stack((margin / 2, -margin / 2)), value


class GraphProofPolicy(ProofPolicy):
    """Keep the lexical workspace, and add a learned full-context readout."""

    def __init__(self, architecture, width=256, *, residual_transport=False, premise_interaction=False,
                 root_fact_adapter=False, arrival_transport=False, adapt_continuation=False, branch_recovery=False):
        super().__init__(architecture, width)
        if residual_transport and architecture != "hybrid":
            raise ValueError("residual transport is a hybrid routing architecture")
        if premise_interaction and not residual_transport:
            raise ValueError("premise interaction requires residual fact transport")
        if root_fact_adapter and not premise_interaction:
            raise ValueError("first-action transport requires premise interaction")
        if arrival_transport and not root_fact_adapter:
            raise ValueError("arrival transport requires first-action adaptation")
        if adapt_continuation and not root_fact_adapter:
            raise ValueError("continuation routing requires an evidence adapter")
        self.adapt_continuation = adapt_continuation
        self.graph = GraphEvidence(architecture, residual_transport=residual_transport,
                                   premise_interaction=premise_interaction, root_fact_adapter=root_fact_adapter,
                                   arrival_transport=arrival_transport)
        self.recovery_head = RecoveryReadout(architecture) if branch_recovery else None

    def recovery_macs(self, passes):
        base = 2 * self.macs(2, passes)
        if self.recovery_head is None or self.recovery_head.transport is None:
            return base
        # The extra path is only the transport subset of a graph pass.
        return base + self.graph.upper_macs(passes)

    def recover(self, packet, passes):
        return self.recovery_head(self, json.loads(packet), passes)

    def macs(self, candidates, passes):
        return super().macs(candidates, passes) + self.graph.upper_macs(passes)

    def forward(self, packet, actions, passes, **kwargs):
        static = visible_graph(packet, actions)
        logits, value = super().forward(packet, actions, passes, **kwargs)
        if static is None:
            return logits, value
        adapted = False
        if self.graph.root_transport is not None:
            first_action = json.loads(packet)["checks_remaining"] == 8
            adapted = first_action != self.adapt_continuation
        graph_logits, graph_value = self.graph(static, passes, use_adapter=adapted)
        return logits + graph_logits, value + graph_value
