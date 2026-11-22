"""Give both recurrent policies the full public Horn premise graph."""
from dataclasses import dataclass
import json

import torch
from torch import nn

from lcgrt.tactic_lm import PostBlock
from .model import ProofPolicy

GRAPH_WIDTH = 32
MAX_ATOMS = 48
MAX_RULES = 64
MAX_PREMISES = 4


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
    def __init__(self, architecture):
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

    @staticmethod
    def combine(nodes, premises, heads):
        padded = torch.cat((nodes.new_zeros(1, GRAPH_WIDTH), nodes), dim=0)
        arity = (premises > 0).sum(dim=1, keepdim=True)
        source = padded[premises].sum(dim=1) / arity.clamp_min(1)
        return torch.cat((source, nodes[heads], arity / MAX_PREMISES), dim=-1)

    def forward(self, graph, passes):
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
        active = features[:, 1:2]
        return torch.cat((scores.new_zeros(1), scores))[selected], self.value((nodes * active).sum(0)).squeeze()

    def upper_macs(self, passes):
        d, n, e = GRAPH_WIDTH, MAX_ATOMS, MAX_RULES
        edge = e * (2 * d + 1) * d
        if self.architecture == "hybrid":
            return 2 * edge + 2 * passes * (edge + 9 * n * d * d) + (4 * n + e) * d
        length = n + e
        attention = 6 * length * d * d + 2 * length * length * d
        return edge + 6 * passes * attention + (4 * n + e) * d


class GraphProofPolicy(ProofPolicy):
    """Keep the lexical workspace, and add a learned full-context readout."""

    def __init__(self, architecture, width=256):
        super().__init__(architecture, width)
        self.graph = GraphEvidence(architecture)

    def macs(self, candidates, passes):
        return super().macs(candidates, passes) + self.graph.upper_macs(passes)

    def forward(self, packet, actions, passes, **kwargs):
        static = visible_graph(packet, actions)
        logits, value = super().forward(packet, actions, passes, **kwargs)
        if static is None:
            return logits, value
        graph_logits, graph_value = self.graph(static, passes)
        return logits + graph_logits, value + graph_value
