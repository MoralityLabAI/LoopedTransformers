"""Learn immediate tactic effects and delayed proof closure on native prefixes."""

from functools import lru_cache

import torch
from torch import nn

from .data import digest
from .mathcan_progress import retry_curve
from .mathcan_route_policy import RoutePolicy
from .mathcan_sparse import CHECKS


DISPOSITIONS = ("closed", "progress", "no_change", "rejected")
BEHAVIOR = ("assumption", "rfl", "constructor", "simp", "omega", "decide", "left", "right", "contradiction")
OUT_OF_RANGE = CHECKS + 1


class NativeFeedbackRoute(RoutePolicy):
    def __init__(self, warm, *, trm=False):
        super().__init__(warm, trm=trm, deadline=True)
        self.disposition = nn.Linear(2 * self.width, len(DISPOSITIONS))

    def factors(self, tokens):
        hidden = self.features(tokens)
        logits = self.disposition(torch.cat((hidden, tokens[..., 0, :]), dim=-1))
        conditional = self.first_hit(hidden).softmax(dim=-1).cumsum(dim=-1)[..., :-1]
        return logits, conditional


def closure_curve(disposition, delayed):
    closed, progress, unchanged, rejected = disposition.unbind(dim=-1)
    later = torch.cat((torch.zeros_like(delayed[..., :1]), delayed[..., :-1]), dim=-1)
    direct = closed.unsqueeze(-1) + progress.unsqueeze(-1) * later
    return direct + (unchanged + rejected).unsqueeze(-1) * retry_curve(direct)


def behavior_choice(menu):
    def order(action):
        return (BEHAVIOR.index(action) if action in BEHAVIOR else len(BEHAVIOR), action)

    return min(menu, key=order)


class PrefixGraph:
    def __init__(self, theorem):
        self.nodes = {tuple(node["prefix"]): node for node in theorem["nodes"]}
        if () not in self.nodes or len(self.nodes) != len(theorem["nodes"]):
            raise ValueError("each source theorem must have one graph node per exact prefix")
        for prefix, node in self.nodes.items():
            if prefix:
                if prefix[:-1] not in self.nodes:
                    raise ValueError("an expanded proof prefix lost its recorded parent")
                previous = self.nodes[prefix[:-1]]["choices"][prefix[-1]]
                if previous["status"] != "progress" or previous["after_sha256"] != digest(node["public"]):
                    raise ValueError("an expanded proof prefix changed its verified Lean successor")

    @lru_cache(maxsize=None)
    def backed_up(self, prefix):
        node = self.nodes.get(prefix)
        if node is None:
            return OUT_OF_RANGE, 1
        closest, unknown = OUT_OF_RANGE, OUT_OF_RANGE
        for action, edge in node["choices"].items():
            status = edge["status"]
            if status == "closed":
                closest = 1
            elif status == "resource_limit":
                unknown = 1
            elif status == "progress":
                future, frontier = self.backed_up((*prefix, action))
                closest = min(closest, 1 + future)
                unknown = min(unknown, 1 + frontier)
        return closest, unknown

    def empirical(self, prefix):
        used = 0
        failed = set()
        while prefix in self.nodes and used < CHECKS:
            options = self.nodes[prefix]["choices"]
            available = options.keys() - failed
            if not available:
                break
            action = behavior_choice(available)
            status = options[action]["status"]
            if status == "resource_limit":
                break
            used += 1
            if status == "closed":
                return used, OUT_OF_RANGE
            if status == "progress":
                prefix = (*prefix, action)
                failed.clear()
            else:
                failed.add(action)
        return OUT_OF_RANGE, used + 1


def horizon_labels(closest, unknown):
    return tuple(1 if horizon >= closest else 0 if horizon < unknown else -1
                 for horizon in range(1, CHECKS + 1))
