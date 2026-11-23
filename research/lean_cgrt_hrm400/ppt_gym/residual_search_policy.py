"""Share typed Lean evidence between a sibling slot and a target slot."""

import re

from .data import digest
from .live_quote_policy import pad_features
from .residual_search import FEES, PROBE, STOP, TARGET


KINDS = ("apply", "cases", "assumption", "rfl", "simp", "omega", "decide", "constructor", "left", "right")
OUTCOMES = ("closed", "progress", "no_change", "rejected", "resource_limit")


def kind(tactic):
    name = tactic.partition(" ")[0]
    return [name == item for item in KINDS]


def disposition(status):
    return [status == name for name in OUTCOMES]


def shape(node):
    goals = node["public"]["goals"]
    focus = goals[0]
    target = focus["target"]
    return [len(goals) / 4, min(len(focus["hypotheses"]), 80) / 80, "and" in target,
            "or" in target, "a" in target, "=" in target.get("raw", "")]


def rule(node, tactic):
    focus = node["public"]["goals"][0]
    _, _, name = tactic.partition(" ")
    ty = next((hyp["type"] for hyp in focus["hypotheses"] if hyp["name"] == name), {})
    raw = ty.get("raw", "")
    premises = raw.count("→")
    while "imp" in ty:
        _, ty = ty["imp"]
        premises += 1
    return [bool(name), min(premises, 6) / 6, bool(name) and not premises,
            ty == focus["target"], "=" in raw, "∧" in raw]


def opaque_context(public, actions, nonce):
    names = {hyp["name"] for goal in public["goals"] for hyp in goal["hypotheses"]}
    aliases = {name: "anon_" + digest([nonce, name])[:12] for name in names}
    if not aliases:
        return public, actions
    pattern = re.compile(r"(?<![\w.'’])(?:" + "|".join(re.escape(name) for name in sorted(aliases)) + r")(?![\w.'’])")

    def rename(value):
        if isinstance(value, str):
            return pattern.sub(lambda match: aliases[match[0]], value)
        if isinstance(value, list):
            return [rename(item) for item in value]
        if isinstance(value, dict):
            return {key: rename(item) for key, item in value.items()}
        return value

    def quoted(action):
        tactic, separator, name = action.partition(" ")
        return tactic + separator + aliases.get(name, name) if tactic in ("apply", "cases") else action

    return rename(public), tuple(quoted(action) for action in actions)


class NativeEvidence:
    def __init__(self, search, prior, nonce=None):
        self.search = search
        self.prior = prior
        self.cache = {}
        self.nonce = nonce

    def curves(self, node):
        key = node["key"]
        if key not in self.cache:
            actions = tuple(node["actions"])
            public, listed = opaque_context(node["public"], actions, self.nonce) if self.nonce else (
                node["public"], actions)
            expected = self.prior.curves(public, listed, ()).tolist()
            self.cache[key] = dict(zip(actions, expected, strict=True))
        return self.cache[key]

    def describe(self, state):
        focus, sibling = self.search.node(state), self.search.sibling
        menu = self.search.menu(state)
        local, remote = self.curves(focus), self.curves(sibling)
        probe, observed = state.probe
        last, status = state.last
        context = [state.left / 2, bool(state.last[0]), bool(probe)]
        active = [*shape(focus), *kind(last), *disposition(status)]
        support = [*shape(sibling), *kind(probe), *disposition(observed)]
        horizon = state.left - 1
        best = max((local[tactic][horizon] - FEES[TARGET] for role, tactic in menu if role == TARGET), default=0.)
        rows, anchors = [], []
        for role, tactic in menu:
            warm = (remote if role == PROBE else local).get(tactic, [0.] * 8)
            cues = [*context, role == PROBE, role == TARGET, role == STOP[0], *kind(tactic), *warm,
                    *rule(sibling if role == PROBE else focus, tactic)]
            rows.append([pad_features([*active, bool(last) and last == tactic]),
                         pad_features([*support, bool(probe) and probe == tactic]), pad_features(cues)])
            anchors.append(best - FEES[PROBE] if role == PROBE else
                           warm[horizon] - FEES[TARGET] if role == TARGET else 0.)
        return rows, anchors
