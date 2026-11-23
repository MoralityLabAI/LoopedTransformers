"""Observed transition and within-fragment utilities for paid Lean proof planning."""

from collections import Counter, defaultdict
from itertools import permutations
from math import fsum

from .data import digest
from .private_quotes import COMMITS, PENALTY, QUOTES, available, metered_rollout, public_view, quote_bit, scenario, REGIMES


def history_key(history):
    return tuple(sorted(history))


def source_fragments(case, regime):
    plans, ledger = case["plans"], case["ledger"]
    prices = [scenario(plans, ledger, case["other"], REGIMES[regime], world) for world in range(4)]
    outcomes = [(tariffs, {action: quote_bit(action, ledger, tariffs) for action in QUOTES}) for tariffs in prices]
    return taped_fragments(case["module"], regime, public_view(case, regime), outcomes)


def taped_fragments(module, regime, view, outcomes):
    slack = view["budget"] - view["min_proof_effort"]
    fees = view["fees"]
    fragments = []
    for length in range(len(QUOTES) + 1):
        for queries in permutations(QUOTES, length):
            if sum(fees[q] for q in queries) > slack:
                continue
            for commit in COMMITS:
                # Cut whole exploratory tapes independently of hidden world and feedback.
                split = int(digest([module, regime, queries, commit])[:8], 16)
                cut = 1 + split % length if length and split % 7 else 0
                for prices, bits in outcomes:
                    actions = iter((*queries, commit))
                    row = metered_rollout(view, prices, bits, lambda *_: next(actions))
                    steps = [{"action": event["action"], "bit": event["seen"], "charge": event["charge"]}
                             for event in row["trace"]]
                    steps.append({"action": commit, "proof": row["proof"], "effort": row["proof_effort"]})
                    fragments.append({"history": (), "steps": steps[:cut] if cut else steps})
                    if cut:
                        fragments.append({"history": history_key((event["action"], event["bit"]) for event in steps[:cut]),
                                          "steps": steps[cut:]})
    return fragments


class QuoteFragmentGraph:
    def __init__(self, view, fragments):
        self.view = view
        self.transitions = defaultdict(Counter)
        self.terminal = defaultdict(Counter)
        self.suffixes = defaultdict(list)
        self.states = defaultdict(set)
        for fragment in fragments:
            self.observe(fragment)
        anchors = {()} | {child for branches in self.transitions.values() for child in branches}
        if not self.states.keys() <= anchors:
            raise ValueError("a fragment restarted at an unvisited price observation")
        self.values = {}
        for history in sorted(self.states, key=len, reverse=True):
            for action in self.states[history]:
                self.values[history, action] = self.backed_up(history, action)

    def remaining(self, history):
        return self.view["budget"] - sum(self.view["fees"][q] for q, _ in history)

    def observe(self, fragment):
        history = tuple(tuple(event) for event in fragment["history"])
        if (history != history_key(history) or len({action for action, _ in history}) != len(history)
                or any(action not in QUOTES or bit not in (0, 1) for action, bit in history)):
            raise ValueError("a price fragment has an invalid or repeated preceding observation")
        preceding = []
        for index, event in enumerate(fragment["steps"]):
            action = event["action"]
            left = self.remaining(history)
            fees = self.view["fees"]
            if action not in available(self.view["quote_order"], fees, history, left):
                raise ValueError("a logged quote or plan choice was unavailable")
            self.states[history].add(action)
            pair = history, action
            if action in COMMITS:
                if index != len(fragment["steps"]) - 1 or event["effort"] > left:
                    raise ValueError("a commitment must terminate inside its remaining allowance")
                self.terminal[pair][event["proof"]] += 1
                value = int(event["proof"])
                self.suffixes[pair].append(value)
                for earlier, charge in reversed(preceding):
                    value -= PENALTY * charge
                    self.suffixes[earlier].append(value)
                return
            if event["charge"] != fees[action] or event["bit"] not in (0, 1):
                raise ValueError("a paid quote has inconsistent feedback or charge")
            successor = history_key((*history, (action, event["bit"])))
            self.transitions[pair][successor] += 1
            preceding.append((pair, event["charge"]))
            history = successor

    def backed_up(self, history, action):
        pair = history, action
        if pair in self.terminal:
            counts = self.terminal[pair]
            return counts[True] / counts.total()
        branches = self.transitions.get(pair)
        if not branches:
            return None
        weighted = []
        for child, count in branches.items():
            options = [self.values[child, following] for following in self.states.get(child, ())]
            known = [score for score in options if score is not None]
            if not known:
                return None
            weighted.append(count * max(known))
        return fsum(weighted) / branches.total() - PENALTY * self.view["fees"][action]

    def labels(self, history, action):
        observed = self.suffixes.get((history, action))
        return self.values.get((history, action)), fsum(observed) / len(observed) if observed else None

    def rows(self):
        for history, actions in sorted(self.states.items()):
            for action in sorted(actions):
                yield history, action, self.labels(history, action)
