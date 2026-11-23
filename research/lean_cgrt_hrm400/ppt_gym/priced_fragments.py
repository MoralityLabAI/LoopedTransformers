"""Budgeted backups through observed native transitions and empirical fragment returns."""

from collections import defaultdict

from .data import digest
from .mathcan_sparse import CHECKS


class FragmentGraph:
    def __init__(self, log):
        self.prices = log["prices"]
        if any(not isinstance(price, int) or price < 1 for price in self.prices.values()):
            raise ValueError("a recorded tactic price must be a positive integer")
        self.states = log["public_states"]
        self.edges = defaultdict(dict)
        self.finished = {}
        self.observed = {}
        if any(digest(state["public"]) != key for key, state in self.states.items()):
            raise ValueError("a recorded fragment changed its public observation")
        for episode in log["fragments"]:
            key = episode["start"]
            spent = 0
            for step in episode["steps"]:
                action = step["action"]
                if (step["before"] != key or action not in self.states[key]["menu"]
                        or step["cost"] != self.prices[action]):
                    raise ValueError("a source fragment changed its action, state or public price")
                previous = self.edges[key].setdefault(action, step)
                if previous != step:
                    raise ValueError("one exact Horn state has conflicting native successors")
                spent += step["cost"]
                key = step["after"]
            if spent > episode["allowance"]:
                raise ValueError("a source fragment exceeded its logged allowance")
            closed = bool(episode["steps"] and episode["steps"][-1]["status"] == "closed")
            suffix = 0
            for step in reversed(episode["steps"]):
                suffix += step["cost"]
                pair = step["before"], step["action"]
                self.observed[pair] = max(suffix, self.observed.get(pair, 0))
                if closed:
                    self.finished[pair] = min(suffix, self.finished.get(pair, CHECKS + 1))
        self.values = [{key: 0 for key in self.states}]
        for cap in range(1, CHECKS + 1):
            self.values.append({key: self.state_value(key, cap) for key in self.states})

    def outcome(self, key, action, cap):
        price = self.prices[action]
        if cap < price:
            return 0
        edge = self.edges[key].get(action)
        if edge is None or edge["status"] == "resource_limit":
            return -1
        status = edge["status"]
        if status == "closed":
            return 1
        if cap == price:
            return 0
        successor = edge["after"] if status == "progress" else key
        return self.values[cap - price].get(successor, -1)

    def state_value(self, key, cap):
        outcomes = [self.outcome(key, action, cap) for action in self.states[key]["menu"]]
        return 1 if 1 in outcomes else -1 if -1 in outcomes else 0

    def labels(self, key, action):
        backed_up = tuple(self.outcome(key, action, cap) for cap in range(1, CHECKS + 1))
        pair = key, action
        first = self.finished.get(pair, CHECKS + 1)
        seen = self.observed[pair]
        empirical = tuple(1 if cap >= first else 0 if cap <= seen else -1 for cap in range(1, CHECKS + 1))
        return backed_up, empirical
