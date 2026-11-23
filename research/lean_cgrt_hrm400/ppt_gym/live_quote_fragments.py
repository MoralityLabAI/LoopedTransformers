"""Bellman intervals over verified Lean transitions and paid tariff observations."""

from collections import defaultdict
from functools import lru_cache
from math import fsum

from .data import digest
from .private_quotes import PENALTY, QUOTES


UNKNOWN = (0., 1.)
FAIL = (0., 0.)
PROOF = (1., 1.)


class NativeFragments:
    """Validate short native fragments and join them by exact public proof state."""

    def __init__(self, log):
        self.root = log["root"]
        self.states = log["states"]
        if self.root not in self.states or any(digest(value["public"]) != key for key, value in self.states.items()):
            raise ValueError("native observation hashes changed")
        self.edges = defaultdict(dict)
        visited = {self.root}
        for fragment in log["fragments"]:
            key = fragment["start"]
            steps = fragment["steps"]
            if key not in visited or not 1 <= len(steps) <= 2:
                raise ValueError("a fragment restarted outside checked history or crossed its cut")
            for index, step in enumerate(steps):
                action, status, after = step["action"], step["status"], step["after"]
                if step["before"] != key or action not in self.states[key]["menu"]:
                    raise ValueError("a fragment changed its checked action or observation")
                if step != self.edges[key].setdefault(action, step):
                    raise ValueError("the same exact native state/action pair has conflicting successors")
                if status not in ("progress", "closed", "resource_limit", "no_change", "rejected"):
                    raise ValueError("an unrecognized native tactic outcome entered the ledger")
                if status == "progress":
                    if after not in self.states:
                        raise ValueError("a checked successor observation is missing")
                    visited.add(after)
                    key = after
                elif after is not None or index != len(steps) - 1:
                    raise ValueError("a terminal action did not end its fragment")
        if visited != self.states.keys():
            raise ValueError("unvisited public states outside the fragment collection")


class NativeQuoteBackup:
    """Back up only recorded edges, retaining an upper bound for unseen tactics."""

    def __init__(self, native, fees, billing, *, quotes_before_tactics=False):
        self.native, self.fees, self.billing = native, fees, billing
        self.initial_belief = tuple(range(len(billing)))
        self.quotes_before_tactics = quotes_before_tactics
        charged = {action for edges in native.edges.values() for action in edges}
        if any(set(world["prices"]) != charged or set(world["bits"]) != set(QUOTES) for world in billing):
            raise ValueError("the source meter must cover exactly the checked tactic identities and purchased quote channels")
        self.value = lru_cache(maxsize=None)(self._value)
        self.action = lru_cache(maxsize=None)(self._action)

    def quote_bit(self, world, name):
        return self.billing[world]["bits"][name]

    def tactic_price(self, world, name):
        return self.billing[world]["prices"][name]

    def groups(self, belief, observe):
        groups = defaultdict(list)
        for world in belief:
            groups[observe(world)].append(world)
        return [(outcome, tuple(worlds)) for outcome, worlds in sorted(groups.items())]

    def menu(self, key, left, ticks, quotes, bought):
        if not left or not ticks:
            return ()
        tactics = self.native.states[key]["menu"]
        can_quote = quotes and (not self.quotes_before_tactics or ticks == 8)
        purchasable = [action for index, action in enumerate(QUOTES) if can_quote and self.fees[action] <= left
                       and not bought & (1 << index)]
        return (*tactics, *purchasable)

    def _value(self, key, left, ticks, belief, quotes, bought):
        bounds = [self.action(key, left, ticks, belief, quotes, bought, move)
                  for move in self.menu(key, left, ticks, quotes, bought)]
        return tuple(max((0., *(b[index] for b in bounds))) for index in range(2))

    def _action(self, key, left, ticks, belief, quotes, bought, move):
        if move in QUOTES:
            fee = self.fees[move]
            branches = self.groups(belief, lambda world: self.quote_bit(world, move))
            used = bought | (1 << QUOTES.index(move))
            bounds = [(len(support), self.value(key, left - fee, ticks, support, quotes - 1, used))
                      for _, support in branches]
            cost = fee * PENALTY
        else:
            edge = self.native.edges[key].get(move)
            if not edge or edge["status"] == "resource_limit":
                return UNKNOWN
            branches = self.groups(belief, lambda world: self.tactic_price(world, move))
            bounds = []
            for price, support in branches:
                if price > left or edge["status"] in ("no_change", "rejected"):
                    value = FAIL
                elif edge["status"] == "closed":
                    value = PROOF
                else:
                    value = self.value(edge["after"], left - price, ticks - 1, support, quotes, bought)
                bounds.append((len(support), value))
            cost = 0.
        total = len(belief)
        return tuple(fsum(count * interval[index] for count, interval in bounds) / total - cost for index in range(2))

    def solve(self, world, budget, quote_limit):
        key, belief, left, quotes = self.native.root, self.initial_belief, budget, quote_limit
        ticks, bought = 8, 0
        script, trace = [], []
        proof = False
        while left and ticks:
            menu = self.menu(key, left, ticks, quotes, bought)
            checked = [move for move in menu if move in QUOTES or move in self.native.edges[key]]
            if not checked:
                break
            choose = lambda action: self.action(key, left, ticks, belief, quotes, bought, action)[0]
            move = max(checked, key=choose)
            if choose(move) <= 0.:
                break
            if move in QUOTES:
                charge, bit = self.fees[move], self.quote_bit(world, move)
                belief = tuple(other for other in belief if self.quote_bit(other, move) == bit)
                trace.append({"state": key, "quote": move, "charge": charge, "bit": bit})
                left -= charge
                quotes -= 1
                bought |= 1 << QUOTES.index(move)
                continue
            edge = self.native.edges[key][move]
            charge = self.tactic_price(world, move)
            if charge > left:
                trace.append({"state": key, "action": move, "charge": 0, "status": "unaffordable"})
                break
            left -= charge
            ticks -= 1
            belief = tuple(other for other in belief if self.tactic_price(other, move) == charge)
            trace.append({"state": key, "action": move, "charge": charge, "status": edge["status"]})
            if edge["status"] not in ("closed", "progress"):
                break
            script.append(move)
            if edge["status"] == "closed":
                proof = True
                break
            key = edge["after"]
        paid = sum(event["charge"] for event in trace if "quote" in event)
        return {"proof": proof, "script": script, "trace": trace, "budget": budget,
                "quote_effort": paid, "proof_effort": budget - left - paid, "utility": int(proof) - paid * PENALTY}
