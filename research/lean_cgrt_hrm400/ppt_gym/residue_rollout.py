"""Sequential Lean recovery with paid membership checks and route attempts."""

from dataclasses import dataclass
from functools import lru_cache
from math import fsum

from .failed_prefix import CHARGE, PREFIX_COST
from .residue_acquisition import QUERY_COST


STOP = 0
FIRST_ROUTE = 1
FIRST_QUERY = 4


@dataclass(frozen=True)
class State:
    remaining: int
    support: int
    queries_left: int


def mask(members):
    return sum(1 << member for member in set(members))


class Recovery:
    """Public transitions shared by learned policies and the exact planner."""

    def __init__(self, target, menu):
        self.routes = target["routes"]
        self.groups = tuple(mask(route["residues"]) for route in self.routes)
        self.queries = tuple(mask(query["members"]) for query in menu)

    def allowed(self, state):
        possible = [i for i, group in enumerate(self.groups) if state.support & group]
        affordable = [i for i in possible if self.routes[i]["pass_cost"] <= state.remaining]
        choices = [STOP, *(FIRST_ROUTE + i for i in affordable)]
        if state.queries_left and any(self.routes[i]["pass_cost"] + QUERY_COST <= state.remaining for i in affordable):
            choices.extend(FIRST_QUERY + i for i, group in enumerate(self.queries)
                           if state.support & group and state.support & ~group)
        return choices

    def transition(self, state, action, passed):
        """Return the immediate charge, closure bit and public successor."""
        if action == STOP:
            return 0, False, None
        if action >= FIRST_QUERY:
            group = self.queries[action - FIRST_QUERY]
            support = state.support & (group if passed else ~group)
            return QUERY_COST, False, State(state.remaining - QUERY_COST, support, state.queries_left - 1)
        index = action - FIRST_ROUTE
        route = self.routes[index]
        if passed:
            return route["pass_cost"], True, None
        charge = route["fail_cost"]
        return charge, False, State(state.remaining - charge, state.support & ~self.groups[index], state.queries_left)

    def planner(self, prior):
        @lru_cache(maxsize=None)
        def mass(support):
            return fsum(weight for index, weight in enumerate(prior) if support & (1 << index))

        @lru_cache(maxsize=None)
        def scores(state):
            total = mass(state.support)
            output = {STOP: 0.}
            if not total:
                return output
            for action in self.allowed(state)[1:]:
                group = (self.queries[action - FIRST_QUERY] if action >= FIRST_QUERY
                         else self.groups[action - FIRST_ROUTE])
                contributions = []
                for passed, branch in ((False, state.support & ~group), (True, state.support & group)):
                    if weight := mass(branch):
                        charge, proved, following = self.transition(state, action, passed)
                        continuation = max(scores(following).values()) if following is not None else 0.
                        contributions.append(weight * (float(proved) - CHARGE * charge + continuation))
                output[action] = fsum(contributions) / total
            return output

        return scores

    def rollout(self, state, route_outcomes, query_outcomes, decide):
        allowance, checks, proved, trace = state.remaining, PREFIX_COST, False, []
        while state is not None:
            action = decide(state, tuple(trace))
            if action not in self.allowed(state):
                raise ValueError("the policy selected an unavailable recovery action")
            if action == STOP:
                break
            passed = (query_outcomes[action - FIRST_QUERY] if action >= FIRST_QUERY
                      else route_outcomes[action - FIRST_ROUTE])
            charge, proved, following = self.transition(state, action, passed)
            if following is not None and not following.support:
                raise ValueError("the witnessed branch contradicted the current support")
            trace.append({"action": action, "passed": passed, "charge": charge, "before": state.remaining,
                          "support": state.support, "queries_left": state.queries_left})
            checks += charge
            state = following
        return {"allowance": allowance, "checks": checks, "proved": proved, "trace": trace,
                "utility": float(proved) - CHARGE * checks}


def greedy(scores):
    # Treat rounding ties as ties, with the first legal action taking priority.
    return max(scores, key=lambda action: round(scores[action], 12))
