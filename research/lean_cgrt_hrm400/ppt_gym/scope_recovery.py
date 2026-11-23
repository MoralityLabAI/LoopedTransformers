"""A proof decision with expiring checks and a tentative two-slot update."""

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product

from .rollback_diagnostics import DIAGNOSTIC_COST


WITNESSES = (0, 1)
EARLY = (2, 3)
OPEN = 4
TENTATIVE = 5
RESOLVE = 6
N_ACTIONS = 7
BUDGETS = (1, 2, 3)
CHECK_INDEX = {EARLY[0]: 0, EARLY[1]: 1, TENTATIVE: 2, RESOLVE: 3}
UNKNOWN = -1


@dataclass(frozen=True)
class State:
    remaining: int
    phase: int = 0
    early: tuple[int, int] = (UNKNOWN, UNKNOWN)
    tentative: int = UNKNOWN
    keep: int = UNKNOWN

    def values(self):
        return (*self.early, self.tentative, self.keep)


def legal(state):
    if state.phase == 0:
        return (*[action for action, value in zip(EARLY, state.early, strict=True)
                  if state.remaining and value == UNKNOWN], OPEN)
    if state.phase == 1:
        return (TENTATIVE, RESOLVE) if state.remaining and state.tentative == UNKNOWN else (RESOLVE,)
    return WITNESSES


def step(state, action, bit=None):
    if action not in legal(state):
        raise ValueError("the scope policy selected an unavailable Lean action")
    if action in WITNESSES:
        return None
    if action == OPEN:
        return replace(state, phase=1)
    if bit not in (0, 1):
        raise ValueError("the branch transition requires a verified bit")
    if action in EARLY:
        address = CHECK_INDEX[action]
        values = tuple(bit if i == address else old for i, old in enumerate(state.early))
        return replace(state, early=values, remaining=state.remaining - 1)
    if action == TENTATIVE:
        return replace(state, tentative=bit, remaining=state.remaining - 1)
    return replace(state, phase=2, keep=bit)


class Layout:
    """All accessibility and reward labels come from the frozen native evidence."""

    def __init__(self, rows):
        self.edited = rows[0]["edited"]
        self.worlds = {tuple(item["value"] for item in row["observations"]): row for row in rows}
        if (set(self.worlds) != set(product((0, 1), repeat=4)) or len(rows) != len(self.worlds)
                or any(row["edited"] != self.edited for row in rows)):
            raise ValueError("scope layout lost its paired native worlds")

    @lru_cache(maxsize=None)
    def compatible(self, state):
        return tuple(bits for bits in self.worlds if all(value in (UNKNOWN, bit)
                     for bit, value in zip(bits, state.values(), strict=True)))

    def transitions(self, state, action):
        compatible = self.compatible(state)
        if action in WITNESSES:
            success = sum(self.worlds[bits]["witness_trials"][action]["passed"] for bits in compatible)
            return success / len(compatible), ((1., None),)
        if action == OPEN:
            return 0., ((1., step(state, action)),)
        index = CHECK_INDEX[action]
        probabilities = [sum(bits[index] == value for bits in compatible) / len(compatible) for value in (0, 1)]
        return -DIAGNOSTIC_COST, tuple((prob, step(state, action, bit))
                                       for bit, prob in enumerate(probabilities) if prob)

    @lru_cache(maxsize=None)
    def optimal(self, state):
        scores = {}
        for action in legal(state):
            reward, following = self.transitions(state, action)
            scores[action] = reward + sum(prob * max(self.optimal(child).values())
                                          for prob, child in following if child is not None)
        return scores


def rollout(row, budget, decide):
    state = State(budget)
    trace = []
    while True:
        action = decide(state)
        if action not in legal(state):
            raise ValueError("the scope policy selected an unavailable Lean action")
        if action in WITNESSES:
            proof = row["witness_trials"][action]
            break
        index = CHECK_INDEX.get(action)
        bit = row["observations"][index]["value"] if index is not None else None
        trace.append({"action": action, "bit": bit})
        state = step(state, action, bit)
    diagnostics = sum(event["action"] in CHECK_INDEX for event in trace)
    if diagnostics != budget - state.remaining + 1:
        raise ValueError("the mandatory scope check or optional probe accounting changed")
    return {"trace": trace, "witness": action, "proved": proof["passed"], "diagnostics": diagnostics,
            "checks": diagnostics + proof["checks"], "utility": float(proof["passed"]) - DIAGNOSTIC_COST * diagnostics}
