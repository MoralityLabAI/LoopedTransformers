"""Buy Lean feedback before choosing a scope on an unseen query key."""

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations


PROBES = (1, 2, 3, 4)
OPEN = 4
BRANCHES = (5, 6)
ACTIONS = 7
FEE = .05
UNKNOWN = -1
SOURCE = ((1, 2), (2, 4), (1, 3), (2, 5), (1, 6), (3, 4), (3, 5), (4, 6), (5, 6))
DEVELOPMENT = ((1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7))


@dataclass(frozen=True)
class State:
    deck: tuple[int, int]
    remaining: int
    observed: tuple[int, ...] = (UNKNOWN,) * len(PROBES)
    key: int = 0


def legal(state):
    if state.key:
        return BRANCHES
    return (*[i for i, bit in enumerate(state.observed) if state.remaining and bit == UNKNOWN], OPEN)


def observe(state, action, bit):
    observed = tuple(bit if index == action else old for index, old in enumerate(state.observed))
    return replace(state, observed=observed, remaining=state.remaining - 1)


def parity_paths(key):
    return tuple((i, j) for i, j in combinations(range(len(PROBES)), 2) if PROBES[i] ^ PROBES[j] == key)


class Support:
    """A complete source observation tree backed by native diagnostic outcomes."""

    def __init__(self, worlds):
        self.worlds = tuple(tuple(item["value"] for item in row["observations"]) for row in worlds)
        if len(set(self.worlds)) != 8 or any(len(bits) != 7 or set(bits) - {0, 1} for bits in self.worlds):
            raise ValueError("the registered native support worlds are incomplete")

    @lru_cache(maxsize=None)
    def possible(self, observed):
        return tuple(world for world in self.worlds if all(value in (UNKNOWN, world[key - 1])
                     for key, value in zip(PROBES, observed, strict=True)))

    def transition(self, state, action):
        if action == OPEN:
            return 0., tuple((.5, replace(state, key=key)) for key in state.deck)
        worlds = self.possible(state.observed)
        if action in BRANCHES:
            branch = action - BRANCHES[0]
            # The native decoder attains 1 for rollback and 1/2 for an unseen replacement.
            rolled_back = sum(bits[state.key - 1] == branch for bits in worlds)
            return (.5 + .5 * rolled_back / len(worlds)), ((1., None),)
        key = PROBES[action]
        branches = [sum(bits[key - 1] == bit for bits in worlds) / len(worlds) for bit in (0, 1)]
        return -FEE, tuple((prob, observe(state, action, bit)) for bit, prob in enumerate(branches) if prob)

    @lru_cache(maxsize=None)
    def exact(self, state):
        output = {}
        for action in legal(state):
            reward, edges = self.transition(state, action)
            output[action] = reward + sum(prob * max(self.exact(child).values())
                                          for prob, child in edges if child is not None)
        return output


def play(world, state, query, choose):
    if query not in state.deck or state.key:
        raise ValueError("a hidden query must be drawn from the advertised deck")
    trace = []
    while True:
        action = choose(state)
        if action not in legal(state):
            raise ValueError("the policy chose an unavailable native support action")
        if action == OPEN:
            state = replace(state, key=query)
        elif action in BRANCHES:
            return {"probes": trace, "branch": action - BRANCHES[0], "query": query}
        else:
            key = PROBES[action]
            bit = world["observations"][key - 1]["value"]
            trace.append((key, bit))
            state = observe(state, action, bit)
