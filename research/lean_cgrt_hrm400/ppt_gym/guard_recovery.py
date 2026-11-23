"""Budgeted native guard checks with optional saved-state recovery."""

import random
from typing import NamedTuple

from .computational_guards import FAMILIES, equation
from .data import digest


SCHEMA = "ppt.native-guard-recovery.v1"
ALLOWANCES = (6, 7, 8)
CHARGE = .025
ENTER = ("enter_0", "enter_1")
PROBE = tuple(f"probe_{i}" for i in range(4))
VERIFY = ("verify_0", "verify_1")
ACTIONS = (*ENTER, *PROBE, *VERIFY, "close")


def cases(partition, count):
    for index in range(count):
        family = FAMILIES[index % len(FAMILIES)]
        rng = random.Random(int(digest([SCHEMA, partition, index])[:15], 16))
        definite = [equation(family, rng, partition) for _ in range(2)]
        failed = rng.randrange(2)
        false = equation(family, rng, partition)
        while false.value == definite[failed].value:
            false = equation(family, rng, partition)
        productive = [e.claim() for e in definite]
        other = [equation(family, rng, partition).claim() for _ in range(2)]
        other[failed] = false.claim(definite[failed].value)
        if rng.getrandbits(1):
            productive.reverse()
            other.reverse()
            failed = 1 - failed
        witnesses = rng.sample(range(2, 50), 2)
        pair = digest([SCHEMA, family, sorted((productive, other)), witnesses])
        for world in range(2):
            guards = (productive, other) if world == 0 else (other, productive)
            binders = ["(P Q : Nat → Prop)"]
            for branch, number in enumerate(witnesses):
                premises = " → ".join(f"({claim})" for claim in guards[branch])
                binders.append(f"(h{branch} : {premises} → P {number})")
                binders.append(f"(h{branch + 2} : Q {number})")
            statement = f"∀ {' '.join(binders)}, ∃ n : Nat, P n ∧ Q n"
            yield {"module": digest([pair, world]), "pair": pair, "family": family,
                   "statement": statement, "world": world, "guards": guards,
                   "false_guard": 2 * (1 - world) + failed}


class State(NamedTuple):
    remaining: int
    known: tuple
    branch: int = -1
    checked: int = 0
    failed: int = 0


def initial(allowance):
    return State(allowance - 2, (-1,) * 4)


def menu(state):
    if not state.remaining:
        return ()
    if state.branch < 0:
        entries = [action for branch, action in enumerate(ENTER) if not state.failed & (1 << branch)]
        probes = [action for index, action in enumerate(PROBE) if state.known[index] < 0]
        return (*entries, *probes)
    if state.checked == 3:
        return ("close",)
    return tuple(action for slot, action in enumerate(VERIFY) if not state.checked & (1 << slot))


def transition(state, action, truth):
    """The policy sees only the observation returned by its chosen operation."""
    if action not in menu(state):
        raise ValueError("an action left the currently affordable native menu")
    left = state.remaining - 1
    if action == "close":
        return None, True
    if action in ENTER:
        return state._replace(remaining=left, branch=ENTER.index(action), checked=0), None
    index = PROBE.index(action) if action in PROBE else 2 * state.branch + VERIFY.index(action)
    bit = int(truth[index])
    known = (*state.known[:index], bit, *state.known[index + 1:])
    failed = state.failed | (0 if bit else 1 << (index // 2))
    if action in PROBE:
        return State(left, known, failed=failed), bit
    if not bit:
        return State(left, known, failed=failed), bit
    return State(left, known, state.branch, state.checked | (1 << VERIFY.index(action)), failed), bit
