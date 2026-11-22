"""Two Lean worlds whose first proof determines a shared existential witness."""

from .data import digest
from .hybrid_rl.corpus import SearchCase


FAMILY = "bpd-dependent-witness"
PREFIX = ("apply Exists.intro", "constructor")
FORK = ("exact h0", "exact h1")


def examples():
    cluster = digest(FAMILY)
    for winner in range(2):
        binders = ("(P Q : Nat → Prop) (h0 : P 0) (h1 : P 1) "
                   f"(h2 : Q {winner}) (h3 : Q {winner} → Q {1 - winner})")
        case = SearchCase(FAMILY, "train", FAMILY, digest([cluster, winner]), cluster,
                          f"bpd_shared_witness_{winner}", binders, "∃ n, P n ∧ Q n")
        yield case, winner


def script(winner, branch):
    close = ("exact h2",) if branch == winner else ("apply h3", "exact h2")
    return (*PREFIX, FORK[branch], *close)


def extra_tactic(action):
    return action in (PREFIX[0], *FORK, "exact h2")
