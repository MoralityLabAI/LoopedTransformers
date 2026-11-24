"""Finite powerset lattices and deterministic, uniform-cost cache transitions."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExpertLattice:
    """Nodes are candidate sets; order is inclusion, meet is ∩ and join is ∪."""

    experts: int = 128

    @property
    def top(self):
        return frozenset(range(self.experts))

    @property
    def bottom(self):
        return frozenset()

    def meet(self, left, right):
        return left & right

    def join(self, left, right):
        return left | right

    def admit(self, scores, old, capacity, budget):
        """Maximize additive soft evidence over feasible fixed-cardinality nodes.

        Each layer has uniform bytes per slot. Keep the best incumbents and
        pair the best outsiders with the worst incumbents in decreasing gain
        order. Stop at zero gain, preferring fewer transfers on a tie.
        """
        scores = np.asarray(scores, dtype=np.float64)
        if (scores.shape != (self.experts,) or not np.isfinite(scores).all()
                or not 1 <= capacity <= self.experts or not 0 <= budget <= capacity):
            raise ValueError("invalid admission state")
        order = np.argsort(-scores, kind="stable").tolist()
        if old is None:
            return frozenset(order[:capacity])
        if len(old) != capacity or any(not 0 <= e < self.experts for e in old):
            raise ValueError("invalid previous set")
        if budget == 0:
            return frozenset(old)
        inside = [e for e in order if e in old]
        outside = [e for e in order if e not in old]
        admitted, evicted = [], []
        for add, remove in zip(outside[:budget], reversed(inside)):
            if scores[add] <= scores[remove]:
                break
            admitted.append(add)
            evicted.append(remove)
        return frozenset(old.difference(evicted).union(admitted))
