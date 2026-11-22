"""Sparse, budgeted episodes over independently audited MathCan transitions."""

from dataclasses import dataclass
from functools import lru_cache

from .data import digest
from .live_lean import MAX_MENU


CHECKS = 8
CHECK_COST = .05
SCHEMA = "ppt.mathcan-sparse-proof-search.v1"


@dataclass(frozen=True)
class Frame:
    key: tuple
    left: int
    history: tuple = ()


class RecordedSearch:
    """Expose a tactic's outcome only after that tactic has been issued."""

    def __init__(self, archive):
        self.archive = archive

    def menu(self, frame):
        rejected = {action for key, action, status in frame.history if key == frame.key and status == "rejected"}
        return tuple(action for action in sorted(self.archive.states[frame.key])[:MAX_MENU] if action not in rejected)

    def step(self, frame, action):
        if frame.left <= 0 or action not in self.menu(frame):
            raise ValueError("the selected tactic is unavailable in the recorded proof state")
        status, successor = self.archive.step(frame.key, action)
        row = self.archive.states[frame.key][action]
        after = None
        if status in {"rejected", "accepted_nonterminal"} and frame.left > 1:
            after = Frame(successor, frame.left - 1, (*frame.history, (frame.key, action, status)))
            if not self.menu(after):
                after = None
        return {"status": status, "after": after, "state_sha256": row["state_sha256"],
                "source_sha256": row["source_sha256"]}

    @lru_cache(maxsize=None)
    def shortest(self, key, budget):
        """Find the shortest recorded closure, leaving missing transitions unknown."""
        if budget == 0:
            return None
        lengths = []
        for action in sorted(self.archive.states[key])[:MAX_MENU]:
            status, child = self.archive.step(key, action)
            if status == "archived_closure":
                lengths.append(1)
            elif status == "accepted_nonterminal":
                steps = self.shortest(child, budget - 1)
                if steps is not None:
                    lengths.append(steps + 1)
        return min(lengths, default=None)


def split_starts(archive):
    """Group the original TRAIN archive by source module before fitting."""
    modules = sorted({key[0] for key in archive.starts}, key=lambda name: digest([SCHEMA, name]))
    dev_modules = set(modules[::3])
    return {split: [key for key in archive.starts if (key[0] in dev_modules) == (split == "dev")]
            for split in ("train", "dev")}


def outcome(steps):
    last = steps[-1]["status"]
    verified = last == "archived_closure"
    unknown = last == "censored"
    return {"proof": verified, "censored": unknown, "checks": len(steps),
            "observed_return": None if unknown else float(verified) - CHECK_COST * len(steps)}
