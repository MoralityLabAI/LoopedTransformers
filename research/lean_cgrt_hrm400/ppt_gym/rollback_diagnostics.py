"""Replay a native diagnostic through its public observations and check budget."""

from copy import deepcopy
from dataclasses import dataclass


SCHEMA = "ppt.rollback-diagnostics.v1"
NAMES = ("probeLeft", "probeRight")
PROBES = tuple(f"have hProbe : {name} = 2 := by decide" for name in NAMES)
DIAGNOSTIC_COST = .05


@dataclass(frozen=True)
class Observation:
    proof_state: dict
    checks_remaining: int
    available_probes: tuple[int, ...]
    latest_diagnostic: tuple[int, int] | None


@dataclass(frozen=True)
class Outcome:
    certificate_passed: bool
    tactic_checks: int
    diagnostics: int
    reward: float


class DiagnosticEpisode:
    """Supply the latest rejection only, leaving earlier evidence to the policy."""

    def __init__(self, row, diagnostic_budget=2):
        if diagnostic_budget not in (0, 1, 2):
            raise ValueError("the two registered probes allow zero, one or two diagnostic checks")
        self._row = row
        self._proof_limit = row["proof_budget"] - 2
        self._budget = self._proof_limit + diagnostic_budget
        self._remaining = diagnostic_budget
        self._available = (0, 1)
        self._latest = None
        self._spent = 0
        self._terminal = False

    def observe(self):
        if self._terminal:
            raise ValueError("a committed witness has no further proof decisions")
        return Observation(deepcopy(self._row["public"]), self._budget - self._spent,
                           self._available if self._remaining else (), self._latest)

    def diagnose(self, index):
        if index not in self.observe().available_probes:
            raise ValueError("the diagnostic is repeated or would spend the proof allowance")
        receipt = self._row["probe_receipts"][index]
        if receipt["action"] != PROBES[index]:
            raise ValueError("the selected diagnostic differs from the verified Lean tactic")
        self._available = tuple(i for i in self._available if i != index)
        self._latest = (index, receipt["observed_value"])
        self._spent += 1
        self._remaining -= 1
        return self.observe()

    def commit(self, witness):
        if self._terminal or witness not in (0, 1):
            raise ValueError("expected one commit to a registered witness")
        self._terminal = True
        trial = self._row["witness_trials"][witness]
        checks = self._spent + trial["checks"]
        passed = trial["passed"] and checks <= self._budget
        return Outcome(passed, checks, self._spent, float(passed) - self._spent * DIAGNOSTIC_COST)
