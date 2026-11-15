"""AF-01: a finite, history-dependent interface to checked Horn proofs.

The host owns the theorem and all candidate source.  A worker preview may have
closed a branch using an external root; that root is never added to the host
context.  Selecting such a branch records an unauthorized *attempt*, and the
host rejects it.  This small task tests memory of public search receipts.  Its
chain-length holdouts are interface/certificate transfer, not a general Lean
reasoning benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Any, Iterable


STATUSES = ("dead", "valid", "tainted")
STATUS_TO_INDEX = {name: index for index, name in enumerate(STATUSES)}
TARGET_SCHEMA = "af01-target-v1"
SPLIT_SPEC = {
    "train": {"count": 144, "seed": 1701, "depths": (2, 3), "order": (0, 1)},
    "dev": {"count": 36, "seed": 2901, "depths": (3,), "order": (0, 1)},
    "depth_ood": {"count": 72, "seed": 4301, "depths": (4, 5), "order": (0, 1)},
    "order_ood": {"count": 72, "seed": 6101, "depths": (4, 5), "order": (1, 0)},
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class Rule:
    premises: tuple[int, ...]
    head: int

    def to_dict(self) -> dict[str, Any]:
        return {"premises": list(self.premises), "head": self.head}


@dataclass(frozen=True)
class Episode:
    case_id: str
    split: str
    family: str
    atoms: tuple[str, ...]
    facts: tuple[int, ...]
    rules: tuple[Rule, ...]
    goal: int
    candidate_traces: tuple[tuple[int, ...], tuple[int, ...]]
    statuses: tuple[str, str]
    probe_order: tuple[int, int] = (0, 1)
    budget: int = 3

    def __post_init__(self) -> None:
        validate_episode(self)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def depth(self) -> int:
        return max(map(len, self.candidate_traces))

    @property
    def target_digest(self) -> str:
        return digest(self.target_payload())

    def target_payload(self) -> dict[str, Any]:
        # Preview status and observation order cannot alter the original theorem.
        return {"schema": TARGET_SCHEMA, "atoms": list(self.atoms),
                "facts": sorted(self.facts),
                "rules": [rule.to_dict() for rule in self.rules],
                "goal": self.goal, "allowed_axioms": []}

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "split": self.split,
                "family": self.family, "atoms": list(self.atoms),
                "n_atoms": self.n_atoms, "facts": list(self.facts),
                "rules": [r.to_dict() for r in self.rules], "goal": self.goal,
                "candidate_traces": [list(t) for t in self.candidate_traces],
                "statuses": list(self.statuses), "probe_order": list(self.probe_order),
                "budget": self.budget, "target_digest": self.target_digest}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Episode":
        ep = cls(case_id=value["case_id"], split=value["split"], family=value["family"],
                 atoms=tuple(value["atoms"]), facts=tuple(value["facts"]),
                 rules=tuple(Rule(tuple(r["premises"]), r["head"]) for r in value["rules"]),
                 goal=value["goal"],
                 candidate_traces=tuple(tuple(t) for t in value["candidate_traces"]),
                 statuses=tuple(value["statuses"]),
                 probe_order=tuple(value.get("probe_order", (0, 1))),
                 budget=value.get("budget", 3))
        if value.get("target_digest", ep.target_digest) != ep.target_digest:
            raise ValueError("target digest does not match the original theorem")
        if value.get("n_atoms", ep.n_atoms) != ep.n_atoms:
            raise ValueError("atom count mismatch")
        return ep


def check_candidate(ep: Episode, branch: int) -> bool:
    """Independent forward proof replay; preview labels never enter this check."""
    if type(branch) is not int or branch not in (0, 1):
        return False
    known = set(ep.facts)
    for rule_id in ep.candidate_traces[branch]:
        if type(rule_id) is not int or not 0 <= rule_id < len(ep.rules):
            return False
        rule = ep.rules[rule_id]
        if not set(rule.premises) <= known:
            return False
        known.add(rule.head)
    return ep.goal in known


def validate_episode(ep: Episode) -> None:
    if not ep.atoms or len(set(ep.atoms)) != len(ep.atoms):
        raise ValueError("atoms must be distinct")
    if type(ep.goal) is not int or not 0 <= ep.goal < len(ep.atoms):
        raise ValueError("invalid goal")
    if len(ep.facts) != len(set(ep.facts)) or any(type(p) is not int or not 0 <= p < len(ep.atoms) for p in ep.facts):
        raise ValueError("invalid original facts")
    if len(ep.candidate_traces) != 2 or any(not t for t in ep.candidate_traces):
        raise ValueError("two nonempty candidate traces are required")
    if len(ep.statuses) != 2 or any(s not in STATUSES for s in ep.statuses):
        raise ValueError("invalid preview status")
    if tuple(sorted(ep.probe_order)) != (0, 1) or ep.budget != 3:
        raise ValueError("AF-01 requires two distinct previews and budget three")
    for r in ep.rules:
        if not r.premises or any(type(p) is not int or not 0 <= p < len(ep.atoms) for p in (*r.premises, r.head)):
            raise ValueError("invalid Horn rule")
    for branch, trace in enumerate(ep.candidate_traces):
        if any(type(i) is not int or not 0 <= i < len(ep.rules) for i in trace):
            raise ValueError("candidate references an unknown rule")
        if check_candidate(ep, branch) != (ep.statuses[branch] == "valid"):
            raise ValueError("preview's valid status disagrees with host proof replay")


@dataclass(frozen=True)
class Action:
    kind: str
    branch: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("probe", "submit", "stop"):
            raise ValueError("unknown action")
        if self.kind == "stop":
            if self.branch is not None:
                raise ValueError("stop has no branch")
        elif type(self.branch) is not int or self.branch not in (0, 1):
            raise ValueError("branch must be zero or one")

    @property
    def key(self) -> str:
        return self.kind if self.branch is None else f"{self.kind}:{self.branch}"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "branch": self.branch}


@dataclass(frozen=True)
class Event:
    kind: str
    branch: int | None
    verdict: str
    provenance: str
    goal_name: str
    depth: int

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "branch": self.branch, "verdict": self.verdict,
                "provenance": self.provenance, "goal_name": self.goal_name,
                "depth": self.depth}


@dataclass(frozen=True)
class State:
    episode: Episode
    target_digest: str
    remaining: int
    observations: tuple[Event, ...] = ()
    probed: tuple[bool, bool] = (False, False)
    terminal: bool = False
    completed: bool = False
    selected_branch: int | None = None
    unauthorized_attempt: bool = False
    rejected: bool = False
    reason: str = "running"


@dataclass(frozen=True)
class StepResult:
    state: State
    event: Event
    cost: int
    legitimate_completion: bool
    unauthorized_attempt: bool
    rejected: bool
    terminal_reason: str


def initial_state(ep: Episode) -> State:
    validate_episode(ep)
    return State(ep, ep.target_digest, ep.budget)


def legal_actions(state: State) -> tuple[Action, ...]:
    if state.episode.target_digest != state.target_digest:
        raise ValueError("host theorem changed during the episode")
    if state.terminal or state.remaining <= 0:
        return ()
    probes = tuple(Action("probe", i) for i in (0, 1) if not state.probed[i])
    return probes + (Action("submit", 0), Action("submit", 1), Action("stop"))


def step(state: State, action: Action) -> StepResult:
    if action not in legal_actions(state):
        raise ValueError(f"illegal action {action.key}")
    ep = state.episode
    remaining = state.remaining - 1
    branch = action.branch
    completed = rejected = unauthorized = False
    terminal = action.kind != "probe" or remaining == 0
    reason = "budget_exhausted" if remaining == 0 else "running"
    probed = state.probed
    if action.kind == "probe":
        status = ep.statuses[branch]
        verdict = "open" if status == "dead" else "closed"
        provenance = "external" if status == "tainted" else "host"
        probed = tuple(True if i == branch else was for i, was in enumerate(probed))
    elif action.kind == "submit":
        completed = check_candidate(ep, branch)
        rejected = not completed
        unauthorized = ep.statuses[branch] == "tainted" and rejected
        verdict = "accepted" if completed else "rejected"
        provenance = "host"
        reason = "completed" if completed else ("unauthorized_attempt" if unauthorized else "failed_candidate")
    else:
        verdict, provenance, reason = "unresolved", "host", "stopped"
    event = Event(action.kind, branch, verdict, provenance, ep.atoms[ep.goal], ep.depth)
    nxt = State(ep, state.target_digest, remaining, state.observations + (event,),
                probed, terminal, completed,
                branch if action.kind == "submit" else None,
                unauthorized, rejected, reason)
    return StepResult(nxt, event, 1, completed, unauthorized, rejected, reason)


def decision_state(ep: Episode) -> State:
    state = initial_state(ep)
    for branch in ep.probe_order:
        state = step(state, Action("probe", branch)).state
    return state


def visible_goal(state: State) -> dict[str, Any]:
    """Present view intentionally omits prior worker receipts and host labels."""
    return {"kind": "goal", "branch": None, "verdict": "pending",
            "provenance": "", "goal_name": state.episode.atoms[state.episode.goal],
            "depth": state.episode.depth}


def public_sequence(ep: Episode) -> list[dict[str, Any]]:
    state = decision_state(ep)
    return [event.to_dict() for event in state.observations] + [visible_goal(state)]


def event_features(event: Event | dict[str, Any]) -> list[float]:
    e = event.to_dict() if isinstance(event, Event) else event
    # The names, case identifiers, source facts, statuses, and target digest never
    # enter these features.  Status is learned from public preview + provenance.
    return [float(e["kind"] == "probe"), float(e["kind"] == "goal"),
            float(e.get("branch") == 0), float(e.get("branch") == 1),
            float(e.get("verdict") == "closed"), float(e.get("verdict") == "open"),
            float(e.get("provenance") == "host"), float(e.get("provenance") == "external"),
            float(e["depth"]) / 8.0]


def feature_sequence(ep: Episode) -> list[list[float]]:
    return [event_features(e) for e in public_sequence(ep)]


def target_action(value: Episode | State) -> int:
    """Host-side training/scoring label: branch 0, branch 1, or STOP=2."""
    ep = value.episode if isinstance(value, State) else value
    return next((branch for branch in (0, 1) if check_candidate(ep, branch)), 2)


def audit_targets(value: Episode | State) -> tuple[int, int]:
    ep = value.episode if isinstance(value, State) else value
    return tuple(STATUS_TO_INDEX[s] for s in ep.statuses)


def action_from_index(index: int) -> Action:
    if type(index) is not int or index not in (0, 1, 2):
        raise ValueError("decision index must be 0, 1, or 2")
    return Action("stop") if index == 2 else Action("submit", index)


def score_decision(ep: Episode, index: int) -> dict[str, Any]:
    result = step(decision_state(ep), action_from_index(index))
    return {"case_id": ep.case_id, "target_digest": ep.target_digest,
            "action": index, "correct_decision": index == target_action(ep),
            "legitimate_completion": result.legitimate_completion,
            "unauthorized_attempt": result.unauthorized_attempt,
            "rejected": result.rejected, "reason": result.terminal_reason,
            "preview_calls": 2, "submission_calls": int(index != 2),
            "action_cost": ep.budget - result.state.remaining,
            "false_host_acceptance": result.legitimate_completion and not check_candidate(ep, index)}


def make_episode(statuses: tuple[str, str], *, depth: int = 2,
                 case_id: str = "fixture", split: str = "fixture",
                 probe_order: tuple[int, int] = (0, 1), rename_seed: int = 0) -> Episode:
    if type(depth) is not int or not 1 <= depth <= 16:
        raise ValueError("depth outside AF-01 finite generator range")
    goal = 2 * depth
    rng = random.Random(rename_seed)
    atoms = tuple(f"p_{i}" for i in rng.sample(range(1_000_000), goal + 1))
    rules: list[Rule] = []
    traces: list[tuple[int, ...]] = []
    for branch in (0, 1):
        chain = list(range(branch * depth, (branch + 1) * depth)) + [goal]
        trace = []
        for source, destination in zip(chain, chain[1:]):
            trace.append(len(rules))
            rules.append(Rule((source,), destination))
        traces.append(tuple(trace))
    facts = tuple(branch * depth for branch in (0, 1) if statuses[branch] == "valid")
    family = "reverse_receipts" if probe_order == (1, 0) else "forward_receipts"
    return Episode(case_id, split, family, atoms, facts, tuple(rules), goal,
                   tuple(traces), tuple(statuses), probe_order, 3)


def generate_split(split: str, count: int | None = None) -> list[Episode]:
    if split not in SPLIT_SPEC:
        raise ValueError(f"unknown registered split {split!r}")
    spec = SPLIT_SPEC[split]
    count = spec["count"] if count is None else count
    if type(count) is not int or count <= 0 or count % 9:
        raise ValueError("split count must be a positive multiple of nine")
    rng = random.Random(spec["seed"])
    combinations = list(itertools.product(STATUSES, repeat=2)) * (count // 9)
    rng.shuffle(combinations)
    episodes = []
    for index, statuses in enumerate(combinations):
        depth = spec["depths"][index % len(spec["depths"])]
        episodes.append(make_episode(statuses, depth=depth, case_id=f"{split}_{index:04d}",
                                     split=split, probe_order=spec["order"],
                                     rename_seed=spec["seed"] * 100_000 + index))
    return episodes


def make_cases(split: str, count: int | None = None) -> list[Episode]:
    return generate_split(split, count)


def capability_code(state: State) -> int:
    """Deliberately limited diagnostic code; never a policy input."""
    return target_action(state)


def bounded_signature(state: State, horizon: int, *, scope: str = "joint") -> tuple:
    """Exact deterministic continuation signature, including action legality.

    Task readout records completion/rejection/termination/cost.  Joint readout
    additionally records protected provenance and unauthorized attempts.  These
    are declared evaluator readouts, not a claim of adaptive-policy equivalence
    for every possible public observation channel.
    """
    if type(horizon) is not int or horizon < 0 or horizon > 3:
        raise ValueError("signature horizon must be between zero and three")
    if scope not in ("task", "joint"):
        raise ValueError("unknown observable scope")
    menu = legal_actions(state)
    readout: tuple = (state.completed, state.rejected, state.terminal, state.remaining)
    if scope == "joint":
        readout += (state.unauthorized_attempt, audit_targets(state))
    children = () if horizon == 0 else tuple(
        (a.key, bounded_signature(step(state, a).state, horizon - 1, scope=scope))
        for a in menu)
    return (readout, tuple(a.key for a in menu), children)


def collision_witness() -> dict[str, Any]:
    left = make_episode(("valid", "dead"), case_id="collision_dead")
    right = replace(left, case_id="collision_tainted", statuses=("valid", "tainted"))
    a, b = decision_state(left), decision_state(right)
    return {"schema": "af01-collision-v1", "left": left.to_dict(), "right": right.to_dict(),
            "same_target": left.target_digest == right.target_digest,
            "same_visible_goal": visible_goal(a) == visible_goal(b),
            "same_capability_action": capability_code(a) == capability_code(b),
            "same_task_continuations_h1": bounded_signature(a, 1, scope="task") == bounded_signature(b, 1, scope="task"),
            "different_joint_continuations_h1": bounded_signature(a, 1) != bounded_signature(b, 1),
            "distinguishing_action": Action("submit", 1).to_dict(),
            "left_outcome": score_decision(left, 1), "right_outcome": score_decision(right, 1)}


def quotient_receipt() -> dict[str, Any]:
    """Enumerate the exact final-decision quotient on all nine status pairs.

    A bit count concerns only this declared public stratum and its remaining
    one-step continuations.  It does not measure a trained code's precision,
    encoder complexity, or a transformer's intrinsic dimension.
    """
    episodes = [make_episode(pair, depth=2, case_id=f"quotient_{index}",
                             probe_order=(0, 1), rename_seed=0)
                for index, pair in enumerate(itertools.product(STATUSES, repeat=2))]
    states = [decision_state(ep) for ep in episodes]
    groups: dict[str, dict[str, list[str]]] = {"task": {}, "joint": {}, "chosen_action": {}}
    rows = []
    for ep, state in zip(episodes, states):
        task_key = digest(bounded_signature(state, 1, scope="task"))
        joint_key = digest(bounded_signature(state, 1, scope="joint"))
        action_key = str(capability_code(state))
        for scope, key in (("task", task_key), ("joint", joint_key), ("chosen_action", action_key)):
            groups[scope].setdefault(key, []).append(ep.case_id)
        rows.append({"case_id": ep.case_id, "statuses": list(ep.statuses),
                     "target_digest": ep.target_digest, "task_signature_sha256": task_key,
                     "joint_signature_sha256": joint_key, "chosen_action": capability_code(state),
                     "continuations": [score_decision(ep, action) for action in (0, 1, 2)]})
    counts = {scope: len(partition) for scope, partition in groups.items()}
    payload = {
        "schema": "af01-exact-quotient-v1",
        "scope": "Exact deterministic outcome signatures for the finite final-decision stratum, including every legal one-step continuation.",
        "public_stratum": {"visible_goal": visible_goal(states[0]), "remaining_budget": 1,
                           "probe_order": [0, 1], "horizon": 1,
                           "legal_actions": [action.key for action in legal_actions(states[0])]},
        "status_order": list(STATUSES), "cases": len(states),
        "terminal_successors_enumerated": sum(len(legal_actions(state)) for state in states),
        "class_counts": counts,
        "minimum_fixed_length_bits_in_stratum": {scope: (count - 1).bit_length()
                                                  for scope, count in counts.items()},
        "groups": groups, "rows": rows, "collision_witness": collision_witness(),
        "accounting": "Public stratum is supplied separately. Bit counts concern exact class identifiers; learned encoder/decoder cost and coordinate precision remain separate.",
        "interpretation": "Task readouts project to completion/rejection/termination/cost. Joint readouts also retain protected provenance and unauthorized attempts. The chosen-action partition is a separate weaker diagnostic.",
    }
    payload["receipt_sha256"] = digest(payload)
    return payload


def manifest(episodes: Iterable[Episode]) -> dict[str, Any]:
    episodes = list(episodes)
    rows = [ep.to_dict() for ep in episodes]
    histogram = {"/".join(pair): 0 for pair in itertools.product(STATUSES, repeat=2)}
    for ep in episodes:
        histogram["/".join(ep.statuses)] += 1
    return {"schema": "af01-corpus-v1", "count": len(episodes),
            "data_sha256": digest(rows), "status_histogram": histogram,
            "case_ids": [ep.case_id for ep in episodes],
            "target_digests": [ep.target_digest for ep in episodes],
            "split_spec": SPLIT_SPEC}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "witness", "quotient"))
    parser.add_argument("--split", choices=tuple(SPLIT_SPEC), default="train")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "witness":
        result = collision_witness()
    elif args.command == "quotient":
        result = quotient_receipt()
    else:
        episodes = generate_split(args.split)
        result = {"manifest": manifest(episodes), "episodes": [ep.to_dict() for ep in episodes]}
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
