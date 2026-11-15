"""AF-02 finite addressed histories and immutable original-host proof tasks.

The host emits a registered sequence of two previews, waits, and one goal
marker. A learned controller chooses a branch or STOP only at the final marker.
WAIT consumes budget without changing any theorem, receipt, or branch status.
Neural inputs are exact public F9 histories; renamed certificates are separate
host records with identical neural inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from experiments.activation_funnel import gym as af01


# Enumeration follows AF-02's registration. Head labels retain AF-01's order.
STATUSES = ("valid", "dead", "tainted")
AUDIT_STATUSES = ("dead", "valid", "tainted")
STATUS_TO_INDEX = {status: index for index, status in enumerate(AUDIT_STATUSES)}
ORDERS = ((0, 1), (1, 0))
RENAMINGS = 2
CONTEXT_SCHEMA = "af02-context-v1"
REGISTERED_PROTOCOL_SHA256 = "f6c3a939a09183c265466d506de985ec1a92c76d2d57f50cdd5f3f984e224c41"
SPLIT_SPEC = {
    "train": {"depths": (2, 3), "schedules": ("PPG", "PWPG", "PPWG"),
              "unique_inputs": 108, "hosts": 216, "seed": 1702},
    "dev": {"depths": (2, 3), "schedules": ("WPPG",),
            "unique_inputs": 36, "hosts": 72, "seed": 2902},
    "depth_context": {"depths": (4, 5), "schedules": ("PPG", "PWPG", "PPWG"),
                      "unique_inputs": 108, "hosts": 216, "seed": 4302},
    "delay2": {"depths": (2, 3), "schedules": ("PPWWG",),
               "unique_inputs": 36, "hosts": 72, "seed": 6102},
    "delay4": {"depths": (2, 3), "schedules": ("PPWWWWG",),
               "unique_inputs": 36, "hosts": 72, "seed": 7902},
}
SCHEDULES = tuple(dict.fromkeys(s for spec in SPLIT_SPEC.values() for s in spec["schedules"]))


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class Case:
    """Original AF-01 host plus a separately bound AF-02 observation context."""
    host: af01.Episode
    schedule: str
    canonical_id: str
    rename_index: int = 0
    context_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.host, af01.Episode):
            raise ValueError("AF-02 requires an immutable AF-01 host")
        af01.validate_episode(self.host)
        if self.schedule not in SCHEDULES:
            raise ValueError("unknown registered AF-02 schedule")
        if not isinstance(self.canonical_id, str) or not self.canonical_id:
            raise ValueError("canonical identifier must be nonempty")
        if type(self.rename_index) is not int or self.rename_index not in range(RENAMINGS):
            raise ValueError("rename index must be zero or one")
        object.__setattr__(self, "context_digest", digest(self.context_payload()))

    @property
    def case_id(self) -> str:
        return self.host.case_id

    @property
    def split(self) -> str:
        return self.host.split

    @property
    def family(self) -> str:
        return f"{self.split}:{self.schedule}"

    @property
    def depth(self) -> int:
        return self.host.depth

    @property
    def probe_order(self) -> tuple[int, int]:
        return self.host.probe_order

    @property
    def statuses(self) -> tuple[str, str]:
        return self.host.statuses

    @property
    def target_digest(self) -> str:
        return self.host.target_digest

    @property
    def budget(self) -> int:
        return 3 + self.schedule.count("W")

    def context_payload(self) -> dict[str, Any]:
        return {"schema": CONTEXT_SCHEMA, "case_id": self.case_id,
                "split": self.split, "canonical_id": self.canonical_id,
                "rename_index": self.rename_index, "target_digest": self.target_digest,
                "candidate_traces": [list(t) for t in self.host.candidate_traces],
                "statuses": list(self.statuses), "probe_order": list(self.probe_order),
                "depth": self.depth, "schedule": self.schedule, "budget": self.budget}

    def validate(self) -> None:
        af01.validate_episode(self.host)
        if digest(self.context_payload()) != self.context_digest:
            raise ValueError("AF-02 context changed after freezing")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = self.host.to_dict()
        data.update({"schema": "af02-case-v1", "family": self.family,
                     "schedule": self.schedule, "depth": self.depth,
                     "budget": self.budget, "canonical_id": self.canonical_id,
                     "rename_index": self.rename_index, "context_digest": self.context_digest})
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Case":
        if data.get("schema") != "af02-case-v1":
            raise ValueError("unknown AF-02 case schema")
        # The reused host schema's original two-preview budget stays internal;
        # AF-02's actual wait budget is checked separately and context-bound.
        host_data = dict(data)
        host_data["budget"] = 3
        host_data["family"] = "reverse_receipts" if tuple(data["probe_order"]) == (1, 0) else "forward_receipts"
        host = af01.Episode.from_dict(host_data)
        case = cls(host, data["schedule"], data["canonical_id"], data["rename_index"])
        if data.get("budget") != case.budget or data.get("depth") != case.depth:
            raise ValueError("AF-02 depth or budget mismatch")
        if data.get("context_digest") != case.context_digest:
            raise ValueError("AF-02 context digest mismatch")
        if data.get("family") != case.family:
            raise ValueError("AF-02 family mismatch")
        return case


def make_case(statuses: tuple[str, str], *, depth: int = 2, schedule: str = "PPG",
              probe_order: tuple[int, int] = (0, 1), case_id: str = "fixture",
              split: str = "fixture", rename_seed: int = 0,
              canonical_id: str | None = None, rename_index: int = 0) -> Case:
    host = af01.make_episode(statuses, depth=depth, probe_order=probe_order,
                             case_id=case_id, split=split, rename_seed=rename_seed)
    return Case(host, schedule, case_id if canonical_id is None else canonical_id, rename_index)


def generate_split(split: str) -> list[Case]:
    if split not in SPLIT_SPEC:
        raise ValueError("unknown registered AF-02 split")
    spec = SPLIT_SPEC[split]
    cases = []
    grid = itertools.product(spec["depths"], spec["schedules"], ORDERS,
                             STATUSES, STATUSES)
    for index, (depth, schedule, order, status0, status1) in enumerate(grid):
        canonical_id = f"af02_{split}_c{index:04d}"
        for rename in range(RENAMINGS):
            cases.append(make_case((status0, status1), depth=depth, schedule=schedule,
                                   probe_order=order, case_id=f"{canonical_id}_r{rename}",
                                   split=split, rename_seed=spec["seed"] * 100000 + index * 2 + rename,
                                   canonical_id=canonical_id, rename_index=rename))
    if len(cases) != spec["hosts"]:
        raise ValueError("registered corpus size mismatch")
    return cases


def canonical_cases(split: str) -> list[Case]:
    return [case for case in generate_split(split) if case.rename_index == 0]


@dataclass(frozen=True)
class Action:
    kind: str
    branch: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("probe", "wait", "submit", "stop"):
            raise ValueError("unknown action")
        if self.kind in ("wait", "stop"):
            if self.branch is not None:
                raise ValueError("wait and stop have no branch")
        elif type(self.branch) is not int or self.branch not in (0, 1):
            raise ValueError("branch must be zero or one")

    @property
    def key(self) -> str:
        return self.kind if self.branch is None else f"{self.kind}:{self.branch}"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "branch": self.branch}


@dataclass(frozen=True)
class State:
    case: Case
    target_digest: str
    context_digest: str
    remaining: int
    cursor: int = 0
    observations: tuple[af01.Event, ...] = ()
    terminal: bool = False
    completed: bool = False
    rejected: bool = False
    unauthorized_attempt: bool = False
    selected_branch: int | None = None
    reason: str = "running"


@dataclass(frozen=True)
class StepResult:
    state: State
    event: af01.Event
    cost: int


def initial_state(case: Case) -> State:
    case.validate()
    return State(case, case.target_digest, case.context_digest, case.budget)


def _scheduled_action(state: State) -> Action | None:
    symbol = state.case.schedule[state.cursor]
    if symbol == "G":
        return None
    if symbol == "W":
        return Action("wait")
    preview_index = state.case.schedule[:state.cursor].count("P")
    return Action("probe", state.case.probe_order[preview_index])


def legal_actions(state: State) -> tuple[Action, ...]:
    state.case.validate()
    if state.case.target_digest != state.target_digest:
        raise ValueError("original host theorem changed during the episode")
    if state.case.context_digest != state.context_digest:
        raise ValueError("AF-02 context changed during the episode")
    if state.terminal or state.remaining <= 0:
        return ()
    if not 0 <= state.cursor < len(state.case.schedule):
        raise ValueError("invalid script cursor")
    scripted = _scheduled_action(state)
    prefix = () if scripted is None else (scripted,)
    return prefix + (Action("submit", 0), Action("submit", 1), Action("stop"))


def check_candidate(case: Case, branch: int) -> bool:
    return af01.check_candidate(case.host, branch)


def step(state: State, action: Action) -> StepResult:
    if action not in legal_actions(state):
        raise ValueError(f"illegal action {action.key}")
    case, branch = state.case, action.branch
    remaining = state.remaining - 1
    completed = rejected = unauthorized = False
    terminal = action.kind in ("submit", "stop") or remaining == 0
    cursor = state.cursor
    reason = "budget_exhausted" if remaining == 0 else "running"
    if action.kind == "probe":
        status = case.statuses[branch]
        verdict = "open" if status == "dead" else "closed"
        provenance = "external" if status == "tainted" else "host"
        cursor += 1
    elif action.kind == "wait":
        verdict, provenance = "", ""
        cursor += 1
    elif action.kind == "submit":
        completed = check_candidate(case, branch)
        rejected = not completed
        unauthorized = rejected and case.statuses[branch] == "tainted"
        verdict, provenance = ("accepted" if completed else "rejected"), "host"
        reason = "completed" if completed else ("unauthorized_attempt" if unauthorized else "failed_candidate")
    else:
        verdict, provenance, reason = "unresolved", "host", "stopped"
    event = af01.Event(action.kind, branch, verdict, provenance,
                       case.host.atoms[case.host.goal], case.depth)
    next_state = State(case, state.target_digest, state.context_digest, remaining, cursor,
                       state.observations + (event,), terminal, completed, rejected,
                       unauthorized, branch if action.kind == "submit" else None, reason)
    return StepResult(next_state, event, 1)


def decision_state(case: Case) -> State:
    state = initial_state(case)
    while case.schedule[state.cursor] != "G":
        state = step(state, _scheduled_action(state)).state
    if state.remaining != 1 or state.terminal:
        raise ValueError("registered script did not reach the one-action decision boundary")
    return state


def visible_goal(state: State) -> dict[str, Any]:
    return {"kind": "goal", "branch": None, "verdict": "pending", "provenance": "",
            "goal_name": state.case.host.atoms[state.case.host.goal], "depth": state.case.depth}


def public_sequence(case: Case) -> list[dict[str, Any]]:
    state = decision_state(case)
    return [event.to_dict() for event in state.observations] + [visible_goal(state)]


def feature_sequence(case: Case) -> list[list[float]]:
    return [af01.event_features(event) for event in public_sequence(case)]


def public_input_hash(case: Case) -> str:
    return digest(feature_sequence(case))


def erased_feature_sequence(case: Case) -> list[list[float]]:
    sequence = feature_sequence(case)
    for event in sequence:
        event[2:4] = [0.0, 0.0]
    return sequence


def target_action(value: Case | State) -> int:
    case = value.case if isinstance(value, State) else value
    return next((branch for branch in (0, 1) if check_candidate(case, branch)), 2)


def audit_targets(value: Case | State) -> tuple[int, int]:
    case = value.case if isinstance(value, State) else value
    return tuple(STATUS_TO_INDEX[status] for status in case.statuses)


def action_from_index(index: int) -> Action:
    if type(index) is not int or index not in (0, 1, 2):
        raise ValueError("decision index must be zero, one, or two")
    return Action("stop") if index == 2 else Action("submit", index)


def score_decision(case: Case, index: int) -> dict[str, Any]:
    state = step(decision_state(case), action_from_index(index)).state
    return {"case_id": case.case_id, "canonical_id": case.canonical_id,
            "target_digest": case.target_digest, "context_digest": case.context_digest,
            "action": index, "correct_decision": index == target_action(case),
            "legitimate_completion": state.completed, "unauthorized_attempt": state.unauthorized_attempt,
            "rejected": state.rejected, "reason": state.reason,
            "preview_calls": 2, "wait_calls": case.schedule.count("W"),
            "submission_calls": int(index != 2), "initial_budget": case.budget,
            "action_cost": case.budget - state.remaining, "remaining_budget": state.remaining,
            "false_host_acceptance": state.completed and not check_candidate(case, index)}


def first_probe_cut(case: Case) -> int:
    """Exclusive event index after the first preview's two recurrent visits."""
    return case.schedule.index("P") + 1


def donor_stratum(case: Case) -> tuple[int, int, str]:
    # graph_layout is fixed to canonical for these semantic records; layout
    # permutations are a separate consistent-transport condition.
    return case.probe_order[0], case.depth, case.schedule


def _canonical_lookup(cases: Sequence[Case]) -> dict[str, int]:
    lookup = {}
    for index, case in enumerate(cases):
        key = public_input_hash(case)
        if key in lookup:
            raise ValueError("pair construction requires unique canonical public inputs")
        lookup[key] = index
    return lookup


@dataclass(frozen=True)
class DonorPair:
    pair_id: str
    receiver_index: int
    donor_index: int
    hybrid_index: int
    receiver_case_id: str
    donor_case_id: str
    hybrid_case_id: str
    first_probed_branch: int
    cut_event: int
    stratum: tuple[int, int, str]
    receiver_prefix_sha256: str
    donor_prefix_sha256: str
    composed_input_sha256: str
    changed_prefix_status: bool
    expected_action: int
    audit_target: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def donor_pairs(canonical: Sequence[Case]) -> list[DonorPair]:
    lookup = _canonical_lookup(canonical)
    groups: dict[tuple[int, int, str], dict[str, int]] = defaultdict(dict)
    sequences = [feature_sequence(case) for case in canonical]
    for index, case in enumerate(canonical):
        prefix_hash = digest(sequences[index][:first_probe_cut(case)])
        groups[donor_stratum(case)].setdefault(prefix_hash, index)
    if any(len(prefixes) != 3 for prefixes in groups.values()):
        raise ValueError("each donor stratum must contain all three observed prefixes")
    pairs = []
    for receiver_index, receiver in enumerate(canonical):
        cut = first_probe_cut(receiver)
        first = receiver.probe_order[0]
        recipient_audit = audit_targets(receiver)
        receiver_prefix_hash = digest(sequences[receiver_index][:cut])
        for donor_prefix_hash, donor_index in groups[donor_stratum(receiver)].items():
            donor = canonical[donor_index]
            composite = list(recipient_audit)
            composite[first] = audit_targets(donor)[first]
            composite = tuple(composite)
            hybrid = sequences[donor_index][:cut] + sequences[receiver_index][cut:]
            hybrid_hash = digest(hybrid)
            if hybrid_hash not in lookup:
                raise ValueError("composed public history is absent from the canonical corpus")
            hybrid_index = lookup[hybrid_hash]
            expected_action = next((b for b in (0, 1) if composite[b] == STATUS_TO_INDEX["valid"]), 2)
            if (audit_targets(canonical[hybrid_index]) != composite or
                    target_action(canonical[hybrid_index]) != expected_action):
                raise ValueError("composed history and independent host oracle disagree")
            pair_id = "af02_cf_" + digest([receiver.case_id, donor_prefix_hash])[:24]
            pairs.append(DonorPair(pair_id, receiver_index, donor_index, hybrid_index,
                                   receiver.case_id, donor.case_id, canonical[hybrid_index].case_id,
                                   first, cut, donor_stratum(receiver), receiver_prefix_hash,
                                   donor_prefix_hash, hybrid_hash,
                                   recipient_audit[first] != composite[first], expected_action, composite))
    return pairs


@dataclass(frozen=True)
class BindingPair:
    pair_id: str
    left_index: int
    right_index: int
    left_case_id: str
    right_case_id: str
    task_nontrivial: bool
    audit_nontrivial: bool
    expected_left_action: int
    expected_right_action: int
    expected_left_audit: tuple[int, int]
    expected_right_audit: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def binding_pairs(canonical: Sequence[Case]) -> list[BindingPair]:
    _canonical_lookup(canonical)
    lookup = {(case.depth, case.schedule, case.probe_order, case.statuses): index
              for index, case in enumerate(canonical)}
    if len(lookup) != len(canonical):
        raise ValueError("binding corpus contains repeated semantic contexts")
    pairs = []
    for left_index, left in enumerate(canonical):
        if left.probe_order != (0, 1):
            continue
        key = (left.depth, left.schedule, (1, 0), tuple(reversed(left.statuses)))
        if key not in lookup:
            raise ValueError("binding companion is absent from the canonical corpus")
        right_index = lookup[key]
        right = canonical[right_index]
        if erased_feature_sequence(left) != erased_feature_sequence(right):
            raise ValueError("binding companions changed more than receipt addresses")
        pairs.append(BindingPair("af02_bind_" + digest([left.case_id, right.case_id])[:24],
                                 left_index, right_index, left.case_id, right.case_id,
                                 left.statuses.count("valid") == 1, left.statuses[0] != left.statuses[1],
                                 target_action(left), target_action(right),
                                 audit_targets(left), audit_targets(right)))
    return pairs


@dataclass(frozen=True)
class GraphPermutation:
    """Map logical atoms (A0,A1,G) and rules (R0,R1) to physical slots."""
    atom_slots: tuple[int, int, int] = (0, 1, 2)
    rule_slots: tuple[int, int] = (0, 1)

    def __post_init__(self) -> None:
        if (type(self.atom_slots) is not tuple or type(self.rule_slots) is not tuple or
                any(type(i) is not int for i in (*self.atom_slots, *self.rule_slots)) or
                tuple(sorted(self.atom_slots)) != (0, 1, 2) or
                tuple(sorted(self.rule_slots)) != (0, 1)):
            raise ValueError("invalid physical graph permutation")

    @property
    def name(self) -> str:
        return "a" + "".join(map(str, self.atom_slots)) + "_r" + "".join(map(str, self.rule_slots))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "atom_slots": list(self.atom_slots), "rule_slots": list(self.rule_slots)}


def graph_permutations() -> list[GraphPermutation]:
    return [GraphPermutation(a, r) for a in itertools.permutations(range(3))
            for r in itertools.permutations(range(2))]


def address_erasure_receipt() -> dict[str, Any]:
    """Exact Bayes ceilings for deterministic address-blind readouts."""
    cases = [make_case((s0, s1), probe_order=order, case_id=f"erasure_{index:02d}")
             for index, (order, s0, s1) in enumerate(itertools.product(ORDERS, STATUSES, STATUSES))]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        groups[digest(erased_feature_sequence(case))].append(index)
    task_best = audit_best = positive_best = positive_count = 0
    rows = []
    for key, indices in sorted(groups.items()):
        action_counts = Counter(target_action(cases[i]) for i in indices)
        audit_counts = Counter(audit_targets(cases[i]) for i in indices)
        positive = [i for i in indices if cases[i].statuses.count("valid") == 1]
        task_best += max(action_counts.values())
        audit_best += max(audit_counts.values())
        positive_count += len(positive)
        positive_best += max(Counter(target_action(cases[i]) for i in positive).values(), default=0)
        rows.append({"erased_input_sha256": key, "case_ids": [cases[i].case_id for i in indices],
                     "task_counts": {str(k): v for k, v in sorted(action_counts.items())},
                     "audit_counts": {",".join(map(str, k)): v for k, v in sorted(audit_counts.items())}})
    pairs = binding_pairs(cases)
    task_pairs = [pair for pair in pairs if pair.task_nontrivial]
    audit_pairs = [pair for pair in pairs if pair.audit_nontrivial]
    task_pair_best = sum(pair.expected_left_action == pair.expected_right_action for pair in task_pairs)
    audit_pair_best = sum(pair.expected_left_audit == pair.expected_right_audit for pair in audit_pairs)
    def rate(correct: int, count: int) -> dict[str, Any]:
        return {"correct": correct, "total": count, "rate": correct / count}
    payload = {"schema": "af02-address-erasure-v1",
               "scope": "Fixed depth2/schedulePPG, all nine status assignments and both orders; deterministic outputs on identical address-erased F9 histories.",
               "cases": len(cases), "erased_input_classes": len(groups),
               "ceilings": {"preferred_action": rate(task_best, len(cases)),
                            "audit_exact": rate(audit_best, len(cases)),
                            "exactly_one_valid_action": rate(positive_best, positive_count),
                            "binding_pair_both_correct": rate(task_pair_best, len(task_pairs)),
                            "audit_pair_both_correct": rate(audit_pair_best, len(audit_pairs))},
               "groups": rows,
               "case_targets": [{"case_id": case.case_id, "statuses": list(case.statuses),
                                 "probe_order": list(case.probe_order), "action": target_action(case),
                                 "audit": list(audit_targets(case))} for case in cases]}
    payload["receipt_sha256"] = digest(payload)
    return payload


def manifest(cases: Iterable[Case]) -> dict[str, Any]:
    cases = list(cases)
    groups: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        groups[public_input_hash(case)].append(case)
    canonical = [copies[0] for copies in groups.values()]
    rows = [{"public_input_sha256": key, "canonical_id": copies[0].canonical_id,
             "case_ids": [case.case_id for case in copies],
             "target_digests": [case.target_digest for case in copies],
             "context_digests": [case.context_digest for case in copies],
             "depth": copies[0].depth, "schedule": copies[0].schedule,
             "probe_order": list(copies[0].probe_order), "statuses": list(copies[0].statuses)}
            for key, copies in groups.items()]
    return {"host_cases": len(cases), "canonical_inputs": len(canonical),
            "copies_per_input": sorted({len(copies) for copies in groups.values()}),
            "status_histogram": dict(sorted(Counter(",".join(case.statuses) for case in canonical).items())),
            "host_cases_sha256": digest([case.to_dict() for case in cases]),
            "canonical_cases_sha256": digest([case.to_dict() for case in canonical]),
            "public_sequences_sha256": digest([feature_sequence(case) for case in canonical]),
            "patterns": rows}


def corpus_manifest() -> dict[str, Any]:
    splits = {split: manifest(generate_split(split)) for split in SPLIT_SPEC}
    hash_sets = {split: {row["public_input_sha256"] for row in item["patterns"]}
                 for split, item in splits.items()}
    intersections = {}
    for left, right in itertools.combinations(SPLIT_SPEC, 2):
        common = hash_sets[left] & hash_sets[right]
        intersections[f"{left}/{right}"] = len(common)
        if common:
            raise ValueError(f"canonical public input leakage between {left} and {right}")
    payload = {"schema": "af02-corpus-v1", "protocol_sha256": REGISTERED_PROTOCOL_SHA256,
               "enumeration_status_order": list(STATUSES), "audit_class_order": list(AUDIT_STATUSES),
               "enumeration_order": ["depth", "schedule", "probe_order", "status0", "status1"],
               "rename_seed_formula": "split_seed*100000 + canonical_index*2 + rename_index",
               "splits": splits, "pairwise_public_input_intersections": intersections,
               "all_public_input_sets_disjoint": True}
    payload["manifest_sha256"] = digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "manifest", "erasure"))
    parser.add_argument("--split", choices=tuple(SPLIT_SPEC), default="train")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("preserve existing corpus evidence; choose a fresh output path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        args.out.write_text("".join(json.dumps(case.to_dict(), sort_keys=True) + "\n"
                                    for case in generate_split(args.split)))
    else:
        payload = corpus_manifest() if args.command == "manifest" else address_erasure_receipt()
        args.out.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
