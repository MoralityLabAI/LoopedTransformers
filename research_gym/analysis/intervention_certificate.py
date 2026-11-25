"""MC-02: declare a cursor intervention before observing its continuation.

This is an empirical causal-abstraction assay. Its symbolic certificates check
only the declared pointer relation; source execution is trusted separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import random
from typing import Sequence

from research_gym.analysis.mechanistic_certificate import (
    Statement, digest, encode, integer, oracle, relation,
)


@dataclass(frozen=True)
class Counterfactual:
    receiver_graph: tuple[int, ...]
    donor_graph: tuple[int, ...]
    receiver_cursor: int
    donor_cursor: int
    horizon: int

    def __post_init__(self) -> None:
        n = len(self.receiver_graph)
        Statement(self.receiver_graph, self.receiver_cursor, self.horizon, self.donor_cursor)
        Statement(self.donor_graph, self.donor_cursor, self.horizon, self.receiver_cursor)
        if len(self.donor_graph) != n:
            raise ValueError("receiver and donor must use the same node alphabet")
        if sorted(self.receiver_graph) != list(range(n)) or sorted(self.donor_graph) != list(range(n)):
            raise ValueError("this assay's native task contract requires permutations")

    def paths(self) -> dict[str, tuple[int, ...]]:
        return {
            "counterfactual": oracle(self.receiver_graph, self.donor_cursor, self.horizon),
            "receiver_baseline": oracle(self.receiver_graph, self.receiver_cursor, self.horizon),
            "donor_baseline": oracle(self.donor_graph, self.donor_cursor, self.horizon),
        }

    def discriminating(self) -> bool:
        paths = self.paths()
        return (self.donor_cursor != self.receiver_cursor
                and paths["counterfactual"][1] != paths["donor_baseline"][1]
                and paths["counterfactual"][1] != paths["receiver_baseline"][1])

    def description(self) -> dict:
        return {"receiver_graph": list(self.receiver_graph), "donor_graph": list(self.donor_graph),
                "receiver_cursor": self.receiver_cursor, "donor_cursor": self.donor_cursor,
                "horizon": self.horizon, "paths": self.paths(),
                "first_step_discriminating": self.discriminating()}

    def statement(self) -> Statement:
        target = self.paths()["counterfactual"]
        return Statement(self.receiver_graph, self.donor_cursor, self.horizon, target[-1])


def select_pairs(graphs: Sequence[Sequence[int]], starts: Sequence[int],
                 endpoints: Sequence[Sequence[int]], visits: Sequence[int], *,
                 horizon: int, max_pairs: int, seed: int) -> dict:
    """Read input graphs and current endpoint labels only, never resumed outcomes.

    At most one donor per (receiver,visit); a seeded cap samples eligible receiver
    slots uniformly. All endpoint-correct receivers remain in the coverage ledger.
    """
    integer(horizon, 1, 64, "horizon")
    integer(max_pairs, 1, 1024, "max_pairs")
    integer(seed, 0, 2**63 - 1, "seed")
    if not graphs or len(graphs) != len(starts) or not visits or len(endpoints) != len(visits):
        raise ValueError("nonempty input tables with matching dimensions required")
    if len(set(visits)) != len(visits) or list(visits) != sorted(visits):
        raise ValueError("visits must be unique and increasing")
    count, n = len(graphs), len(graphs[0])
    gs = [tuple(g) for g in graphs]
    for graph, start in zip(gs, starts):
        Statement(graph, start, horizon, start)
        if len(graph) != n or sorted(graph) != list(range(n)):
            raise ValueError("all graphs must be permutations of the same alphabet")
    for visit, row in zip(visits, endpoints):
        integer(visit, 1, 64, "visit")
        if len(row) != count:
            raise ValueError("endpoint matrix width differs from item count")
        for value in row:
            integer(value, 0, n - 1, "endpoint")
    rng = random.Random(seed)
    eligible_slots = []
    candidate_count = endpoint_correct_slots = 0
    for visit, row in zip(visits, endpoints):
        truth = [oracle(g, start, visit)[-1] for g, start in zip(gs, starts)]
        for receiver in range(count):
            if row[receiver] != truth[receiver]:
                continue
            endpoint_correct_slots += 1
            donors = []
            for donor in range(count):
                if donor == receiver or row[donor] != truth[donor]:
                    continue
                claim = Counterfactual(gs[receiver], gs[donor], truth[receiver], truth[donor], horizon)
                if claim.discriminating():
                    donors.append(donor)
            candidate_count += len(donors)
            if donors:
                eligible_slots.append((visit, receiver, donors, truth))
    chosen = sorted(rng.sample(range(len(eligible_slots)), min(max_pairs, len(eligible_slots))))
    rows = []
    for index in chosen:
        visit, receiver, donors, truth = eligible_slots[index]
        donor = rng.choice(donors)
        claim = Counterfactual(gs[receiver], gs[donor], truth[receiver], truth[donor], horizon)
        rows.append({"visit": visit, "receiver": receiver, "donor": donor,
                     "eligible": True, "claim": claim.description()})
    manifest = {"schema": "mc02.selection.v1", "seed": seed, "horizon": horizon,
                "max_pairs": max_pairs, "visits": list(visits), "items": count,
                "receiver_slots": count * len(visits), "endpoint_correct_slots": endpoint_correct_slots,
                "eligible_receiver_slots": len(eligible_slots), "candidate_ordered_pairs": candidate_count,
                "selected_pairs": rows, "graphs": gs, "starts": list(starts),
                "endpoint_predictions": [list(row) for row in endpoints],
                "selection_inputs": "graphs, starts, visits and current predictions; no future network outputs"}
    return {**manifest, "manifest_sha256": digest(manifest)}


def validate_manifest(manifest: dict) -> None:
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != digest(payload):
        raise ValueError("selection manifest identity mismatch")


def score_continuation(claim: Counterfactual, current: int, future: Sequence[int]) -> dict:
    """Audit captured labels without asserting that they came from honest inference."""
    n = len(claim.receiver_graph)
    integer(current, 0, n - 1, "post-patch current label")
    if len(future) != claim.horizon:
        raise ValueError("future length differs from the declared horizon")
    for value in future:
        integer(value, 0, n - 1, "future label")
    paths = claim.paths()
    expected = paths["counterfactual"]
    observed = (current,) + tuple(future)
    statement = claim.statement()
    failures = relation(n, claim.horizon).failures(encode(statement, observed))
    mismatches = [k for k in range(1, claim.horizon + 1) if observed[k] != expected[k]]
    return {"post_patch_label": current, "future_labels": list(future),
            "lift_matches": current == claim.donor_cursor,
            "counterfactual_future_match": not mismatches,
            "counterfactual_trace_match": not failures,
            "first_wrong_future_offset": min(mismatches) if mismatches else None,
            "first_future_matches_donor": future[0] == paths["donor_baseline"][1],
            "first_future_matches_receiver": future[0] == paths["receiver_baseline"][1],
            "first_future_matches_counterfactual": future[0] == expected[1],
            "oracle_target": list(expected), "constraint_failures": failures,
            "statement_sha256": digest(statement.identity()),
            "source_binding": "trusted_execution_required", "cryptographic_source_binding": False,
            "causal_scope": "declared cursor lift and tested finite continuation only"}


def exact_controls() -> dict:
    """Enumerate all discriminating N=3 graph/cursor pairs; no neural measurement."""
    tested = correct = donor_rejected = receiver_rejected = final_collisions = 0
    example = None
    for receiver_graph, donor_graph in itertools.product(itertools.permutations(range(3)), repeat=2):
        for receiver, donor in itertools.product(range(3), repeat=2):
            claim = Counterfactual(receiver_graph, donor_graph, receiver, donor, 4)
            if not claim.discriminating():
                continue
            paths = claim.paths()
            truth = score_continuation(claim, donor, paths["counterfactual"][1:])
            copied = score_continuation(claim, donor, paths["donor_baseline"][1:])
            ignored = score_continuation(claim, donor, paths["receiver_baseline"][1:])
            tested += 1
            final_collisions += int(paths["counterfactual"][-1] == paths["donor_baseline"][-1])
            correct += int(truth["counterfactual_trace_match"])
            donor_rejected += int(not copied["counterfactual_trace_match"])
            receiver_rejected += int(not ignored["counterfactual_trace_match"])
            if example is None:
                example = {"claim": claim.description(), "correct": truth,
                           "donor_copy": copied, "ignored_intervention": ignored}
    if not (tested == correct == donor_rejected == receiver_rejected and tested > 0):
        raise RuntimeError("exhaustive counterfactual controls failed")
    return {"status": "EXACT_CONTROLS_ONLY", "challenges": tested, "correct_accepted": correct,
            "donor_copies_rejected": donor_rejected, "ignored_interventions_rejected": receiver_rejected,
            "donor_copy_correct_final_answer": final_collisions,
            "example": example, "trained_neural_runs": 0}
