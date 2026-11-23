"""LSA-02 finite algebra checks; pure CPU, no neural-model dependency."""

from __future__ import annotations

from itertools import product

import pytest

from lsa.partition_lattice import (
    assert_type_safe_ties,
    canonical_partition,
    enumerate_partitions,
    partition_join,
    partition_meet,
    refines,
    tied_pair_distance,
    unique_block_count,
)
from lsa.schedule import VisitSymbol, parse_word
from lsa.topology import canonical_module_word
from research_gym.core.hybrid import CandidateState


def test_partition_enumeration_and_canonical_renaming() -> None:
    assert [len(enumerate_partitions(n)) for n in range(6)] == [1, 1, 2, 5, 15, 52]
    assert canonical_partition(("fast", "slow", "fast", "probe")) == (0, 1, 0, 2)
    assert canonical_partition(("x", "y", "x")) == canonical_module_word((8, 9, 8))
    assert unique_block_count((0, 1, 0, 2)) == 3
    assert unique_block_count(()) == 0
    with pytest.raises(ValueError):
        enumerate_partitions(-1)
    with pytest.raises(ValueError):
        partition_join((0,), (0, 1))


def test_partition_meet_and_join_universal_properties() -> None:
    partitions = enumerate_partitions(4)
    for a, b in product(partitions, repeat=2):
        meet = partition_meet(a, b)
        join = partition_join(a, b)
        assert refines(meet, a) and refines(meet, b)
        assert refines(a, join) and refines(b, join)
        assert partition_meet(a, b) == partition_meet(b, a)
        assert partition_join(a, b) == partition_join(b, a)
        assert partition_meet(a, a) == a
        assert partition_join(a, a) == a
        assert partition_meet(a, join) == a
        assert partition_join(a, meet) == a
        for candidate in partitions:
            assert (refines(candidate, a) and refines(candidate, b)) == refines(candidate, meet)
            assert (refines(a, candidate) and refines(b, candidate)) == refines(join, candidate)


def test_lattice_associativity_and_nondistributivity() -> None:
    parts = enumerate_partitions(4)
    for a, b, c in product(parts, repeat=3):
        assert partition_meet(partition_meet(a, b), c) == partition_meet(a, partition_meet(b, c))
        assert partition_join(partition_join(a, b), c) == partition_join(a, partition_join(b, c))
    # Partition lattices contain an M3 sublattice and need not be distributive.
    a, b, c = (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0)
    assert partition_meet(a, partition_join(b, c)) != partition_join(
        partition_meet(a, b), partition_meet(a, c)
    )


def test_parameter_schema_fibres_closed_under_lattice_operations() -> None:
    visit_schemas = ("dense:128", "dense:128", "graph:64", "graph:64")
    a, b = (0, 0, 1, 2), (0, 1, 2, 2)
    assert assert_type_safe_ties(visit_schemas, a) == a
    assert assert_type_safe_ties(visit_schemas, b) == b
    assert_type_safe_ties(visit_schemas, partition_meet(a, b))
    assert_type_safe_ties(visit_schemas, partition_join(a, b))
    with pytest.raises(ValueError, match="incompatible"):
        assert_type_safe_ties(visit_schemas, (0, 1, 0, 1))
    with pytest.raises(ValueError, match="one parameter type"):
        assert_type_safe_ties(("dense",), (0, 1))


def test_partition_distance_and_order_are_distinct_properties() -> None:
    fully_tied = (0, 0, 0, 0)
    untied = (0, 1, 2, 3)
    assert refines(untied, fully_tied)
    assert not refines(fully_tied, untied)
    assert tied_pair_distance(fully_tied, untied) == 6
    assert tied_pair_distance(untied, fully_tied) == 6
    assert tied_pair_distance(fully_tied, fully_tied) == 0


def test_same_sharing_structure_does_not_make_schedules_equal() -> None:
    symbols = {
        "A": VisitSymbol("A", "a", frozenset({"state"}), "state"),
        "B": VisitSymbol("B", "b", frozenset({"state"}), "state"),
    }
    ab = parse_word("A.B", symbols)
    ba = parse_word("B.A", symbols)
    assert partition_meet((0, 1), (0, 1)) == (0, 1)
    assert unique_block_count((0, 1)) == 2

    operators = {"a": lambda x: x + 1, "b": lambda x: 2 * x}

    def execute(word) -> int:
        state = 1
        for visit in word.expand():
            state = operators[visit.symbol.module](state)
        return state

    assert execute(ab) == 4
    assert execute(ba) == 3


def test_candidate_lattice_certified_descent_and_order_independent_meets() -> None:
    initial = CandidateState({"goal": frozenset({"a", "b", "c", "d"})})
    first = CandidateState({"goal": frozenset({"b", "c"})})
    second = CandidateState({"goal": frozenset({"a", "b", "d"})})
    assert initial.meet(first).meet(second) == initial.meet(second).meet(first)
    assert initial.meet(first).meet(second).domains["goal"] == frozenset({"b"})
    # The actual state is b, retained by each independent sound proposal.
    state = initial
    for proposal in (first, second):
        next_state = state.meet(proposal)
        assert next_state.is_refinement_of(state)
        assert "b" in next_state.domains["goal"]
        state = next_state
    # A finite product powerset loses a candidate at every strict refinement.
    assert len(state.domains["goal"]) <= len(initial.domains["goal"])
