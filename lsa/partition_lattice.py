"""Finite lattice of parameter-sharing patterns for looped architectures.

A partition describes *which visit positions share one physical parameter block*.
It says nothing about the order of those visits. Dispatch order belongs to
`lsa.schedule` and `lsa.topology`.

Partitions are represented by restricted-growth tuples, e.g. (0, 1, 0, 2).
The order is refinement: P <= Q iff every class of P is contained in a
class of Q. Consequently, more weight tying is *higher* in this order.
This is a purely combinatorial module with no torch dependency.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence


Partition = tuple[int, ...]


def canonical_partition(labels: Iterable[Hashable]) -> Partition:
    """Canonicalize equality classes by first appearance.

    An empty visit schedule has the unique empty partition.
    """
    names: dict[Hashable, int] = {}
    result: list[int] = []
    for label in labels:
        if label not in names:
            names[label] = len(names)
        result.append(names[label])
    return tuple(result)


def _compatible(left: Sequence[Hashable], right: Sequence[Hashable]) -> tuple[Partition, Partition]:
    a, b = canonical_partition(left), canonical_partition(right)
    if len(a) != len(b):
        raise ValueError("parameter partitions must cover the same visit positions")
    return a, b


def refines(finer: Sequence[Hashable], coarser: Sequence[Hashable]) -> bool:
    """Whether every tied class in `finer` sits in one class of `coarser`."""
    a, b = _compatible(finer, coarser)
    image: dict[int, int] = {}
    for x, y in zip(a, b):
        if x in image and image[x] != y:
            return False
        image[x] = y
    return True


def partition_meet(left: Sequence[Hashable], right: Sequence[Hashable]) -> Partition:
    """Greatest common refinement: intersect the two equivalence relations."""
    a, b = _compatible(left, right)
    return canonical_partition(zip(a, b))


def partition_join(left: Sequence[Hashable], right: Sequence[Hashable]) -> Partition:
    """Least common coarsening: transitive closure of both tie relations."""
    a, b = _compatible(left, right)
    size = len(a)
    parent = list(range(size))
    rank = [0] * size

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(i: int, j: int) -> None:
        x, y = find(i), find(j)
        if x == y:
            return
        if rank[x] < rank[y]:
            x, y = y, x
        parent[y] = x
        if rank[x] == rank[y]:
            rank[x] += 1

    for partition in (a, b):
        representatives: dict[int, int] = {}
        for i, label in enumerate(partition):
            if label in representatives:
                union(i, representatives[label])
            else:
                representatives[label] = i

    return canonical_partition(find(i) for i in range(size))


def unique_block_count(partition: Sequence[Hashable]) -> int:
    """Number of independent parameter blocks, excluding embedding/readout."""
    return len(set(canonical_partition(partition)))


def tied_pair_distance(left: Sequence[Hashable], right: Sequence[Hashable]) -> int:
    """How many visit pairs disagree about sharing parameters."""
    a, b = _compatible(left, right)
    return sum(
        (a[i] == a[j]) != (b[i] == b[j])
        for i in range(len(a))
        for j in range(i + 1, len(a))
    )


def assert_type_safe_ties(visit_parameter_types: Sequence[Hashable], partition: Sequence[Hashable]) -> Partition:
    """Require all visits tied to one parameter object to share a parameter schema.

    A schema should include shape, role and any implementation constraints
    relevant to aliasing. Equal schemas permit tying; they do not require it.
    """
    if len(visit_parameter_types) != len(partition):
        raise ValueError("one parameter type is required per visit")
    canonical = canonical_partition(partition)
    types_by_class: dict[int, Hashable] = {}
    for position, (kind, group) in enumerate(zip(visit_parameter_types, canonical)):
        if group in types_by_class and types_by_class[group] != kind:
            raise ValueError(f"incompatible parameter types tied at visit {position}")
        types_by_class[group] = kind
    return canonical


def enumerate_partitions(size: int) -> tuple[Partition, ...]:
    """Enumerate all set partitions (Bell-number growth; use only for small n).

    The n=0 case includes the unique partition of the empty set.
    """
    if size < 0:
        raise ValueError("visit count must be non-negative")
    if size == 0:
        return ((),)

    result: list[Partition] = []

    def extend(prefix: tuple[int, ...], greatest: int) -> None:
        if len(prefix) == size:
            result.append(prefix)
            return
        for group in range(greatest + 2):
            extend(prefix + (group,), max(group, greatest))

    extend((0,), 0)
    return tuple(result)
