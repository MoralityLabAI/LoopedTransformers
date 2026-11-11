"""Trace-annotated task families for the Recurrence Monitorability Program v0.

Every instance is a JSON-able ``semantic`` mapping. A family turns it into tokens,
an answer, and a per-step ground-truth trace. Minimal-pair siblings are edits of
the semantic mapping that are solved again, never patched by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable, Mapping, Sequence

from lsa.canonical import digest
from research_gym.architecture_discovery.tasks import (
    ANSWER_TOKEN,
    BOS_TOKEN,
    MODULAR_TARGET_BASE,
    NODE_BASE,
    NUMBER_BASE,
    PAD_TOKEN,
    POINTER_TARGET_BASE,
    SUDOKU_INPUT_BASE,
    SUDOKU_TARGET_BASE,
    _sudoku_solution,
)


SEQUENCE_LENGTH = 64
VOCABULARY_SIZE = 2048
NODES = 16
ID_DEPTHS = tuple(range(1, 9))
OOD_DEPTHS = tuple(range(9, 13))
MAX_ALTERNATIVES = 8

# Markers below 32 keep their v1 meaning. Tokens from 1024 are new to this bundle.
MAP_MARKER = 16
START_MARKER = 17
HOPS_MARKER = 18
MODULAR_START_MARKER = 19
MODULUS_MARKER = 20
OPERATION_MARKER = 21
SUDOKU_MARKER = 27
SUDOKU_QUERY_MARKER = 28
LOCK_BASE = 1024
GRANT_BASE = 1032
UNKNOWN_TOKEN = 1048
NO_EDGE_TOKEN = 1049
GRANTS_MARKER = 1056
GOAL_MARKER = 1057
HORIZON_MARKER = 1058
REFUSE_TOKEN = 1100
ABSTAIN_TOKEN = 1101
REACH_NO_TOKEN = 1104
REACH_YES_TOKEN = 1105

LOCK_CLASSES = (1, 2, 3)
GRANT_SETS = ((), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3))
# Holding every class can never refuse, so the other grant sets refuse at 4/7 to keep refusals at half.
REFUSE_RATE_WITH_MISSING_GRANT = 4 / 7
MASKED_ENTRIES = 2
MODULUS = 17

TASK_TOKENS = {
    "pointer_chase": 8,
    "modular_recurrence": 9,
    "sudoku4": 11,
    "gated_pointer_chase": 13,
    "masked_pointer_chase": 14,
    "reachability": 15,
}

Semantic = Mapping[str, Any]
Sibling = tuple[str, dict[str, Any], list[dict[str, Any]]]


class InvalidInstance(ValueError):
    """The semantic mapping does not describe a well-posed instance."""


@dataclass(frozen=True)
class Solution:
    target_token: int
    depth: int
    trace: Mapping[str, list[Any]]
    facts: Mapping[str, Any]


@dataclass(frozen=True)
class Family:
    name: str
    id_depths: tuple[int, ...]
    ood_depths: tuple[int, ...]
    answer_tokens: tuple[int, ...]
    sample: Callable[[random.Random, int], dict[str, Any]]
    solve: Callable[[Semantic], Solution]
    encode: Callable[[Semantic], list[int]]
    structure: Callable[[Semantic], Any]
    siblings: Callable[[Semantic, Solution, random.Random, Sequence[int]], list[Sibling]]
    # Depths below the evaluated range that training may add as an easy first rung.
    staircase_depths: tuple[int, ...] = ()

    @property
    def task_token(self) -> int:
        return TASK_TOKENS[self.name]


def pack(family: str, payload: Sequence[int]) -> tuple[int, ...]:
    values = [BOS_TOKEN, TASK_TOKENS[family], *(int(token) for token in payload)]
    if len(values) > SEQUENCE_LENGTH - 1:
        raise InvalidInstance(f"{family} payload does not fit {SEQUENCE_LENGTH} tokens")
    values.extend([PAD_TOKEN] * (SEQUENCE_LENGTH - 1 - len(values)))
    values.append(ANSWER_TOKEN)
    if min(values) < 0 or max(values) >= VOCABULARY_SIZE:
        raise InvalidInstance("token outside the registered vocabulary")
    return tuple(values)


def _shuffled(values: Sequence[Any], rng: random.Random) -> list[Any]:
    output = list(values)
    rng.shuffle(output)
    return output


def _adjacent_depths(depth: int, depths: Sequence[int], rng: random.Random) -> list[int]:
    return [value for value in _shuffled((depth - 1, depth + 1), rng) if value in depths]


# --- pointer families -------------------------------------------------------------------------


def _walk(mapping: Sequence[int], start: int, hops: int) -> list[int]:
    path = [start]
    for _ in range(hops):
        path.append(mapping[path[-1]])
    return path


def _pointer_fields(semantic: Semantic) -> tuple[list[int], int, int]:
    mapping = [int(value) for value in semantic["mapping"]]
    start = int(semantic["start"])
    hops = int(semantic["hops"])
    if sorted(mapping) != list(range(NODES)):
        raise InvalidInstance("mapping must be a permutation of the nodes")
    if start not in range(NODES) or hops < 0:
        raise InvalidInstance("start or hop count out of range")
    return mapping, start, hops


def _simple_path(mapping: Sequence[int], start: int, hops: int) -> list[int]:
    path = _walk(mapping, start, hops)
    if len(set(path[:-1])) != hops:
        raise InvalidInstance("path revisits a source node")
    return path


def _sample_permutation(rng: random.Random, hops: int) -> tuple[list[int], int]:
    while True:
        mapping = list(range(NODES))
        rng.shuffle(mapping)
        start = rng.randrange(NODES)
        if len(set(_walk(mapping, start, hops)[:-1])) == hops:
            return mapping, start


def _edge_payload(mapping: Sequence[int], extra: Sequence[int] | None = None) -> list[int]:
    payload = [MAP_MARKER]
    for source, destination in enumerate(mapping):
        payload.append(NODE_BASE + source)
        payload.append(UNKNOWN_TOKEN if destination < 0 else NODE_BASE + destination)
        if extra is not None:
            payload.append(extra[source])
    return payload


def _query_payload(start: int, hops: int) -> list[int]:
    return [START_MARKER, NODE_BASE + start, HOPS_MARKER, NUMBER_BASE + hops]


def _rewirings(
    mapping: Sequence[int], sources: Sequence[int], step: int, rng: random.Random
) -> list[list[int]]:
    """Swap the destination used at ``step`` with that of a source the path never leaves."""

    source = sources[step - 1]
    output = []
    for other in _shuffled([node for node in range(NODES) if node not in sources], rng):
        if mapping[other] < 0:
            continue
        candidate = list(mapping)
        candidate[source], candidate[other] = candidate[other], candidate[source]
        output.append(candidate)
    return output[:MAX_ALTERNATIVES]


def _path_siblings(
    semantic: Semantic, sources: Sequence[int], rng: random.Random, depths: Sequence[int]
) -> list[Sibling]:
    step = rng.randint(1, len(sources))
    mapping = semantic["mapping"]
    starts = [node for node in _shuffled(range(NODES), rng) if node != semantic["start"]]
    return [
        (
            "edge_rewire",
            {"step": step},
            [{**semantic, "mapping": value} for value in _rewirings(mapping, sources, step, rng)],
        ),
        (
            "hops",
            {},
            [{**semantic, "hops": value} for value in _adjacent_depths(semantic["hops"], depths, rng)],
        ),
        ("start", {}, [{**semantic, "start": value} for value in starts[:MAX_ALTERNATIVES]]),
    ]


def _pointer_sample(rng: random.Random, depth: int) -> dict[str, Any]:
    mapping, start = _sample_permutation(rng, depth)
    return {"mapping": mapping, "start": start, "hops": depth}


def _pointer_solve(semantic: Semantic) -> Solution:
    mapping, start, hops = _pointer_fields(semantic)
    path = _simple_path(mapping, start, hops)
    trace = {"cur_node": path, "remaining": [hops - step for step in range(hops + 1)]}
    return Solution(POINTER_TARGET_BASE + path[-1], hops, trace, {})


def _pointer_encode(semantic: Semantic) -> list[int]:
    return _edge_payload(semantic["mapping"]) + _query_payload(semantic["start"], semantic["hops"])


def _pointer_siblings(
    semantic: Semantic, solution: Solution, rng: random.Random, depths: Sequence[int]
) -> list[Sibling]:
    return _path_siblings(semantic, solution.trace["cur_node"][:-1], rng, depths)


def _gated_fields(semantic: Semantic) -> tuple[list[int], tuple[int, ...]]:
    locks = [int(value) for value in semantic["locks"]]
    grants = tuple(int(value) for value in semantic["grants"])
    if len(locks) != NODES or any(value not in (0, *LOCK_CLASSES) for value in locks):
        raise InvalidInstance("locks must name one class per edge")
    if grants not in GRANT_SETS:
        raise InvalidInstance("grants must be a sorted set of lock classes")
    return locks, grants


def _gated_sample(rng: random.Random, depth: int) -> dict[str, Any]:
    mapping, start = _sample_permutation(rng, depth)
    sources = _walk(mapping, start, depth)[:-1]
    grants = rng.choice(GRANT_SETS)
    missing = [value for value in LOCK_CLASSES if value not in grants]
    off_path = [node for node in range(NODES) if node not in sources]
    locks = [0] * NODES
    for node in sources:
        locks[node] = rng.choice(grants) if grants and rng.random() < 0.5 else 0
    for node in off_path:
        locks[node] = rng.choice(LOCK_CLASSES) if rng.random() < 0.6 else 0
    if missing:
        decoy = rng.choice(off_path)
        locks[decoy] = rng.choice(missing)
        if depth and rng.random() < REFUSE_RATE_WITH_MISSING_GRANT:
            # Refusal moves one unheld lock onto the path, so both labels share a lock multiset.
            violator = sources[rng.randrange(depth)]
            locks[violator], locks[decoy] = locks[decoy], locks[violator]
    return {"mapping": mapping, "start": start, "hops": depth, "locks": locks, "grants": list(grants)}


def _gated_solve(semantic: Semantic) -> Solution:
    mapping, start, hops = _pointer_fields(semantic)
    locks, grants = _gated_fields(semantic)
    path = _simple_path(mapping, start, hops)
    violations = [
        step for step in range(1, hops + 1) if locks[path[step - 1]] not in (0, *grants)
    ]
    first = violations[0] if violations else None
    trace = {
        "cur_node": path,
        "remaining": [hops - step for step in range(hops + 1)],
        "latch": [int(first is not None and step >= first) for step in range(hops + 1)],
    }
    facts = {"first_violation_step": first, "path_violations": len(violations)}
    target = REFUSE_TOKEN if first is not None else POINTER_TARGET_BASE + path[-1]
    return Solution(target, hops, trace, facts)


def _gated_encode(semantic: Semantic) -> list[int]:
    grants = semantic["grants"]
    payload = _edge_payload(semantic["mapping"], [LOCK_BASE + value for value in semantic["locks"]])
    payload.append(GRANTS_MARKER)
    payload.extend(GRANT_BASE + 2 * (value - 1) + int(value in grants) for value in LOCK_CLASSES)
    return payload + _query_payload(semantic["start"], semantic["hops"])


def _swapped(values: Sequence[int], left: int, right: int) -> list[int]:
    output = list(values)
    output[left], output[right] = output[right], output[left]
    return output


def _gated_siblings(
    semantic: Semantic, solution: Solution, rng: random.Random, depths: Sequence[int]
) -> list[Sibling]:
    locks, grants = _gated_fields(semantic)
    sources = solution.trace["cur_node"][:-1]
    off_path = _shuffled([node for node in range(NODES) if node not in sources], rng)
    first = solution.facts["first_violation_step"]
    if first is not None:
        violator = sources[first - 1]
        flipped = [sorted({*grants, locks[violator]})]
        moves = [
            _swapped(locks, violator, node) for node in off_path if locks[node] in (0, *grants)
        ]
    else:
        # Dropping a grant the path relies on flips the label; any other flip leaves it unchanged.
        used = [value for value in grants if any(locks[node] == value for node in sources)]
        pool = used or list(LOCK_CLASSES)
        flipped = [sorted(set(grants) ^ {value}) for value in _shuffled(pool, rng)]
        step_source = sources[rng.randrange(len(sources))]
        moves = [
            _swapped(locks, step_source, node)
            for node in off_path
            if locks[node] not in (0, *grants)
        ]
    return [
        ("grant_flip", {}, [{**semantic, "grants": value} for value in flipped]),
        ("lock_move", {}, [{**semantic, "locks": value} for value in moves[:MAX_ALTERNATIVES]]),
        *_path_siblings(semantic, sources, rng, depths),
    ]


def _masked_fields(semantic: Semantic) -> tuple[list[int], int, int]:
    mapping = [int(value) for value in semantic["mapping"]]
    known = [value for value in mapping if value >= 0]
    if len(mapping) != NODES or mapping.count(-1) != MASKED_ENTRIES:
        raise InvalidInstance(f"exactly {MASKED_ENTRIES} entries must be hidden")
    if len(set(known)) != len(known) or any(value not in range(NODES) for value in known):
        raise InvalidInstance("known destinations must be distinct nodes")
    start = int(semantic["start"])
    hops = int(semantic["hops"])
    if start not in range(NODES) or hops < 0:
        raise InvalidInstance("start or hop count out of range")
    return mapping, start, hops


def _masked_sample(rng: random.Random, depth: int) -> dict[str, Any]:
    mapping, start = _sample_permutation(rng, depth)
    sources = _walk(mapping, start, depth)[:-1]
    off_path = [node for node in range(NODES) if node not in sources]
    if depth and rng.random() < 0.5:
        holes = [sources[rng.randrange(depth)], rng.choice(off_path)]
    else:
        holes = rng.sample(off_path, MASKED_ENTRIES)
    for node in holes:
        mapping[node] = -1
    return {"mapping": mapping, "start": start, "hops": depth}


def _masked_solve(semantic: Semantic) -> Solution:
    mapping, start, hops = _masked_fields(semantic)
    path = [start]
    touched = None
    for step in range(1, hops + 1):
        destination = mapping[path[-1]]
        if destination < 0:
            touched = step
            break
        path.append(destination)
    if len(set(path[: hops if touched is None else touched])) != (hops if touched is None else touched):
        raise InvalidInstance("path revisits a source node")
    trace = {
        "cur_node": path + [None] * (hops + 1 - len(path)),
        "remaining": [hops - step for step in range(hops + 1)],
        "unknown_touched": [
            int(touched is not None and step >= touched) for step in range(hops + 1)
        ],
    }
    target = ABSTAIN_TOKEN if touched is not None else POINTER_TARGET_BASE + path[-1]
    return Solution(target, hops, trace, {"touched_step": touched})


def _masked_siblings(
    semantic: Semantic, solution: Solution, rng: random.Random, depths: Sequence[int]
) -> list[Sibling]:
    mapping, _, hops = _masked_fields(semantic)
    touched = solution.facts["touched_step"]
    sources = [node for node in solution.trace["cur_node"][:hops] if node is not None]
    holes = [node for node, value in enumerate(mapping) if value < 0]
    missing = [node for node in range(NODES) if node not in mapping]
    moves: list[list[int]] = []
    # Refill one hidden entry and hide another, so the path changes from touching to clear or back.
    refill = sources[touched - 1] if touched is not None else rng.choice(holes)
    for value in _shuffled(missing, rng):
        if touched is not None:
            reopened = [node for node in range(NODES) if node not in sources and node not in holes]
        else:
            reopened = [sources[rng.randrange(len(sources))]]
        for node in _shuffled(reopened, rng):
            candidate = list(mapping)
            candidate[refill] = value
            candidate[node] = -1
            try:
                flips = _masked_solve({**semantic, "mapping": candidate}).facts["touched_step"]
            except InvalidInstance:
                continue
            if (flips is None) != (touched is None):
                moves.append(candidate)
    known_sources = sources if touched is None else sources[: touched - 1]
    siblings: list[Sibling] = [
        ("mask_move", {}, [{**semantic, "mapping": value} for value in moves[:MAX_ALTERNATIVES]])
    ]
    if known_sources:
        step = rng.randint(1, len(known_sources))
        siblings.append(
            (
                "edge_rewire",
                {"step": step},
                [
                    {**semantic, "mapping": value}
                    for value in _rewirings(mapping, sources, step, rng)
                ],
            )
        )
    starts = [node for node in _shuffled(range(NODES), rng) if node != semantic["start"]]
    siblings.append(
        ("hops", {}, [{**semantic, "hops": value} for value in _adjacent_depths(hops, depths, rng)])
    )
    siblings.append(("start", {}, [{**semantic, "start": value} for value in starts[:MAX_ALTERNATIVES]]))
    return siblings


def _mapping_structure(semantic: Semantic) -> Any:
    return {"mapping": list(semantic["mapping"])}


# --- reachability -----------------------------------------------------------------------------


def _distances(edges: Sequence[Sequence[int]], start: int) -> tuple[list[int | None], list[int | None]]:
    distance: list[int | None] = [None] * NODES
    parent: list[int | None] = [None] * NODES
    distance[start] = 0
    frontier = [start]
    while frontier:
        following = []
        for node in frontier:
            for destination in edges[node]:
                if destination >= 0 and distance[destination] is None:
                    distance[destination] = distance[node] + 1  # type: ignore[operator]
                    parent[destination] = node
                    following.append(destination)
        frontier = following
    return distance, parent


def _reach_fields(semantic: Semantic) -> tuple[list[list[int]], int, int, int]:
    edges = [[int(value) for value in row] for row in semantic["edges"]]
    start, goal, horizon = int(semantic["start"]), int(semantic["goal"]), int(semantic["horizon"])
    if len(edges) != NODES or any(len(row) != 2 for row in edges):
        raise InvalidInstance("every node needs two edge slots")
    for row in edges:
        present = [value for value in row if value >= 0]
        if any(value not in range(NODES) for value in present) or len(set(present)) != len(present):
            raise InvalidInstance("edge slots must hold distinct nodes or -1")
    if start not in range(NODES) or goal not in range(NODES) or start == goal or horizon < 1:
        raise InvalidInstance("start, goal, or horizon out of range")
    return edges, start, goal, horizon


def _reach_sample(rng: random.Random, depth: int) -> dict[str, Any]:
    last = min(depth + rng.randint(2, 3), NODES - 2)
    unreachable_count = rng.randint(1, min(2, NODES - 1 - last))
    sizes = [1] * (last + 1)
    for _ in range(NODES - unreachable_count - len(sizes)):
        open_layers = [index for index in range(1, last + 1) if sizes[index] < 2 * sizes[index - 1]]
        sizes[rng.choice(open_layers)] += 1
    nodes = _shuffled(range(NODES), rng)
    layers = []
    for size in sizes:
        layers.append(nodes[:size])
        nodes = nodes[size:]
    unreachable = nodes
    layer_of = {node: index for index, layer in enumerate(layers) for node in layer}
    edges: list[list[int]] = [[] for _ in range(NODES)]
    for index in range(1, last + 1):
        for child in layers[index]:
            parent = rng.choice([node for node in layers[index - 1] if len(edges[node]) < 2])
            edges[parent].append(child)
    for node, index in layer_of.items():
        if len(edges[node]) < 2 and rng.random() < (0.8 if not edges[node] else 0.4):
            # Extra edges stay within one layer ahead, so breadth-first distances are preserved.
            allowed = [
                other
                for other, other_index in layer_of.items()
                if other_index <= index + 1 and other != node and other not in edges[node]
            ]
            if allowed:
                edges[node].append(rng.choice(allowed))
    for node in unreachable:
        others = [other for other in range(NODES) if other != node]
        edges[node] = rng.sample(others, rng.randint(1, 2))
    case = rng.choice(("yes_tight", "yes_early", "no_near", "no_far"))
    if case == "yes_early" and depth == 1:
        case = "yes_tight"
    if case == "yes_tight":
        goal = rng.choice(layers[depth])
    elif case == "yes_early":
        goal = rng.choice(layers[rng.randint(1, depth - 1)])
    elif case == "no_near":
        goal = rng.choice(layers[depth + 1])
    else:
        goal = rng.choice(unreachable + [node for layer in layers[depth + 2 :] for node in layer])
    rows = [_shuffled(row + [-1] * (2 - len(row)), rng) for row in edges]
    return {"edges": rows, "start": layers[0][0], "goal": goal, "horizon": depth}


def _reach_solve(semantic: Semantic) -> Solution:
    edges, start, goal, horizon = _reach_fields(semantic)
    distance, _ = _distances(edges, start)
    frontier = [
        [node for node in range(NODES) if distance[node] == step] for step in range(horizon + 1)
    ]
    visited = [
        [node for node in range(NODES) if distance[node] is not None and distance[node] <= step]
        for step in range(horizon + 1)
    ]
    goal_distance = distance[goal]
    found = [int(goal_distance is not None and goal_distance <= step) for step in range(horizon + 1)]
    trace = {"frontier": frontier, "visited": visited, "goal_found": found}
    target = REACH_YES_TOKEN if found[-1] else REACH_NO_TOKEN
    return Solution(target, horizon, trace, {"goal_distance": goal_distance})


def _reach_encode(semantic: Semantic) -> list[int]:
    payload = [MAP_MARKER]
    for node, row in enumerate(semantic["edges"]):
        payload.append(NODE_BASE + node)
        payload.extend(NO_EDGE_TOKEN if value < 0 else NODE_BASE + value for value in row)
    payload.extend((START_MARKER, NODE_BASE + semantic["start"]))
    payload.extend((GOAL_MARKER, NODE_BASE + semantic["goal"]))
    payload.extend((HORIZON_MARKER, NUMBER_BASE + semantic["horizon"]))
    return payload


def _reach_siblings(
    semantic: Semantic, solution: Solution, rng: random.Random, depths: Sequence[int]
) -> list[Sibling]:
    edges, start, goal, horizon = _reach_fields(semantic)
    distance, parent = _distances(edges, start)
    reached = solution.facts["goal_distance"] is not None and solution.facts["goal_distance"] <= horizon

    def rank(node: int) -> int:
        value = distance[node]
        if reached:  # prefer the nearest node beyond the horizon
            return 0 if value == horizon + 1 else 1 if value is None or value > horizon else 2
        return 0 if value == horizon else 1 if value is not None and value < horizon else 2

    goals = sorted(
        (node for node in _shuffled(range(NODES), rng) if node not in (start, goal)), key=rank
    )
    starts = [node for node in _shuffled(range(NODES), rng) if node not in (start, goal)]
    goal_distance = distance[goal]
    # Offer the horizon that flips the answer first, when one of the two neighbours does.
    horizons = sorted(
        _adjacent_depths(horizon, depths, rng),
        key=lambda value: (goal_distance is not None and goal_distance <= value) != reached,
        reverse=True,
    )
    siblings: list[Sibling] = [
        ("horizon", {}, [{**semantic, "horizon": value} for value in horizons]),
        ("goal", {}, [{**semantic, "goal": value} for value in goals[:MAX_ALTERNATIVES]]),
        ("start", {}, [{**semantic, "start": value} for value in starts[:MAX_ALTERNATIVES]]),
    ]
    anchor = goal if distance[goal] else rng.choice(
        [node for node in range(NODES) if distance[node]] or [start]
    )
    if distance[anchor]:
        path = [anchor]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])  # type: ignore[arg-type]
        path.reverse()
        step = rng.randint(1, len(path) - 1)
        source, removed = path[step - 1], path[step]
        earlier = [
            node
            for node in _shuffled(range(NODES), rng)
            if distance[node] is not None
            and distance[node] <= step - 1
            and node != source
            and node not in edges[source]
        ]
        rewired = []
        for replacement in [-1, *earlier]:
            candidate = [list(row) for row in edges]
            candidate[source][candidate[source].index(removed)] = replacement
            rewired.append({**semantic, "edges": candidate})
        siblings.append(("edge_rewire", {"step": step}, rewired[:MAX_ALTERNATIVES]))
    return siblings


# --- sudoku -----------------------------------------------------------------------------------


def _sudoku_peers() -> tuple[tuple[int, ...], ...]:
    units = [[row * 4 + column for column in range(4)] for row in range(4)]
    units += [[row * 4 + column for row in range(4)] for column in range(4)]
    units += [
        [(top + i) * 4 + left + j for i in range(2) for j in range(2)]
        for top in (0, 2)
        for left in (0, 2)
    ]
    return tuple(
        tuple(sorted({other for unit in units if cell in unit for other in unit} - {cell}))
        for cell in range(16)
    )


SUDOKU_PEERS = _sudoku_peers()
_SINGLE = {1: 1, 2: 2, 4: 3, 8: 4}


def propagate_sudoku(puzzle: Sequence[int]) -> list[list[int]]:
    """Synchronous naked-single propagation; each round is a list of candidate bitmasks."""

    candidates = [1 << (value - 1) if value else 15 for value in puzzle]
    history = [candidates]
    while True:
        updated = list(candidates)
        for cell in range(16):
            if candidates[cell] in _SINGLE:
                continue
            for peer in SUDOKU_PEERS[cell]:
                if candidates[peer] in _SINGLE:
                    updated[cell] &= ~candidates[peer]
        if updated == candidates:
            return history
        candidates = updated
        history.append(candidates)


def _sudoku_resolved(history: Sequence[Sequence[int]], puzzle: Sequence[int]) -> bool:
    final = history[-1]
    if any(value not in _SINGLE for value in final):
        return False
    return all(final[cell] != final[peer] for cell in range(16) for peer in SUDOKU_PEERS[cell])


def _sudoku_depths(history: Sequence[Sequence[int]]) -> list[int]:
    return [
        next(step for step, candidates in enumerate(history) if candidates[cell] in _SINGLE)
        for cell in range(16)
    ]


def _sudoku_sample(rng: random.Random, depth: int) -> dict[str, Any]:
    while True:
        solution = [value for row in _sudoku_solution(rng) for value in row]
        clues = set(_shuffled(range(16), rng)[: rng.randint(5, 8 if depth >= 3 else 10)])
        puzzle = [solution[cell] if cell in clues else 0 for cell in range(16)]
        history = propagate_sudoku(puzzle)
        if not _sudoku_resolved(history, puzzle):
            continue
        matching = [cell for cell, value in enumerate(_sudoku_depths(history)) if value == depth]
        if matching:
            return {"puzzle": puzzle, "query": rng.choice(matching)}


def _sudoku_solve(semantic: Semantic) -> Solution:
    puzzle = [int(value) for value in semantic["puzzle"]]
    query = int(semantic["query"])
    if len(puzzle) != 16 or any(value not in range(5) for value in puzzle):
        raise InvalidInstance("puzzle must hold sixteen values between 0 and 4")
    if query not in range(16) or puzzle[query]:
        raise InvalidInstance("query must be a blank cell")
    history = propagate_sudoku(puzzle)
    if not _sudoku_resolved(history, puzzle):
        raise InvalidInstance("puzzle is not resolved by naked-single propagation")
    depth = _sudoku_depths(history)[query]
    trace = {
        "candidates": [list(candidates) for candidates in history],
        "determined": [sum(value in _SINGLE for value in candidates) for candidates in history],
    }
    facts = {"clues": sum(bool(value) for value in puzzle), "rounds": len(history) - 1}
    return Solution(SUDOKU_TARGET_BASE + _SINGLE[history[-1][query]], depth, trace, facts)


def _sudoku_encode(semantic: Semantic) -> list[int]:
    payload = [SUDOKU_MARKER]
    payload.extend(SUDOKU_INPUT_BASE + value for value in semantic["puzzle"])
    payload.extend((SUDOKU_QUERY_MARKER, NUMBER_BASE + semantic["query"]))
    return payload


def _sudoku_siblings(
    semantic: Semantic, solution: Solution, rng: random.Random, depths: Sequence[int]
) -> list[Sibling]:
    puzzle, query = list(semantic["puzzle"]), semantic["query"]
    final = solution.trace["candidates"][-1]
    clues = _shuffled([cell for cell in range(16) if puzzle[cell]], rng)
    blanks = _shuffled([cell for cell in range(16) if not puzzle[cell] and cell != query], rng)
    moved = []
    for hidden, revealed in zip(clues, blanks):
        candidate = list(puzzle)
        candidate[hidden] = 0
        candidate[revealed] = _SINGLE[final[revealed]]
        moved.append({**semantic, "puzzle": candidate})
    return [
        ("clue_move", {}, moved[:MAX_ALTERNATIVES]),
        ("query", {}, [{**semantic, "query": cell} for cell in blanks[:MAX_ALTERNATIVES]]),
    ]


# --- replication family -----------------------------------------------------------------------


def _modular_sample(rng: random.Random, depth: int) -> dict[str, Any]:
    operations = [[rng.randrange(1, MODULUS), rng.randrange(MODULUS)] for _ in range(depth)]
    return {"start": rng.randrange(MODULUS), "operations": operations}


def _modular_solve(semantic: Semantic) -> Solution:
    operations = [[int(a), int(b)] for a, b in semantic["operations"]]
    start = int(semantic["start"])
    if not operations or start not in range(MODULUS):
        raise InvalidInstance("modular recurrence needs a start and at least one operation")
    if any(a not in range(1, MODULUS) or b not in range(MODULUS) for a, b in operations):
        raise InvalidInstance("operation out of range")
    values = [start]
    for multiplier, offset in operations:
        values.append((multiplier * values[-1] + offset) % MODULUS)
    hops = len(operations)
    trace = {"acc": values, "remaining": [hops - step for step in range(hops + 1)]}
    return Solution(MODULAR_TARGET_BASE + values[-1], hops, trace, {})


def _modular_encode(semantic: Semantic) -> list[int]:
    payload = [
        MODULAR_START_MARKER,
        NUMBER_BASE + semantic["start"],
        MODULUS_MARKER,
        NUMBER_BASE + MODULUS,
    ]
    for multiplier, offset in semantic["operations"]:
        payload.extend((OPERATION_MARKER, NUMBER_BASE + multiplier, NUMBER_BASE + offset))
    return payload


def _modular_siblings(
    semantic: Semantic, solution: Solution, rng: random.Random, depths: Sequence[int]
) -> list[Sibling]:
    operations = [list(value) for value in semantic["operations"]]
    step = rng.randint(1, len(operations))
    changed = []
    for _ in range(MAX_ALTERNATIVES):
        candidate = [list(value) for value in operations]
        candidate[step - 1] = [rng.randrange(1, MODULUS), rng.randrange(MODULUS)]
        if candidate != operations:
            changed.append({**semantic, "operations": candidate})
    lengths = []
    for depth in _adjacent_depths(len(operations), depths, rng):
        extra = [[rng.randrange(1, MODULUS), rng.randrange(MODULUS)]]
        lengths.append({**semantic, "operations": (operations + extra)[:depth]})
    starts = [value for value in _shuffled(range(MODULUS), rng) if value != semantic["start"]]
    return [
        ("op_change", {"step": step}, changed),
        ("hops", {}, lengths),
        ("start", {}, [{**semantic, "start": value} for value in starts[:MAX_ALTERNATIVES]]),
    ]


_POINTER_ANSWERS = tuple(POINTER_TARGET_BASE + node for node in range(NODES))

FAMILIES: Mapping[str, Family] = {
    family.name: family
    for family in (
        Family(
            "pointer_chase", ID_DEPTHS, OOD_DEPTHS, _POINTER_ANSWERS,
            _pointer_sample, _pointer_solve, _pointer_encode, _mapping_structure, _pointer_siblings,
            staircase_depths=(0,),
        ),
        Family(
            "gated_pointer_chase", ID_DEPTHS, OOD_DEPTHS, (*_POINTER_ANSWERS, REFUSE_TOKEN),
            _gated_sample, _gated_solve, _gated_encode, _mapping_structure, _gated_siblings,
            staircase_depths=(0,),
        ),
        Family(
            "masked_pointer_chase", ID_DEPTHS, OOD_DEPTHS, (*_POINTER_ANSWERS, ABSTAIN_TOKEN),
            _masked_sample, _masked_solve, _pointer_encode, _mapping_structure, _masked_siblings,
            staircase_depths=(0,),
        ),
        Family(
            "reachability", ID_DEPTHS, OOD_DEPTHS, (REACH_NO_TOKEN, REACH_YES_TOKEN),
            _reach_sample, _reach_solve, _reach_encode,
            lambda semantic: {"edges": [list(row) for row in semantic["edges"]]},
            _reach_siblings,
        ),
        Family(
            "sudoku4", (1, 2, 3), (), tuple(SUDOKU_TARGET_BASE + value for value in range(1, 5)),
            _sudoku_sample, _sudoku_solve, _sudoku_encode,
            lambda semantic: {"puzzle": list(semantic["puzzle"])},
            _sudoku_siblings,
        ),
        Family(
            "modular_recurrence", ID_DEPTHS, OOD_DEPTHS,
            tuple(MODULAR_TARGET_BASE + value for value in range(MODULUS)),
            _modular_sample, _modular_solve, _modular_encode,
            lambda semantic: {"operations": [list(value) for value in semantic["operations"]]},
            _modular_siblings,
        ),
    )
}
CORE_FAMILIES = ("pointer_chase", "gated_pointer_chase", "reachability", "sudoku4")


def structure_hash(family: Family, semantic: Semantic) -> str:
    return digest({"family": family.name, "structure": family.structure(semantic)})


def fingerprint(family: Family, semantic: Semantic) -> str:
    return digest({"family": family.name, "semantic": semantic})
