from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import random

from research_gym.interp import tasks as T
from research_gym.interp.bundle import (
    BundleConfig,
    TraceExample,
    first_difference,
    iter_family_examples,
    materialize_bundle,
    read_shard,
    region_of,
    train_example,
)
from research_gym.interp.tasks import FAMILIES, InvalidInstance


ROOT = Path(__file__).resolve().parents[1]
TINY_COUNTS = {
    "validation": 48,
    "monitor_train": 48,
    "monitor_calibration": 24,
    "locked": 24,
    "ood_stress": 16,
}


def _config() -> BundleConfig:
    raw = json.loads((ROOT / "configs" / "rmp_task_bundle_v0.json").read_text(encoding="utf-8"))
    return BundleConfig.from_mapping(raw).scaled(TINY_COUNTS)


@lru_cache(maxsize=None)
def _tiny(family: str) -> tuple[TraceExample, ...]:
    return tuple(iter_family_examples(family, _config()))


def _samples(family: str, per_depth: int, seed: int = 5):
    rng = random.Random(seed)
    definition = FAMILIES[family]
    for depth in definition.id_depths + definition.ood_depths:
        for _ in range(per_depth):
            semantic = definition.sample(rng, depth)
            yield depth, semantic, definition.solve(semantic)


# --- independent oracles that read only the token sequence ------------------------------------


def _edges_from_tokens(tokens, width):
    body = tokens[3 : 3 + T.NODES * width]
    assert tokens[2] == T.MAP_MARKER
    assert [body[node * width] for node in range(T.NODES)] == [
        T.NODE_BASE + node for node in range(T.NODES)
    ]
    return body, list(tokens[3 + T.NODES * width :])


def _pointer_oracle(tokens, family):
    width = 3 if family == "gated_pointer_chase" else 2
    body, rest = _edges_from_tokens(tokens, width)
    destinations = [body[node * width + 1] for node in range(T.NODES)]
    held = set()
    if family == "gated_pointer_chase":
        assert rest[0] == T.GRANTS_MARKER
        held = {index + 1 for index, token in enumerate(rest[1:4]) if (token - T.GRANT_BASE) % 2}
        rest = rest[4:]
    assert rest[0] == T.START_MARKER and rest[2] == T.HOPS_MARKER
    node, hops = rest[1] - T.NODE_BASE, rest[3] - T.NUMBER_BASE
    for _ in range(hops):
        if destinations[node] == T.UNKNOWN_TOKEN:
            return T.ABSTAIN_TOKEN
        if family == "gated_pointer_chase":
            lock = body[node * width + 2] - T.LOCK_BASE
            if lock and lock not in held:
                return T.REFUSE_TOKEN
        node = destinations[node] - T.NODE_BASE
    return T.POINTER_TARGET_BASE + node


def _reach_oracle(tokens):
    body, rest = _edges_from_tokens(tokens, 3)
    assert (rest[0], rest[2], rest[4]) == (T.START_MARKER, T.GOAL_MARKER, T.HORIZON_MARKER)
    reached = {rest[1] - T.NODE_BASE}
    for _ in range(rest[5] - T.NUMBER_BASE):
        reached |= {
            token - T.NODE_BASE
            for node in reached
            for token in body[node * 3 + 1 : node * 3 + 3]
            if token != T.NO_EDGE_TOKEN
        }
    return T.REACH_YES_TOKEN if rest[3] - T.NODE_BASE in reached else T.REACH_NO_TOKEN


def _sudoku_oracle(tokens):
    assert tokens[2] == T.SUDOKU_MARKER and tokens[19] == T.SUDOKU_QUERY_MARKER
    grid = [token - T.SUDOKU_INPUT_BASE for token in tokens[3:19]]
    solutions = []

    def search(cell):
        if cell == 16:
            solutions.append(list(grid))
            return
        if grid[cell]:
            search(cell + 1)
            return
        for value in range(1, 5):
            if all(grid[peer] != value for peer in T.SUDOKU_PEERS[cell]):
                grid[cell] = value
                search(cell + 1)
                grid[cell] = 0

    search(0)
    assert len(solutions) == 1
    return T.SUDOKU_TARGET_BASE + solutions[0][tokens[20] - T.NUMBER_BASE]


def _modular_oracle(tokens):
    value = tokens[3] - T.NUMBER_BASE
    modulus = tokens[5] - T.NUMBER_BASE
    position = 6
    while tokens[position] == T.OPERATION_MARKER:
        value = ((tokens[position + 1] - T.NUMBER_BASE) * value + tokens[position + 2] - T.NUMBER_BASE) % modulus
        position += 3
    return T.MODULAR_TARGET_BASE + value


ORACLES = {
    "pointer_chase": lambda tokens: _pointer_oracle(tokens, "pointer_chase"),
    "gated_pointer_chase": lambda tokens: _pointer_oracle(tokens, "gated_pointer_chase"),
    "masked_pointer_chase": lambda tokens: _pointer_oracle(tokens, "masked_pointer_chase"),
    "reachability": _reach_oracle,
    "sudoku4": _sudoku_oracle,
    "modular_recurrence": _modular_oracle,
}


def test_tokens_alone_determine_the_stored_answer_at_every_depth() -> None:
    assert set(ORACLES) == set(FAMILIES)
    for name, family in FAMILIES.items():
        for depth, semantic, solution in _samples(name, per_depth=40):
            tokens = T.pack(name, family.encode(semantic))
            assert len(tokens) == T.SEQUENCE_LENGTH
            assert tokens[:2] == (T.BOS_TOKEN, family.task_token)
            assert tokens[-1] == T.ANSWER_TOKEN
            assert solution.depth == depth
            assert solution.target_token in family.answer_tokens
            assert ORACLES[name](tokens) == solution.target_token


def test_traces_follow_the_step_semantics_of_each_family() -> None:
    for depth, semantic, solution in _samples("gated_pointer_chase", per_depth=30):
        trace, first = solution.trace, solution.facts["first_violation_step"]
        assert trace["cur_node"][0] == semantic["start"] and len(trace["cur_node"]) == depth + 1
        assert trace["remaining"] == list(range(depth, -1, -1))
        assert trace["latch"] == [int(first is not None and step >= first) for step in range(depth + 1)]
        assert (solution.target_token == T.REFUSE_TOKEN) == (first is not None)
    for depth, semantic, solution in _samples("masked_pointer_chase", per_depth=30):
        touched = solution.facts["touched_step"]
        assert semantic["mapping"].count(-1) == T.MASKED_ENTRIES
        assert sorted(solution.trace["unknown_touched"]) == solution.trace["unknown_touched"]
        assert [node is None for node in solution.trace["cur_node"]] == [
            touched is not None and step >= touched for step in range(depth + 1)
        ]
    for depth, semantic, solution in _samples("reachability", per_depth=30):
        frontier, visited = solution.trace["frontier"], solution.trace["visited"]
        assert frontier[0] == [semantic["start"]]
        for step in range(depth + 1):
            assert sorted(node for layer in frontier[: step + 1] for node in layer) == visited[step]
        distance = solution.facts["goal_distance"]
        assert solution.trace["goal_found"] == [
            int(distance is not None and distance <= step) for step in range(depth + 1)
        ]
    for depth, semantic, solution in _samples("sudoku4", per_depth=30):
        rounds = solution.trace["candidates"]
        query = semantic["query"]
        for before, after in zip(rounds, rounds[1:]):
            assert all(new & old == new for old, new in zip(before, after))
        singles = [bin(candidates[query]).count("1") == 1 for candidates in rounds]
        assert singles.index(True) == depth
        assert solution.trace["determined"][-1] == 16


def test_gated_refusals_are_balanced_and_invisible_to_lock_counts() -> None:
    rng = random.Random(31)
    family = FAMILIES["gated_pointer_chase"]
    refused, steps, by_grant = [], Counter(), Counter()
    unheld = {True: [], False: []}
    locked = {True: [], False: []}
    for _ in range(6000):
        semantic = family.sample(rng, 4)
        solution = family.solve(semantic)
        label = solution.target_token == T.REFUSE_TOKEN
        refused.append(label)
        grants = tuple(semantic["grants"])
        by_grant[(grants, label)] += 1
        if label:
            steps[solution.facts["first_violation_step"]] += 1
            assert solution.facts["path_violations"] == 1
        if grants != T.LOCK_CLASSES:
            unheld[label].append(sum(lock not in (0, *grants) for lock in semantic["locks"]))
            locked[label].append(sum(lock != 0 for lock in semantic["locks"]))
    mean = lambda values: sum(values) / len(values)
    assert abs(mean(refused) - 0.5) < 0.03
    assert by_grant[(T.LOCK_CLASSES, True)] == 0
    assert max(steps.values()) - min(steps.values()) < 0.2 * sum(steps.values()) / 4
    assert abs(mean(unheld[True]) - mean(unheld[False])) < 0.15
    assert abs(mean(locked[True]) - mean(locked[False])) < 0.15


def test_masked_and_reachability_labels_are_balanced() -> None:
    abstain = [
        solution.target_token == T.ABSTAIN_TOKEN
        for _, _, solution in _samples("masked_pointer_chase", per_depth=150)
    ]
    assert abs(sum(abstain) / len(abstain) - 0.5) < 0.05
    cases = Counter()
    for depth, _, solution in _samples("reachability", per_depth=150):
        distance = solution.facts["goal_distance"]
        if solution.target_token == T.REACH_YES_TOKEN:
            cases["yes_tight" if distance == depth else "yes_early"] += 1
        else:
            cases["no_near" if distance == depth + 1 else "no_far"] += 1
    total = sum(cases.values())
    assert abs((cases["yes_tight"] + cases["yes_early"]) / total - 0.5) < 0.05
    assert cases["yes_tight"] / total > 0.2 and cases["no_near"] / total > 0.2


def test_invalid_instances_are_rejected() -> None:
    cycle = {"mapping": [1, 0, *range(2, T.NODES)], "start": 0, "hops": 3}
    unresolved = {"puzzle": [0] * 16, "query": 0}
    overlong = [T.NODE_BASE] * T.SEQUENCE_LENGTH
    for check in (
        lambda: FAMILIES["pointer_chase"].solve(cycle),
        lambda: FAMILIES["sudoku4"].solve(unresolved),
        lambda: T.pack("pointer_chase", overlong),
    ):
        try:
            check()
        except InvalidInstance:
            continue
        raise AssertionError("invalid instance was accepted")


def _changed_keys(base: TraceExample, sibling: TraceExample) -> set[str]:
    return {key for key in base.semantic if base.semantic[key] != sibling.semantic[key]}


def test_siblings_edit_one_field_and_record_where_traces_diverge() -> None:
    expected_field = {
        "edge_rewire": {"mapping", "edges"},
        "hops": {"hops", "operations"},
        "horizon": {"horizon"},
        "start": {"start"},
        "goal": {"goal"},
        "grant_flip": {"grants"},
        "lock_move": {"locks"},
        "mask_move": {"mapping"},
        "clue_move": {"puzzle"},
        "query": {"query"},
        "op_change": {"operations"},
    }
    kinds = Counter()
    for name in FAMILIES:
        examples = {example.example_id: example for example in _tiny(name)}
        for sibling in examples.values():
            if sibling.sibling is None:
                continue
            base = examples[sibling.sibling["base_id"]]
            kind = sibling.sibling["kind"]
            kinds[(name, kind)] += 1
            changed = _changed_keys(base, sibling)
            assert len(changed) == 1 and changed <= expected_field[kind], (name, kind, changed)
            assert (sibling.split, sibling.group_id) == (base.split, base.group_id)
            assert base.sibling is None and base.split in _config().sibling_splits
            assert sibling.sibling["target_changed"] == (sibling.target_token != base.target_token)
            assert sibling.sibling["diverges_at"] == {
                key: first_difference(values, sibling.trace[key]) for key, values in base.trace.items()
            }
            if kind == "edge_rewire" and name in ("pointer_chase", "gated_pointer_chase"):
                assert sibling.sibling["diverges_at"]["cur_node"] == sibling.sibling["step"]
            if kind == "lock_move":
                assert sorted(base.semantic["locks"]) == sorted(sibling.semantic["locks"])
                assert sibling.sibling["target_changed"]
            if kind == "grant_flip":
                assert len(set(base.semantic["grants"]) ^ set(sibling.semantic["grants"])) == 1
            if kind in ("mask_move", "goal"):
                assert sibling.sibling["target_changed"]
            if kind == "clue_move":
                clues = lambda example: sum(bool(value) for value in example.semantic["puzzle"])
                assert clues(base) == clues(sibling)
    for name, kind in (
        ("pointer_chase", "edge_rewire"),
        ("pointer_chase", "hops"),
        ("pointer_chase", "start"),
        ("gated_pointer_chase", "grant_flip"),
        ("gated_pointer_chase", "lock_move"),
        ("masked_pointer_chase", "mask_move"),
        ("reachability", "horizon"),
        ("reachability", "goal"),
        ("reachability", "edge_rewire"),
        ("sudoku4", "clue_move"),
    ):
        assert kinds[(name, kind)] >= 24, (name, kind, kinds[(name, kind)])
    # Every gated base carries a one-grant sibling, as the program requires.
    assert kinds[("gated_pointer_chase", "grant_flip")] == 48


def test_fixed_splits_are_region_disjoint_depth_balanced_and_deterministic() -> None:
    config = _config()
    for name, family in FAMILIES.items():
        examples = _tiny(name)
        assert examples == tuple(iter_family_examples(name, config))
        assert len({example.fingerprint for example in examples}) == len(examples)
        for example in examples:
            own_region = region_of(T.structure_hash(family, example.semantic), config.regions)
            assert own_region != "train"
            if example.sibling is None:
                assert own_region == example.split
                assert example.group_id == T.structure_hash(family, example.semantic)
        bases = [example for example in examples if example.sibling is None]
        for split, count in TINY_COUNTS.items():
            depths = family.ood_depths if split == config.ood_split else family.id_depths
            by_depth = Counter(example.difficulty for example in bases if example.split == split)
            if not depths:
                assert not by_depth
            elif name in T.CORE_FAMILIES:
                assert sum(by_depth.values()) == count
                assert set(by_depth) == set(depths)
                assert max(by_depth.values()) - min(by_depth.values()) <= 1


def test_training_stream_is_deterministic_and_never_meets_a_fixed_split() -> None:
    config = _config()
    for name in T.CORE_FAMILIES:
        family = FAMILIES[name]
        fixed = {example.fingerprint for example in _tiny(name)}
        stream = [train_example(name, index, config) for index in range(200)]
        assert stream == [train_example(name, index, config) for index in range(200)]
        for example in stream:
            assert example.split == "train" and example.fingerprint not in fixed
            assert region_of(example.group_id, config.regions) == "train"
            assert example.difficulty in family.id_depths
        assert len({example.difficulty for example in stream}) == len(family.id_depths)


def test_staircase_stream_gives_zero_hop_examples_that_answer_the_start_node() -> None:
    config = _config()
    for name in ("pointer_chase", "gated_pointer_chase", "masked_pointer_chase"):
        assert FAMILIES[name].staircase_depths == (0,)
        for index in range(40):
            example = train_example(name, index, config, staircase=True)
            assert example.difficulty == 0
            assert example.target_token == T.POINTER_TARGET_BASE + example.semantic["start"]
            assert ORACLES[name](example.tokens) == example.target_token
            assert region_of(example.group_id, config.regions) == "train"
    assert all(FAMILIES[name].staircase_depths == () for name in ("reachability", "sudoku4"))


def test_materialized_shards_round_trip_and_reject_tampering(tmp_path: Path) -> None:
    config = BundleConfig.from_mapping({**_config().to_dict(), "families": ["sudoku4", "reachability"]})
    manifest = materialize_bundle(config, tmp_path, write_splits=["validation", "locked"])
    again = materialize_bundle(config, tmp_path / "again", write_splits=[])

    assert manifest["bundle_hash"] == again["bundle_hash"]
    assert "ood_stress" not in manifest["families"]["sudoku4"]
    assert manifest["families"]["reachability"]["ood_stress"]["written"] is False
    assert manifest["families"]["reachability"]["locked"]["sibling_rows"] > 0
    assert manifest["families"]["reachability"]["monitor_train"]["sibling_rows"] == 0
    assert all(
        shard["shortfall"] == 0 for shards in manifest["families"].values() for shard in shards.values()
    )
    loaded = read_shard(tmp_path, "validation", "reachability")
    assert loaded == tuple(
        sorted(
            (example for example in _tiny("reachability") if example.split == "validation"),
            key=lambda example: example.example_id,
        )
    )
    shard = tmp_path / "validation" / "reachability.jsonl"
    shard.write_bytes(shard.read_bytes() + b"\n")
    try:
        read_shard(tmp_path, "validation", "reachability")
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("tampered task shard was accepted")
