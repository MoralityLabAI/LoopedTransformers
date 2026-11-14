"""TFH-01: exact Hall deficit, maximum b-matching and small-fibre sampling.

Input contains original-Lean TRAIN-only compiler-progress edges. Unknown
cells are omitted from the certified incidence matrix; no heldout data is
loaded. Full-fibre uniform sampling uses exact integer continuation counts.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Iterable


@dataclass(frozen=True)
class Fibre:
    states: tuple[str, ...]
    families: tuple[str, ...]
    quotas: tuple[int, ...]
    legal: tuple[frozenset[int], ...]

    @classmethod
    def from_dict(cls, data: dict) -> "Fibre":
        if data.get("split") != "train":
            raise ValueError("only explicitly training-only incidence data is accepted")
        quotas_raw = data["quotas"]
        if not isinstance(quotas_raw, dict) or not quotas_raw:
            raise ValueError("nonempty quota dict required")
        families = tuple(sorted(quotas_raw))
        quotas = tuple(quotas_raw[f] for f in families)
        if any(type(q) is not int or q < 0 for q in quotas):
            raise ValueError("quotas must be nonnegative integers")
        rows = data["states"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("nonempty state list required")
        states = tuple(row["id"] for row in rows)
        if any(not isinstance(s, str) or not s for s in states) or len(set(states)) != len(states):
            raise ValueError("state IDs must be distinct nonempty strings")
        if sum(quotas) != len(states):
            raise ValueError("quota total must equal distinct state count")
        index = {name: j for j, name in enumerate(families)}
        legal = []
        for row in rows:
            if row.get("split", "train") != "train":
                raise ValueError("heldout state in training graph")
            names = row["allowed"]
            if not isinstance(names, list) or any(name not in index for name in names):
                raise ValueError("unknown action family")
            legal.append(frozenset(index[name] for name in names))
        return cls(states, families, quotas, tuple(legal))

    def to_dict(self) -> dict:
        return {
            "split": "train",
            "quotas": dict(zip(self.families, self.quotas)),
            "states": [
                {"id": self.states[i], "allowed": [self.families[j] for j in sorted(row)]}
                for i, row in enumerate(self.legal)
            ],
        }


def from_compiler_probes(rows: Iterable[dict], quotas: dict[str, int]) -> Fibre:
    """Only actual, accepted, progressing original-Lean TRAIN probes make edges."""
    states: dict[str, set[str]] = {}
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("all compiler probes require split=train")
        if row["family"] not in quotas:
            continue
        state = states.setdefault(row["stage_id"], set())
        if (row.get("source_kind") == "original_lean"
                and row.get("compiler_accepted") is True
                and row.get("progress") is True):
            state.add(row["family"])
    return Fibre.from_dict({
        "split": "train", "quotas": quotas,
        "states": [{"id": sid, "allowed": sorted(allowed)}
                   for sid, allowed in sorted(states.items())],
    })


def hall_deficit(fibre: Fibre) -> dict:
    """Exact max over action subsets of (quota - distinct eligible states)_+."""
    k = len(fibre.families)
    if k > 16:
        raise ValueError("exhaustive Hall cut supports at most 16 families")
    best, best_mask, best_neighbors = 0, 0, set()
    for mask in range(1, 1 << k):
        quota = sum(fibre.quotas[j] for j in range(k) if mask & (1 << j))
        neighbors = {i for i, row in enumerate(fibre.legal)
                     if any(mask & (1 << j) for j in row)}
        shortage = quota - len(neighbors)
        if shortage > best:
            best, best_mask, best_neighbors = shortage, mask, neighbors
    return {
        "minimum_unfilled_slots": best,
        "witness_families": [fibre.families[j] for j in range(k) if best_mask & (1 << j)],
        "witness_quota": sum(fibre.quotas[j] for j in range(k) if best_mask & (1 << j)),
        "witness_neighbors": [fibre.states[i] for i in sorted(best_neighbors)],
        "total_variation_lower_bound": {"numerator": best, "denominator": len(fibre.states)},
    }


def maximum_assignment(fibre: Fibre) -> dict[str, str]:
    """Independent augmenting-path certificate for a maximum b-matching."""
    slots = [j for j, quota in enumerate(fibre.quotas) for _ in range(quota)]
    by_family = [[s for s, fam in enumerate(slots) if fam == j]
                 for j in range(len(fibre.families))]
    occupant: list[int | None] = [None] * len(slots)

    def augment(i: int, visited: set[int]) -> bool:
        for family in sorted(fibre.legal[i]):
            for slot in by_family[family]:
                if slot in visited:
                    continue
                visited.add(slot)
                if occupant[slot] is None or augment(occupant[slot], visited):
                    occupant[slot] = i
                    return True
        return False

    for i in range(len(fibre.states)):
        augment(i, set())
    return {fibre.states[i]: fibre.families[slots[s]]
            for s, i in enumerate(occupant) if i is not None}


def _counter(fibre: Fibre):
    @lru_cache(None)
    def count(i: int, rem: tuple[int, ...]) -> int:
        if i == len(fibre.states):
            return int(all(q == 0 for q in rem))
        total = 0
        for j in sorted(fibre.legal[i]):
            if rem[j]:
                nxt = list(rem)
                nxt[j] -= 1
                total += count(i + 1, tuple(nxt))
        return total
    return count


def exact_count(fibre: Fibre, max_states: int = 16) -> int:
    if len(fibre.states) > max_states:
        raise ValueError("exact counting cap exceeded; no mixing claim is made")
    return _counter(fibre)(0, fibre.quotas)


def sample_uniform(fibre: Fibre, seed: int, max_states: int = 16) -> dict[str, str]:
    """Ideal uniform integer branching, reproducible seeded pseudorandom draws."""
    if len(fibre.states) > max_states:
        raise ValueError("exact uniform sampler cap exceeded")
    count = _counter(fibre)
    rem = fibre.quotas
    if count(0, rem) == 0:
        raise ValueError("structural zeros forbid a full matching")
    rng = random.Random(seed)
    result = {}
    for i, sid in enumerate(fibre.states):
        draw = rng.randrange(count(i, rem))
        for j in sorted(fibre.legal[i]):
            if rem[j] == 0:
                continue
            nxt = list(rem)
            nxt[j] -= 1
            nxt = tuple(nxt)
            ways = count(i + 1, nxt)
            if draw < ways:
                result[sid] = fibre.families[j]
                rem = nxt
                break
            draw -= ways
        else:
            raise AssertionError("weighted branching failed")
    assert all(q == 0 for q in rem)
    return result


def legal_pair_swaps(fibre: Fibre, assigned: dict[str, str]) -> list[dict[str, str]]:
    indices = {f: j for j, f in enumerate(fibre.families)}
    moves = []
    for i, j in itertools.combinations(range(len(fibre.states)), 2):
        a, b = fibre.states[i], fibre.states[j]
        ai, bj = assigned[a], assigned[b]
        if ai != bj and indices[bj] in fibre.legal[i] and indices[ai] in fibre.legal[j]:
            swapped = dict(assigned)
            swapped[a], swapped[b] = bj, ai
            moves.append(swapped)
    return moves


def six_cycle_fixture() -> Fibre:
    return Fibre.from_dict({
        "split": "train", "quotas": {"A": 1, "B": 1, "C": 1},
        "states": [{"id": "row0", "allowed": ["A", "B"]},
                   {"id": "row1", "allowed": ["B", "C"]},
                   {"id": "row2", "allowed": ["C", "A"]}],
    })


def published_fc12_capacity() -> dict:
    """Use published success *attempt* counts as support upper bounds only."""
    counts = {"exact_local": 1, "rw_local": 13,
              "rcases_local": 1, "induction_list": 30}
    target = {"exact_local": 13, "rw_local": 12,
              "rcases_local": 13, "induction_list": 13}
    assert sum(target.values()) == 51
    deficit = target["exact_local"] + target["rcases_local"] - (
        counts["exact_local"] + counts["rcases_local"])
    assert deficit == 24
    return {
        "mode": "aggregate_capacity_only",
        "source": "experiments/refutation_transfer/FC12_FIRST_RUN_20261114.md",
        "train_distinct_goal_states": 51,
        "positive_compiler_attempts": counts,
        "hypothetical_four_family_distinct_state_quotas": target,
        "witness_subset": ["exact_local", "rcases_local"],
        "witness_quota": 26,
        "maximum_distinct_neighbors_from_attempt_counts": 2,
        "minimum_additional_successfully_labeled_distinct_states": deficit,
        "tv_lower_bound_for_one_pass_assignment": {"numerator": deficit, "denominator": 51},
        "does_not_reconstruct_unreleased_training_incidence": True,
    }


def audit(fibre: Fibre, max_exact_states: int = 16) -> dict:
    witness = hall_deficit(fibre)
    matching = maximum_assignment(fibre)
    if len(matching) != len(fibre.states) - witness["minimum_unfilled_slots"]:
        raise AssertionError("matching disagrees with the exact Hall cut")
    receipt = {
        "mode": "exact_full_incidence",
        "input_sha256": hashlib.sha256(json.dumps(fibre.to_dict(), sort_keys=True,
                                                  separators=(",", ":")).encode()).hexdigest(),
        "distinct_states": len(fibre.states),
        "quotas": dict(zip(fibre.families, fibre.quotas)),
        "maximum_filled_slots": len(matching),
        "hall_certificate": witness,
        "matching": matching,
    }
    if len(fibre.states) <= max_exact_states:
        receipt["exact_full_assignment_count"] = exact_count(fibre, max_exact_states)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path)
    mode.add_argument("--probe-jsonl", type=Path)
    mode.add_argument("--six-cycle", action="store_true")
    mode.add_argument("--fc12-published", action="store_true")
    parser.add_argument("--quotas", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--max-exact-states", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.fc12_published:
        result = published_fc12_capacity()
    else:
        if args.six_cycle:
            fibre = six_cycle_fixture()
        elif args.input:
            fibre = Fibre.from_dict(json.loads(args.input.read_text(encoding="utf-8")))
        else:
            if args.quotas is None:
                parser.error("--probe-jsonl requires --quotas")
            rows = [json.loads(s) for s in args.probe_jsonl.read_text(encoding="utf-8").splitlines()
                    if s.strip()]
            fibre = from_compiler_probes(rows, json.loads(args.quotas.read_text(encoding="utf-8")))
        result = audit(fibre, args.max_exact_states)
        if args.sample:
            result["uniform_sample"] = sample_uniform(fibre, args.seed, args.max_exact_states)
        if args.six_cycle:
            a = {"row0": "A", "row1": "B", "row2": "C"}
            b = {"row0": "B", "row1": "C", "row2": "A"}
            result["two_matching_witness"] = [a, b]
            result["elementary_pair_swaps"] = [len(legal_pair_swaps(fibre, a)),
                                               len(legal_pair_swaps(fibre, b))]
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
