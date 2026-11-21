"""NEC-03G: deterministic fresh Boolean-function/position world registry.

128 distinct four-variable truth functions, 128 unique focal-position sets,
independent training seeds, 64 development worlds and 64 evaluation-sealed
confirmation worlds. Confirmation models and outcomes are NEVER run here.
A task/world, including its trained checkpoint, is the unit of inference.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path

SCHEMA = "NEC-03G/fresh-composed-boolean-worlds/v1"
SALT = b"nec03g-worlds-20261121-v1"
OPS = ("and", "or", "xor", "nand", "nor", "xnor")
STRUCTURES = ("balanced", "left_nested", "right_nested")
DEV_COUNT = 64
CONFIRM_COUNT = 64
COUNTS = {"train": 160, "development": 48, "reserved": 48}
TRUNK_STEPS = 450
SAE_STEPS = 480
MIN_CLEAN_ACC = 0.90
PRIMARY_MEANINGFUL_DELTA = 0.02
TOPK = 16
NATIVE_SELECTION_CONTEXTS = 80
FEATURE_SELECTION_CONTEXTS = 80


def encoded(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(obj: object) -> str:
    return sha256(encoded(obj)).hexdigest()


def gate(a: int, b: int, op: str) -> int:
    if op == "and":
        return a & b
    if op == "or":
        return a | b
    if op == "xor":
        return a ^ b
    if op == "nand":
        return 1 - (a & b)
    if op == "nor":
        return 1 - (a | b)
    if op == "xnor":
        return 1 - (a ^ b)
    raise ValueError("unknown Boolean gate")


def program_truth(structure: str, g1: str, g2: str, g3: str, inv: int) -> int:
    """Little-endian 16-bit truth table for a four-input Boolean program."""
    if structure not in STRUCTURES or inv not in range(16):
        raise ValueError("program outside registered grammar")
    table = 0
    for mask in range(16):
        a, b, c, d = [((mask >> i) & 1) ^ ((inv >> i) & 1) for i in range(4)]
        if structure == "balanced":
            result = gate(gate(a, b, g1), gate(c, d, g2), g3)
        elif structure == "left_nested":
            result = gate(gate(gate(a, b, g1), c, g2), d, g3)
        else:
            result = gate(a, gate(b, gate(c, d, g1), g2), g3)
        table |= int(result) << mask
    return table


def essential(table: int) -> bool:
    if not 0 <= table < 65536:
        raise ValueError("truth table requires 16 bits")
    return all(any(((table >> i) & 1) != ((table >> (i ^ (1 << bit))) & 1)
                   for i in range(16)) for bit in range(4))


def teacher_functions() -> list[dict]:
    """Source-independent grammar with one canonical program per truth table."""
    canonical = {}
    for structure, g1, g2, g3, inv in product(
            STRUCTURES, OPS, OPS, OPS, range(16)):
        truth = program_truth(structure, g1, g2, g3, inv)
        if not (5 <= truth.bit_count() <= 11) or not essential(truth):
            continue
        if truth in (0x6ca0, 0xe8e8):  # excludes two legacy-style patterns
            continue
        candidate = {"structure": structure, "gates": [g1, g2, g3],
                     "negated_inputs_mask": inv, "truth_table_16": truth}
        if truth not in canonical or encoded(candidate) < encoded(canonical[truth]):
            canonical[truth] = candidate
    items = list(canonical.values())
    items.sort(key=lambda item: sha256(SALT + b"|truth|" +
               str(item["truth_table_16"]).encode()).digest())
    if len(items) < DEV_COUNT + CONFIRM_COUNT:
        raise AssertionError("grammar lacks 128 distinct essential balanced functions")
    return items[:DEV_COUNT + CONFIRM_COUNT]


def focal_positions() -> list[tuple[int, ...]]:
    """Require memory-sensitive early bits and at least one later bit."""
    available = [
        p for p in combinations(range(12), 4)
        if p[0] <= 4 and p[3] >= 7 and sum(x <= 5 for x in p) >= 2
    ]
    available.sort(key=lambda p: sha256(
        SALT + b"|positions|" + bytes(p)).digest())
    if len(available) < 128:
        raise AssertionError("not enough unique position sets")
    return available[:128]


def worlds() -> list[dict]:
    functions = teacher_functions()
    positions = focal_positions()
    out = []
    for index in range(DEV_COUNT + CONFIRM_COUNT):
        cohort = "development" if index < DEV_COUNT else "confirmation"
        ordinal = index if cohort == "development" else index-DEV_COUNT
        world_id = f"{cohort[:3]}-{ordinal:03d}"
        program = functions[index]
        pos = list(positions[index])
        seed = 1 + int.from_bytes(sha256(
            SALT + b"|checkpoint|" + world_id.encode()).digest()[:4], "big") % 2100000000
        row = {"schema": SCHEMA, "world_id": world_id, "cohort": cohort,
               "world_index": index, "program": program,
               "focal_positions": pos,
               "additive_control_weights": [1, 2, 4, 8],
               "trunk_seed": seed,
               "context_salt": f"nec03g-context-v1:{world_id}",
               "trunk_steps": TRUNK_STEPS, "sae_steps": SAE_STEPS,
               "truth_table_16": program["truth_table_16"]}
        row["world_sha256"] = digest(row)
        out.append(row)
    if len({w["truth_table_16"] for w in out}) != 128:
        raise AssertionError("truth-function disjointness violated")
    if len({tuple(w["focal_positions"]) for w in out}) != 128:
        raise AssertionError("focal-position disjointness violated")
    if len({w["trunk_seed"] for w in out}) != 128:
        raise AssertionError("checkpoint seed collision")
    return out


def additive_control_truth(primary_truth: int) -> int:
    """Additive prethreshold baseline with identical positive class count.

    Unique weighted scores 1,2,4,8 allow a linear separator for ANY
    desired number of positives, matching the primary's class prior.
    """
    n = primary_truth.bit_count()
    if not 0 < n < 16:
        raise ValueError("degenerate teacher")
    return sum(1 << mask for mask in range(16-n, 16))


def target_bits(bits, world: dict, *, control: bool = False):
    """Boolean target for a torch [batch,12] long tensor, no fitting."""
    import torch
    if bits.ndim != 2 or bits.shape[1] != 12:
        raise ValueError("need twelve input bits")
    focal = world["focal_positions"]
    index = sum((bits[:,pos].long() << i for i,pos in enumerate(focal)))
    truth = additive_control_truth(world["truth_table_16"]) if control else world["truth_table_16"]
    return ((torch.full_like(index, truth) >> index) & 1).long()


def context_catalog(world: dict) -> tuple[dict, dict]:
    """World-specific 160/48/48 split; all 16 focal flips stay together."""
    remaining = [p for p in range(12) if p not in world["focal_positions"]]
    contexts = [tuple((x >> j) & 1 for j in range(8)) for x in range(256)]
    salt = world["context_salt"].encode()
    contexts.sort(key=lambda c: (sha256(salt + bytes(c)).digest(), bytes(c)))
    counts = dict(COUNTS)
    catalog = {k: [] for k in counts}
    owners = {}
    offset = 0
    for partition, n in counts.items():
        for context in contexts[offset:offset+n]:
            key = "".join(map(str, context))
            owners[key] = partition
            for focal in range(16):
                bits = [0]*12
                for j,pos in enumerate(remaining):
                    bits[pos] = context[j]
                for j,pos in enumerate(world["focal_positions"]):
                    bits[pos] = (focal >> j) & 1
                catalog[partition].append(tuple(bits))
        offset += n
    if offset != 256 or len(set.union(*(set(rows) for rows in catalog.values()))) != 4096:
        raise AssertionError("world context split fails complete 12-bit closure")
    manifest = {"schema": SCHEMA, "world_id": world["world_id"],
                "world_sha256": world["world_sha256"],
                "contexts_ordered": ["".join(map(str,c)) for c in contexts],
                "context_assignment": owners,
                "context_positions": remaining,
                "focal_positions": world["focal_positions"],
                "counts": counts,
                "no_context_crosses_split": True,
                "reserved_contexts_untouched_by_development": True}
    manifest["sha256"] = digest(manifest)
    return catalog, manifest


def registry() -> dict:
    all_worlds = worlds()
    dev = all_worlds[:DEV_COUNT]
    confirm = all_worlds[DEV_COUNT:]
    out = {"schema": SCHEMA, "salt_utf8": SALT.decode(),
           "development_count": DEV_COUNT,
           "confirmation_count": CONFIRM_COUNT,
           "dev_full_world_specs": dev,
           "confirmation_commitments": [{
                "world_id": w["world_id"],
                "world_sha256": w["world_sha256"]}
                for w in confirm],
           "confirmation_full_spec_sha256": digest(confirm),
           "development_full_spec_sha256": digest(dev),
           "shared_grammar_predefined": True,
           "function_truth_tables_disjoint_between_cohorts": True,
           "focal_position_quartets_disjoint_between_cohorts": True,
           "confirmation_outcomes_exposed": False,
           "current_evidence_class": "engineering-and-development-only",
           "confirmation_requires_separate_preflight_authorization": True,
           "planned_comparisons": 3,
           "desired_confirmation_roots": 64,
           "smallest_meaningful_primary_correctness_gain": "0.02",
           "world_unit": "one distinct Boolean truth function, focal-position quartet and trained checkpoint"}
    out["registry_sha256"] = digest(out)
    return out


def write_registry(out: Path) -> dict:
    item=registry()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_bytes(encoded(item))
    return item


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args=parser.parse_args()
    r=write_registry(args.out)
    print(json.dumps({"worlds_development": r["development_count"],
                      "worlds_confirmation_sealed": r["confirmation_count"],
                      "registry_sha256": r["registry_sha256"],
                      "full_confirmation_specs_exported": False},sort_keys=True))
