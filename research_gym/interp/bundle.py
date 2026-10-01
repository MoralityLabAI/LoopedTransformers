"""Hash-partitioned splits, minimal-pair siblings, and manifests for the RMP task bundle."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Iterator, Mapping, Sequence

from lsa.canonical import digest
from research_gym.integrity import canonical_file_sha256

from .tasks import (
    FAMILIES,
    Family,
    InvalidInstance,
    Solution,
    fingerprint,
    pack,
    structure_hash,
)


TRAIN_REGION = "train"
GENERATOR_SOURCES = ("research_gym/interp/tasks.py", "research_gym/interp/bundle.py")
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TraceExample:
    example_id: str
    fingerprint: str
    group_id: str
    family: str
    split: str
    tokens: tuple[int, ...]
    target_token: int
    difficulty: int
    semantic: Mapping[str, Any]
    trace: Mapping[str, Any]
    facts: Mapping[str, Any]
    sibling: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = dict(self.__dict__)
        value["tokens"] = list(self.tokens)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TraceExample":
        return cls(**{**value, "tokens": tuple(int(token) for token in value["tokens"])})


@dataclass(frozen=True)
class BundleConfig:
    bundle_id: str
    seed: int
    families: tuple[str, ...]
    counts: Mapping[str, int]
    regions: Mapping[str, float]
    sibling_splits: tuple[str, ...]
    ood_split: str

    def __post_init__(self) -> None:
        if abs(sum(self.regions.values()) - 1.0) > 1e-9 or TRAIN_REGION not in self.regions:
            raise ValueError("regions must include train and sum to one")
        unknown = [name for name in self.families if name not in FAMILIES]
        if unknown:
            raise ValueError(f"unknown task families: {unknown}")
        splits = set(self.counts) | set(self.sibling_splits) | {self.ood_split}
        if not splits <= set(self.regions) - {TRAIN_REGION}:
            raise ValueError("every fixed split needs its own structure-hash region")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "seed": self.seed,
            "families": list(self.families),
            "counts": dict(self.counts),
            "regions": dict(self.regions),
            "sibling_splits": list(self.sibling_splits),
            "ood_split": self.ood_split,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BundleConfig":
        return cls(
            bundle_id=str(value["bundle_id"]),
            seed=int(value["seed"]),
            families=tuple(value["families"]),
            counts={str(key): int(count) for key, count in value["counts"].items()},
            regions={str(key): float(share) for key, share in value["regions"].items()},
            sibling_splits=tuple(value["sibling_splits"]),
            ood_split=str(value["ood_split"]),
        )

    def scaled(self, counts: Mapping[str, int]) -> "BundleConfig":
        return BundleConfig.from_mapping({**self.to_dict(), "counts": dict(counts)})


def region_of(structure: str, regions: Mapping[str, float]) -> str:
    """Map a structure hash to the split that owns that part of hash space."""

    position = int(structure[:13], 16) / 16**13
    upper = 0.0
    names = sorted(regions)
    for name in names:
        upper += regions[name]
        if position < upper:
            return name
    return names[-1]


def first_difference(left: Sequence[Any], right: Sequence[Any]) -> int | None:
    """First step at which two trace variables differ, or where only one continues."""

    for step, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return step
    return None if len(left) == len(right) else min(len(left), len(right))


def build_example(
    family: Family,
    semantic: Mapping[str, Any],
    split: str,
    *,
    solution: Solution | None = None,
    group_id: str | None = None,
    sibling: Mapping[str, Any] | None = None,
) -> TraceExample:
    solution = solution or family.solve(semantic)
    identity = fingerprint(family, semantic)
    return TraceExample(
        example_id=f"{family.name}-{split}-{identity[:16]}",
        fingerprint=identity,
        group_id=group_id or structure_hash(family, semantic),
        family=family.name,
        split=split,
        tokens=pack(family.name, family.encode(semantic)),
        target_token=solution.target_token,
        difficulty=solution.depth,
        semantic=dict(semantic),
        trace=dict(solution.trace),
        facts=dict(solution.facts),
        sibling=sibling,
    )


def _trial_rng(seed: int, family: str, stream: str, trial: int) -> random.Random:
    material = digest({"seed": seed, "family": family, "stream": stream, "trial": trial})
    return random.Random(int(material[:16], 16))


def train_example(
    family_name: str, index: int, config: BundleConfig, *, staircase: bool = False
) -> TraceExample:
    """Training example ``index``: a pure function of the seed, family, and index.

    ``staircase`` draws from the family's easier-than-evaluated depths, in a separate stream.
    """

    family = FAMILIES[family_name]
    depths = family.staircase_depths if staircase else family.id_depths
    stream = "staircase" if staircase else TRAIN_REGION
    rng = _trial_rng(config.seed, family_name, stream, index)
    while True:
        semantic = family.sample(rng, rng.choice(depths))
        if region_of(structure_hash(family, semantic), config.regions) == TRAIN_REGION:
            return build_example(family, semantic, TRAIN_REGION)


def depth_quotas(count: int, depths: Sequence[int]) -> dict[int, int]:
    base, remainder = divmod(count, len(depths))
    return {depth: base + int(index < remainder) for index, depth in enumerate(depths)}


def _siblings(
    family: Family,
    base: TraceExample,
    rng: random.Random,
    depths: Sequence[int],
    config: BundleConfig,
    seen: set[str],
) -> Iterator[TraceExample]:
    solved = Solution(base.target_token, base.difficulty, base.trace, base.facts)
    for kind, info, alternatives in family.siblings(base.semantic, solved, rng, depths):
        for semantic in alternatives:
            try:
                solution = family.solve(semantic)
            except InvalidInstance:
                continue
            identity = fingerprint(family, semantic)
            # A sibling that edits the structure must not land where the training stream draws.
            if (
                solution.depth not in depths
                or identity in seen
                or region_of(structure_hash(family, semantic), config.regions) == TRAIN_REGION
            ):
                continue
            seen.add(identity)
            yield build_example(
                family,
                semantic,
                base.split,
                solution=solution,
                group_id=base.group_id,
                sibling={
                    "base_id": base.example_id,
                    "kind": kind,
                    **info,
                    "target_changed": solution.target_token != base.target_token,
                    "diverges_at": {
                        name: first_difference(values, solution.trace[name])
                        for name, values in base.trace.items()
                    },
                },
            )
            break


def iter_family_examples(
    family_name: str, config: BundleConfig, *, stale_limit: int = 200_000
) -> Iterator[TraceExample]:
    """Yield every fixed-split example of one family, each base followed by its siblings."""

    family = FAMILIES[family_name]
    seen: set[str] = set()
    in_distribution = [split for split in config.counts if split != config.ood_split]
    streams = (
        ("fixed", family.id_depths, in_distribution),
        ("ood", family.ood_depths, [config.ood_split]),
    )
    for stream, depths, splits in streams:
        if not depths:
            continue
        open_quotas = {
            (split, depth): count
            for split in splits
            for depth, count in depth_quotas(config.counts[split], depths).items()
        }
        remaining = sum(open_quotas.values())
        trial = stale = 0
        while remaining and stale < stale_limit:
            rng = _trial_rng(config.seed, family_name, stream, trial)
            depth = depths[trial % len(depths)]
            trial += 1
            stale += 1
            semantic = family.sample(rng, depth)
            split = region_of(structure_hash(family, semantic), config.regions)
            identity = fingerprint(family, semantic)
            if not open_quotas.get((split, depth)) or identity in seen:
                continue
            seen.add(identity)
            open_quotas[(split, depth)] -= 1
            remaining -= 1
            stale = 0
            example = build_example(family, semantic, split)
            yield example
            if split in config.sibling_splits:
                yield from _siblings(family, example, rng, depths, config, seen)


def _shard_stats(examples: Iterable[TraceExample], requested: int) -> dict[str, Any]:
    bases = [example for example in examples if example.sibling is None]
    siblings = [example for example in examples if example.sibling is not None]
    targets = Counter(example.target_token for example in bases)
    return {
        "base_rows": len(bases),
        "sibling_rows": len(siblings),
        "shortfall": requested - len(bases),
        "by_depth": dict(sorted(Counter(str(example.difficulty) for example in bases).items())),
        "best_constant": max(targets.values()) / len(bases) if bases else None,
        "sibling_kinds": dict(sorted(Counter(e.sibling["kind"] for e in siblings).items())),
    }


def _line(example: TraceExample) -> str:
    return json.dumps(example.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


def materialize_bundle(
    config: BundleConfig, directory: Path, *, write_splits: Sequence[str] | None = None
) -> dict[str, Any]:
    """Generate every fixed split, hash each shard, and write the requested ones."""

    directory.mkdir(parents=True, exist_ok=True)
    families: dict[str, dict[str, Any]] = {}
    for name in config.families:
        by_split: dict[str, list[TraceExample]] = {}
        for example in iter_family_examples(name, config):
            by_split.setdefault(example.split, []).append(example)
        shards = {}
        for split in sorted(config.counts):
            if split == config.ood_split and not FAMILIES[name].ood_depths:
                continue
            examples = by_split.get(split, [])
            examples.sort(key=lambda example: example.example_id)
            if len({example.example_id for example in examples}) != len(examples):
                raise ValueError(f"example IDs collide in {name}/{split}")
            payload = "".join(_line(example) + "\n" for example in examples).encode("utf-8")
            written = write_splits is None or split in write_splits
            if written:
                path = directory / split / f"{name}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            shards[split] = {
                "path": f"{split}/{name}.jsonl",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "written": written,
                **_shard_stats(examples, config.counts[split]),
            }
        families[name] = shards
    manifest = {
        "schema_version": 1,
        "config": config.to_dict(),
        "config_sha256": digest(config.to_dict()),
        "generator_sha256": {
            source: canonical_file_sha256(ROOT / source) for source in GENERATOR_SOURCES
        },
        "families": families,
    }
    manifest["bundle_hash"] = digest(
        {
            "config_sha256": manifest["config_sha256"],
            "shards": {
                name: {split: shard["sha256"] for split, shard in shards.items()}
                for name, shards in families.items()
            },
        }
    )
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def read_shard(directory: Path, split: str, family: str) -> tuple[TraceExample, ...]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    shard = manifest["families"][family][split]
    payload = (directory / shard["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != shard["sha256"]:
        raise ValueError(f"task shard hash mismatch: {split}/{family}")
    return tuple(
        TraceExample.from_mapping(json.loads(line))
        for line in payload.decode("utf-8").splitlines()
    )
