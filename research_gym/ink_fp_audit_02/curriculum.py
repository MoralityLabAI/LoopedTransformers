"""DataGoblin curriculum interface: provenance-carrying tasks, partitions, leakage guard.

Every Task carries source root, source world, family, generator identity and version,
difficulty, verifier identity, expected answer or certificate, and estimated verifier cost.
Partition is a function of the source *world* (not the task), so transformed descendants of a
world always land in the same partition. Sealed tasks come from a disjoint seed range and are
never materialised by training code paths.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator

PARTITIONS = ("train", "dev", "accept", "sealed")
# Disjoint root ranges per partition; partition also re-derived from world hash and checked.
ROOT_BASE = {"train": 10_000_000, "dev": 20_000_000, "accept": 30_000_000, "sealed": 40_000_000}


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    prompt: str
    answer: str
    source_root: int
    source_world: str
    source_repo: str
    generator: str
    generator_version: str
    difficulty: float
    verifier: str
    verify_cost_ms: float
    partition: str
    certificate: dict = field(default_factory=dict)

    def receipt(self) -> dict:
        d = asdict(self)
        d.pop("prompt")
        d["prompt_sha256"] = hashlib.sha256(self.prompt.encode()).hexdigest()
        return d

    def wrong_answer(self, rng) -> str:
        if re.fullmatch(r"[+-]?\d+", self.answer):
            return str(int(self.answer) + rng.choice([-2, -1, 1, 2, 3]))
        return self.answer + "x"


def semantic_id(family: str, canonical_content) -> str:
    """Identity of the underlying problem independent of surface wording or seed."""
    blob = json.dumps([family, canonical_content], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


# ---- verification ----------------------------------------------------------------------
_TERMINATORS = ("<|message_end|>", "<|endoftext|>", "<|im_end|>")


def final_answer(text: str) -> tuple[str | None, bool]:
    cleaned = text
    for t in _TERMINATORS:
        cleaned = cleaned.replace(t, "")
    cleaned = cleaned.strip()
    lines = re.findall(r"(?m)^ANSWER:[ \t]*(.+?)[ \t]*$", cleaned)
    last = re.search(r"(?:^|\n)ANSWER:[ \t]*(.+?)[ \t]*$", cleaned)
    if len(lines) != 1 or last is None:
        return None, False
    return last.group(1).strip(), True


def repeated(text: str, n: int = 8, k: int = 5) -> bool:
    words = text.split()
    grams: dict = {}
    for i in range(max(0, len(words) - n + 1)):
        g = tuple(words[i:i + n])
        grams[g] = grams.get(g, 0) + 1
        if grams[g] >= k:
            return True
    return False


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


VERIFIERS: dict[str, Callable[[Task, str], bool]] = {
    "exact_int.v1": lambda t, a: bool(re.fullmatch(r"[+-]?\d+", a)) and int(a) == int(t.answer),
    "exact_token.v1": lambda t, a: _norm(a) == _norm(t.answer),
    "exact_set.v1": lambda t, a: sorted(_norm(x) for x in a.split(",")) == sorted(_norm(x) for x in t.answer.split(",")),
}


def verify(task: Task, text: str) -> dict:
    ans, completed = final_answer(text)
    rep = repeated(text)
    ok = bool(completed and not rep and ans is not None and VERIFIERS[task.verifier](task, ans))
    return {"correct": ok, "completed": completed, "repetition": rep, "parsed_answer": ans}


# ---- registry ---------------------------------------------------------------------------
GENERATORS: dict[str, Callable[[int, str], Task]] = {}


def register(family: str):
    def deco(fn):
        GENERATORS[family] = fn
        return fn
    return deco


def world_partition(world: str) -> str:
    """Partition implied by the root range encoded in the world id 'family:root'."""
    root = int(world.rsplit(":", 1)[1])
    for name in reversed(PARTITIONS):
        if root >= ROOT_BASE[name]:
            return name
    raise ValueError("world root outside registered ranges")


def generate(family: str, partition: str, n: int, offset: int = 0) -> Iterator[Task]:
    if partition not in PARTITIONS:
        raise ValueError(partition)
    for i in range(offset, offset + n):
        t = GENERATORS[family](ROOT_BASE[partition] + i, partition)
        if world_partition(t.source_world) != partition:
            raise AssertionError("generator broke the world/partition contract")
        yield t


def leakage_report(partitions: dict[str, list[Task]]) -> dict:
    """Cross-partition collisions on source world, semantic task id, prompt or certificate."""
    keys = {"source_world": lambda t: t.source_world, "task_id": lambda t: t.task_id,
            "prompt": lambda t: hashlib.sha256(_norm(t.prompt).encode()).hexdigest(),
            "certificate": lambda t: json.dumps(t.certificate, sort_keys=True) if t.certificate.get("proof") else None}
    seen: dict = {k: {} for k in keys}
    collisions = []
    for part, tasks in partitions.items():
        for t in tasks:
            for k, fn in keys.items():
                v = fn(t)
                if v is None:
                    continue
                other = seen[k].setdefault(v, part)
                if other != part:
                    collisions.append({"key": k, "a": other, "b": part, "task_id": t.task_id})
    return {"collisions": collisions, "ok": not collisions,
            "counts": {p: len(ts) for p, ts in partitions.items()}}


def manifest(families: list[str], counts: dict[str, int]) -> dict:
    """Versioned curriculum manifest; sealed partition is counted, never materialised here."""
    from . import gyms  # noqa: F401  registers generators
    rows = []
    for f in families:
        t = GENERATORS[f](ROOT_BASE["train"], "train")
        rows.append({"family": f, "generator": t.generator, "generator_version": t.generator_version,
                     "verifier": t.verifier, "source_repo": t.source_repo})
    m = {"schema": "ink-fp-audit-02.curriculum.v1", "families": rows, "counts": counts,
         "root_base": ROOT_BASE, "partition_rule": "world root range; descendants inherit world"}
    m["manifest_sha256"] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
    return m


def build_partitions(families: list[str], counts: dict[str, int], max_tries: int = 20) -> dict:
    """Unique tasks per family and partition, with semantic exclusion across partitions.

    Filled in priority order sealed -> accept -> dev -> train, so a held-out problem is never
    displaced by a training duplicate. Sealed tasks contribute only their semantic ids here; the
    sealed prompts are not returned.
    """
    from . import gyms  # noqa: F401  registers generators
    used: set[str] = set()
    out: dict[str, list[Task]] = {}
    shortfall = {}
    for part in ("sealed", "accept", "dev", "train"):
        n = counts.get(part, 0)
        out[part] = []
        for fam in families:
            got, offset = [], 0
            while len(got) < n and offset < n * max_tries:
                for t in generate(fam, part, n, offset):
                    if t.task_id not in used and len(got) < n:
                        used.add(t.task_id)
                        got.append(t)
                offset += n
            if len(got) < n:
                shortfall[f"{part}/{fam}"] = n - len(got)
            out[part].extend(got)
    sealed_ids = sorted(t.task_id for t in out.pop("sealed"))
    return {"partitions": out, "sealed_task_ids": sealed_ids, "shortfall": shortfall}
