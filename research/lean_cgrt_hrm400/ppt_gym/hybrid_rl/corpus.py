"""Freeze development-only Lean policy cohorts without exposing source witnesses."""
from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from lcgrt import fragment

from ..data import canonical, digest, sha256_file

REPO = Path(__file__).resolve().parents[4]
BPD_GENERATOR = REPO / "experiments" / "bpd_lean_curriculum_v2" / "generate.py"
MATHCAN_INCIDENCE = REPO / "experiments" / "refutation_transfer" / "data" / "tfh01_fc12_verified_incidence.json"
SCHEMA = "ppt.hybrid_rl.cohort.v1"
DEADLINE_FAMILY = "first-choice-proof-deadline"
CONTINUATION_DEADLINE_FAMILY = "continuation-proof-deadline"
RECOVERY_FAMILY = "early-branch-recovery"
DEADLINE_FAMILIES = frozenset((DEADLINE_FAMILY, CONTINUATION_DEADLINE_FAMILY))
PAIRED_FAMILIES = frozenset(("first-choice-alias", "first-choice-conjunction", RECOVERY_FAMILY)) | DEADLINE_FAMILIES


def proposal_attempt(family):
    return 2 if family == CONTINUATION_DEADLINE_FAMILY else 1


@dataclass(frozen=True)
class SearchCase:
    source: str
    split: str
    family: str
    lineage: str
    cluster: str
    tid: str
    binders: str
    target: str

    @property
    def root(self):
        return digest([self.source, self.lineage, self.statement])

    @property
    def statement(self):
        return f"∀ {self.binders}, {self.target}"


def bpd_generator():
    loader = importlib.util.spec_from_file_location("ppt_bpd_lg02", BPD_GENERATOR)
    generator = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(generator)
    return generator


def _bpd():
    generator = bpd_generator()
    for (family, depth, shape), split in sorted(generator.assign_groups().items()):
        if family != "exact" or split not in ("train", "dev"):
            continue
        for variant in (0, 1):
            g = generator.graph_spec(family, depth, shape, 0, variant)
            if split == "dev":
                path = generator.directed_path(g["main"][0], g["main"][-1], g["edges"])
                if path is None or len(path) + 1 > 8:
                    continue
            tid = "thm_bpd2_" + generator.digest((family, depth, shape, 0, variant))[:16]
            raw, _ = generator.source(g, tid)
            # Public declaration only. Alpha-rename the fact binder to the existing
            # live Horn tactic grammar. Leave all graph edges and atom names intact.
            binders, target = raw.removeprefix(f"theorem {tid} ").removesuffix(" := by").rsplit(" : ", 1)
            if not re.fullmatch(r"\([^\n]+\)", binders) or not re.fullmatch(r"v[0-9a-f]+", target):
                raise ValueError("unexpected BPD exact declaration")
            binders = binders.replace("(hs : ", "(h9999 : ")
            yield SearchCase("bpd_lg02", split, family, g["alpha_hash"],
                             generator.digest(generator.group_signature(family, depth, shape)),
                             tid, binders, target)


def _horn(seed, train, dev):
    seen = set()
    for split, n, families in (("train", train, ("chain-dead", "chain-deep", "alt-dead")),
                               ("dev", dev, ("alt-cycle", "tree-deep"))):
        for i in range(n):
            family = families[i % len(families)]
            for attempt in range(128):
                draw = int(digest(["hybrid-rl", seed, split, i, attempt])[:16], 16)
                tid = f"hybrid_{split}_{seed}_{i}"
                problem = fragment.generate(draw, 2 + draw % 3, family, tid=tid)
                lineage = fragment.canonical_hash(problem)
                length = sum(fragment.reference_sizes(problem)[goal] for goal in problem.goal)
                length += int(len(problem.goal) > 1)
                if lineage not in seen and 3 <= length <= 8:
                    break
            else:
                raise ValueError("could not freeze a fresh budget-feasible Horn root")
            seen.add(lineage)
            yield SearchCase("loopt_horn", split, family, lineage, lineage, tid,
                             fragment.binders(problem), fragment.target(problem))


def freeze(path, seed, train, dev):
    rows = [*_horn(seed, train, dev), *_bpd()]
    record = {"schema": SCHEMA, "seed": seed, "cases": [asdict(case) for case in rows],
              "source_sha256": {"bpd_lg02": sha256_file(BPD_GENERATOR),
                                "loopt_horn": sha256_file(Path(fragment.__file__)),
                                "mathcan_train_incidence": sha256_file(MATHCAN_INCIDENCE)},
              "registration": {"purpose": "development pilot only", "unit": "source theorem cluster",
                               "primary": "hybrid RL versus TRM RL at common proof-check and MAC ceilings",
                               "meaningful_gain": 0.05, "mathcan_role": "provenance only, Lean 4.35 replay pending",
                               "confirmation": "unopened", "tactic_checks": 8}}
    with path.open("x", encoding="utf-8", newline="\n") as out:
        out.write(canonical(record) + "\n")


def load(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != SCHEMA:
        raise ValueError("unknown hybrid proof cohort schema")
    selected = {"train": [], "dev": []}
    seen = set()
    clusters = {}
    for raw in record["cases"]:
        case = SearchCase(**raw)
        if (case.source not in ("bpd_lg02", "loopt_horn", "evalmath_lg01") or case.split not in selected
                or (case.source == "evalmath_lg01" and case.split != "train")
                or case.lineage in seen or clusters.get(case.cluster, case.split) != case.split
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", case.tid)):
            raise ValueError("invalid identity, duplicate theorem or cross-split cluster")
        seen.add(case.lineage)
        clusters[case.cluster] = case.split
        selected[case.split].append(case)
    if not all(selected.values()):
        raise ValueError("empty train or development data")
    return selected
