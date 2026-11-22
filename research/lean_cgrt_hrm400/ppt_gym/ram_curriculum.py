"""Freeze generated Horn cohorts and define deterministic PPT RAM research schedules."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from lcgrt import fragment

from .data import canonical, digest, sha256_file
from .live_suite import LiveCase

SCHEMA = "ppt_gym.ram_curriculum.v1"
DOSES = (0, 1, 2, 4, 8, 16)


def freeze(path, seed, families, queries, rounds):
    chosen = fragment.TRAIN_FAMILIES[:families]
    if not 1 <= families <= len(fragment.TRAIN_FAMILIES) or not 1 <= queries <= 256 or not 1 <= rounds <= 16:
        raise ValueError("invalid family, query or curriculum round count")
    seen, cohorts = set(), {"warmup": [], "support": [], "within": [], "ood": []}
    for cohort, sources, count in (
        ("warmup", chosen, 4), ("support", chosen, 16 * rounds),
        ("within", chosen, queries), ("ood", fragment.HELDOUT_FAMILIES, queries),
    ):
        split = "train" if cohort in ("support", "warmup") else cohort
        for family in sources:
            for index in range(count):
                for attempt in range(4096):
                    token = f"ppt-g04/{seed}/{cohort}/{family}/{index}/{attempt}"
                    draw = hashlib.sha256(token.encode()).digest()
                    problem_seed, depth = int.from_bytes(draw[:8], "big"), 2 + draw[8] % 5
                    tid = "ppt_g04_" + draw.hex()[:20]
                    thm = fragment.generate(problem_seed, depth, family, arity3=False, tid=tid)
                    lineage = fragment.canonical_hash(thm)
                    if lineage not in seen:
                        break
                else:
                    raise ValueError("could not allocate independent generated theorem roots")
                seen.add(lineage)
                statement = fragment.statement(thm)
                root = digest([split, lineage, statement])
                cohorts[cohort].append(asdict(LiveCase(root, lineage, split, family,
                                                      problem_seed, depth, tid, statement)))
    record = {"schema": SCHEMA, "seed": seed, "families": list(chosen), "rounds": rounds,
              "cohorts": cohorts, "generator": sha256_file(Path(fragment.__file__)),
              "registration": {"unit": "generated theorem lineage; repeated measures paired by root",
                               "primary_contrast": "SGD at 16 support roots versus retrieval and frozen",
                               "minimum_meaningful_gain": 0.05, "familywise_alpha": 0.05,
                               "query_doses": list(DOSES), "purpose": "pilot_only",
                               "archive_input": None, "external_overlap": "pending"}}
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(record) + "\n")


def load(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["schema"] != SCHEMA or record["generator"] != sha256_file(Path(fragment.__file__)):
        raise ValueError("frozen curriculum or generator changed")
    families = set(record["families"])
    roots, lineages, cohorts = set(), set(), {}
    if not families or not families <= set(fragment.TRAIN_FAMILIES):
        raise ValueError("training curriculum has unregistered families")
    for name in ("warmup", "support", "within", "ood"):
        cases = cohorts[name] = [LiveCase(**item) for item in record["cohorts"][name]]
        for case in cases:
            case.theorem()
            sources = set(fragment.HELDOUT_FAMILIES) if name == "ood" else families
            split = "train" if name in ("warmup", "support") else name
            if (case.split != split or case.family not in sources or
                    case.root in roots or case.lineage in lineages):
                raise ValueError("overlapping or invalid curriculum split lineage")
            roots.add(case.root)
            lineages.add(case.lineage)
    return record, cohorts


def schedule(record, cohorts, *, support, rounds, persistent, meta_steps, backbone_every,
             arms, depths, final_only):
    tasks = []
    if meta_steps or backbone_every:
        for case in cohorts["warmup"]:
            tasks.append({"kind": "warmup", "root": case.root})
        for step in range(meta_steps):
            family = record["families"][step % len(record["families"])]
            warm = [case.root for case in cohorts["warmup"] if case.family == family]
            index = (step // len(record["families"])) % len(warm)
            tasks.append({"kind": "fit", "support": warm[index], "query": warm[(index + 1) % len(warm)]})

    visited = []
    fit_roots = {}
    for turn in range(rounds):
        for family in record["families"]:
            if family not in visited:
                visited.append(family)
            if not persistent or not tasks or (turn == 0 and family == record["families"][0]):
                tasks.append({"kind": "reset", "family": family})
            source = [case for case in cohorts["support"] if case.family == family][16 * turn:]
            if len(source) < support:
                raise ValueError("frozen curriculum has insufficient unique support")
            for dose in range(support + 1):
                if dose:
                    root = source[dose - 1].root
                    tasks.append({"kind": "support", "root": root, "family": family, "dose": dose})
                    if backbone_every and dose % backbone_every == 0:
                        warm = [case.root for case in cohorts["warmup"] if case.family == family]
                        tasks.append({"kind": "fit", "support": fit_roots.get(family, warm[0]), "query": root})
                        fit_roots[family] = root
                if dose not in DOSES or (final_only and dose not in (0, support)):
                    continue
                targets = [c for c in cohorts["within"] if c.family == family]
                if persistent and dose == support:
                    targets = [c for c in cohorts["within"] if c.family in visited]
                if family == record["families"][-1] and dose == support:
                    targets += cohorts["ood"]
                for case in targets:
                    for arm in arms:
                        for depth in ((1,) if arm in ("auto8", "horn", "static") else depths):
                            tasks.append({"kind": "query", "root": case.root, "family": family,
                                          "round": turn, "dose": dose, "arm": arm, "depth": depth})
    return tasks
