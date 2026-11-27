"""Independent interpreter for the COG-02 finite Horn task contract.

The checker receives one proposed rule application and charges one call.
Policy inputs contain the world and the current ledger. Source metadata and
the retrospective goal-ancestry audit travel separately.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path


SPLITS = ("train", "calibration", "ordinary_test", "ood_test")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_tasks(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        tasks = [json.loads(line) for line in handle if line.strip()]
    roots, worlds = {}, {}
    for task in tasks:
        expected = hashlib.sha256(canonical({k: v for k, v in task.items()
                                            if k != "task_digest"}).encode()).hexdigest()
        if task.get("schema_version") != "cogentic.ledger.v1" or task.get("task_digest") != expected:
            raise ValueError("task schema or content digest mismatch")
        split, root = task["split"], task["root_id"]
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        if root in roots:
            raise ValueError("each root must have one task and one split")
        roots[root] = split
        world = task["world"]
        atoms = world["atoms"]
        if len(atoms) != len(set(atoms)) or not all(type(a) is int for a in atoms):
            raise ValueError("atoms must be distinct integers")
        atom_set = set(atoms)
        if not set(world["facts"]) <= atom_set or world["goal"] not in atom_set:
            raise ValueError("facts and goal must name declared atoms")
        for rule in world["rules"]:
            if not rule["premises"] or not set(rule["premises"]) <= atom_set:
                raise ValueError("rule premises must name declared atoms")
            if rule["conclusion"] not in atom_set:
                raise ValueError("rule conclusion must name a declared atom")
        fingerprint = hashlib.sha256(canonical(world).encode()).hexdigest()
        if fingerprint in worlds:
            raise ValueError("duplicate public world across corpus roots")
        worlds[fingerprint] = root
    if set(roots.values()) != set(SPLITS):
        raise ValueError("corpus must provide all four split roles")
    return tasks


def public_observation(task, known, history):
    return {"world": copy.deepcopy(task["world"]), "known": sorted(known),
            "history": [{key: copy.deepcopy(event[key])
                         for key in ("action", "accepted", "code", "new_fact")}
                        for event in history]}


def check_step(world, known, action):
    """Check one rule using only currently established facts."""
    known = set(known)
    receipt = {"accepted": False, "code": "invalid_action", "next_known": sorted(known),
               "new_fact": None, "solved": world["goal"] in known, "verifier_calls": 1}
    if type(action) is not int or not 0 <= action < len(world["rules"]):
        return receipt
    rule = world["rules"][action]
    if not set(rule["premises"]) <= known:
        receipt["code"] = "missing_premise"
        return receipt
    receipt["accepted"] = True
    conclusion = rule["conclusion"]
    receipt["code"] = "duplicate_lemma" if conclusion in known else "proved_lemma"
    if conclusion not in known:
        receipt["new_fact"] = conclusion
    known.add(conclusion)
    receipt["next_known"] = sorted(known)
    receipt["solved"] = world["goal"] in known
    return receipt


def goal_ancestors(world):
    """Structural teacher and audit ceiling, computed outside neural features."""
    needed = {world["goal"]}
    while True:
        before = len(needed)
        for rule in world["rules"]:
            if rule["conclusion"] in needed:
                needed.update(rule["premises"])
        if len(needed) == before:
            return needed


def useful_actions(world, known):
    needed = goal_ancestors(world)
    known = set(known)
    return [i for i, rule in enumerate(world["rules"])
            if rule["conclusion"] in needed and rule["conclusion"] not in known
            and set(rule["premises"]) <= known]


def available_actions(world, known, failed_at_state):
    """Shared public no-repeat rule; a new ledger fact clears rejected actions."""
    return [i for i, rule in enumerate(world["rules"])
            if rule["conclusion"] not in known and i not in failed_at_state]
