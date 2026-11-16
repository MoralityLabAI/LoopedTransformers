"""ROC-02 source-family-disjoint delayed-rule gym and exact symbolic oracle.

Registered: configs/roc02_delayed_rule_source_impl_v1.json, committed before
any source cases were generated. Training and selection may use train/dev only.
The sealed test split is published without labels until checkpoints and
observers are locked. The rules remain mechanically inspectable because
inputs explicitly contain the binding table; 'sealed' is a procedural gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

DOMAIN = 8
SOURCES = ("A", "B", "C")
FAMILIES = ((1, 1), (1, 3), (1, 5), (1, 7),
            (3, 1), (3, 5), (5, 3), (7, 5))
SPLITS = {
    "train": {"seed": 2026111621, "n": 1024, "families": (0, 1, 2, 3)},
    "calibration": {"seed": 2026111623, "n": 256, "families": (4, 5)},
    "sealed_test": {"seed": 2026111627, "n": 384, "families": (6, 7)},
}
EVENT_TYPES = ("bind", "distractor", "carry", "query")
FEATURE_DIM = 4 + 3 + 64 + 8


def rule_table(family: int) -> list[int]:
    if type(family) is not int or not 0 <= family < len(FAMILIES):
        raise ValueError("unregistered rule family")
    mult, shift = FAMILIES[family]
    table = [(mult * value + shift) % DOMAIN for value in range(DOMAIN)]
    if sorted(table) != list(range(DOMAIN)):
        raise AssertionError("registered rule is not a permutation")
    return table


def symbolic_execute(events: list[dict]) -> dict:
    """Independent source-binding interpreter with error labels."""
    bindings: dict[str, tuple[int, ...]] = {}
    observed_queries = []
    for event in events:
        op = event.get("op")
        if op in ("bind", "distractor"):
            source, table = event.get("src"), event.get("table")
            if source not in SOURCES or not isinstance(table, list) or len(table) != DOMAIN:
                raise ValueError("invalid source/table binding")
            if sorted(table) != list(range(DOMAIN)):
                raise ValueError("nonbijective or invalid table")
            bindings[source] = tuple(table)
        elif op == "carry":
            if set(event) != {"op"}:
                raise ValueError("carry carries unexpected data")
        elif op == "query":
            source, value = event.get("src"), event.get("value")
            if source not in SOURCES:
                return {"status": "no_rule", "label": None}
            if type(value) is not int or not 0 <= value < DOMAIN:
                return {"status": "out_of_domain", "label": None}
            if source not in bindings:
                return {"status": "no_rule", "label": None}
            observed_queries.append(bindings[source][value])
        else:
            raise ValueError("unknown event op")
    if len(observed_queries) != 1:
        raise ValueError("exactly one query required")
    return {"status": "ok", "label": observed_queries[0]}


def encode_event(event: dict) -> list[float]:
    """79 binary features: 4 event, 3 source, 64 table, 8 query."""
    op = event["op"]
    if op not in EVENT_TYPES:
        raise ValueError("unknown event")
    x = [0.0] * FEATURE_DIM
    x[EVENT_TYPES.index(op)] = 1.0
    source = event.get("src")
    if source is not None:
        if source not in SOURCES:
            raise ValueError("invalid source")
        x[4 + SOURCES.index(source)] = 1.0
    if "table" in event:
        table = event["table"]
        if not isinstance(table, list) or len(table) != DOMAIN:
            raise ValueError("invalid lookup table")
        for row, mapped in enumerate(table):
            if type(mapped) is not int or mapped not in range(DOMAIN):
                raise ValueError("invalid mapped label")
            x[7 + row * DOMAIN + mapped] = 1.0
    if "value" in event:
        value = event["value"]
        if type(value) is not int or value not in range(DOMAIN):
            raise ValueError("out-of-domain query cannot be neural encoded")
        x[71 + value] = 1.0
    return x


def canonical_hash(record: dict) -> str:
    payload = {key: record[key] for key in ("family", "delay", "events")}
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sample_case(rng: random.Random, families: tuple[int, ...]) -> dict:
    family = rng.choice(families)
    delay = rng.randint(2, 16)
    active = {"op": "bind", "src": "A", "table": rule_table(family)}
    distractor_count = rng.randint(0, min(4, delay))
    where = set(rng.sample(range(delay), distractor_count))
    events = [active]
    sources_used = []
    for t in range(delay):
        if t in where:
            source = rng.choice(("B", "C"))
            dist_family = rng.choice(families)
            events.append({"op": "distractor", "src": source,
                           "table": rule_table(dist_family)})
            sources_used.append(source)
        else:
            events.append({"op": "carry"})
    query = rng.randrange(DOMAIN)
    events.append({"op": "query", "src": "A", "value": query})
    record = {"family": family, "delay": delay,
              "distractor_composition": "".join(sources_used) or "none",
              "events": events}
    record["id"] = canonical_hash(record)
    answer = symbolic_execute(events)
    arithmetic = (FAMILIES[family][0] * query + FAMILIES[family][1]) % DOMAIN
    if answer != {"status": "ok", "label": arithmetic}:
        raise AssertionError("two exact oracle paths disagree")
    record["label"] = arithmetic
    return record


def build_split(name: str) -> list[dict]:
    if name not in SPLITS:
        raise ValueError("unknown split")
    cfg = SPLITS[name]
    rng = random.Random(cfg["seed"])
    out = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < cfg["n"]:
        attempts += 1
        if attempts > cfg["n"] * 200:
            raise RuntimeError("source uniqueness exhausted")
        rec = sample_case(rng, cfg["families"])
        if rec["id"] in seen:
            continue
        seen.add(rec["id"])
        out.append(rec)
    return out


def invalid_fixture_suite() -> list[dict]:
    out = []
    for i in range(12):
        source = "A" if i % 2 == 0 else "C"
        events = ([{"op": "carry"}] * (2 + i % 3)
                  + [{"op": "query", "src": source, "value": i % DOMAIN}])
        out.append({"kind": "no_rule", "events": events,
                    "expected": "no_rule"})
    for i in range(12):
        events = [{"op": "bind", "src": "A", "table": rule_table(i % 8)},
                  {"op": "carry"},
                  {"op": "query", "src": "A", "value": DOMAIN + i}]
        out.append({"kind": "out_of_domain", "events": events,
                    "expected": "out_of_domain"})
    return out


def write_dataset(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"study": "ROC-02-delayed-rule-source-implementation-v1",
                "original_registration": "MathCan ROC-01 Part4 and Amendment C",
                "generator": "roc02_delayed_rule_source.py",
                "rule_families": {k: list(v["families"]) for k, v in SPLITS.items()},
                "splits": {}, "invalid": {}}
    globally_seen: set[str] = set()
    for name in SPLITS:
        cases = build_split(name)
        ids = {x["id"] for x in cases}
        if len(cases) != len(ids) or globally_seen.intersection(ids):
            raise AssertionError("duplicate source event cases across splits")
        globally_seen.update(ids)
        for row in cases:
            if row["family"] not in SPLITS[name]["families"]:
                raise AssertionError("family leakage")
            if symbolic_execute(row["events"])["label"] != row["label"]:
                raise AssertionError("source label differs from exact oracle")
        file = destination / f"{name}_inputs.jsonl"
        # The sealed source has no 'label' field until the model/observer freeze.
        with file.open("w", encoding="utf-8") as f:
            for row in cases:
                visible = row if name != "sealed_test" else {k:v for k,v in row.items() if k!="label"}
                f.write(json.dumps(visible, sort_keys=True, separators=(",", ":")) + "\n")
        manifest["splits"][name] = {
            "n": len(cases), "source_seed": SPLITS[name]["seed"],
            "family_counts": dict(sorted(Counter(x["family"] for x in cases).items())),
            "file_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "file_name": file.name,
            "labels_in_public_file": name != "sealed_test",
            "oracle_accuracy": 1.0}
    invalid = invalid_fixture_suite()
    for case in invalid:
        if symbolic_execute(case["events"])["status"] != case["expected"]:
            raise AssertionError("invalid symbolic outcome")
    file = destination / "invalid_oracle_fixtures.jsonl"
    file.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in invalid),encoding="utf-8")
    manifest["invalid"] = {"n": len(invalid), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    (destination / "source_manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    return manifest


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(write_dataset(a.out),indent=2))


if __name__ == "__main__":
    main()
