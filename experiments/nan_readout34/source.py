"""Build the fixed source roster and audit it without model or tokenizer access."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from . import oracle, parser, worlds


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_bytes())
BACKENDS = ("nf4", "bf16")
WORLDS = ("A", "B")


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(payload.encode("ascii")).hexdigest()


def build():
    rows = []
    for split in ("CAL", "DEV"):
        for query in CONFIG["query_types"]:
            for index in range(CONFIG[f"{split}_roots_per_query_type"]):
                for length in CONFIG["event_counts"]:
                    source = worlds.draw(CONFIG, split, query, index, length)
                    for world in WORLDS:
                        label = worlds.gold(source["worlds"][world], length)
                        arms = [(readout, arm) for readout in CONFIG["readouts"] for arm in CONFIG["arms"]]
                        arms.append(("final", "erased"))
                        for readout, arm in arms:
                            rows.append({
                                "id": f"{source['root']}:{length}:{world}:{readout}:{arm}",
                                "root": source["root"],
                                "split": split,
                                "query_type": query,
                                "event_count": length,
                                "world": world,
                                "arm": arm,
                                "schema": {
                                    "readout": readout,
                                    "checkpoints": label["checkpoints"],
                                    "nodes": list(worlds.NODES),
                                },
                                "prompt": worlds.render(source, world, readout, arm),
                                "label": label,
                            })
    return rows


def public(row):
    return {key: value for key, value in row.items() if key != "label"}


def audit(rows, check_budget):
    assert tuple(CONFIG["backends"]) == BACKENDS
    assert CONFIG["additional_control"] == "erased_final"
    assert CONFIG["raw_reply_character_cap"] == 512
    assert len(rows) == len({row["id"] for row in rows}) == CONFIG["rows_per_backend"] == 432
    roots = {row["root"]: row["split"] for row in rows}
    assert len(roots) == CONFIG["roots"] == 12
    assert Counter(roots.values()) == {"CAL": 3, "DEV": 9}
    contexts = {}
    pairs = {}
    for row in rows:
        key = (row["root"], row["event_count"], row["world"])
        contexts.setdefault(key, []).append(row)
        pairs.setdefault(key[:2], set()).add(key[2])
    assert len(contexts) == CONFIG["contexts"] == 48
    assert len(pairs) == 24 and all(values == set(WORLDS) for values in pairs.values())

    parse_checks = 0
    contrasts = Counter()
    event_lines = {}
    for key, group in contexts.items():
        check_budget()
        assert len(group) == 9
        indexed = {(row["schema"]["readout"], row["arm"]): row for row in group}
        assert len(indexed) == 9 and all(row["label"] == group[0]["label"] for row in group)
        example = indexed["final", "compute"]["prompt"]
        event_lines[key] = example.split("\nQuery:", 1)[0].split("\nEvents:\n", 1)[1].splitlines()
        for readout in CONFIG["readouts"]:
            compute = indexed[readout, "compute"]
            supplied = indexed[readout, "copy"]
            text = worlds.canonical(compute["label"], readout)
            separator = "Copy control: the complete verified answer is supplied below. Copy it exactly; do not recompute:\n"
            assert separator not in compute["prompt"]
            assert supplied["prompt"] == compute["prompt"] + separator + text + "\n"
            assert oracle.replay_text(compute["prompt"]) == compute["label"]
            assert oracle.replay_text(supplied["prompt"]) == supplied["label"]
            schema = compute["schema"]
            assert parser.parse(text, schema) == oracle.packet(text, schema)
            assert oracle.grade(text, schema, compute["label"])["joint_exact"] == 1
            invalid_room = text.replace("R", "X", 1)
            invalid = ("", text[:-1], text + "\ncomment", "prose " + text, text + "é",
                       " " * 512 + text, invalid_room, text + ";" + text)
            for reply in invalid:
                assert parser.parse(reply, schema) is None and oracle.packet(reply, schema) is None
            for reply in (text, " \n" + text + "\t"):
                assert parser.parse(reply, schema) == oracle.packet(reply, schema) is not None
            parse_checks += len(invalid) + 2
            contrasts[(compute["split"], compute["query_type"], compute["event_count"], readout)] += 1

        erased = indexed["final", "erased"]["prompt"]
        assert "Initial parents: [not provided]\nEvents: [not provided]\n" in erased
        assert not any(line.startswith("T") and ": actor=" in line for line in erased.splitlines())

    for root, length in pairs:
        a = contexts[root, length, "A"][0]
        b = contexts[root, length, "B"][0]
        events_a, events_b = event_lines[root, length, "A"], event_lines[root, length, "B"]
        assert len(events_a) == len(events_b) == length
        changes = [(left, right) for left, right in zip(events_a, events_b) if left != right]
        assert len(changes) == 1
        left, right = changes[0]
        left_before, left_after = left.split("=", 2)[-1].split(";", 1)
        right_before, right_after = right.split("=", 2)[-1].split(";", 1)
        assert left.rsplit("=", 2)[0] == right.rsplit("=", 2)[0]
        assert left.split(");", 1)[0].split(")=", 1)[0] == right.split(");", 1)[0].split(")=", 1)[0]
        assert left_before != right_before and left_after == right_after
        assert a["label"]["final_room"] != b["label"]["final_room"]
        assert contexts[root, length, "A"][-1]["prompt"] == contexts[root, length, "B"][-1]["prompt"]

    assert set(contrasts.values()) == {2, 6}
    return {
        "roots_by_split": dict(Counter(roots.values())),
        "contexts": len(contexts),
        "one_event_twin_pairs": len(pairs),
        "precision_matched_slots": len(rows),
        "rows_per_backend": dict.fromkeys(BACKENDS, len(rows)),
        "total_registered_slots": len(rows) * len(BACKENDS),
        "rows_by_split_and_backend": dict(Counter(row["split"] for row in rows)),
        "rows_by_arm_per_backend": dict(Counter(row["arm"] for row in rows)),
        "independent_rendered_replays": 2 * len(contexts) * len(CONFIG["readouts"]),
        "parser_agreement_checks": parse_checks,
        "public_roster_sha256": digest([public(row) for row in rows]),
        "label_roster_sha256": digest([{"id": row["id"], "label": row["label"]} for row in rows]),
    }
