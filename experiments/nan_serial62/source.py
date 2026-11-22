"""Outcome-blind matched lookup tasks for the Nanbeige depth gym."""
from collections import Counter
import random

from experiments.nan_readout34.source import digest

SEED = "20261122-nan-serial62-v1"
SPLITS = {"CAL": 2, "DEV": 6}
HOPS = (1, 3, 6)
ROWS = (24, 72)
ARMS = (("direct", "compute"), ("trace", "compute"),
        ("direct", "copy"), ("trace", "copy"), ("direct", "erased"))
DEPTHS = (1, 2)
BACKENDS = ("nf4", "bf16")
RULE = "Look up the current code on the left, then replace it with the code on the right."
MISSING = "[table withheld]"
COPY = "The verified reply is supplied. Copy it exactly:\n"


def root_graph(root):
    rng = random.Random(int(digest([SEED, root]), 16))
    codes = [f"{i:02d}" for i in range(96)]
    rng.shuffle(codes)
    start = codes[0]
    lanes = {"A": codes[1:7], "B": codes[7:13]}
    # Place the two equally long paths among irrelevant functional edges.
    # The longer context inserts rows while preserving the relative order of all short rows.
    keys = codes[:ROWS[-1]]
    exits = dict(zip(keys, rng.choices(keys, k=len(keys))))
    for lane in lanes.values():
        exits.update(zip(lane, lane[1:]))
    rng.shuffle(keys)
    return start, lanes, set(codes[:ROWS[0]]), keys, exits


def canonical(path, readout):
    final = f"FINAL={path[-1]};END"
    return final if readout == "direct" else "PATH=" + ">".join(path[1:]) + ";" + final


def render(edges, start, hops, readout):
    table = "\n".join(f"{key} -> {value}" for key, value in edges) if edges else MISSING
    form = "FINAL=dd;END" if readout == "direct" else "PATH=dd>...>dd;FINAL=dd;END"
    detail = "" if readout == "direct" else " PATH lists the code reached after each lookup, excluding the start."
    return (
        f"{RULE}\nTable:\n{table}\n"
        f"Start at {start}. Lookups: {hops}.\n"
        f"Give exactly one line in the form {form} (dd is a two-digit code).{detail} No prose.\n"
    )


def build():
    result = []
    for split, count in SPLITS.items():
        for index in range(count):
            root = f"serial62:{split}:{index:02d}"
            start, lanes, short_keys, keys, edges = root_graph(root)
            for size in ROWS:
                for hops in HOPS:
                    for world, lane in lanes.items():
                        path = [start, *lane[:hops]]
                        table = [(key, lane[0] if key == start else edges[key])
                                 for key in keys if size == ROWS[-1] or key in short_keys]
                        for readout, arm in ARMS:
                            prompt = render([] if arm == "erased" else table, start, hops, readout)
                            if arm == "copy":
                                prompt += COPY + canonical(path, readout) + "\n"
                            result.append({
                                "id": f"{root}:{size}:{hops}:{world}:{readout}:{arm}",
                                "root": root, "split": split, "table_rows": size, "hops": hops,
                                "world": world, "readout": readout, "arm": arm, "prompt": prompt,
                                "path": path,
                            })
    return result


def audit(rows, guard):
    from .oracle import grade, replay, read_table

    expected_count = sum(SPLITS.values()) * len(ROWS) * len(HOPS) * 2 * len(ARMS)
    assert len(rows) == len({row["id"] for row in rows}) == expected_count
    roots = {row["root"]: row["split"] for row in rows}
    assert Counter(roots.values()) == SPLITS
    roster = {(r["root"], r["table_rows"], r["hops"], r["readout"], r["arm"], r["world"]): r
              for r in rows}
    tables = {}
    for key, row in roster.items():
        guard()
        root, size, hops, readout, arm, world = key
        prompt = row["prompt"]
        gold = canonical(row["path"], readout)
        grade_gold = grade(row, gold)
        assert grade_gold["joint_exact"] and len(gold) < 64
        assert not grade(row, gold + "\nEXTRA")["format_valid"]
        if arm == "erased":
            assert replay(prompt) is None
            assert prompt == roster[(*key[:-1], "B" if world == "A" else "A")]["prompt"]
            continue
        edges, start, steps = read_table(prompt)
        assert len(edges) == size and steps == hops and len(set(row["path"])) == hops + 1
        assert replay(prompt) == row["path"]
        if arm == "copy":
            main = roster[(root, size, hops, readout, "compute", world)]
            assert prompt == main["prompt"] + COPY + gold + "\n"
            continue
        assert COPY not in prompt
        tables.setdefault((root, size, world), edges)
        assert tables[root, size, world] == edges
        if world == "A":
            twin = roster[(*key[:-1], "B")]
            other, other_start, other_steps = read_table(twin["prompt"])
            assert (other_start, other_steps) == (start, steps)
            assert list(edges) == list(other)
            assert [entry for entry in edges if edges[entry] != other[entry]] == [start]
            assert set(row["path"][1:]).isdisjoint(twin["path"][1:])
    for root in roots:
        for world in ("A", "B"):
            short = tables[root, ROWS[0], world]
            long = tables[root, ROWS[1], world]
            assert list(short.items()) == [(key, value) for key, value in long.items() if key in short]

    return {
        "roots_by_split": dict(Counter(roots.values())),
        "prompts": len(rows),
        "independently_replayed_prompts": sum(row["arm"] != "erased" for row in rows),
        "erased_twin_pairs": sum(row["arm"] == "erased" and row["world"] == "A" for row in rows),
        "slots_per_backend": len(rows) * len(DEPTHS),
        "roster_sha256": digest([{k: v for k, v in row.items() if k != "path"} for row in rows]),
        "gold_sha256": digest([[row["id"], row["path"]] for row in rows]),
    }
