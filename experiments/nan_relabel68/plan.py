"""Compile paired prefills and grade first-token transfer across label encodings."""
from collections import defaultdict
from itertools import product

from experiments.nan_comp14.parser import label
from experiments.nan_readout34.runtime_audit import paired

from .audit import parse
from .source import ENCODINGS, HOPS, ARMS, SPLITS, WORLDS

DEPTHS = (1, 2)


def pair_roster(rows, bindings):
    cases = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ("split", "root", "hops", "encoding", "arm"))
        worlds = cases[key]
        if row["world"] in worlds:
            raise ValueError("Repeated source world")
        worlds[row["world"]] = row["id"]
    counts = {split: len({k[1] for k in cases if k[0] == split}) for split in SPLITS}
    if counts != SPLITS or len(cases) != sum(SPLITS.values()) * len(HOPS) * len(ENCODINGS) * len(ARMS):
        raise ValueError("Fixed B2 source roster is incomplete")
    plan = []
    for i, ((split, root, hops, enc, arm), worlds) in enumerate(sorted(cases.items())):
        if set(worlds) != set(WORLDS):
            raise ValueError("A registered B2 job requires both worlds")
        slots = [worlds[world] for world in WORLDS]
        a, b = (bindings[slot] for slot in slots)
        if a["tokens"] != b["tokens"] or (arm == "focused_erased" and a != b):
            raise ValueError("B2 token lengths or erased prompts differ")
        plan.append({"ordinal": i, "split": split, "root": root, "hops": hops, "encoding": enc,
                     "arm": arm, "source_ids": slots, "length": a["tokens"], "observe_phases": list(DEPTHS)})
    if {slot for pair in plan for slot in pair["source_ids"]} != set(bindings):
        raise ValueError("The plan does not bind every tokenizer receipt row")
    return plan


def relabel_maps(public):
    by_root = defaultdict(dict)
    for r in public:
        if (r["hops"], r["world"], r["arm"]) == (1, "A", "focused"):
            by_root[r["root"]][r["encoding"]] = parse(r["prompt"].split("\n\n")[-1])[0]
    result = {}
    for root, d in by_root.items():
        mapping = dict(zip(d["original"], d["relabel"], strict=True))
        if len(mapping) != 16 or set(mapping) != set(mapping.values()):
            raise ValueError("Source relabeling is not a complete permutation")
        result[root] = mapping
    return result


def and_complete(values):
    return None if None in values else all(values)


def count(values):
    return {"registered": len(values), "correct": sum(value is True for value in values),
            "incorrect": sum(value is False for value in values), "unknown": values.count(None)}


def score(public, gold, replies):
    """Reconcile independently verified first-token text closures against the gold roster."""
    source = {row["id"]: row for row in public}
    maps = relabel_maps(public)
    if len(source) != len(public) or set(source) != {r["id"] for r in gold}:
        raise ValueError("Public and oracle rosters differ")
    if any(key not in source or depth not in DEPTHS or not isinstance(text, str)
           for (key, depth), text in replies.items()):
        raise ValueError("A reply is outside the registered source or physical depth")
    values = defaultdict(dict)
    for r in gold:
        src = source[r["id"]]
        identity = ("id", "split", "root", "hops", "encoding", "world", "arm")
        if any(r[k] != src[k] for k in identity):
            raise ValueError("Response closure binds a different gold row")
        cell = values[r["split"], r["hops"], r["arm"]]
        for depth in DEPTHS:
            text = replies.get((r["id"], depth))
            prediction = label(text)
            correct = None if text is None else prediction == r["answer"]
            cell[r["root"], r["encoding"], r["world"], depth] = (correct, prediction)
    result = []
    for (split, hops, arm), cell in sorted(values.items()):
        roots = sorted({root for root, _, _, _ in cell})

        def results(depth):
            world = [cell[r, enc, side, depth][0] for r,enc,side in product(roots, ENCODINGS, WORLDS)]
            twin = [and_complete([cell[r, enc, side, depth][0] for side in WORLDS])
                    for r,enc in product(roots, ENCODINGS)]
            orbit = [and_complete([cell[r,enc,side,depth][0] for enc,side in product(ENCODINGS,WORLDS)])
                     for r in roots]
            equivariant = []
            for r,side in product(roots,WORLDS):
                a, b = (cell[r,enc,side,depth] for enc in ENCODINGS)
                valid = None if a[0] is None or b[0] is None else (
                    a[1] is not None and b[1] is not None and maps[r][a[1]] == b[1])
                equivariant.append(valid)
            return {"world_correct": world, "both_worlds_same_encoding_correct": twin,
                    "all_worlds_both_encodings_correct": orbit, "semantic_prediction_equivariant": equivariant}

        one, two = (results(d) for d in DEPTHS)
        result.append({"split": split, "hops": hops, "arm": arm, "independent_roots": len(roots),
                       "per_phase": {"1": {k: count(v) for k,v in one.items()},
                                     "2": {k: count(v) for k,v in two.items()}},
                       "phase1_to_phase2": {k: paired(list(zip(one[k], two[k], strict=True))) for k in one}})
    return result
