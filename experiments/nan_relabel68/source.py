"""Draw paired lookup roots with target-label-disjoint exposures and graph relabelings."""
from collections import Counter
import random

from experiments.nan_comp14.sources import NODES, question
from experiments.nan_readout34.source import digest

SEED = "20261122-nan-relabel68-v1"
SPLITS = {"calibration": 8, "development": 16}
POOLS = {"calibration": NODES[:8], "development": NODES[8:]}
ENCODINGS = ("original", "relabel")
HOPS = (1, 3)
WORLDS = ("A", "B")
ARMS = ("focused", "copy", "focused_erased")
INTRO = "Follow the directed edges for exactly the requested number of hops. Reply only with the final node, one letter A through P."
COPY = "Copy the following verified answer exactly, with no explanation:\n"


def generator(root):
    return random.Random(int(digest([SEED, root])[:32], 16))


def advance(edges, start, hops):
    for _ in range(hops):
        start = edges[start]
    return start


def draw(root):
    rng = generator(root)
    nodes = list(range(len(NODES)))
    rng.shuffle(nodes)
    start, owner, a1, a2, a3, b1, b2, b3 = nodes[:8]
    fixed = {start: a1, a1: a2, a2: a3, owner: b1, b1: b2, b2: b3}
    free = sorted(set(nodes)-set(fixed.values()))
    rng.shuffle(free)
    graph_a = fixed | dict(zip(sorted(set(nodes)-set(fixed)), free, strict=True))
    graph_b = graph_a | {start: b1, owner: a1}
    order = rng.sample(nodes, len(nodes))
    demos = {}
    for hops in HOPS:
        prompts = []
        for endpoint in (a1, a3):
            perm = rng.sample(nodes, len(nodes))
            example = dict(zip(sorted(nodes), perm, strict=True))
            inverse = {v: k for k, v in example.items()}
            query = advance(inverse, endpoint, hops)
            prompts.append((example, rng.sample(nodes, len(nodes)), query, endpoint))
        demos[hops] = prompts
    return {"start": start, "order": order, "graphs": {"A": graph_a, "B": graph_b},
            "targets": (a1, b1, a3, b3), "demos": demos}


def labels(draw_data, root, split, index, encoding):
    pool = POOLS[split]
    other = "".join(c for c in NODES if c not in pool)
    base = dict(zip(draw_data["targets"], (pool[(index+i) % len(pool)] for i in range(4)), strict=True))
    left = sorted(set(range(len(NODES)))-set(base))
    available = sorted(set(NODES)-set(base.values()))
    rng = generator(root + ":surface")
    rng.shuffle(available)
    base.update(zip(left, available, strict=True))
    if encoding == "original":
        return base
    relabel = dict(zip(pool, pool[4:]+pool[:4], strict=True))
    relabel.update(zip(other, other[3:]+other[:3], strict=True))
    return {code: relabel[label] for code, label in base.items()}


def mapped_question(edges, order, start, hops, surface, erased=False):
    mapped = None if erased else {surface[k]: surface[v] for k, v in edges.items()}
    return question(mapped, [surface[k] for k in order], surface[start], hops)


def build():
    public, host = [], []
    for split, count in SPLITS.items():
        for index in range(count):
            root = f"relabel68:{split}:{index:03d}"
            tape = draw(root)
            for encoding in ENCODINGS:
                surface = labels(tape, root, split, index, encoding)
                for hops in HOPS:
                    examples = [mapped_question(edges, order, start, hops, surface) + " " + surface[answer]
                                for edges, order, start, answer in tape["demos"][hops]]
                    for world in WORLDS:
                        edges = tape["graphs"][world]
                        final = surface[advance(edges, tape["start"], hops)]
                        for arm in ARMS:
                            meta = {"id": f"{root}:{encoding}:{hops}:{world}:{arm}", "root": root, "split": split,
                                    "encoding": encoding, "hops": hops, "world": world, "arm": arm}
                            if arm == "copy":
                                prompt = COPY + final
                            else:
                                target = mapped_question(edges, tape["order"], tape["start"], hops, surface,
                                                         erased=arm == "focused_erased")
                                prompt = "\n\n".join((INTRO, *examples, target))
                            public.append(meta | {"prompt": prompt})
                            host.append(meta | {"answer": final})
    return public, host


def balance(gold):
    counts = {}
    for split, count in SPLITS.items():
        rows = [row for row in gold if row["split"] == split and row["arm"] == "focused"]
        for enc in ENCODINGS:
            for hops in HOPS:
                for world in WORLDS:
                    counter = Counter(r["answer"] for r in rows
                                      if (r["encoding"], r["hops"], r["world"]) == (enc, hops, world))
                    assert counter == dict.fromkeys(POOLS[split], count//len(POOLS[split]))
        counts[split] = sorted({row["answer"] for row in rows})
    assert set(counts["calibration"]).isdisjoint(counts["development"])
    return counts
