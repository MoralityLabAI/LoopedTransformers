"""Independent rendered-graph and exact symbol-equivariance checks."""
from collections import defaultdict
import re

from .source import ARMS, COPY, ENCODINGS, HOPS, NODES, POOLS, SPLITS, WORLDS, balance

BODY = re.compile(r"Edges: (.+)\nStart: ([A-P])\nHops: ([0-9]+)\nAnswer:(?: ([A-P]))?")
EDGE = re.compile(r"([A-P])->([A-P])")


def parse(body):
    match = BODY.fullmatch(body)
    if not match:
        raise ValueError("Malformed pointer prompt")
    nodes, start, hops, answer = match.groups()
    if nodes == "[not provided]":
        return None, start, int(hops), answer
    pairs = [EDGE.fullmatch(edge) for edge in nodes.split(", ")]
    if not all(pairs):
        raise ValueError("Malformed pointer table")
    edges = dict((pair[1], pair[2]) for pair in pairs)
    if len(pairs) != len(edges) or set(edges) != set(edges.values()) or set(edges) != set(NODES):
        raise ValueError("Pointer map is not a permutation")
    return edges, start, int(hops), answer


def solve(parsed):
    graph, start, hops, _ = parsed
    if graph is None:
        raise ValueError("An erased graph cannot be solved")
    return _walk(graph, start, hops)[-1]


def _walk(graph, start, hops):
    path = [start]
    for _ in range(hops):
        path.append(graph[path[-1]])
    return path


def relabel_match(left, right, names):
    a, start, hop, answer = left
    b, other, other_hop, other_answer = right
    return (
        other == names[start] and hop == other_hop and other_answer == (names[answer] if answer else None)
        and (list(b.items()) if b else None) == ([(names[k], names[v]) for k, v in a.items()] if a else None)
    )


def ordered_signature(parsed):
    edges, start, _, _ = parsed
    position = {k: i for i, k in enumerate(edges)}
    return (position[start], tuple(position[v] for v in edges.values()))


def check(public, gold, earlier, guard):
    expected = sum(SPLITS.values()) * len(ENCODINGS) * len(HOPS) * len(WORLDS) * len(ARMS)
    assert len(public) == len(gold) == len({p["id"] for p in public}) == expected
    roster, seen, seen_topologies = defaultdict(dict), set(), set()
    verified_main = 0
    for row, answer in zip(public, gold, strict=True):
        guard()
        assert {k: v for k, v in row.items() if k != "prompt"} == {k: v for k, v in answer.items() if k != "answer"}
        root, split, enc, hops, world, arm = (row[k] for k in ("root", "split", "encoding", "hops", "world", "arm"))
        cells = roster[split, root]
        key = (enc, hops, world, arm)
        assert key not in cells and answer["answer"] in POOLS[split]
        if arm == "copy":
            assert row["prompt"] == COPY + answer["answer"]
            cells[key] = (answer["answer"], ())
            continue
        _, *parts = row["prompt"].split("\n\n")
        assert len(parts) == 3
        examples = tuple(parse(p) for p in parts[:-1])
        for p in examples:
            assert p[2] == hops and p[3] in POOLS[split] and solve(p) == p[3]
        target = parse(parts[-1])
        assert target[2] == hops and target[3] is None
        if arm == "focused":
            assert row["prompt"] not in earlier and row["prompt"] not in seen
            seen.add(row["prompt"])
            assert solve(target) == answer["answer"]
            assert len(set(_walk(*target[:3]))) == hops + 1
            verified_main += 1
        else:
            assert target[0] is None
        cells[key] = (answer["answer"], (*examples, target))

    for (split, _), rows in roster.items():
        assert len(rows) == len(ENCODINGS)*len(HOPS)*len(WORLDS)*len(ARMS)
        orig = rows["original", 1, "A", "focused"][1][-1]
        rel = rows["relabel", 1, "A", "focused"][1][-1]
        names = dict(zip(orig[0], rel[0], strict=True))
        assert set(names) == set(names.values()) == set(NODES) and all(k != v for k,v in names.items())
        assert set(names[x] for x in POOLS[split]) == set(POOLS[split])
        assert ordered_signature(orig) not in seen_topologies
        seen_topologies.add(ordered_signature(orig))
        for h in HOPS:
            for w in WORLDS:
                for arm in ARMS:
                    g, x = rows["original", h, w, arm]
                    other_gold, y = rows["relabel", h, w, arm]
                    assert other_gold == names[g] and other_gold != g
                    assert all(relabel_match(a,b,names) for a,b in zip(x,y,strict=True))
            for enc in ENCODINGS:
                a, b = (rows[enc,h,w,"focused"] for w in WORLDS)
                assert a[0] != b[0] and a[1][:-1] == b[1][:-1]
                p, q = a[1][-1], b[1][-1]
                assert p[1:] == q[1:] and list(p[0]) == list(q[0])
                changed = {k for k in p[0] if p[0][k] != q[0][k]}
                assert len(changed) == 2 and p[1] in changed
                other = next(k for k in changed if k != p[1])
                assert p[0][p[1]] == q[0][other] and p[0][other] == q[0][p[1]]
                path_a, path_b = (_walk(*parsed[:3]) for parsed in (p,q))
                assert set(path_a[1:]).isdisjoint(path_b[1:])
                erased_a, erased_b = (rows[enc,h,w,"focused_erased"] for w in WORLDS)
                assert erased_a[1] == erased_b[1] and [erased_a[0],erased_b[0]] == [a[0],b[0]]
                assert [rows[enc,h,w,"copy"][0] for w in WORLDS] == [a[0],b[0]]
        answers = {enc: {rows[enc,h,w,"focused"][0] for h in HOPS for w in WORLDS} for enc in ENCODINGS}
        assert all(len(v) == 4 for v in answers.values()) and answers["original"].isdisjoint(answers["relabel"])
        # Within each encoding and world, short and long queries share the same graph bytes and start.
        for enc in ENCODINGS:
            for w in WORLDS:
                x, y = (rows[enc,h,w,"focused"][1][-1] for h in HOPS)
                assert x[:2] == y[:2]

    roots = {split: sum(k[0] == split for k in roster) for split in SPLITS}
    assert roots == SPLITS
    return {"independent_roots_by_split": roots, "prompts": len(public), "independently_replayed": verified_main,
            "distinct_ordered_topologies": len(seen_topologies), "target_labels_by_split": balance(gold),
            "paired_world_batches_per_backend": len(public)//2, "verified_relabelings": len(roster)}
