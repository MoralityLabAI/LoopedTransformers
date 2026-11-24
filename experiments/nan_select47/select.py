"""Profile ranked paths, choose equal per-phase sets and deduplicate intervention masks."""
from dataclasses import dataclass
import math
from statistics import fmean

from experiments.nan_readpath37.run import ranked_entries


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    weight: float
    rank: float
    normalized_strength: float


def ranked_stage(links):
    if any(not math.isfinite(weight) for _, _, weight in links):
        raise ValueError("Incomplete or nonfinite measured edge")
    weights = [w for _, _, w in links]
    scale = sum(map(abs, weights))
    return [Edge(source, target, weight, entry, abs(weight) / scale if scale else 0.0)
            for (source, target, weight), entry in zip(links, ranked_entries(weights))]


def layered(source, middle, sink):
    n, m = len(source), len(sink)
    if not n or not m or len(middle) != n or any(len(row) != m for row in middle):
        raise ValueError("Measured edge arrays have incompatible physical occurrences")
    end = 1 + n + m
    return end + 1, (
        ranked_stage([(0, i + 1, x) for i, x in enumerate(source)])
        + ranked_stage([(i + 1, n + j + 1, x)
                        for i, row in enumerate(middle) for j, x in enumerate(row)])
        + ranked_stage([(n + j + 1, end, x) for j, x in enumerate(sink)])
    )


def node_entries(nodes, edges):
    """Exact bottleneck forward/reverse sweep over topologically sorted edge groups."""
    forward, backward = [math.inf] * nodes, [math.inf] * nodes
    forward[0], backward[-1] = 0., 0.
    for edge in edges:
        forward[edge.target] = min(forward[edge.target], max(forward[edge.source], edge.rank))
    for edge in reversed(edges):
        backward[edge.source] = min(backward[edge.source], max(backward[edge.target], edge.rank))
    return [max(a, b) for a, b in zip(forward, backward)]


def unit_rank(values):
    normalized = ranked_entries(values)
    live = sum(math.isfinite(x) for x in normalized)
    return [0.0 if not math.isfinite(x) else 1.0 + 1.0 / live - x for x in normalized]


def score_graph(graph, cutoff):
    nodes, edges = layered(graph["source"], graph["middle"], graph["sink"])
    bottleneck = node_entries(nodes, edges)
    degree_in, degree_out = [0] * nodes, [0] * nodes
    strength_in, strength_out, attribution = ([0.0] * nodes for _ in range(3))
    for e in edges:
        attribution[e.source] += e.normalized_strength
        attribution[e.target] += e.normalized_strength
        if e.rank <= cutoff:
            degree_out[e.source] += 1
            degree_in[e.target] += 1
            strength_out[e.source] += e.normalized_strength
            strength_in[e.target] += e.normalized_strength
    n = len(graph["source"])
    activity = [*unit_rank(graph["activity1"]), *unit_rank(graph["activity2"])]
    effect = [*unit_rank(graph["isolated1"]), *unit_rank(graph["isolated2"])]
    full_rank = [*unit_rank(attribution[1:n + 1]), *unit_rank(attribution[n + 1:-1])]
    if len(activity) != nodes - 2 or len(effect) != nodes - 2 or n != len(graph["activity1"]):
        raise ValueError("Missing phase-conditioned measurements for a competing selector")
    answer = []
    for i in range(1, nodes - 1):
        a, b = degree_in[i], degree_out[i]
        u, v = strength_in[i], strength_out[i]
        answer.append({
            "sieve": (float(bottleneck[i] <= cutoff), max(0.0, 1.0 - bottleneck[i]), -bottleneck[i]
                      if math.isfinite(bottleneck[i]) else -2.0),
            "local_degree": (float(a > 0 and b > 0), min(a, b), a + b),
            "local_weighted_degree": (math.sqrt(u * v), u + v),
            "full_attribution": (full_rank[i - 1],),
            "isolated_margin_response": (effect[i - 1],),
            "fixed_bank_activity": (activity[i - 1],),
        })
    return answer


def across_worlds(a, b):
    return [{name: tuple((x + y) / 2 for x, y in zip(left[name], right[name]))
             for name in left} for left, right in zip(a, b, strict=True)]


def mask_sets(paired, cfg):
    """Aggregate root means; select on fixed CAL data with deterministic tie handling."""
    roots = [across_worlds(score_graph(a, cfg["calibration_cutoff"]), score_graph(b, cfg["calibration_cutoff"]))
             for a, b in paired]
    sending, receiving = cfg["source_components"], cfg["receiver_components"]
    if len(roots) < cfg["minimum_complete_CAL_roots"]:
        return {"status": "insufficient_complete_CAL_roots", "complete_CAL_roots": len(roots)}
    num = len(sending) + len(receiving)
    if any(len(group) != num for group in roots) or len(set(sending)) != len(sending) or len(set(receiving)) != len(receiving):
        raise ValueError("Static mask roster differs from the published factor bank")
    quotas = [cfg["source_quota"], cfg["receiver_quota"]]
    if not (0 < quotas[0] <= len(sending) and 0 < quotas[1] <= len(receiving)):
        raise ValueError("Fixed selector quota is out of range")
    masks = {}
    for method in roots[0][0]:
        pooled = []
        for slot in range(num):
            pooled.append(tuple(fmean(values) for values in zip(*(group[slot][method] for group in roots))))

        def top(index, start, quota):
            order = sorted(range(len(index)), key=lambda i: tuple(-x for x in pooled[start + i]) + (index[i],))
            return sorted(index[i] for i in order[:quota])
        masks[method] = {"source": top(sending, 0, quotas[0]), "receiver_observer": top(receiving, len(sending), quotas[1])}

    aliases = {}
    for name, mask in masks.items():
        key = tuple(mask["source"])
        aliases.setdefault(key, []).append(name)
    reference = masks["sieve"]
    overlap = {}
    for name, mask in masks.items():
        record = {}
        for scope in reference:
            a, b = set(reference[scope]), set(mask[scope])
            record[scope] = {"same": a == b, "replacements": len(a - b), "Jaccard": len(a & b) / len(a | b)}
        overlap[name] = record
    return {"status": "frozen", "complete_CAL_roots": len(roots), "masks": masks, "overlap_vs_sieve": overlap,
            "physical_source_intervention_groups": [{"source_mask": list(key), "selectors": names}
                                                     for key, names in sorted(aliases.items())],
            "sieve_vs_degree_requires_distinct_source_interventions": not overlap["local_degree"]["source"]["same"]}
