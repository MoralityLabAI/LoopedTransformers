"""NEC-01: context-block neuron co-attribution and uncertainty estimates.

Each item supplied to graph discovery represents one independently sampled
PEF-07 eight-bit context, aggregated across its full 16-state focal quartet.
Nodes are (recurrent visit, native MLP neuron), not distinct parameters.
This is an exploratory Louvain surrogate for Goodfire's Leiden method.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import random
from statistics import mean


def node(visit: int, neuron: int) -> str:
    if visit < 0 or neuron < 0:
        raise ValueError("negative visit/neuron")
    return f"{visit}:{neuron}"


def decode(key: str) -> tuple[int, int]:
    parts = key.split(":")
    if len(parts) != 2 or any(not x.isdecimal() for x in parts):
        raise ValueError("bad visit:neuron key")
    return int(parts[0]), int(parts[1])


def build_graph(context_rows: list[list[str]], *, hub_rate: float = 0.9,
                min_pair_count: int = 3, min_phi: float = 0.18) -> dict:
    """Create frequency-adjusted positive co-attribution edges.

    One row per source context, rather than per counterfactual history.
    Pearson's phi is the Bernoulli co-occurrence correlation; perfectly
    ubiquitous nodes are recorded as hubs and excluded from the graph.
    """
    if not context_rows or not 0 < hub_rate <= 1 or min_pair_count < 1:
        raise ValueError("invalid graph settings")
    n = len(context_rows)
    counts = Counter()
    pairs = Counter()
    for row in context_rows:
        present = sorted(set(row), key=decode)
        for key in present:
            decode(key)
        counts.update(present)
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                pairs[(a, b)] += 1
    hubs = sorted((k for k, f in counts.items() if f/n >= hub_rate),
                  key=decode)
    hub_set = set(hubs)
    edges = []
    for (a, b), joint in pairs.items():
        if a in hub_set or b in hub_set or joint < min_pair_count:
            continue
        fa, fb = counts[a], counts[b]
        denominator = fa * fb * (n-fa) * (n-fb)
        if denominator <= 0:
            continue
        phi = (n*joint - fa*fb)/math.sqrt(denominator)
        if phi >= min_phi:
            edges.append({"a": a, "b": b, "phi": round(phi, 9),
                          "joint_contexts": joint})
    edges.sort(key=lambda e: (decode(e["a"]), decode(e["b"])))
    return {"contexts": n, "frequency": dict(sorted(counts.items(),
              key=lambda kv: decode(kv[0]))),
            "hubs": hubs, "edges": edges,
            "method": "context-aggregated-Pearson-phi-then-seeded-Louvain",
            "parameters": {"hub_rate": hub_rate,
                           "min_pair_count": min_pair_count,
                           "min_phi": min_phi}}


def discover(graph: dict, *, max_size: int = 24, min_size: int = 2,
             max_candidates: int = 8, seed: int = 0) -> list[dict]:
    """Rank candidates using discovery-only graph information."""
    import networkx as nx
    G = nx.Graph()
    for e in graph["edges"]:
        G.add_edge(e["a"], e["b"], weight=e["phi"])
    if G.number_of_edges() == 0:
        return []
    partitions = nx.community.louvain_communities(G, weight="weight",
                                                  seed=seed, resolution=1.0)
    candidates = []
    for part in partitions:
        members = sorted(part, key=decode)
        visits = {decode(k)[0] for k in members}
        if not min_size <= len(members) <= max_size or len(visits) < 2:
            continue
        H = G.subgraph(members)
        total_weight = sum(e["weight"] for *_, e in H.edges(data=True))
        score = total_weight/max(1, H.number_of_edges())
        digest = hashlib.sha256((",".join(members)).encode()).hexdigest()[:12]
        candidates.append({"id": "nec-" + digest, "nodes": members,
                           "visits": sorted(visits), "graph_mean_phi": score,
                           "graph_edges": H.number_of_edges()})
    candidates.sort(key=lambda x: (-x["graph_mean_phi"],
                                  -x["graph_edges"], x["id"]))
    return candidates[:max_candidates]


def matched_sham(candidate: dict, graph: dict, *, mlp_width: int = 64) -> list[str]:
    """Preselect equally many same-visit neurons of closest train frequency.

    Uses discovery context frequencies exclusively. Intended as a
    frequency-matched control, not an activation/output-norm-matched control.
    """
    selected = candidate["nodes"]
    freq = graph["frequency"]
    by_visit = {}
    for key in selected:
        visit, _ = decode(key)
        by_visit.setdefault(visit, []).append(key)
    sham = []
    for visit, wanted in sorted(by_visit.items()):
        wanted_freq = mean(freq.get(k, 0) for k in wanted)
        candidates = [node(visit, j) for j in range(mlp_width)
                      if node(visit, j) not in selected]
        candidates.sort(key=lambda k: (abs(freq.get(k, 0)-wanted_freq), decode(k)))
        if len(candidates) < len(wanted):
            raise ValueError("not enough same-visit sham neurons")
        sham.extend(candidates[:len(wanted)])
    return sham


def interval(values: list[float], *, iterations: int = 2000,
             seed: int = 1907) -> dict:
    """Percentile interval over independently held-out context group means."""
    if not values or iterations < 100 or not all(math.isfinite(v) for v in values):
        raise ValueError("bad clustered bootstrap input")
    r = random.Random(seed)
    n = len(values)
    samples = sorted(sum(values[r.randrange(n)] for _ in range(n))/n
                     for _ in range(iterations))
    return {"contexts": n, "mean": mean(values),
            "ci95": [samples[int(0.025*iterations)],
                     samples[min(iterations-1, int(0.975*iterations))]],
            "method": "paired-context-percentile-bootstrap",
            "bootstrap_iterations": iterations}


def summary(context_records: list[dict], *, seed: int = 1907) -> dict:
    """Context means are the independent observations, and stay paired."""
    keys = ("kl", "correctness_drop", "top_probability_drop")
    out = {}
    for key in keys:
        values = [float(x[key]) for x in context_records]
        out[key] = interval(values, seed=seed)
    return out
