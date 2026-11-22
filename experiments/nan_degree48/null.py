"""Count labelled completions under a fixed bipartite degree sequence."""
from collections import Counter
from functools import cache
from itertools import combinations

from experiments.nan_select47.select import layered, node_entries


def completion_count(inactive):
    """Columns with the same capacity are exchangeable, with labelled multiplicity."""
    @cache
    def count(index, sorted_remaining):
        if index == len(inactive):
            return int(not any(sorted_remaining))
        next_rows = len(inactive) - index - 1
        available = [j for j, r in enumerate(sorted_remaining) if r]
        cases = Counter()
        for touched in combinations(available, inactive[index]):
            left = list(sorted_remaining)
            for j in touched:
                left[j] -= 1
            if max(left, default=0) <= next_rows:
                cases[tuple(sorted(left))] += 1
        return sum(multiplicity * count(index + 1, caps) for caps, multiplicity in cases.items())

    return count


def exact_distribution(rows, cols, active_rows, sink_cols, *, check=lambda: None):
    """Return counts indexed by (source-side backbone size, sink-side backbone size)."""
    a, b = len(rows), len(cols)
    if (not (1 <= a <= 8 and 1 <= b <= 8) or any(type(x) is not int or x < 0 or x > b for x in rows)
            or any(type(x) is not int or x < 0 or x > a for x in cols) or sum(rows) != sum(cols)
            or not set(active_rows) <= set(range(a)) or not set(sink_cols) <= set(range(b))):
        raise ValueError("Invalid supported fixed-degree bipartite problem")
    active_rows, sink_cols = set(active_rows), set(sink_cols)
    active = sorted((r for i, r in enumerate(rows) if i in active_rows), reverse=True)
    inactive = sorted((r for i, r in enumerate(rows) if i not in active_rows), reverse=True)
    finish = completion_count(inactive)
    initial = tuple(sorted((int(j in sink_cols), r) for j, r in enumerate(cols)))
    states = {(initial, 0): 1}
    for index, degree in enumerate(active, 1):
        next_states = Counter()
        nleft = a - index
        for k, ((capacities, source_hits), multiplicity) in enumerate(states.items()):
            if k % 128 == 0:
                check()
            slots = [j for j, (_, left) in enumerate(capacities) if left]
            for chosen in combinations(slots, degree):
                nxt = list(capacities)
                hits_sink = False
                for j in chosen:
                    group, r = nxt[j]
                    hits_sink |= group != 0
                    nxt[j] = (2 if group else 0, r - 1)
                if any(r > nleft for _, r in nxt):
                    continue
                key = tuple(sorted(nxt)), source_hits + hits_sink
                next_states[key] += multiplicity
        states = next_states
    results = Counter()
    for k, ((capacities, source_hits), multiplicity) in enumerate(states.items()):
        if k % 128 == 0:
            check()
        receiver_hits = sum(g == 2 for g, _ in capacities)
        completions = finish(0, tuple(sorted(r for _, r in capacities)))
        if completions:
            results[source_hits, receiver_hits] += completions * multiplicity
    if not results:
        raise ValueError("The specified labelled bipartite degree sequence has no realization")
    return results


def analyze(source, middle, sink, cutoff, check):
    a, b = len(source), len(sink)
    size, edges = layered(source, middle, sink)
    entry = node_entries(size, edges)
    active_rows, sink_cols = set(), set()
    kept = list()
    row_degrees, col_degrees = [0] * a, [0] * b
    for edge in edges:
        if edge.rank > cutoff:
            continue
        if edge.source == 0:
            active_rows.add(edge.target - 1)
        elif edge.target == size - 1:
            sink_cols.add(edge.source - a - 1)
        else:
            i, j = edge.source - 1, edge.target - a - 1
            kept.append((i, j))
            row_degrees[i] += 1
            col_degrees[j] += 1
    retained_paths = [(i, j) for i, j in kept if i in active_rows and j in sink_cols]
    observed = (len({i for i, _ in retained_paths}), len({j for _, j in retained_paths}))
    if observed != (sum(x <= cutoff for x in entry[1:a + 1]), sum(x <= cutoff for x in entry[a + 1:-1])):
        raise ValueError("Four-layer and bottleneck backbone closures disagree")
    local = (sum(row_degrees[i] > 0 for i in active_rows), sum(col_degrees[j] > 0 for j in sink_cols))
    distribution = exact_distribution(row_degrees, col_degrees, active_rows, sink_cols, check=check)
    if observed not in distribution or any(x > local[0] or y > local[1] for x, y in distribution):
        raise ValueError("Observed topology is outside the conditional null support")
    total = sum(distribution.values())
    count_above = sum(n for (x, y), n in distribution.items() if x + y >= sum(observed))
    return {
        "source_links": len(active_rows), "sink_links": len(sink_cols), "middle_links": len(kept),
        "observed_backbone": list(observed), "local_degree_positives": list(local),
        "null_configurations": str(total), "null_upper_tail_count": str(count_above),
        "null_spanning_count": str(total - distribution.get((0, 0), 0)),
        "null_total_backbone_numerator": str(sum((x + y) * n for (x, y), n in distribution.items())),
        "conditional_null_distribution": [[x, y, str(n)] for (x, y), n in sorted(distribution.items())],
    }
