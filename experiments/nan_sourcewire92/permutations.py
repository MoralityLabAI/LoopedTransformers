"""Enumerate source reassignments while freezing the receiver-local graph."""

from collections import Counter
from itertools import permutations
from math import factorial, lcm

from experiments.nan_readpath37.run import ranked_entries
from experiments.nan_select47.select import mask_sets


CUTOFFS = (.25, .5, .75)


def ranks(values, scale, np):
    flat = values.reshape(-1)
    ranked = np.asarray(ranked_entries(flat))
    present = np.isfinite(ranked)
    out = np.full(len(flat), 2 * scale, dtype=np.int32)
    reached = ranked[present] * scale
    if not np.allclose(np.rint(reached), reached, rtol=0, atol=1e-10):
        raise ValueError("Exact staged rank scaling did not close")
    out[present] = np.rint(reached).astype(np.int32)
    return out.reshape(values.shape)


def condition_graphs(paired, np):
    flat = [world for pair in paired for world in pair]
    raw = [[np.asarray(w[name]) for w in flat] for name in ("source", "middle", "sink")]
    sizes = [np.count_nonzero(stage) for group in raw for stage in group]
    scale = lcm(4, *(int(v) for v in sizes if v))
    if scale > 2**24:
        raise ValueError("Staged denominator exceeds exact integer storage")
    encoded = [np.stack([ranks(v, scale, np) for v in group]) for group in raw]
    return scale, encoded, [np.stack(v) for v in raw]


def permute(paired, order):
    return [[world | {"source": [world["source"][i] for i in order]} for world in root] for root in paired]


def blocks(scale, rank, raw, orders, cutoff, quota, np):
    source, middle, sink = rank
    changed = source[:, orders].transpose(1, 0, 2)
    entry = np.minimum.reduce(np.maximum(changed[:, :, :, None], middle[None]), axis=2)
    entries = np.maximum(entry, sink[None])
    threshold = int(scale * cutoff)
    hits = (entries <= threshold).sum(axis=1)
    salience = np.maximum(0, scale - entries).sum(axis=1, dtype=np.int64)
    total = entries.sum(axis=1, dtype=np.int64)
    index = np.broadcast_to(np.arange(middle.shape[-1]), hits.shape)
    winners = np.lexsort((index, total, -salience, -hits), axis=1)[:, :quota]
    masks = np.sort(winners, axis=1)
    if cutoff != .5:
        return masks, None
    source_value, middle_value, sink_value = raw
    mapped = source_value[:, orders].transpose(1, 0, 2)
    admissible = ((changed[:, :, :, None] <= threshold) & (middle[None] <= threshold)
                  & (sink[None, :, None, :] <= threshold))
    products = np.where(admissible, mapped[:, :, :, None] * middle_value[None] * sink_value[None, :, None], 0.)
    net = products.sum(axis=(2, 3))
    unsigned = abs(products).sum(axis=(2, 3))
    return masks, np.stack((net.mean(axis=1), abs(net).mean(axis=1), unsigned.mean(axis=1)), axis=1)


def summarize(paired, cfg, check, np):
    width = len(cfg["source_components"])
    orders = np.asarray(list(permutations(range(width))), dtype=np.int16)
    base = tuple(range(width))
    if tuple(orders[0]) != base or len(orders) != factorial(width):
        raise ValueError("Source assignments are incomplete")
    denom, encoded, raw = condition_graphs(paired, np)
    counts = {}
    primary_values = None
    for cutoff in CUTOFFS:
        options = cfg | {"calibration_cutoff": cutoff}
        reference = mask_sets(paired, options)["masks"]
        identity = tuple(reference["sieve"]["receiver_observer"])
        local = {k: tuple(reference[k]["receiver_observer"]) for k in (
            "local_degree", "local_weighted_degree", "isolated_margin_response")}
        counter = Counter()
        signed = []
        selected = None
        for start in range(0, len(orders), 256):
            check()
            end = min(start + 256, len(orders))
            block, diagnostics = blocks(denom, encoded, raw, orders[start:end], cutoff,
                                        cfg["receiver_quota"], np)
            if start == 0:
                selected = tuple(block[0])
            counter.update(map(tuple, block.tolist()))
            if diagnostics is not None:
                signed.append(diagnostics)
        if selected != identity or sum(counter.values()) != len(orders):
            raise ValueError("Vectorized exact ranks disagree with the original sieve")
        for i in (0, 1, len(orders) // 3, len(orders) // 2, len(orders) - 1):
            actual, _ = blocks(denom, encoded, raw, orders[i:i+1], cutoff, cfg["receiver_quota"], np)
            checked = mask_sets(permute(paired, orders[i]), options)["masks"]
            if actual[0].tolist() != checked["sieve"]["receiver_observer"] or any(
                    tuple(checked[k]["receiver_observer"]) != v for k, v in local.items()):
                raise ValueError("A source permutation changed an isolated receiver comparator")
        ranked = sorted(counter.items(), key=lambda value: (-value[1], value[0]))
        counts[str(cutoff)] = {"identity_mask": list(identity), "identity_matches": counter[identity],
                               "local_masks": {k: list(v) for k, v in local.items()},
                               "local_alias_counts": {k: counter[v] for k, v in local.items()},
                               "distinct_masks": len(counter),
                               "frequencies": [{"mask": list(k), "assignments": v} for k, v in ranked]}
        if signed:
            values = np.concatenate(signed)
            primary_values = {
                name: {"identity": float(v[0]), "median": float(np.median(v)), "min": float(v.min()),
                       "max": float(v.max()), "assignments_at_least_identity": int((v >= v[0] - 1e-12).sum())}
                for name, v in zip(("mean_signed_path_sum", "mean_absolute_net", "mean_absolute_path_mass"),
                                   values.T, strict=True)}
    return {"source_assignments": len(orders), "exact_rank_denominator": denom,
            "cutoffs": counts, "primary_signed_paths": primary_values}


def fixture(np):
    item = {"source": [2., 2., -1.], "middle": [[2., 0., -2.], [1., 2., 0.], [-1., 1., 2.]],
            "sink": [1., -1., 2.], "activity1": [1., 2., 3.], "activity2": [1., 1., 3.],
            "isolated1": [1., 1., 1.], "isolated2": [1., 2., 3.]}
    paired = [[item, item | {"source": [0., -2., 4.]}]]
    cfg = {"source_components": [0, 1, 2], "source_quota": 1, "receiver_components": [0, 1, 2],
           "receiver_quota": 2, "minimum_complete_CAL_roots": 1}
    scale, encoded, raw = condition_graphs(paired, np)
    orders = np.asarray(list(permutations(range(3))))
    for cutoff in CUTOFFS:
        proposed, _ = blocks(scale, encoded, raw, orders, cutoff, 2, np)
        for row, order in zip(proposed, orders, strict=True):
            expected = mask_sets(permute(paired, order), cfg | {"calibration_cutoff": cutoff})
            assert row.tolist() == expected["masks"]["sieve"]["receiver_observer"]
    return {"permutation_cutoff_cells_exact": 18, "zero_and_tied_weights": True}
