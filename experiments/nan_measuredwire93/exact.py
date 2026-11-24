"""Exact sender and receiver selection for heterogeneous signed graph ranks."""

from collections import Counter
from fractions import Fraction
from itertools import permutations
from math import factorial, lcm

from experiments.nan_select47.select import mask_sets
from experiments.nan_sourcewire92.permutations import permute


CUTOFFS = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))


def stage_ranks(values, denominator, dtype, np):
    magnitudes = np.abs(values).reshape(-1)
    nonzero = magnitudes[magnitudes > 0.]
    scale = denominator // len(nonzero) if len(nonzero) else 0
    integers = [int((nonzero >= v).sum()) * scale if v else 2 * denominator for v in magnitudes]
    return np.asarray(integers, dtype=dtype).reshape(values.shape)


def encoded(paired, np):
    worlds = [world for pair in paired for world in pair]
    raw = {name: np.asarray([w[name] for w in worlds], dtype=np.float64) for name in ("source", "middle", "sink")}
    if any(not np.isfinite(v).all() for v in raw.values()):
        raise ValueError("The measured graph includes a nonfinite edge")
    live_counts = [int(np.count_nonzero(w)) for group in raw.values() for w in group]
    denominator = lcm(4, *(n for n in live_counts if n))
    # Every pooled criterion sums at most twice this denominator per world.
    # Python integers retain an exact order when heterogeneous stages overflow int64.
    dtype = object if denominator * 2 * len(worlds) > np.iinfo(np.int64).max else np.int64
    rank = {name: np.stack([stage_ranks(w, denominator, dtype, np) for w in group]) for name, group in raw.items()}
    return raw, rank, denominator, dtype


def score(entries, cutoff, denominator, dtype, np):
    hits = (entries <= cutoff).sum(axis=1)
    strength = np.maximum(0, denominator - entries).sum(axis=1, dtype=dtype)
    summed = entries.sum(axis=1, dtype=dtype)
    return hits, strength, summed


def pick(scores, component_ids, quota, np):
    hits, strength, summed = scores
    component_ids = np.asarray(component_ids)
    labels = np.broadcast_to(component_ids, hits.shape)
    selected = np.lexsort((labels, summed, -strength, -hits), axis=1)[:, :quota]
    return np.sort(component_ids[selected], axis=1)


def block(order, raw, rank, scale, cutoff, cfg, dtype, np, *, with_scores=False):
    left, middle, sink = (rank[k] for k in ("source", "middle", "sink"))
    source = left[:, order].transpose(1, 0, 2)
    reverse = np.min(np.maximum(middle, sink[:, None, :]), axis=2)
    sender = np.maximum(source, reverse[None])
    forward = np.min(np.maximum(source[:, :, :, None], middle[None]), axis=2)
    receiver = np.maximum(forward, sink[None])
    threshold = int(scale * cutoff)
    scores = [score(entries, threshold, scale, dtype, np) for entries in (sender, receiver)]
    a = pick(scores[0], cfg["source_components"], cfg["source_quota"], np)
    b = pick(scores[1], cfg["receiver_components"], cfg["receiver_quota"], np)
    if with_scores:
        return a, b, scores
    if cutoff != Fraction(1, 2):
        return a, b, None
    full = ((source[:, :, :, None] <= threshold) & (middle[None] <= threshold)
            & (sink[None, :, None, :] <= threshold))
    products = np.where(full, raw["source"][:, order].transpose(1, 0, 2)[:, :, :, None]
                        * raw["middle"][None] * raw["sink"][None, :, None, :], 0.)
    net = products.sum(axis=(2, 3))
    return a, b, np.stack((net.mean(axis=1), abs(net).mean(axis=1), abs(products).sum(axis=(2, 3)).mean(axis=1)),
                          axis=1)


def ordered_counts(counts):
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))


def equivalent_choice(new, old, ids, scores):
    """A wired difference can be accepted only across an exact score boundary tie."""
    old, new = set(old), set(new)
    if old == new:
        return True
    ranking = dict(zip(ids, zip(*(criterion[0].tolist() for criterion in scores), strict=True), strict=True))
    return len(new) == len(old) and len({ranking[i] for i in (old ^ new)}) == 1


def summarize(paired, cfg, check, np):
    width = len(cfg["source_components"])
    order = np.asarray(list(permutations(range(width))), dtype=np.int16)
    if len(order) != factorial(width):
        raise ValueError("The source-assignment domain did not close")
    raw, ranked, scale, dtype = encoded(paired, np)
    result, signed = {}, None
    for cutoff in CUTOFFS:
        cut_cfg = cfg | {"calibration_cutoff": float(cutoff)}
        reference = mask_sets(paired, cut_cfg)["masks"]
        legacy_identity = (tuple(reference["sieve"]["source"]), tuple(reference["sieve"]["receiver_observer"]))
        frequencies, diagnostics = Counter(), []
        baseline = None
        for lo in range(0, len(order), 128):
            check()
            a, b, stats = block(order[lo:lo + 128], raw, ranked, scale, cutoff, cfg, dtype, np)
            if lo == 0:
                baseline = tuple(a[0]), tuple(b[0])
            frequencies.update((tuple(s), tuple(r)) for s, r in zip(a.tolist(), b.tolist(), strict=True))
            if stats is not None:
                diagnostics.append(stats)
        if sum(frequencies.values()) != len(order):
            raise ValueError("The exact assignment frequency did not close")
        local = {method: record["receiver_observer"] for method, record in reference.items() if method != "sieve"}
        tied = []
        for i in (0, len(order) // 3, len(order) // 2, len(order) - 1):
            computed = mask_sets(permute(paired, order[i]), cut_cfg)["masks"]
            a, b, scores = block(order[i:i + 1], raw, ranked, scale, cutoff, cfg, dtype, np, with_scores=True)
            actual = (a[0].tolist(), b[0].tolist())
            legacy = (computed["sieve"]["source"], computed["sieve"]["receiver_observer"])
            if actual != legacy:
                if not all(equivalent_choice(got, before, ids, points) for got, before, ids, points in zip(
                        actual, legacy, (cfg["source_components"], cfg["receiver_components"]), scores, strict=True)):
                    raise ValueError("Reference selector disagrees beyond an exact boundary tie")
                tied.append(i)
            if any(computed[method]["receiver_observer"] != values for method, values in local.items()):
                raise ValueError("The source-only control changed a receiver-local comparator")
        send, receive = Counter(), Counter()
        for (s, r), count in frequencies.items():
            send[s] += count
            receive[r] += count
        result[str(float(cutoff))] = {
            "identity": {"source": list(baseline[0]), "receiver_observer": list(baseline[1])},
            "legacy_identity": reference["sieve"], "checked_assignment_tie_differences": tied,
            "legacy_identity_joint_frequency": frequencies[legacy_identity],
            "identity_joint_frequency": frequencies[baseline],
            "identity_sender_frequency": send[baseline[0]], "identity_receiver_frequency": receive[baseline[1]],
            "different_receiver_same_sender_frequency": send[baseline[0]] - frequencies[baseline],
            "different_sender_same_receiver_frequency": receive[baseline[1]] - frequencies[baseline],
            "unique_joint_masks": len(frequencies), "unique_senders": len(send), "unique_receivers": len(receive),
            "joint_frequencies": [{"source": list(s), "receiver_observer": list(r), "count": n}
                                  for (s, r), n in ordered_counts(frequencies)],
        }
        if diagnostics:
            values = np.concatenate(diagnostics)
            signed = {name: {"identity": float(v[0]), "median": float(np.median(v)),
                             "min": float(v.min()), "max": float(v.max()),
                             "assignments_at_least_identity": int((v >= v[0] - 1e-12).sum())}
                      for name, v in zip(("mean_signed_path_sum", "mean_absolute_net", "mean_unsigned_path_mass"),
                                         values.T, strict=True)}
    return {"assignments": len(order), "rank_denominator": scale, "arbitrary_precision_ranks": dtype is object,
            "cutoffs": result, "primary_signed_paths": signed}


def nominations(panel, observed):
    primary = panel["cutoffs"]["0.5"]
    joint = primary["joint_frequencies"]
    present = {tuple(mask["source"]) for mask in observed.values()}
    counts = Counter()
    for mask in joint:
        counts[tuple(mask["source"])] += mask["count"]
    chosen = {"sender_reassignment_mode": joint[0]}
    new = [s for s, _ in ordered_counts(counts) if s not in present]
    if new:
        chosen["sender_reassignment_distinct"] = next(mask for mask in joint if tuple(mask["source"]) == new[0])
    current = primary["legacy_identity"]
    pair = next((mask for mask in joint if mask["source"] == current["source"]
                 and mask["receiver_observer"] != current["receiver_observer"]), None)
    if pair:
        chosen["sender_reassignment_receiver_only"] = pair
    return chosen
