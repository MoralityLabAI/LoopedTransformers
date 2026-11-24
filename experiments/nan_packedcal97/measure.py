"""Measure the same signed DELAY39 edits using independent packed B2 branches."""

from experiments.nan_delay39.contrasts import contrast_directions, dose_status, numpy_state, qualified_pair, response
from experiments.nan_delay39.instrument import clean_row_closes, prepare
from experiments.nan_delay39.sink_edges import sink_edges


def check_row(clean, repeated, first, second, *, compare_late=True):
    import torch

    if (clean["physical_length"] != repeated["physical_length"]
            or clean["semantic_lengths"] != repeated["semantic_lengths"]):
        raise RuntimeError("Packed branch changed the B2 canvas")
    phases = (1, 2) if compare_late else (1,)
    for phase in phases:
        for key in ("inputs", "residual"):
            if not torch.equal(clean[key][phase, first], repeated[key][phase, second]):
                raise RuntimeError("Packed branch failed physical row parity")
    if compare_late and not torch.equal(clean["logits"][:, first], repeated["logits"][:, second]):
        raise RuntimeError("Packed branch failed exact full-vocabulary row parity")


def collect(reference, shared, ids, donor, V, U, labels, cfg, tag, packs, np, guard, *, audit=False):
    """Return the DELAY39 scientific record separately from execution accounting."""
    import torch

    counts = {"full_B2_prefills": 0, "phase1_primes": 0, "packed_phase2_branches": 0,
              "audited_edited_rows": 0, "qualified_edited_rows": 0}

    def full(prompts, patch=None):
        guard()
        try:
            value = reference.capture(prompts, patch)
            counts["full_B2_prefills"] += 1
            return value
        finally:
            reference.cache.reset()

    native = full([ids, donor])
    read, donor_read = (numpy_state(native["inputs"][1, i]) @ V for i in (0, 1))
    pair = [ids, ids]
    try:
        guard()
        shared.prime(pair)
        counts["phase1_primes"] += 1
        baseline = shared.branch(pair)
        counts["packed_phase2_branches"] += 1
        if not torch.equal(native["inputs"][1, 0], baseline["inputs"][1, 0]):
            raise RuntimeError("Mixed native row and shared duplicate disagree at the original writer")
        check_row(baseline, baseline, 0, 1)
        clean_row_closes(baseline, full(pair), expect_edit=False)
        indices = cfg["phase1_components"]
        directions = {f"component_{i}_{name}": U[i] * sign
                      for i in indices for name, sign in (("positive", 1), ("negative", -1))}
        extra = contrast_directions(read, donor_read, U, indices, tag, np)
        if extra:
            directions.update(extra)
        offered = {name for pair_names in packs for name in pair_names}
        if (len(offered) != sum(map(len, packs)) or len(offered) != 2 * len(indices) + 3
                or set(directions) - offered or any(not 1 <= len(names) <= 2 for names in packs)):
            raise ValueError("Packed edit schedule does not close the registered arms")
        before = numpy_state(baseline["residual"][1, 1])
        edits, cases = {}, {}
        for name, vector in directions.items():
            guard()
            patch = prepare(before, vector, cfg["phase1_dose_fraction"], cfg["dose_relative_tolerance"],
                            cfg["minimum_direction_cosine"], np)
            cases[name] = dose_status(patch)
            if patch.eligible:
                edits[name] = patch
        for name in ("twin_donor", "span_sham", "ambient_sham"):
            if name not in cases:
                cases[name] = {"eligible": False, "reason": "zero_twin_projection"}
        for names in packs:
            selected = [name for name in names if name in edits]
            if not selected:
                continue
            # A singleton keeps the original clean row 0 and edited row 1.
            row_names = selected if len(selected) == 2 else [None, selected[0]]
            guard()
            packed = shared.branch(pair, tuple(edits.get(name) for name in row_names))
            counts["packed_phase2_branches"] += 1
            if packed["applied"] != len(selected):
                raise RuntimeError("Packed branch did not deliver all admitted edits")
            for i, name in enumerate(row_names):
                check_row(baseline, packed, i, i, compare_late=name is None)
                if name is None:
                    continue
                patch = edits[name]
                if audit:
                    exact = full(pair, patch)
                    clean_row_closes(baseline, exact, expect_edit=True)
                    check_row(exact, packed, 1, i)
                    counts["audited_edited_rows"] += 1
                item = response(packed, V, labels, np, row=i)
                item["realized_residual_delta"] = (patch.after - patch.before).tolist()
                cases[name]["response"] = item
                counts["qualified_edited_rows"] += 1
        if counts["qualified_edited_rows"] != len(edits):
            raise RuntimeError("Packed scheduler lost an admitted edit")
        edges = {}
        for i in indices:
            pos, neg = (cases[f"component_{i}_{name}"] for name in ("positive", "negative"))
            if qualified_pair(pos, neg, cfg):
                change = np.asarray(pos["response"]["phase2_read"]) - neg["response"]["phase2_read"]
                edges[str(i)] = (change / (pos["realized_l2"] + neg["realized_l2"])).tolist()
            else:
                edges[str(i)] = None
        pairs = {name: qualified_pair(cases["twin_donor"], cases[name], cfg) for name in ("span_sham", "ambient_sham")}
        terminal = sink_edges(reference, baseline, U, labels, cfg, np, guard)
        scientific = {"baseline": response(baseline, V, labels, np), "self_closes": True,
                      "phase1_source_writes": (abs(read[indices]) * np.linalg.norm(U[indices], axis=1)).tolist(),
                      "phase2_terminal": terminal, "cases": cases, "signed_phase2_edges_per_l2": edges,
                      "donor_sham_pairs": pairs}
        counts["physical_B2_pass_traversals"] = (2 * counts["full_B2_prefills"] + counts["phase1_primes"]
                                                 + counts["packed_phase2_branches"])
        return scientific, counts
    finally:
        shared.cache.reset()
