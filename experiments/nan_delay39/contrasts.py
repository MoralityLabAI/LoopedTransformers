"""Registered IO18 phase-to-phase edge and donor contrasts."""
from hashlib import sha256

from .instrument import clean_row_closes, prepare


def contrast_directions(read, donor_read, writes, component_indices, tag, np):
    """Use the pinned component gauge; keep shams orthogonal to the donor."""
    selected = writes[component_indices]
    span, r = np.linalg.qr(selected.T, mode="reduced")
    if any(abs(r[i, i]) <= 1e-10 for i in range(len(selected))):
        raise ValueError("Seven independent directions are required for the span sham")
    donor = (donor_read[component_indices] - read[component_indices]) @ selected
    if not np.isfinite(donor).all() or np.linalg.norm(donor) == 0:
        return None
    unit = donor / np.linalg.norm(donor)
    seed = int.from_bytes(sha256(tag.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    raw = (span @ rng.normal(size=span.shape[1]), rng.normal(size=len(unit)))
    shams = [v - float(v @ unit) * unit for v in raw]
    if any(np.linalg.norm(v) <= 1e-12 for v in shams):
        raise RuntimeError("Degenerate registered sham direction")
    return {"twin_donor": donor, "span_sham": shams[0], "ambient_sham": shams[1]}


def numpy_state(tensor):
    import torch

    return tensor.detach().to(device="cpu", dtype=torch.float64).numpy()


def dose_status(patch):
    return {"eligible": patch.eligible, "requested_l2": patch.requested_l2,
            "realized_l2": patch.realized_l2, "direction_cosine": patch.cosine}


def qualified_pair(left, right, cfg):
    return (left["eligible"] and right["eligible"] and left["requested_l2"] > 0
            and left["requested_l2"] == right["requested_l2"]
            and abs(left["realized_l2"] - right["realized_l2"])
            <= cfg["max_contrast_dose_gap_fraction"] * left["requested_l2"])


def response(capture, bank, labels, np, row=1):
    heads = numpy_state(capture["logits"][:, row])
    return {
        "phase2_read": (numpy_state(capture["inputs"][2, row]) @ bank).tolist(),
        "phase1_label_logits": heads[0, labels].tolist(),
        "phase2_label_logits": heads[1, labels].tolist(),
        "top_token": heads.argmax(axis=1).tolist(),
    }


def measure_pair(engine, ids, donor_ids, V, U, labels, cfg, tag, np, guard):
    """Measure one recipient and independently streamed clean companion."""
    guard()
    native_pair = engine.capture([ids, donor_ids])
    read = numpy_state(native_pair["inputs"][1, 0]) @ V
    donor_read = numpy_state(native_pair["inputs"][1, 1]) @ V
    guard()
    baseline = engine.capture([ids, ids])
    if not np.array_equal(numpy_state(native_pair["inputs"][1, 0]), numpy_state(baseline["inputs"][1, 0])):
        raise RuntimeError("Matched native row and duplicate geometry disagree")
    # Compare both baseline rows before trusting either as a control.
    for phase in (1, 2):
        for site in ("inputs", "residual"):
            if not np.array_equal(numpy_state(baseline[site][phase, 0]), numpy_state(baseline[site][phase, 1])):
                raise RuntimeError("Duplicate rows produced unequal native physical states")
    heads = numpy_state(baseline["logits"])
    if not np.array_equal(heads[:, 0], heads[:, 1]):
        raise RuntimeError("Duplicate rows produced unequal native readouts")
    before = numpy_state(baseline["residual"][1, 1])
    guard()
    self_result = engine.capture([ids, ids])
    clean_row_closes(baseline, self_result, expect_edit=False)

    indices = cfg["phase1_components"]
    directions = {f"component_{i}_{sign}": U[i] * factor
                  for i in indices for sign, factor in (("positive", 1), ("negative", -1))}
    extra = contrast_directions(read, donor_read, U, indices, tag, np)
    if extra:
        directions.update(extra)

    cases = {}
    for name, direction in directions.items():
        guard()
        patch = prepare(before, direction, cfg["phase1_dose_fraction"],
                        cfg["dose_relative_tolerance"], cfg["minimum_direction_cosine"], np)
        cases[name] = status = dose_status(patch)
        if patch.eligible:
            value = engine.capture([ids, ids], patch)
            clean_row_closes(baseline, value, expect_edit=True)
            include = response(value, V, labels, np)
            include["realized_residual_delta"] = (patch.after - patch.before).tolist()
            status["response"] = include
    for key in ("twin_donor", "span_sham", "ambient_sham"):
        if key not in cases:
            cases[key] = {"eligible": False, "reason": "zero_twin_projection"}
    edges = {}
    for i in indices:
        neg, pos = [cases[f"component_{i}_{sign}"] for sign in ("negative", "positive")]
        if qualified_pair(neg, pos, cfg):
            change = np.asarray(pos["response"]["phase2_read"]) - neg["response"]["phase2_read"]
            edges[str(i)] = (change / (pos["realized_l2"] + neg["realized_l2"])).tolist()
        else:
            edges[str(i)] = None
    donor_pairs = {name: qualified_pair(cases["twin_donor"], cases[name], cfg)
                   for name in ("span_sham", "ambient_sham")}
    from .sink_edges import sink_edges

    terminal = sink_edges(engine, baseline, U, labels, cfg, np, guard)
    source_writes = (abs(read[indices]) * np.linalg.norm(U[indices], axis=1)).tolist()
    return {"baseline": response(baseline, V, labels, np), "self_closes": True,
            "phase1_source_writes": source_writes, "phase2_terminal": terminal,
            "cases": cases, "signed_phase2_edges_per_l2": edges, "donor_sham_pairs": donor_pairs}
