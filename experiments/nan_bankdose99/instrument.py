"""Collect all unique signed bank edits on one independently closed B2 canvas."""

from experiments.nan_delay39.contrasts import numpy_state
from experiments.nan_delay39.instrument import clean_row_closes
from experiments.nan_packedcal97.measure import check_row

from .geometry import CONTRASTS, CURRENT_DRIFT_LIMIT, METHODS, SIGN, materialize, plan


def measure(head, native, labels, torch):
    rows = []
    for logits, clean in zip(head, native, strict=True):
        peak = int(logits.argmax())
        masked = logits.clone()
        masked[peak] = -torch.inf
        rows.append({
            "winner": peak,
            "peak_logit": float(logits[peak]),
            "best_other_logit": float(masked.max()),
            "selected_logits": logits[labels].tolist(),
            "max_abs_logit_drift": float((logits.float() - clean.float()).abs().max()),
        })
    return rows


def collect(reference, shared, ids, peer, banks, labels, cfg, root, world, np, guard, *, audit=False):
    import torch

    def full(prompts, edit=None):
        guard()
        try:
            return reference.capture(prompts, edit)
        finally:
            reference.cache.reset()

    mixed = full([ids, peer])
    prompts = [ids, ids]
    try:
        guard()
        shared.prime(prompts)
        baseline = shared.branch(prompts)
        check_row(baseline, baseline, 0, 1)
        clean_row_closes(baseline, full(prompts), expect_edit=False)
        if not torch.equal(mixed["inputs"][1, 0], baseline["inputs"][1, 0]):
            raise RuntimeError("Donor canvas and duplicate disagree at the native recipient input")
        x, y = (numpy_state(mixed["inputs"][1, i]) for i in (0, 1))
        directions = materialize(y - x, banks, root, world, np)
        before = numpy_state(baseline["residual"][1, 1])
        geometry, patches = plan(before, directions, cfg, np)
        head = baseline["logits"][:, 1]
        responses = []
        for start in range(0, len(patches), 2):
            guard()
            edits = list(patches[start:start + 2])
            if len(edits) == 1:
                edits.insert(0, None)
            out = shared.branch(prompts, tuple(edits))
            if out["applied"] != sum(edit is not None for edit in edits):
                raise RuntimeError("The packed branch failed to deliver the unique edit set")
            for row, edit in enumerate(edits):
                check_row(baseline, out, row, row, compare_late=edit is None)
                if edit is None:
                    continue
                if audit:
                    exact = full(prompts, edit)
                    clean_row_closes(baseline, exact, expect_edit=True)
                    check_row(exact, out, 1, row)
                responses.append(measure(out["logits"][:, row], head, labels, torch))
        return geometry | {"baseline": measure(head, head, labels, torch), "edited_heads": responses,
                           "packed_phase2_branches": (len(patches) + 1) // 2,
                           "independent_full_prefills": 2 + (len(patches) if audit else 0)}
    finally:
        shared.cache.reset()


def score(episode, labels, target, foil):
    if len(set(labels)) != len(labels) or target == foil:
        raise ValueError("Scoring requires unique labels and unequal target and foil")
    a, b = labels.index(target), labels.index(foil)
    before = episode["baseline"]

    def margin(row):
        logits = row["selected_logits"]
        return logits[a] - logits[b]

    def unrestricted(row):
        competitor = row["best_other_logit"] if row["winner"] == target else row["peak_logit"]
        return row["selected_logits"][a] - competitor

    methods = {}
    for method in METHODS:
        cells = episode["cases"][method]
        if any(cells[sign]["physical_edit"] is None for sign, _ in SIGN):
            continue
        rows = [episode["edited_heads"][cells[sign]["physical_edit"]] for sign, _ in SIGN]
        pos, neg = [row[1] for row in rows]
        immediate = [row[0] for row in rows]
        passes = all(row["winner"] == before[0]["winner"]
                     and row["max_abs_logit_drift"] <= CURRENT_DRIFT_LIMIT
                     and abs(margin(row) - margin(before[0])) <= CURRENT_DRIFT_LIMIT for row in immediate)
        methods[method] = {"phase2_paired_label_half_effect": (margin(pos) - margin(neg)) / 2,
                           "phase2_unrestricted_half_effect": (unrestricted(pos) - unrestricted(neg)) / 2,
                           "phase2_winners": [pos["winner"], neg["winner"]], "passes_current_head_gate": passes}
    contrasts = {}
    for left, right in CONTRASTS:
        key = f"{left}_vs_{right}"
        if episode["qualified_panels"][key]:
            x, y = methods[left], methods[right]
            contrasts[key] = {
                "phase2_paired_label_half_effect_delta": x["phase2_paired_label_half_effect"]
                - y["phase2_paired_label_half_effect"],
                "phase2_unrestricted_half_effect_delta": x["phase2_unrestricted_half_effect"]
                - y["phase2_unrestricted_half_effect"],
                "passes_current_head_gate": x["passes_current_head_gate"] and y["passes_current_head_gate"],
            }
    return {"methods": methods, "contrasts": contrasts}
