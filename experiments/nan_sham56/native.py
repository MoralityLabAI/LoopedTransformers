"""Apply the matched SHAM56 control through the live pass-one admission gate."""
from experiments.nan_delay39.contrasts import numpy_state
from experiments.nan_delay39.instrument import clean_row_closes
from experiments.nan_dose51.route import directions, patches
from experiments.nan_gated46.gated import margin, preservation
from experiments.nan_gauge42.transport import head_gradient
from experiments.nan_round54.geometry import make_pair, match

from .geometry import candidate, drive


def continuation(engine, prompt, label_pair, tag, model, policy, cfg, round_cfg, dose, check, masks, np):
    """One fresh duplicate baseline and four signed continuations when admitted."""
    import torch

    model_bank, controller, eps, static_anchor = model
    check()
    baseline = engine.capture([prompt, prompt])
    for value in ("inputs", "residual", "native_writes"):
        if any(not torch.equal(baseline[value][phase, 0], baseline[value][phase, 1]) for phase in (1, 2)):
            raise RuntimeError("Identical baseline rows reached different physical states")
    if any(not torch.equal(baseline["logits"][phase, 0], baseline["logits"][phase, 1]) for phase in (0, 1)):
        raise RuntimeError("Identical baseline rows reached different vocabulary heads")
    h = numpy_state(baseline["residual"][1, 1])
    x = numpy_state(baseline["inputs"][1, 1]).astype(np.float32)
    y = numpy_state(baseline["native_writes"][1, 1]).astype(np.float32)
    effects, _ = directions(x, y, masks(x[None], model_bank, controller)[0], static_anchor, model_bank, eps, np)
    raw_vector = effects["learned"] - effects["static"]
    norm = engine.cache.loop.base.norm
    head_diff = numpy_state(engine.head.weight[label_pair[0]]) - numpy_state(engine.head.weight[label_pair[1]])
    gradient = head_gradient(h, numpy_state(norm.weight) * head_diff, norm.variance_epsilon, np)
    radius = float(np.linalg.norm(h)) * dose["phase1_dose_fraction"]
    dense, original, _ = make_pair(h, raw_vector, gradient, radius, None, tag, round_cfg, np)
    failed = {"status": "physical_unqualified", "edited_pass2_runs": 0}
    if dense is None:
        return failed | {"reason": "degenerate_active_direction"}
    active, _, eligible = patches(h, {"contrast": dense["contrast"]}, dose, np)
    if not eligible:
        return failed | {"reason": "active_dose"}
    method, cap = policy
    v, support = candidate(h, model_bank["U"], original, gradient, [p.after - p.before for p in active.values()],
                           radius, cap, method, drive(tag, cfg["sham_namespace"], len(h), np), np)
    if v is None:
        return failed | {"reason": "degenerate_sham", "sham_support": support}
    edits, cases = match(h, {"contrast": dense["contrast"], "sham": v}, original, round_cfg, dose, np)
    if not cases["physical_gate"]:
        return failed | {"sham_support": support, "cases": cases}
    for key, p in active.items():
        if not np.array_equal(p.after, edits[key].after):
            raise RuntimeError("Sham construction changed the active state")
    current = baseline["logits"][0]
    planned = {}
    for key, edit in edits.items():
        check()
        planned[key] = engine.planned_head(baseline, edit)
        cases["arm_dose"][key]["live_phase1"] = preservation(current, planned[key], label_pair, dose)
    if not all(row["live_phase1"]["passed"] for row in cases["arm_dose"].values()):
        return {"status": "head_unqualified", "cases": cases, "sham_support": support, "edited_pass2_runs": 0}

    def outcome(packet):
        logits = packet["logits"][1, 1]
        return {"bare_margin": margin(logits, label_pair), "winner": int(logits.argmax()),
                "phase2_bpd_read": (numpy_state(packet["inputs"][2, 1]) @ model_bank["V"]).tolist()}

    reached = {}
    for key, patch in edits.items():
        check()
        live = engine.capture([prompt, prompt], patch, planned=planned[key], baseline=current, tokens=label_pair, cfg=dose)
        clean_row_closes(baseline, live, expect_edit=True)
        if (any(not torch.equal(baseline["native_writes"][phase, 0], live["native_writes"][phase, 0])
                for phase in (1, 2))
                or any(not torch.equal(baseline[field][1, 1], live[field][1, 1]) for field in ("inputs", "native_writes"))):
            raise RuntimeError("Unedited projection changed during the live gate")
        reached[key] = outcome(live)
    effects = {arm: .5 * (reached[f"{arm}/positive"]["bare_margin"] - reached[f"{arm}/negative"]["bare_margin"])
               for arm in ("contrast", "sham")}
    return {"status": "closed", "cases": cases, "baseline": outcome(baseline), "sham_support": support,
            "outcomes": reached, "signed_pass2_margin_contrast_minus_sham": effects["contrast"] - effects["sham"],
            "edited_pass2_runs": len(reached)}
