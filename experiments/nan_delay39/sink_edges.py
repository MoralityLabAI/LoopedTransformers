"""Tape the signed final-label response to each phase-2 output write."""
from .contrasts import dose_status, numpy_state, qualified_pair
from .instrument import prepare


def sink_edges(engine, native, U, labels, cfg, np, guard):
    import torch
    import torch.nn.functional as F

    loop = engine.cache.loop
    target = engine.head.weight[labels].to(loop.device)
    h = numpy_state(native["residual"][2, 1])
    dtype = engine.head.weight.dtype
    dose = cfg["phase2_readout_dose_fraction"]
    before = numpy_state(native["logits"][1, 1])[labels]
    with torch.inference_mode():
        # Bind the smaller fixed-row calculation to the corresponding full-head rows.
        base = torch.as_tensor(h, dtype=dtype, device=loop.device)[None]
        replay = numpy_state(F.linear(loop.base.norm(base), target))[0]
        replay_error = float(abs(before - replay).max())
        if replay_error > cfg["maximum_phase2_readout_replay_logit_delta"]:
            raise RuntimeError("Direct phase-2 head rows no longer close against the full head")
        patches = {}
        for component, u in enumerate(U):
            guard()
            pair = [prepare(h, sign * u, dose, cfg["dose_relative_tolerance"],
                            cfg["minimum_direction_cosine"], np) for sign in (1.0, -1.0)]
            a, b = [dose_status(p) for p in pair]
            entry = {"positive": a, "negative": b, "signed_label_slopes_per_read_coefficient": None}
            if qualified_pair(a, b, cfg):
                values = torch.as_tensor(np.stack([p.after for p in pair]), dtype=dtype, device=loop.device)
                logits = numpy_state(F.linear(loop.base.norm(values), target))
                delta_l2 = sum(p.realized_l2 for p in pair)
                entry["signed_label_slopes_per_read_coefficient"] = (
                    (logits[0] - logits[1]) * float(np.linalg.norm(u)) / delta_l2
                ).tolist()
            patches[str(component)] = entry
    return {"registered_phase2_dose_fraction": dose, "selected_head_replay_max_delta": replay_error,
            "phase2_write_responses": patches}
