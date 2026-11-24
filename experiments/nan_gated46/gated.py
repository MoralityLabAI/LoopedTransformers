"""Physically gate an edited B2 trajectory between pass 1 and pass 2."""
from hashlib import sha256
from itertools import combinations

from experiments.nan_delay39.contrasts import dose_status, numpy_state, qualified_pair
from experiments.nan_delay39.instrument import BoundaryInstrument, DelayedCapture, clean_row_closes, prepare
from experiments.nan_gauge42.transport import head_gradient
from mechinterp.nanbeige_bpd_gym.chunked_cache import ChunkedTwoLoopCache


SIGNS = {"positive": 1, "negative": -1}
ARMS = ("margin_null_donor", "matched_span_sham")


def margin(logits, ids):
    return float(logits[ids[0]] - logits[ids[1]])


def preservation(baseline, reached, tokens, cfg):
    import torch

    if baseline.ndim != 2 or baseline.shape != reached.shape or baseline.shape[0] != 2:
        raise ValueError("Finite gate requires the complete B2 vocabulary tensors")
    if not torch.isfinite(baseline).all() or not torch.isfinite(reached).all():
        raise ValueError("Nonfinite vocabulary tensor")
    if not torch.equal(baseline[0], reached[0]):
        raise RuntimeError("The clean companion head changed during gate evaluation")
    before, after = baseline[1], reached[1]
    current_delta = abs(margin(after, tokens) - margin(before, tokens))
    peak = float((after - before).abs().max())
    same = int(before.argmax()) == int(after.argmax())
    return {"margin_delta_abs": current_delta, "max_abs_logit_delta": peak, "greedy_same": same,
            "passed": same and peak <= cfg["maximum_head_delta"] and current_delta <= cfg["maximum_margin_delta"]}


class BeforeSecond(ChunkedTwoLoopCache):
    def __init__(self, *args, **kwargs):
        self.boundary_gate = None
        super().__init__(*args, **kwargs)

    def _walk(self, h, phase, targets, prefill):
        if phase == 2 and self.boundary_gate is not None:
            import torch

            if not prefill or self.instrument.applied != 1 or len(targets) != 2:
                raise RuntimeError("The pending edit did not reach exactly one last-token row")
            self.boundary_gate(torch.stack([h[b, p] for b, p in enumerate(targets)]))
        return super()._walk(h, phase, targets, prefill)


class GateCapture(DelayedCapture):
    def __init__(self, loop, head, *, layer, chunk=128, cache_factory=None):
        self.cache = BeforeSecond(loop, prefill_chunk=chunk, observe_layers=(), cache_factory=cache_factory,
                                  instrument_factory=lambda model: BoundaryInstrument(model, layer))
        self.instrument = self.cache.instrument
        self.head = head

    def planned_head(self, reference, patch):
        import torch

        loop = self.cache.loop
        with torch.inference_mode():
            raw = torch.stack([reference["residual"][1, 0],
                               torch.as_tensor(patch.after.copy(), dtype=torch.bfloat16)]).to(loop.device)
            return self.head(loop.base.norm(raw)).detach().cpu()

    def capture(self, prompts, patch=None, *, planned=None, baseline=None, tokens=None, cfg=None):
        import torch

        if patch is None:
            if any(item is not None for item in (planned, baseline, tokens, cfg)):
                raise ValueError("Unedited native capture has no pending gate")
            return super().capture(prompts)
        if self.cache.boundary_gate is not None or any(item is None for item in (planned, baseline, tokens, cfg)):
            raise ValueError("Every edited prefill requires an approved all-vocabulary plan")
        if not preservation(baseline, planned, tokens, cfg)["passed"]:
            raise ValueError("An edit with a refused head plan cannot reach the decoder")
        checked = False

        def confirm(boundary):
            nonlocal checked
            actual = self.head(boundary).detach().cpu()
            if checked or not torch.equal(actual, planned):
                raise RuntimeError("Live phase-1 head differs from the gated plan; second pass refused")
            checked = True

        self.cache.boundary_gate = confirm
        try:
            result = super().capture(prompts, patch)
            if not checked or not torch.equal(result["logits"][0], planned):
                raise RuntimeError("An edited trajectory did not close the gated first-pass receipt")
            return result
        finally:
            self.cache.boundary_gate = None


def physical_directions(read_delta, U, gradient, tag, cfg, np):
    selected = U[cfg["phase1_components"]]
    basis, r = np.linalg.qr(selected.T, mode="reduced")
    if any(abs(v) < 1e-10 for v in r.diagonal()):
        raise ValueError("The eligible BPD writes lost physical rank")
    donor = (read_delta[cfg["phase1_components"]] @ selected)
    g = basis.T @ gradient
    v = basis.T @ donor
    if np.linalg.norm(g) <= 1e-12 or np.linalg.norm(v) <= 1e-12:
        return None, None
    g /= np.linalg.norm(g)
    d = v - float(v @ g) * g
    length = float(np.linalg.norm(d))
    if length <= 1e-12:
        return None, 0.0
    d /= length
    gen = np.random.default_rng(int.from_bytes(sha256(f"{cfg['sham_seed_namespace']}|{tag}".encode()).digest()[:8], "big"))
    sham = gen.normal(size=len(d))
    sham -= float(sham @ g) * g + float(sham @ d) * d
    sham_norm = np.linalg.norm(sham)
    if sham_norm <= 1e-12:
        return None, None
    sham /= sham_norm
    directions = dict(zip(ARMS, (basis @ d, basis @ sham)))
    if abs(float(directions[ARMS[0]] @ directions[ARMS[1]])) > 1e-10 or any(
            abs(float(gradient @ value)) > 1e-10 for value in directions.values()):
        raise ValueError("Span sham or margin-null alignment invariant failed")
    return directions, float(length / np.linalg.norm(v))


def outcome(capture, tokens, bank, tok):
    logits = capture["logits"][1, 1]
    winner = int(logits.argmax())
    return {"margin": margin(logits, tokens), "token_id": winner,
            "text": tok.decode([winner], skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "phase2_reads": (numpy_state(capture["inputs"][2, 1]) @ bank).tolist()}


def run_world(engine, baseline, ids, read_delta, bank, writes, tokens, tag, cfg, tok, guard, np):
    norm = engine.cache.loop.base.norm
    raw = numpy_state(baseline["residual"][1, 1])
    w = numpy_state(norm.weight) * (
        numpy_state(engine.head.weight[tokens[0]]) - numpy_state(engine.head.weight[tokens[1]]))
    gradient = head_gradient(raw, w, norm.variance_epsilon, np)
    directions, alignment = physical_directions(read_delta, writes, gradient, tag, cfg, np)
    result = {"source_id": tag, "clean": outcome(baseline, tokens, bank, tok),
              "donor_cosine": alignment, "phase2_edit_count": 0, "cases": {}}
    if directions is None or alignment < cfg["minimum_donor_cosine"]:
        return result | {"status": "direction_unqualified"}
    guard()
    plans, patches = {}, {}
    for arm in ARMS:
        for name, sign in SIGNS.items():
            key = arm + "/" + name
            p = prepare(raw, sign * directions[arm], cfg["phase1_dose_fraction"],
                        cfg["dose_relative_tolerance"], cfg["minimum_direction_cosine"], np)
            patches[key] = p
            record = {"eligible_dose": p.eligible, "requested_l2": p.requested_l2,
                      "realized_l2": p.realized_l2, "direction_cosine": p.cosine}
            if p.eligible:
                guard()
                plans[key] = engine.planned_head(baseline, p)
                record["phase1_gate"] = preservation(baseline["logits"][0], plans[key], tokens, cfg)
            result["cases"][key] = record
    if len(plans) != 4 or any(not case["phase1_gate"]["passed"] for case in result["cases"].values() if case["eligible_dose"]):
        return result | {"status": "phase1_gate_unqualified"}
    if any(not qualified_pair(dose_status(a), dose_status(b), cfg)
           for a, b in combinations(patches.values(), 2)):
        return result | {"status": "matched_dose_unqualified"}
    for key, p in patches.items():
        guard()
        changed = engine.capture([ids, ids], p, planned=plans[key], baseline=baseline["logits"][0], tokens=tokens, cfg=cfg)
        clean_row_closes(baseline, changed, expect_edit=True)
        result["cases"][key]["future"] = outcome(changed, tokens, bank, tok)
        result["phase2_edit_count"] += 1
    return result | {"status": "closed"}
