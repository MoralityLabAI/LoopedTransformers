"""Native continuation helper for matched learned/static edited B2 conditions."""
from experiments.nan_delay39.instrument import BoundaryInstrument, clean_row_closes
from experiments.nan_gated46.gated import BeforeSecond, GateCapture, preservation

from .route import ARMS, SIGNS, patches


class WriteCapture(BoundaryInstrument):
    def __init__(self, loop, layer):
        super().__init__(loop, layer)
        self.writes = {}
        modules = [loop.layers[layer]]
        if loop.stage_layer is not None and layer >= loop.n_gpu_layers:
            modules.append(loop.stage_layer)
        for module in modules:
            self.handles.append(module.mlp.down_proj.register_forward_hook(self.capture_write))

    def capture_write(self, _module, _args, result):
        self.observations(result, self.writes)


class RouteCapture(GateCapture):
    def __init__(self, loop, head, *, layer, chunk=128, cache_factory=None):
        self.cache = BeforeSecond(loop, prefill_chunk=chunk, observe_layers=(), cache_factory=cache_factory,
                                  instrument_factory=lambda model: WriteCapture(model, layer))
        self.instrument = self.cache.instrument
        self.head = head

    def capture(self, prompts, patch=None, **gate):
        self.instrument.writes.clear()
        result = super().capture(prompts, patch, **gate)
        actual = dict(self.instrument.writes)
        if set(actual) != {(phase, row) for phase in (1, 2) for row in (0, 1)}:
            raise RuntimeError("Missing exact native module write observation")
        return result | {"native_writes": actual}


def gated_continuations(engine, ids, reference, effects, label_pair, cfg, np):
    import torch
    from experiments.nan_delay39.contrasts import numpy_state

    raw = numpy_state(reference["residual"][1, 1])
    prepared, summaries, matched = patches(raw, effects, cfg, np)
    if not matched:
        return {"status": "dose_unqualified", "cases": summaries, "edited_pass2_runs": 0}
    clean_head = reference["logits"][0]
    planned = {key: engine.planned_head(reference, value) for key, value in prepared.items()}
    for key, head in planned.items():
        summaries[key]["phase1_gate"] = preservation(clean_head, head, label_pair, cfg)
    if not all(c["phase1_gate"]["passed"] for c in summaries.values()):
        return {"status": "head_unqualified", "cases": summaries, "edited_pass2_runs": 0}
    outcomes, calls = {}, 0
    for sign in SIGNS:
        key_a, key_b = (f"{a}/{sign}" for a in ARMS)
        if np.array_equal(prepared[key_a].after, prepared[key_b].after):
            keys = [(key_a, [key_a, key_b])]
        else:
            keys = [(key_a, [key_a]), (key_b, [key_b])]
        for lane, aliases in keys:
            reached = engine.capture([ids, ids], prepared[lane], planned=planned[lane], baseline=clean_head,
                                     tokens=label_pair, cfg=cfg)
            clean_row_closes(reference, reached, expect_edit=True)
            if any(not torch.equal(reference["native_writes"][p, 0], reached["native_writes"][p, 0]) for p in (1, 2)):
                raise RuntimeError("Unedited companion native projection changed")
            for name in aliases:
                outcomes[name] = reached
            calls += 1
    return {"status": "closed", "cases": summaries, "outcomes": outcomes, "edited_pass2_runs": calls}
