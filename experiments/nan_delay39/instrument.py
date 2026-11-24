"""Last-valid-token phase-1 boundary patch with clean-row closure."""
from contextlib import contextmanager
from dataclasses import dataclass

from experiments.nan_dose38.run import MAX_MULTIPLE, STEPS, displacement


@dataclass(frozen=True)
class Patch:
    before: object
    after: object
    requested_l2: float
    realized_l2: float
    cosine: float
    eligible: bool


def prepare(before, direction, fraction, relative_error, minimum_cosine, np):
    """Return the actual reached, BF16-rounded residual and audit its dose."""
    from experiments.nan_readpath37.run import bf16

    base = np.asarray(before, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if base.shape != direction.shape or base.ndim != 1 or not np.isfinite(direction).all():
        raise ValueError("A direction must be finite and match the captured residual")
    if not np.array_equal(bf16(base, np), base):
        raise ValueError("Residue was not captured in the registered BF16 coordinates")
    target = float(np.linalg.norm(base) * fraction)
    length = float(np.linalg.norm(direction))
    if length <= 0 or target <= 0 or not np.isfinite(target):
        return Patch(base, base.copy(), target, 0.0, 0.0, False)
    unit = direction[None, :] / length
    low, high = 0.0, target * MAX_MULTIPLE
    brackets = float(np.linalg.norm(displacement(base, unit, np.array([high]), np))) >= target
    for _ in range(STEPS):
        m = .5 * (low + high)
        if np.linalg.norm(displacement(base, unit, np.array([m]), np)) < target:
            low = m
        else:
            high = m
    candidates = [displacement(base, unit, np.array([value]), np)[0] for value in (low, high)]
    patch = min(candidates, key=lambda v: abs(float(np.linalg.norm(v)) - target))
    realized = float(np.linalg.norm(patch))
    cosine = float(patch @ unit[0] / realized) if realized > 0 else 0.0
    eligible = brackets and abs(realized / target - 1.0) <= relative_error and cosine >= minimum_cosine
    return Patch(base, base + patch, target, realized, cosine, bool(eligible))


class BoundaryInstrument:
    def __init__(self, loop, layer):
        self.loop = loop
        self.layer = layer
        self.frame = None
        self.enabled = False
        self.patch = None
        self.applied = 0
        self.residual = {}
        self.inputs = {}
        self.handles = []
        modules = [loop.layers[layer]]
        if loop.stage_layer is not None and layer >= loop.n_gpu_layers:
            modules.append(loop.stage_layer)
        for module in modules:
            self.handles.append(module.register_forward_hook(self.at_layer_output))
            self.handles.append(module.mlp.down_proj.register_forward_pre_hook(self.at_projection_input))

    def set_frame(self, frame):
        self.frame = frame

    @contextmanager
    def request(self, patch):
        if self.patch is not None or self.enabled or (patch is not None and not patch.eligible):
            raise ValueError("Cannot nest or apply an unqualified boundary edit")
        self.patch = patch
        try:
            yield
        finally:
            self.patch = None

    @contextmanager
    def context(self, modes):
        if self.enabled or tuple(modes) != ("native", "native"):
            raise ValueError("Boundary capture requires two physical native-mode rows")
        self.enabled = True
        self.frame = None
        self.applied = 0
        self.residual = {}
        self.inputs = {}
        try:
            yield
            if self.applied != int(self.patch is not None):
                raise RuntimeError("One eligible phase-1 row must be edited exactly once")
            expected = {(phase, row) for phase in (1, 2) for row in (0, 1)}
            if set(self.residual) != expected or set(self.inputs) != expected:
                raise RuntimeError("Incomplete paired residual or projection observations")
        finally:
            self.enabled = False
            self.frame = None

    def observations(self, value, out):
        if not self.enabled or self.loop.active_layer != self.layer:
            return
        frame = self.frame
        if (frame is None or not frame.prefill or frame.phase not in (1, 2)
                or len(frame.capture_offsets) != 2 or value.shape[0] != 2):
            raise RuntimeError("A current B2 prefill frame is required")
        for row, offset in enumerate(frame.capture_offsets):
            if offset is None:
                continue
            key = frame.phase, row
            if key in out or offset != frame.valid_lengths[row] - 1:
                raise RuntimeError("Physical observation is not the last semantic token")
            out[key] = value[row, offset].detach().cpu().clone()

    def at_projection_input(self, module, args):
        self.observations(args[0], self.inputs)

    def at_layer_output(self, module, args, result):
        import torch

        states = result[0]
        self.observations(states, self.residual)
        if (not self.enabled or self.loop.active_layer != self.layer or self.patch is None
                or self.frame.phase != 1 or self.frame.capture_offsets[1] is None):
            return None
        offset = self.frame.capture_offsets[1]
        original = states[1, offset]
        before = torch.as_tensor(self.patch.before, dtype=states.dtype, device=states.device)
        after = torch.as_tensor(self.patch.after, dtype=states.dtype, device=states.device)
        if states.dtype != torch.bfloat16 or not torch.equal(original, before):
            raise RuntimeError("Pre-edit bytes differ from the frozen B2 row")
        if torch.equal(original, after):
            raise RuntimeError("The qualified edit rounded away")
        edited = states.clone()
        edited[1, offset].copy_(after)
        self.applied += 1
        return (edited, *result[1:])

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class DelayedCapture:
    def __init__(self, loop, head, *, layer, chunk=128, cache_factory=None):
        from mechinterp.nanbeige_bpd_gym.chunked_cache import ChunkedTwoLoopCache

        self.cache = ChunkedTwoLoopCache(loop, prefill_chunk=chunk, observe_layers=(),
                                         cache_factory=cache_factory,
                                         instrument_factory=lambda model: BoundaryInstrument(model, layer))
        self.instrument = self.cache.instrument
        self.head = head

    def capture(self, prompts, patch=None):
        import torch

        if len(prompts) != 2 or (patch is not None and prompts[0] != prompts[1]):
            raise ValueError("An edited prefill requires its duplicate clean row")
        with self.instrument.request(patch):
            result = self.cache.prefill(prompts, capture=False)
        lengths = list(map(len, prompts))
        if self.cache.semantic_lengths != lengths or self.cache.physical_length != max(lengths):
            raise RuntimeError("Physical and semantic cache lengths disagree")
        logits = torch.stack([self.head(h).detach().cpu() for h in result["boundaries"]])
        if logits.shape[:2] != (2, 2) or logits.shape[2] != self.head.weight.shape[0] or not torch.isfinite(logits).all():
            raise RuntimeError("Incomplete full-vocabulary phase readout")
        return {"logits": logits, "inputs": dict(self.instrument.inputs),
                "residual": dict(self.instrument.residual), "physical_length": self.cache.physical_length,
                "semantic_lengths": lengths, "applied": self.instrument.applied}

    def close(self):
        self.cache.close()


def clean_row_closes(reference, observed, expect_edit):
    import torch

    if (reference["semantic_lengths"] != observed["semantic_lengths"]
            or reference["physical_length"] != observed["physical_length"]
            or observed["applied"] != int(expect_edit)):
        raise RuntimeError("Native and intervention batch identities differ")
    if not torch.equal(reference["logits"][:, 0], observed["logits"][:, 0]):
        raise RuntimeError("The companion's full-vocabulary logits changed")
    for site in ("inputs", "residual"):
        for phase in (1, 2):
            if not torch.equal(reference[site][phase, 0], observed[site][phase, 0]):
                raise RuntimeError("The companion's physical states changed")
    if not expect_edit:
        for site in ("inputs", "residual"):
            for phase in (1, 2):
                if not torch.equal(reference[site][phase, 1], observed[site][phase, 1]):
                    raise RuntimeError("The self row's physical states changed")
        if not torch.equal(reference["logits"], observed["logits"]):
            raise RuntimeError("The self row's full-vocabulary logits changed")
