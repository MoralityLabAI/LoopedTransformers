"""Share an immutable raw phase-one boundary across same-shape B2 branches."""

from contextlib import contextmanager

from experiments.nan_delay39.instrument import BoundaryInstrument
from mechinterp.nanbeige_bpd_gym.chunked_cache import ChunkedTwoLoopCache


class PhaseObserver(BoundaryInstrument):
    @contextmanager
    def capture_phase(self, phase):
        if self.enabled or self.patch is not None or phase not in (1, 2):
            raise ValueError("Require a single unmodified layer-observation context")
        self.enabled = True
        self.frame = None
        self.applied = 0
        self.residual, self.inputs = {}, {}
        try:
            yield
            expected = {(phase, i) for i in (0, 1)}
            if set(self.residual) != expected or set(self.inputs) != expected or self.applied:
                raise RuntimeError("An independent physical pass did not close both observations")
        finally:
            self.enabled = False
            self.frame = None


class SharedPhaseCache(ChunkedTwoLoopCache):
    def __init__(self, loop, layer, chunk):
        if layer != loop.n - 1:
            raise ValueError("Phase reuse is defined only after the last physical layer")
        super().__init__(loop, prefill_chunk=chunk, observe_layers=(), inplace=True,
                         instrument_factory=lambda model: PhaseObserver(model, layer))
        self.first = None
        self.tokens = None

    def reset(self):
        super().reset()
        self.first = None
        self.tokens = None

    def prime(self, prompts):
        if len(prompts) != 2 or prompts[0] != prompts[1] or not prompts[0]:
            raise ValueError("Phase reuse requires a complete duplicate prompt B2 prefill")
        super().prefill(prompts, capture=False)
        self.tokens = tuple(tuple(row) for row in prompts)

    def _capture(self, h, targets, prefill, capture):
        if not prefill or capture or not self.inplace:
            raise ValueError("The shared boundary supports prefill only")
        with self.instrument.capture_phase(1):
            self._walk(h, 1, targets, True)
            if any(self.caches[0].get_seq_length(i) != h.shape[1] for i in range(self.loop.n)):
                raise RuntimeError("The primed phase-one canvas is incomplete")
            self.first = {"raw": h.detach(), "targets": tuple(targets), "inputs": dict(self.instrument.inputs),
                          "residual": dict(self.instrument.residual)}
        # Phase 2 starts without phase-one K/V or a previously edited branch.
        self.caches = [self.factory(), self.factory()]
        return {}

    def apply(self, prompts, patches):
        import torch

        if (self.first is None or tuple(tuple(row) for row in prompts) != self.tokens or len(patches) != 2):
            raise ValueError("Branched edits must bind to the exact primed B2 tokens")
        with torch.inference_mode():
            canvas = self.first["raw"].clone()
            targets = self.first["targets"]
            for i, patch in enumerate(patches):
                if patch is None:
                    continue
                at = canvas[i, targets[i]]
                if not patch.eligible or canvas.dtype != torch.bfloat16:
                    raise ValueError("The requested boundary replacement did not qualify")
                before = torch.as_tensor(patch.before, dtype=canvas.dtype, device=canvas.device)
                after = torch.as_tensor(patch.after, dtype=canvas.dtype, device=canvas.device)
                if (at.shape != after.shape or not torch.equal(at, before) or torch.equal(at, after)
                        or not torch.isfinite(after).all()):
                    raise ValueError("The branched boundary replacement is stale or degenerate")
                at.copy_(after)
            activated = self.loop.base.norm(canvas)
            a = torch.stack([activated[i, j] for i, j in enumerate(targets)]).detach().clone()
            self.caches[1] = self.factory()
            try:
                with self.instrument.capture_phase(2):
                    later = self._walk(activated, 2, targets, True)
                    if any(self.caches[1].get_seq_length(i) != canvas.shape[1] for i in range(self.loop.n)):
                        raise RuntimeError("A branch did not close its independent phase-two cache")
                    b = torch.stack([later[i, j] for i, j in enumerate(targets)]).detach().clone()
                inputs, residual = ({k: v.clone() for k, v in self.first[name].items()} |
                                    dict(getattr(self.instrument, name)) for name in ("inputs", "residual"))
                return {"boundaries": (a, b), "inputs": inputs, "residual": residual}
            finally:
                self.caches[1] = self.factory()

    def step(self, *_args, **_kwargs):
        raise ValueError("The shared boundary supports prefill only")


class SharedPassCapture:
    def __init__(self, loop, head, *, layer, chunk=128):
        self.cache = SharedPhaseCache(loop, layer, chunk)
        self.head = head

    def prime(self, prompts):
        import torch

        with torch.inference_mode():
            self.cache.prime(prompts)

    def branch(self, prompts, patches=(None, None)):
        import torch

        captured = self.cache.apply(prompts, patches)
        logits = torch.stack([self.head(row).detach().cpu() for row in captured.pop("boundaries")])
        if logits.shape != (2, 2, len(self.head.weight)) or not torch.isfinite(logits).all():
            raise ValueError("The shared phase readout lacks a complete B2 vocabulary head")
        return captured | {"logits": logits, "physical_length": self.cache.physical_length,
                           "semantic_lengths": list(self.cache.semantic_lengths),
                           "applied": sum(p is not None for p in patches)}

    def close(self):
        self.cache.close()
