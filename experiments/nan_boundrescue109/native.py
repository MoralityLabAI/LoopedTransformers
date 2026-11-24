"""Before-pass-2 qualification and reader rescue for raw phase-1 residual edits."""

from contextlib import contextmanager
from dataclasses import dataclass

from experiments.nan_gated46.gated import BeforeSecond, preservation
from experiments.nan_premlp64.capture import same_bf16
from experiments.nan_radial76.native import WRITER, WRITER_SITES, WriterReaderCapture, check_writer
from experiments.nan_reader71.native import SELF, SITES, ReaderInstrument, summarize


class ResidualReader(ReaderInstrument):
    def __init__(self, loop):
        super().__init__(loop, 0, None)
        self.writer = loop.n - 1
        module = loop.layers[self.writer] if self.writer < loop.n_gpu_layers else loop.stage_layer
        if module is None:
            raise ValueError("The last physical layer has no execution module")
        self.handles.extend([
            module.mlp.down_proj.register_forward_pre_hook(self.read_writer_input),
            module.register_forward_hook(self.write_boundary),
        ])
        self.writer_edit = None
        self.writer_edits = 0

    @property
    def applied(self):
        return self.writer_edits

    @contextmanager
    def context(self, modes):
        with super().context(modes):
            self.values.update({name: {} for name in WRITER_SITES})
            self.writer_edits = 0
            yield
            if self.writer_edits != int(self.writer_edit is not None):
                raise RuntimeError("A scoped raw boundary edit must run exactly once")

    def read_writer_input(self, _module, args):
        if self.active and self.loop.active_layer == self.writer:
            self.record("writer_input", args[0])

    def write_boundary(self, _module, _args, result):
        if not self.active or self.loop.active_layer != self.writer:
            return None
        raw = result[0]
        self.record("writer_raw", raw)
        if self.writer_edit is not None and self.frame.phase == 1 and self.frame.capture_offsets[1] is not None:
            reached = self.replace(raw, self.writer_edit)
            self.record("writer_reached", reached)
            self.writer_edits += 1
            return (reached, *result[1:])
        self.record("writer_reached", raw)
        return None


@dataclass(frozen=True)
class HeadPlan:
    normalized: object
    logits: object


class BoundaryCapture(WriterReaderCapture):
    def __init__(self, loop, head, *, chunk=128):
        self.cache = BeforeSecond(loop, prefill_chunk=chunk, observe_layers=(), instrument_factory=ResidualReader)
        self.instrument = self.cache.instrument
        self.head = head

    def predict(self, clean, after):
        import torch

        loop = self.cache.loop
        reference = clean["writer_raw"][1, 0]
        if after.shape != reference.shape or after.dtype != reference.dtype or after.device.type != "cpu":
            raise ValueError("The plan needs a native BF16 raw-boundary tensor on CPU")
        with torch.inference_mode():
            normalized = loop.base.norm(torch.stack([reference, after]).to(loop.device))
            return HeadPlan(normalized.detach().cpu().clone(), self.head(normalized).detach().cpu().clone())

    def capture(self, prompts, *, edit=None, rescue=None, planned=None):
        import torch

        if edit is None:
            if planned is not None:
                raise ValueError("An unedited trajectory cannot carry an edit plan")
            return super().capture(prompts, rescue=rescue)
        if planned is None or self.cache.boundary_gate is not None:
            raise ValueError("Every raw edit requires a frozen first-pass plan")
        checked = False

        def confirm(boundary):
            nonlocal checked
            if checked or not same_bf16(boundary.detach().cpu(), planned.normalized):
                raise RuntimeError("The planned first-pass boundary differs before pass 2")
            checked = True

        self.cache.boundary_gate = confirm
        try:
            result = super().capture(prompts, edit=edit, rescue=rescue)
            if not checked or not torch.equal(result["logits"][0], planned.logits):
                raise RuntimeError("The physical first-pass head differs from the checked plan")
            return result
        finally:
            self.cache.boundary_gate = None


def execute_arm(capture, prompt, clean, after, planned, labels, check):
    import torch

    prompts = [prompt, prompt]
    edit = clean["writer_raw"][WRITER], after
    check()
    observed = capture.capture(prompts, edit=edit, planned=planned)
    check_writer(clean, observed)
    if not same_bf16(observed["writer_reached"][WRITER], after):
        raise RuntimeError("The raw residual write differs from the selected pair")
    rescue = observed["k_raw"][SELF], clean["k_raw"][SELF]
    check()
    rescued = capture.capture(prompts, edit=edit, planned=planned, rescue=rescue)
    check_writer(clean, rescued)
    if (not torch.equal(rescued["logits"][0], observed["logits"][0])
            or not same_bf16(rescued["writer_reached"][WRITER], after)
            or not same_bf16(rescued["k_reached"][SELF], rescue[1])
            or any(not same_bf16(rescued[site][SELF], observed[site][SELF]) for site in SITES if site != "k_reached")):
        raise RuntimeError("Restoring pass-2 K changed an unintended site")
    return {"edited": summarize(clean, observed, labels), "k_rescue": summarize(clean, rescued, labels),
            "zeroed_k_delta": same_bf16(clean["k_raw"][SELF], rescued["k_reached"][SELF])}


def grade_first_head(clean, heads, labels, np, *, design):
    from experiments.nan_bankdose99.geometry import CURRENT_DRIFT_LIMIT
    from experiments.nan_chordsham106.geometry import MAX_RMS_GAP
    from experiments.nan_headswap107.search import pair_rms

    cfg = dict(maximum_head_delta=CURRENT_DRIFT_LIMIT, maximum_margin_delta=CURRENT_DRIFT_LIMIT)
    gates = [preservation(clean, p.logits, labels, cfg) for p in heads]
    values = np.stack([p.logits[1].double().numpy() for p in heads]).reshape(2, 2, -1)
    baseline = clean[1].double().numpy()
    ids = np.asarray(design)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or not 0 < len(ids) < len(baseline) or (
            len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(baseline))):
        raise ValueError("The selection row grid does not partition the native vocabulary")
    keep = np.ones(len(baseline), dtype=bool)
    keep[ids] = False
    a, s = pair_rms(values, baseline, np)
    blind_active, blind_sham = pair_rms(values[:, :, keep], baseline[keep], np)
    gaps, blind_gaps = abs(a-s), abs(blind_active-blind_sham)
    contrast = ((values[0, 1] - values[0, 0]) - (values[1, 1] - values[1, 0])) / 2
    return {
        "passed": all(g["passed"] for g in gates) and bool(max(max(gaps), max(blind_gaps)) <= MAX_RMS_GAP),
        "arm_gates": gates, "active_rms": a.tolist(), "sham_rms": s.tolist(), "absolute_rms_gaps": gaps.tolist(),
        "excluded_rms_gaps": blind_gaps.tolist(),
        "paired_label_contrast": float(contrast[labels[0]] - contrast[labels[1]]),
    }
