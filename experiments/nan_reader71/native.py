"""Instrument the actual pass-2 carry and projection outputs in a B2 prefill."""
from contextlib import contextmanager
from statistics import fmean

from experiments.nan_premlp64.capture import PAIRED_PHASES, same_bf16
from mechinterp.nanbeige_bpd_gym.chunked_cache import ChunkedTwoLoopCache

from .controls import COORDINATE, SHAM_COUNT, plan


CHANNELS = ("q", "k", "v")
SELF = (2, 1)
SITES = ("carry", "carry_reached", *(f"{channel}_{side}" for channel in CHANNELS for side in ("input", "raw", "reached")))


class ReaderInstrument:
    def __init__(self, loop, reader, replay):
        self.loop = loop
        self.reader = reader
        self.replay = replay
        self.frame = None
        self.active = False
        self.edit = self.rescue = None
        self.handles = []
        module = loop.layers[reader] if reader < loop.n_gpu_layers else loop.stage_layer
        if module is None:
            raise ValueError("The reader has no execution layer")
        self.handles.append(module.register_forward_pre_hook(self.carry))
        for channel in CHANNELS:
            projection = getattr(module.self_attn, f"{channel}_proj")

            def hook(projection, args, output, name=channel):
                return self.project(name, projection, args[0], output)

            self.handles.append(projection.register_forward_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def set_frame(self, frame):
        self.frame = frame

    @contextmanager
    def context(self, modes):
        if self.active or tuple(modes) != ("native", "native"):
            raise ValueError("Reader capture requires two native rows")
        self.active = True
        self.values = {site: {} for site in SITES}
        self.edits = self.rescues = 0
        try:
            yield
            if (any(set(rows) != PAIRED_PHASES for rows in self.values.values())
                    or self.edits != int(self.edit is not None)
                    or self.rescues != int(self.rescue is not None)):
                raise RuntimeError("Incomplete last-token reader capture or unexpected hook count")
        finally:
            self.active = False
            self.frame = None

    def record(self, site, tensor):
        import torch

        frame = self.frame
        if (frame is None or not frame.prefill or frame.phase not in (1, 2)
                or len(frame.capture_offsets) != 2 or tensor.shape[:2] != (2, frame.stop - frame.start)
                or tensor.ndim != 3 or tensor.dtype != torch.bfloat16):
            raise RuntimeError("Reader requires a framed native BF16 B2 prefill")
        for row, offset in enumerate(frame.capture_offsets):
            if offset is None:
                continue
            key = frame.phase, row
            if offset != frame.valid_lengths[row] - 1 or key in self.values[site]:
                raise RuntimeError("Reader state does not bind one last semantic token")
            state = tensor[row, offset].detach().cpu().clone()
            if not torch.isfinite(state).all():
                raise RuntimeError("Nonfinite physical reader state")
            self.values[site][key] = state

    def selected(self):
        return self.frame.phase == 2 and self.frame.capture_offsets[1] is not None

    def replace(self, tensor, pair):
        import torch

        before, after = pair
        original = tensor[1, self.frame.capture_offsets[1]]
        if (not same_bf16(original.detach().cpu(), before) or after.dtype != torch.bfloat16
                or after.shape != before.shape or not torch.isfinite(after).all()):
            raise RuntimeError("Patch did not bind the planned native reader bytes")
        updated = tensor.clone()
        updated[1, self.frame.capture_offsets[1]].copy_(after)
        return updated

    def carry(self, _module, args):
        if self.active and self.loop.active_layer == self.reader:
            self.record("carry", args[0])
            if self.edit is not None and self.selected():
                updated = self.replace(args[0], self.edit)
                self.record("carry_reached", updated)
                self.edits += 1
                return (updated, *args[1:])
            self.record("carry_reached", args[0])
        return None

    def project(self, channel, projection, value, output):
        if not self.active or self.loop.active_layer != self.reader:
            return None
        if self.replay is not None:
            self.replay(projection, value, output)
        self.record(f"{channel}_input", value)
        self.record(f"{channel}_raw", output)
        if channel == "k" and self.rescue is not None and self.selected():
            reached = self.replace(output, self.rescue)
            self.rescues += 1
            self.record("k_reached", reached)
            return reached
        self.record(f"{channel}_reached", output)
        return None


class ReaderCapture:
    def __init__(self, loop, head, *, chunk=128, replay=None):
        self.cache = ChunkedTwoLoopCache(loop, prefill_chunk=chunk, observe_layers=(),
                                         instrument_factory=lambda model: ReaderInstrument(model, 0, replay))
        self.instrument = self.cache.instrument
        self.head = head

    def capture(self, prompts, *, edit=None, rescue=None):
        import torch

        hook = self.instrument
        if (len(prompts) != 2 or hook.active or (rescue is not None and edit is None)
                or (edit is not None and prompts[0] != prompts[1])):
            raise ValueError("Reader edits and rescues require a paired duplicate baseline")
        hook.edit, hook.rescue = edit, rescue
        try:
            result = self.cache.prefill(prompts, capture=False)
            logits = torch.stack([self.head(state).detach().cpu() for state in result["boundaries"]])
            if logits.shape != (2, 2, len(self.head.weight)) or not torch.isfinite(logits).all():
                raise RuntimeError("Incomplete unrestricted vocabulary head")
            return hook.values | {"logits": logits, "edits": hook.edits, "rescues": hook.rescues}
        finally:
            hook.edit = hook.rescue = None
            self.cache.reset()

    def close(self):
        self.cache.close()


def check_control(clean, observed, *, identity=False):
    import torch

    keys = PAIRED_PHASES if identity else PAIRED_PHASES - {SELF}
    if (any(not same_bf16(clean[site][key], observed[site][key]) for site in SITES for key in keys)
            or any(not torch.equal(clean["logits"][phase - 1, row], observed["logits"][phase - 1, row])
                   for phase, row in keys)
            or not same_bf16(clean["carry"][SELF], observed["carry"][SELF])):
        raise RuntimeError("The clean companion, pass-1 head, or pre-edit carry changed")


def summarize(clean, observed, labels):
    import torch

    def delta(site):
        before, after = (row[site][SELF].float() for row in (clean, observed))
        l2 = float((after - before).norm())
        norm = float(before.norm())
        return {"l2_change": l2, "relative_l2_change": l2 / norm if norm else None}

    before, after = (row["logits"][1, 1].float() for row in (clean, observed))
    own, other = labels
    clean_id, reached_id = int(before.argmax()), int(after.argmax())
    return {
        "projection_input": {name: delta(f"{name}_input") for name in CHANNELS},
        "raw_projection": {name: delta(f"{name}_raw") for name in CHANNELS},
        "reached_projection": {name: delta(f"{name}_reached") for name in CHANNELS},
        "max_full_head_logit_change": float((after - before).abs().max()),
        "correct_label_log_probability_change": float(
            torch.log_softmax(after, dim=0)[own] - torch.log_softmax(before, dim=0)[own]),
        "answer_margin_change": float((after[own] - after[other]) - (before[own] - before[other])),
        "greedy_token_changed": clean_id != reached_id,
        "clean_greedy_correct": clean_id == own,
        "reached_greedy_correct": reached_id == own,
    }


def evaluate(capture, ids, labels, root_key, guard, np, *, coordinate=COORDINATE):
    """Run one world after the caller supplies source bindings and a resource guard."""
    import torch

    prompts = [ids, ids]
    guard()
    clean = capture.capture(prompts)
    if len(labels) != 2 or len(set(labels)) != 2 or any(type(t) is not int or not 0 <= t < clean["logits"].shape[-1]
                                                       for t in labels):
        raise ValueError("Two distinct in-vocabulary answer tokens are required")
    base = clean["carry"][SELF]
    changes, receipt = plan(base.float().numpy(), coordinate, root_key, np)
    guard()
    zero = capture.capture(prompts, edit=(base, base))
    check_control(clean, zero, identity=True)
    out = receipt | {"zero_hook_closed": True, "prefills": 2, "effects": {}}
    if not changes:
        return out

    for name, after in changes.items():
        guard()
        edited = torch.as_tensor(after, dtype=torch.bfloat16)
        cfg = {"edit": (base, edited)}
        reached = capture.capture(prompts, **cfg)
        check_control(clean, reached)
        if not same_bf16(reached["carry_reached"][SELF], edited):
            raise RuntimeError("Reached BF16 carry differs from its matched-dose plan")
        out["effects"][name] = summarize(clean, reached, labels)
        if name == "target":
            guard()
            rescued = capture.capture(prompts, **cfg, rescue=(reached["k_raw"][SELF], clean["k_raw"][SELF]))
            check_control(clean, rescued)
            unchanged = set(SITES) - {"k_reached"}
            if (any(not same_bf16(reached[site][SELF], rescued[site][SELF]) for site in unchanged)
                    or not same_bf16(rescued["k_reached"][SELF], clean["k_raw"][SELF])):
                raise RuntimeError("Only the intended pass-2 reader K output may be rescued")
            out["effects"]["target_k_rescue"] = summarize(clean, rescued, labels)
    guard()
    restored = capture.capture(prompts)
    check_control(clean, restored, identity=True)
    effects = out["effects"]
    target = effects["target"]["raw_projection"]["k"]["relative_l2_change"]
    shams = [effects[f"coordinate_sham_{i + 1}"]["raw_projection"]["k"]["relative_l2_change"]
             for i in range(SHAM_COUNT)]
    contrast = None if target is None or any(value is None for value in shams) else {
        "target_minus_coordinate_sham_mean": target - fmean(shams),
        "target_minus_coordinate_sham_max": target - max(shams),
    }
    return out | {"status": "closed", "clean_repeat_closed": True, "prefills": len(changes) + 4,
                  "relative_k_change_contrast": contrast}
