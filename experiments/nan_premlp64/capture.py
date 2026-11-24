"""Add read-only residual and attention captures to the gated native B2 path."""
from contextlib import contextmanager

from experiments.nan_dose51.native import RouteCapture, WriteCapture
from experiments.nan_gated46.gated import BeforeSecond


SITES = ("attention_input", "attention_write", "pre_mlp")
PAIRED_PHASES = {(phase, row) for phase in (1, 2) for row in (0, 1)}


def same_bf16(left, right):
    import torch

    return (left.shape == right.shape and left.dtype == right.dtype == torch.bfloat16
            and torch.equal(left.view(torch.int16), right.view(torch.int16)))


class CarryInstrument(WriteCapture):
    def __init__(self, loop, layer):
        if layer != loop.n - 1:
            raise ValueError("The existing phase-boundary gate requires the last physical layer")
        modules = [loop.layers[layer]]
        if loop.stage_layer is not None and layer >= loop.n_gpu_layers:
            modules.append(loop.stage_layer)
        if any(module.enable_hyper_connection for module in modules):
            raise ValueError("The registered carry capture requires additive residuals")
        super().__init__(loop, layer)
        self.carry = {}
        for module in modules:
            self.handles.extend([
                module.input_layernorm.register_forward_pre_hook(self.attention_input),
                module.self_attn.o_proj.register_forward_hook(self.attention_write),
                module.post_attention_layernorm.register_forward_pre_hook(self.pre_mlp),
            ])

    @contextmanager
    def context(self, modes):
        # The parent sets enabled before entering any module and clears it on errors.
        with super().context(modes):
            self.carry = {site: {} for site in SITES}
            yield
            if any(set(self.carry[site]) != PAIRED_PHASES for site in SITES):
                raise RuntimeError("A native carry capture lacks a phase or last-valid row")

    def close(self):
        super().close()
        self.carry.clear()

    def attention_input(self, _module, args):
        self.store_carry("attention_input", args[0])

    def attention_write(self, _module, _args, result):
        self.store_carry("attention_write", result)

    def pre_mlp(self, _module, args):
        self.store_carry("pre_mlp", args[0])

    def store_carry(self, site, value):
        if self.enabled:
            self.observations(value, self.carry[site])


def check_additions(packet):
    """Recompute each BF16 residual sum from the recorded native operands."""
    import torch

    for key in sorted(PAIRED_PHASES):
        incoming, attention, pre_mlp = (packet[site][key] for site in SITES)
        write, output = (packet[site][key] for site in ("native_writes", "residual"))
        inputs = packet["inputs"][key]
        if (len(incoming.shape) != 1 or any(value.shape != incoming.shape
                                          for value in (attention, pre_mlp, write, output))
                or any(value.dtype != torch.bfloat16 or not torch.isfinite(value).all()
                       for value in (incoming, attention, pre_mlp, write, output, inputs))
                or len(inputs.shape) != 1):
            raise ValueError("Invalid physical BF16 carry vector")
        if not same_bf16(incoming + attention, pre_mlp):
            raise ValueError("Direct pre-MLP residual does not close the native attention add")
        if not same_bf16(pre_mlp + write, output):
            raise ValueError("Layer output does not close the native MLP add")


class CarryCapture(RouteCapture):
    """Preserve the registered phase-1 gate, then return its native operands."""

    def __init__(self, loop, head, *, layer, chunk=128, cache_factory=None):
        self.cache = BeforeSecond(loop, prefill_chunk=chunk, observe_layers=(),
                                  cache_factory=cache_factory,
                                  instrument_factory=lambda model: CarryInstrument(model, layer))
        self.instrument = self.cache.instrument
        self.head = head

    def capture(self, prompts, patch=None, **gate):
        result = super().capture(prompts, patch, **gate)
        result.update({name: dict(values) for name, values in self.instrument.carry.items()})
        check_additions(result)
        return result
