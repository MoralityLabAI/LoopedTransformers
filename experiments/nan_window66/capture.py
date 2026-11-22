"""Observe a contiguous carry window while the registered late gate still owns the edit."""
from contextlib import contextmanager

from experiments.nan_gated46.gated import BeforeSecond
from experiments.nan_premlp64.capture import CarryCapture, CarryInstrument, PAIRED_PHASES, check_additions, same_bf16


EARLY_SITES = ("attention_input", "attention_write", "pre_mlp", "inputs", "native_writes", "residual")
FOCUS = tuple(range(11, 18)) + (21,)


class WindowInstrument(CarryInstrument):
    def __init__(self, loop, layers):
        if not layers or tuple(sorted(set(layers))) != layers or layers[0] < 0 or layers[-1] != loop.n - 1:
            raise ValueError("Capture window needs unique ordered layers and the final layer")
        self.earlier = layers[:-1]
        modules = [loop.layers[layer] for layer in self.earlier if layer < loop.n_gpu_layers]
        if loop.stage_layer is not None and any(layer >= loop.n_gpu_layers for layer in self.earlier):
            modules.append(loop.stage_layer)
        if any(module.enable_hyper_connection for module in modules):
            raise ValueError("Earlier carry capture requires additive residuals")
        super().__init__(loop, layers[-1])
        self.early = {}
        for module in modules:
            self.handles.extend([
                module.input_layernorm.register_forward_pre_hook(self.early_input),
                module.self_attn.o_proj.register_forward_hook(self.early_attention),
                module.post_attention_layernorm.register_forward_pre_hook(self.early_pre_mlp),
                module.mlp.down_proj.register_forward_hook(self.early_projection),
                module.register_forward_hook(self.early_residual),
            ])

    @contextmanager
    def context(self, modes):
        with super().context(modes):
            self.early = {layer: {site: {} for site in EARLY_SITES} for layer in self.earlier}
            yield
            if any(set(values) != PAIRED_PHASES for sites in self.early.values() for values in sites.values()):
                raise RuntimeError("Earlier carry capture lacks a phase or semantic row")

    def close(self):
        super().close()
        self.early.clear()

    def store_earlier(self, site, value):
        if not self.enabled or self.loop.active_layer not in self.earlier:
            return
        frame = self.frame
        if (frame is None or not frame.prefill or frame.phase not in (1, 2)
                or len(frame.capture_offsets) != 2 or value.ndim != 3 or value.shape[0] != 2
                or value.shape[1] != frame.stop - frame.start):
            raise RuntimeError("Earlier carry capture requires an active B2 chunk")
        dest = self.early[self.loop.active_layer][site]
        for row, offset in enumerate(frame.capture_offsets):
            if offset is None:
                continue
            key = frame.phase, row
            if key in dest or offset != frame.valid_lengths[row] - 1 or not 0 <= offset < value.shape[1]:
                raise RuntimeError("Earlier state does not bind one last semantic token")
            dest[key] = value[row, offset].detach().cpu().clone()

    def early_input(self, _module, args):
        self.store_earlier("attention_input", args[0])

    def early_attention(self, _module, _args, output):
        self.store_earlier("attention_write", output)

    def early_pre_mlp(self, _module, args):
        self.store_earlier("pre_mlp", args[0])

    def early_projection(self, _module, args, output):
        self.store_earlier("inputs", args[0])
        self.store_earlier("native_writes", output)

    def early_residual(self, _module, _args, output):
        self.store_earlier("residual", output[0])


class WindowCapture(CarryCapture):
    def __init__(self, loop, head, *, layers=FOCUS, chunk=128, cache_factory=None):
        self.cache = BeforeSecond(loop, prefill_chunk=chunk, observe_layers=(), cache_factory=cache_factory,
                                  instrument_factory=lambda model: WindowInstrument(model, layers))
        self.instrument = self.cache.instrument
        self.head = head
        self.layers = layers

    def capture(self, prompts, patch=None, **gate):
        packet = super().capture(prompts, patch, **gate)
        window = {layer: {site: dict(values) for site, values in cells.items()}
                  for layer, cells in self.instrument.early.items()}
        window[self.layers[-1]] = {site: packet[site] for site in EARLY_SITES}
        for layer in self.layers[:-1]:
            check_additions(window[layer])
        for before, after in zip(self.layers, self.layers[1:]):
            if after == before + 1 and any(not same_bf16(window[before]["residual"][key],
                                                        window[after]["attention_input"][key])
                                           for key in PAIRED_PHASES):
                raise ValueError("Adjacent layer captures break native residual continuity")
        return packet | {"window": window}
