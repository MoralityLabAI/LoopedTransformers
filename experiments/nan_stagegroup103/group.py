"""Share phase-two stage copies while preserving independent B2 calls and caches."""

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
from types import SimpleNamespace

from experiments.nan_passreuse96.capture import SharedPhaseCache


MAX_BRANCHES = 2


@dataclass(frozen=True)
class Prepared:
    owner: object
    tokens: tuple
    raw: object
    attention: object
    targets: tuple
    lengths: tuple
    inputs: dict
    residual: dict


@dataclass(frozen=True)
class Request:
    prepared: Prepared
    prompts: tuple
    patches: tuple = (None, None)


class GroupedPassCapture:
    def __init__(self, loop, head, *, layer, chunk=128):
        self.cache = SharedPhaseCache(loop, layer, chunk)
        self.head = head
        self.loop = loop
        self.owner = object()

    def prime(self, prompts):
        import torch

        previous = self.loop.phase, self.loop.active_layer
        try:
            with torch.inference_mode():
                self.cache.prime(prompts)
                first = self.cache.first
                return Prepared(self.owner, self.cache.tokens, first["raw"], self.cache.attention,
                                first["targets"], tuple(self.cache.semantic_lengths), first["inputs"], first["residual"])
        finally:
            self.cache.reset()
            self.loop.phase, self.loop.active_layer = previous

    def initialize(self, request, torch):
        src = request.prepared
        if (src.owner is not self.owner or tuple(tuple(row) for row in request.prompts) != src.tokens
                or len(request.patches) != 2 or tuple(src.raw.shape[:2]) != (2, len(src.tokens[0]))):
            raise ValueError("Branch must bind to the exact primed B2 prompts on this capture")
        prepared_edits = []
        for row, patch in enumerate(request.patches):
            if patch is None:
                continue
            token = src.raw[row, src.targets[row]]
            if not patch.eligible or token.dtype != torch.bfloat16:
                raise ValueError("The grouped phase-one edit did not qualify")
            before = torch.as_tensor(patch.before, dtype=token.dtype, device=token.device)
            after = torch.as_tensor(patch.after, dtype=token.dtype, device=token.device)
            if not torch.equal(before, token) or not torch.isfinite(after).all() or torch.equal(before, after):
                raise ValueError("The grouped phase-one edit is stale or degenerate")
            if after.shape != token.shape:
                raise ValueError("The grouped phase-one edit has the wrong shape")
            prepared_edits.append((row, after))
        canvas = src.raw.clone()
        for row, after in prepared_edits:
            canvas[row, src.targets[row]].copy_(after)
        values = self.loop.base.norm(canvas)
        current = torch.stack([values[r, p] for r, p in enumerate(src.targets)]).detach().clone()
        return SimpleNamespace(src=src, value=values, cache=self.cache.factory(), current=current,
                               applied=len(prepared_edits), inputs={}, residual={})

    def layer(self, branch, module, index, observed, torch):
        h, cache, src = branch.value, branch.cache, branch.src
        instrument = self.cache.instrument
        context = instrument.capture_phase(2) if observed else nullcontext()
        with context:
            for start in range(0, h.shape[1], self.cache.chunk):
                stop = min(start + self.cache.chunk, h.shape[1])
                physical = torch.arange(start, stop, device=self.loop.device)
                positions = physical.unsqueeze(0).expand(2, -1)
                if observed:
                    instrument.set_frame(SimpleNamespace(
                        phase=2, prefill=True, valid_lengths=[max(0, min(stop, n)-start) for n in src.lengths],
                        capture_offsets=[p-start if start <= p < stop else None for p in src.targets]))
                mask = self.loop.base._update_causal_mask(src.attention, h[:, start:stop], physical, cache, False)
                values = module(h[:, start:stop], attention_mask=mask, position_ids=positions,
                                past_key_value=cache, output_attentions=False, use_cache=True,
                                cache_position=physical, loop_idx=0)[0]
                for cached in (cache.key_cache[index], cache.value_cache[index]):
                    if cached.untyped_storage().data_ptr() == h.untyped_storage().data_ptr():
                        raise RuntimeError("Grouped K/V aliases the overwritten residual canvas")
                h[:, start:stop].copy_(values)
        if observed:
            branch.inputs = dict(instrument.inputs)
            branch.residual = dict(instrument.residual)

    def finish(self, branch, torch):
        src = branch.src
        n = branch.value.shape[1]
        if any(branch.cache.get_seq_length(i) != n for i in range(self.loop.n)):
            raise RuntimeError("A grouped phase-two cache did not close every physical layer")
        normalized = self.loop.base.norm(branch.value)
        later = torch.stack([normalized[r, p] for r, p in enumerate(src.targets)]).detach().clone()
        logits = torch.stack([self.head(row).detach().cpu() for row in (branch.current, later)])
        if logits.shape != (2, 2, len(self.head.weight)) or not torch.isfinite(logits).all():
            raise RuntimeError("Grouped pass did not close the exact two-row full-vocabulary heads")
        return {"logits": logits, "inputs": {k: v.clone() for k, v in src.inputs.items()} | branch.inputs,
                "residual": {k: v.clone() for k, v in src.residual.items()} | branch.residual,
                "physical_length": n, "semantic_lengths": list(src.lengths), "applied": branch.applied}

    def run(self, requests, guard):
        import torch

        if not 1 <= len(requests) <= MAX_BRANCHES:
            raise ValueError("A group must contain one or two B2 branches")
        previous = self.loop.phase, self.loop.active_layer
        branches = []
        try:
            with torch.inference_mode():
                for request in requests:
                    guard()
                    branches.append(self.initialize(request, torch))
                if len({id(b.cache) for b in branches}) != len(branches):
                    raise ValueError("Grouped branches require independent phase-two caches")
                self.loop.phase = 2
                for i in range(self.loop.n):
                    guard()
                    self.loop.active_layer = i
                    layer = self.loop.layers[i] if i < self.loop.n_gpu_layers else self.loop._load_stage(self.loop.layers[i])
                    if layer.self_attn.layer_idx != i:
                        raise RuntimeError("Grouped staging changed the physical cache layer")
                    for branch in branches:
                        self.layer(branch, layer, i, i == self.loop.n - 1, torch)
                return [self.finish(branch, torch) for branch in branches]
        finally:
            for branch in branches:
                branch.cache = branch.value = branch.current = None
            branches.clear()
            self.loop.phase, self.loop.active_layer = previous

    def stream(self, requests, guard):
        pending = iter(requests)
        while group := tuple(islice(pending, MAX_BRANCHES)):
            yield from self.run(group, guard)

    def close(self):
        self.cache.close()
