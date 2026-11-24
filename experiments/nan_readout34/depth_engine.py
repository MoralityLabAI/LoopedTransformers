"""One- and two-physical-pass cached decoding using the same weights and head."""
from mechinterp.nanbeige_bpd_gym.chunked_cache import ChunkedTwoLoopCache


class OnePassCache(ChunkedTwoLoopCache):
    def _capture(self, hidden, targets, prefill, capture):
        import torch

        if capture or self.instrument is not None:
            raise ValueError("The single-pass READOUT34 cache only supports native boundaries")
        self.caches = self.caches[:1]
        hidden = self._walk(hidden, 1, targets, prefill)
        length = hidden.shape[1] if prefill else self.physical_length + 1
        if any(self.caches[0].get_seq_length(i) != length for i in range(self.loop.n)):
            raise RuntimeError("Single-pass cache canvas differs across layers")
        boundary = torch.stack([hidden[row, position] for row, position in enumerate(targets)])
        return {"boundaries": [boundary.detach().clone()], "states": {}}


class DepthEngine:
    def __init__(self, loop, head, *, chunk=128):
        self.caches = {
            1: OnePassCache(loop, prefill_chunk=chunk, observe_layers=()),
            2: ChunkedTwoLoopCache(loop, prefill_chunk=chunk, observe_layers=()),
        }
        self.head = head
        self.vocabulary_size = int(head.weight.shape[0])
        self.cache = None
        self.depth = None

    def reset(self, depth):
        if self.cache is not None:
            self.cache.reset()
        self.cache = self.caches[depth]
        self.depth = depth

    def logits(self, capture, expected_length):
        import torch

        if (len(capture["boundaries"]) != self.depth or capture["states"]
                or self.cache.semantic_lengths != [expected_length]
                or self.cache.physical_length != expected_length
                or len(self.cache.caches) != self.depth):
            raise RuntimeError("Decoder phase count or B1 cache history changed")
        value = self.head(capture["boundaries"][-1])
        if value.shape != (1, self.vocabulary_size) or not torch.isfinite(value).all():
            raise RuntimeError("Invalid full-vocabulary B1 readout")
        return value

    def first(self, ids):
        return int(self.logits(self.cache.prefill([ids], capture=False), len(ids)).argmax())

    def next(self, token):
        expected = self.cache.physical_length + 1
        return int(self.logits(self.cache.step([token], capture=False), expected).argmax())

    def close(self):
        for cache in self.caches.values():
            cache.close()

    def preflight(self, ids, maximum_position, guard):
        import torch

        reports = {}
        first_boundaries = {}
        for depth in (1, 2):
            self.reset(depth)
            guard()
            initial = self.cache.prefill([ids], capture=False)
            first_boundaries[depth] = initial["boundaries"][0]
            base = self.logits(initial, len(ids))
            next_token = int(base.argmax())
            cached = self.logits(self.cache.step([next_token], capture=False), len(ids) + 1)
            self.reset(depth)
            replay = self.logits(self.cache.prefill([ids], capture=False), len(ids))
            if not torch.equal(base, replay):
                raise RuntimeError("Repeated same-shape prefill differs")
            self.reset(depth)
            full = self.logits(self.cache.prefill([ids + [next_token]], capture=False), len(ids) + 1)
            incremental_vs_full = (cached - full).abs().max().item()
            same_next_argmax = int(cached.argmax()) == int(full.argmax())
            self.reset(depth)
            guard()
            capacity_ids = (ids * ((maximum_position - 2) // len(ids) + 1))[:maximum_position - 1]
            self.cache.prefill([capacity_ids], capture=False)
            self.next(next_token)
            if self.cache.physical_length != maximum_position:
                raise RuntimeError("Verified position did not reach the scheduled maximum")
            reports[str(depth)] = {
                "max_position": maximum_position,
                "capacity_cache_bytes": self.cache.cache_bytes(),
                "repeat_same_shape_exact": True,
                "diagnostic_incremental_vs_full_max_abs_logit": incremental_vs_full,
                "diagnostic_incremental_vs_full_argmax_equal": same_next_argmax,
            }
            guard()
            self.reset(depth)
        if not torch.equal(first_boundaries[1], first_boundaries[2]):
            raise RuntimeError("First-phase boundary differs between depth arms at the same prompt")
        return reports
