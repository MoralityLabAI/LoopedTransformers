"""Model-lane adapter for one- and two-pass physical B2 caches and streamed head."""
from experiments.nan_readout34.depth_engine import DepthEngine


class BatchedDepthEngine(DepthEngine):
    def _read(self, captured, lengths, canvas):
        import torch

        if (len(captured["boundaries"]) != self.depth or captured["states"]
                or self.cache.semantic_lengths != lengths or self.cache.physical_length != canvas
                or len(self.cache.caches) != self.depth):
            raise RuntimeError("Physical pass count, semantic history or padded canvas changed")
        logits = self.head(captured["boundaries"][-1])
        if logits.shape != (len(lengths), self.vocabulary_size) or not torch.isfinite(logits).all():
            raise RuntimeError("Incomplete or nonfinite full-vocabulary batch")
        return logits

    def prefill_logits(self, inputs):
        if not 1 <= len(inputs) <= 2:
            raise ValueError("One or two full prompts are required")
        lengths = list(map(len, inputs))
        captured = self.cache.prefill(inputs, capture=False)
        return self._read(captured, lengths, max(lengths))

    def advance_logits(self, keep, tokens):
        old = self.cache.semantic_lengths
        if (not keep or keep != sorted(set(keep)) or keep[0] < 0 or keep[-1] >= len(old)
                or len(tokens) != len(keep)):
            raise ValueError("Invalid survivor-to-physical-row binding")
        canvas = self.cache.physical_length + 1
        if keep != list(range(len(old))):
            self.cache.select_rows(keep)
        captured = self.cache.step(tokens, capture=False)
        return self._read(captured, [old[i] + 1 for i in keep], canvas)

    def prefill(self, inputs):
        return self.prefill_logits(inputs).argmax(dim=-1).tolist()

    def advance(self, keep, tokens):
        return self.advance_logits(keep, tokens).argmax(dim=-1).tolist()
