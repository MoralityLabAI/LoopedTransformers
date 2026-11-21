"""Ordered source slots preserve duplicate controls without fresh retry attempts."""
from experiments.nan_prime06.wire import prompt_sha, validate_request, stream_chunks, sse_bytes


class OrderedSlots:
    def __init__(self, public):
        self.public = list(public); self.next_index = 0; self.pending = None; self.attempted = set()
        assert len({r['id'] for r in self.public}) == len(self.public)
        # With official max-concurrent1, no shuffle, immediate transport retries
        # cannot be mistaken for the next legitimate source, including repeats.
        assert all(a['prompt'] != b['prompt'] for a, b in zip(self.public, self.public[1:]))

    def claim(self, body, backend, cfg):
        if self.pending is not None or self.next_index >= len(self.public):
            raise ValueError('No unattempted source slot is available; no fresh retry')
        row = self.public[self.next_index]
        key, matched = validate_request(body, {prompt_sha(row['prompt']): row}, backend, cfg)
        assert matched is row and row['id'] not in self.attempted
        self.attempted.add(row['id']); self.pending = row['id']
        return self.next_index, key, row

    def commit(self, source_id):
        if self.pending != source_id:
            raise ValueError('A different source slot cannot be completed')
        self.pending = None; self.next_index += 1
