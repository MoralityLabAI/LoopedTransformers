"""SAU-OBS01: independent CPU arithmetic witness for observer-footprint controls.

Synthetic floating-point graph only. No GPU compilation or language-model claims.
Run: python observer_footprint.py [--self-test]
"""
from __future__ import annotations

import json
import math
import struct
import sys
import unittest

N = 64


def f32(x: float) -> float:
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bf16_round_nearest_even(x: float) -> float:
    """Round a finite FP32 scalar to BF16 with ties to even."""
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    if (bits & 0x7f800000) == 0x7f800000:
        raise ValueError('finite witness inputs only')
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def forward(prompt_id: int, mode: str) -> dict:
    """Same physical inputs; different arithmetic graph at a hidden boundary.

    capture: split materialization plus observer. barrier: same split without
    observer. matched_capture: same split plus observer with barrier in both
    arms. passive_copy reads the existing float without an extra round.
    """
    if not 0 <= prompt_id < N:
        raise ValueError('prompt outside frozen synthetic set')
    if mode not in {'plain', 'capture', 'barrier', 'matched_capture', 'passive_copy'}:
        raise ValueError('unknown mode')
    prefix = f32(f32(0.376194 * 1.73111) + 0.225431)
    materialize = mode in {'capture', 'barrier', 'matched_capture'}
    site = bf16_round_nearest_even(prefix) if materialize else prefix
    retained = site if mode in {'capture', 'matched_capture', 'passive_copy'} else None
    state = f32(site * 1.41 + 0.0353)
    for _ in range(4):
        state = f32(f32(state * 1.012) - 0.007)
    # Context enters after the identical causal prefix, as in a prefix fingerprint.
    shift = math.sin((prompt_id + 1) * 0.419) * 0.12
    logits = tuple(f32(a * state + b * shift) for a, b in
                   ((1.0, 1.0), (-0.7, 0.3), (0.4, -1.2), (-1.2, -0.2)))
    return {'site': site, 'state': state, 'logits': logits, 'retained': retained}


def audit() -> dict:
    counters = {'capture_changed_logits': 0, 'barrier_changed_logits': 0,
                'matched_capture_changed_logits': 0, 'passive_changed_logits': 0,
                'capture_equals_barrier': 0, 'barrier_equals_matched_capture': 0}
    for i in range(N):
        runs = {mode: forward(i, mode) for mode in
                ('plain', 'capture', 'barrier', 'matched_capture', 'passive_copy')}
        ref = runs['plain']['logits']
        counters['capture_changed_logits'] += runs['capture']['logits'] != ref
        counters['barrier_changed_logits'] += runs['barrier']['logits'] != ref
        counters['matched_capture_changed_logits'] += (
            runs['matched_capture']['logits'] != runs['barrier']['logits'])
        counters['passive_changed_logits'] += runs['passive_copy']['logits'] != ref
        counters['capture_equals_barrier'] += runs['capture']['logits'] == runs['barrier']['logits']
        counters['barrier_equals_matched_capture'] += (
            runs['barrier']['logits'] == runs['matched_capture']['logits'])
    return {'protocol': 'SAU-OBS01', 'kind': 'synthetic_cpu_arithmetic_witness',
            'prompt_pairs': N, 'site_delta': forward(0, 'capture')['site'] -
            forward(0, 'plain')['site'], **counters}


class Tests(unittest.TestCase):
    def test_bf16_exact_roundtrip(self):
        self.assertEqual(bf16_round_nearest_even(1.0), 1.0)
        self.assertEqual(bf16_round_nearest_even(0.0), 0.0)
        self.assertEqual(bf16_round_nearest_even(1.00390625), 1.0)  # halfway, ties-even

    def test_matched_observer_invariance(self):
        for i in range(N):
            self.assertEqual(forward(i, 'capture')['logits'], forward(i, 'barrier')['logits'])
            self.assertEqual(forward(i, 'barrier')['logits'],
                             forward(i, 'matched_capture')['logits'])
            self.assertEqual(forward(i, 'plain')['logits'],
                             forward(i, 'passive_copy')['logits'])

    def test_pure_barrier_explains_footprint(self):
        result = audit()
        self.assertNotEqual(result['site_delta'], 0.0)
        self.assertEqual(result['capture_changed_logits'], N)
        self.assertEqual(result['barrier_changed_logits'], N)
        self.assertEqual(result['matched_capture_changed_logits'], 0)
        self.assertEqual(result['capture_equals_barrier'], N)
        self.assertEqual(result['passive_changed_logits'], 0)


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
    else:
        print(json.dumps(audit(), indent=2, sort_keys=True))
