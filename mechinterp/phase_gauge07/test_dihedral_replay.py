"""COE-07R exact finite tests, with optional pinned native-source replay."""
from __future__ import annotations
from pathlib import Path
import unittest
try:
    from .dihedral_replay import (analyze, bracelet, descriptor, necklace_counts,
                                  pair_and_decoder_ledger, reverse_bits)
    from .score import rotate
    from .telemetry_replay import BUNDLE, TRACE, load_and_verify
except ImportError:
    from dihedral_replay import (analyze, bracelet, descriptor, necklace_counts,
                                 pair_and_decoder_ledger, reverse_bits)
    from score import rotate
    from telemetry_replay import BUNDLE, TRACE, load_and_verify


class TestDihedralGauge(unittest.TestCase):
    def test_reversal_is_involutive(self):
        for T in (4, 8, 12):
            for word in range(1 << T):
                self.assertEqual(reverse_bits(reverse_bits(word, T), T), word)

    def test_orbit_invariance(self):
        for T in (4,8,12):
            for word in range(1 << T):
                b = bracelet(word, T)
                for shift in range(T):
                    self.assertEqual(bracelet(rotate(word, shift, T), T), b)
                    self.assertEqual(bracelet(rotate(reverse_bits(word,T),shift,T),T),b)

    def test_orientation_strict_pair(self):
        word = 0b1011
        other = reverse_bits(word,8)
        self.assertNotEqual(descriptor(word,8,'cyclic'),descriptor(other,8,'cyclic'))
        self.assertEqual(descriptor(word,8,'dihedral'),descriptor(other,8,'dihedral'))
        self.assertEqual(descriptor(word,8,'count'),descriptor(other,8,'count'))
        s=pair_and_decoder_ledger([word,other],[1,2],8)
        self.assertEqual((s['phase_sensitive'],s['time_orientation_sensitive'],s['dihedral_beyond_count']), (0,1,0))

    def test_weight_four_burnside(self):
        self.assertEqual(necklace_counts(8,4),{
            'width':8,'weight':4,'necklaces':10,'reversal_fixed_necklaces':6,'bracelets':8})
        self.assertEqual(necklace_counts(8,3)['necklaces'],7)
        self.assertEqual(necklace_counts(8,3)['bracelets'],5)

    def test_unequal_digit_exclusion_and_bayes_upper_bound(self):
        words=[0,0,0b11,0b11,0b100]
        labels=[1,1,1,2,2]
        result=pair_and_decoder_ledger(words,labels,4)
        self.assertEqual(result['eligible'],6)
        self.assertEqual(result['ordered']['distinguished'],5)
        self.assertEqual(result['ordered']['in_sample_oracle_correct'],4)
        for a,b in [('ordered','cyclic'),('cyclic','dihedral'),('dihedral','count')]:
            self.assertGreaterEqual(result[a]['distinguished'],result[b]['distinguished'])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): bracelet(256,8)
        with self.assertRaises(ValueError): necklace_counts(17,4)
        with self.assertRaises(ValueError): pair_and_decoder_ledger([1],[1,2],4)

    @unittest.skipUnless(BUNDLE.is_file() and TRACE.is_file(),
                         'native pinned LoopScope-02 JSON is not mounted locally')
    def test_pinned_native_exact_dihedral_ledger(self):
        old=load_and_verify()
        import json
        data=analyze(json.loads(BUNDLE.read_text()),json.loads(TRACE.read_text()))
        self.assertEqual(data['model_checkpoint_sha256'],old['checkpoint_sha256'])
        for T,expect in [('4',(1278,1244,1244,1244,23,18,18,18)),
                         ('8',(1414,1389,1389,1369,30,24,24,19)),
                         ('16',(1447,1428,1428,1422,31,26,26,24))]:
            r=data['windows'][T]['S4_stalled']['all_64']
            self.assertEqual(tuple(r[k]['distinguished'] for k in ('ordered','cyclic','dihedral','count')) +
                             tuple(r[k]['in_sample_oracle_correct'] for k in ('ordered','cyclic','dihedral','count')),expect)
            self.assertEqual(r['eligible'],1795)

if __name__ == '__main__': unittest.main()
