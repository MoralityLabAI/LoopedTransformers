"""Exact sealed finite cells from earlier published LoopScope-02 native receipts."""
import unittest
try:
    from .telemetry_replay import pair_stats, trace_word, analyze, SOURCE_BUNDLE_SHA256, MODEL_SHA256
except ImportError:
    from telemetry_replay import pair_stats, trace_word, analyze, SOURCE_BUNDLE_SHA256, MODEL_SHA256

class TestTelemetry(unittest.TestCase):
    def test_s4_eight_step_phase_advantage_witnesses(self):
        words=[0b11111000,0b01001111,0b01011011,0b01110011]
        x=pair_stats(words,[1,2,3,4],8)
        self.assertEqual(x['count'],0)
        self.assertEqual(x['cyclic'],6)
        self.assertEqual(x['ordered'],6)
    def test_native_source_requires_exact_shape_and_pin(self):
        with self.assertRaises(ValueError):
            analyze({'items':[],'bundle_sha256':SOURCE_BUNDLE_SHA256},
                    {'checkpoint_sha256':MODEL_SHA256,'bundle_sha256':SOURCE_BUNDLE_SHA256,
                     'steps':16,'items_target_correct_by_step':[[1]]})
    def test_quotient_nesting_on_numeric_fixture(self):
        x=pair_stats([255,0,248,254,192,239,129,16,36],list(range(1,10)),8)
        self.assertTrue(x['ordered']>=x['cyclic']>=x['count'])
        self.assertEqual(x['denominator'] if 'denominator' in x else x['eligible'],36)
if __name__=='__main__': unittest.main()
