"""Coverage and endpoint guards for the official/native presentation."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from mechinterp.nanbeige_bpd_gym.prime20_results import validate,render


class OfficialPresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=(Path(__file__).parent/'fixtures/prime20_76020.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest()=='3caaabb29aeae20a4b7a27940329030c0302bd2b6be41f8644c6bd88bd3c68fa'
        fixture=json.loads(raw);cls.data=fixture['prime20'];cls.native=fixture['native_comparator_projection']

    def test_pending_official_grades_remain_unknown(self):
        text=render(self.data,self.native,'immutable','source-sha')
        self.assertIn('| bf16 | focused | 0 / 32 | unknown | 32 | unknown / 0 | 0.00000 to 1.00000 | 32 / 32 |',text)

    def test_full_coverage_cannot_disagree_with_matched_native(self):
        data=copy.deepcopy(self.data);data['backends']['nf4']['groups'][0]['correct']-=1
        with self.assertRaises(AssertionError):validate(data,self.native)

    def test_copy_control_cannot_disappear(self):
        data=copy.deepcopy(self.data);data['backends']['nf4']['groups'].pop(1)
        with self.assertRaises(AssertionError):validate(data,self.native)

    def test_completed_coverage_cannot_hide_an_orphan(self):
        data=copy.deepcopy(self.data);data['backends']['nf4']['orphan_or_unbound_neural_replies']=1
        with self.assertRaises(AssertionError):validate(data,self.native)

    def test_partial_official_coverage_keeps_bounds_and_native_separate(self):
        data=copy.deepcopy(self.data);r=data['backends']['nf4']
        r.update(execution_status='terminal_with_missing_coverage',successful_official_episodes=60,
            missing_graded_contexts=36,failed_official_episodes=36,orphan_or_unbound_neural_replies=36)
        for group,correct,both in zip(r['groups'],(18,20,1),(8,10,0)):
            group.update(graded_contexts=20,missing_graded_contexts=12,correct=correct,format_valid=20,
                graded_source_roots=10,complete_world_pairs=10,both_world_roots_correct=both,
                whole_registered_correctness_lower_bound=correct/32,whole_registered_correctness_upper_bound=(correct+12)/32)
        validate(data,self.native)
        text=render(data,self.native,'software-only-partial-fixture','source-sha')
        self.assertIn('| nf4 | focused | 20 / 32 | 18 | 12 | 8 / 10 | 0.56250 to 0.93750 | 29 / 32 |',text)


if __name__=='__main__':unittest.main()
