import unittest
from mechinterp.nanbeige_bpd_gym.audit_prime_attempts import distinct_occurrences,summaries,policy_unfinished


class PrimeAttemptAuditTests(unittest.TestCase):
    def test_duplicate_prompts_require_distinct_neural_occurrences(self):
        assigned,ambiguous=distinct_occurrences(['a','b','a'],['a','a'])
        self.assertEqual(assigned,[0,2]);self.assertEqual(ambiguous,1)
        with self.assertRaises(ValueError):distinct_occurrences(['a'],['a','a'])

    def test_missing_sdk_result_remains_missing_with_fixed_cohort_bounds(self):
        source=[dict(id=str(i),root_id=str(i),suite='pointer',private_label=dict(copy_control=False,hops=1)) for i in range(3)]
        r=summaries(source,{'0':dict(correct=1,format_valid=True),'1':dict(correct=0,format_valid=False)})[0]
        self.assertEqual((r['registered_contexts'],r['graded_contexts'],r['missing_graded_contexts'],r['correct']),(3,2,1,1))
        self.assertEqual(r['whole_registered_correctness_lower_bound'],1/3)
        self.assertEqual(r['whole_registered_correctness_upper_bound'],2/3)

    def test_operational_selection_never_uses_reward_or_format(self):
        expected=[dict(id=str(i)) for i in range(3)]
        def episode(i,ok,reward):
            return dict(ok=ok,task=dict(data=dict(id=str(i))),traces=[dict(ok=ok,is_completed=ok,is_timeout=False,calls=[{}],reward=reward)])
        self.assertEqual([r['id'] for r in policy_unfinished(expected,[episode(0,True,0),episode(1,False,1)])],['1','2'])


if __name__=='__main__':unittest.main()
