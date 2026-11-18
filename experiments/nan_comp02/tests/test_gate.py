import copy
import unittest
from experiments.nan_comp02.grade import gate,correct

class GateTests(unittest.TestCase):
    def test_source_blind_ceiling_cannot_pass_and_constrained_scores_do_not_rescue(self):
        cfg={'minimum_free_future_correct':12,'minimum_both_correct_pairs':6,'minimum_source_use_difference':'0.25'}
        pairs=[{'split':'development','recipient':f'r{i}','donor':f'd{i}'} for i in range(8)]
        grades={cid:{'worked_native2':{'future_correct':int(cid[0]=='r'),'constrained_future_correct':1},'erased_native2':{'future_correct':int(cid[0]=='r')}} for p in pairs for cid in [p['recipient'],p['donor']]}
        self.assertFalse(gate(grades,pairs,cfg)['passed']);self.assertEqual(gate(grades,pairs,cfg)['both_correct_pairs'],0)
        for r in grades.values():r['worked_native2']['future_correct']=1
        self.assertTrue(gate(grades,pairs,cfg)['passed'])
        for r in grades.values():r['worked_native2']['future_correct']=0
        self.assertFalse(gate(grades,pairs,cfg)['passed'])
    def test_verified_future_requires_a_valid_single_operator(self):
        c={'inputs':[[1,1,0,0,0]],'outputs':[[1,-1,0,0,0]],'query':[1,1,1,0,0],'future':[1,-1,1,0,0]}
        self.assertEqual(correct(c,'N'),1);self.assertEqual(correct(c,'I'),0);self.assertEqual(correct(c,None),0);self.assertEqual(correct(c,'IN'),0)

if __name__=='__main__':unittest.main()
