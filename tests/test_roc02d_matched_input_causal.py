"""ROC-02D legal source and norm-matched hidden-direction tests."""
import unittest,torch
try:
    from research_gym.analysis.roc02d_matched_input_causal import changed_legal_binding,seeded_vector
    from research_gym.analysis.roc02_delayed_rule_source import rule_table,symbolic_execute,canonical_hash
except ModuleNotFoundError:
    from roc02d_matched_input_causal import changed_legal_binding,seeded_vector
    from roc02_delayed_rule_source import rule_table,symbolic_execute,canonical_hash
class Geometry(unittest.TestCase):
 def test_legal_flip(self):
    e=[{'op':'bind','src':'A','table':rule_table(4)},{'op':'carry'},{'op':'carry'},{'op':'query','src':'A','value':1}]
    case={'family':4,'delay':2,'events':e,'label':symbolic_execute(e)['label']}
    nxt=changed_legal_binding(case);self.assertNotEqual(nxt['label'],case['label']);self.assertEqual(nxt['id'],canonical_hash(nxt));self.assertEqual(case['events'][0]['table'],rule_table(4))
 def test_keyed_unit_vectors(self):
    U=torch.eye(128,dtype=torch.float64)[:,:8]
    for kind in ('observed','dark','isotropic'):
      a=seeded_vector('hash',2,kind,'joint',U);self.assertAlmostEqual(float(torch.linalg.norm(a)),1.,places=12)
      self.assertTrue(torch.equal(a,seeded_vector('hash',2,kind,'joint',U)))
      if kind=='observed':self.assertLess(float(torch.linalg.norm(a[8:])),1e-12)
      if kind=='dark':self.assertLess(float(torch.linalg.norm(a[:8])),1e-12)
if __name__=='__main__':unittest.main()
