import unittest
import torch
from reader import Reader,packet,fit,costs,ARMS
from run08 import Meter,Exhausted,fit_prior,FRESH,read,HOME
from search08 import reprice
class Tests(unittest.TestCase):
 def test_graph_and_matched_arms(self):
    torch.set_num_threads(1)
    g={'nodes':[{'k':'fvar','n':'x'},{'k':'const','n':'Nat'},{'k':'app'}], 'edges':[[0,2,'arg0'],[1,0,'typeof']], 'obligations':[{'target':2}]}
    l=[{'name':'foo','root':0,'nodes':[{'k':'const','n':'foo'}],'edges':[]}]
    a=[(1.,'apply_lemma','foo','apply foo'),(2.,'close','','simp_all')]
    p=packet(g,l,a,0,[1.,2.]);hs=[]
    for arm in ARMS:
      m,s=fit([(p,0),(p,1)],arm,17,steps=3)
      self.assertLess(sum(x.numel() for x in m.parameters()),146000)
      self.assertTrue(all(c>=0 for c in costs(m,p)))
      if arm!='linear':hs.append((s['initial_sha256'],s['sampler_sha256']))
    self.assertEqual(len(set(hs)),1)
 def test_budget_all_rpc(self):
    class Mock:
      def request(self,**k):return {'request_hb_raw':5,'request_ms':1,'ok':True}
    m=Meter(Mock(),steps=1,rpcs=4,hb=20)
    m.call('premises');m.call('graph');m.call('step');m.call('cert')
    self.assertEqual(m.report()['hb_raw'],20);self.assertEqual(m.report()['steps'],1)
    with self.assertRaises(Exhausted):m.call('step')
 def test_prior_uses_exact_event_ids(self):
    r={'role':'train','on_path':[1],'events':[{'kind':'close','tactic':'rfl','rank':0,'depth':0,'hb_raw':10},{'kind':'close','tactic':'simp','rank':0,'depth':0,'hb_raw':15}]}
    p=fit_prior([r]);self.assertGreater(p['weight']['0|simp'],p['weight']['0|rfl'])
    with self.assertRaises(ValueError):fit_prior([{**r,'role':'evaluation'}])
 def test_fresh_roster(self):
    old=set(read(HOME/'spent.json')['files']);fresh=[f'src/Init/Data/{f}/{n}.lean' for _,fs in FRESH.items() for f,ns in fs.items() for n in ns]
    self.assertEqual(len(fresh),len(set(fresh)));self.assertFalse(old&set(fresh));self.assertTrue(all('/SInt/' not in n for n in fresh))
if __name__=='__main__':unittest.main()
