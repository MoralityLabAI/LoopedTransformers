import unittest
import numpy as np
from mechinterp.nanbeige_bpd_gym.context_probes import root_macro,displacements
from mechinterp.nanbeige_bpd_gym.context_probe_report import physical_pair


class ContextProbeUnitTests(unittest.TestCase):
    def test_physical_loop_difference_retains_magnitude_despite_high_cosine(self):
        meta=[dict(id='one',root='one',split='development',context_budget=1024,position='early',task='lookup',copy_control=False,loop=t) for t in (1,2)]
        bf={21:{'residual_after_layer':np.array([[1.,0.],[2.,1.]])}}
        nf={21:{'residual_after_layer':np.array([[1.,0.],[3.,1.]])}}
        result=physical_pair(bf,nf,meta)['21']
        self.assertAlmostEqual(result['loop_displacements'][0]['delta_relative_l2'],1/np.sqrt(2))
        self.assertGreater(result['states'][1]['physical_cosine'],.98)

    def test_correlated_variants_do_not_reweight_source_roots(self):
        meta=[{'root':r} for r in ('a','b','b','b')]
        self.assertEqual(root_macro(np.array([0.,1.,1.,1.]),list(range(4)),meta),.5)

    def test_world_and_loop_pairs_are_separate_with_shared_source_count(self):
        meta=[]
        for world in ('a','b'):
            for loop in (1,2):
                meta.append(dict(id=world,root='one',split='development',context_budget=1024,position='early',
                    task='lookup',copy_control=False,world=world,loop=loop))
        h=np.array([[0.],[1.],[3.],[4.]])
        loop=displacements(h,h,meta,'loop');world=displacements(h,h,meta,'world')
        self.assertEqual(len(loop),1);self.assertEqual(loop[0]['vectors'],2)
        self.assertEqual(loop[0]['source_root_groups'],1);self.assertEqual(loop[0]['root_macro_physical_delta_l2'],1.)
        self.assertEqual(len(world),2)
        self.assertTrue(all(g['source_root_groups']==1 and g['root_macro_physical_delta_l2']==3. for g in world))


if __name__=='__main__':unittest.main()
