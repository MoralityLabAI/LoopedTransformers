"""MC-03 data, source and training-contract tests; no experiment tuning."""
import unittest
from pathlib import Path
import tempfile
import numpy as np
import torch
from research_gym.scripts.mc03_trained_pilot import (
    SEEDS, RECIPE, graph_split, render, sampled_batch, heldout_panel,
    lr_at, source_identity, trace_metrics, run,
)
from research_gym.scripts.loopscope_toy import LoopedChase
from research_gym.scripts import loopscope_semantic_echo as native


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_recipe(self):
        self.assertEqual(SEEDS,(201,202))
        self.assertEqual((RECIPE['steps'],RECIPE['loops'],RECIPE['lam']),(3000,4,1.0))
        self.assertEqual(RECIPE['wall_limit_seconds'],240)

    def test_graph_disjoint_complete_split(self):
        split=graph_split(); sets={k:set(v) for k,v in split.items()}
        self.assertEqual([len(sets[k]) for k in ('train','calibration','test')],[480,120,120])
        self.assertFalse(sets['train']&sets['test'])
        self.assertFalse(sets['train']&sets['calibration'])
        self.assertFalse(sets['test']&sets['calibration'])
        self.assertEqual(len(set.union(*sets.values())),720)
        self.assertEqual(split,graph_split())

    def test_panel_every_start_fixed_before_model(self):
        panel=heldout_panel(graph_split()['test'])
        g,s,t,y,ix=panel
        self.assertEqual(t.shape,(720,13))
        self.assertEqual(y.shape,(720,16))
        self.assertEqual(len({(tuple(a),int(b)) for a,b in zip(g.tolist(),s)}),720)
        self.assertEqual(len(set(ix.tolist())),64)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(panel,heldout_panel(graph_split()['test']))))

    def test_token_layout_and_oracle(self):
        gs=torch.tensor([[1,2,0],[2,0,1]])
        st=torch.tensor([0,1]); order=torch.tensor([[2,0,1],[1,2,0]])
        tok,y=render(gs,st,order,4)
        self.assertEqual(tok.tolist(),[[2,0,0,1,1,2,0],[1,0,2,1,0,2,1]])
        self.assertEqual(y.tolist(),[[1,2,0,1],[0,2,1,0]])

    def test_edge_order_preserves_targets(self):
        gs=torch.tensor([[1,2,0]]);st=torch.tensor([0])
        a,y=render(gs,st,torch.tensor([[0,1,2]]),4)
        b,z=render(gs,st,torch.tensor([[2,0,1]]),4)
        self.assertTrue(torch.equal(y,z));self.assertFalse(torch.equal(a,b))

    def test_reject_bad_shapes_dtypes_graphs(self):
        gs=torch.tensor([[1,2,0]]);st=torch.tensor([0]);order=torch.tensor([[0,1,2]])
        for args in ((gs.float(),st,order,4),(gs,torch.tensor([-1]),order,4),
                     (gs,st,torch.tensor([[0,0,2]]),4),(gs,st,order,0)):
            with self.assertRaises(ValueError):render(*args)

    def test_sampled_graphs_stay_training_only(self):
        split=graph_split();gs=torch.tensor(split['train']);rng=torch.Generator().manual_seed(1)
        tokens,_=sampled_batch(gs,64,rng,4)
        for row in tokens.tolist():
            g=[0]*6
            for src,dst in zip(row[:12:2],row[1:12:2]):g[src]=dst
            self.assertIn(tuple(g),split['train'])
            self.assertNotIn(tuple(g),split['test'])

    def test_batch_seed_replay(self):
        gs=torch.tensor(graph_split()['train'])
        a=sampled_batch(gs,16,torch.Generator().manual_seed(2),4)
        b=sampled_batch(gs,16,torch.Generator().manual_seed(2),4)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))

    def test_lr_frozen_warmup_and_end(self):
        self.assertAlmostEqual(lr_at(0),1e-5)
        self.assertAlmostEqual(lr_at(99),1e-3)
        self.assertAlmostEqual(lr_at(3000),5e-5)

    def test_full_source_identity(self):
        self.assertEqual(len(source_identity()),7)

    def test_native_shapes_backward_and_parity(self):
        torch.manual_seed(7001)
        model=LoopedChase(6,48,1,4)
        tok,target=sampled_batch(torch.tensor(graph_split()['train']),4,torch.Generator().manual_seed(2),4)
        states=model(tok,4); logits=model.readout(states)
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,6),target.T.reshape(-1))*4
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        with torch.inference_mode():
            _,full=native.native_trace(model,tok,4)
            self.assertTrue(native.verify_native_trace(model,tok,full,rtol=2e-5,atol=2e-6)['passed'])

    def test_relation_checks_intermediate_wrong_same_endpoint(self):
        g,s,_,target,_=heldout_panel(graph_split()['test'])
        truth=target[:2].T.numpy();pred=truth.copy()
        pred[0,0]=(pred[0,0]+1)%6
        result=trace_metrics(pred,truth,g[:2].tolist(),s[:2].tolist())
        self.assertEqual(result['full_trace_16_accept_count'],1)
        self.assertEqual(result['accuracy_by_visit'][-1],1)
        self.assertEqual(result['first_wrong_visit'],[1,None])

    def test_existing_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'prior').write_text('retain')
            with self.assertRaises(ValueError):run(p)
            self.assertEqual((p/'prior').read_text(),'retain')


if __name__=='__main__':unittest.main()
