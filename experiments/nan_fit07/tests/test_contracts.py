import ast
from pathlib import Path
import re
import unittest
import numpy as np
import torch
from experiments.nan_fit07.io import read,rows
from experiments.nan_fit07.sources import materialize
from experiments.nan_fit07.gate import grade
from experiments.nan_fit07.pointwise import masks
from mechinterp.nanbeige_bpd_gym.native_components import load_bpd

ROOT=Path(__file__).resolve().parents[1]


class FreshSourceTests(unittest.TestCase):
    def setUp(self):
        self.cfg=read(ROOT/'config.json');self.public,self.host=materialize(self.cfg)

    def test_all_64_graph_walks_and_fresh_main_prompts(self):
        labels={r['id']:r for r in self.host};count=0
        old={r['prompt'] for r in rows(ROOT.parent/'nan_gym05/fixtures/public.jsonl')}
        for r in self.public:
            if r['copy_control']:
                self.assertTrue(r['prompt'].endswith('\n'+labels[r['id']]['answer']));continue
            self.assertNotIn(r['prompt'],old)
            pieces=r['prompt'].split('Edges: ')[1:];self.assertEqual(len(pieces),4)
            for i,piece in enumerate(pieces):
                edge,start,depth,answer=re.fullmatch(r'([^\n]+)\nStart: ([A-P])\nHops: ([1-3])\nAnswer:([^\n]*)(?:\n\n)?',piece,flags=re.S).groups()
                graph=dict(pair.split('->') for pair in edge.split(', '));node=start
                self.assertEqual(set(graph),set('ABCDEFGHIJKLMNOP'));self.assertEqual(len(set(graph.values())),16)
                for _ in range(int(depth)):node=graph[node]
                self.assertEqual(node,answer.strip() if i<3 else labels[r['id']]['answer']);count+=1
                if i==3:self.assertEqual(answer,'')
        self.assertEqual(count,64);self.assertEqual(len(self.public),32);self.assertEqual(len({r['root'] for r in self.public}),16)

    def test_gate_keeps_all_cases_and_fails_missing_copy_correctness(self):
        values={r['id']:r['answer'] for r in self.host}
        baseline={'records':[{'id':r['id'],'phases':[{'free_token_text':values[r['id']]}]*2} for r in self.public]}
        self.assertTrue(grade(baseline,self.public,self.host,self.cfg)['allow_fits'])
        baseline['records'][1]['phases']=[{'free_token_text':'Unparseable'}]*2
        self.assertFalse(grade(baseline,self.public,self.host,self.cfg)['allow_fits'])
        self.assertEqual(sum(r['records'] for r in grade(baseline,self.public,self.host,self.cfg)['groups']),32)

    def test_worker_does_not_import_host_oracle_or_read_gold_rows(self):
        module=ast.parse((ROOT/'worker.py').read_text())
        imports=[n.module or '' for n in ast.walk(module) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(n.endswith(('.sources','.gate')) or n in ('sources','gate') for n in imports))
        calls=[n for n in ast.walk(module) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='rows']
        self.assertEqual(len(calls),1);self.assertIn('public.jsonl',ast.unparse(calls[0]))

    def test_registered_pointwise_replay_matches_real_framework(self):
        import os
        bpd=Path(os.environ.get('NAN_BPD_SOURCE',str(ROOT.parents[2]/'BPD')))
        framework=load_bpd(bpd)
        from mechinterp.bpd.ci import TiedCINetwork
        torch.manual_seed(2026112007)
        cfg=framework['config'].CIConfig(variant='state_only',d_model=32,n_blocks=0,n_heads=4,mlp_hidden=64,max_passes=2)
        name='site';net=TiedCINetwork({name:7},{name:8},cfg,is_causal=True)
        x=torch.randn(2,2,5,7)
        with torch.no_grad():expected=net({name:x}).lower_leaky[name].numpy()
        bank={'ci__'+key:value.detach().numpy() for key,value in net.state_dict().items()}
        actual=masks(x.numpy(),bank,{'variant':'state_only','n_blocks':0})
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6)
        np.testing.assert_allclose(actual[:,1,-1],masks(x.numpy()[:,1,-1],bank,{'variant':'state_only','n_blocks':0}),rtol=2e-6,atol=2e-6)


if __name__=='__main__':unittest.main()
