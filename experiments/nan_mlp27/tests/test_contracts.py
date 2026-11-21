import ast
import json
from pathlib import Path
import unittest
from experiments.nan_mlp27.common import qualify

class Contracts(unittest.TestCase):
    def test_cuda_capacity_runs_both(self):
        a=dict(status='capacity_stopped',failure=dict(type='OutOfMemoryError',message='CUDA out of memory'))
        b=dict(status='completed',failure=None)
        self.assertTrue(qualify(dict(nf4=a,bf16=b))['run_both_backends'])
        a['correctness']=0;b['correctness']=1
        self.assertTrue(qualify(dict(nf4=a,bf16=b))['run_both_backends'])
    def test_admission_is_resource_failure(self):
        for msg in ('Unadmitted input capacity; preserve and stop','A/B capacity unavailable; no serial replacement'):
            a=dict(failure=dict(type='AssertionError',message=msg))
            self.assertTrue(qualify(dict(nf4=a,bf16=dict(failure=None)))['run_both_backends'])
    def test_not_accuracy_or_generic_failure(self):
        for kind,msg in [('AssertionError','all-one closure'),('MemoryError','CPU allocation'),('RuntimeError','backend_time_budget')]:
            a=dict(status='failed',failure=dict(type=kind,message=msg),correct_worlds=0)
            self.assertFalse(qualify(dict(nf4=a,bf16=dict(failure=None)))['run_both_backends'])
        self.assertFalse(qualify(dict(nf4=dict(status='completed'),bf16=dict(status='completed')))['run_both_backends'])
    def test_missing_closed_receipt_unrun(self):
        self.assertFalse(qualify(dict(nf4=dict(failure=dict(type='OutOfMemoryError',message='CUDA'))))['run_both_backends'])
    def test_full_scope_and_same_criteria(self):
        old=json.loads(Path('experiments/nan_full26/config.json').read_text())
        new=json.loads(Path('experiments/nan_mlp27/config.json').read_text())
        for k,v in old.items():
            if k not in ('experiment','capacity_shapes'):self.assertEqual(new[k],v,k)
        self.assertEqual(new['baseline_contexts_per_backend'],1549);self.assertEqual(new['AB_worlds_per_route'],72)
        self.assertEqual(new['required_capacity'],[dict(batch=1,tokens=9216),dict(batch=2,tokens=4608)])
    def test_frozen_generation_is_reused(self):
        from experiments.nan_mlp27.engine import Engine
        from experiments.nan_full26.worker import Engine as Original
        self.assertIs(Engine.generate,Original.generate);self.assertIs(Engine.ids,Original.ids)
    def test_host_numerical_function_identical(self):
        def process(path):
            t=ast.parse(Path(path).read_text());return ast.dump(next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='process'))
        self.assertEqual(process('experiments/nan_mlp27/host.py'),process('experiments/nan_full26/host.py'))
    def test_exact_batch_and_padded_capacity(self):
        from experiments.nan_mlp27.engine import Engine
        e=Engine.__new__(Engine);e.capacity=[dict(status='passed',batch=2,length=9216,includes_reached_state_route=True,right_padding=1)]
        self.assertTrue(e.admitted(2,9000));self.assertFalse(e.admitted(1,9000))
        e.capacity[0]['right_padding']=0;self.assertFalse(e.admitted(2,9000))
    def test_no_original_writer_stop(self):
        s=Path('experiments/nan_mlp27/queue.py').read_text()
        self.assertNotIn('terminate(',s);self.assertNotIn('TerminateProcess',s);self.assertNotIn('Stop-Process',s)
    def test_engineering_source_bindings(self):
        from experiments.nan_mlp27.common import ROOT,read,sha
        r=read(ROOT/'engineering_receipt.json');self.assertEqual(r['status'],'passed')
        self.assertEqual(r['pretrained_model_forwards'],0);self.assertEqual(r['GPU_allocations'],0)
        for name,h in r['source_sha256'].items():self.assertEqual(sha(ROOT/name),h,name)

if __name__=='__main__':unittest.main()
