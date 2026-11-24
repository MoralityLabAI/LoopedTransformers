"""CPU-only safety/metadata contracts; no snapshot or pretrained model."""
import ast
import copy
from pathlib import Path
import subprocess
import sys
import unittest
from .paired import batch_signature
from .common import read,ROOT


class Contracts(unittest.TestCase):
    def row(self,i=0):
        return dict(source_id='MOCK',mode='native',prefix_ids_sha256='prefix',history_sha256='history',batch_size=2,batch_row=i,
            padded_length=7,actual_length=7-i,padding_tokens=1)
    def test_complete_batch_requires_every_unique_row(self):
        self.assertIsNone(batch_signature([self.row()]));self.assertIsNone(batch_signature([self.row(),self.row()]))
        self.assertIsNotNone(batch_signature([self.row(1),self.row(0)]))
    def test_companion_history_mode_shape_and_order_are_bound(self):
        rows=[self.row(0),self.row(1)];before=batch_signature(rows)
        for field,value in [('source_id','other'),('prefix_ids_sha256','other'),('history_sha256','other'),('mode','reached_state'),('actual_length',4),('padding_tokens',3)]:
            changed=copy.deepcopy(rows);changed[1][field]=value;self.assertNotEqual(batch_signature(changed),before,field)
        changed=copy.deepcopy(rows);changed[0]['batch_row']=1;changed[1]['batch_row']=0;self.assertNotEqual(batch_signature(changed),before)
    def test_same_batch_row_order_is_canonical(self):
        rows=[self.row(0),self.row(1)];self.assertEqual(batch_signature(rows),batch_signature(rows[::-1]))
        rows[1]['padded_length']=8;self.assertIsNone(batch_signature(rows))
    def test_unfrozen_loader_fails_before_torch_import(self):
        code="""import pathlib,sys,tempfile
from unittest.mock import patch
import experiments.nan_bf31.common as common
from experiments.nan_bf31.loader import load_backend
with tempfile.TemporaryDirectory(dir='runs') as d,patch.object(common,'ROOT',pathlib.Path(d)):
 try:load_backend('unused','unused','bf16')
 except FileNotFoundError:assert 'torch' not in sys.modules
 else:raise AssertionError('Missing manifest admitted')
"""
        subprocess.run([sys.executable,'-B','-S','-c',code],check=True)
    def test_loader_requires_closed_predecessor_lane(self):
        from unittest.mock import patch
        import experiments.nan_bf31.admission as admission
        with patch.object(admission,'read',side_effect=FileNotFoundError('No authenticated queue admission')):
            with self.assertRaises(FileNotFoundError):admission.verify_worker()
    def test_queue_admission_binds_current_actual_pid(self):
        from unittest.mock import patch
        import experiments.nan_bf31.admission as admission
        def data(path):
            if path.name=='predecessor_lane_closed.json':return dict(source_manifest_sha256='seal',all_authenticated_predecessors_ended=True,known_model_host_audit_inventory_clear=True)
            if path.name=='bf16_runtime.json':return dict(pid=-1,backend='bf16',condition='NAN-BF-31')
            return dict(source_manifest_sha256='seal',remote_commit='frozen')
        with patch.object(admission,'read',side_effect=data),patch.object(admission,'sha',return_value='seal'):
            with self.assertRaises(AssertionError):admission.verify_worker()
    def test_source_publication_precedes_wait_and_model(self):
        source=(ROOT/'queue.py').read_text();self.assertLess(source.index("['git','ls-remote'"),source.index('wait(cfg,event)'))
        loader=(ROOT/'loader.py').read_text();self.assertLess(loader.index('verify_worker()'),loader.index('    import torch'))
    def test_host_inner_math_and_grades_are_unchanged(self):
        def process(path):
            tree=ast.parse(Path(path).read_text())
            node=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='process')
            # Only the explicit raw source-root assignment changes.
            first=node.body[0];assert isinstance(first,ast.Assign) and first.targets[0].id=='base'
            first.value=ast.Name(id='EXPLICIT_BACKEND_ARTIFACT_ROOT',ctx=ast.Load())
            return ast.dump(node,include_attributes=False)
        self.assertEqual(process(ROOT/'host.py'),process('experiments/nan_mlp27/host.py'))
    def test_full_scope_and_numerical_method_budgets_are_retained(self):
        cfg=read(ROOT/'config.json');old=read('experiments/nan_ram28/config.json')
        for key in ('source_seed','checkpoint_revision','baseline_contexts_per_backend','AB_worlds_per_route','AB_routes','maximum_generated_tokens',
                    'story_generated_tokens','other_generated_tokens','capacity_shapes','required_capacity','BF16_resident_decoder_layers',
                    'vocabulary_chunk_rows','layers_zero_based','learned_site','component_token_chunk_rows','MLP_token_chunk_rows',
                    'maximum_backend_seconds','maximum_pair_seconds','maximum_artifact_bytes_per_backend','minimum_free_disk_bytes',
                    'cuda_memory_fraction','formal_inferential_comparisons','neural_retries'):
            self.assertEqual(cfg[key],old[key],key)
        self.assertEqual(cfg['baseline_contexts_per_backend'],1549);self.assertEqual(cfg['AB_worlds_per_route']*len(cfg['AB_routes']),216)
        self.assertEqual(cfg['backends'],['bf16']);self.assertEqual(cfg['minimum_start_available_RAM_bytes'],5*2**30)
    def test_predecessors_get_no_termination_rights(self):
        source=(ROOT/'predecessors.py').read_text();self.assertNotIn('TerminateProcess',source)
        self.assertIn('Handle(pid,creation)',source);self.assertIn('worker|server|host|run',source)
        q=(ROOT/'queue.py').read_text();self.assertLess(q.index('wait(cfg,event)'),q.index("command=[a.python"))
        self.assertIn("actual_worker_ended=True",q);self.assertIn("actual_host_ended=True",q)


if __name__=='__main__':unittest.main()
