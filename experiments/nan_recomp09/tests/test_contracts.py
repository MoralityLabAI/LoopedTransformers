import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from experiments.nan_recomp09.common import target_digest,read


class ExecutionAmendmentTests(unittest.TestCase):
    def test_bounded_hash_retains_original_parameter_byte_digest(self):
        model=torch.nn.Linear(7,9,dtype=torch.bfloat16);digest=hashlib.sha256()
        for name,p in model.named_parameters():
            digest.update(name.encode());digest.update(memoryview(p.detach().contiguous().view(torch.uint8).numpy()))
        self.assertEqual(target_digest(SimpleNamespace(model=model),chunk_bytes=7),digest.hexdigest())

    def test_recipe_retains_fit07_hyperparameters_and_CPU_cap(self):
        root=Path(__file__).resolve().parents[1];cfg=read(root/'config.json');parent=read(root.parent/'nan_fit07/config.json')
        for key in ('fit_steps','rank','site','input_width','output_width','main_lr','coeff_faith','coeff_stoch','coeff_adv',
                    'adversary_inner_steps','maximum_added_saved_tensor_CPU_bytes','source_seed','diagnostic_modes'):
            self.assertEqual(cfg[key],parent[key])
        self.assertEqual(cfg['fit_passes'],[2]);self.assertEqual(cfg['backend'],'bf16')

    def test_worker_reads_only_registered_public_rows_not_gold(self):
        root=Path(__file__).resolve().parents[1];tree=ast.parse((root/'worker.py').read_text())
        imported=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(n.endswith(('.sources','.gate','.report')) for n in imported))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='rows']
        self.assertEqual(len(calls),1);self.assertEqual(ast.unparse(calls[0].args[0]),'public_file')
        strings=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        self.assertNotIn('host_labels.jsonl',strings)


if __name__=='__main__':unittest.main()
