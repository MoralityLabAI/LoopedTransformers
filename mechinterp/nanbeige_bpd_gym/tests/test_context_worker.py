import ast
from pathlib import Path
import unittest


class ContextWorkerIsolationTests(unittest.TestCase):
    def test_model_worker_reads_public_rows_without_host_source_or_grader_import(self):
        root=Path(__file__).resolve().parents[3]/'experiments/nan_ctx08'
        module=ast.parse((root/'worker.py').read_text())
        imported=[n.module or '' for n in ast.walk(module) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(n.endswith(('.sources','.report','.gate')) for n in imported))
        calls=[n for n in ast.walk(module) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='rows']
        self.assertEqual(len(calls),1);self.assertIn('public.jsonl',ast.unparse(calls[0]))
        strings=[n.value for n in ast.walk(module) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        self.assertNotIn('fixtures/host_labels.jsonl',strings)


if __name__=='__main__':unittest.main()
