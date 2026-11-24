import ast
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from experiments.nan_fit16_io18.common import ROOT, read


class WriterRepair(unittest.TestCase):
    def test_exact_single_binding_recipe_change(self):
        parent = ast.parse(Path('experiments/nan_fit16/fit.py').read_text(encoding='utf-8'))
        derived = ast.parse((ROOT / 'fit.py').read_text(encoding='utf-8'))
        imports = [n for n in derived.body if isinstance(n, ast.ImportFrom) and n.module == 'experiments.nan_fit07.io']
        self.assertEqual(len(imports), 1); self.assertEqual([(n.name, n.asname) for n in imports[0].names], [('write', 'write_steps')])
        derived.body.remove(imports[0]); repaired = []
        for node in ast.walk(derived):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'write_steps':
                self.assertEqual(ast.unparse(node.args[0]), "cell / 'steps_local.json'")
                repaired.append(node); node.func.id = 'write'
        self.assertEqual(len(repaired), 1)
        self.assertEqual(ast.dump(derived), ast.dump(parent))

    def test_actual_selected_writer_keeps_all_sixteen_snapshots(self):
        tree = ast.parse((ROOT / 'fit.py').read_text(encoding='utf-8'))
        binding = next(n for n in tree.body if isinstance(n, ast.ImportFrom) and any(a.asname == 'write_steps' for a in n.names))
        alias = next(a for a in binding.names if a.asname == 'write_steps')
        writer = getattr(importlib.import_module(binding.module), alias.name)
        from experiments.nan_fit16.common import write as original_writer
        workspace = Path.cwd().resolve()
        with tempfile.TemporaryDirectory(prefix='io18-writer-', dir=workspace) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(workspace))
            target = Path(directory) / 'steps_local.json'
            for count in range(1, 17):
                values = [{'step': i} for i in range(count)]; writer(target, values)
                self.assertEqual(json.loads(target.read_bytes()), values)
            original = Path(directory) / 'original_steps.json'; original_writer(original, [0])
            with self.assertRaises(FileExistsError):
                original_writer(original, [0, 1])

    def test_sources_and_numerical_config_unchanged(self):
        old = read('experiments/nan_fit16/config.json'); new = read(ROOT / 'config.json')
        old.pop('experiment'); new.pop('experiment'); self.assertEqual(old, new)
        for name in ('public.jsonl', 'host_labels.jsonl'):
            self.assertEqual((ROOT / 'fixtures' / name).read_bytes(), (Path('experiments/nan_fit16/fixtures') / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
