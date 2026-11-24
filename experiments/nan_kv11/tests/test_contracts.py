import ast
from collections import Counter
from pathlib import Path
import unittest
from experiments.nan_kv11.common import public_rows,read


class CacheConditionContracts(unittest.TestCase):
    def test_complete_public_DEV_roster_includes_every_suite_and_copy(self):
        _,rows=public_rows();self.assertEqual(len(rows),130);self.assertEqual(len({r['root'] for r in rows}),41)
        self.assertEqual(Counter(r['suite'] for r in rows),dict(pointer=80,arc1=6,arc2=6,sudoku=6,storyworld=32))
        self.assertEqual(sum(r['id'].endswith(':copy') for r in rows),41)

    def test_two_native_loops_and_reference_history_are_not_reinterpreted(self):
        root=Path('experiments/nan_kv11');cfg=read(root/'config.json');self.assertEqual(cfg['native_loops'],2);self.assertEqual(cfg['maximum_generated_tokens'],8)
        text=(root/'worker.py').read_text();tree=ast.parse(text)
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('oracle' in n or n.endswith(('.report','.gate','.sources')) for n in imports))
        self.assertIn('history.append(token)',text);self.assertNotIn('history.append(candidate)',text)
        strings=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        self.assertNotIn('host_labels.jsonl',strings);self.assertNotIn('progress.json',strings)


if __name__=='__main__':unittest.main()
