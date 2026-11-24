import ast
from pathlib import Path
import unittest
from experiments.nan_xref25.repair import repaired_tree


class Repair(unittest.TestCase):
    def test_one_float64_cast_in_each_complete_frozen_auditor(self):
        for source in ('nan_xref19','nan_xref22'):
            tree,changes=repaired_tree((Path('experiments')/source/'audit.py').read_text())
            self.assertEqual(changes['physical_geometry_float64_direction_casts'],1)
            nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='U']
            self.assertEqual(len(nodes),1);call=nodes[0].value
            self.assertEqual(call.func.attr,'astype');self.assertEqual(call.args[0].id,'float')
            self.assertIsInstance(call.func.value,ast.IfExp)
            # The AST reverse assertion in repaired_tree verifies that no
            # tolerance, root equation or other operation was changed.

    def test_unexpected_direction_assignment_is_rejected(self):
        original=Path('experiments/nan_xref19/audit.py').read_text()
        with self.assertRaises(AssertionError):repaired_tree(original+'\nU = unknown_array\n')


if __name__=='__main__':unittest.main()
