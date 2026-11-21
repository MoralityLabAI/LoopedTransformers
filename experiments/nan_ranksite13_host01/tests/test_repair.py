import ast
import symtable
import unittest
from experiments.nan_ranksite13_host01.repair import amended_source, ORIGINAL


class HostRepair(unittest.TestCase):
    def test_old_failure_and_new_helper_binding(self):
        old = symtable.symtable(ORIGINAL.read_text(), str(ORIGINAL), 'exec')
        new = symtable.symtable(amended_source(), 'amended', 'exec')
        old_probe = next(x for x in old.get_children() if x.get_name() == 'probe')
        new_probe = next(x for x in new.get_children() if x.get_name() == 'probe')
        self.assertTrue(old_probe.lookup('key').is_local())
        self.assertTrue(new_probe.lookup('condition_key').is_global())
        self.assertTrue(new_probe.lookup('condition_name').is_local())

    def test_only_registered_identifier_repairs_change_ast(self):
        tree = ast.parse(amended_source())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                node.id = {'condition_name': 'name', 'condition_key': 'key'}.get(node.id, node.id)
            if isinstance(node, ast.alias) and node.name == 'key':
                node.asname = None
        self.assertEqual(ast.dump(tree), ast.dump(ast.parse(ORIGINAL.read_text())))

    def test_condition_identifier_is_assigned_once(self):
        tree = ast.parse(amended_source())
        probe = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'probe')
        stores = [x for x in ast.walk(probe) if isinstance(x, ast.Name) and x.id == 'condition_name' and isinstance(x.ctx, ast.Store)]
        self.assertEqual(len(stores), 1)
        loads = [x for x in ast.walk(probe) if isinstance(x, ast.Name) and x.id == 'condition_name' and isinstance(x.ctx, ast.Load)]
        self.assertEqual(len(loads), 3)


if __name__ == '__main__':
    unittest.main()
