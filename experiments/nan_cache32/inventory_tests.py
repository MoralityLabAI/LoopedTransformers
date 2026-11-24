"""Metadata-only admission contracts; no Torch, NumPy or process control."""
import copy
from pathlib import Path
import tempfile
import unittest
from .inventory import arguments,owned_host_launcher,blockers


class InventoryContracts(unittest.TestCase):
    def setUp(self):
        self.launcher=r'C:\known\venv\Scripts\python.exe'
        tail='-X utf8 -B -u -m experiments.nan_cache32.host --artifact-root MOCK_ONLY --snapshot MOCK_ONLY'
        self.parent=dict(pid=10,parent_pid=1,name='python.exe',exe=self.launcher,command='"'+self.launcher+'" '+tail,creation=100)
        self.own=dict(pid=11,parent_pid=10,name='python.exe',exe=r'C:\Python312\python.exe',command='"C:\\Python312\\python.exe" '+tail,creation=101)
        self.foreign=dict(pid=12,parent_pid=1,name='python.exe',exe=self.launcher,command=self.launcher+' '+tail,creation=102)
    def test_exact_owned_host_launcher_is_not_a_second_host(self):
        self.assertTrue(owned_host_launcher(self.own,self.parent,self.launcher))
        self.assertEqual(blockers([self.parent,self.own],11,self.launcher),[])
    def test_foreign_identical_host_remains_blocked(self):
        self.assertEqual([r['pid'] for r in blockers([self.parent,self.own,self.foreign],11,self.launcher)],[12])
    def test_parent_pid_reuse_and_missing_times_never_exempt(self):
        for changes in (dict(pid=9),dict(creation=102),dict(creation=0),dict(creation=True)):
            self.assertFalse(owned_host_launcher(self.own,dict(self.parent,**changes),self.launcher))
    def test_wrong_executable_or_nonpython_parent_never_exempt(self):
        for changes in (dict(exe=r'C:\other\python.exe'),dict(name='other.exe'),dict(exe=None)):
            self.assertFalse(owned_host_launcher(self.own,dict(self.parent,**changes),self.launcher))
    def test_argument_or_module_changes_never_exempt(self):
        for text in (self.parent['command']+' --other',self.parent['command'].replace('.host ','.worker '),self.parent['command']+'\n'):
            self.assertFalse(owned_host_launcher(self.own,dict(self.parent,command=text),self.launcher))
        own=copy.deepcopy(self.own);own['command']=own['command'].replace('.host ','.worker ')
        parent=copy.deepcopy(self.parent);parent['command']=parent['command'].replace('.host ','.worker ')
        self.assertFalse(owned_host_launcher(own,parent,self.launcher))
        own['command']=own['command'].replace('.worker ','.worker --label "-m experiments.nan_cache32.host" ')
        parent['command']=parent['command'].replace('.worker ','.worker --label "-m experiments.nan_cache32.host" ')
        self.assertFalse(owned_host_launcher(own,parent,self.launcher))
    def test_foreign_model_and_host_lanes_still_blocked(self):
        records=[self.parent,self.own]+[dict(self.foreign,pid=20+i,command=self.foreign['command'].replace('.host ','.'+name+' ')) for i,name in enumerate(('worker','server','host','run'))]
        self.assertEqual([r['pid'] for r in blockers(records,11,self.launcher)],[20,21,22,23])
    def test_malformed_or_ambiguous_metadata_never_exempt(self):
        self.assertIsNone(arguments('unparseable'))
        self.assertFalse(owned_host_launcher(None,self.parent,self.launcher))
        with self.assertRaises(AssertionError):blockers([self.own,self.own],11,self.launcher)
    def test_original_predecessor_wait_AST_remains_exact(self):
        from .inventory_amendment import assert_predecessor_equivalence
        root=Path(__file__).parent;original=root/'predecessors_before_inventory.py.txt';current=root/'predecessors.py'
        assert_predecessor_equivalence(current,original)
        with tempfile.NamedTemporaryFile(mode='w',suffix='.py',encoding='utf-8',delete=False) as stream:
            temporary=Path(stream.name);stream.write(current.read_text().replace('alive(30000)','alive(10000)'))
        try:
            with self.assertRaises(AssertionError):assert_predecessor_equivalence(temporary,original)
        finally:temporary.unlink()


if __name__=='__main__':unittest.main()
