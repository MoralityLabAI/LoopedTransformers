from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime

SOURCE='import Init\nnamespace T\ntheorem toy (h : True) : True := by\n  exact h\nend T\n'

class RuntimeTests(unittest.TestCase):
    def test_startup_resource_abort_closes_child(self):
        process=Mock();process.poll.return_value=None
        guard=Mock();guard.check.side_effect=runtime.ResourceLimit('injected startup memory limit')
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)/'probe'
            with patch.object(runtime.subprocess,'Popen',return_value=process):
                with self.assertRaises(runtime.ResourceLimit):
                    runtime.Session('lean',SOURCE,'toy',out,guard)
            process.wait.assert_called_once()
            self.assertTrue((out/'request-1.json').is_file())

    def test_existing_mailbox_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(runtime.subprocess,'Popen') as launch:
                with self.assertRaises(FileExistsError):
                    runtime.Session('lean',SOURCE,'toy',temporary,Mock())
                launch.assert_not_called()

if __name__=='__main__':unittest.main()
