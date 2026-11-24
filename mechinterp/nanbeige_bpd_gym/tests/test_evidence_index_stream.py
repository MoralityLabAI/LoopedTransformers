import io
import unittest
from unittest.mock import patch
from mechinterp.nanbeige_bpd_gym.evidence_index import OUTPUTS,BRANCH
from mechinterp.nanbeige_bpd_gym.evidence_index_stream import publish_generated


class EvidencePublicationOrderTests(unittest.TestCase):
    def test_unpushed_worker_commit_never_enters_Git_mutation_path(self):
        with patch('subprocess.check_output',side_effect=['a'*40,'b'*40+'\trefs/heads/'+BRANCH+'\n']),patch('subprocess.run') as run:
            self.assertFalse(publish_generated('a'*40,io.StringIO()));run.assert_not_called()

    def test_worker_advancing_HEAD_during_index_generation_defers_all_writes(self):
        with patch('subprocess.check_output',return_value='b'*40),patch('subprocess.run') as run:
            self.assertFalse(publish_generated('a'*40,io.StringIO()));run.assert_not_called()

    def test_published_unchanged_worker_commit_stages_only_named_index_files(self):
        with patch('subprocess.check_output',side_effect=['a'*40,'a'*40+'\trefs/heads/'+BRANCH+'\n']),patch('subprocess.run') as run:
            self.assertTrue(publish_generated('a'*40,io.StringIO()))
        commands=[call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0],['git','add','--sparse','--',*OUTPUTS])
        self.assertIn('--only',commands[1]);self.assertEqual(commands[1][-2:],list(OUTPUTS))
        self.assertEqual(commands[2],['git','push','origin','HEAD:refs/heads/'+BRANCH])


if __name__=='__main__':unittest.main()
