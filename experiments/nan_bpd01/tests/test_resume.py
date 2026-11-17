import json
from pathlib import Path
import tempfile
import unittest
from experiments.nan_bpd01.run import checked_resume

class ResumeTests(unittest.TestCase):
    def test_resume_uses_only_bound_complete_pairs_and_rejects_changed_sources(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as name:
            out=Path(name);pair={'root':12,'split':'development','recipient':'r','donor':'d'}
            meta={k:k for k in ['public_sha256','pairs_sha256','config_sha256','backend','checkpoint_revision']}
            (out/'receipt.json').write_text(json.dumps(meta|{'completed_cases':['r','d']}))
            for cid in ['r','d']:(out/(cid+'.json')).write_text(json.dumps({'case_id':cid,'root':12,'split':'development'}))
            prior,completed=checked_resume(out,[pair],meta,True);self.assertEqual(completed,['r','d'])
            with self.assertRaises(ValueError):checked_resume(out,[pair],meta|{'config_sha256':'changed'},True)
            with self.assertRaises(FileExistsError):checked_resume(out,[pair],meta,False)
            (out/'receipt.json').write_text(json.dumps(meta|{'completed_cases':['r']}))
            with self.assertRaises(ValueError):checked_resume(out,[pair],meta,True)

if __name__=='__main__':unittest.main()
