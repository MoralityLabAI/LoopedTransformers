"""MOCK_ONLY new decoder paths, dimensions, hash fence and partial-world metadata."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from experiments.nan_xref22 import cache
from experiments.nan_xref22.common import rows,sha,write


class CacheDecoder(unittest.TestCase):
    def fixture(self,folder):
        row=rows('experiments/nan_ctx21/fixtures/public.jsonl')[0]
        widths=dict(residual_after_layer=3072,down_proj_input=10752,down_proj_output=3072,o_proj_input=6144,o_proj_output=3072)
        arrays={};index=[]
        for t in (1,2):
            for l in (5,11,17,21):
                for site,width in widths.items():
                    key='v'+str(len(arrays));arrays[key]=np.full(width,t+l/100,dtype=np.float32)
                    index.append(dict(loop=t,layer_zero_based=l,site=site,array_key=key))
        archive=folder/'MOCK_ONLY_states.npz'
        with archive.open('xb') as stream:
            np.savez_compressed(stream,**arrays)
        record=folder/'MOCK_ONLY_case.json';write(record,row|dict(state_index=index))
        audit=folder/'MOCK_ONLY_audit.json';receipt=folder/'MOCK_ONLY_receipt.json';write(audit,dict(MOCK_ONLY=True));write(receipt,dict(MOCK_ONLY=True))
        admission=dict(status='admitted',audit=str(audit),audit_sha256=sha(audit),receipt=str(receipt),receipt_sha256=sha(receipt),available_contexts=1,
            files=[dict(record=str(record),record_sha256=sha(record),state=str(archive),state_sha256=sha(archive))])
        write(folder/'admission/nf4.json',admission)
        return admission

    def exercise(self,corrupt):
        workspace=Path.cwd().resolve();parent=workspace/'runs/xref22_decoder_tests';parent.mkdir(parents=True,exist_ok=True)
        self.assertTrue(parent.resolve().is_relative_to(workspace))
        with tempfile.TemporaryDirectory(prefix='MOCK_ONLY_',dir=parent) as temporary:
            folder=Path(temporary).resolve();self.assertTrue(folder.is_relative_to(parent.resolve()))
            admission=self.fixture(folder)
            if corrupt:
                admission['files'][0]['state_sha256']='0'*64
                (folder/'admission/nf4.json').write_text(json.dumps(admission),encoding='utf-8')
            with patch.object(cache,'RUNS',folder):
                if corrupt:
                    with self.assertRaises(AssertionError):
                        cache.load('ctx21','nf4')
                else:
                    H,X,Y,meta=cache.load('ctx21','nf4')
                    self.assertEqual((X.shape,Y.shape),((2,10752),(2,3072)))
                    self.assertEqual([r['loop'] for r in meta],[1,2]);self.assertEqual([r['world'] for r in meta],['A','A'])
                    self.assertEqual(H[21].shape,(2,3072));self.assertAlmostEqual(H[21][0,0],1.21,places=6)
                    self.assertAlmostEqual(H[21][1,0],2.21,places=6)
            self.assertTrue(folder.is_relative_to(parent.resolve()) and parent.resolve().is_relative_to(workspace),'Check containment before TemporaryDirectory cleanup')

    def test_partial_case_both_loops_keep_exact_physical_sites(self):
        self.exercise(False)

    def test_archive_hash_mismatch_prevents_any_vector_rescue(self):
        self.exercise(True)


if __name__=='__main__':
    unittest.main()
