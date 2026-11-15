"""Synthetic *software-only* CPU smoke, never a source-label or theorem result.

Exercise data-loader isolation, model forward/backward, optimizer state,
DEV-only checkpoint selection, SHA verification. All original MathCan Lean
TRAIN/DEV labels must still come from the separate live compiler collector.
"""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import torch
import unittest
from ram10_core import sha_file
from train import train

HERE=Path(__file__).resolve().parent

class SyntheticTrainSmoke(unittest.TestCase):
    def test_one_model_hundred_cpu_updates_and_frozen_byte_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=json.loads((HERE/'config.json').read_text())
            cfg['training']['models']=[{'name':'loop32','width':32,'loops':4}]
            cfg['training']['seeds']=[17]
            cfg['training']['updates']=100
            config=root/'test_config.json';config.write_text(json.dumps(cfg))
            data=root/'fake_compiler_software_only';data.mkdir()
            synthetic=[]
            for part in ('train','dev'):
                for i in range(96 if part=='train' else 32):
                    family=('exact_typed','rw_local','rcases_local','omega')[i%4]
                    label=(i//4)%4
                    goals=['x : Nat\n⊢ x = x']+(['y : Nat\n⊢ y = y'] if i%2 else [])
                    synthetic.append(dict(partition=part,
                        source_file='MathCan/BasinCapture.lean' if part=='train' else 'MathCan/OrderedJet.lean',
                        theorem='software_fixture_train' if part=='train' else 'software_fixture_dev',
                        family=family,script='exact h' if family=='exact_typed' else family,
                        label=label,delta=1 if label in (1,3) else 0,
                        before={'goals':goals},after={'accepted':True,'goals':[] if label==3 else goals},
                        state_sha256=f'{part}:{i}',source_sha256='synthetic-not-a-real-compiler-effect'))
            (data/'transitions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in synthetic))
            hash_=sha_file(data/'transitions.jsonl');config_hash=sha_file(config)
            (data/'collection_report.json').write_text(json.dumps(dict(status='complete',config_sha256=config_hash,dataset_sha256=hash_)))
            (data/'independent_audit.json').write_text(json.dumps(dict(status='passed',dataset_sha256=hash_)))
            (data/'training_gate.json').write_text(json.dumps(dict(admitted=True,config_sha256=config_hash,
                                                            dataset_sha256=hash_,independently_replayed=len(synthetic))))
            output=root/'software_only_weights'
            result=train(data,config,output)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(len(result['models']),1)
            self.assertEqual(result['fresh_test_proof_cases_scored'],0)
            rec=result['selected']['loop32']
            best=output/rec['file']
            self.assertEqual(sha_file(best),rec['sha256'])
            weights=torch.load(best,map_location='cpu',weights_only=True)
            self.assertEqual(weights['schema'],'RAM10-v1')
            self.assertEqual(weights['step'],100)
            self.assertEqual(weights['seed'],17)
            self.assertIn('optimizer',weights)
            self.assertIn('sampler_rng',weights)
            self.assertEqual(weights['data_sha256'],hash_)

if __name__=='__main__':unittest.main()
