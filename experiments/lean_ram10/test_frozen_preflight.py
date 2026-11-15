"""RAM-10 operational integrity tests; no prospective PEF or new TEST states."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch
from ram10_core import sha_file, verify_checkout
from evaluate import verify_frozen

HERE=Path(__file__).resolve().parent

class FrozenIntegrity(unittest.TestCase):
    def config(self):
        return json.loads((HERE/'config.json').read_text())

    def init_fixture(self,root):
        config=self.config()
        conf=root/'config.json';conf.write_text(json.dumps(config,sort_keys=True))
        data=root/'data';data.mkdir()
        (data/'transitions.jsonl').write_text('registered original TRAIN/DEV, fixture only\n')
        sha=sha_file(data/'transitions.jsonl')
        csha=sha_file(conf)
        (data/'independent_audit.json').write_text(json.dumps({'status':'passed','dataset_sha256':sha}))
        (data/'collection_report.json').write_text(json.dumps({'status':'complete','dataset_sha256':sha,'config_sha256':csha}))
        (data/'training_gate.json').write_text(json.dumps({'admitted':True,'dataset_sha256':sha,'config_sha256':csha}))
        training=root/'training';training.mkdir()
        report=dict(status='complete',models=[],selected={},frozen_model_count=9,
                    original_mathcan_source=config['source_mathcan'],
                    development_only_selection=True,fresh_test_proof_cases_scored=0,
                    config_sha256=csha,dataset_sha256=sha)
        for spec in config['training']['models']:
            for i,seed in enumerate(config['training']['seeds']):
                name=spec['name'];sub=training/f'{name}_s{seed}';sub.mkdir()
                score=0.4+i*0.1
                receipt=dict(schema='RAM10-v1',model_config=spec,seed=seed,
                             step=100,state_dict={'dummy':torch.tensor([seed],dtype=torch.float32)},
                             optimizer={},sampler_rng=torch.get_rng_state(),torch_rng=torch.get_rng_state(),
                             config_sha256=csha,data_sha256=sha,original_mathcan_sha=config['source_mathcan'])
                torch.save(receipt,sub/'best.pt');torch.save(receipt,sub/'last.pt')
                row=dict(name=name,seed=seed,parameters=100,best_step=100,last_step=100,best_score=score,
                         best_sha256=sha_file(sub/'best.pt'),last_sha256=sha_file(sub/'last.pt'),
                         relative_best_file=f'{name}_s{seed}/best.pt',
                         history=[dict(step=100,dev=dict(selection_loss=score))])
                report['models'].append(row)
                if i==0:
                    report['selected'][name]=dict(seed=seed,step=100,sha256=row['best_sha256'],file=row['relative_best_file'])
        (training/'training_report.json').write_text(json.dumps(report,sort_keys=True))
        return conf,data,training,report

    def test_rechecks_all_nine_original_best_and_last_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            conf,data,training,receipt=self.init_fixture(Path(temporary))
            _,_,models=verify_frozen(conf,data,training)
            self.assertEqual(set(models),{'loop32','loop64','flat64'})
            # Corrupt an *unselected* model's LAST checkpoint. The original
            # test preflight checked only selected three best checkpoints.
            with (training/'loop64_s43/last.pt').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(AssertionError,'nine original'):
                verify_frozen(conf,data,training)

    def test_rejects_selection_not_supported_by_dev_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            conf,data,training,receipt=self.init_fixture(Path(temporary))
            receipt['selected']['flat64']=dict(seed=43,step=100,
                sha256=next(x['best_sha256'] for x in receipt['models'] if x['name']=='flat64' and x['seed']==43),
                file='flat64_s43/best.pt')
            (training/'training_report.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(AssertionError,'non-DEV'):
                verify_frozen(conf,data,training)

    def test_rejects_tampered_imports_and_untracked_mathcan_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            mathcan=Path(temporary)
            subprocess.run(['git','init','-q',str(mathcan)],check=True)
            (mathcan/'MathCan').mkdir()
            (mathcan/'MathCan/Training.lean').write_text('import Init\ntheorem x : True := by\n  trivial\n')
            (mathcan/'MathCan/Test.lean').write_text('theorem y : True := by trivial\n')
            (mathcan/'MathCan/Imported.lean').write_text('theorem z : True := by trivial\n')
            (mathcan/'lean-toolchain').write_text('leanprover/lean4:v4.35.0\n')
            subprocess.run(['git','-C',str(mathcan),'add','.'],check=True)
            subprocess.run(['git','-C',str(mathcan),'-c','user.name=Research',
                            '-c','user.email=research@example.com','commit','-qm','fixture'],check=True)
            sha=subprocess.check_output(['git','-C',str(mathcan),'rev-parse','HEAD'],text=True).strip()
            cfg=dict(source_mathcan=sha,toolchain='leanprover/lean4:v4.35.0',modules=dict(test={'Test':'gitsha'}))
            (mathcan/'MathCan/Test.lean').unlink()
            self.assertEqual(verify_checkout(mathcan,cfg,'train'),sha)
            (mathcan/'MathCan/Imported.lean').write_text('altered dependency\n')
            with self.assertRaisesRegex(AssertionError,'tracked worktree'):
                verify_checkout(mathcan,cfg,'train')
            (mathcan/'MathCan/Imported.lean').write_text('theorem z : True := by trivial\n')
            (mathcan/'MathCan/Surprise.lean').write_text('theorem w : False := by sorry\n')
            with self.assertRaisesRegex(AssertionError,'untracked original'):
                verify_checkout(mathcan,cfg,'train')
            (mathcan/'MathCan/Surprise.lean').unlink()
            with self.assertRaisesRegex(AssertionError,'tracked worktree'):
                (mathcan/'MathCan/Training.lean').write_text('modified\n')
                verify_checkout(mathcan,cfg,'train')

if __name__=='__main__':unittest.main()
