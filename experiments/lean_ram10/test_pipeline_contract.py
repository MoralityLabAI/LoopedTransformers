"""RAM-10 isolation and fake-compiler contract checks, no synthetic TRAIN labels stored."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from ram10_core import source_order_roots,verify_checkout,sha_file,sha_bytes
from collect import collect_root,independent_audit
from evaluate import verify_frozen

class FakeLeanCore:
    @staticmethod
    def signature(goals):return sha_bytes(json.dumps(goals).encode())
    @staticmethod
    def sha(text):return sha_bytes(text.encode())
    @staticmethod
    def label(before,after):
        if not after['accepted'] or after['goals']==before['goals']:return 0
        if not after['goals']:return 3
        return 1 if len(after['goals'])<len(before['goals']) else 2
    @staticmethod
    def _ram10_original_candidates(state):
        if '⊢ B' in state['goals'][0]:return [('exact_local','exact h'),('rfl','rfl')]
        return [('intro_named','intro ram07_arg0'),('rfl','rfl')]
    candidates=_ram10_original_candidates

class FakeSession:
    def __init__(self,lean,source,theorem,out,guard,timeout=10.):
        self.initial=dict(accepted=True,goals=['⊢ A'],definitions=[])
        self.events=[]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def query(self,actions,timeout):
        self.events.append(actions)
        if actions==['intro ram07_arg0']:
            return dict(accepted=True,goals=['h : A\n⊢ B'],definitions=[])
        if actions==['intro ram07_arg0','exact h']:
            return dict(accepted=True,goals=[],definitions=[])
        return dict(accepted=False,goals=['⊢ A'],definitions=[])

class Guard:
    def check(self):pass

class PipelineTests(unittest.TestCase):
    def test_pinned_blob_selection_and_fs_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);(repo/'MathCan').mkdir();subprocess.run(['git','init','-q',str(repo)],check=True)
            (repo/'lean-toolchain').write_text('leanprover/lean4:v4.35.0\n')
            (repo/'MathCan/A.lean').write_text('import Init\ntheorem x : True := by\n  trivial\ntheorem y : True := by\n  trivial\n')
            path=repo/'MathCan/A.lean';blob=subprocess.check_output(['git','-C',str(repo),'hash-object',str(path)],text=True).strip()
            subprocess.run(['git','-C',str(repo),'add','MathCan/A.lean','lean-toolchain'],check=True)
            subprocess.run(['git','-C',str(repo),'-c','user.name=Research','-c','user.email=research@example.com','commit','-qm','fixture'],check=True)
            revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
            cfg=dict(source_mathcan=revision,
                     toolchain='leanprover/lean4:v4.35.0',modules=dict(train={'A':blob},dev={},test={'TEST':'unknown'}),root_selector={'first_train_dev':6,'first_test':2})
            selected=source_order_roots(repo,cfg,'train')
            self.assertEqual([x['theorem'] for x in selected],['x','y'])
            self.assertTrue(all(x['git_blob_sha']==blob for x in selected))
            (repo/'MathCan/TEST.lean').write_text('theorem secret : True := by trivial\n')
            # Source-only train workspace must *not* contain original TEST files.
            with self.assertRaises(AssertionError):
                verify_checkout(repo,cfg,'train')
    def test_real_state_actions_and_independent_compiler_replay_contract(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            (root/'MathCan').mkdir();source='theorem example : True := by\n  trivial\n'
            (root/'MathCan/Sample.lean').write_text(source)
            rec=dict(source_file='MathCan/Sample.lean',theorem='example',
                     source_sha256=sha_bytes(source.encode()),partition='train')
            cfg={'collector':{'states_per_root':3,'max_transitions':6,'max_goals':8,'per_request_seconds':4}}
            artifact=root/'output';artifact.mkdir()
            summary=collect_root(root,rec,0,artifact/'collect',artifact,'lean',FakeLeanCore,FakeSession,Guard(),cfg)
            self.assertEqual(summary['observed_states'],2)
            self.assertEqual(summary['candidate_records'],4)
            data=[json.loads(s) for s in (artifact/'transitions.jsonl').read_text().splitlines()]
            self.assertEqual([r['label'] for r in data],[2,0,3,0])
            audited=independent_audit(root,artifact,'lean',FakeLeanCore,FakeSession,Guard(),cfg)
            self.assertEqual(audited['replayed'],4)
            self.assertEqual(audited['status'],'passed')
    def test_frozen_test_precondition_rejects_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);data=root/'data';data.mkdir();weight=root/'training';weight.mkdir()
            cfg=root/'config.json';cfg.write_text('{}')
            (data/'transitions.jsonl').write_text('a\n')
            digest=sha_file(data/'transitions.jsonl')
            (data/'independent_audit.json').write_text(json.dumps({'status':'passed','dataset_sha256':digest}))
            (data/'training_gate.json').write_text(json.dumps({'admitted':True,'dataset_sha256':digest}))
            (weight/'training_report.json').write_text(json.dumps({'status':'complete','models':[{}]*9,
              'selected':{'loop32':{'sha256':'bad','file':'missing.pt'},'loop64':{'sha256':'bad','file':'missing.pt'},'flat64':{'sha256':'bad','file':'missing.pt'}},
              'config_sha256':sha_file(cfg),'dataset_sha256':digest}))
            with self.assertRaises(FileNotFoundError):verify_frozen(cfg,data,weight)

if __name__=='__main__':unittest.main()
