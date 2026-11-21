import ast
from collections import Counter
from pathlib import Path
import unittest
import tempfile
import numpy as np
from experiments.nan_kv12.common import public_rows,read
from experiments.nan_kv12.common import sha,write
from experiments.nan_kv12.report import historical_prefixes


class CacheConditionContracts(unittest.TestCase):
    def test_explicit_policy_is_mounted_before_observers_without_a_second_BF16_run(self):
        root=Path('experiments/nan_kv12');cfg=read(root/'config.json')
        self.assertEqual(cfg['backends'],['nf4']);text=(root/'worker.py').read_text()
        self.assertLess(text.index('replaced=mount_effective_nf4(loop)'),text.index('observer=TowerObserver(loop'))
        self.assertIn('assert target_digest(loop)==before',text)
        self.assertIn('numerical_policy=cfg',text)

    def test_historical_comparison_uses_only_common_initial_histories(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);new=root/'new';old=root/'old';source=dict(id='r:main',root='r',suite='pointer')
            for folder in (new/'nf4',old/'nf4',old/'bf16'):
                folder.mkdir(parents=True)
                arrays={}
                for t in (1,2):
                    arrays[f'reference_boundary_{t}']=np.array([1,t],dtype=np.float32)
                    arrays[f'reference_logits_{t}']=np.array([0,1],dtype=np.float32)
                    for layer in (5,11,17,21):arrays[f'reference_{t}_{layer}_residual_after_layer']=np.array([1,t],dtype=np.float32)
                np.savez(folder/'initial.npz',**arrays)
                write(folder/'case_0000.json',dict(id=source['id'],initial_prompt_ids=[4,5],copy_control=False,
                    steps=[dict(artifact_file='initial.npz',artifact_sha256=sha(folder/'initial.npz')),
                        dict(artifact_file='diverged_tail_must_never_be_opened.npz')]))
                if folder.parent==old:write(folder/'receipt.json',dict(status='completed',source_manifest_sha256='bf9cb5d06ffa94d57de11f54dc4d4d4b2ab473c4ccf9f22715fb8d6d7960de67'))
            result=historical_prefixes(new,[source],historical_root=old)
            self.assertEqual(len(result['groups']),4)
            for group in result['groups']:
                self.assertEqual(group['contexts'],1)
                self.assertEqual(group['root_macro']['boundary_relative_l2'],0)
                self.assertEqual(group['root_macro']['normalized_loop_delta_relative_l2'],0)
                self.assertEqual(group['root_macro']['free_argmax_equal'],1)

    def test_complete_public_DEV_roster_includes_every_suite_and_copy(self):
        _,rows=public_rows();self.assertEqual(len(rows),130);self.assertEqual(len({r['root'] for r in rows}),41)
        self.assertEqual(Counter(r['suite'] for r in rows),dict(pointer=80,arc1=6,arc2=6,sudoku=6,storyworld=32))
        self.assertEqual(sum(r['id'].endswith(':copy') for r in rows),41)

    def test_two_native_loops_and_reference_history_are_not_reinterpreted(self):
        root=Path('experiments/nan_kv12');cfg=read(root/'config.json');self.assertEqual(cfg['native_loops'],2);self.assertEqual(cfg['maximum_generated_tokens'],8)
        text=(root/'worker.py').read_text();tree=ast.parse(text)
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('oracle' in n or n.endswith(('.report','.gate','.sources')) for n in imports))
        self.assertIn('history.append(token)',text);self.assertNotIn('history.append(candidate)',text)
        strings=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        self.assertNotIn('host_labels.jsonl',strings);self.assertNotIn('progress.json',strings)


if __name__=='__main__':unittest.main()
