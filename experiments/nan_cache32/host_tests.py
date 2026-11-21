"""Actual full host/paired/report entry with four synthetic rows and 3958 unknown."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from .tests import source_fixture
from .plan import build,digest
from . import host,paired,common,host_budget,predecessors


class FullHostFixture(unittest.TestCase):
    def test_actual_host_covers_both_precisions_and_all_scheduled_slots_without_Torch(self):
        from mechinterp.nanbeige_bpd_gym.tests.test_chunked_audit import Fixture,Layout,SMALL,ids_hash,save_json
        from mechinterp.nanbeige_bpd_gym.tests.test_chunked_probes import assets
        import mechinterp.nanbeige_bpd_gym.chunked_audit as audit_module
        import mechinterp.nanbeige_bpd_gym.chunked_probes as probes
        import experiments.nan_full26.host as original_host
        import experiments.nan_gym05.common as prompts
        from .loader import TARGETS
        from .report import collect
        source,bindings=source_fixture()
        with tempfile.TemporaryDirectory(prefix='CACHE32_MOCK_ONLY_') as directory:
            root=Path(directory);artifact=root/'artifact';artifact.mkdir();snap=root/'snapshot';snap.mkdir()
            save_json(root/'SOURCE_MANIFEST.json',dict(MOCK_ONLY=True));manifest_sha=common.sha(root/'SOURCE_MANIFEST.json')
            cfg=common.read(common.ROOT/'config.json');cfg['host_packages']={};save_json(root/'config.json',cfg)
            for backend in ('bf16','nf4'):
                base=artifact/backend;base.mkdir();f=Fixture(base)
                for row in (0,1):
                    r=copy.deepcopy(f.sources[row]);r.update(source_id=r['id'],arm='main',prompt=r['id'])
                    r['schema'].update(checkpoints=[1,2,3],nodes=['E00','P0','R0','R1','R2'])
                    source[row]=r;bindings[r['id']]=dict(tokens=len(f.initial[row]),ids_sha256=ids_hash(f.initial[row]))
                with (base/'closed.jsonl').open('w') as stream:
                    for entry in f.entries:stream.write(json.dumps(entry)+'\n')
            token_map={r['prompt']:([1,3,5,7] if r['id']=='MOCK_A' else [2,4] if r['id']=='MOCK_B' else [3,5,7,11]) for r in source}
            for r in source:bindings[r['id']]=dict(tokens=len(token_map[r['prompt']]),ids_sha256=ids_hash(token_map[r['prompt']]))
            plan=build(source,bindings);tokens=dict(contexts=bindings,tokenizer_files_sha256={})
            save_json(artifact/'model_lane_terminal.json',dict(source_manifest_sha256=manifest_sha,all_actual_workers_ended=True,MOCK_ONLY=True))
            gold={r['id']:dict(final_room='R0',intermediate_rooms=['R0']*3,path=['E00','P0']) for r in source}
            actual_read=host.read
            def read(path):return tokens if str(path).replace('\\','/').endswith('experiments/nan_full26/tokenizer_receipt.json') else actual_read(path)
            class Tokenizer:
                eos_token_id=31
                def decode(self,ids,**kw):return ' '.join(map(str,ids))
            class Budget:
                def check(self):assert 'torch' not in sys.modules
                def close(self):pass
            def fixed(backend,target,guard):
                a=assets();a.backend=backend;a.target_digest=TARGETS[backend];return a
            layout=Layout(SMALL.widths,4,32,32,8)
            with (patch.object(host_budget,'Budget',Budget),patch.object(host,'ROOT',root),patch.object(paired,'ROOT',root),
                  patch.object(host,'verify',return_value=dict(public_source_pins={},plan_sha256=digest(plan),MOCK_ONLY=True)),
                  patch.object(host,'load_sources',return_value=(source,{})),patch.object(host,'read',side_effect=read),
                  patch.object(host,'terminal_integrity',return_value=True),patch.object(predecessors,'ensure_inventory_clear'),
                  patch.object(original_host,'labels',return_value=gold),patch.object(prompts,'prompt_ids',side_effect=lambda t,s,text:token_map[text]),
                  patch.object(audit_module,'pinned_layout',return_value=layout),patch.object(probes,'load_fixed_assets',side_effect=fixed),
                  patch.dict(sys.modules,{'transformers':SimpleNamespace(PreTrainedTokenizerFast=lambda **kw:Tokenizer())})):
                host.run(artifact,snap);items=paired.batches(artifact)
            receipt=json.loads((artifact/'host/receipt.json').read_bytes())
            self.assertEqual(sum(v['primary_verified_rows'] for v in receipt['counts'].values()),4)
            self.assertEqual(sum(v['registered_rows'] for v in receipt['counts'].values()),3962)
            table=collect(plan,source,gold,items)
            self.assertEqual(sum(r['verified'] for r in table['accuracy']),4)
            self.assertEqual(sum(r['unknown'] for r in table['accuracy']),3958)
            geometry=json.loads((artifact/'host/paired_geometry.json').read_bytes())
            self.assertEqual(geometry['exact_complete_cache_lineage_prefix_pairs'],4)
            self.assertEqual(geometry['unpaired_or_unverified_scheduled_rows'],1979)
            self.assertEqual(len(geometry['root_level_geometry']),8)
            self.assertNotIn('torch',sys.modules)
    def test_missing_normal_target_receipt_cannot_qualify_raw_closed_rows(self):
        from .loader import TARGETS
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(host.terminal_integrity(folder,'seal','plan',TARGETS['nf4']))


if __name__=='__main__':unittest.main()
