"""NumPy-only synthetic replay against the actual frozen inner host processor."""
import ast
from collections import defaultdict
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from mechinterp.nanbeige_bpd_gym.chunked_probes import FixedAssets,grade_packet,project_episode
from mechinterp.nanbeige_bpd_gym.chunked_audit import Layout,audit_episode,complete_batch_signatures,ids_hash
from mechinterp.nanbeige_bpd_gym.tests.test_chunked_audit import Fixture,npy,SMALL


def assets():
    def dictionary(sign):
        return dict(encoder=sign*np.arange(24,dtype=float).reshape(6,4)/32,bias=np.arange(6)/16,
            decoder=sign*np.arange(24,dtype=float).reshape(4,6)/64,mean=np.zeros(4),scale=np.ones(4),decoder_bias=np.zeros(4))
    bank=dict(V=np.eye(8),U=np.ones((8,4))/16,mean_comp_acts=np.ones((2,8)),mean_delta_out=np.zeros((2,4)),mean_write_rms=np.ones(2)*.5)
    bank.update({'ci__proj_in.weight':np.zeros((6,8)),'ci__proj_in.bias':np.ones(6),
                 'ci__out_proj.weight':np.zeros((8,6)),'ci__out_proj.bias':np.ones(8)*.25})
    asset=dict(static_CAL_mask=np.ones((2,8))*.25,
        receipt=dict(final_checkpoint=dict(CI_config=dict(variant='state_only',n_blocks=0)),
                     training_contract=dict(ablation=dict(sphere_rescale=False,eps=1e-6))))
    return FixedAssets('bf16','MOCK_TARGET',{l:[dictionary(1),dictionary(-1)] for l in SMALL.widths},bank,asset,{'MOCK_ONLY':'no trained asset'},True)


def fixture(base,route):
    f=Fixture(base,route)
    for r in f.sources:r['schema'].update(checkpoints=[1,2,3],nodes=['E00','P0','R0','R1','R2'])
    if route!='native':
        for step in range(3):f.mutate_state(1,step,'actual_masks',npy((2,8),kind='F32',word=1. if route=='component_only' else .25))
    layout=Layout(SMALL.widths,4,32,32,8)
    audits=[audit_episode(base,f.entries[i],f.sources[i],members=f.members,ordinal=0,lane=f.lane,original_row=i,
        decode=lambda ids:' '.join(map(str,ids)),parse=lambda text,row:None,eos_token_id=31,layout=layout,
        token_bound=dict(tokens=len(f.initial[i]),ids_sha256=ids_hash(f.initial[i]))) for i in range(2)]
    gold=dict(final_room='R0',intermediate_rooms=['R0']*3,path=['E00','P0'])
    return f,audits,gold


def original_processor(artifact,out,f,gold,fixed):
    # Execute the actual, unchanged nested processor AST, not a reimplementation.
    import experiments.nan_full26.host as original
    from experiments.nan_ram28.common import read,write,sha,append
    from experiments.nan_full26.common import parse
    from mechinterp.nanbeige_bpd_gym.probe_towers import encoding
    from experiments.nan_fit07.pointwise import masks
    from experiments.nan_story23.host import local_replay
    run=ast.parse(inspect.getsource(original.run)).body[0];node=next(x for x in run.body if isinstance(x,ast.FunctionDef) and x.name=='process')
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    env=dict(np=np,Path=Path,defaultdict=defaultdict,artifact=artifact,out=out,read=read,write=write,sha=sha,append=append,
        source={r['id']:r for r in f.sources},gold={r['id']:gold for r in f.sources},parse=parse,decode=original.decode,
        dictionaries={'bf16':fixed.dictionaries},banks={'bf16':fixed.bank},assets={'bf16':fixed.asset},encoding=encoding,masks=masks,local_replay=local_replay)
    exec(compile(module,'frozen_FULL26_inner_process','exec'),env)
    return env['process']


class CachedNumericalProbeTests(unittest.TestCase):
    def test_original_SAE_controller_write_and_recurrence_outputs_are_unchanged(self):
        for route in ('native','component_only','static_CAL_mean','reached_state'):
            with self.subTest(route=route),tempfile.TemporaryDirectory() as directory:
                artifact=Path(directory);base=artifact/'bf16';base.mkdir();new=artifact/'new';new.mkdir();old=artifact/'original';old.mkdir()
                f,audits,gold=fixture(base,route);fixed=assets();keys=complete_batch_signatures(audits)
                reference=original_processor(artifact,old,f,gold,fixed)
                for i,audit in enumerate(audits):
                    result=project_episode(base,audit,f.sources[i],gold,fixed,new,geometry_keys=keys,expected_target_digest='MOCK_TARGET',guard=lambda:None)
                    reference('bf16',f.entries[i]);ref=json.loads((old/('bf16_'+Path(f.entries[i]['file']).stem+'_audit.json')).read_bytes())
                    for name in ('grade','metrics','prefixes','physical_vectors','maximum_mask_abs_error','maximum_local_write_relative_L2_error'):
                        self.assertEqual(result[name],ref[name],name)
                    self.assertEqual(result['status'],'MOCK_ONLY')
                    if i==1 and route!='native':self.assertEqual(result['maximum_local_write_relative_L2_error'],0.)
                    with np.load(new/result['code_file'],allow_pickle=False) as a,np.load(old/ref['code_file'],allow_pickle=False) as b:
                        for name in a.files:self.assertTrue(np.array_equal(a[name],b[name]),name)
    def test_wrong_model_binding_stale_episode_and_bad_mask_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);f,audits,gold=fixture(base,'reached_state');fixed=assets();keys=complete_batch_signatures(audits)
            with self.assertRaises(ValueError):project_episode(base,audits[1],f.sources[1],gold,fixed,base,geometry_keys=keys,expected_target_digest='OTHER',guard=lambda:None)
            fixed.bank['ci__out_proj.bias'][:]=.75
            with self.assertRaises(AssertionError):project_episode(base,audits[1],f.sources[1],gold,fixed,base,geometry_keys=keys,expected_target_digest='MOCK_TARGET',guard=lambda:None)
            self.assertFalse(list(base.glob('*_codes.npz')))
            (base/f.entries[1]['file']).write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):project_episode(base,audits[1],f.sources[1],gold,fixed,base,geometry_keys=keys,expected_target_digest='MOCK_TARGET',guard=lambda:None)
    def test_story_endpoints_and_bounded_label_packets_retain_original_grades(self):
        r=dict(stage='nan_story23',schema=dict(checkpoints=[1,2,3],nodes=['E00','P0','R0','R1','R2']))
        g=dict(final_room='R1',intermediate_rooms=['R0','R1','R2'],path=['E00','P0','R1'])
        text='T1=R0;T2=R1;T3=R2;PATH=E00>P0>R1;FINAL=R1;END'
        self.assertTrue(all(grade_packet(text,r,g).values()))
        wrong=grade_packet(text.replace('PATH=E00>P0>R1','PATH=E00>P0>R0'),r,g)
        self.assertTrue(wrong['final_correct']);self.assertFalse(wrong['path_correct']);self.assertFalse(wrong['joint_correct'])
        for stage in ('nan_gym05','nan_comp14','nan_ctx08','nan_ctx21'):
            row=dict(stage=stage,schema=dict(type='label',allowed=['A','B'],max_characters=64))
            label=dict(answer='A',copy_control=True)
            self.assertEqual(grade_packet('{"trace":["fact"],"answer":"A"};END',row,label),dict(format_valid=True,final_correct=True))
            self.assertFalse(grade_packet('A',row,label)['format_valid'])
            self.assertFalse(grade_packet('{"trace":["fact"],"answer":"B"};END',row,label)['final_correct'])


if __name__=='__main__':unittest.main()
