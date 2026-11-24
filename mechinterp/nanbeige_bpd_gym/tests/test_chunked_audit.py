"""Independent manual cached-v2/NPY fixtures, zero numerical dependencies."""
import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile
from mechinterp.nanbeige_bpd_gym.chunked_audit import (Layout,LAYERS,SITES,audit_episode,audit_states,
    complete_batch_signatures,context_hash,file_hash,ids_hash)


WIDTHS={s:8 if s=='down_proj_input' else 6 if s=='o_proj_input' else 4 for s in SITES}
SMALL=Layout({l:dict(WIDTHS) for l in LAYERS},4,32,32,2)


def save_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def npy(shape,*,kind='BF16',word=0x3f80):
    descr='<u2' if kind=='BF16' else '<f4';header=repr(dict(descr=descr,fortran_order=False,shape=shape))
    size=10+len(header)+1;header=(header+' '*((-size)%64)+'\n').encode('ascii')
    raw=struct.pack('<H',word) if kind=='BF16' else struct.pack('<f',word)
    return b'\x93NUMPY\x01\x00'+len(header).to_bytes(2,'little')+header+raw*math.prod(shape)


class Fixture:
    def __init__(self,base,route='native'):
        self.base=Path(base);self.lane='baseline' if route=='native' else 'AB';self.route=route
        self.sources=[dict(id='MOCK_A',root='ROOT_A',stage='nan_story23',split='DEV',schema=dict(type='story')),
                      dict(id='MOCK_B',root='ROOT_B',stage='nan_story23',split='DEV',schema=dict(type='story'))]
        if route!='native':self.sources[1]=copy.deepcopy(self.sources[0])
        self.initial=[[1,3,5,7],[2,4] if route=='native' else [1,3,5,7]]
        self.generated=[[31],[7,7,31]];self.modes=['native',route]
        self.members=[dict(source_id=r['id'],mode=m,initial_ids=ids) for r,m,ids in zip(self.sources,self.modes,self.initial)]
        self.episodes=[];self.entries=[];parent='0'*64;prefixes=[[],[]]
        for step,active in enumerate(([0,1],[1],[1])):
            P=4+step;lengths=[len(self.initial[i])+step for i in active]
            # Construct expected masks directly, independently of producer code.
            masks=[[int(p<len(self.initial[i]) or p>=4) for p in range(P)] for i in active]
            runs=[]
            for i in active:
                n=len(self.initial[i]);runs.append([[0,P]] if n==4 else [[0,n]]+([[4,P]] if step else []))
            context=dict(source_ids=[self.sources[i]['id'] for i in active],modes=[self.modes[i] for i in active],
                prefix_ids_sha256=[ids_hash(self.initial[i]+self.generated[i][:step]) for i in active],
                history_sha256=[ids_hash(self.generated[i][:step]) for i in active],semantic_lengths=lengths,
                initial_semantic_lengths=[len(self.initial[i]) for i in active],physical_cache_length=P,
                attention_mask_encoding='half-open-runs-v1',attention_mask_runs=runs,attention_mask_sha256=ids_hash(masks),
                parent_cache_context_sha256=parent,original_row_indices=active,original_prefill_canvas=4)
            parent=context_hash(context)
            for b,i in enumerate(active):
                stem=f'{self.lane}_0000_r{i}_p{step:03d}';index=[]
                state=self.base/(stem+'.npz')
                with zipfile.ZipFile(state,'w') as archive:
                    for phase in (1,2):
                        for layer in LAYERS:
                            for site in SITES:
                                key='v'+str(len(index));index.append(dict(loop=phase,layer=layer,site=site,key=key))
                                word=0x3f00 if self.modes[i]!='native' and layer==21 and site=='down_proj_output' else 0x3f80
                                archive.writestr(key+'.npy',npy((WIDTHS[site],),word=word))
                    for name,shape in (('boundaries',(2,4)),('native_at_reached_input',(2,4)),('instrument_input_last_token',(2,8))):
                        archive.writestr(name+'.npy',npy(shape))
                    if self.modes[i]!='native':archive.writestr('actual_masks.npy',npy((2,2),kind='F32',word=1. if route=='component_only' else .25))
                p=dict(schema='cached-prefix-v2',step=step,source_id=self.sources[i]['id'],mode=self.modes[i],token=self.generated[i][step],
                    history_sha256=context['history_sha256'][b],prefix_ids_sha256=context['prefix_ids_sha256'][b],
                    batch_context=context,batch_size=len(active),batch_row=b,actual_length=lengths[b],padded_length=P,
                    padding_tokens=sum(P-n for n in lengths),full_vocabulary_size=32,top_logit=10.,log_normalizer=10.01,
                    cache_bytes_for_entire_batch=32*P*len(active),cuda_elapsed_ms_for_entire_batch=0.,wall_seconds_for_entire_batch=0.,
                    state_file=state.name,state_sha256=file_hash(state),state_index=index,
                    full_logits_independently_reconstructible_from_export=False,complete_batch_context_sha256=parent)
                path=self.base/(stem+'.json');save_json(path,p);prefixes[i].append(dict(file=path.name,sha256=file_hash(path)))
        for i in range(2):
            path=self.base/f'{self.lane}_0000_r{i}_closed.json';r=self.sources[i]
            episode=dict(schema='cached-episode-v2',source_id=r['id'],root=r['root'],stage=r['stage'],split=r['split'],mode=self.modes[i],lane=self.lane,
                initial_ids=self.initial[i],generated_ids=self.generated[i],text='' if i==0 else '7 7',stop_reason='eos',
                fragments=prefixes[i],prefix_count=len(prefixes[i]),wall_seconds=0.)
            save_json(path,episode);self.episodes.append(episode);self.entries.append(dict(file=path.name,sha256=file_hash(path)))
    def audit(self,i):
        return audit_episode(self.base,self.entries[i],self.sources[i],members=self.members,ordinal=0,lane=self.lane,original_row=i,
            decode=lambda ids:' '.join(map(str,ids)),parse=lambda text,row:None,eos_token_id=31,layout=SMALL,
            token_bound=dict(tokens=len(self.initial[i]),ids_sha256=ids_hash(self.initial[i])))
    def rewrite(self,i,step,change):
        fragment=self.episodes[i]['fragments'][step];path=self.base/fragment['file'];p=json.loads(path.read_bytes());change(p)
        p['complete_batch_context_sha256']=context_hash(p['batch_context']);save_json(path,p);fragment['sha256']=file_hash(path)
        closed=self.base/self.entries[i]['file'];save_json(closed,self.episodes[i]);self.entries[i]['sha256']=file_hash(closed)
    def mutate_state(self,i,step,key,raw):
        fragment=self.episodes[i]['fragments'][step];p=json.loads((self.base/fragment['file']).read_bytes());path=self.base/p['state_file']
        with zipfile.ZipFile(path) as archive:items={name:archive.read(name) for name in archive.namelist()}
        items[key+'.npy']=raw
        with zipfile.ZipFile(path,'w') as archive:
            for name,data in items.items():archive.writestr(name,data)
        self.rewrite(i,step,lambda p:p.update(state_sha256=file_hash(path)))


class IndependentCachedAuditTests(unittest.TestCase):
    def test_full_unequal_widths_token_stops_and_cache_pruning(self):
        with tempfile.TemporaryDirectory() as base:
            f=Fixture(base);audits=[f.audit(0),f.audit(1)];keys=complete_batch_signatures(audits)
            self.assertEqual(sum(a['physical_vectors'] for a in audits),160)
            self.assertEqual(len(keys),4);self.assertTrue(all(p['signature'] for p in keys.values()))
            self.assertTrue(all(a['instrumented_snapshots_verified'] for a in audits))
            self.assertFalse(any(a['full_vocabulary_argmax_independently_reconstructed'] for a in audits))
    def test_missing_companion_excludes_its_cache_descendants(self):
        with tempfile.TemporaryDirectory() as base:
            f=Fixture(base);b=f.audit(1);keys=complete_batch_signatures([b])
            self.assertEqual(len(keys),3);self.assertTrue(all(x['signature'] is None for x in keys.values()))
            self.assertEqual(keys[b['prefixes'][-1]['file']]['reason'],'unverified_parent_cache_context')
    def test_self_consistent_forged_companion_hash_is_not_a_batch_proof(self):
        with tempfile.TemporaryDirectory() as base:
            f=Fixture(base)
            f.rewrite(0,0,lambda p:p['batch_context']['history_sha256'].__setitem__(1,ids_hash([19])))
            audits=[f.audit(0),f.audit(1)];keys=complete_batch_signatures(audits)
            self.assertTrue(all(x['signature'] is None for x in keys.values()))
            self.assertEqual(keys[audits[1]['prefixes'][-1]['file']]['reason'],'unverified_parent_cache_context')
    def test_reset_or_missing_lineage_or_wrong_stop_is_rejected(self):
        changes=[lambda p:p['batch_context'].update(parent_cache_context_sha256='0'*64),
                 lambda p:p['batch_context'].update(physical_cache_length=3),
                 lambda p:p['batch_context'].update(attention_mask_runs=[[[0,3]]]),
                 lambda p:p.update(token=31),lambda p:p.update(actual_length=True),
                 lambda p:p.update(full_logits_independently_reconstructible_from_export=True)]
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as base:
                f=Fixture(base);f.rewrite(1,1,change)
                with self.assertRaises((ValueError,KeyError)):f.audit(1)
    def test_nonfinite_wrong_width_and_native_input_output_mismatch_are_rejected(self):
        corruptions=[('v0',npy((4,),word=0x7f80)),('v1',npy((4,))),
                     ('native_at_reached_input',npy((2,4),word=0x3f00)),('instrument_input_last_token',npy((2,8),word=0x3f00))]
        for key,data in corruptions:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as base:
                f=Fixture(base);f.mutate_state(0,0,key,data)
                with self.assertRaises(ValueError):f.audit(0)
    def test_real_AB_structure_and_replay_pending_are_explicit(self):
        for route in ('component_only','static_CAL_mean','reached_state'):
            with self.subTest(route=route),tempfile.TemporaryDirectory() as base:
                f=Fixture(base,route);a,b=f.audit(0),f.audit(1)
                self.assertTrue(b['BPD_shadow_and_local_write_replay_still_required'])
                self.assertFalse(a['BPD_shadow_and_local_write_replay_still_required'])
                self.assertTrue(all(x['signature'] for x in complete_batch_signatures([a,b]).values()))
    def test_duplicate_journal_row_and_missing_instrument_or_null_CUDA_are_not_admitted(self):
        with tempfile.TemporaryDirectory() as base:
            f=Fixture(base);a=f.audit(0)
            with self.assertRaises(ValueError):complete_batch_signatures([a,a])
            f.rewrite(0,0,lambda p:p.update(cuda_elapsed_ms_for_entire_batch=None))
            with self.assertRaises(ValueError):f.audit(0)
        with tempfile.TemporaryDirectory() as base:
            f=Fixture(base);f.mutate_state(0,0,'actual_masks',npy((2,2),kind='F32',word=1.))
            with self.assertRaises(ValueError):f.audit(0)


if __name__=='__main__':unittest.main()
