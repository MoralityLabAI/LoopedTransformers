"""Source/control coverage, stop/binding failures and a real uint16 archive fixture."""
import ast
import json
import os
from pathlib import Path
import unittest
import uuid
from .common import ROOT,Guard,Handle,select,sha,read,write,no_torch
from .audit import WIDTHS,digest,check_episode,check_pair_shapes,grade
from .run import aggregate,roster,ENDPOINTS


class Tests(unittest.TestCase):
    def test_source_order_and_controls(self):
        from experiments.nan_full26.common import load_sources
        public=roster();self.assertEqual([r['id'] for r in public],[r['id'] for r in load_sources()[0][:72]])
        self.assertEqual(len(public),72);self.assertEqual(len({r['root'] for r in public}),12)
        self.assertEqual({r['arm'] for r in public},{'main','copy','erased'})

    def test_selector_partial_and_duplicate(self):
        entries=[dict(file='baseline_0000_r0_closed.json',sha256='x'),dict(file='baseline_0000_r1_closed.json',sha256='y'),
                 dict(file='AB_0000_r0_closed.json',sha256='z'),dict(file='baseline_0072_r0_closed.json',sha256='z')]
        self.assertEqual([v['slot'] for v in select(entries,list(range(72)))],[0,1])
        with self.assertRaises(AssertionError):select(entries+[entries[0]],list(range(72)))
        with self.assertRaises(AssertionError):select([dict(file='baseline_0001_r0_closed.json')],list(range(72)))

    def test_original_story_endpoint_AST(self):
        original=ast.parse(Path('experiments/nan_mlp27/host.py').read_text(encoding='utf-8'))
        branch=next(v for v in ast.walk(original) if isinstance(v,ast.If) and ast.unparse(v.test)=="r['stage'] == 'nan_story23'")
        current=next(v for v in ast.parse((ROOT/'audit.py').read_text(encoding='utf-8')).body if isinstance(v,ast.FunctionDef) and v.name=='grade')
        assignments={v.targets[0].id:v for v in current.body if isinstance(v,ast.Assign) and isinstance(v.targets[0],ast.Name)}
        for v in branch.body[:3]:self.assertEqual(ast.dump(v),ast.dump(assignments[v.targets[0].id]))
        self.assertEqual(ast.dump(branch.body[3].value),ast.dump(current.body[-1].value))

    def test_whole_roster_bounds(self):
        public=roster();empty=aggregate(public,[])
        self.assertEqual(sum(g['missing_worlds'] for g in empty),72)
        for g in empty:
            for e in ENDPOINTS:self.assertEqual(g['endpoints'][e]['whole_roster_correct_bounds'],[0,g['registered_worlds']])
        r=public[0];record=dict(source_id=r['id'],grade={e:True for e in ENDPOINTS})
        partial=aggregate(public,[record]);self.assertEqual(sum(g['observed_worlds'] for g in partial),1)
        with self.assertRaises(AssertionError):aggregate(public,[record,record])

    def test_memory_boundaries_and_model_exit(self):
        g=Guard();g.cfg=dict(maximum_private_bytes=100,minimum_available_RAM_bytes=200);g.samples=[]
        g.memory=lambda:dict(private_bytes=100,available_physical_bytes=200);g.check()
        for private,available in ((101,200),(100,199)):
            g.memory=lambda:dict(private_bytes=private,available_physical_bytes=available)
            with self.assertRaises(AssertionError):g.check()
        g.memory=lambda:dict(private_bytes=100,available_physical_bytes=200)
        class Ended:
            def alive(self):return False
        g.model=Ended()
        with self.assertRaises(AssertionError):g.check()

    @unittest.skipUnless(os.name=='nt','Windows handle contract')
    def test_actual_handle_creation(self):
        h=Handle(os.getpid())
        try:
            self.assertTrue(h.alive())
            with self.assertRaises(RuntimeError):Handle(os.getpid(),h.creation+1)
        finally:h.close()

    def test_pair_padding_rejects_serial_substitute(self):
        a=dict(slot=0,prefixes=1,batch_shapes=[(2,0,8,9,1)])
        b=dict(slot=1,prefixes=2,batch_shapes=[(2,1,9,9,1),(1,0,10,10,0)])
        check_pair_shapes([a,b]);b['batch_shapes'][0]=(1,0,9,9,0)
        with self.assertRaises(AssertionError):check_pair_shapes([a,b])

    def test_uint16_episode_and_corruption(self):
        try:import numpy as np
        except ImportError:self.skipTest('NumPy required; explicitly run with existing numerical environment')
        folder=Path('runs/nan_storyaudit30_synthetic')/uuid.uuid4().hex;folder.mkdir(parents=True)
        try:
            r=dict(id='nan_story23/mock',root='MOCK_ONLY',world='A',split='calibration',stage='nan_story23',arm='main',query_type='actual',event_count=32,
                   schema=dict(checkpoints=[1,2,3],nodes=['R0','P0']))
            gold=dict(final_room='R0',intermediate_rooms=['R0']*3,path=['P0','R0'])
            class Tokenizer:
                eos_token_id=1
                def decode(self,ids,**kwargs):return '' if not ids else 'bad'
            state={};index=[]
            for phase in (1,2):
                for layer in (5,11,17,21):
                    for site,width in WIDTHS.items():
                        key='v'+str(len(index));state[key]=np.zeros(width,dtype=np.uint16);index.append(dict(loop=phase,layer=layer,site=site,key=key))
            state.update(boundaries=np.zeros((2,3072),dtype=np.uint16),native_at_reached_input=np.zeros((2,3072),dtype=np.uint16))
            archive=folder/'baseline_0000_r0_p000.npz'
            with archive.open('xb') as f:np.savez(f,**state)
            prefix=dict(step=0,source_id=r['id'],mode='native',token=1,history_sha256=digest([]),prefix_ids_sha256=digest([5]),
                actual_length=1,padded_length=1,batch_size=1,batch_row=0,padding_tokens=0,full_vocabulary_size=166144,
                top_logit=0.,log_normalizer=1.,cuda_elapsed_ms_for_entire_batch=1.,wall_seconds_for_entire_batch=.001,
                state_file=archive.name,state_sha256=sha(archive),state_index=index)
            file=folder/'baseline_0000_r0_p000.json';write(file,prefix)
            record=dict(source_id=r['id'],root=r['root'],stage=r['stage'],split=r['split'],mode='native',lane='baseline',
                initial_ids=[5],generated_ids=[1],prefix_count=1,text='',stop_reason='eos',fragments=[dict(file=file.name,sha256=sha(file))])
            closed=folder/'baseline_0000_r0_closed.json';write(closed,record)
            entry=dict(slot=0,file=closed.name,sha256=sha(closed));bound=dict(tokens=1,ids_sha256=digest([5]))
            result=check_episode(folder,entry,r,gold,Tokenizer(),[5],bound)
            self.assertEqual(result['physical_vectors'],40);self.assertFalse(result['grade']['joint_correct'])
            prefix['history_sha256']='corrupted'
            file.write_text(json.dumps(prefix),encoding='utf-8');record['fragments'][0]['sha256']=sha(file)
            closed.write_text(json.dumps(record),encoding='utf-8');entry['sha256']=sha(closed)
            with self.assertRaises(AssertionError):check_episode(folder,entry,r,gold,Tokenizer(),[5],bound)
            no_torch()
        finally:
            for path in folder.iterdir():path.unlink()
            folder.rmdir()


if __name__=='__main__':unittest.main()
