"""Scripted CPU trajectory fixtures; no model answers or competence evidence."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import torch
from mechinterp.nanbeige_bpd_gym.chunked_gym import CachedGym,digest_ids,batch_signature


class ScriptedCache:
    def __init__(self):
        self.loop=SimpleNamespace(device=torch.device('cpu'));self.observer=SimpleNamespace(layers=(5,11,17,21),values={})
        self.instrument=None;self.reset()
    def reset(self):self.physical_length=0;self.semantic_lengths=[];self.modes=();self.markers=[]
    def cache_bytes(self):return self.physical_length*len(self.markers)*32
    def value(self):
        boundary=torch.tensor([[marker,self.physical_length,0,0] for marker in self.markers],dtype=torch.bfloat16)
        sites=('residual_after_layer','down_proj_input','down_proj_output','o_proj_input','o_proj_output')
        states={(row,phase,layer,site):boundary[row].clone() for row in range(len(self.markers))
                for phase in (1,2) for layer in (5,11,17,21) for site in sites}
        return dict(boundaries=[boundary.clone(),boundary.clone()],states=states)
    def prefill(self,sequences,*,modes):
        self.semantic_lengths=list(map(len,sequences));self.physical_length=max(self.semantic_lengths);self.markers=[x[0] for x in sequences]
        self.modes=tuple(modes);self.attention=torch.zeros(len(sequences),self.physical_length,dtype=torch.long)
        for row,n in enumerate(self.semantic_lengths):self.attention[row,:n]=1
        return self.value()
    def select_rows(self,indices):
        self.markers=[self.markers[i] for i in indices];self.semantic_lengths=[self.semantic_lengths[i] for i in indices]
        self.modes=tuple(self.modes[i] for i in indices);self.attention=self.attention[indices]
    def step(self,tokens):
        self.physical_length+=1;self.semantic_lengths=[n+1 for n in self.semantic_lengths]
        self.attention=torch.cat([self.attention,torch.ones(len(tokens),1,dtype=torch.long)],dim=1);return self.value()


def scripted_head(boundary):
    logits=torch.zeros(len(boundary),32)
    for row,(marker,physical,*_) in enumerate(boundary.tolist()):logits[row,31 if marker==1 or physical==6 else 7]=10
    return logits


class CachedGymTests(unittest.TestCase):
    def rows(self):return [dict(id='SCRIPTED_A',root='ROOT_A',stage='nan_story23',split='DEV',schema=dict(type='story'),ids=[1,3,5,7]),
                           dict(id='SCRIPTED_B',root='ROOT_B',stage='nan_story23',split='DEV',schema=dict(type='story'),ids=[2,4])]
    def engine(self,out,*,admitted=lambda b,p:True,parser=lambda text,row:None,budget=lambda start,n:None):
        return CachedGym(ScriptedCache(),scripted_head,out,ids=lambda row:row['ids'],decode=lambda ids:' '.join(map(str,ids)),
                         parse=parser,eos_token_id=31,budget=budget,admitted=admitted)
    def test_full_vocab_greedy_pruning_holes_and_immutable_closed_exports(self):
        import numpy as np
        from experiments.nan_ram28.common import sha
        with tempfile.TemporaryDirectory() as directory:
            sizes=[];engine=self.engine(directory,budget=lambda start,n:sizes.append(n));base=Path(directory)
            entries=engine.generate(self.rows(),['native','native'],0,'baseline');self.assertEqual(len(entries),2)
            episodes=[json.loads((base/entry['file']).read_bytes()) for entry in entries]
            self.assertEqual([row['generated_ids'] for row in episodes],[[31],[7,7,31]])
            self.assertEqual([row['text'] for row in episodes],['','7 7']);self.assertEqual([row['stop_reason'] for row in episodes],['eos','eos'])
            contexts=[]
            for episode in episodes:
                for fragment in episode['fragments']:
                    path=base/fragment['file'];self.assertEqual(sha(path),fragment['sha256']);record=json.loads(path.read_bytes())
                    self.assertEqual(record['full_vocabulary_size'],32);self.assertEqual(len(record['state_index']),40)
                    self.assertEqual(record['complete_batch_context_sha256'],batch_signature(record))
                    self.assertEqual(sha(base/record['state_file']),record['state_sha256'])
                    with np.load(base/record['state_file'],allow_pickle=False) as arrays:
                        self.assertEqual(len(arrays.files),41);self.assertTrue(all(arrays[k].dtype==np.uint16 for k in arrays.files))
                    contexts.append(record['batch_context'])
            tail=contexts[-2:];self.assertEqual([x['physical_cache_length'] for x in tail],[5,6])
            self.assertEqual([x['semantic_lengths'] for x in tail],[[3],[4]])
            self.assertEqual(tail[-1]['attention_mask'],[[1,1,0,0,1,1]])
            self.assertEqual(tail[-1]['attention_mask_sha256'],digest_ids([[1,1,0,0,1,1]]))
            self.assertEqual(tail[-1]['source_ids'],['SCRIPTED_B']);self.assertEqual(tail[-1]['original_row_indices'],[1])
            self.assertEqual(sizes[0],0);self.assertGreater(sizes[-1],sizes[1])
            with self.assertRaises(FileExistsError):engine.generate(self.rows(),['native','native'],0,'baseline')
    def test_companion_history_and_padding_are_required_for_geometry_matching(self):
        import copy
        with tempfile.TemporaryDirectory() as directory:
            engine=self.engine(directory);engine.generate(self.rows(),['native','native'],0,'baseline')
            record=json.loads((Path(directory)/'baseline_0000_r0_p000.json').read_bytes())
            different=copy.deepcopy(record);different['batch_context']['history_sha256'][1]=digest_ids([13])
            self.assertNotEqual(batch_signature(record),batch_signature(different))
            changed=copy.deepcopy(record);changed['batch_context']['attention_mask'][1]=[1,0,1,0]
            with self.assertRaises(ValueError):batch_signature(changed)
            missing=copy.deepcopy(record);missing['batch_context']['source_ids'].pop()
            with self.assertRaises(ValueError):batch_signature(missing)
    def test_parser_stop_and_capacity_or_resource_failure_do_not_promote_fragments(self):
        with tempfile.TemporaryDirectory() as directory:
            engine=self.engine(directory,parser=lambda text,row:dict(done=True))
            entries=engine.generate(self.rows()[1:],['native'],0,'baseline')
            self.assertEqual(json.loads((Path(directory)/entries[0]['file']).read_bytes())['stop_reason'],'complete_packet')
        for which in ('capacity','budget'):
            with tempfile.TemporaryDirectory() as directory:
                def fail(start,n):raise RuntimeError('scripted memory stop')
                engine=self.engine(directory,admitted=lambda b,p:False if which=='capacity' else True,
                                   budget=fail if which=='budget' else lambda start,n:None)
                with self.assertRaises(RuntimeError):engine.generate(self.rows(),['native','native'],0,'baseline')
                self.assertFalse((Path(directory)/'closed.jsonl').exists());self.assertEqual(engine.cache.physical_length,0)
        with tempfile.TemporaryDirectory() as directory:
            engine=self.engine(directory,admitted=lambda b,p:p<=4)
            with self.assertRaisesRegex(RuntimeError,'physical cache'):engine.generate(self.rows(),['native','native'],0,'baseline')
            self.assertTrue((Path(directory)/'baseline_0000_r0_p000.json').exists())
            self.assertTrue((Path(directory)/'baseline_0000_r1_p000.json').exists())
            self.assertFalse((Path(directory)/'closed.jsonl').exists())


if __name__=='__main__':unittest.main()
