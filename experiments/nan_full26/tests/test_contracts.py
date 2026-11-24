import json
from pathlib import Path
import unittest
from experiments.nan_full26.common import load_sources,intervention_rows,parse,prompt

class Contracts(unittest.TestCase):
    def test_full_source_coverage(self):
        if not Path('runs/nan_gym05/cohort/public.jsonl').exists():self.skipTest('32 private routing rows stay local; full1549 coverage is verified locally before freeze')
        public,pins=load_sources();self.assertEqual(len(public),1549)
        from collections import Counter
        self.assertEqual(Counter(r['stage'] for r in public),dict(nan_gym05=322,nan_comp14=240,nan_ctx08=291,nan_ctx21=624,nan_story23=72))
        self.assertEqual(len(intervention_rows(public)),72)
        self.assertEqual(sum(r['stage']=='nan_story23' for r in intervention_rows(public)),24)
        self.assertEqual(sum(r['stage']=='nan_comp14' for r in intervention_rows(public)),48)
        self.assertTrue(all('test' not in p.lower() for p in pins))
    def test_strict_packet(self):
        r=dict(stage='nan_comp14',schema=dict(type='label',allowed=list('AB'),max_characters=64))
        packet='{"trace":["A goes to B"],"answer":"B"};END'
        self.assertEqual(parse(packet,r)['answer'],'B')
        for value in [packet+' prose','prose '+packet,packet[:-4],packet.replace('"B"','"Z"'),
            '{"trace":[],"answer":"B"};END','{"trace":["a"],"answer":"B","extra":0};END',
            '{"trace":["a"],"answer":"A","answer":"B"};END',
            '{"trace":["'+('x'*121)+'"],"answer":"B"};END']:
            self.assertIsNone(parse(value,r),value)
    def test_array_answer(self):
        r=dict(stage='nan_gym05',schema=dict(type='grid',max_characters=2048))
        self.assertEqual(parse('{"trace":["copy"],"answer":[[1,2],[3,4]]};END',r)['answer'],[[1,2],[3,4]])
    def test_replaced_instruction_explicit(self):
        text=prompt('Reply only A.');self.assertIn('are replaced by this protocol',text)
        self.assertIn('TASK BEGIN\nReply only A.\nTASK END',text)
    def test_worker_no_host_oracle(self):
        text=Path('experiments/nan_full26/worker.py').read_text()
        self.assertNotIn('host_labels',text);self.assertNotIn('sealed_test/',text)
        self.assertNotIn('resume',text)
    def test_batch_is_not_serial_rows(self):
        text=Path('experiments/nan_full26/batch.py').read_text()
        body=text.split('def run_pairs(self,items):',1)[1].split('loop.run_pairs=',1)[0]
        self.assertNotIn('for b',body);self.assertIn('h=layer(h,',body)
    def test_capacity_only(self):
        cfg=json.loads(Path('experiments/nan_full26/config.json').read_text())
        self.assertEqual(cfg['AB_routes'],['component_only','static_CAL_mean','reached_state'])
        self.assertEqual(cfg['neural_retries'],0);self.assertEqual(cfg['formal_inferential_comparisons'],0)
    def test_engineering_source_bindings(self):
        from experiments.nan_full26.common import ROOT,read,sha
        for name in ('engineering_receipt.json','generation_receipt.json'):
            r=read(ROOT/name);self.assertEqual(r['status'],'passed');self.assertEqual(r['pretrained_model_calls'],0)
            for file,h in r['source_sha256'].items():self.assertEqual(sha(ROOT/file),h)

if __name__=='__main__':unittest.main()
