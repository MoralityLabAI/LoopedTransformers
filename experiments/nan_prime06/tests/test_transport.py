import json
import unittest
from experiments.nan_prime06.wire import prompt_sha,validate_request,stream_chunks,sse_bytes


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.cfg={'request_max_tokens':192};self.row={'prompt':'registered prompt','schema':{'type':'label'}}
        self.prompts={prompt_sha(self.row['prompt']):self.row}
        self.body={'model':'nanbeige-451ed48c-nf4','messages':[{'role':'user','content':self.row['prompt']}],
            'stream':True,'temperature':0,'max_tokens':192}
    def test_exact_registered_input(self):
        _,row=validate_request(self.body,self.prompts,'nf4',self.cfg);self.assertEqual(row,self.row)
    def test_reject_prompt_system_tool_and_sampling_changes(self):
        cases=[{'messages':[{'role':'user','content':'other'}]}, {'messages':[{'role':'system','content':'secret'}]},
            {'tools':[{}]},{'temperature':0.1},{'max_tokens':16},{'model':'hosted-model'},{'stream':'yes'}]
        for change in cases:
            with self.subTest(change=change),self.assertRaises(ValueError):validate_request(self.body|change,self.prompts,'nf4',self.cfg)
    def test_sse_roundtrip_preserves_whitespace_and_complete_finish(self):
        text='\n [1, 2] \n';usage={'prompt_tokens':3,'completion_tokens':4,'total_tokens':7}
        raw=sse_bytes(stream_chunks('local-id','model',text,usage,'length')).decode()
        chunks=[json.loads(x[6:]) for x in raw.split('\n\n') if x.startswith('data: {')]
        self.assertEqual(chunks[0]['choices'][0]['delta']['content'],text)
        self.assertEqual(chunks[1]['choices'][0]['finish_reason'],'length');self.assertEqual(chunks[2]['usage'],usage)
        self.assertTrue(raw.endswith('data: [DONE]\n\n'))


if __name__=='__main__':unittest.main()
