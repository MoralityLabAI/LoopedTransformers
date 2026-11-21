import unittest
from mechinterp.nanbeige_bpd_gym.prime_evidence import raw_reply_bound,successful_episode


class RawSDKBindingTests(unittest.TestCase):
    def test_zero_reward_success_is_completed_but_timeout_is_not(self):
        episode=dict(ok=True,is_timeout=False,traces=[dict(ok=True,is_completed=True,is_timeout=False,errors=[],reward=0.)])
        self.assertTrue(successful_episode(episode))
        episode['traces'][0]['is_timeout']=True;self.assertFalse(successful_episode(episode))
        episode['traces'][0]['is_timeout']=False;episode['ok']=False;self.assertFalse(successful_episode(episode))

    def test_requires_exact_raw_node_and_only_sdk_outer_whitespace_trim(self):
        record=dict(mock_only=False,text='\n[[1,2]] ')
        trace=dict(last_reply='[[1,2]]',nodes=[dict(message=dict(role='assistant',content=record['text']))])
        self.assertTrue(raw_reply_bound(record,trace))
        trace['nodes'][0]['message']['content']='[[1,2]]'
        self.assertFalse(raw_reply_bound(record,trace))
        trace['nodes'][0]['message']['content']=record['text'];trace['last_reply']='[[1,3]]'
        self.assertFalse(raw_reply_bound(record,trace))
        record['mock_only']=True;trace['last_reply']='[[1,2]]'
        self.assertFalse(raw_reply_bound(record,trace))


if __name__=='__main__':unittest.main()
