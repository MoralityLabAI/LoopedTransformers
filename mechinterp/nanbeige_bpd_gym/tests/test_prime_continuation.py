import unittest
from mechinterp.nanbeige_bpd_gym.prime_continuation import unfinished


class ContinuationSelectionTests(unittest.TestCase):
    def test_selection_is_independent_of_reward_and_format(self):
        public=[dict(id=str(i),split='development',prompt=str(i)) for i in range(4)]
        def episode(i,**changes):
            trace=dict(ok=True,is_completed=True,is_timeout=False,calls=[{}])
            trace.update(changes)
            return dict(task=dict(data=public[i]),ok=True,traces=[trace],reward=0,format_valid=False)
        self.assertEqual(unfinished(public,[episode(0),episode(1,is_timeout=True),episode(2,is_completed=False)]),['1','2','3'])
        bad=episode(0);bad['task']['data']=dict(public[0],prompt='changed')
        with self.assertRaises(ValueError):unfinished(public,[bad])


if __name__=='__main__':unittest.main()
