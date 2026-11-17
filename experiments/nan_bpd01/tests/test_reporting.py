import json
from pathlib import Path
import tempfile
import unittest
from experiments.nan_bpd01.analyze import memory,ranks
from experiments.nan_bpd01.verify_sources import execute

class ReportingTests(unittest.TestCase):
    def test_rank_ties_have_average_rank(self):
        self.assertEqual(ranks([4,2,2,9]).tolist(),[2,0.5,0.5,3])
    def test_memory_measures_are_separate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            p=Path(t)
            (p/'receipt.json').write_text(json.dumps({'peak_cuda_allocated_bytes':11,'peak_cuda_reserved_bytes':29,'elapsed_seconds':7,'head_remained_cpu':True}))
            (p/'memory_samples.json').write_text(json.dumps([{'driver_device_global_mib':[100,4096]}]))
            (p/'external_memory_samples.json').write_text(json.dumps([{'gpu_memory':[{'DedicatedUsage':23,'SharedUsage':8},{'DedicatedUsage':2,'SharedUsage':3}]}]))
            r=memory(p);self.assertEqual([r[k] for k in ['cuda_allocated_bytes','cuda_reserved_bytes','sampled_process_dedicated_bytes','sampled_process_shared_bytes']],[11,29,25,11])
    def test_independent_formal_inverse(self):
        f=[1,1,0,0,0,0];identity=[1,0,0,0,0,0]
        self.assertEqual(execute(f,'fork',['I','R','I']),identity)
        self.assertEqual(execute(f,'cascade',['N','I','S']),[1,0,-2,0,1,0])

if __name__=='__main__':unittest.main()
