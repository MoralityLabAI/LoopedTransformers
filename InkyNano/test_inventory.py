import json
import struct
import unittest
from inventory import header, audit


class InventoryTests(unittest.TestCase):
    def test_header_read(self):
        raw = json.dumps({"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}).encode()
        data = struct.pack("<Q",len(raw))+raw+b"abcd"
        h, n, sha = header("test",lambda name,start,size:(data[start:start+size],len(data)))
        self.assertEqual(n,4)
        self.assertEqual(h["x"]["shape"],[4])
        self.assertEqual(len(sha),64)

    def test_reject_huge_header(self):
        with self.assertRaises(ValueError):
            header("test",lambda name,start,size:(struct.pack("<Q",2**40),2**41))

    def test_full_synthetic_index(self):
        cfg={"text_config":{"hidden_size":2304,"intermediate_size":1152,"num_hidden_layers":24,
                            "n_routed_experts":128,"num_experts_per_tok":8,
                            "loop_config":{"core_start":6,"core_end":18,"num_passes":3}}}
        h={}
        pos=0
        for l in range(24):
            for p in ("w13","w2"):
                for suffix in ("","_scale"):
                    name=f"model.llm.layers.{l}.mlp.experts.{p}_weight{suffix}"
                    h[name]={"shape":[128,1],"dtype":"U8","data_offsets":[pos,pos+128]}
                    pos+=128
        index={"weight_map":{k:"s" for k in h}}
        result=audit(cfg,index,{"s":(h,pos,"test")})
        self.assertEqual(len(result["expert_groups"]),48)
        self.assertEqual(result["bucket_bytes"]["expert_bank"],pos)
        index["weight_map"].pop(next(iter(h)))
        with self.assertRaises(ValueError):
            audit(cfg,index,{"s":(h,pos,"test")})


if __name__ == "__main__":
    unittest.main()
