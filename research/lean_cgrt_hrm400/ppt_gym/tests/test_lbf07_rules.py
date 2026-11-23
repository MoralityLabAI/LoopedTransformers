"""LBF-07: exact-literal and causal-control contracts for typed rule mapping."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import tempfile
import unittest

import torch

from ppt_gym.data import canonical, sha256_file
from ppt_gym.lbf_hybrid import SCHEMAS
from ppt_gym.native_train import fit
from ppt_gym.tests.test_native_train import fake_teacher
from ppt_gym.typed_state import VIEWS, encode_public_packet

V = ("hybrid_rule_intact", "hybrid_rule_rebind", "hybrid_rule_order")
RULE = re.compile(r"([hr][0-9]{1,4}):([^;]*);")


def packet():
    atom = lambda n: {"a": f"P{n}"}
    imp = lambda a,b: {"imp": [atom(a),atom(b)]}
    return canonical({
        "state": {"goals": [{
            "target": atom(4),
            "hypotheses": [
                {"name": "P0", "type": {"sort": "Prop"}},
                {"name": "P1", "type": {"sort": "Prop"}},
                {"name": "P2", "type": {"sort": "Prop"}},
                {"name": "h12", "type": imp(0,4)},
                {"name": "r24", "type": imp(1,4)},
                {"name": "r56", "type": imp(2,3)},
            ],
            "pp": "h12 : P0 -> P4; r24 : P1 -> P4; r56 : P2 -> P3; ⊢ P4",
        }]},
        "failed_choices": [],
        "checks_remaining": 8,
    })


def rule_rows(text):
    return RULE.findall(text.split("P:",1)[0])


class TypedRuleCausalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.public=packet()
        self.menu=["apply h12","apply r24"]

    def test_distinct_effects_same_character_multiset(self):
        outputs={v:encode_public_packet(self.public,self.menu,view=v,max_bytes=384)
                 for v in V}
        values=list(outputs.values())
        self.assertEqual(len(set(values)),3)
        self.assertEqual(len({len(x.encode("utf-8")) for x in values}),1)
        self.assertLessEqual(len(values[0].encode("utf-8")),384)
        self.assertTrue(all(Counter(x)==Counter(values[0]) for x in values))
        self.assertTrue(all(x.startswith("B8|G:P4|") for x in values))
        self.assertEqual(outputs[V[0]],encode_public_packet(self.public,self.menu,view="hybrid_nopp"))
        original=rule_rows(outputs[V[0]])
        rebound=rule_rows(outputs[V[1]])
        ordered=rule_rows(outputs[V[2]])
        self.assertEqual([x[0] for x in original],[x[0] for x in rebound])
        self.assertEqual(sorted(original),sorted(ordered))
        self.assertEqual([x[0] for x in original],[x[0] for x in original])
        self.assertNotEqual(dict(original),dict(rebound))
        self.assertEqual(dict(original),dict(ordered))
        self.assertEqual(sorted(x[1] for x in original),sorted(x[1] for x in rebound))

    def test_determinism_and_public_only_fingerprint(self):
        a=encode_public_packet(self.public,self.menu,view=V[1])
        b=encode_public_packet(self.public,self.menu,view=V[1])
        self.assertEqual(a,b)
        c=json.loads(self.public)
        c["state"]["goals"][0]["hypotheses"][0]["type"]["sort"]="Type"
        # Changing the public context changes the source hash or raises
        # under strict grammar. Both are acceptable protocol outcomes.
        with self.assertRaises(ValueError):
            encode_public_packet(canonical(c),self.menu,view=V[1])
        for v in V:
            self.assertIn(v,VIEWS)
            self.assertTrue(SCHEMAS[v].startswith("lbf07."))

    def test_labels_and_fake_proof_receipts_rejected(self):
        fake=json.loads(self.public)
        fake["native_certificate"]="passed"
        for v in V:
            with self.assertRaises(ValueError):
                encode_public_packet(canonical(fake),self.menu,view=v)
        fake=json.loads(self.public)
        fake["state"]["goals"][0]["verifier_outcome"]=1
        for v in V:
            with self.assertRaises(ValueError):
                encode_public_packet(canonical(fake),self.menu,view=v)

    def test_single_clause_registered_as_noop(self):
        packet1=json.loads(self.public)
        packet1["state"]["goals"][0]["hypotheses"]=[
            x for x in packet1["state"]["goals"][0]["hypotheses"]
            if x["name"] in ("P0","h12")
        ]
        for v in V:
            rendered=encode_public_packet(canonical(packet1),["apply h12"],
                                         view=v,max_bytes=384)
            self.assertEqual(rendered,encode_public_packet(canonical(packet1),["apply h12"],
                                         view=V[0],max_bytes=384))

    def test_matched_initial_checkpoint_and_source_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            teacher=fake_teacher(root)
            source=teacher/"teacher.jsonl"
            frames=[json.loads(l) for l in source.read_text().splitlines() if l]
            for row in frames:
                row["goal"]=self.public
                row["candidates"]=list(self.menu)
                row["labels"]=[1,0]
            source.write_text("".join(canonical(r)+"\n" for r in frames),encoding="utf-8")
            manifest=teacher/"manifest.json"
            meta=json.loads(manifest.read_text())
            meta["artifacts"]["teacher.jsonl"]=sha256_file(source)
            manifest.write_text(canonical(meta)+"\n",encoding="utf-8")
            fitted=[]
            for v in V:
                m=fit(teacher,root/v,order="ab",seed=107,steps=2,
                      depth=2,width=16,max_bytes=384,threads=1,
                      max_seconds=40,input_view=v,allow_synthetic=True)
                self.assertEqual(m["optimizer_steps"],2)
                checkpoint=torch.load(root/v/"checkpoint.pt",map_location="cpu",weights_only=True)
                self.assertEqual(checkpoint["input_view_schema"],SCHEMAS[v])
                self.assertEqual(checkpoint["input_view"],v)
                fitted.append(m)
            self.assertEqual(len({f["initial_weights_sha256"] for f in fitted}),1)
            self.assertEqual(len({f["unique_parameter_count"] for f in fitted}),1)


if __name__ == "__main__":
    unittest.main()
