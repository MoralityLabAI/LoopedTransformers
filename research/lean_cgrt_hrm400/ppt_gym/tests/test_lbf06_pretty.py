"""LBF-06 exact byte-custody and public-only pretty-context controls."""
from __future__ import annotations

import json
import unittest

from ppt_gym.data import canonical
from ppt_gym.live_lean import candidate_menu, state_input
from ppt_gym.tests.test_live_lean import FakeLean
from ppt_gym.typed_state import encode_public_packet, VIEWS
from ppt_gym.lbf_hybrid import SCHEMAS


class PrettyCausalControls(unittest.TestCase):
    def setUp(self):
        fake=FakeLean()
        initial=fake.init("dummy")
        raw,_=state_input(fake,initial,(),8)
        packet=json.loads(raw)
        packet["state"]["goals"][0]["pp"] = (
            "h0 : P0   r0 : P0 → P1   ⊢ P1   "
            "an additional ∀ Unicode contextual trace 𝜙 α β δ")
        self.packet=canonical(packet)
        self.menu=candidate_menu(fake,initial)

    def test_exact_byte_count_and_structure(self):
        results={mode:encode_public_packet(self.packet,self.menu,view=mode,max_bytes=384)
                 for mode in ("hybrid","hybrid_nopp","hybrid_shuffle")}
        self.assertEqual(len({len(x.encode('utf-8')) for x in results.values()}),1)
        self.assertEqual(len(set(results.values())),3)
        self.assertTrue(all('r0:P0>P1;' in x for x in results.values()))
        self.assertTrue(all(x.startswith("B8|G:P1|") for x in results.values()))
        self.assertLessEqual(len(results["hybrid"].encode('utf-8')),384)

    def test_exact_sham_bag_of_unicode_characters(self):
        first=encode_public_packet(self.packet,self.menu,view="hybrid")
        shuffled=encode_public_packet(self.packet,self.menu,view="hybrid_shuffle")
        p1=first.split("P:",1)[1].rsplit("|",1)[0]
        p2=shuffled.split("P:",1)[1].rsplit("|",1)[0]
        self.assertEqual(sorted(p1),sorted(p2))
        self.assertEqual(len(p1.encode("utf-8")),len(p2.encode("utf-8")))
        self.assertNotEqual(p1,p2)
        self.assertEqual(shuffled,encode_public_packet(self.packet,self.menu,view="hybrid_shuffle"))

    def test_paired_source_change_without_hidden_labels(self):
        x=json.loads(self.packet)
        x["future_proof"]={"winner":"apply r0"}
        for mode in ("hybrid","hybrid_nopp","hybrid_shuffle"):
            with self.assertRaises(ValueError):
                encode_public_packet(canonical(x),self.menu,view=mode)

    def test_variant_schema_and_fixed_legacy_hybrid(self):
        self.assertEqual(VIEWS[:3],("raw","typed","hybrid"))
        self.assertEqual(SCHEMAS["hybrid"],"lbf05.hybrid_public_state.v1")
        self.assertNotEqual(SCHEMAS["hybrid_shuffle"],SCHEMAS["hybrid_nopp"])
        self.assertEqual(
            encode_public_packet(self.packet,self.menu,view="hybrid"),
            encode_public_packet(self.packet,self.menu,view="hybrid"))
        

if __name__=="__main__":
    unittest.main()
