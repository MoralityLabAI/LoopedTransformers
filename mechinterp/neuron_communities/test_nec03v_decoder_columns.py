"""Pure-standard-library source-site and visitation checks for decoder extraction.

Full original column arithmetic is independently tested by the native CPU
artifact workflow, using the byte-pinned historical checkpoints.
"""
from __future__ import annotations

import unittest

from mechinterp.neuron_communities.nec03v_decoder_columns import original_visits
from mechinterp.neuron_communities.nec03v_decoder_columns_audit import parse_node


class DecoderRosterTests(unittest.TestCase):
    def test_original_A_and_B_roster_forms(self):
        candidate={"nodes":["3:5","7:11"]}
        for label in ("roster","source_roster"):
            roster={label:{"source_candidate":candidate,
                           "source_graph":{"hubs":["3:12","4:20","7:2"]}}}
            result=original_visits(roster,candidate)
            self.assertEqual(result,{3:{5,-13},7:{11,-3}})
            self.assertEqual(parse_node("3:12"),(3,12))

    def test_missing_source_candidate_is_empty(self):
        self.assertEqual(original_visits({},None),{})

    def test_source_roster_identity_is_mandatory(self):
        candidate={"nodes":["3:5"]}
        roster={"roster":{"source_candidate":{"nodes":["4:5"]},
                           "source_graph":{"hubs":[]}}}
        with self.assertRaisesRegex(ValueError,"roster differs"):
            original_visits(roster,candidate)

    def test_duplicate_or_out_of_range_original_source_sites_rejected(self):
        for bad in (["3:5","3:5"],["12:5"],["-1:3"],["3/5"]):
            candidate={"nodes":bad}
            roster={"source_roster":{"source_candidate":candidate,
                                    "source_graph":{"hubs":[]}}}
            with self.assertRaises(ValueError):
                original_visits(roster,candidate)

    def test_protected_world_neuron_node_format_rejected(self):
        for node in ("12:1","0:64","3:x","1:1:1"):
            with self.assertRaises(ValueError):
                parse_node(node)


if __name__=="__main__":
    unittest.main()
