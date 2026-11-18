"""Tokenization preflight contracts without any model weights or HF downloads."""
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.nanbeige.nan_comp04 import cohort,prompt,run,preflight_tokenizer


def fake_tokens(_tokenizer,_snapshot,row):
    # One token per character tests visible-text equality/length, not BPE.
    return list(map(ord,prompt.body(row)))


class TestTokenizerPreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=cohort.build()["public"]

    def test_all_source_groups_have_nine_unique_paired_prompts(self):
        with patch.object(run,"encode",side_effect=fake_tokens):
            result=preflight_tokenizer.scan(
                self.rows,None,Path("."),max_prompt=2048)
        self.assertEqual((result["n_roots"],result["n_model_visible_tasks"]), (20,220))
        self.assertEqual(result["unique_rendered_prompts"],180)
        self.assertEqual(len(result["cards"]),20)
        self.assertTrue(all(c["distinct_prompt_hashes"]==9 for c in result["cards"]))
        self.assertFalse(result["pretrained_model_loaded"])
        self.assertFalse(result["heldout_168_accessed"])

    def test_fail_closed_on_insufficient_token_budget(self):
        with patch.object(run,"encode",side_effect=fake_tokens):
            with self.assertRaises(ValueError):
                preflight_tokenizer.scan(
                    self.rows,None,Path("."),max_prompt=10)

    def test_identical_source_early_pairs_are_detected(self):
        with patch.object(run,"encode",side_effect=fake_tokens):
            chosen=run.group_rows(self.rows,"development",1)
            self.assertEqual(len(chosen),1)
            _,eleven=chosen[0]
            data=run.tokenize_root(None,Path("."),eleven,2048)
            early=[row["case_id"] for row in eleven if row["tier"]=="early"]
            erased=[row["case_id"] for row in eleven if row["tier"]=="erased"]
            self.assertEqual(data[early[0]][0],data[early[1]][0])
            self.assertEqual(data[erased[0]][0],data[erased[1]][0])


if __name__=="__main__":
    unittest.main()
