"""CPU unit and leakage tests for the LR-01 RefutationBench transfer pilot."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import random
import sys
import unittest

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runner import (
    MAX_ATOMS, MAX_EVIDENCE, MAX_RULES, RefutationPolicy, batch_encode,
    load_bench, loss_function, respond, shuffled_case, stable_tokens, train_model,
)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("RB_BENCH_ROOT")
        if not root:
            raise RuntimeError("RB_BENCH_ROOT must be an exact RefutationBench repository checkout")
        cls.bench, cls.train, cls.dev, cls.answers = load_bench(Path(root))
        torch.set_num_threads(2)

    def test_family_split_and_labels(self):
        self.assertEqual((len(self.train), len(self.dev)), (6, 19))
        self.assertFalse({c["family_id"] for c in self.train} &
                         {c["family_id"] for c in self.dev})
        self.assertFalse(any(c["kind"] == "source_audit" for c in self.train))

    def test_shapes_and_finite_forward(self):
        b = batch_encode(self.train, self.answers)
        self.assertEqual(tuple(b["ids"].shape), (6, MAX_ATOMS, 80))
        self.assertEqual(tuple(b["prem"].shape), (6, MAX_RULES, MAX_ATOMS))
        for loops in (1, 4):
            model = RefutationPolicy(loops)
            verdict, flaw, evidence = model(b)
            self.assertEqual(tuple(verdict.shape), (6, 4))
            self.assertEqual(tuple(flaw.shape), (6, 8))
            self.assertEqual(tuple(evidence.shape), (6, MAX_EVIDENCE))
            self.assertTrue(torch.isfinite(verdict).all())
            self.assertTrue(torch.isfinite(loss_function((verdict, flaw, evidence), b)))

    def test_input_only_encoder_ignores_gold_and_metadata(self):
        a = deepcopy(self.train[:1])
        b = deepcopy(a)
        b[0]["case_id"] = "INDEPENDENT_CASE_ID"
        b[0]["split"] = "dev"
        b[0]["family_id"] = "OTHER_FAMILY"
        for e in b[0]["evidence"]:
            e["locator"] = "changed locator"
            e["source_id"] = "changed source"
        enc_a, enc_b = batch_encode(a), batch_encode(b)
        for key in ("ids", "roles", "rule_types", "prem", "head"):
            self.assertTrue(torch.equal(enc_a[key], enc_b[key]), key)
        self.assertTrue(torch.equal(enc_a["verdict_y"], enc_b["verdict_y"]))
        self.assertEqual(int(enc_a["verdict_y"][0]), -1)

    def test_evidence_order_permutation_equivariance(self):
        torch.manual_seed(17)
        model = RefutationPolicy(4).eval()
        case = self.dev[0]
        permuted = shuffled_case(case, random.Random(101))
        x = model(batch_encode([case]))[0]
        y = model(batch_encode([permuted]))[0]
        self.assertTrue(torch.allclose(x, y, atol=2e-5, rtol=2e-5),
                        f"{x.tolist()} != {y.tolist()}")

    def test_training_is_reproducible_without_dev_labels(self):
        model_a, meta_a = train_model(self.train, self.answers, loops=1, seed=17, steps=3)
        model_b, meta_b = train_model(self.train, self.answers, loops=1, seed=17, steps=3)
        self.assertEqual(meta_a["losses"], meta_b["losses"])
        for key in model_a.state_dict():
            self.assertTrue(torch.equal(model_a.state_dict()[key],
                                        model_b.state_dict()[key]), key)
        predictions = respond(model_a, self.dev)
        self.assertEqual(len(predictions), 19)
        for p, case in zip(predictions, self.dev):
            self.assertEqual(p["case_id"], case["case_id"])
            self.assertEqual(self.bench.validate_response(p, case), [])

    def test_tokens_are_stable_and_no_external_tokenizer(self):
        self.assertEqual(stable_tokens("unique factorization in cyclotomic rings"),
                         stable_tokens("unique factorization in cyclotomic rings"))
        self.assertNotEqual(stable_tokens("Wiles patched the argument"),
                            stable_tokens("Kempe recolouring interference"))


if __name__ == "__main__":
    unittest.main()
