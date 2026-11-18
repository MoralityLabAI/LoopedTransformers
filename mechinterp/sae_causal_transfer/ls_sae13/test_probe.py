"""Small CPU-only LS-SAE13 tests: exact edits, margins, layouts and replay."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from mechinterp.sae_causal_transfer.active_reasoning_v1 import schedule as S
from mechinterp.sae_causal_transfer.ls_sae13 import run_pilot as P


P17 = [int(c) for c in
       "000000010400000000020000000000050407008000300001090000300400200050100000000806000"]
P17_SOL = [int(c) for c in
           "693784512487512936125963874932651487568247391741398625319475268856129743274836159"]


class FakeTRM:
    puzzle_emb_len = 16

    def act_step(self, zh, zl, emb):
        return zh + 0.25 * zl + 0.001 * emb, zl + 0.125 * zh

    def digit_logits(self, zh):
        return zh[:, 16:97, :9]


class LockedEditTests(unittest.TestCase):
    def test_hashes_and_equal_l2(self):
        for pos, site in P.SITES.items():
            v = P.locked_edit(pos, site)
            self.assertEqual(v.dtype, np.dtype("<f4"))
            self.assertEqual(len(v.tobytes()), 2048)
            self.assertAlmostEqual(float(np.linalg.norm(v.astype(np.float64))), 0.02, places=7)
            self.assertTrue(np.array_equal(v, P.locked_edit(pos, site)))
        with self.assertRaises(ValueError):
            P.locked_edit(23, 39)
        with self.assertRaises(ValueError):
            P.locked_edit(99, 40)

    def test_all_batch_layouts(self):
        self.assertEqual(P.layouts(1), [["clean"], ["plus"], ["minus"]])
        self.assertEqual(P.layouts(2), [["clean", "plus"], ["clean", "minus"]])
        for n in (21, 61):
            a = P.layouts(n)
            self.assertEqual(len(a), 1)
            self.assertEqual(len(a[0]), n)
            self.assertEqual(a[0][:3], ["clean", "plus", "minus"])
            self.assertTrue(all(x == "clean" for x in a[0][3:]))
        with self.assertRaises(ValueError):
            P.layouts(4)


class MarginTests(unittest.TestCase):
    def test_sufficient_certificate_excludes_actual_flips(self):
        clean = np.full((81, 9), -2.0, dtype=np.float32)
        clean[:, 0] = 2.0
        edit = clean.copy()
        edit[0, 1] = 3.0
        x = [0] * 81
        m = P.logits_pair(clean, edit, x)
        self.assertEqual(m["actual_blank_argmax_flips"], 1)
        self.assertEqual(m["certified_blank_cells"], 80)
        self.assertEqual(m["uncertified_blank_cells"], 1)
        m0 = P.logits_pair(clean, clean, x)
        self.assertEqual(m0["certified_blank_cells"], 81)
        self.assertEqual(m0["actual_blank_argmax_flips"], 0)

    def test_centering_rejects_common_logit_offset(self):
        clean = np.arange(729, dtype=np.float32).reshape(81, 9)
        shifted = clean + 0.25
        m = P.logits_pair(clean, shifted, [0] * 81)
        self.assertAlmostEqual(m["centered_logit_rms"], 0.0, places=7)
        self.assertAlmostEqual(m["max_abs_logit_delta"], 0.25, places=7)

    def test_nonfinite_rejected(self):
        x = np.zeros((81, 9), dtype=np.float32)
        y = x.copy()
        y[0, 0] = np.nan
        with self.assertRaises(ArithmeticError):
            P.logits_pair(x, y, [0] * 81)


class NativeToyTests(unittest.TestCase):
    def test_complete_four_batch_audit(self):
        torch.set_num_threads(2)
        model = FakeTRM()
        zh = torch.linspace(-0.04, 0.04, 97 * 512, dtype=torch.float32).reshape(1, 97, 512)
        anchor = {"zh": zh, "zl": torch.ones_like(zh) * 0.02,
                  "x_emb": torch.ones_like(zh) * 0.05}
        site = S.control_site(P17)
        edit = P.locked_edit(23, 40)
        original, _ = P.forward_layout(model, anchor, site, edit, ["clean"])
        prefix = [[1] * 81 for _ in range(4)]
        preds = prefix + [P.RN.argmax_digits(r["logits"]) for r in original["clean"]]
        screen_logits = [np.zeros((81, 9), dtype=np.float32).tolist() for _ in range(4)]
        screen_logits += [r["logits"].tolist() for r in original["clean"]]
        output = P.run_batches(model, anchor, P17, P17_SOL, site, edit, preds, screen_logits)
        self.assertEqual(set(output), {"1", "2", "21", "61"})
        for batch in P.BATCHES:
            result = output[str(batch)]
            self.assertIsNone(result["first_clean_digit_divergence_from_batch1"])
            self.assertEqual(len(result["per_step"]), 12)
            self.assertEqual(len(result["trajectories"]["plus"]["predicted_digits_by_act"]), 16)
            self.assertEqual(result["trajectories"]["clean"]["schedule"].keys(),
                             result["trajectories"]["minus"]["schedule"].keys())
            for step in result["per_step"]:
                self.assertEqual(set(step["edits"]), {"plus", "minus"})
                if step["act"] in P.RETAIN_LOGITS_STEPS:
                    self.assertIsNotNone(step["retained_float32_logits"])
                else:
                    self.assertIsNone(step["retained_float32_logits"])
        self.assertTrue(all(x["bit_identical"] for x in output["2"]["baseline_repeat_comparison"]))
        for batch in ("21", "61"):
            self.assertEqual(len(output[batch]["padding_checks"]), 12)
            self.assertTrue(all(p["padding_rows_different"] == 0
                                for p in output[batch]["padding_checks"]))


if __name__ == "__main__":
    unittest.main()
