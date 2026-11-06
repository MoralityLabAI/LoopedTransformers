"""Tests for the v2 gym (lcgrt). Lean-dependent tests skip when the pinned toolchain or
the built lgym server is missing; everything else runs on CPU in seconds.

    python -m unittest discover -s tests -v
"""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from lcgrt import config
from lcgrt import fragment as F

HAVE_LEAN = config.LEAN_EXE.exists() and config.REPL_EXE.exists()


class GeneratorTests(unittest.TestCase):
    def test_goals_provable_and_bounded(self):
        for fam in F.FAMILIES:
            for d in (2, 5, 9):
                for s in range(5):
                    t = F.generate(s, d, fam)
                    size = F.reference_sizes(t)
                    self.assertTrue(all(size[g] < F.INF for g in t.goal))
                    self.assertLessEqual(max(len(p) for p, _ in t.rules), 2)
        t = F.generate(1, 4, "chain-dead", arity3=True)
        self.assertLessEqual(max(len(p) for p, _ in t.rules), 3)

    def test_decoys_create_real_choices(self):
        t = F.generate(3, 6, "chain-deep")
        heads = [h for _, h in t.rules]
        self.assertTrue(any(heads.count(h) > 1 for h in t.gen["solution_atoms"]))

    def test_canonical_hash_ignores_names_and_order(self):
        t = F.generate(7, 4, "tree-cycle")
        perm = list(range(t.n_atoms)); random.Random(0).shuffle(perm)
        u = F.Theorem(t.tid, t.family, t.depth, t.seed, t.n_atoms, [perm[a] for a in reversed(t.facts)],
                      [[[perm[p] for p in pre], perm[h]] for pre, h in reversed(t.rules)],
                      [perm[g] for g in t.goal])
        self.assertEqual(F.canonical_hash(t), F.canonical_hash(u))
        self.assertNotEqual(F.canonical_hash(t), F.canonical_hash(F.generate(8, 4, "tree-cycle")))


@unittest.skipUnless(HAVE_LEAN, "Lean toolchain or lgym not built")
class CertifierTests(unittest.TestCase):
    def setUp(self):
        from lcgrt.extract import explore
        from lcgrt.lean_repl import LeanRepl
        self.thm = F.generate(11, 3, "chain-dead")
        with LeanRepl() as repl:
            self.rec = explore(repl, self.thm)

    def run_cert(self, script, thm=None):
        from lcgrt.certify import certify, decl_name
        r = certify([("x", thm or self.thm, script)])
        return r["status"][decl_name("x")], r

    def test_good_proof_passes(self):
        st, r = self.run_cert(self.rec["proof"]["script"])
        self.assertEqual(st, "passed", r["output_tail"])

    def test_forbidden_tactics_rejected_before_lean(self):
        for bad in (["sorry"], ["admit"], ["native_decide"], ["exact?"], ["apply h0 <;> sorry"],
                    ["set_option debug.skipKernelTC true in apply h0"], ["apply foo"]):
            st, r = self.run_cert(bad)
            self.assertEqual(st, "rejected", bad)
            self.assertIn("t_x", r["whitelist_rejections"])

    def test_incomplete_replay_rejected(self):
        st, _ = self.run_cert(self.rec["proof"]["script"][:-1])
        self.assertEqual(st, "rejected")

    def test_proof_bound_to_dataset_statement(self):
        other = F.generate(12, 3, "chain-dead")
        st, _ = self.run_cert(self.rec["proof"]["script"], thm=other)
        self.assertEqual(st, "rejected")

    def test_repl_kernel_reports_sorry_axiom(self):
        from lcgrt.lean_repl import LeanRepl
        with LeanRepl() as repl:
            root = repl.init(F.statement(self.thm))
            r = repl.run(root["sid"], "sorry")
            self.assertTrue(r["ok"])
            k = repl.kernel(r["state"]["sid"])
            self.assertIn("sorryAx", k["axioms"])
            bad = repl.run(root["sid"], "apply nonexistent")
            self.assertFalse(bad["ok"])


@unittest.skipUnless(HAVE_LEAN, "Lean toolchain or lgym not built")
class ExtractionAndSearchTests(unittest.TestCase):
    def test_lean_agrees_with_reference_and_replays(self):
        from lcgrt.encode import path_states
        from lcgrt.extract import explore
        from lcgrt.lean_repl import LeanRepl
        with LeanRepl() as repl:
            for fam in F.FAMILIES:
                t = F.generate(5, 4, fam, conj_p=1.0)
                rec = explore(repl, t)               # raises on any Lean/reference mismatch
                self.assertEqual(rec["agreement_checks"], 1)
                self.assertEqual(len(path_states(rec)), rec["proof"]["size"])
                self.assertEqual(rec["proof"]["size"], sum(F.reference_sizes(t)[g] for g in t.goal) + 1)

    def test_packed_features_equal_search_features(self):
        from lcgrt.encode import build_graph, path_states, state_features
        from lcgrt.extract import explore
        from lcgrt.lean_repl import LeanRepl
        with LeanRepl() as repl:
            rec = explore(repl, F.generate(2, 4, "tree-dead"))
        g = build_graph(rec)
        goals, anc, _ = path_states(rec)[1]
        x, _ = state_features(g, goals, anc, rec["table"][goals[0]], set())
        q = np.round(x * 255).astype(np.uint8).astype(np.float32) / 255
        self.assertLess(np.abs(q - x).max(), 1 / 255)

    def test_search_finds_certified_proofs_and_respects_control_layer(self):
        from lcgrt.evaluate import certify_results
        from lcgrt.lean_repl import LeanRepl
        from lcgrt.search import Controller, run_searches
        items = []
        for i, fam in enumerate(("chain-cycle", "alt-cycle", "tree-deep")):
            t = F.generate(20 + i, 5, fam)
            items.append((t.tid, F.statement(t), {"size": sum(F.reference_sizes(t)[g] for g in t.goal),
                          "seed": t.seed, "depth": t.depth, "family": fam, "arity3": False,
                          "stmt_sha256": F.stmt_sha(t)}))
        with LeanRepl() as repl:
            res = run_searches(repl, items, "bfs", None, budget_fn=lambda m: 200)
            self.assertTrue(all(r["success"] for r in res))
            certify_results(res)
            self.assertTrue(all(r["certified"] for r in res))
            ctl = Controller(repl, items[0][1])
            frontier = [ctl.root]
            for _ in range(30):
                if not frontier:
                    break
                node = frontier.pop()
                if not node.goals:
                    continue
                for ch in ctl.expand(node):
                    for k, anc in ch.goals:
                        self.assertNotIn(k, anc)        # no goal repeats on its own path
                    frontier.append(ch)


@unittest.skipUnless(HAVE_LEAN, "Lean toolchain or lgym not built")
class TrainingResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.old = config.DATA_ROOT
        config.DATA_ROOT = Path(cls.tmp.name) / "data"
        from lcgrt import dataset
        spec = {"val": (F.TRAIN_FAMILIES, range(2, 4), False, 6), "train": (F.TRAIN_FAMILIES, range(2, 4), False, 40)}
        old_spec = dataset.SPEC
        dataset.SPEC = spec
        try:
            dataset.build("t", workers=1)
        finally:
            dataset.SPEC = old_spec

    @classmethod
    def tearDownClass(cls):
        config.DATA_ROOT = cls.old
        cls.tmp.cleanup()

    def args(self, out, steps, lr="3e-4"):
        return ["--data", "t", "--preset", "tiny", "--steps", str(steps), "--batch", "8", "--lr", lr,
                "--log-every", "5", "--eval-every", "10", "--save-every", "5", "--device", "cpu",
                "--warmup", "4", "--out", str(out), "--allow-battery", "--min-disk-gib", "0"]

    def test_resume_is_bitwise_identical_and_identity_checked(self):
        from lcgrt import train
        from lcgrt.errors import GymError
        from lcgrt.train import main as train_main
        d = Path(self.tmp.name)
        lim = train.hw.Limits
        train.hw.Limits = lambda **k: lim(**{**k, "min_ram_avail_gib": 0.0, "min_power_limit_w": 0.0,
                                                   "gpu_temp_c": 1e9})
        try:
            self.assertEqual(train_main(self.args(d / "a", 10)), 0)
            # Same identity (steps=10), interrupted by a 6-step window, then resumed.
            a2 = self.args(d / "b", 10) + ["--max-minutes", "0.0001", "--guard-every", "5"]
            self.assertEqual(train_main(a2), 76)
            self.assertEqual(train_main(self.args(d / "b", 10)), 0)
            wa = torch.load(d / "a" / "latest.pt", weights_only=False)["model"]
            wb = torch.load(d / "b" / "latest.pt", weights_only=False)["model"]
            self.assertTrue(all(torch.equal(wa[k], wb[k]) for k in wa))
            with self.assertRaises(GymError):
                train_main(self.args(d / "b", 10, lr="1e-3"))
            # A corrupted latest.pt falls back to the last-known-good copy.
            (d / "b" / "latest.pt").write_bytes(b"corrupt")
            which, ck = train.load_verified(d / "b", "cpu")
            self.assertEqual(which, "lkg")
        finally:
            train.hw.Limits = lim



@unittest.skipUnless(config.LEAN_EXE.exists() and config.TRACE_EXE.exists(), "lgym_trace not built")
class TraceTests(unittest.TestCase):
    SRC = """theorem and_swap' (p q : Prop) (h : p ∧ q) : q ∧ p := by
  constructor
  · exact h.2
  · exact h.1

theorem add_zero' (n : Nat) : n + 0 = n := by
  simp

theorem broken (p : Prop) : p := by
  exact trivial
"""

    def test_trace_steps_errors_axioms_candidates(self):
        from lcgrt.lean_repl import LeanRepl
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "Demo.lean"; src.write_text(self.SRC, encoding="utf-8")
            out = Path(d) / "steps.jsonl"
            with LeanRepl(exe=config.TRACE_EXE, timeout=300) as r:
                res = r.request(op="trace", path=str(src), out=str(out), candidates=["simp", "rfl"], cand_every=1)
            rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        tactics = [x["tactic"] for x in rows]
        self.assertIn("constructor", tactics)
        self.assertIn("exact h.2", tactics)
        self.assertNotIn("by", tactics)                 # structural duplicates are skipped
        self.assertNotIn("·", tactics)
        cons = next(x for x in rows if x["tactic"] == "constructor")
        self.assertEqual((len(cons["goals_before"]), len(cons["goals_after"])), (1, 2))
        simp = next(x for x in rows if x["tactic"] == "simp")
        self.assertTrue(any(c["tactic"] == "simp" and c["ok"] for c in simp["candidates"]))
        axioms = {d["decl"]: d["axioms"] for d in res["decls"]}
        self.assertEqual(axioms["and_swap'"], [])
        self.assertIn("sorryAx", axioms["broken"])
        self.assertTrue(any(e["line"] == 10 for e in res["errors"]))


@unittest.skipUnless(config.TRACE_EXE.exists() and
                     (config.MATHLIB_DIR / ".lake/build/lib/lean/Mathlib/Data/Option/Basic.olean").exists(),
                     "Mathlib v4.19.0 cache not installed")
class MathlibTraceTests(unittest.TestCase):
    def test_trace_mathlib_file(self):
        from lcgrt import mathlib
        from lcgrt.lean_repl import LeanRepl
        env = config.lean_env() | {"LEAN_PATH": mathlib.lean_path()}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "steps.jsonl"
            with LeanRepl(exe=config.TRACE_EXE, env=env, cwd=str(config.MATHLIB_DIR), timeout=900) as r:
                res = r.request(op="trace", path=str(config.MATHLIB_DIR / "Mathlib/Data/Option/Basic.lean"),
                                out=str(out), candidates=["simp"], cand_every=20)
            rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(res["errors"], [])
        self.assertGreater(len(rows), 100)
        self.assertTrue(all(r["goals_before"] for r in rows))
        status, _ = mathlib.decl_status(res)
        self.assertTrue(all(s["certified"] for s in status.values()))


@unittest.skipUnless(config.LEAN_EXE.exists() and config.CERT_EXE.exists(), "lgym_cert not built")
class SelectCertTests(unittest.TestCase):
    SRC = """universe u

theorem and_swap' (p q : Prop) (h : p ∧ q) : q ∧ p := by
  constructor
  · exact h.2
  · exact h.1

theorem refl' {α : Sort u} (a : α) : a = a := by
  rfl
"""

    def test_select_runs_proposals_and_kernel_certifies(self):
        from lcgrt.lean_repl import LeanRepl
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "Demo.lean"; src.write_text(self.SRC, encoding="utf-8")
            out = Path(d) / "sel.jsonl"
            sel = [{"line": 5, "col": 4, "tactics": ["exact h.2", "exact h.1", "sorry", "simp_all"]},
                   {"line": 9, "col": 2, "tactics": ["rfl", "exact rfl"]}]
            with LeanRepl(exe=config.CERT_EXE, timeout=300) as r:
                r.request(op="trace", path=str(src), out=str(out), select=sel)
            rows = {(x["line"], x["col"]): {c["tactic"]: c for c in x["results"]}
                    for x in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
        a = rows[(5, 4)]
        self.assertEqual(a["exact h.2"]["cert"], {"kernel": True, "axioms": []})
        self.assertFalse(a["exact h.1"]["ok"])                       # Lean rejects the wrong component
        self.assertEqual(a["sorry"]["cert"]["kernel"], False)        # closes the goal, never certified
        self.assertTrue(a["simp_all"]["cert"]["kernel"])
        b = rows[(9, 2)]                                             # universe-polymorphic goal
        self.assertTrue(b["rfl"]["cert"]["kernel"] and b["exact rfl"]["cert"]["kernel"])


class TacticLMTests(unittest.TestCase):
    def test_atom_rows_and_normalisation(self):
        from lcgrt.mathlib import is_atom_row
        from lcgrt.tactic_lm import norm_tactic
        self.assertTrue(is_atom_row({"kind": "«]»", "tactic": "]"}))
        self.assertFalse(is_atom_row({"kind": "«tactic_<;>_»", "tactic": "cases a <;> simp"}))
        nl = chr(10)
        self.assertEqual(norm_tactic("rw [a," + nl + "    b]"), "rw [a, b]")

    def test_tiny_model_memorises_and_beam_recovers(self):
        from lcgrt import tactic_lm as T
        nl, turnstile, lor = chr(10), chr(0x22A2), chr(0x2228)     # multi-byte UTF-8, as in Lean output
        goals = ["a : Nat" + nl + turnstile + " a = a", "h : p" + nl + turnstile + " p " + lor + " q",
                 turnstile + " 2 + 2 = 4", "h : q" + nl + turnstile + " p " + lor + " q"]
        pairs = list(zip((g.encode() for g in goals),
                         (b"rfl", b"exact Or.inl h", b"norm_num", b"exact Or.inr h")))
        enc = np.zeros((4, T.CTX), np.uint8); dec = np.zeros((4, T.TGT), np.uint8)
        el = np.zeros(4, np.int16); dl = np.zeros(4, np.int16)
        for i, (g, t) in enumerate(pairs):
            enc[i, :len(g)] = np.frombuffer(g, np.uint8); el[i] = len(g)
            dec[i, :len(t)] = np.frombuffer(t, np.uint8); dl[i] = len(t)
        torch.manual_seed(0)
        m = T.TacticLM(T.PRESETS["tlm-tiny"])
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        b = T.make_batch(enc, el, "cpu", dec, dl)
        for _ in range(250):
            loss = T.nll(m, b); opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        best = [row[0][0] for row in m.beam(b["enc"], b["enc_mask"], k=3, max_len=24)]
        self.assertEqual(best, [t.decode() for _, t in pairs])


if __name__ == "__main__":
    unittest.main()
