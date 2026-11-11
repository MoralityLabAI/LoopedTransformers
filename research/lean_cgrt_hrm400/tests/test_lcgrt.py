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

theorem rw_nested (a b : Nat) (h : ∀ x : Nat, x = x → x + 0 = b) : a + 0 = b := by
  rw [h _ (by rfl)]

def idProof := fun (α : Sort _) (a : α) => (show a = a by rfl)
"""

    def test_select_runs_proposals_and_kernel_certifies(self):
        from lcgrt.lean_repl import LeanRepl
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "Demo.lean"; src.write_text(self.SRC, encoding="utf-8")
            out = Path(d) / "sel.jsonl"
            sel = [{"line": 5, "col": 4, "tactics": ["exact h.2", "exact h.1", "sorry", "simp_all"]},
                   {"line": 9, "col": 2, "tactics": ["rfl", "exact rfl"]},
                   {"line": 12, "col": 2, "tactics": ["rw [h _ (by rfl)]", "exact h _ (by rfl)"]},
                   {"line": 14, "col": 58, "tactics": ["rfl", "exact rfl"]}]
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
        c = rows[(12, 2)]           # a nested `by` postponed until rw fixes `_` is run before the proof is read
        self.assertEqual([c[t].get("cert") for t in c], [{"kernel": True, "axioms": []}] * 2)
        d = rows[(14, 58)]          # universe still a metavariable when the tactic ran (`Sort _`)
        self.assertEqual([d[t].get("cert") for t in d],
                         [{"kernel": True, "axioms": [], "levels_generalized": True}] * 2)

    SRC2 = """universe u v

def pickNat : Nat := by
  exact 3

theorem ulift_down {α : Type u} (a : α) : (ULift.up.{v} a).down = a := by
  exact (rfl : (ULift.up.{v, u} a).down = a)

theorem cls (p q : Prop) (hq : q) : p → q := by
  classical
  intro _h
  exact hq
"""

    def test_select_flags_data_goals_universe_names_and_layout(self):
        from lcgrt.lean_repl import LeanRepl
        from lcgrt.mathlib_eval import certifiable, human_text
        from lcgrt.tactic_lm import norm_tactic
        nl = chr(10)
        src_cls = "classical" + nl + "  intro _h" + nl + "  exact hq"      # the source text at 10:2
        human = human_text({"tactic": src_cls, "col": 2})
        self.assertEqual(human, "classical" + nl + "intro _h" + nl + "exact hq")
        ulift = "exact (rfl : (ULift.up.{v, u} a).down = a)"
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "Demo2.lean"; src.write_text(self.SRC2, encoding="utf-8")
            out = Path(d) / "sel.jsonl"
            sel = [{"line": 4, "col": 2, "tactics": ["exact 3"]},
                   {"line": 7, "col": 2, "tactics": [ulift]},
                   {"line": 10, "col": 2, "tactics": [human, norm_tactic(src_cls)]}]
            with LeanRepl(exe=config.CERT_EXE, timeout=300) as r:
                r.request(op="trace", path=str(src), out=str(out), select=sel)
            rows = {(x["line"], x["col"]): x for x in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
        res = {k: {c["tactic"]: c for c in v["results"]} for k, v in rows.items()}
        # a data goal closes but can never be a theorem: flagged, so it is counted and not scored
        self.assertEqual((rows[(4, 2)]["main_prop"], rows[(4, 2)]["main_closed"]), (False, True))
        self.assertFalse(certifiable(rows[(4, 2)]))
        self.assertFalse(res[(4, 2)]["exact 3"]["cert"]["kernel"])
        # `.{v, u}` resolves: the goal's universe parameters are in scope for the tactic
        self.assertTrue(certifiable(rows[(7, 2)]))
        self.assertEqual(res[(7, 2)][ulift]["cert"], {"kernel": True, "axioms": []})
        # the dedented block certifies; the whitespace-collapsed one intros three names and fails
        self.assertEqual(res[(10, 2)][human]["cert"], {"kernel": True, "axioms": []})
        self.assertFalse(res[(10, 2)][norm_tactic(src_cls)]["ok"])


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

    def test_xs_presets_parameter_matched(self):
        from lcgrt import tactic_lm as T
        n = {k: T.count_parameters(T.TacticLM(T.PRESETS[f"tlm-{k}-XS"]))["total"] for k in ("flat", "hrm", "loop")}
        self.assertLess(max(n.values()) / min(n.values()), 1.015)

    @staticmethod
    def _pairs_batch(T):
        nl, turnstile, lor = chr(10), chr(0x22A2), chr(0x2228)
        goals = ["a : Nat" + nl + turnstile + " a = a", "h : p" + nl + turnstile + " p " + lor + " q",
                 turnstile + " 2 + 2 = 4", "h : q" + nl + turnstile + " p " + lor + " q"]
        tacs = (b"rfl", b"exact Or.inl h", b"norm_num", b"exact Or.inr h")
        enc = np.zeros((4, T.CTX), np.uint8); dec = np.zeros((4, T.TGT), np.uint8)
        el = np.zeros(4, np.int16); dl = np.zeros(4, np.int16)
        for i, (g, t) in enumerate(zip((g.encode() for g in goals), tacs)):
            enc[i, :len(g)] = np.frombuffer(g, np.uint8); el[i] = len(g)
            dec[i, :len(t)] = np.frombuffer(t, np.uint8); dl[i] = len(t)
        return T.make_batch(enc, el, "cpu", dec, dl), [t.decode() for t in tacs]

    def test_trm_preset_size_and_identity(self):
        from lcgrt import tactic_lm as T
        self.assertTrue(3.9e6 < T.count_parameters(T.TacticLM(T.PRESETS["tlm-trm-4M"]))["total"] < 4.1e6)
        for k in ("tlm-flat-S", "tlm-hrm-S", "tlm-loop-XS"):          # the other arms keep their hashes
            self.assertFalse(set(T.TRM_FIELDS) & set(T.cfg_identity(T.PRESETS[k])))
        self.assertTrue(set(T.TRM_FIELDS) <= set(T.cfg_identity(T.PRESETS["tlm-trm-4M"])))

    def test_trm_carry_continues_until_halting(self):
        """A slot keeps its sample and state until it halts (at trm_sup steps while the halting
        logit is negative); then it takes the new batch's sample and starts again."""
        from lcgrt import tactic_lm as T
        torch.manual_seed(0)
        m = T.TacticLM(T.PRESETS["tlm-trm-tiny"])
        b, _ = self._pairs_batch(T)
        b2 = {k: v.flip(0) for k, v in b.items()}
        rng = np.random.default_rng(0)
        loss, carry, info = m.act_step(None, b, rng)
        self.assertEqual(info["fresh"], 4)
        self.assertTrue(torch.isfinite(loss))
        for step in range(2, m.cfg.trm_sup + 1):
            self.assertFalse(bool(carry["halted"].any()))
            _, carry, info = m.act_step(carry, b2, rng)
            self.assertEqual(info["fresh"], 0)
            self.assertTrue(torch.equal(carry["enc"], b["enc"]))
            self.assertEqual(carry["steps"].tolist(), [step] * 4)
        self.assertTrue(bool(carry["halted"].all()))
        _, carry, info = m.act_step(carry, b2, rng)
        self.assertEqual(info["fresh"], 4)
        self.assertTrue(torch.equal(carry["enc"], b2["enc"]))
        self.assertEqual(carry["steps"].tolist(), [1] * 4)

    def test_trm_tiny_learns_and_beam_recovers(self):
        """Deep supervision learns the pairs under the trainer's optimizer recipe (AdamW, warmup,
        clipping). Plain Adam at 3e-3 without clipping collapses this tiny TRM to an
        input-independent encoder; one recursion with full gradient does not."""
        from lcgrt import tactic_lm as T
        torch.manual_seed(0)
        m = T.TacticLM(T.PRESETS["tlm-trm-tiny"])
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05, betas=(0.9, 0.95))
        b, want = self._pairs_batch(T)
        rng, carry = np.random.default_rng(0), None
        for i in range(1000):
            for g in opt.param_groups:
                g["lr"] = 1e-3 * min(1.0, (i + 1) / 100)
            loss, carry, _ = m.act_step(carry, b, rng)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        best = [row[0][0] for row in m.beam(b["enc"], b["enc_mask"], k=3, max_len=24)]
        self.assertEqual(best, want)

    def test_proposal_cache_reuses_only_on_exact_key(self):
        from dataclasses import asdict
        from lcgrt import mathlib_eval as M, tactic_lm as T
        from lcgrt.train import sha256
        nl, turnstile, lor = chr(10), chr(0x22A2), chr(0x2228)
        torch.manual_seed(0)
        cfg = T.PRESETS["tlm-tiny"]
        goals = ["a : Nat" + nl + turnstile + " a = a", turnstile + " 2 + 2 = 4", "h : p" + nl + turnstile + " p " + lor + " q"]
        states = [{"file": "A.lean", "line": i, "col": 2, "goals_before": [g]} for i, g in enumerate(goals)]
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())   # _proposal_job sets the thread count
        with tempfile.TemporaryDirectory() as d:
            run, cache = Path(d) / "run_x", Path(d) / "cache"
            run.mkdir()
            torch.save({"model": T.TacticLM(cfg).state_dict(), "config": asdict(cfg)}, run / "best.pt")
            (run / "best.json").write_text(json.dumps({"step": 1, "sha256": sha256(run / "best.pt")}))
            digest = M.states_sha256(states)
            key = M.proposal_key(run, digest, 2, None, 1, "cpu")
            job = (str(run), states, key, str(cache), 2, None, 1, "cpu")
            name, first, hit = M._proposal_job(job)
            self.assertEqual((name, hit, len(first)), ("run_x", False, 3))
            entry = json.loads((cache / "run_x.json").read_text(encoding="utf-8"))
            entry["proposals"] = [["sentinel"]] * 3                      # a hit must return what is stored
            (cache / "run_x.json").write_text(json.dumps(entry), encoding="utf-8")
            self.assertEqual(M._proposal_job(job)[1:], ([["sentinel"]] * 3, True))
            other = M.proposal_key(run, digest, 2, None, 2, "cpu")          # another thread count: recomputed
            self.assertEqual(M._proposal_job((str(run), states, other, str(cache), 2, None, 1, "cpu"))[1:], (first, False))
            self.assertNotEqual(M.states_sha256(states[:2]), digest)


class RapidGymTests(unittest.TestCase):
    def test_overrides_and_round1_lines_parse(self):
        from lcgrt import rapid as R, tactic_lm as T
        cfg = T.apply_overrides(T.PRESETS["tlm-trm-XS"], ["trm_sup=4", "trm_ema=0"])
        self.assertEqual((cfg.trm_sup, cfg.trm_ema, cfg.width), (4, 0.0, 176))
        with self.assertRaises(Exception):
            T.apply_overrides(cfg, ["no_such_field=1"])
        self.assertEqual(sum(p.numel() for p in T.TacticLM(T.PRESETS["tlm-trm-XS"]).parameters()), 1491204)
        lines = [R.train_line(1, s) for s in R.ROUND1]
        self.assertEqual(len({l["name"] for l in lines}), len(lines))
        for l in lines:
            a = T.parser().parse_args(l["args"] + ["--out", "x"])
            T.apply_overrides(T.PRESETS[a.preset], a.cfg)
            self.assertEqual((a.steps, a.eval_every, a.warmup), (1000, 1000, 100))
        ev = R.eval_line(1, [l["name"] for l in lines], 120)
        self.assertEqual(ev["args"][ev["args"].index("--splits") + 1], "val")

    def test_next_round_survivors_and_paired_mutants(self):
        from lcgrt import rapid as R
        specs = [R.full(s) for s in R.ROUND1]
        rows = [{"round": 1, "spec": s, "dev_close@8": c, "val_bpb": 2.0} for s, c in zip(specs, (.10, .08, .12, .14, .05, .09))]
        nxt = R.next_specs(rows, rows, 2, 6, 2000)
        self.assertEqual(len(nxt), 6)
        self.assertEqual([(s["preset"], s["cfg"]) for s in nxt[:3]],
                         [("tlm-trm-XS", {"trm_sup": 4}), ("tlm-trm-XS", {}), ("tlm-flat-XS", {})])
        self.assertTrue(all(s["steps"] == 2000 and s["seed"] == 29 for s in nxt))
        mutants = nxt[3:]
        self.assertTrue(all(m["parent"] == R.key(nxt[0]) and "knob" in m for m in mutants))
        self.assertEqual(len({R.key(s) for s in nxt}), 6)
        later = [dict(r, round=2, spec=s) for r, s in zip(rows, nxt)]
        for r, d in zip(later, (.20, .15, .11, .18, .22, .17)):
            r["dev_close@8"] = d
        eff = R.knob_effects(later)
        self.assertEqual(sum(v["n"] for v in eff.values()), 3)
        r3 = R.next_specs(later, rows + later, 3, 6, 2000)       # a mutant can survive; it loses its tags
        self.assertTrue(any("knob" in r["spec"] for r in sorted(later, key=lambda r: -r["dev_close@8"])[:3]))
        self.assertTrue(all("knob" not in s for s in r3[:3]))
        self.assertTrue(all(s["steps"] == 2000 and s["seed"] == 43 for s in r3))

    def test_unscored_arms_return_and_leases_fit(self):
        from lcgrt import rapid as R
        flat = R.full({"preset": "tlm-flat-XS"})
        rows = [{"round": 1, "spec": flat, "dev_close@8": .17, "val_bpb": 2.5}]
        retry = [R.full(s) for s in R.ROUND1[1:]]                      # round 1's arms that failed to train
        nxt = R.next_specs(rows, rows, 2, 6, 2000, retry)
        self.assertEqual(len(nxt), 6)                                 # survivor + 5 returning arms, no mutant
        self.assertEqual([R.arm(s) for s in nxt[1:]], [R.arm(s) for s in retry])
        self.assertTrue(all(s["steps"] == 2000 and s["seed"] == 29 and "knob" not in s for s in nxt))
        self.assertEqual(R.next_specs(rows, rows, 2, 6, 2000, [flat])[1:], R.next_specs(rows, rows, 2, 6, 2000)[1:])
        lines = {s["preset"]: R.train_line(2, s) for s in nxt}
        self.assertEqual(lines["tlm-loop-XS"]["vram"], "6G")
        self.assertIn("--random-depth", lines["tlm-loop-XS"]["args"])
        self.assertTrue(all("--grad-ckpt" in l["args"] for l in lines.values()))


class CpuProfileTests(unittest.TestCase):
    def test_parse_size(self):
        from lcgrt.ramcap import parse_size
        self.assertEqual(parse_size("9G"), 9 * 2**30)
        self.assertEqual(parse_size("512M"), 512 * 2**20)
        self.assertEqual(parse_size("1.5GiB"), 3 * 2**29)

    @unittest.skipUnless(sys.platform == "win32", "Windows job objects")
    def test_ram_cap_holds_for_grandchildren(self):
        """A grandchild started through the interpreter launcher cannot commit past the cap, runs at
        Idle priority, and a small allocation still succeeds."""
        import subprocess
        code = ("import subprocess, sys; from lcgrt.ramcap import Job, parse_size; "
                "j = Job(parse_size('256M')); j.assign_self(); "
                "c = 'import psutil, sys; b = bytearray(int(sys.argv[1]) << 20); print(psutil.Process().nice())'; "
                "r = [subprocess.run([sys.executable, '-c', c, n], capture_output=True, text=True) for n in ('32', '512')]; "
                "print(r[0].returncode, r[0].stdout.strip(), r[1].returncode, 'MemoryError' in r[1].stderr)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT)).stdout.split()
        self.assertEqual(out, ["0", "64", "1", "True"])

    def test_cpu_only_lane_selection_and_command(self):
        from lcgrt import runq
        gpu = {"name": "g", "module": "lcgrt.tactic_lm", "vram": "6G", "args": ["--steps", "1"]}
        cpu = {"name": "c", "module": "lcgrt.tactic_lm", "cpu": True, "args": ["--steps", "1"]}
        ev = {"name": "e", "module": "lcgrt.mathlib_eval", "args": []}
        self.assertEqual([runq.wanted(j, "train", False) for j in (gpu, cpu, ev)], [True, True, False])
        self.assertEqual([runq.wanted(j, "train", True) for j in (gpu, cpu, ev)], [False, True, False])
        self.assertEqual([runq.wanted(j, "eval", True) for j in (gpu, cpu, ev)], [False, False, True])
        cmd = runq.job_command(cpu, Path("out"), "train", 4, cpu_only=True)
        self.assertNotIn("vramorama", " ".join(cmd).lower())
        self.assertEqual(cmd[cmd.index("--device") + 1], "cpu")
        self.assertEqual(cmd[cmd.index("--threads") + 1], "4")
        self.assertNotIn("--precision", cmd)

    def test_gpu_lane_command(self):
        """On the GPU path a CPU-sized line keeps its fp32 recipe, every tactic-LM job is capped at
        its lease, and the duty share is passed on; an eval line can be moved to the GPU runner."""
        from lcgrt import runq
        orig, runq._VRAM_TOTAL = runq._VRAM_TOTAL, 16384
        try:
            gpu = {"name": "g", "module": "lcgrt.tactic_lm", "vram": "8G", "args": ["--steps", "1"]}
            cpu = {"name": "c", "module": "lcgrt.tactic_lm", "cpu": True, "vram": "6G", "args": ["--steps", "1"]}
            c = runq.job_command(cpu, Path("out"), "train", 4, gpu_duty=0.6)
            self.assertEqual(c[c.index("--precision") + 1], "fp32")
            self.assertEqual(c[c.index("--cuda-memory-fraction") + 1], "0.375")
            self.assertEqual(c[c.index("--gpu-duty") + 1], "0.6")
            self.assertNotIn("--device", c)
            g = runq.job_command(gpu, Path("out"), "train", 4)
            self.assertNotIn("--precision", g); self.assertNotIn("--gpu-duty", g)
            self.assertEqual(g[g.index("--cuda-memory-fraction") + 1], "0.500")
            moved = {"name": "e", "module": "lcgrt.evaluate", "lane": "train", "vram": "3G", "args": []}
            self.assertEqual(runq.lane_of(moved), "train")
            self.assertTrue(runq.wanted(moved, "train", False))
            self.assertFalse(runq.wanted(moved, "eval", True))
            m = runq.job_command(moved, Path("out"), "train", 4, gpu_duty=0.6)
            self.assertNotIn("--device", m); self.assertNotIn("--gpu-duty", m)
            self.assertEqual(m[m.index("--cuda-memory-fraction") + 1], "0.188")
        finally:
            runq._VRAM_TOTAL = orig

    def test_priority_then_file_order(self):
        from lcgrt import runq
        jobs = [{"name": "a"}, {"name": "b", "priority": 2}, {"name": "c"}, {"name": "d", "priority": 2}]
        self.assertEqual([j["name"] for j in runq.by_priority(jobs)], ["b", "d", "a", "c"])

    def test_cpu_job_ignores_gpu_state(self):
        from lcgrt import hw
        orig, hw.gpu = hw.gpu, (lambda: {"temp_c": 99.0, "power_limit_w": 15.0})     # a hot, capped GPU
        try:
            self.assertFalse(hw.check(hw.Limits(require_ac=False, min_disk_gib=0), ".")[0])
            self.assertTrue(hw.check(hw.Limits(require_ac=False, min_disk_gib=0, use_gpu=False), ".")[0])
        finally:
            hw.gpu = orig

    def test_report_arm_names(self):
        from lcgrt.report_mathlib import ARM_RE
        self.assertEqual(ARM_RE.match("tlmcpu_hrmXS_s29").groupdict(), {"arm": "hrmXS", "seed": "29"})
        self.assertEqual(ARM_RE.match("tlm3_flatS_s17")["arm"], "flatS")
        self.assertIsNone(ARM_RE.match("tlm3_flatS_s17+auto"))
        self.assertEqual(ARM_RE.match("tlm4_trm_s43").groupdict(), {"arm": "trm", "seed": "43"})


if __name__ == "__main__":
    unittest.main()
