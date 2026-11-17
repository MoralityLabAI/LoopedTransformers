"""LS-SAE12 CPU unit tests: solvers, schedules, site rules, gate, directions, toy end-to-end."""
from __future__ import annotations
import json
import random
import unittest

import numpy as np
import torch

from mechinterp.loopscope02.scripts import trm_sudoku as TRM
from mechinterp.sae_causal_transfer.active_reasoning_v1 import audit as A
from mechinterp.sae_causal_transfer.active_reasoning_v1 import families as F
from mechinterp.sae_causal_transfer.active_reasoning_v1 import run_native as RUN
from mechinterp.sae_causal_transfer.active_reasoning_v1 import schedule as S
from mechinterp.sae_causal_transfer.native_withinrun_v1 import families as S11

# A published 17-clue puzzle with a unique solution, and that solution.
P17 = [int(c) for c in
       "000000010400000000020000000000050407008000300001090000300400200050100000000806000"]
P17_SOL = [int(c) for c in
           "693784512487512936125963874932651487568247391741398625319475268856129743274836159"]


def toy_model(seed=3):
    torch.manual_seed(seed)
    m = TRM.PureSudokuTRM(hidden_size=512, num_heads=8, l_layers=1, h_cycles=1, l_cycles=1)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(0.5)
    return m.float().eval().requires_grad_(False)


class SolverTests(unittest.TestCase):
    def test_independent_solvers_agree_unique(self):
        a = S11.solve_unique(P17)
        b = F.solve_dfs(P17)
        self.assertEqual(a["solutions"], 1)
        self.assertEqual(b["solutions"], 1)
        self.assertEqual(a["target"], P17_SOL)
        self.assertEqual(b["target"], P17_SOL)
        self.assertTrue(F.valid_complete(P17_SOL, P17))

    def test_two_solutions_and_conflicts(self):
        loose = list(P17)
        loose[[i for i, v in enumerate(P17) if v][0]] = 0
        self.assertEqual(F.solve_dfs(loose)["solutions"], S11.solve_unique(loose)["solutions"])
        self.assertEqual(F.solve_dfs(loose)["solutions"], 2)
        bad = list(P17)
        bad[0], bad[1] = 5, 5
        self.assertEqual(F.solve_dfs(bad)["solutions"], 0)

    def test_pool_registration_shape(self):
        self.assertEqual(len(F.POOL_FILES), 48)
        names = [n for n, _ in F.POOL_FILES]
        self.assertEqual(names, sorted(names))
        self.assertTrue(set(names).isdisjoint({n for n, _ in S11.SOURCE_FILES}))
        self.assertGreater(names[0], S11.SOURCE_FILES[-1][0])


class ScheduleTests(unittest.TestCase):
    def test_verified_schedule_definition(self):
        x = [0] + [1] * 80
        target = [7] + [1] * 80
        steps = [[0] * 81 for _ in range(16)]
        for s in range(16):
            steps[s] = [7 if s >= 9 else 3] + [1] * 80   # correct from ACT10 onward
        self.assertEqual(S.verified_schedule(steps, x, target), {0: 10})
        steps[15] = [3] + [1] * 80
        self.assertEqual(S.verified_schedule(steps, x, target), {0: 17})
        steps = [[7] + [1] * 80 for _ in range(16)]
        self.assertEqual(S.verified_schedule(steps, x, target), {0: 1})

    def test_runner_and_auditor_schedules_and_outcomes_agree(self):
        rng = random.Random(5)
        x = list(P17)
        blanks = S.blanks(x)
        for _ in range(40):
            traj = []
            for s in range(16):
                p = list(P17_SOL)
                for c in blanks:
                    if rng.random() < max(0.0, 0.6 - 0.05 * s):
                        p[c] = rng.randint(1, 9)
                traj.append(p)
            self.assertEqual(S.verified_schedule(traj, x, P17_SOL), A.sched(traj, x, P17_SOL))
            site, rec, sched = S.select_site(traj, x, P17_SOL)
            for c in rec:
                self.assertNotEqual(traj[3][c], P17_SOL[c])
                self.assertTrue(6 <= sched[c] <= 16)
            if site is not None:
                self.assertEqual(sched[site], min(sched[c] for c in rec))

    def test_deduction_change_counts(self):
        clean = {1: 6, 2: 17, 3: 9}
        edit = {1: 17, 2: 12, 3: 7}
        out = S.deduction_change(clean, edit, 1)
        self.assertEqual(out["vdd"], 11 + 5 + 2)
        self.assertEqual((out["blocked_recoveries"], out["new_recoveries"]), (1, 1))
        self.assertEqual((out["accelerated"], out["delayed"]), (2, 1))
        self.assertTrue(out["site_blocked"])

    def test_gate_matches_auditor_and_is_deterministic(self):
        rng = random.Random(11)
        for fam in S.FAMILIES:
            diffs = [rng.uniform(-1, 3) for _ in range(12)]
            g = S.family_gate(diffs, fam)
            self.assertEqual(g["bootstrap95"], A.gate(diffs, fam)["bootstrap95"])
            self.assertEqual(g["registered_success"], A.gate(diffs, fam)["registered_success"])
            self.assertEqual(g, S.family_gate(diffs, fam))
        self.assertNotEqual(S.BOOT_SEEDS["active_a"], S.BOOT_SEEDS["active_b"])
        self.assertIsNone(S.family_gate([1.0] * 11, "active_a")["registered_success"])

    def test_family_assignment(self):
        f = S.assign_families(list(range(20, 50)))
        self.assertEqual(f["active_a"], list(range(20, 32)))
        self.assertEqual(f["active_b"], list(range(32, 44)))
        self.assertFalse(S.assign_families([20, 21])["complete"]["active_a"])

    def test_isotropic_vectors_regenerate_exactly(self):
        seed = S.direction_seed(S.ISO_SEED_BASE, 33, "observable_error", 5)
        self.assertEqual(seed, 2026111800000 + 33000 + 5)
        self.assertEqual(S.gaussian_unit(seed), A.gauss_unit(seed))
        self.assertAlmostEqual(sum(v * v for v in S.gaussian_unit(seed)), 1.0, places=12)


class NativeToyTests(unittest.TestCase):
    def test_power_iteration_matches_explicit_jacobian_svd(self):
        model = toy_model()
        _, _, anchor = RUN.clean_trajectory(model, P17)
        token = model.puzzle_emb_len + S.blanks(P17)[0]
        vec, meta = RUN.top_singular_direction(model, anchor, token, "logit", 123, iters=60)

        def f(v):
            zh = anchor["zh"].clone()
            zh[0, token] = zh[0, token] + v
            out_h, _ = model.act_step(zh, anchor["zl"], anchor["x_emb"])
            lg = model.digit_logits(out_h)
            return (lg - lg.mean(dim=-1, keepdim=True)).reshape(-1)

        with torch.enable_grad():
            J = torch.autograd.functional.jacobian(f, torch.zeros(512)).double().numpy()
        _, sv, vt = np.linalg.svd(J, full_matrices=False)
        self.assertGreater(abs(float(vt[0] @ vec)), 0.99)
        self.assertAlmostEqual(meta["sigma_estimate"], sv[0], delta=1e-3 * sv[0])

    def test_toy_end_to_end_state_passes_independent_audit(self):
        model = toy_model()
        q, _ = RUN.readout_q(model)
        src = {"x": P17, "target": P17_SOL, "seed": 1, "source_file": "toy"}
        scr = RUN.screen_row(model, src, 20)
        if scr["selected_site"] is None:
            scr["selected_site"] = S.blanks(P17)[0]
        for rule in S.SITE_RULES:
            st = json.loads(json.dumps(RUN.run_state(model, src, scr, rule, q)))
            self.assertTrue(A.state_audit(st, src, scr, rule))
            self.assertEqual(st["outcomes"][0]["vdd"], 0)
            n = len(st["variants"])
            self.assertEqual(n, 61 if rule == "observable_error" else 21)
            # Clean row of the 41-variant batch reproduces the batch-1 clean run.
            self.assertTrue(all(st["same_forward_clean_matches_screen_steps"][:4]))

    def test_zero_dose_is_identity(self):
        model = toy_model()
        _, _, anchor = RUN.clean_trajectory(model, P17)
        with torch.no_grad():
            a = model.act_step(anchor["zh"].repeat(2, 1, 1), anchor["zl"].repeat(2, 1, 1), anchor["x_emb"].repeat(2, 1, 1))[0]
        self.assertTrue(torch.equal(a[0], a[1]))


if __name__ == "__main__":
    unittest.main()
