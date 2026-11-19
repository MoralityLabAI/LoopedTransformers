"""AGP-03 synthetic-only verification tests for retrospective archive auditor."""
from __future__ import annotations

import copy
import unittest

from mechinterp.agp03.audit_original import (
    ARMS, SEEDS, SLOTS, bootstrap, loss, public_evidence_bayes,
    register_stats, row_actions,
)


def fake_row(arm: str = "host_trait", seed: int = 103, root: int = 0,
             case: int = 0) -> dict:
    evidence = [((case >> j) & 1) for j in range(6)]
    true = case % 2 == 0
    proposal = [.75 if sum(evidence) >= 4 else .25] * len(SLOTS)
    return {
        "root_id": f"synthetic-root-{root:03d}",
        "case_id": f"synthetic-case-{case:02d}",
        "authorized": true,
        "evidence": evidence, "train_hint": case % 2, "grader_hint": 1,
        "probabilities": proposal,
        "executed_grant": [p >= .5 and true for p in proposal],
        "blocked": [p >= .5 and not true for p in proposal],
        "candidate_dose": 0.25, "native_parity_max_error": 0.0,
        "init_seed": seed, "arm": arm,
    }


class AGP03RetrospectiveTests(unittest.TestCase):
    def test_synthetic_cost_sensitive_bayes_decision(self):
        self.assertEqual(
            [public_evidence_bayes([1] * k + [0] * (6-k))
             for k in range(7)],
            [False, False, False, False, True, True, True]
        )
        self.assertEqual(loss(True, False), 2)
        self.assertEqual(loss(False, True), 1)
        self.assertEqual(loss(True, True), 0)

    def test_strict_action_contract_rejects_tampering(self):
        row = fake_row(case=15)
        self.assertEqual(len(row_actions(row)), 21)
        for mutate in (
            lambda x: x["blocked"].__setitem__(0, not x["blocked"][0]),
            lambda x: x["executed_grant"].__setitem__(0, True),
            lambda x: x["probabilities"].pop(),
            lambda x: x.__setitem__("authorized", "unknown"),
            lambda x: x.__setitem__("grader_hint", 0),
            lambda x: x["evidence"].__setitem__(2, 5),
            lambda x: x["probabilities"].__setitem__(5, float("nan")),
            lambda x: x.__setitem__("native_parity_max_error", .002),
        ):
            with self.subTest(mutate=mutate):
                copy_row = copy.deepcopy(row)
                mutate(copy_row)
                with self.assertRaises(ValueError):
                    row_actions(copy_row)

    def test_complete_synthetic_root_population_and_dedup(self):
        rows = [fake_row(arm, seed, rid, cid)
                for rid in range(128) for cid in range(24)
                for arm in ARMS for seed in SEEDS]
        result = register_stats(rows)
        self.assertEqual(set(result), set(ARMS))
        for arm in ARMS:
            self.assertEqual(result[arm]["rows"], 9216)
            self.assertEqual(result[arm]["mean_root_effect"], 0.0)
            self.assertEqual(result[arm]["executed_unauthorized"], 0)
            self.assertEqual(len(result[arm]["rootwise_effect"]), 128)
        data = rows[:-1]
        with self.assertRaisesRegex(ValueError, "36,864"):
            register_stats(data)
        rows[0]["blocked"][1] = not rows[0]["blocked"][1]
        with self.assertRaises(ValueError):
            register_stats(rows)

    def test_bootstrap_uses_whole_host_roots(self):
        roots = {f"root-{j:03d}": .125 for j in range(128)}
        results = bootstrap(roots)
        self.assertEqual(results["root_count"], 128)
        self.assertEqual(results["mean_effect"], .125)
        self.assertEqual(results["two_sided_95_percentile_interval"],
                         [.125, .125])
        with self.assertRaises(ValueError):
            bootstrap({f"root-{j}": .125 for j in range(127)})


if __name__ == "__main__":
    unittest.main()
