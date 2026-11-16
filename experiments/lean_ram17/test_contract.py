"""Source-only RAM17 manifest, candidate, gate and leakage contracts."""
from __future__ import annotations
import json
from pathlib import Path
import random
import sys
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(1,str(HERE.parent/"lean_ram16"))
import grammar
from balanced import FAMILIES,VerifiedCurriculum,draw_batch

CONF=json.loads((HERE/"config.json").read_text())
HIDDEN=set(CONF["source_modules"]["test"])
HISTORICAL16={"TemporalCutCapacity","PhaseLatch","ObserverLabelAlignment","ContractionAliasing"}

def sample_rows():
    rows=[]
    for fam in FAMILIES:
        for module in ("M1","M2","M3"):
            for root in ("a","b"):
                for j in range(2):
                    rows.append(dict(role="train",label=3,source_compiles=3,
                        family=fam,source_file=f"MathCan/{module}.lean",
                        theorem=f"{root}{module}",script=f"{fam}_{j}",
                        before={"goals":["n : Nat\n⊢ n = n"],"definitions":[]}))
    return rows

class RAM17Contracts(unittest.TestCase):
    def test_exact_source_split(self):
        self.assertEqual((len(CONF["source_modules"]["train"]),
                          len(CONF["source_modules"]["dev"]),
                          len(CONF["source_modules"]["test"])),(14,4,5))
        roles=[set(CONF["source_modules"][role]) for role in ("train","dev","test")]
        self.assertFalse(roles[0]&roles[1] or roles[0]&roles[2] or roles[1]&roles[2])
        self.assertFalse(set(CONF["source_modules"]["train"])&HISTORICAL16)
        self.assertFalse(set(CONF["source_modules"]["dev"])&HISTORICAL16)
        self.assertNotIn("GramianTransfer",roles[0]|roles[1])
        self.assertTrue(HISTORICAL16.issubset(set(CONF["protected_test_modules_older"])))
        self.assertEqual(set(CONF["chronologically_new_test_modules"]),
                         {"NOR19StreamedVerifier","NOR20BigintWidthBounds"})
        self.assertEqual(set(HIDDEN)-set(CONF["chronologically_new_test_modules"]),
                         set(CONF["protected_test_modules_older"])-HISTORICAL16)
        self.assertEqual(CONF["train_source_commit"],"f427e0ef4a61093304d12ca2c0dc943d8d2429e5")
        self.assertEqual(CONF["test_source_commit"],"3749c84cc6d47297faaf94404af6d1fd36916812")
        self.assertEqual(CONF["prospective_TEST_scored"],0)
        for role in ("train","dev","test"):
            self.assertTrue(all(len(v)==40 for v in CONF["source_modules"][role].values()))

    def test_source_positive_gate_is_strict(self):
        self.assertEqual(CONF["gate"]["min_distinct_nat_modules"],2)
        self.assertEqual(CONF["gate"]["min_distinct_prior_modules"],2)
        self.assertTrue(CONF["gate"]["require_all_families"])
        self.assertEqual(CONF["learning"]["train_seeds"],[17,29,43])
        self.assertEqual(CONF["learning"]["novel_families"],23)
        self.assertEqual(len(CONF["primary"]["policies"]),16)
        self.assertEqual(CONF["primary"]["evaluations"],48)
        self.assertEqual(CONF["primary"]["seconds"],30)

    def test_original_source_grammar_proof_body_independent(self):
        src="""import Init
def oldDef (x : Nat) := x
theorem old_eq (n : Nat) : oldDef n = n := by
  rfl
theorem new_eq (n : Nat) : oldDef n = n := by
  exact old_eq n
theorem future : False := by
  sorry
"""
        a=grammar.generate(src,"new_eq")
        b=grammar.generate(src.replace("  exact old_eq n","  omega"),"new_eq")
        self.assertEqual(a,b)
        self.assertFalse(any("future" in x.script for x in a))
        self.assertTrue(all(grammar._safe(x.family,x.script) for x in a))

    def test_exact_family_balance_and_module_cycles(self):
        sampler=VerifiedCurriculum(sample_rows(),seed=17)
        rows=[sampler.sample() for _ in range(2560)]
        receipt=sampler.receipt()
        self.assertEqual(receipt["total_draws"],2560)
        self.assertEqual(set(receipt["by_family"].values()),{853,854})
        self.assertEqual(len({x["source_file"] for x in rows}),3)
        for fam in FAMILIES:
            mods=list(receipt["by_module"][fam].values())
            self.assertLessEqual(max(mods)-min(mods),1)

    def test_reproducible_sampler_with_distinct_seeds(self):
        a=VerifiedCurriculum(sample_rows(),seed=29)
        b=VerifiedCurriculum(sample_rows(),seed=29)
        self.assertEqual([(a.sample()["family"],a.sample()["theorem"]) for _ in range(50)],
                         [(b.sample()["family"],b.sample()["theorem"]) for _ in range(50)])

    def test_reject_invented_or_exposed_actions(self):
        for mutation in (
            {"label":0},{"source_compiles":2},{"role":"dev"},
            {"family":"not_a_family"},{"source_file":"MathCan/GramianTransfer.lean"}):
            rows=sample_rows()
            rows[0].update(mutation)
            with self.assertRaises(ValueError):
                VerifiedCurriculum(rows,seed=17)
        with self.assertRaises(ValueError):
            VerifiedCurriculum(sample_rows()[:-12],seed=17)

    def test_registered_macro_fraction(self):
        sample=sample_rows();sampler=VerifiedCurriculum(sample,seed=43)
        dummy=dict(before={"goals":["n : Nat\n⊢ n=n"]},label=3,delta=1,script="rfl",family="rfl")
        rng=random.Random(43)
        used=sum(draw_batch([dummy],sampler,i,"task_plus_structural",rng)[1] for i in range(400))
        self.assertEqual(used,2560)
        self.assertEqual(sampler.receipt()["total_draws"],2560)
        other=VerifiedCurriculum(sample,seed=43)
        self.assertEqual(draw_batch([dummy],other,0,"task_only",rng)[1],0)
        self.assertEqual(other.receipt()["total_draws"],0)

if __name__=="__main__":unittest.main()
