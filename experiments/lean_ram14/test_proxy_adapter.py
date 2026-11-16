"""Source-only RAM14 adapter tests. No private models or imported Lean data."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from proxy_adapter import RAM08_FAMILIES,MAX_EXTRA,extend,family_proxy,\
    fingerprint,safe_extended_script


def observed():
    return [("omega","omega"),("rfl","rfl"),("induction_list","induction xs")]


def extensions():
    return [
      dict(family="nat_induction",
           script="induction n generalizing x <;> simp_all [iterateState, transport, g.left_inv]",
           witnesses=["binder:n:Nat","generalizing:x","prior_or_field:iterateState",
                      "prior_or_field:transport","prior_or_field:g.left_inv"]),
      dict(family="rewrite_prior",
           script="simp [transport_iterate, g.left_inv]",
           witnesses=["earlier_theorem:transport_iterate","field:g.left_inv"]),
      dict(family="rewrite_prior",
           script="rw [transport_iterate]",
           witnesses=["earlier_theorem:transport_iterate"]),
      dict(family="simp_prior",
           script="simp_all [iterateState, transport]",
           witnesses=["prior_or_field:iterateState","prior_or_field:transport"])
    ]


class Adapter(unittest.TestCase):
    def test_frozen_family_names_equal_20(self):
        self.assertEqual(len(RAM08_FAMILIES),20)
        self.assertEqual(len(set(RAM08_FAMILIES)),20)
        self.assertIn("induction_list",RAM08_FAMILIES)
    def test_proxy_and_provenance(self):
        old=observed()
        after,receipt=extend(old,extensions())
        self.assertEqual(after[:len(old)],old)
        self.assertEqual(len(receipt),4)
        self.assertEqual([x.embedding_family for x in receipt],
                         ["induction_list","simp_all_only_visible","rw_local",
                          "simp_all_only_visible"])
        self.assertTrue(all(x.declared_posthoc_origin for x in receipt))
    def test_model_family_ids_remain_compatible(self):
        out,receipt=extend(observed(),extensions())
        self.assertTrue(all(f in RAM08_FAMILIES for f,s in out))
        self.assertTrue(all(s in [p[1] for p in out] for s in [
            "simp [transport_iterate, g.left_inv]",
            "induction n generalizing x <;> simp_all [iterateState, transport, g.left_inv]"]))
    def test_invalid_source_script_rejected(self):
        bad=[
           ("nat_induction","induction n generalizing x <;> sorry"),
           ("rewrite_prior","run_tac unsafe"),
           ("simp_prior","simp [foo]; exact h"),
           ("rewrite_prior","rw [later]; sorry"),
           ("nat_induction","induction n generalizing x\n  exact h"),
        ]
        for fam,script in bad:
            self.assertFalse(safe_extended_script(fam,script),script)
    def test_refuses_missing_source_witnesses(self):
        altered=extensions()
        altered[0]=dict(altered[0],witnesses=[])
        with self.assertRaises(ValueError):extend(observed(),altered)
    def test_registered_cap(self):
        out,receipts=extend(observed(),extensions(),extra_cap=2)
        self.assertEqual(len(receipts),2)
        self.assertEqual(len(out),len(observed())+2)
        with self.assertRaises(ValueError):extend(observed(),extensions(),extra_cap=MAX_EXTRA+1)
    def test_duplicate_scripts_are_suppressed(self):
        cases=extensions()
        cases.append(cases[0])
        out,receipts=extend(observed(),cases)
        self.assertEqual(len(receipts),4)
        self.assertEqual(len(out),len(observed())+4)
    def test_original_source_pool_not_modified(self):
        original=observed()
        copy=list(original)
        extend(original,extensions())
        self.assertEqual(original,copy)
    def test_fingerprint_changes_with_candidate(self):
        a,p=extend(observed(),extensions())
        v=fingerprint(observed(),extensions(),p)
        self.assertEqual(v,fingerprint(observed(),extensions(),p))
        changed=extensions()
        changed[0]=dict(changed[0],witnesses=["binder:n:Nat","generalizing:x"])
        b,q=extend(observed(),changed)
        self.assertNotEqual(v,fingerprint(observed(),changed,q))


if __name__ == "__main__":
    unittest.main()
