"""LBF-09 source invariance and genuine equal-shortest proposition branches."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile
import unittest

from lcgrt import fragment as F
from ppt_gym.data import sha256_file
from ppt_gym.live_suite import create_suite, load_suite


class Lbf09EqualCostSource(unittest.TestCase):
    def test_default_generator_unchanged_by_opt_in_flag(self):
        for family in ("chain-dead", "alt-deep", "tree-cycle"):
            original = F.generate(109, 3, family, tid="lbf09_default_test")
            explicit = F.generate(109, 3, family, tid="lbf09_default_test",
                                  equal_alt_p=0.0)
            self.assertEqual(original.to_json(), explicit.to_json())
            self.assertEqual(F.statement(original), F.statement(explicit))
            self.assertEqual(F.canonical_hash(original), F.canonical_hash(explicit))

    def test_finite_distinct_proposition_twins_preserve_shortest_cost(self):
        for family in ("chain-dead", "tree-cycle", "alt-deep"):
            source=F.generate(57,3,family,tid="lbf09_twins",equal_alt_p=0.0)
            new=F.generate(57,3,family,tid="lbf09_twins",equal_alt_p=1.0)
            self.assertEqual(source.goal,new.goal)
            self.assertEqual(F.reference_sizes(source)[source.goal[0]],
                             F.reference_sizes(new)[new.goal[0]])
            self.assertGreater(new.n_atoms,source.n_atoms)
            self.assertEqual(new.gen["lbf09_equal_cost"]["p"],1.0)
            self.assertGreater(new.gen["lbf09_equal_cost"]["shortest_rule_branch_twins"],0)
            self.assertGreater(new.gen["lbf09_equal_cost"]["alternate_fact_witnesses"],0)
            self.assertTrue(all(len(pre)<=2 for pre,_ in new.rules))
            self.assertNotEqual(F.canonical_hash(source),F.canonical_hash(new))

            sizes=F.reference_sizes(new)
            rule_choices={}
            for idx,(pre,head) in enumerate(new.rules):
                if all(sizes[x]<F.INF for x in pre) and 1+sum(sizes[x] for x in pre)==sizes[head]:
                    rule_choices.setdefault(head,[]).append((idx,tuple(pre)))
            self.assertTrue(any(len(choices)>=2 and
                                len({p for _,p in choices})>=2
                                for choices in rule_choices.values()))

    def test_distinct_one_step_named_fact_proofs(self):
        base=F.generate(12,3,"chain-dead",tid="lbf09_fact_duality",equal_alt_p=0)
        t=F.generate(12,3,"chain-dead",tid="lbf09_fact_duality",equal_alt_p=1)
        facts=Counter(t.facts)
        # The original leaf facts are duplicated. Newly cloned premise facts
        # can appear once because they are distinct new propositions.
        self.assertTrue(all(facts[atom]>=2 for atom in base.facts))
        fact=base.facts[0]
        options=F.reference_legal(t,fact)
        self.assertGreaterEqual(len([x for x in options if x.startswith("apply h")]),2)

    def test_rejects_invalid_augmentation(self):
        for p in (-.1,1.01,float("nan")):
            with self.assertRaises(ValueError):
                F.generate(7,3,"chain-dead",equal_alt_p=p)

    def test_suite_frozen_lineage_and_immutable_equal_source_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/"suite"
            freeze=create_suite(folder,seed=20261130,per_split=3,equal_alt_p=1.0)
            self.assertEqual(freeze["equal_alt_p"],1.0)
            self.assertEqual(freeze["source_family"],"lbf09_true_equal_cost_premise_twins.v1")
            for shard in ("train","calibration","ordinary_test","ood_family"):
                cases,info=load_suite(folder,shard)
                self.assertEqual(len(cases),3)
                self.assertEqual(info["equal_alt_p"],1.0)
                self.assertTrue(all(c.equal_alt_p==1.0 for c in cases))
                self.assertTrue(all(c.theorem().gen["lbf09_equal_cost"]["shortest_rule_branch_twins"]>0 for c in cases))
            with (folder/"calibration.jsonl").open("a") as f:
                f.write("{}\n")
            with self.assertRaisesRegex(ValueError,"bytes changed"):
                load_suite(folder,"train")

    def test_legacy_suite_source_format_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/"legacy"
            m=create_suite(folder,seed=73,per_split=2)
            self.assertNotIn("equal_alt_p",m)
            cases,_=load_suite(folder,"train")
            self.assertTrue(all(c.equal_alt_p==0 for c in cases))


if __name__=="__main__":
    unittest.main()
