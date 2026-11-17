"""CPU-only independent source-oracle tests; sealed ARG-03 test never loaded."""
from __future__ import annotations

from hashlib import sha256
from itertools import product
from pathlib import Path
import unittest

from mechinterp.dark_fiber_gyms.arg03_preunseal.prepare_final_r import make_fresh, public
from mechinterp.dark_fiber_gyms.arg03_preunseal.score_final_r_future import (
    FIT, VALID, evaluate, score_one, guard_dev,
)


class DummyCase:
    def __init__(self, identifier, source_graph, family):
        self.identifier = identifier
        self.source_graph = source_graph
        self.family = family
        self.signature = sha256(identifier.encode()).hexdigest()
        self.public = {
            "case_id": identifier, "topology": "fork", "observed_degree": 12,
            "demos": [{}, {}], "query": {"synthetic": True},
        }


class DummyOracle:
    CAL_FAMILIES = ("signed",)
    PROGRAMS = tuple(product("INSR", repeat=3))
    TOPOLOGIES = ("fork",)

    def roster(self):
        return {"calibration": []}

    def make_case(self, rng, *, identifier, family, graph, **kw):
        return DummyCase(identifier, graph, family)

    def search(self, case, ordered, budget):
        if budget < 1 or len(ordered) < 1:
            raise ValueError("empty program selection")
        return {"graph": tuple(ordered[0])}

    def grade(self, case, search_result):
        correct = search_result["graph"] == case.source_graph
        return {"future_correct": correct, "accepted": True}


class DummyARG02:
    @staticmethod
    def dev_cases():
        return []
    @staticmethod
    def shot_cases():
        return []


def _fixtures(oracle, cohorts):
    pub,host,tapes = {},[],{}
    raw = [[[0.0] * 4 for _ in range(3)] for _ in range(16)]
    for split,cases in cohorts.items():
        pub[split] = [public(c) for c in cases]
        tapes[split] = []
        for c in cases:
            index = oracle.PROGRAMS.index(c.source_graph)
            host.append({"case_id":c.identifier,"split":split,
                         "source_sha256":c.signature,"gold_program_index":index})
            baseline = {str(t): [float(i==index) if t==2 else float(i==0)
                                for i in range(64)]
                        for t in (1,2,4,8)}
            tapes[split].append({
                "case_id":c.identifier,"public_sha256":c.signature,
                "loop1_operator_site_logits":raw,
                "baseline":baseline, "open_loop":{"2":baseline["1"]},
            })
    return pub,host,tapes


class TestNewDevOracleGrading(unittest.TestCase):
    def test_future_12_requires_source_oracle_and_frozen_fit(self):
        oracle = DummyOracle()
        cohorts,_ = make_fresh(oracle,DummyARG02(),set(),per_program=1)
        self.assertEqual({k:len(v) for k,v in cohorts.items()},{FIT:8,VALID:8})
        pub,host,tapes = _fixtures(oracle,cohorts)
        report = evaluate(oracle,cohorts,pub,host,tapes)
        self.assertFalse(report["sealed_test_loaded"])
        for split in (FIT,VALID):
            d = report["splits"][split]
            self.assertEqual(d["baseline@1"]["future_correct_at12"],0)
            self.assertEqual(d["baseline@2"]["future_correct_at12"],8)
            self.assertEqual(d["baseline@2"]["top1_future_correct"],8)
            self.assertEqual(d["baseline@2"]["paired_vs_baseline1"]["paired_diff"],8)
            self.assertIn("bias12_devfit",d)
        self.assertEqual(report["source_program_rank"]["frozen_bias"]["fit_split"],FIT)
        corrupted={k:list(v) for k,v in tapes.items()}
        corrupted[VALID]=[dict(x) for x in corrupted[VALID]]
        corrupted[VALID][0]["public_sha256"]="invalid"
        with self.assertRaises(ValueError):
            evaluate(oracle,cohorts,pub,host,corrupted)

    def test_future_grader_rejects_test_paths(self):
        for path in ("runs/test.jsonl","runs/test/inputs.jsonl",
                     "runs/heldout/data.json","runs/test_full168/grade.json"):
            with self.assertRaises(ValueError):
                guard_dev(Path(path))
        guard_dev(Path("runs/development_finalR_valid_public.jsonl"))




class TestTimeLocalOracleGrade(unittest.TestCase):
    def test_matched_dev_continuations_receive_exact_future_grades(self):
        from mechinterp.dark_fiber_gyms.arg03_preunseal.score_tlocal_future import evaluate as score_local
        oracle=DummyOracle()
        cohorts,_=make_fresh(oracle,DummyARG02(),set(),per_program=1)
        cases=cohorts[VALID]
        public_rows=[public(c) for c in cases]
        tape=[]
        for c in cases:
            index=oracle.PROGRAMS.index(c.source_graph)
            sc_clean=[0.0]*64
            sc_good=[-1.0]*64
            sc_good[index]=5.0
            slots={}
            for depth in ("2","4","8"):
                slots[depth]={
                  "pre_head_logprob_error":{"structured":0.0,"random_matched":0.0},
                  "matched_l2_dose":{"structured":0.3,"random_matched":0.3,
                                     "relative_error":0.0},
                  "next_64_program_scores":{"clean":sc_clean,
                                             "structured":sc_good,
                                             "random_matched":sc_clean},
                  "next_full_vocab_l2":{"structured":2.0,"random_matched":1.0}
                }
            tape.append({"case_id":c.identifier,"source_split":"development_only",
                         "public_sha256":c.signature,
                         "historical_h2a_unchanged":True,"depths":slots})
        report=score_local(oracle,cases,public_rows,tape)
        self.assertFalse(report["sealed_test_loaded"])
        self.assertEqual(report["n"],8)
        for depth in ("2","4","8"):
            r=report["depths"][depth]
            self.assertEqual(r["eligible"],8)
            self.assertEqual(r["structured"]["future_correct_at12"],8)
            self.assertEqual(r["random_matched"]["future_correct_at12"],0)
            self.assertEqual(r["structured_vs_random_mcnemar"]["paired_diff"],8)
        bad=[dict(x) for x in tape]
        bad[0]=dict(bad[0],source_split="test")
        with self.assertRaises(ValueError):
            score_local(oracle,cases,public_rows,bad)


if __name__=="__main__":
    unittest.main()
