"""Host-grading and model-runner source gates tested without GPU or checkpoints."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.nanbeige.nan_comp04 import cohort,grade,run


class TestDevOnlyScorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset=cohort.build()
        cls.host={x["case_id"]:x for x in cls.dataset["host"]}
        cls.by_group={}
        for p in cls.dataset["public"]:
            if p["split"]=="development":
                cls.by_group.setdefault(p["root_group"],[]).append(p)
        cls.first=list(cls.by_group.items())[:2]

    def fake_result(self,row):
        label=self.host[row["case_id"]]
        if row["kind"]=="coefficient":
            parsed={"status":"accepted","value":label["oracle_answer"],
                    "reason":"complete_grammar"}
            gen=[1,2,3]
        else:
            choice="N" if row["tier"] in ("early","erased") else label["truth"]
            parsed={"status":"accepted","operator":choice,"reason":"complete_grammar"}
            gen={"I":[11],"N":[12],"S":[13],"R":[14]}[choice]
        return {"case_id":row["case_id"],"public_sha256":row["public_sha256"],
                "kind":row["kind"],"tier":row["tier"],"result":{
                    "kind":row["kind"],"parser":parsed,"generated_ids":gen,
                    "initial_prompt_sha256":cohort.fingerprint({
                        "kind":row["kind"],"tier":row["tier"],"m":row["m"],
                        "degree":row["degree"],"demos":row["demos"],
                        "query":row["query"]}),
                    "tokens_generated":len(gen)
                }}

    def test_exact_grader_uses_host_only_and_counts_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source_dir=base/"source"
            receipt=cohort.save(source_dir,self.dataset)
            self.assertEqual(run.read_public(source_dir)[1]["worlds"],220)
            loc=base/"runs"/"bf16"/"development"
            loc.mkdir(parents=True)
            selected=[group for group,_ in self.first]
            manifest={"status":"completed","backend":"bf16","split":"development",
                      "roots_selected":selected,
                      "original_ARG03_168_test_accessed":False,
                      "source_public_sha256":receipt["files_sha256"]["public.jsonl"]}
            (loc/"manifest.json").write_text(json.dumps(manifest)+"\n")
            for group,items in self.first:
                payload={"root_group":group,"backend":"bf16","split":"development",
                         "source_public_sha256":receipt["files_sha256"]["public.jsonl"],
                         "original_ARG03_168_test_accessed":False,
                         "cases":[self.fake_result(p) for p in items]}
                (loc/("root_"+group+".json")).write_text(json.dumps(payload)+"\n")
            report=grade.grade(source_dir,base/"runs","bf16","development")
            self.assertEqual(report["decision"],"ENGINEERING_PILOT_ONLY")
            self.assertEqual(report["source_root_count"],2)
            self.assertEqual(report["model_case_count"],22)
            self.assertEqual(report["summary"]["counts"]["nr_both"],2)
            self.assertEqual(report["summary"]["counts"]["cross_query"],2)
            self.assertEqual(report["summary"]["counts"]["joint"],2)
            self.assertEqual(report["summary"]["counts"]["early_nr_worlds"],2)
            self.assertEqual(report["summary"]["counts"]["erased_nr_worlds"],2)
            self.assertFalse(report["original_ARG03_168_test_opened"])
            # Exact case SHA and outputs are indispensable to the independent grade.
            corrupted=json.loads((loc/("root_"+selected[0]+".json")).read_text())
            corrupted["cases"][0]["public_sha256"]="bad"
            (loc/("root_"+selected[0]+".json")).write_text(json.dumps(corrupted)+"\n")
            with self.assertRaises(ValueError):
                grade.grade(source_dir,base/"runs","bf16","development")

    def test_hard_seal_and_grammar(self):
        with self.assertRaises(ValueError):
            run.forbid_test_paths(Path("runs/heldout/source.jsonl"))
        with self.assertRaises(ValueError):
            run.group_rows(self.dataset["public"],"test",0)
        with self.assertRaises(RuntimeError):
            # A deliberately absent stage lock should prevent any model process.
            import unittest.mock
            with unittest.mock.patch.object(run,"HERE",Path("/tmp/nan_comp04_lock_absent")):
                run.lock_source()

    def test_no_success_for_undistinguished_early_pair(self):
        root,items=self.first[0]
        targets=[p for p in items if p["tier"]=="early"]
        fake=[self.fake_result(p) for p in targets]
        grades=[grade.score_case(p,self.host[p["case_id"]],t)
                for p,t in zip(targets,fake)]
        self.assertEqual(sum(x["correct"] for x in grades),1)


if __name__=="__main__":
    unittest.main()
