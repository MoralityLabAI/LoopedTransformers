"""Independent pre-inference NAN-COMP-04 source, archive and grammar tests."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.nanbeige.nan_comp04 import cohort, source, prompt, parser

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT.parents[1]


class TestExactLateSeries(unittest.TestCase):
    def test_all_registered_integer_coefficient_witnesses(self):
        checked=0
        for m in (3,5):
            for a in source.COEFFICIENTS:
                for c in source.COEFFICIENTS:
                    z=source.certificate(m,a,c)
                    self.assertEqual(z["difference_at_2m"],a*a)
                    self.assertEqual(z["difference_at_2m_plus_2"],2*a*c)
                    n=z["series"]["N"];r=z["series"]["R"]
                    self.assertEqual(n[:2*m],r[:2*m])
                    self.assertEqual(len({tuple(v[:2*m+1]) for v in z["series"].values()}),4)
                    self.assertEqual(source.multiply(z["source"],r),
                                     [1]+[0]*(len(z["source"])-1))
                    checked+=1
        self.assertEqual(checked,200)

    def test_invalid_sources_are_refused(self):
        for m,a,c in ((2,1,1),(3,0,1),(3,1,0),(4,2,2),(3,10,1)):
            with self.assertRaises(ValueError):
                source.make_source(m,a,c)
        self.assertEqual(source.signed_digits(-12),"M 0 1 2")
        self.assertEqual(source.signed_digits(13),"P 0 1 3")


class TestStagedCohort(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=cohort.config()
        cls.dataset=cohort.build(cls.cfg)
        cls.labels={r["case_id"]:r for r in cls.dataset["host"]}

    def test_roots_and_sealed_splits(self):
        d=self.dataset
        self.assertEqual(len(d["roots"]),20)
        self.assertEqual(len(d["public"]),220)
        self.assertEqual({x["split"] for x in d["host"]},{"calibration","development"})
        self.assertEqual(sum(x["split"]=="development" for x in d["host"]),176)
        self.assertEqual(len({x["case_id"] for x in d["public"]}),220)
        self.assertEqual(len({x["source_identity"] for x in d["roots"]}),20)
        self.assertTrue(set(self.cfg["reserved_confirmation_roots"]).isdisjoint(
            x["root"] for x in d["host"]))
        self.assertFalse(self.cfg["BPD_transfer_enabled"])
        self.assertEqual(self.cfg["original_ARG03_168_case_test"][:6],"sealed")

    def test_each_root_has_identical_low_information_pairs_and_separate_cross(self):
        groups={}
        for row in self.dataset["public"]:
            groups.setdefault(row["root_group"],[]).append(row)
            self.assertEqual(set(row),cohort.PUBLIC_KEYS)
            self.assertNotIn("truth",row)
            self.assertEqual(row["public_sha256"],cohort.fingerprint(
                {k:v for k,v in row.items() if k not in ("case_id","public_sha256")}))
        for rows in groups.values():
            self.assertEqual(len(rows),11)
            by_tier={}
            for r in rows:
                by_tier.setdefault((r["kind"],r["tier"]),[]).append(r)
            self.assertEqual(len(by_tier["classification","first"]),4)
            for tier in ("early","cross","erased"):
                self.assertEqual(len(by_tier["classification",tier]),2)
            self.assertEqual(len(by_tier["coefficient","query_cross"]),1)
            for tier in ("early","erased"):
                a,b=by_tier["classification",tier]
                self.assertEqual(prompt.body(a),prompt.body(b))
                self.assertEqual(a["public_sha256"],b["public_sha256"])
            mid=by_tier["classification","first"]
            self.assertEqual(len({prompt.body(r) for r in mid}),4)
            a,b=by_tier["classification","cross"]
            self.assertNotEqual(prompt.body(a),prompt.body(b))
            for row in rows:
                self.assertNotIn(row["case_id"],prompt.body(row))
                self.assertNotIn(row["root_group"],prompt.body(row))
                self.assertEqual(self.labels[row["case_id"]]["public_sha256"],
                                 row["public_sha256"])
            num=by_tier["coefficient","query_cross"][0]
            self.assertEqual(self.labels[num["case_id"]]["oracle_answer"],
                             2*num["query"][num["m"]]*num["query"][num["m"]+2])

    def test_source_output_isolated_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"fresh_dev"
            r=cohort.save(out,self.dataset)
            self.assertEqual(r["worlds"],220)
            self.assertFalse(r["original_ARG03_168_test_opened"])
            self.assertFalse(r["reserved_confirmation_sources_generated"])
            self.assertEqual(len((out/"public.jsonl").read_text().splitlines()),220)
            with self.assertRaises(FileExistsError):
                cohort.save(out,self.dataset)


class TestBoundedAnswerParser(unittest.TestCase):
    def test_operator_parser_unchanged(self):
        self.assertEqual(parser.parse("**R**","classification")["operator"],"R")
        self.assertEqual(parser.parse("**","classification")["status"],"pending")
        self.assertEqual(parser.parse("R then N","classification")["status"],"rejected")

    def test_numeric_parser_is_anchored_and_waits_for_closing_bracket(self):
        for a in ("[12]","**[-12]**","[+12]"):
            result=parser.parse(a,"coefficient")
            self.assertEqual(result["status"],"accepted")
            self.assertEqual(result["value"],-12 if "-" in a else 12)
        for s in ("[","**[","[-1","**[+12]"):
            self.assertEqual(parser.parse(s,"coefficient")["status"],"pending")
        for s in ("12","[12] and [13]","[1234]","[1] nonsense","foo"):
            self.assertEqual(parser.parse(s,"coefficient")["status"],"rejected")


class TestArchiveIntegrity(unittest.TestCase):
    def test_colocated_frozen_preexisting_data_are_identical_git_blobs(self):
        manifest=json.loads((ROOT/"archive"/"PROVENANCE.json").read_text())
        assert (ROOT/"NAN_COMP03_RESEARCH_NOTE.md").is_file()
        files=[*manifest["files"],manifest["note"]]
        for spec in files:
            original=REPO/spec["source"]
            copied=REPO/spec.get("archive",spec.get("copy"))
            self.assertTrue(original.is_file(),str(original))
            self.assertEqual(original.read_bytes(),copied.read_bytes())
            raw=copied.read_bytes()
            git=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
            self.assertEqual(git,spec["git_blob_sha1"])


if __name__=="__main__":
    unittest.main()
