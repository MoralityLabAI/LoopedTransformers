import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import controller as C
A=C.A

class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.source=("namespace "+A.NS+"\n"
                     "theorem demo_monitor_correct (s : Bool × Bool) (a : Bool) : True := by trivial\n"
                     "end "+A.NS+"\n")
        self.actions=["unpack","enumerate"]
        self.request={"observation":1,"action":True,"root":C.ROOT,"actions":self.actions}
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)
        self.out=self.path/"evidence"
        self.out.mkdir()
        with patch.object(A,"checkout",return_value=self.source):
            self.controller=C.Controller(self.path,"compatible")

    def synthetic_receipt(self):
        one,two,header,_=A.render(self.source,C.ROOT,self.actions)
        receipt=dict(schema="LRF03/v1",source_commit=A.COMMIT,source_blob=A.BLOB,
            root=C.ROOT,actions=self.actions,fixture=None,classification="verified_target_proof",
            header_sha256=header,reward=1,kernel=True,foundation=True,contract=True,
            compiler_calls=2,deadline_ms=A.DEADLINE*1000,elapsed_ms=3.,allowlist=sorted(A.ALLOW),axioms=[])
        for key,src,names in (("first",one,[A.SUB]),("second",two,[A.SUB,A.CON])):
            log="".join(f"'{name}' does not depend on any axioms\n" for name in names)
            (self.out/(key+".lean")).write_text(src)
            (self.out/(key+".log")).write_text(log)
            receipt[key]=dict(status="compiled",returncode=0,reports={n:[] for n in names},
                source_sha256=A.sha(src),log_sha256=A.sha(log))
        (self.out/"receipt.json").write_text(json.dumps(receipt))
        return receipt

    def test_finite_truth_table(self):
        table=[C.policy(mode,o,a)[0] for mode in C.MODES for o in (0,1,2) for a in (False,True)]
        self.assertEqual(table,[False,False,False,True,False,False]+[False]*6)
        self.assertEqual(C.policy("collision",1,True),(False,"authorization_collision"))
        self.assertEqual(C.policy("compatible",2,True),(False,"unknown_observation"))

    def test_preflight_denies_without_execution(self):
        for change in ({"observation":2},{"observation":0},{"action":False},{"observation":True},
                       {"root":"factor_of_section"},{"actions":["arbitrary Lean"]},{"receipt":{}}):
            with patch.object(A,"evaluate_actions",side_effect=AssertionError("must not execute")) as f:
                row=self.controller.review({**self.request,**change},self.out)
                self.assertFalse(row["allowed"])
                self.assertEqual(row["compiler_calls"],0)
                f.assert_not_called()
        with patch.object(A,"checkout",return_value=self.source):
            collision=C.Controller(self.path,"collision")
        with patch.object(A,"evaluate_actions") as f:
            self.assertEqual(collision.review(self.request,self.out)["reason"],"authorization_collision")
            f.assert_not_called()

    def test_faults_fail_closed(self):
        original=self.synthetic_receipt()
        self.assertTrue(C._trusted_receipt_valid(original,self.source,self.actions,self.out))
        faults={"source_commit":"stale","source_blob":"bad","header_sha256":"bad","actions":["skip"],
                "root":"another","reward":True,"kernel":False,"foundation":False,"contract":None,
                "compiler_calls":1,"elapsed_ms":float("inf"),"deadline_ms":999999,
                "classification":"deadline_unknown","axioms":["Classical.choice"],"fixture":"unused_axiom"}
        for field,value in faults.items():
            bad=copy.deepcopy(original);bad[field]=value
            (self.out/"receipt.json").write_text(json.dumps(bad))
            self.assertFalse(C._trusted_receipt_valid(bad,self.source,self.actions,self.out),field)
        original=self.synthetic_receipt()
        (self.out/"second.log").write_text("tampered")
        self.assertFalse(C._trusted_receipt_valid(original,self.source,self.actions,self.out))

    def test_verifier_errors_and_unknown(self):
        with patch.object(A,"checkout",return_value=self.source):
            with patch.object(A,"evaluate_actions",side_effect=OSError("mock failure")):
                self.assertEqual(self.controller.review(self.request,self.out)["reason"],"verifier_error")
            with patch.object(A,"evaluate_actions",return_value={"classification":"deadline_unknown","reward":None,"compiler_calls":1}):
                self.assertEqual(self.controller.review(self.request,self.out)["reason"],"certificate_denied")
        with patch.object(A,"checkout",return_value="changed"),patch.object(A,"evaluate_actions") as f:
            self.assertEqual(self.controller.review(self.request,self.out)["reason"],"source_changed")
            f.assert_not_called()

if __name__=="__main__":unittest.main()
