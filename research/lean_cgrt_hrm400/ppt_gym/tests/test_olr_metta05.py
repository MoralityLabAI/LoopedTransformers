"""OLR-METTA-05 tests of the restricted observation and matched model interfaces.

The Python tests assert protocol/architecture behavior only. Native proof
claims require the separate real Lean/Hyperon workflow.
"""
import copy
import unittest
from unittest.mock import patch

import torch

from ppt_gym.olr_metta05_native import (
    roots,root_row,from_row,SPLITS,N,lookup,admit,QUERY,
)
from ppt_gym.olr_metta05_model import (
    ReceiptLedger,ExplicitMap,packets,event,flip_receipt,fit,choose,weight_hash,
)


def history(root):
    """Reference fixture; this is not a native Lean receipt."""
    winner={s.addr:s.winner for s in root.slots}
    return [{"addr":a,"branch":b,"ok":b==winner[a]}
            for a,b in zip(root.probe_order,root.probe_branches,strict=True)]


class HistoryProtocolTests(unittest.TestCase):
    def test_frozen_cohort_and_matched_aliases(self):
        bank=roots();seen=set()
        self.assertEqual(bank,roots())
        for split,(count,k) in SPLITS.items():
            self.assertEqual(len(bank[split]),2*count)
            for a,b in zip(bank[split][::2],bank[split][1::2],strict=True):
                self.assertEqual(a.lineage(),b.lineage())
                self.assertNotIn(a.lineage(),seen);seen.add(a.lineage())
                self.assertEqual(a.projection(),b.projection())
                self.assertEqual(a.probe_order,b.probe_order)
                self.assertEqual(a.probe_branches,b.probe_branches)
                self.assertTrue(all(x.winner!=y.winner for x,y in zip(a.slots,b.slots)))
                self.assertEqual(len(a.slots),k)
                self.assertNotEqual(a.statement(),b.statement())
                self.assertEqual(from_row(root_row(a)),a)

    def test_recall_packet_is_identical_and_receipt_labels_opposite(self):
        a,b=roots()["train"][:2]
        h1,q1,t1=packets(history(a),a.recall_order)
        h2,q2,t2=packets(history(b),b.recall_order)
        self.assertTrue(torch.equal(q1,q2))
        self.assertFalse(torch.equal(h1,h2))
        self.assertTrue(all(lookup(history(a))[i]!=lookup(history(b))[i] for i in a.recall_order))
        self.assertEqual(tuple(h1.shape),(8,11))
        self.assertEqual(tuple(q1.shape),(3,11))
        self.assertEqual(tuple(t1.shape),(3,3))

    def test_focal_and_sham_edit_same_single_verdict_magnitude(self):
        a=roots()["train"][0];r=history(a)
        h,q,_=packets(r,a.recall_order)
        h1,q1,_=packets(flip_receipt(r,a.focal),a.recall_order)
        h2,q2,_=packets(flip_receipt(r,a.sham),a.recall_order)
        self.assertTrue(torch.equal(q,q1) and torch.equal(q,q2))
        self.assertEqual(int(torch.count_nonzero(h1-h)),1)
        self.assertEqual(int(torch.count_nonzero(h2-h)),1)
        self.assertEqual(float((h1-h).abs().sum()),2.)
        self.assertEqual(float((h2-h).abs().sum()),2.)
        self.assertEqual(r,history(a))

    def test_no_verdict_on_unadmitted_or_recall_tokens(self):
        for k in ("veto","delay","query"):
            with self.assertRaises(ValueError):
                event(k,0 if k=="query" else None,0,True)

    def test_admission_rejects_native_disagreement(self):
        a=roots()["train"][0];r=history(a)
        with patch("ppt_gym.olr_metta05_native._native_metta",return_value=([],"fake")):
            with self.assertRaises(ValueError):
                admit(r)


class MatchedArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled=False
        torch.use_deterministic_algorithms(True)

    def test_identical_parameterization_and_initialization(self):
        torch.manual_seed(17);a=ReceiptLedger(mode="addressed")
        torch.manual_seed(17);d=ReceiptLedger(mode="dense")
        self.assertEqual(weight_hash(a),weight_hash(d))
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in d.parameters()))
        root=roots()["train"][0]
        h,q,_=packets(history(root),root.recall_order)
        self.assertEqual(a(h[None],q[None]).shape,(1,3,2))
        self.assertEqual(d(h[None],q[None]).shape,(1,3,2))

    def test_erased_memory_cannot_distinguish_pair(self):
        a,b=roots()["train"][:2]
        model=ReceiptLedger().eval()
        p1,v1=choose(model,"addressed",history(a),a.recall_order,erase=True)
        p2,v2=choose(model,"addressed",history(b),b.recall_order,erase=True)
        self.assertEqual(p1,p2)
        self.assertEqual(v1,v2)
        self.assertEqual(sum(x==lookup(history(r))[i] for r in (a,b)
            for i,x in zip(r.recall_order,p1,strict=True)),3)

    def test_source_gradient_and_training_sampler(self):
        rows=[]
        for root in roots()["train"][:4]:
            r=history(root);lab=lookup(r)
            rows.append({"receipts":r,"recall_order":list(root.recall_order),
                         "labels":[lab[i] for i in root.recall_order]})
        a,ra=fit(rows,seed=29,arm="addressed",steps=2,batch_size=4)
        d,rd=fit(rows,seed=29,arm="dense",steps=2,batch_size=4)
        self.assertEqual(ra["initial_sha256"],rd["initial_sha256"])
        self.assertEqual(ra["sampler_sha256"],rd["sampler_sha256"])
        self.assertNotEqual(ra["weights_sha256"],rd["weights_sha256"])
        e,re=fit(rows,seed=29,arm="explicit_mlp",steps=2,batch_size=4)
        self.assertEqual(choose(e,"explicit_mlp",rows[0]["receipts"],rows[0]["recall_order"])[1],0)


if __name__=="__main__":unittest.main()
