import unittest
import torch
from ppt_gym import olr07b_cpu as B
from ppt_gym import olr07a_library as A

def fake(i):
    rows=[]
    for j,t in enumerate(B.MENU):
        rows.append({"tactic":t,"ok":j==i%4,"goals_after":0 if j==i%4 else 2,
                     "cert":{"kernel":True,"axioms":[]} if j==i%4 else {}})
    return {"id":str(i),"role":"train","file":f"file{i%2}","family":"fixture","goal":"x : Nat\n⊢ "+str(i),
            "receipts":rows,"scored":True,"human_closed":True}

class LocalGymTests(unittest.TestCase):
    def test_receipt_mask_no_unseen_features(self):
        g,r,y=B.row_features(fake(0));_,x,_,mask=B.frame(g,r,y,(1,3))
        self.assertTrue(torch.equal(x[1],r[1]));self.assertTrue(mask[0])
        self.assertEqual(int((~mask).sum()),2)
        self.assertTrue(torch.equal(x[0,1:],torch.zeros_like(x[0,1:])))
        self.assertEqual(float(x[0,0]),1.)
    def test_train_all_readers_and_paid_policy(self):
        torch.set_num_threads(1)
        data=B.dataset([fake(i) for i in range(4)])
        hashes=[]
        for arm in B.ARMS:
            m,meta=B.fit(data,17,arm,steps=3)
            if arm!="direct":hashes.append((meta["initial_sha256"],meta["sampler_sha256"]))
            e=B.neural_episode(m,fake(1),17,arm)
            self.assertLessEqual(e["calls"],4);self.assertEqual(e["calls"],len({x["action"] for x in e["events"]}))
        self.assertEqual(len(set(hashes)),1)
    def test_classical_receipts(self):
        rows=[fake(i) for i in range(8)]
        e,s,b=B.classical(rows,rows)
        self.assertEqual(len(e),len(A.ARMS)*len(rows));self.assertIn(b,A.ARMS)
        self.assertTrue(all(x["closed"] for x in e if x["arm"]=="fixed"))
    def test_roster_disjoint_and_wrapper_exclusion(self):
        paths=[p for _,_,p in B.choose_files("full")]
        self.assertEqual(len(paths),len(set(paths)))
        raw=[{"decl":"d","line":1,"col":0,"tactic":"decreasing_by omega","goals_before":["G"],"goals_after":[]}]
        rep={"decls":[{"decl":"d","in_env":True,"axioms":[]}]}
        self.assertFalse(list(B.eligible(raw,rep,"fixture",set())))
if __name__=="__main__":unittest.main()
