"""Empirical policy accounting tests; no substitute for native execution."""
import unittest
from ppt_gym.olr07a_library import MENU,closed,category,Empirical,fixed_order,episode

def receipt(j,ok=False,error="failed"):
    return {"tactic":MENU[j],"ok":True,"goals_after":0,"cert":{"kernel":True,"axioms":[]}} if ok else {"tactic":MENU[j],"ok":False,"error":error}

class ReceiptTests(unittest.TestCase):
    def test_kernel_and_censoring(self):
        self.assertTrue(closed(receipt(0,True)))
        self.assertFalse(closed({"ok":True,"goals_after":0,"cert":{"kernel":True,"axioms":["sorryAx"]}}))
        self.assertEqual(category({"ok":False,"error":"maximum heartbeats exceeded"}),"resource")
        self.assertEqual(category({"ok":True,"goals_after":2}),"open:2")
    def test_exact_tree_and_paid_actions(self):
        rows=[]
        for i in range(8):
            rows.append({"id":str(i),"goal":"example "+str(i),"role":"evaluation","file":"fixture",
                         "receipts":[receipt(j,j==i%6,error="type mismatch" if i%2 else "no matching") for j in range(len(MENU))]})
        binary=Empirical(rows,False);diag=Empirical(rows,True)
        self.assertGreaterEqual(diag.best(diag.all,0,4)[0],binary.best(binary.all,0,4)[0])
        order=fixed_order(rows)
        for arm in ("fixed","binary","diagnostic","local_binary","local_diagnostic"):
            r=episode(rows[0],arm,order,{"binary":binary,"diagnostic":diag})
            self.assertLessEqual(r["calls"],4)
            self.assertEqual(len({e["action"] for e in r["events"]}),r["calls"])

if __name__=="__main__":unittest.main()
