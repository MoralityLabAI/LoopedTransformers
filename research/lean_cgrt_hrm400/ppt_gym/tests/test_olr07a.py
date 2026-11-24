"""Empirical policy accounting tests; no substitute for native execution."""
import unittest
from itertools import combinations
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

    def test_tree_matches_independent_small_finite_horizon(self):
        # Restrict to a tiny independent fixture. Several tactics close the
        # same root and some rejected receipts distinguish subpopulations.
        import random
        rng=random.Random(70111)
        rows=[]
        for i in range(7):
            rs=[]
            for j in range(len(MENU)):
                if j<4 and rng.random()<.30:
                    rs.append(receipt(j,True))
                else:
                    rs.append(receipt(j,error=rng.choice(("type mismatch","no matching","heartbeat","failed"))))
            rows.append({"id":str(i),"goal":"fixture "+str(i),"receipts":rs})
        # Independent recursion on explicit source row indices, using only
        # four legal actions. The implementation represents beliefs by bits.
        def brute(ids,available,left,diag):
            if not ids or not left or not available:return (0,0,-1)
            options=[]
            for j in available:
                branches={};gain=0;cost=len(ids)
                for i in ids:
                    r=rows[i]["receipts"][j]
                    if closed(r):gain+=1
                    else:
                        label=category(r) if diag else "nonclosure"
                        branches.setdefault(label,[]).append(i)
                for b in branches.values():
                    a,c,_=brute(b,[k for k in available if k!=j],left-1,diag)
                    gain+=a;cost+=c
                options.append((gain,cost,j))
            return max(options,key=lambda v:(v[0],-v[1],-v[2]))
        disabled=sum(1<<j for j in range(4,len(MENU)))
        for diag in (False,True):
            tree=Empirical(rows,diag)
            self.assertEqual(tree.best(tree.all,disabled,3),
                             brute(list(range(len(rows))),list(range(4)),3,diag))
            # Unseen receipt histories fall back without re-enabling any
            # tried action. This is a belief backoff, not a proof fact.
            b=tree.reveal(tree.all,[(0,"never_observed_fixture")])
            self.assertEqual(b,tree.all)
            self.assertNotEqual(tree.best(b,disabled|1,2)[2],0)
        order=fixed_order(rows)
        def covered(ids):
            return sum(any(closed(r["receipts"][j]) for j in ids) for r in rows)
        self.assertEqual(covered(order),max(covered(a) for a in combinations(range(len(MENU)),4)))

if __name__=="__main__":unittest.main()
