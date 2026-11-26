"""Unit and mocked native-meter regression tests; no Lean cohort is opened."""
import tempfile
import unittest
from pathlib import Path

import campaign
from portfolio import ARMS, ArmReceipt, ExactMenuCache, caps, choose, headroom


class PortfolioTests(unittest.TestCase):
    def test_budget_scaling(self):
        self.assertEqual(caps(300), (300, 1200, 512_000_000))
        self.assertEqual(caps(1200), (1200, 4800, 2_048_000_000))
        self.assertEqual(caps(75), (75, 300, 128_000_000))
        with self.assertRaises(ValueError):
            caps(301)

    def test_cache_immutable_and_bounded(self):
        cache = ExactMenuCache(64)
        cache.cap = 80  # Small test limit, same byte-accounting implementation.
        menu = [(1.0, "close", "", "rfl")]
        cache.put(1, menu)
        retrieved = cache.get(1)
        retrieved.clear()
        self.assertEqual(cache.get(1), menu)
        for i in range(2, 10):
            cache.put(i, menu)
        self.assertLessEqual(cache.peak, cache.cap)
        self.assertGreater(cache.evictions, 0)
        self.assertIsNone(cache.get(1))
        disabled = ExactMenuCache(0)
        disabled.put(1, menu)
        self.assertIsNone(disabled.get(1))
        self.assertEqual(disabled.bytes, 0)

    def test_round_robin_and_live_filter(self):
        r = {a: ArmReceipt() for a in ARMS}
        self.assertEqual([choose("rr", ARMS, ARMS, r, i) for i in range(4)], list(ARMS))
        self.assertEqual(choose("rr", ARMS, ["s3"], r, 0), "s3")
        with self.assertRaises(ValueError):
            choose("rr", ARMS, [], r, 0)

    def test_ucb_observation_only(self):
        r = {a: ArmReceipt() for a in ARMS}
        for a in ARMS:
            self.assertEqual(choose("ucb", ARMS, ARMS, r, 0), a)
            r[a].observe({"ok": True, "ngoals": 1 if a == "s3" else 0,
                          "depth": 0}, 1000, 3)
        self.assertEqual(choose("ucb", ARMS, ARMS, r, 0), "s3")
        self.assertEqual(r["s1"].opened, 0)
        r["s3"].observe({"ok": True, "ngoals": 1, "depth": 2}, 1000, 3)
        self.assertEqual(r["s3"].opened, 1)

    def test_headroom_pairing_and_exact_threshold(self):
        def rows(solved):
            return [{"id":str(i),"file":str(i//10),"solved":i in solved}
                    for i in range(120)]
        a, b = rows(set(range(60))), rows(set(range(48,108)))
        ref = rows(set(range(72)))
        table = {"a":a,"b":b,"long":ref}
        g = headroom(table,"long",["a","b"])
        self.assertTrue(g["passes"])
        self.assertEqual(g["offline_control_union"],108)
        self.assertAlmostEqual(g["margin"],.1)
        self.assertFalse(headroom({**table,"long":rows(set(range(71)))},"long",["a","b"])["passes"])
        with self.assertRaises(ValueError):
            headroom({**table,"b":b[:-1]},"long",["a","b"])
        with self.assertRaises(ValueError):
            headroom({**table,"b":b+[b[0]]},"long",["a","b"])

    def test_reserved_and_spec_isolation(self):
        self.assertTrue(campaign.reserved("src/Init/Data/SInt.lean"))
        self.assertTrue(campaign.reserved("src/Init/Data/SInt/Basic.lean"))
        self.assertFalse(campaign.reserved("src/Init/Data/UInt/Basic.lean"))
        self.assertEqual(len(campaign.specs("spent")),40)
        self.assertEqual(len(campaign.specs("fresh-calibration")),15)
        self.assertTrue(all(s[3]==3 and not s[4] for s in campaign.specs("fresh-calibration")))
        with self.assertRaises(ValueError):
            campaign.search(None,None,{"role":"evaluation"},None,"s1",300,0)

    def test_roster_rejects_spent_reserved_and_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            files=[f"src/Init/Data/Test/Nested/F{i}.lean" for i in range(12)]
            for f in files:
                p=root/f
                p.parent.mkdir(parents=True,exist_ok=True)
                p.touch()
            inv={"spent_files":{},"reserved_files":[]}
            fams=campaign.validate_roster(files,[],inv,root)
            self.assertIn("Nested/F0",fams["Test"])
            with self.assertRaises(ValueError):
                campaign.validate_roster(files,[],{**inv,"spent_files":{files[0]:[]}},root)
            with self.assertRaises(ValueError):
                campaign.validate_roster(files+[files[0]],[],inv,root)
            with self.assertRaises(ValueError):
                campaign.validate_roster(files[1:]+["src/Init/Data/SInt/Basic.lean"],[],inv,root)


class MockServer:
    def __init__(self, timeout=False, certify=True):
        self.calls=[]
        self.timeout=timeout
        self.certify=certify

    def request(self, op, **kwargs):
        self.calls.append((op,kwargs))
        base={"request_hb_raw":5,"request_ms":1}
        if op=="premises":
            if self.timeout:
                return {**base,"ok":False,"error":"maximum number of heartbeats"}
            return {**base,"ok":True,"rw":[],"apply":[]}
        if op=="step":
            return {**base,"ok":True,"ngoals":0,"sid":99}
        if op=="cert":
            return {**base,"ok":True,"cert":{"kernel":self.certify,"axioms":[]}}
        raise AssertionError(op)


class NativeAdapterTests(unittest.TestCase):
    root={"sid":1,"info":{"hyps":[],"defs":[]}}

    def test_shared_cache_reuses_menu_only(self):
        server=MockServer()
        m=campaign.old.Meter(server,300,1200,512_000_000)
        cache=ExactMenuCache(64)
        a=campaign.Queue(m,self.root,"s1",{},cache,3,False)
        b=campaign.Queue(m,self.root,"s2",{},cache,3,False)
        self.assertEqual([x[0] for x in server.calls],["premises"])
        a.step()
        b.step()
        self.assertEqual([x[0] for x in server.calls],
                         ["premises","step","cert","step","cert"])
        self.assertEqual(m.report()["steps"],2)
        self.assertEqual(m.report()["rpcs"],5)
        self.assertIsNotNone(a.path)
        self.assertIsNotNone(b.path)

    def test_timeout_is_charged_and_not_cached(self):
        server=MockServer(timeout=True)
        m=campaign.old.Meter(server,300,1200,512_000_000)
        cache=ExactMenuCache(64)
        a=campaign.menu_for(m,self.root,cache)
        b=campaign.menu_for(m,self.root,cache)
        self.assertEqual(a,b)
        self.assertTrue(a)
        self.assertEqual(m.report()["rpcs"],2)
        self.assertEqual(cache.hits,0)
        self.assertEqual(cache.bytes,0)

    def test_kernel_rejection_is_not_a_win(self):
        server=MockServer(certify=False)
        m=campaign.old.Meter(server,300,1200,512_000_000)
        q=campaign.Queue(m,self.root,"s1",{},ExactMenuCache(0),3,False)
        q.step()
        self.assertIsNone(q.path)
        self.assertEqual(m.report()["rpcs"],3)


if __name__=="__main__":
    unittest.main()
