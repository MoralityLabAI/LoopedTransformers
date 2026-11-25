import itertools
import unittest
from ld_moe_cache import coords, normalize_events, replay, SLOT_ORDER


def fixture(n=2):
    support=[list(range(12)) for _ in range(24)]
    out=[]
    for t in range(n):
        for slot in SLOT_ORDER:
            l,p=coords(slot)
            ids=[(t+p+x)%12 for x in range(8)]
            scores=[.8 if i in ids else .1 for i in range(12)]
            out.append(dict(token=t,phase="prefill" if t==0 else "decode",slot=slot,
                layer=l,**{"pass":p},experts=ids,scores=scores,rank_scores=scores,
                selected_scores=[.8]*8,scaled_weights=[1.]*8))
    return out,support


def small(sets):
    return [dict(layer=0,phase="decode",experts=list(s),**{"pass":0}) for s in sets]


class CacheTests(unittest.TestCase):
    def test_remap_and_coverage(self):
        e,s=fixture()
        s=[list(range(20,32)) for _ in s]
        n=normalize_events(e,s,1,1)
        self.assertEqual(n[0]["experts"],list(range(20,28)))
        self.assertEqual(n,normalize_events(n,s,1,1,original_ids=True))
        with self.assertRaises(ValueError):normalize_events(e[:-1],s,1,1)
        e[0]["scaled_weights"][0]=.5
        with self.assertRaises(ValueError):normalize_events(e,s,1,1)

    def test_pass_order(self):
        e,s=fixture()
        e[18],e[19]=e[19],e[18]
        with self.assertRaises(ValueError):normalize_events(e,s,1,1)

    def test_capacity_active_set(self):
        e,s=fixture()
        n=normalize_events(e,s,1,1)
        a=replay(n,8,"lru")
        for p in ("ema","previous_pass","oracle"):
            self.assertEqual(a["admissions"],replay(n,8,p)["admissions"])
        b=replay(n,12,"oracle")
        self.assertLess(b["admissions"],a["admissions"])
        self.assertEqual(b["estimated_bytes"],sum(x["admissions"]*size for x,size in zip(b["layers"],[4810752]*20+[7174656]*4)))
        self.assertEqual(b["phases"]["prefill"]["visits"],48)
        self.assertEqual(b["phases"]["decode"]["visits"],48)

    def test_belady_against_exact_small_dp(self):
        # Enumerate 3^5 traces of two-expert visits, capacity three over four
        # expert IDs. Dynamic program considers every valid demand eviction.
        choices=[{0,1},{1,2},{0,3}]
        for seq in itertools.product(choices,repeat=5):
            dp={frozenset():0}
            for req in seq:
                after={}
                for cache,cost in dp.items():
                    pool=cache|req
                    for keep in itertools.combinations(pool-req,min(3-len(req),len(pool-req))):
                        state=frozenset(req|set(keep))
                        after[state]=min(after.get(state,10**9),cost+len(req-cache))
                dp=after
            got=replay(small(seq),3,"oracle")["admissions"]
            self.assertEqual(got,min(dp.values()))

    def test_forecast_keeps_native(self):
        e=small([{0,1},{1,2},{0,3}])
        f=[[0.]*128 for _ in e]
        f[1][0]=10.
        a=replay(e,3,"forecast",forecasts=f)
        self.assertEqual(a["phases"]["decode"]["requests"],6)
        self.assertEqual(a["admissions"],4)


if __name__=="__main__":unittest.main()
