"""OLR-08B shared-budget scheduling primitives. Standard library only."""
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import time

ARMS = ("s1", "s2", "s3", "s3c")
BUDGETS = (75, 150, 300, 600, 1200)
TIERS = (0, 64, 256, 1024, 4096)


class ExactMenuCache:
    """Root/session-local immutable menu bytes keyed only by native snapshot ID.

    Never cache tactic outcomes, errors, certificates, or pretty-printed states.
    This is a serialized payload cap, not a claim about total process RSS.
    """
    def __init__(self, mib):
        if mib not in TIERS:
            raise ValueError("unregistered memory tier")
        self.cap = mib * 2**20
        self.items = OrderedDict()
        self.bytes = self.peak = self.hits = self.misses = self.evictions = 0
        self.seconds = 0.0

    def get(self, sid):
        t = time.perf_counter()
        try:
            if sid not in self.items:
                self.misses += 1
                return None
            self.hits += 1
            self.items.move_to_end(sid)
            return [tuple(a) for a in json.loads(self.items[sid])]
        finally:
            self.seconds += time.perf_counter() - t

    def put(self, sid, menu):
        t = time.perf_counter()
        try:
            if not self.cap:
                return
            data = json.dumps(menu, ensure_ascii=False, separators=(",", ":")).encode()
            if len(data) > self.cap:
                return
            if sid in self.items:
                self.bytes -= len(self.items.pop(sid))
            while self.bytes + len(data) > self.cap:
                _, v = self.items.popitem(last=False)
                self.bytes -= len(v)
                self.evictions += 1
            self.items[sid] = data
            self.bytes += len(data)
            self.peak = max(self.peak, self.bytes)
        finally:
            self.seconds += time.perf_counter() - t

    def report(self):
        return {k: getattr(self, k) for k in
                ("cap", "bytes", "peak", "hits", "misses", "evictions", "seconds")}


@dataclass
class ArmReceipt:
    pulls: int = 0
    opened: int = 0
    heartbeats: int = 0

    def observe(self, event, charged_hb, max_depth):
        self.pulls += 1
        self.heartbeats += charged_hb
        # A bounded progress proxy, not a closure label. No reward for a
        # locally closed state rejected by the kernel.
        self.opened += int(bool(event.get("ok")) and
                           isinstance(event.get("ngoals"), int) and
                           event["ngoals"] > 0 and event["depth"] + 1 < max_depth)


def choose(policy, arms, live, receipts, turn):
    """Deterministic selectors. No oracle outcomes or future events."""
    if not live:
        raise ValueError("no live arms")
    live = set(live)
    if policy == "rr":
        return next(arms[(turn+i) % len(arms)] for i in range(len(arms))
                    if arms[(turn+i) % len(arms)] in live)
    if policy != "ucb":
        raise ValueError("unknown scheduler")
    candidates = [a for a in arms if a in live]
    untried = [a for a in candidates if receipts[a].pulls == 0]
    if untried:
        return untried[0]
    total = sum(r.pulls for r in receipts.values())
    def score(a):
        r = receipts[a]
        mean = r.opened / r.pulls
        # Cost penalty is dimensionless; raw heartbeat accounting remains
        # authoritative in the native meter.
        penalty = 0.05 * math.log1p(r.heartbeats / r.pulls / 100_000)
        return mean - penalty + math.sqrt(2 * math.log(total + 1) / r.pulls)
    return max(candidates, key=score)


def caps(steps):
    if steps not in BUDGETS:
        raise ValueError("unregistered step budget")
    return (steps, 4 * steps, (512_000_000 * steps) // 300)


def headroom(rows_by_arm, reference, controls, min_roots=120, min_files=12):
    """Paired cohort gate; reference is measured on the same roots.

    Gate thresholds are proposals for 08B. They do not amend 08A.
    """
    if not controls or reference in controls:
        raise ValueError("invalid gate arms")
    indexes = {}
    for arm in (*controls, reference):
        rows = rows_by_arm[arm]
        ix = {r["id"]: (r["file"], bool(r["solved"])) for r in rows}
        if len(ix) != len(rows):
            raise ValueError("duplicate root")
        indexes[arm] = ix
    ref = indexes[reference]
    for ix in indexes.values():
        if set(ix) != set(ref) or any(ix[k][0] != ref[k][0] for k in ref):
            raise ValueError("unpaired cohort")
    n = len(ref)
    nfiles = len({v[0] for v in ref.values()})
    counts = {a: sum(v[1] for v in ix.values()) for a, ix in indexes.items()}
    best = max(counts[a] for a in controls)
    long = counts[reference]
    oracle = sum(any(indexes[a][k][1] for a in controls) for k in ref)
    checks = {"hard_roots": n >= min_roots, "file_clusters": nfiles >= min_files,
              "control_ceiling": n > 0 and 5*best <= 4*n,
              "long_margin": n > 0 and 10*(long-best) >= n}
    return {"roots": n, "files": nfiles, "counts": counts,
            "best_control": best, "long_reference": long,
            "offline_control_union": oracle, "margin": (long-best)/max(1,n),
            "portfolio_oracle_margin": (oracle-best)/max(1,n),
            "checks": checks, "passes": all(checks.values())}
