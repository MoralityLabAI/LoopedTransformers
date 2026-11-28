"""LDT-MOE-02: exact-route demand-cache replay, independent of GPU runtime.

All policies execute the same original selections. This is a traffic model,
not a measured PCIe benchmark. Fixed per-layer slots, uniform cost within each
layer, no speculative loads, a visit's entire active set is pinned until the
visit finishes. Prefill and decode share the cold-start cache.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path

LAYER_BYTES = (4810752,)*20 + (7174656,)*4  # mixed bank payload, not PCIe counters
SLOT_ORDER = tuple(range(18))+tuple(range(24,48))+tuple(range(18,24))


def coords(slot):
    if type(slot) is not int or not 0 <= slot < 48:
        raise ValueError("invalid logical slot")
    return (slot,0) if slot < 24 else (6+(slot-24)%12,1+(slot-24)//12)


def normalize_events(events, support, prefill, decode, *, original_ids=False):
    """Validate single-token native captures; map local expert IDs to original.

    Full score vectors are indexed by support[layer], not by physical ID
    unless the support is the full ordered range(128). Reject phase totals.
    The runtime scores only evaluated inputs; the final sampled but
    unevaluated token is intentionally not in the sequence.
    """
    if (len(support)!=24 or any(len(s)<8 or len(s)>128 or len(set(s))!=len(s)
        or any(type(e) is not int or not 0<=e<128 for e in s) for s in support)):
        raise ValueError("invalid original expert support")
    if (type(prefill) is not int or type(decode) is not int or prefill<1 or decode<0
        or len(events)!=(prefill+decode)*48):
        raise ValueError("incomplete ordered trace")
    clean=[]
    for t in range(prefill+decode):
        visits=events[t*48:(t+1)*48]
        if tuple(e["slot"] for e in visits)!=SLOT_ORDER:
            raise ValueError("unexpected native execution order: do not reconstruct from phase totals")
        for e in visits:
            l,p=coords(e["slot"])
            phase="prefill" if t<prefill else "decode"
            if (e["token"],e["layer"],e["pass"],e["phase"])!=(t,l,p,phase):
                raise ValueError("event index/phase mismatch")
            available=tuple(support[l]); width=len(available)
            ids=e["experts"]
            if len(ids)!=8 or len(set(ids))!=8 or any(type(x) is not int for x in ids):
                raise ValueError("invalid expert selection")
            if original_ids:
                if not set(ids)<=set(available):raise ValueError("unsupported original ID")
                local=[available.index(x) for x in ids]
            else:
                local=ids
                if any(not 0<=x<width for x in local):raise ValueError("out-of-support native ID")
            scores,rank,selected,scaled=(e[k] for k in ("scores","rank_scores","selected_scores","scaled_weights"))
            if (len(scores)!=width or len(rank)!=width or len(selected)!=8 or len(scaled)!=8
                or any(not math.isfinite(v) for vec in (scores,rank,selected,scaled) for v in vec)
                or any(not 0<=v<=1 for v in scores) or any(v<0 for v in scaled)):
                raise ValueError("invalid native router values")
            if any(abs(scores[i]-v)>2e-6 for i,v in zip(local,selected)):
                raise ValueError("selected weights are not gathered from unbiased scores")
            outside=set(range(width))-set(local)
            if outside and min(rank[i] for i in local)+2e-6<max(rank[i] for i in outside):
                raise ValueError("native selection violates biased top-k")
            total=sum(selected)
            if total<=0 or any(abs(v-8*w/(total+1e-20))>2e-5 for v,w in zip(scaled,selected)):
                raise ValueError("native scaled mixture mismatch")
            clean.append({"token":t,"phase":phase,"slot":e["slot"],"layer":l,"pass":p,
                "experts":[available[i] for i in local],"scores":list(scores),"rank_scores":list(rank),
                "selected_scores":list(selected),"scaled_weights":list(scaled)})
    return clean


def next_pass(layer,pass_id):
    return (pass_id+1)%3 if 6<=layer<18 else 0


def replay(events, capacity, policy, layer_bytes=LAYER_BYTES, forecasts=None):
    """Demand-only eviction. At each visit load every missing native expert.

    Oracle is farthest-next-use among nonrequired residents; independently
    optimal for each fixed-capacity, uniform-byte layer under these demand
    rules. All requested IDs must coexist; no bypass or speculation.
    Forecasts, if supplied, are per-event scores for retention at that event.
    They may be produced only from observations through that event.
    """
    caps=(capacity,)*24 if isinstance(capacity,int) else tuple(capacity)
    if len(caps)!=24 or len(layer_bytes)!=24 or any(type(c) is not int or not 1<=c<=128 for c in caps) or any(type(b) is not int or b<=0 for b in layer_bytes):
        raise ValueError("invalid capacity or byte accounting")
    if policy not in ("lru","ema","previous_pass","oracle","forecast"):
        raise ValueError("unknown policy")
    if policy=="forecast" and (forecasts is None or len(forecasts)!=len(events)):
        raise ValueError("forecast count mismatch")
    if not events:raise ValueError("empty trace")
    caches=[set() for _ in caps]
    last={}; ema=[defaultdict(float) for _ in caps]; history={}
    future=defaultdict(deque)
    if policy=="oracle":
        for i,e in enumerate(events):
            for x in e["experts"]:future[e["layer"],x].append(i)
    phases={p:{"requests":0,"hits":0,"admissions":0,"estimated_bytes":0,"visits":0} for p in ("prefill","decode")}
    layers=[{"requests":0,"hits":0,"admissions":0,"estimated_bytes":0} for _ in caps]
    for i,e in enumerate(events):
        l,p=e["layer"],e["pass"]
        req=set(e["experts"])
        if len(req)!=len(e["experts"]) or len(req)>caps[l] or any(not 0<=x<128 for x in req):
            raise ValueError("invalid request or active set exceeds capacity")
        old=caches[l]; missing=req-old
        if policy=="oracle":
            for x in req:
                if future[l,x].popleft()!=i:raise AssertionError("bad next-use index")
        for x in list(ema[l]):ema[l][x]*=.85
        for x in req:ema[l][x]+=.15
        history[l,p]=req.copy()
        if policy=="forecast":
            f=forecasts[i]
            if len(f)!=128 or any(not math.isfinite(v) for v in f):raise ValueError("bad forecast")
        # Victims are old nonrequired entries. With capacity>=|req|, enough
        # victims always exist; never evict a native active expert.
        need=max(0,len(old)+len(missing)-caps[l])
        victims=old-req
        def priority(x):
            # Lower priority evicted first. Deterministic ID tie-break.
            recency=last.get((l,x),-1)
            if policy=="oracle":
                q=future[l,x]; return (-(q[0] if q else math.inf),x)
            if policy=="ema":return (ema[l][x],recency,x)
            if policy=="previous_pass":return (x in history.get((l,next_pass(l,p)),set()),recency,x)
            if policy=="forecast":return (float(f[x]),recency,x)
            return (recency,x)
        evict=set(sorted(victims,key=priority)[:need])
        if len(evict)!=need:raise AssertionError("cannot keep required expert set")
        new=(old-evict)|req
        if not req<=new or len(new)>caps[l]:raise AssertionError("invalid cache transition")
        caches[l]=new
        for x in req:last[l,x]=i
        for out in (phases[e["phase"]],layers[l]):
            out["requests"]+=len(req);out["hits"]+=len(req)-len(missing)
            out["admissions"]+=len(missing);out["estimated_bytes"]+=len(missing)*layer_bytes[l]
        phases[e["phase"]]["visits"]+=1
    result={"policy":policy,"capacity":list(caps),"phases":phases,"layers":layers,
            "allocated_slot_payload_bytes":sum(c*b for c,b in zip(caps,layer_bytes))}
    result["admissions"]=sum(v["admissions"] for v in phases.values())
    result["estimated_bytes"]=sum(v["estimated_bytes"] for v in phases.values())
    result["hit_rate"]=sum(v["hits"] for v in phases.values())/sum(v["requests"] for v in phases.values())
    return result


def rows(path):
    with path.open(encoding="utf8") as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            if row.get("ordered_trace_version")!=1 or "routing_events" not in row:
                raise ValueError("requires new ordered events, not archived aggregated traces")
            if row.get("split") not in ("train","dev"):raise ValueError("invalid split")
            runtime=row["traced_runtime"]
            support=row["expert_support"]
            row["routing_events"]=normalize_events(row["routing_events"],support,
                            runtime["input_tokens"],runtime["decoded_tokens"],original_ids=True)
            yield row


def audit(path,capacities=(8,12,16,24,32),forecast_factory=None):
    reports=[]; cases=set(); identity=None; splits={}
    for row in rows(path):
        if row["case_id"] in cases:raise ValueError("duplicate case")
        cases.add(row["case_id"])
        if splits.setdefault(row["root"],row["split"])!=row["split"]:raise ValueError("root split leakage")
        support=row["expert_support"]
        key=(row["model_sha256"],row["runtime_revision"],json.dumps(support))
        if identity is None:identity=key
        if key!=identity:raise ValueError("mixed teacher identities")
        if row["split"]!="dev":continue
        events=row["routing_events"]
        forecasts=forecast_factory(row) if forecast_factory else None
        for k in capacities:
            # Clip only for a declared scoped teacher; preserve actual capacity
            # in the report. Avoid redundant capacities above support width.
            if any(k>len(s) for s in support):continue
            for policy in ("lru","ema","previous_pass","oracle")+(("forecast",) if forecasts is not None else ()):
                result=replay(events,k,policy,forecasts=forecasts)
                reports.append({"case_id":row["case_id"],"root":row["root"],"task":row["task"],
                                "model_sha256":key[0],"scope":"unrestricted_support" if all(len(s)==128 for s in support) else "scoped_teacher",
                                **result})
    if not reports:raise ValueError("no usable development cases/capacities")
    groups=defaultdict(lambda:defaultdict(list))
    for q in reports:groups[(q["capacity"][0],q["policy"])][q["root"]].append(q)
    summary=[]
    for (k,p),by_root in sorted(groups.items()):
        def macro(fn):return sum(sum(fn(x) for x in v)/len(v) for v in by_root.values())/len(by_root)
        summary.append({"capacity":k,"policy":p,"n_roots":len(by_root),
                        "root_mean_hit_rate":macro(lambda q:q["hit_rate"]),
                        "root_mean_admissions":macro(lambda q:q["admissions"]),
                        "root_mean_estimated_bytes":macro(lambda q:q["estimated_bytes"])})
    return {"status":"observational_exact_route_cache_replay","summary":summary,"cases":reports,
            "limitations":["estimated demand payload only; no measured H2D or stall time",
                           "cold start per case; all prefill and evaluated decode tokens counted",
                           "oracle has future information; not an inference policy"]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routing",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--capacities",type=int,nargs="+",default=[8,12,16,24,32])
    a=p.parse_args()
    if any(not 8<=k<=128 for k in a.capacities):p.error("capacity must be 8..128")
    result=audit(a.routing,tuple(a.capacities))
    h=hashlib.sha256()
    with a.routing.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    result["routing_sha256"]=h.hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open("x",encoding="utf8") as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result["summary"],indent=2))


if __name__=="__main__":main()
