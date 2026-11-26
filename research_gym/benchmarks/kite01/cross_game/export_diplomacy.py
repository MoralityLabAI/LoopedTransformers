"""Offline AI_Diplomacy engine adapter. No LLM clients, live orders or private messages."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def signature(g):
    return {"phase":g.get_current_phase(),"units":{p:sorted(v) for p,v in g.get_units().items()},
            "centers":{p:sorted(v) for p,v in g.get_centers().items()}}

def holds(g,p):
    return sorted(u+" H" for u in g.get_units(p) if not u.startswith("*"))

def submit(g,p,orders):
    g.set_orders(p,list(orders),expand=False)
    if sorted(g.get_orders(p))!=sorted(orders):
        raise ValueError(f"Order mismatch {p}: {orders!r} -> {g.get_orders(p)!r}")

def pool(g,p,possible):
    base=holds(g,p)
    options=[]
    for h in base:
        u=h[:-2]
        loc=u[2:]
        for o in sorted(set(possible.get(loc,[]))|set(possible.get(loc[:3],[]))):
            if o.startswith(u+" ") and not o.endswith(" VIA") and (o.startswith(u+" - ") or o.startswith(u+" S ")):
                options.append((o.split()[-1][:3],o,sorted(o if x==h else x for x in base)))
    return base,options

def roster(g,p,possible,shared,cap):
    base,opts=pool(g,p,possible)
    enemy_centers={s[:3] for s in g.map.scs}-{s[:3] for s in g.get_centers(p)}
    opts.sort(key=lambda v:(v[0] not in shared,v[0] not in enemy_centers,v[1]))
    result=[{"id":"hold","orders":base}]
    seen={tuple(base)}
    for _,_,bundle in opts:
        if tuple(bundle) not in seen:
            result.append({"id":digest(bundle)[:16],"orders":bundle})
            seen.add(tuple(bundle))
        if len(result)>=cap:
            break
    return result

def resolve(base,orders):
    g=deepcopy(base)
    g.clear_orders()
    for p in sorted(orders):
        submit(g,p,orders[p])
    g.process()
    if g.get_current_phase().endswith("R"):
        for p in sorted(g.powers):
            retreat=sorted(u[1:]+" D" for u in g.get_units(p) if u.startswith("*"))
            if retreat:
                submit(g,p,retreat)
        g.process()
    return g

def utility(g,p,centers):
    owned=len({c[:3] for c in g.get_centers(p)}&centers)
    occupied=len({u[2:5] for u in g.get_units(p) if not u.startswith("*")}&centers)
    return {"owned_centers":owned,"occupied_centers":occupied,"numerator":owned+occupied}

def export_pair(g,row,col,cap):
    if row==col or row not in g.powers or col not in g.powers or not g.get_current_phase().endswith("M"):
        raise ValueError("Invalid focal pair or phase")
    frozen=signature(g)
    possible=g.get_all_possible_orders()
    shared={v[0] for v in pool(g,row,possible)[1]}&{v[0] for v in pool(g,col,possible)[1]}
    rows,cols=roster(g,row,possible,shared,cap),roster(g,col,possible,shared,cap)
    if min(len(rows),len(cols))<2:
        raise ValueError("Insufficient candidates")
    others=sorted(set(g.powers)-{row,col})
    centers={s[:3] for s in g.map.scs}
    worlds=[]
    for mode in ("all_hold","fixed_pressure"):
        fixed={}
        for p in others:
            a=roster(g,p,possible,shared,cap)
            fixed[p]=a[int(mode=="fixed_pressure" and len(a)>1)]["orders"]
        R,C,outcomes=[],[],[]
        for x in rows:
            rr,cc=[],[]
            for y in cols:
                h=resolve(g,{**fixed,row:x["orders"],col:y["orders"]})
                a,b=utility(h,row,centers),utility(h,col,centers)
                rr.append(a["numerator"]);cc.append(b["numerator"])
                outcomes.append({"row":x["id"],"col":y["id"],"row_utility":a,"col_utility":b,
                                 "post_phase":h.get_current_phase(),"post_state_hash":digest(signature(h))})
            R.append(rr);C.append(cc)
        worlds.append({"id":mode,"fixed_orders":fixed,"R":R,"C":C,"outcomes":outcomes})
    if signature(g)!=frozen:
        raise AssertionError("Source state mutated")
    return {"schema":"kite.conditional-bimatrix.v1","id":f"diplomacy:{digest(frozen)[:16]}:{row}:{col}",
            "actors":[row,col],"actions":{"row":rows,"col":cols},"denominator":2*len(centers),
            "worlds":worlds,"beliefs":{"row":[1,1],"col":[1,1]},
            "source":{"engine":"native Game.process","board_hash":digest(frozen),"board":frozen,
                      "utility":"owned+occupied supply centers, global scale","phase":frozen["phase"],
                      "retreat_policy":"disband","scope":"movement plus immediate retreat; fixed other powers"}}

def pure_profiles(g,cap):
    """Exact restricted pure-profile deviations for every power, with no pairwise shortcut."""
    frozen=signature(g)
    possible=g.get_all_possible_orders()
    powers=sorted(g.powers)
    target_sets={p:{v[0] for v in pool(g,p,possible)[1]} for p in powers}
    shared={t for p in powers for t in target_sets[p] if sum(t in target_sets[q] for q in powers)>=2}
    rosters={p:roster(g,p,possible,shared,cap) for p in powers}
    centers={s[:3] for s in g.map.scs}
    results=[]
    for mode in ("all_hold","fixed_pressure"):
        anchor={p:rosters[p][int(mode=="fixed_pressure" and len(rosters[p])>1)] for p in powers}
        orders={p:anchor[p]["orders"] for p in powers}
        baseline=resolve(g,orders)
        incumbent={p:utility(baseline,p,centers)["numerator"] for p in powers}
        deviations={}
        for p in powers:
            options=[]
            for a in rosters[p]:
                if a["id"]==anchor[p]["id"]:
                    value=incumbent[p]
                    post=digest(signature(baseline))
                else:
                    h=resolve(g,{**orders,p:a["orders"]})
                    value=utility(h,p,centers)["numerator"]
                    post=digest(signature(h))
                options.append({"id":a["id"],"numerator":value,"post_state_hash":post})
            deviations[p]=options
        results.append({"id":f"dip-pure:{digest(frozen)[:16]}:{mode}","players":powers,
                        "board_hash":digest(frozen),"denominator":2*len(centers),"mode":mode,
                        "rosters":rosters,"anchor":{p:anchor[p]["id"] for p in powers},
                        "incumbent":incumbent,"deviations":deviations,
                        "native_adjudications":1+sum(len(rosters[p])-1 for p in powers)})
    assert signature(g)==frozen
    return results


def main():
    from diplomacy import Game
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--cap",default=4,type=int)
    a=p.parse_args()
    if not 2<=a.cap<=8:
        raise ValueError("cap must be 2..8")
    game=Game(map_name="standard")
    roots=[]
    pure=[]
    for phase in range(2):
        for row,col in [("FRANCE","GERMANY"),("ENGLAND","FRANCE"),("AUSTRIA","RUSSIA")]:
            roots.append(export_pair(game,row,col,a.cap))
        pure.extend(pure_profiles(game,a.cap))
        if phase==0:
            game=resolve(game,{p:holds(game,p) for p in game.powers})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps({"roots":roots,"pure_profiles":pure},indent=2)+"\n")
    print(f"NATIVE_DIPLOMACY_ROOTS {len(roots)}")

if __name__=="__main__":
    main()
