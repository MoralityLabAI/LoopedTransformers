"""Adjudicate only the frozen sparse mixed-strategy frontier."""
import argparse
import json
from pathlib import Path
from export_diplomacy import digest,signature,holds,resolve,utility

def main():
    from diplomacy import Game
    p=argparse.ArgumentParser()
    p.add_argument("plan",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    m=json.loads(a.plan.read_text())
    assert m["schema"]=="kite.native-joint-plan.v1" and m["budget"]==256
    base=Game(map_name="standard")
    boards={base.get_current_phase():base}
    nxt=resolve(base,{p:holds(base,p) for p in base.powers})
    boards[nxt.get_current_phase()]=nxt
    result=[]
    for j in m["jobs"]:
        s=j["spec"]
        g=boards[s["source"]["phase"]]
        frozen=signature(g)
        assert digest(frozen)==s["source"]["board_hash"]
        players=s["players"]
        assert sorted(players)==sorted(g.powers)
        centers={x[:3] for x in g.map.scs}
        assert s["denominator"]==2*len(centers)
        cells=[]
        assert len(j["plan"]["cells"])==j["plan"]["required_cells"]<=m["budget"]
        seen=set()
        for t in j["plan"]["cells"]:
            assert len(t)==len(players) and tuple(t) not in seen
            seen.add(tuple(t))
            assert all(type(i) is int and 0<=i<len(s["rosters"][p]) for p,i in zip(players,t))
            orders={p:s["rosters"][p][i]["orders"] for p,i in zip(players,t)}
            h=resolve(g,orders)
            cells.append({"actions":t,"payoffs":[utility(h,p,centers)["numerator"] for p in players],
                          "post_state_hash":digest(signature(h))})
        assert signature(g)==frozen
        result.append({"id":s["id"],"digest":j["digest"],"board_hash":digest(frozen),"cells":cells})
    a.out.write_text(json.dumps({"schema":"kite.native-joint-receipts.v1","jobs":result},indent=2)+"\n")
    print("JOINT_DIPLOMACY_CELLS",sum(len(j["cells"]) for j in result))

if __name__=="__main__":
    main()
