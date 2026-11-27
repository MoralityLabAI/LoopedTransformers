"""Read-only native Diplomacy oracle: JSONL request/reply, one frozen board per id."""
from contextlib import redirect_stdout
import json
import sys
from export_diplomacy import digest, signature, holds, resolve, utility, pool, roster

def describe(g, campaign):
    assert g.get_current_phase().endswith("M")
    frozen=signature(g)
    possible=g.get_all_possible_orders()
    players=sorted(g.powers)
    assert len(players)==7
    targets={p:{v[0] for v in pool(g,p,possible)[1]} for p in players}
    shared={t for p in players for t in targets[p] if sum(t in targets[q] for q in players)>=2}
    rosters={p:roster(g,p,possible,shared,4) for p in players}
    centers={x[:3] for x in g.map.scs}
    board_id="dip-adaptive:"+digest(frozen)[:16]
    spec={"schema":"kite.sparse-joint.v1","id":board_id,"players":players,"rosters":rosters,
          "denominator":2*len(centers),"profiles":[{"id":"start","weights":{p:["1"]+["0"]*(len(rosters[p])-1) for p in players}}],
          "source":{"board_hash":digest(frozen),"campaign_id":campaign,"phase":frozen["phase"],"engine":"native Game.process"}}
    return spec, frozen, centers

def main():
    from diplomacy import Game
    with redirect_stdout(sys.stderr):
        g=Game(map_name="standard")
        h=resolve(g,{p:holds(g,p) for p in g.powers})
        campaign="standard-1901-hold-chain"
        boards={}
        for b in (g,h):
            spec,frozen,centers=describe(b,campaign)
            boards[spec["id"]]=(b,spec,frozen,centers)
        assert len(boards)==2
    seen={}
    for line in sys.stdin:
        request=json.loads(line)
        with redirect_stdout(sys.stderr):
            if request["op"]=="describe":
                answer={"specs":[v[1] for v in boards.values()]}
            elif request["op"]=="evaluate":
                board_id=request["board"]
                mode=request["mode"]
                assert mode in ("br","hybrid","hybrid_prune")
                b,spec,frozen,centers=boards[board_id]
                players=spec["players"]
                used=seen.setdefault((board_id,mode),set())
                tuples=request["tuples"]
                assert isinstance(tuples,list) and len(tuples)>0 and len(used)+len(tuples)<=256
                rows=[]
                for t in tuples:
                    assert len(t)==len(players) and tuple(t) not in used
                    assert all(type(i) is int and 0<=i<len(spec["rosters"][p]) for p,i in zip(players,t))
                    used.add(tuple(t))
                    orders={p:spec["rosters"][p][i]["orders"] for p,i in zip(players,t)}
                    after=resolve(b,orders)
                    rows.append({"actions":t,"payoffs":[utility(after,p,centers)["numerator"] for p in players],
                                 "post_state_hash":digest(signature(after))})
                assert signature(b)==frozen
                answer={"rows":rows}
            else:
                raise ValueError("unknown request")
        print(json.dumps(answer,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
