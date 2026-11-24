"""Independently replay the rendered suffixes and verify matched source structure."""
from collections import Counter, defaultdict
import re

from .source import NODES,POOLS,REMAINING,SPLITS


EDGE=re.compile(r"([A-P])->([A-P])")


def query(text):
    match=re.fullmatch(r"Edges: (.+)\nStart: ([A-P])\nHops: ([123])\nAnswer:(?: ([A-P]))?",text)
    if match is None:
        raise ValueError("Pointer query grammar differs from the frozen source")
    graph,start,hops,answer=match.groups()
    if graph=="[not provided]":
        return None,(),start,int(hops),answer
    parts=graph.split(", ")
    parsed=[EDGE.fullmatch(v) for v in parts]
    if len(parsed)!=16 or any(v is None for v in parsed):
        raise ValueError("The rendered graph must have 16 bound edges")
    order=tuple(v.group(1) for v in parsed)
    edges={v.group(1):v.group(2) for v in parsed}
    if set(edges)!=set(NODES) or set(edges.values())!=set(NODES):
        raise ValueError("The rendered graph is not a permutation")
    return edges,order,start,int(hops),answer


def walk(edges,start,hops):
    result=[start]
    for _ in range(hops):
        result.append(edges[result[-1]])
    return result


def check(public,gold,prior):
    pairs=defaultdict(dict)
    by_id={v["id"]:v for v in gold}
    if len(public)!=len(by_id) or len(public)!=sum(SPLITS.values())*14:
        raise ValueError("Unexpected source cardinality")
    seen=set()
    endpoints=Counter()
    for p in public:
        id=p["id"];g=by_id[id]
        keys=("root","split","world","remaining","arm")
        if any(p[k]!=g[k] for k in keys):
            raise ValueError("Source oracle is bound to the wrong slot")
        group=pairs[p["root"]]
        slot=p["world"],p["arm"],p["remaining"]
        if slot in group or g["answer"] not in POOLS[p["split"]]:
            raise ValueError("Duplicate source or invalid answer-role partition")
        group[slot]=p,g["answer"]
        if p["arm"]=="focused":
            if p["prompt"] in seen or p["prompt"] in prior:
                raise ValueError("A focused prompt repeats an exposed source")
            seen.add(p["prompt"])
            if p["remaining"]==3:
                endpoints[p["split"],p["world"],g["answer"]]+=1

    graphs=set()
    for group in pairs.values():
        tapes={}
        for world in ("A","B"):
            main,answer=group[world,"focused",3]
            context=main["prompt"].split("\n\n")
            if len(context)!=5:
                raise ValueError("Expected three demonstration blocks")
            demo_answers=set()
            for expected,demo in zip(REMAINING,context[1:-1],strict=True):
                m,_,start,hops,observed=query(demo)
                assert m is not None and hops==expected and walk(m,start,hops)[-1]==observed in POOLS[main["split"]]
                demo_answers.add(observed)
            assert len(demo_answers)==3 and all(group[w,"focused",3][1] not in demo_answers for w in ("A","B"))
            graph,order,start,_,blank=query(context[-1])
            assert graph is not None and blank is None
            full=walk(graph,start,3)
            assert len(set(full))==4 and full[-1]==answer
            copy,copy_gold=group[world,"copy",0]
            assert copy["prompt"].splitlines()[-1]==copy_gold==answer
            for n in REMAINING:
                p,a=group[world,"focused",n]
                e,ea=group[world,"erased",n]
                split=p["prompt"].split("\n\n")
                other=e["prompt"].split("\n\n")
                assert split[:-1]==other[:-1]==context[:-1]
                m,o,s,h,w=query(split[-1]); em,eo,es,eh,ew=query(other[-1])
                assert (m,o,s,h,w)==(graph,order,full[3-n],n,None)
                assert (em,eo,es,eh,ew)==(None,(),full[3-n],n,None)
                assert walk(m,s,n)[-1]==a==ea==answer
            tapes[world]=(graph,order,full,context[:-1])
        a,ao,ap,ac=tapes["A"]
        b,bo,bp,bc=tapes["B"]
        changed=sorted(k for k in NODES if a[k]!=b[k])
        assert ao==bo and ac==bc and ap[:-1]==bp[:-1] and ap[-1]!=bp[-1]
        assert len(changed)==2 and ap[-2] in changed
        x,y=changed
        assert a[x]==b[y] and a[y]==b[x]
        signature=tuple(a[k] for k in NODES)
        assert signature not in graphs
        graphs.add(signature)
        for n in REMAINING:
            assert group["A","erased",n][0]["prompt"]==group["B","erased",n][0]["prompt"]
    assert endpoints==Counter({(split,world,token):count//len(POOLS[split])
                                for split,count in SPLITS.items() for world in ("A","B") for token in POOLS[split]})
    return {"roots":len(pairs),"distinct_focused_prompts":len(seen),"verified_world_suffixes":len(seen),
            "prior_prompt_count":len(prior),"verified_three_edge_swaps":len(graphs)}
