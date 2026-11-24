"""Draw paired suffixes with balanced answer labels and constant graph context."""
from hashlib import sha256
import random

from experiments.nan_comp14.sources import NODES, question


SPLITS={"calibration":16,"development":16}
POOLS={"calibration":NODES[:8],"development":NODES[8:]}
REMAINING=(1,2,3)
ARMS=("focused","erased","copy")
INTRO="Starting from the given node, follow exactly the requested number of edges. Reply only with the final node, one letter A through P."
COPY="Copy the following verified answer exactly, with no explanation:\n"


def seeded(root):
    return random.Random(int.from_bytes(sha256(f"NAN-SUFFIX-80-v1|{root}".encode()).digest()[:16],"big"))


def demonstration(rng,hops,answer):
    edges=dict(zip(NODES,rng.sample(list(NODES),len(NODES)),strict=True))
    inverse={v:k for k,v in edges.items()}
    start=answer
    for _ in range(hops):
        start=inverse[start]
    return question(edges,rng.sample(list(NODES),len(NODES)),start,hops)+" "+answer


def draw(split,index):
    root=f"suffix80:{split}:{index:03d}"
    rng=seeded(root)
    pool=POOLS[split]
    a,b=pool[index%len(pool)],pool[(index+3)%len(pool)]
    s,p,q,owner=rng.sample(sorted(set(NODES)-{a,b}),4)
    edges={s:p,p:q,q:a,owner:b}
    available=sorted(set(NODES)-set(edges.values()))
    rng.shuffle(available)
    edges.update(zip(sorted(set(NODES)-set(edges)),available,strict=True))
    other=edges|{q:b,owner:a}
    examples=rng.sample(sorted(set(pool)-{a,b}),3)
    demos=[demonstration(rng,n,label) for n,label in zip(REMAINING,examples,strict=True)]
    order=rng.sample(list(NODES),len(NODES))
    return root,dict(zip((3,2,1),(s,p,q),strict=True)),order,(edges,other),(a,b),demos


def build():
    public,gold=[],[]
    for split,count in SPLITS.items():
        for index in range(count):
            root,starts,order,worlds,answers,demos=draw(split,index)
            for world,edges,answer in zip(("A","B"),worlds,answers,strict=True):
                for remaining,arm in [(n,which) for n in REMAINING for which in ("focused","erased")]+[(0,"copy")]:
                    meta={"id":f"{root}:{remaining}:{arm}:{world}","root":root,"split":split,
                          "world":world,"remaining":remaining,"arm":arm}
                    if arm=="copy":
                        prompt=COPY+answer
                    else:
                        body=question(None if arm=="erased" else edges,order,starts[remaining],remaining)
                        prompt="\n\n".join((INTRO,*demos,body))
                    public.append(meta|{"prompt":prompt})
                    gold.append(meta|{"answer":answer})
    return public,gold
