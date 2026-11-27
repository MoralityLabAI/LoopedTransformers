"""Independent set-graph replay of the two deterministic graph cut selectors."""
import json, random, hashlib
from pathlib import Path
H=Path(__file__).resolve().parent

def vertices(mask):
    return {j for j in range(mask.bit_length()) if mask & (1<<j)}

def adjacency(rows, remaining):
    g={j:set() for j in remaining}
    for mask in rows:
        vs=vertices(mask)&remaining
        for j in vs:g[j].update(vs-{j})
    return g

def simulate(rows, remaining):
    g=adjacency(rows,remaining);sizes=[]
    while g:
        i=min(g,key=lambda j:(len(g[j]),j));nbr=g.pop(i);sizes.append(len(nbr)+1)
        for j in nbr:g[j].update(nbr-{j});g[j].discard(i)
    return max(sizes,default=0),sum(max(0,s-3) for s in sizes)

def structural(rows, remaining, queries):
    masks=[sum(1<<i for i in vertices(a)&remaining) for a in rows]
    r=[simulate(masks,remaining)]
    for q in queries:
        q=sum(1<<i for i in vertices(q)&remaining)
        if not q:r.append(r[0]);continue
        pivot=min(vertices(q));r.append(simulate([a^q if a&(1<<pivot) else a for a in masks],remaining-{pivot}))
    return max(x for x,y in r),sum(y for x,y in r)

def choose(w,arm):
    rest=set(range(w['n']));rows=w['rows'];qs=w['queries'];out=set()
    if arm.startswith('none'):return out
    if arm=='prefix2':return {0,1}
    if arm=='random2':return set(random.Random(w['seed']+17017).sample(range(w['n']),2))
    for _ in range(2):
        g=adjacency(rows,rest)
        if arm=='degree2':j=min(rest,key=lambda j:(-len(g[j]),j))
        else:j=min(rest,key=lambda j:(*structural(rows,rest-{j},qs),-len(g[j]),j))
        rest.remove(j);out.add(j)
    return out

def main():
    raw=(H/'receipt.json').read_bytes();r=json.loads(raw);worlds={w['seed']:w for w in r['worlds']};cache={};checks=0;scope_checks=0
    for row in r['results']:
        w=worlds[row['seed']];key=(row['seed'],row['arm'])
        if key not in cache:cache[key]=choose(w,row['arm'])
        cut=cache[key];assert cut==vertices(row['cut']);checks+=1
        assert list(structural(w['rows'],set(range(w['n']))-cut,w['queries']))==row['scope'];scope_checks+=1
    result=dict(status='PASS',unique_world_arm_choices=len(cache),selector_replays=checks,scope_score_replays=scope_checks,receipt_sha256=hashlib.sha256(raw).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (H/'selector_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
