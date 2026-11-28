"""Exact finite check of parity-query identifiability, known coordinate mapping."""
import itertools, json

def dot(x,y):return (x&y).bit_count()%2

def span(rows):
    s={0}
    for a in rows:s|={x^a for x in s}
    return s

def check(k=3):
    n=1<<k; cases=0
    for rows in itertools.product(range(n),repeat=k):
        groups={}
        for h in range(n):groups.setdefault(tuple(dot(r,h) for r in rows),[]).append(h)
        sp=span(rows)
        for q in range(n):
            for values in groups.values():
                identifiable=len({dot(q,h) for h in values})==1
                assert identifiable==(q in sp)
                cases+=1
    return dict(k=k,matrices=n**k,checks=cases,status='PASS')
if __name__=='__main__':print(json.dumps(check(),indent=2))
