"""Small exact-answer synthetic smoke suite; template families are clustered."""
import argparse
import json
from pathlib import Path

def make():
    rows=[]
    # Distinct generator families; two examples each. No independent test.
    families=[
        ("train","math","sum",[(f"What is {a}+{b}?",str(a+b)) for a,b in [(37,86),(258,471)]]),
        ("train","math","product",[(f"What is {a} times {b}?",str(a*b)) for a,b in [(14,17),(23,19)]]),
        ("train","code","index",[(f"In Python, what is [4, 7, 11, 18][{i}]?",str([4,7,11,18][i])) for i in [1,-1]]),
        ("train","logic","order",[(f"A precedes B, B precedes C. Which comes {q}?",ans) for q,ans in [("first","A"),("last","C")]]),
        ("dev","math","remainder",[(f"What is the remainder when {a} is divided by {b}?",str(a%b)) for a,b in [(127,9),(251,17)]]),
        ("dev","math","equation",[(f"Solve {a}x + {b} = {a*x+b} for x.",str(x)) for a,b,x in [(7,3,12),(13,5,8)]]),
        ("dev","code","slice",[(f"In Python, what is len(list(range({n}))[{a}:{b}])?",str(len(list(range(n))[a:b]))) for n,a,b in [(12,3,8),(9,4,14)]]),
        ("dev","logic","set",[(f"How many elements are in the intersection of {a} and {b}?",str(len(set(a)&set(b)))) for a,b in [([1,2,4],[2,3,4]),([2,5,7],[1,3,8])]])
    ]
    for split,category,family,items in families:
        for i,(prompt,answer) in enumerate(items):
            rows.append({"root":f"{family}-{i}","family":family,"split":split,"category":category,
                         "prompt":prompt+" Respond with FINAL: followed by the answer only.","answer":answer})
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    with a.out.open("x",encoding="utf8") as f:
        for r in make():f.write(json.dumps(r)+"\n")

if __name__=="__main__":main()
