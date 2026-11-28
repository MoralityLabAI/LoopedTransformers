"""Sequential local server benchmark. No external APIs or executable answers."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

FINAL=re.compile(r"^\s*FINAL\s*:\s*([^\r\n]+)\s*$",re.I|re.M)
def score(text,answer):
    matches=FINAL.findall(text)
    return len(matches)==1 and matches[0].strip().casefold()==str(answer).strip().casefold()

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def load_roster(path):
    rows=[json.loads(l) for l in path.read_text(encoding="utf8").splitlines() if l.strip()]
    if not rows or len({r["root"] for r in rows})!=len(rows):raise ValueError("empty or duplicate roots")
    families={}
    for r in rows:
        if r["split"] not in ("train","dev") or families.setdefault(r["family"],r["split"])!=r["split"]:
            raise ValueError("family leakage or invalid split")
    return rows

def local_url(url):
    p=urlsplit(url)
    if p.scheme!="http" or p.hostname not in ("127.0.0.1","localhost","::1") or p.username or p.password or p.query or p.fragment:
        raise ValueError("local HTTP server only")
    return url.rstrip("/")+"/v1/chat/completions"

def request(url,payload,timeout):
    req=Request(url,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    with urlopen(req,timeout=timeout) as f:
        raw=f.read(8*1024*1024+1)
    if len(raw)>8*1024*1024:raise ValueError("oversized response")
    return json.loads(raw)

def run(root,endpoint,max_tokens,seed,timeout,call=request):
    payload={"model":"dory","messages":[{"role":"user","content":root["prompt"]}],
             "temperature":0,"top_p":1,"seed":seed,"max_tokens":max_tokens,"stream":False}
    start=time.monotonic()
    try:
        response=call(endpoint,payload,timeout)
        choice=response["choices"][0]
        msg=choice["message"]
        text=msg.get("content") or ""
        usage=response.get("usage",{})
        return {"status":"ok","correct":score(text,root["answer"]),"text":text,
                "reasoning":msg.get("reasoning_content"),"finish_reason":choice.get("finish_reason"),
                "prompt_tokens":usage.get("prompt_tokens"),"completion_tokens":usage.get("completion_tokens"),
                "server_timings":response.get("timings"),"wall_seconds":time.monotonic()-start}
    except Exception as e:
        return {"status":"error","correct":None,"error":str(e),"wall_seconds":time.monotonic()-start}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--roster",type=Path,required=True)
    p.add_argument("--loops",type=int,choices=(1,2,3,4),required=True)
    p.add_argument("--cache",choices=("per_loop","last_loop"),required=True)
    p.add_argument("--server-log",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--url",default="http://127.0.0.1:8080")
    p.add_argument("--max-tokens",type=int,default=128)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--timeout",type=float,default=180)
    a=p.parse_args()
    if a.max_tokens<1 or a.timeout<=0 or a.server_log.stat().st_size==0:raise ValueError("invalid budget or empty log")
    rows=load_roster(a.roster)
    endpoint=local_url(a.url)
    config={"loops":a.loops,"cache":a.cache,"roster_sha256":digest(a.roster),
            "startup_log_sha256":digest(a.server_log),"max_tokens":a.max_tokens,"seed":a.seed,
            "temperature":0,"top_p":1}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open("x",encoding="utf8") as f:
        for root in rows:
            result=run(root,endpoint,a.max_tokens,a.seed,a.timeout)
            f.write(json.dumps({"config":config,"root":root,**result},allow_nan=False)+"\n");f.flush()
            print(root["root"],result["status"],result["correct"],round(result["wall_seconds"],2))
            if result["status"]=="error":
                raise RuntimeError("stop on server error; preserve partial run")

if __name__=="__main__":main()
