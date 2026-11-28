"""Local LCD campaign driver: smoke checks by default; explicit native run.

The driver runs synchronously in the invoking process and stops on failure or
budget exhaustion. It never downloads model weights or overwrites receipts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

MODULES=("lcd01_huginn","lcd02_spectral","lcd03_anderson","lcd04_adaptive",
         "lcd05_precision","lcd06_sudoku","lcd06_lean","lcd07_compose")

def command(module,*args):
    return [sys.executable,"-m","research_gym.scripts."+module,*map(str,args)]

def load_status(p):
    return json.loads(p.read_text(encoding="utf-8"))["status"]

def build_plan(a):
    root=a.out.resolve()
    jobs=[]
    def pair(module,folder,fit_stage,fit_file,fit_status,extra=()):
        f=root/folder/fit_file
        t=root/folder/"test.json"
        jobs.append((module,fit_stage,f,{fit_status},("--out",f,*extra)))
        jobs.append((module,"test",t,{"PILOT_COMPLETE"},("--freeze",f,"--out",t,*extra)))
    if a.native_huginn:
        pair("lcd01_huginn","lcd01","calibrate","cal.json","CALIBRATED")
        pair("lcd02_spectral","lcd02","fit","fit.json","FROZEN")
        pair("lcd03_anderson","lcd03","calibrate","cal.json","FROZEN")
        pair("lcd04_adaptive","lcd04","calibrate","cal.json","FROZEN")
        pair("lcd05_precision","lcd05","fit","fit.json","FROZEN")
        p=root/"lcd07"/"test.json"
        jobs.append(("lcd07_compose","test",p,{"PILOT_COMPLETE","SKIPPED_INSUFFICIENT_COMPONENTS"},
                     ("--results",root,"--out",p)))
    if a.sudoku:
        pair("lcd06_sudoku","lcd06a","calibrate","cal.json","FROZEN",("--device",a.sudoku_device))
    if a.lean_ckpt:
        pair("lcd06_lean","lcd06b","calibrate","cal.json","FROZEN",
             ("--ckpt",a.lean_ckpt.resolve(),"--device",a.lean_device,"--data",a.lean_data))
    return jobs

def run(a):
    jobs=build_plan(a)
    if a.plan:
        for m in MODULES:
            print(" ".join(command(m,"--stage","smoke")))
        for m,s,_,_,extra in jobs:
            print(" ".join(command(m,"--stage",s,*extra)))
        return 0
    deadline=time.monotonic()+a.max_hours*3600
    def invoke(args):
        remaining=deadline-time.monotonic()
        if remaining<=0:
            raise RuntimeError("campaign time cap exhausted")
        subprocess.run(args,check=True,timeout=remaining)
    for m in MODULES:
        invoke(command(m,"--stage","smoke"))
    if not jobs:
        print("All source smoke checks passed. No native jobs requested.")
        return 0
    for m,s,p,accepted,extra in jobs:
        if p.exists():
            status=load_status(p)
            if a.resume and status in accepted:
                print("Preserving existing completed stage",m,s,flush=True)
                continue
            raise RuntimeError("Existing receipt; use fresh output root or --resume with completed receipts: "+str(p))
        print("LCD campaign",m,s,flush=True)
        invoke(command(m,"--stage",s,*extra))
        if not p.exists() or load_status(p) not in accepted:
            raise RuntimeError("Missing or incomplete stage receipt: "+str(p))
    print("Requested native stages finished or explicitly skipped by evidence gate.")
    return 0

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-huginn",action="store_true")
    p.add_argument("--sudoku",action="store_true")
    p.add_argument("--sudoku-device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--lean-ckpt",type=Path)
    p.add_argument("--lean-data",default="lcgrt-v1")
    p.add_argument("--lean-device",choices=("cuda","cpu"),default="cpu")
    p.add_argument("--out",type=Path,default=Path("results/lcd-campaign"))
    p.add_argument("--max-hours",type=float,default=4.)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--plan",action="store_true")
    a=p.parse_args()
    if not 0<a.max_hours<=72:
        p.error("max-hours must be in (0,72]")
    raise SystemExit(run(a))
