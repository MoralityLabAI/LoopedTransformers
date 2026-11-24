"""Run the local 07B campaign under process-tree memory/CPU limits."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
import zipfile

ROOT=Path(__file__).resolve().parent

def parse_ram(s):
    from lcgrt.ramcap import parse_size
    return parse_size(s)

def guard(ram):
    if ram=="0":return None,"explicit no hard cap"
    limit=parse_ram(ram)
    if sys.platform=="win32":
        from lcgrt.ramcap import Job
        job=Job(limit,"below_normal");job.assign_self()
        return job,"Windows process-tree committed-memory limit"
    if sys.platform.startswith("linux"):
        # systemd-run --scope cgroup v2 MemoryMax must already enforce cap.
        rel=next((s.split("::",1)[1] for s in Path("/proc/self/cgroup").read_text().splitlines() if s.startswith("0::")),None)
        if rel is not None:
            d=Path("/sys/fs/cgroup")/rel.lstrip("/")
            for q in (d,*d.parents):
                p=q/"memory.max"
                if p.exists():
                    v=p.read_text().strip()
                    if v!="max" and int(v)<=limit:return None,"verified cgroup v2 cap "+v
        raise SystemExit("Use systemd-run --user --scope -p MemoryMax="+ram+" with this command; or --ram 0 explicitly.")
    raise SystemExit("Hard cap supported on Windows/Linux cgroup v2; --ram 0 opts out.")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lean-source",type=Path,required=True)
    p.add_argument("--prior-07a",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--profile",choices=("smoke","full"),default="full")
    p.add_argument("--threads",type=int,default=8);p.add_argument("--workers",type=int,default=2)
    p.add_argument("--ram",default="12G");p.add_argument("--timeout",type=int,default=900)
    p.add_argument("--collect-only",action="store_true")
    args=p.parse_args()
    if not 1<=args.workers<=args.threads<=64 or args.timeout<180:raise SystemExit("invalid resource configuration")
    args.out=args.out.resolve();args.lean_source=args.lean_source.resolve();args.prior_07a=args.prior_07a.resolve()
    os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads),LEAN_NUM_THREADS=str(max(1,args.threads//args.workers)))
    job,limit_kind=guard(args.ram)
    import psutil
    proc=psutil.Process()
    if hasattr(proc,"cpu_affinity"):
        avail=proc.cpu_affinity()
        proc.cpu_affinity(avail[:min(args.threads,len(avail))])
    # Resolve the installed pinned Lean via elan when no explicit override.
    if "LCGRT_LEAN_HOME" not in os.environ and shutil.which("elan"):
        exe=subprocess.check_output(["elan","which","lean"],cwd=ROOT/"lean",text=True).strip()
        os.environ["LCGRT_LEAN_HOME"]=str(Path(exe).resolve().parent.parent)
    from lcgrt import config
    if not config.REPL_EXE.is_file() or not config.LEAN_EXE.is_file():
        raise SystemExit("Build the pinned Lean backend first: cd lean; lake build")
    args.out.mkdir(parents=True,exist_ok=True)
    if not (args.out/"contract.json").exists() and any(args.out.iterdir()):
        raise SystemExit("New output directory must be empty.")
    resource={"os":platform.platform(),"python":sys.version,"logical_cpu":psutil.cpu_count(),
              "physical_ram":psutil.virtual_memory().total,"threads":args.threads,"workers":args.workers,
              "requested_ram":args.ram,"memory_limit":limit_kind,
              "cpu_only":True,"peak_sampled_tree_rss":0,"peak_sampled_processes":0}
    done=threading.Event()
    def monitor():
        while not done.is_set():
            try:
                ps=[proc,*proc.children(recursive=True)];total=0
                for p in ps:
                    try:total+=p.memory_info().rss
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                resource["peak_sampled_tree_rss"]=max(total,resource["peak_sampled_tree_rss"])
                resource["peak_sampled_processes"]=max(len(ps),resource["peak_sampled_processes"])
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            done.wait(.5)
    t=threading.Thread(target=monitor,daemon=True);t.start()
    start=time.perf_counter()
    from ppt_gym import olr07b_cpu as B
    try:
        report=B.run(args.lean_source,args.prior_07a,args.out,args.profile,args.workers,args.threads,args.timeout,args.collect_only)
        resource["completed"]=True
    except BaseException as e:
        resource["completed"]=False;resource["error"]=repr(e);raise
    finally:
        done.set();t.join()
        resource["seconds"]=time.perf_counter()-start
        if job is not None:resource["job_peak_commit_bytes"]=job.peak_bytes()
        B.write(args.out/"resource_latest.json",resource)
    code=args.out/"review_code";code.mkdir(exist_ok=True)
    for rel in ("run_olr07b.py","ppt_gym/olr07b_cpu.py","ppt_gym/olr07a_library.py",
                "ppt_gym/tests/test_olr07b.py","lcgrt/ramcap.py","lean/Trace.lean","lean/Main.lean",
                "OLR_07A_RESULTS.md","OLR_07B_REGISTRATION.md","OLR_07B_LOCAL_RUN.md"):
        p=code/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,p)
    names=sorted(p.relative_to(args.out).as_posix() for p in args.out.rglob("*")
                 if p.is_file() and p.name!="audit_manifest.json")
    B.write(args.out/"audit_manifest.json",{"schema":"olr07b.audit.v1","files":B.stamp(args.out,names)})
    full=args.out.with_name(args.out.name+"_AUDIT.zip")
    review=args.out.with_name(args.out.name+"_FABLE.zip")
    with zipfile.ZipFile(full,"w",zipfile.ZIP_DEFLATED) as z:
        for n in [*names,"audit_manifest.json"]:z.write(args.out/n,n)
    with zipfile.ZipFile(review,"w",zipfile.ZIP_DEFLATED) as z:
        for n in [*filter(lambda x:x.startswith("review_code/"),names),"summary.json","contract.json",
                  "frozen_policy.json","resource_latest.json","audit_manifest.json"]:
            z.write(args.out/n,n)
    print(json.dumps({"summary":report,"audit_zip":str(full),"audit_sha256":B.sha(full),
                      "fable_zip":str(review),"fable_sha256":B.sha(review)},indent=2),flush=True)

if __name__=="__main__":main()
