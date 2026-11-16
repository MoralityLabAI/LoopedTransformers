"""RAM-16 original Lean4.35 zero-action goal probe.

This source is a portable, resource-guarded version of the original RAM13
mailbox-based compiler probe. It observes genuine goals before any tactic,
using the unchanged source-backed RAM03 ProbeServer. The source reference proof
is replaced, and no original target gold tactic text enters model data.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from native_replay import render_source


class ProbeTimeout(RuntimeError):
    pass


class Session:
    def __init__(self,lean,original,theorem,out,guard,timeout=15.):
        self.out=Path(out)
        self.out.mkdir(parents=True,exist_ok=False)
        self.guard=guard
        self.closed=False
        self.number=0
        self.source=self.out/"Session.lean"
        self.source.write_text("import ProbeServer\n"+
             render_source(original,theorem,["ram03_probe_server"]),encoding="utf8")
        self.log=(self.out/"stdout.log").open("wb")
        env=dict(os.environ,RAM03_MAILBOX=str(self.out.resolve()))
        self.proc=subprocess.Popen([str(lean),str(self.source.resolve())],
              stdin=subprocess.DEVNULL,stdout=self.log,stderr=subprocess.STDOUT,
              env=env,start_new_session=(os.name!="nt"),
              creationflags=(0x08000000 if os.name=="nt" else 0))
        try:
            response=self.await_response(0,timeout)
            if response.get("ready") is not True:
                raise ProbeTimeout("original compiler did not signal ready state")
            self.initial=response["state"]
            if self.initial.get("accepted") is not True:
                raise ProbeTimeout("original compiler root failed to elaborate")
        except BaseException:
            self.close()
            raise

    def await_response(self,n,seconds):
        end=time.perf_counter()+seconds
        target=self.out/f"response-{n}.json"
        while time.perf_counter()<end:
            self.guard.check()
            if target.exists():
                try:return json.loads(target.read_text(encoding="utf8"))
                except (FileNotFoundError,PermissionError,json.JSONDecodeError):pass
            if self.proc.poll() is not None:
                self.log.flush()
                details=(self.out/"stdout.log").read_text(encoding="utf8",errors="replace")[-1600:]
                raise ProbeTimeout("original Lean server exited unexpectedly: "+details)
            time.sleep(.005)
        raise ProbeTimeout(f"genuine original Lean goal response {n} timed out")

    def close(self):
        if self.closed:return
        self.closed=True
        if self.proc.poll() is None:
            n=self.number+1
            tmp=self.out/f"request-{n}.tmp"
            tmp.write_text(json.dumps(dict(id=n,actions=[],stop=True)))
            tmp.replace(self.out/f"request-{n}.json")
            try:self.proc.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                if os.name!="nt":
                    try:os.killpg(self.proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                else:self.proc.kill()
                self.proc.wait(timeout=2)
        self.log.close()

    def __enter__(self):return self
    def __exit__(self,*args):
        self.close()
        if args[0] is None and self.proc.returncode!=0:
            raise ProbeTimeout("original source probe stopped with nonzero Lean exit")
