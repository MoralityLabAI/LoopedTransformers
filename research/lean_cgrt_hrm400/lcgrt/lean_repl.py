"""Client for the `lgym` Lean proof-state server (lean/Main.lean).

Every goal, hypothesis, tactic outcome and error message returned here was
produced by Lean 4. The client only transports JSON and caches local contexts.
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time

from . import config
from .errors import LeanError


class LeanRepl:
    def __init__(self, imports: tuple[str, ...] = (), timeout: float = 60.0):
        self.imports = imports
        self.timeout = timeout
        self.proc = None
        self.contexts: dict[str, list] = {}
        self.calls = 0
        self.seconds = 0.0
        self.version = None
        self._start()

    def _start(self):
        if not config.REPL_EXE.exists():
            raise LeanError("lean_unavailable", f"{config.REPL_EXE} missing; run `lake build` in lean/")
        self.proc = subprocess.Popen([str(config.REPL_EXE), *self.imports], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     env=config.lean_env(), text=True, encoding="utf-8", bufsize=1)
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        ready = self._get()
        if not ready.get("ready"):
            raise LeanError("lean_unavailable", f"lgym did not start: {ready}")
        self.version = ready.get("lean")
        self.contexts.clear()

    def _reader(self):
        for line in self.proc.stdout:
            self._q.put(line)
        self._q.put(None)

    def _get(self):
        try:
            line = self._q.get(timeout=self.timeout)
        except queue.Empty:
            self.restart()
            raise LeanError("lean_timeout", f"no reply within {self.timeout}s")
        if line is None:
            raise LeanError("lean_crashed", f"lgym exited with {self.proc.poll()}")
        return json.loads(line)

    def restart(self):
        self.close()
        self._start()

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()

    def request(self, **req) -> dict:
        t = time.perf_counter()
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        out = self._get()
        self.calls += 1
        self.seconds += time.perf_counter() - t
        if "error" in out and "ok" not in out:
            raise LeanError("lean_request_error", out["error"])
        self._absorb(out)
        return out

    def _absorb(self, obj):
        """Remember each local context the first time Lean sends it."""
        if isinstance(obj, dict):
            if "ctx" in obj and "hyps" in obj:
                self.contexts[obj["ctx"]] = obj["hyps"]
            for v in obj.values():
                self._absorb(v)
        elif isinstance(obj, list):
            for v in obj:
                self._absorb(v)

    # Thin op wrappers -------------------------------------------------------
    def init(self, stmt: str, pp: bool = False) -> dict:
        return self.request(op="init", stmt=stmt, pp=pp)

    def expand(self, sid: int, errors: int = 2) -> dict:
        return self.request(op="expand", sid=sid, errors=errors)

    def run(self, sid: int, tactic: str) -> dict:
        return self.request(op="run", sid=sid, tactic=tactic)

    def focus(self, sid: int, goal: int) -> dict:
        return self.request(op="focus", sid=sid, goal=goal)

    def kernel(self, sid: int) -> dict:
        return self.request(op="kernel", sid=sid)

    def reset(self):
        self.contexts.clear()
        return self.request(op="reset")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
