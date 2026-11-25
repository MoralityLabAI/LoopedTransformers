"""LBF-19 native Lean verifier inside the proof-search loop (restricted equational fragment).

Fragment.  A word w over {a..f} denotes the unary-function composition `f_w1 (f_w2 (... x))` on an
arbitrary type `A`; rule i is a hypothesis `h_i : forall z : A, term(a_i, z) = term(b_i, z)`.  The theorem is
`term(S) = term(T)`.

Correspondence (abstract move  <->  native tactic).  A move `(rule i, direction d, position p)` on word w, with
`left -> right` the oriented sides and `suffix = w[p+|left|:]`, is the tactic
    forward  (side 0):  conv => lhs; rw [h_i (term suffix)]     (rw [<- h_i ...] if d = 1)
    backward (side 1):  conv => rhs; rw [h_i (term suffix)]
applied to the state whose goal is `term(w) = term(T)` (side 0) or `term(S) = term(w)` (side 1).  Because the
explicit suffix fixes the nesting depth of the pattern, the instance is unique, so Lean's resulting goal must be
exactly `term(w') = term(T)` / `term(S) = term(w')`.  Every call checks this *and* that the tactic is accepted;
any divergence raises `CorrespondenceError` (the run is then invalid, never silently continued).

Cost accounting (all measured, never modelled):
  * `init`      one elaboration of the theorem statement per episode (binders + both terms)
  * `tactic`    each `run` request, split by outcome and by word length
  * `cache`     a (side, word) already holding a verified state is reused without a call
  * `kernel`    kernel check of the replayed final proof + standalone `lean` compile of the certificate
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics
import subprocess
import sys
import time

import os

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "lean_cgrt_hrm400"))
from lcgrt.lean_repl import LeanRepl  # noqa: E402
from lcgrt import config as lcgrt_config  # noqa: E402

LGYM_EXE = pathlib.Path(os.environ.get("LBF19_LGYM_EXE", str(lcgrt_config.REPL_EXE)))
LEAN_EXE = pathlib.Path(os.environ.get("LBF19_LEAN_EXE", str(lcgrt_config.LEAN_EXE)))
ALPHABET = "abcdef"


class CorrespondenceError(RuntimeError):
    pass


def term(word: str, leaf: str = "x") -> str:
    s = leaf
    for c in reversed(word):
        s = f"f_{c} {s}" if s == leaf else f"f_{c} ({s})"
    return s


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def statement(case) -> str:
    binders = ["(A : Type)", "(" + " ".join(f"f_{c}" for c in ALPHABET) + " : A -> A)", "(x : A)"]
    for i, (a, b) in enumerate(case.rules):
        binders.append(f"(h{i} : forall z : A, {term(a, 'z')} = {term(b, 'z')})")
    return f"forall {' '.join(binders)}, {term(case.source)} = {term(case.target)}"


def tactic_for(side: int, word: str, rule: int, direction: int, position: int, rules) -> str:
    a, b = rules[rule]
    left = a if direction == 0 else b
    suffix = word[position + len(left):]
    arrow = "<- " if direction else ""
    arg = term(suffix)
    arg = f"({arg})" if " " in arg else arg
    return f"conv => {'lhs' if side == 0 else 'rhs'}; rw [{arrow}h{rule} {arg}]"


class NativeSession:
    """One proof-search episode against Lean (`mode='lean'`) or the exact call counter (`mode='count'`).

    The count mode executes no tactic but applies identical deduplication, so verifier-call counts are
    exactly those the Lean mode will make (checked by a test on every archived root).
    """

    def __init__(self, case, repl: LeanRepl | None, mode: str = "lean", *, call_cap: int | None = None,
                 check_goals: bool = True):
        assert mode in ("lean", "count")
        self.case = case
        self.repl = repl
        self.mode = mode
        self.call_cap = call_cap
        self.check_goals = check_goals
        self.states: list[dict[str, int]] = [{}, {}]   # side -> word -> sid
        self.edges_requested = 0       # successor enumerations presented to the verifier layer
        self.calls = 0                 # actual tactic requests (novel states only)
        self.cache_hits = 0            # edges whose child already had a verified state
        self.tactic_seconds = 0.0
        self.init_seconds = 0.0
        self.fail_calls = 0
        self.lat_ms: list[float] = []
        self.lat_by_len: dict[int, list[float]] = {}
        self.mismatch = 0
        self.script: list[tuple[int, str]] = []   # (side, tactic) tape of verified novel steps
        self.exceeded = False
        if mode == "lean":
            t0 = time.perf_counter()
            o = repl.init(statement(case))
            self.init_seconds = time.perf_counter() - t0
            if o.get("closed"):
                raise CorrespondenceError("theorem closed at init")
            root = o["sid"]
            self.states[0][case.source] = root
            self.states[1][case.target] = root
            self._root_sid = root
            self._expect_root = _norm(o["goals"][0]["target"]["raw"])
        else:
            self.states[0][case.source] = -1
            self.states[1][case.target] = -1

    # -- core call -------------------------------------------------------------------------------
    def discover(self, side: int, parent: str, mv, child: str) -> None:
        """Present successor `child` of `parent` (already known novel on `side`) to the verifier layer."""
        self.edges_requested += 1
        if child in self.states[side]:
            self.cache_hits += 1
            return
        if self.call_cap is not None and self.calls >= self.call_cap:
            self.exceeded = True
            self.states[side][child] = -2
            return
        self.calls += 1
        if self.mode == "count":
            self.states[side][child] = -1
            return
        rule, direction, position = _mv_fields(mv)
        tac = tactic_for(side, parent, rule, direction, position, self.case.rules)
        psid = self.states[side][parent]
        t0 = time.perf_counter()
        o = self.repl.run(psid, tac)
        dt = time.perf_counter() - t0
        self.tactic_seconds += dt
        self.lat_ms.append(dt * 1000)
        self.lat_by_len.setdefault(len(parent), []).append(dt * 1000)
        if not o.get("ok"):
            self.fail_calls += 1
            raise CorrespondenceError(f"Lean rejected abstract move {tac}: {o.get('error')}")
        st = o["state"]
        if self.check_goals:
            if st["closed"]:
                # try-rfl closed the goal: only legitimate when child equals the opposite endpoint.
                ok = (child == self.case.target) if side == 0 else (child == self.case.source)
            else:
                got = _norm(st["goals"][0]["target"]["raw"])
                exp = (f"{term(child)} = {term(self.case.target)}" if side == 0
                       else f"{term(self.case.source)} = {term(child)}")
                ok = got == _norm(exp)
            if not ok:
                self.mismatch += 1
                raise CorrespondenceError(f"goal mismatch after {tac}")
        self.states[side][child] = st["sid"]
        self.script.append((side, tac))

    def summary(self) -> dict:
        lat = self.lat_ms
        return {
            "mode": self.mode,
            "edges_requested": self.edges_requested,
            "verifier_calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": (self.cache_hits / self.edges_requested) if self.edges_requested else 0.0,
            "tactic_seconds": self.tactic_seconds,
            "init_seconds": self.init_seconds,
            "mean_call_ms": (statistics.fmean(lat) if lat else None),
            "p50_call_ms": (statistics.median(lat) if lat else None),
            "p95_call_ms": (sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else None),
            "mismatches": self.mismatch,
            "exceeded_call_cap": self.exceeded,
        }

    # -- native certificate ------------------------------------------------------------------------
    def certify(self, path: list[dict], out_dir: pathlib.Path | None = None, name: str = "cert") -> dict:
        """Replay the full path as one native tactic proof from a fresh state, kernel-check it, and compile it."""
        case = self.case
        t0 = time.perf_counter()
        o = self.repl.init(statement(case))
        sid = o["sid"]
        tape = []
        word = case.source
        for m in path:
            tac = tactic_for(0, word, m["rule"], m["direction"], m["position"], case.rules)
            r = self.repl.run(sid, tac)
            if not r.get("ok"):
                raise CorrespondenceError(f"certificate replay rejected {tac}: {r.get('error')}")
            sid = r["state"]["sid"]
            tape.append(tac)
            word = m["successor"]
            if r["state"]["closed"]:
                break
        if not r["state"]["closed"]:
            r2 = self.repl.run(sid, "rfl")
            if not r2.get("ok") or not r2["state"]["closed"]:
                raise CorrespondenceError("final goal not closed by rfl")
            sid = r2["state"]["sid"]
            tape.append("rfl")
        replay_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        k = self.repl.kernel(sid)
        kernel_s = time.perf_counter() - t1
        if not k.get("ok") or k.get("axioms"):
            raise CorrespondenceError(f"kernel check failed or axioms present: {k.get('error')} {k.get('axioms')}")
        res = {"native_tactic_replay_seconds": replay_s, "kernel_seconds": kernel_s,
               "kernel_ok": True, "kernel_axioms": k.get("axioms", []), "tape_len": len(tape), "tape": tape}
        if out_dir is not None:
            res.update(compile_standalone(case, tape, out_dir, name))
        return res


def _mv_fields(mv):
    if isinstance(mv, tuple):
        return mv[0], mv[1], mv[2]
    if isinstance(mv, dict):
        return mv["rule"], mv["direction"], mv["position"]
    return mv.rule, mv.direction, mv.position


def standalone_source(case, tape: list[str], name: str) -> str:
    """Standalone Lean 4 file: the same tactic tape wrapped as a closed theorem (kernel-checked by `lean`)."""
    binders = ["(A : Type)", "(" + " ".join(f"f_{c}" for c in ALPHABET) + " : A -> A)", "(x : A)"]
    for i, (a, b) in enumerate(case.rules):
        binders.append(f"(h{i} : forall z : A, {term(a, 'z')} = {term(b, 'z')})")
    lines = [f"theorem {name} {' '.join(binders)} : {term(case.source)} = {term(case.target)} := by"]
    lines += [f"  {t}" for t in tape]
    return "namespace LBF19Cert\n\n" + "\n".join(lines) + "\n\nend LBF19Cert\n#print axioms LBF19Cert." + name + "\n"


def compile_standalone(case, tape, out_dir: pathlib.Path, name: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"{name}.lean"
    f.write_text(standalone_source(case, tape, name), encoding="utf-8")
    t0 = time.perf_counter()
    p = subprocess.run([str(LEAN_EXE), "--threads=1", str(f)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    dt = time.perf_counter() - t0
    log = p.stdout + p.stderr
    ok = p.returncode == 0 and f"'LBF19Cert.{name}' does not depend on any axioms" in log
    return {"standalone_compile_seconds": dt, "standalone_ok": bool(ok), "standalone_file": f.name,
            "standalone_sha256": __import__("hashlib").sha256(f.read_bytes()).hexdigest(),
            "standalone_log_tail": log[-400:] if not ok else ""}


def start_repl() -> LeanRepl:
    return LeanRepl(exe=LGYM_EXE, timeout=300.0)


def compile_batch(case, tapes: dict[str, list[str]], out_dir: pathlib.Path, file_stem: str) -> dict:
    """Compile many verified tactic tapes for one root as one standalone Lean file (axiom audit per theorem)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    binders = ["(A : Type)", "(" + " ".join(f"f_{c}" for c in ALPHABET) + " : A -> A)", "(x : A)"]
    for i, (a, b) in enumerate(case.rules):
        binders.append(f"(h{i} : forall z : A, {term(a, 'z')} = {term(b, 'z')})")
    body = ["namespace LBF19Cert", ""]
    for name, tape in tapes.items():
        body.append(f"theorem {name} {' '.join(binders)} : {term(case.source)} = {term(case.target)} := by")
        body += [f"  {t}" for t in tape]
        body.append("")
    body.append("end LBF19Cert")
    body += [f"#print axioms LBF19Cert.{n}" for n in tapes]
    f = out_dir / f"{file_stem}.lean"
    f.write_text("\n".join(body) + "\n", encoding="utf-8")
    t0 = time.perf_counter()
    p = subprocess.run([str(LEAN_EXE), "--threads=1", str(f)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1800)
    dt = time.perf_counter() - t0
    log = p.stdout + p.stderr
    (out_dir / f"{file_stem}.log").write_text(log, encoding="utf-8")
    passed = set(re.findall(r"'LBF19Cert\.([A-Za-z0-9_]+)' does not depend on any axioms", log))
    ok = p.returncode == 0 and passed == set(tapes)
    return {"file": f.name, "n_theorems": len(tapes), "compile_seconds": dt, "all_axiom_free": bool(ok),
            "sha256": __import__("hashlib").sha256(f.read_bytes()).hexdigest()}
