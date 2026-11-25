"""Pure helper functions from Fable's OLR08 archive. Legacy execution drivers removed."""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SEED = 20270119
MENU = ("rfl", "assumption", "simp", "simp_all", "simp [*]", "decide", "omega", "trivial",
        "contradiction", "exact?", "constructor", "ac_rfl")
AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
FORBIDDEN = re.compile(r"\b(sorry|admit|native_decide|run_tac|run_cmd|run_meta|by_elab|set_option|unsafe|trustCompiler)\b|#|IO\.")
ROSTER = {
 "train": {"Nat": ("Bitwise", "Compare", "Div", "Dvd", "Gcd", "Linear", "Log2", "Mod"),
           "List": ("Count", "Erase", "Find", "MapIdx", "MinMax", "Nat", "OfFn", "Pairwise", "Perm", "Range", "Sublist", "TakeDrop", "Zip"),
           "Option": ("Attach", "List", "Monadic"), "Char": ("Lemmas",), "String": ("Lemmas",)},
 "calibration": {"Int": ("Bitwise", "Compare", "DivMod", "Gcd", "Linear", "LemmasAux", "Order", "Pow")},
 "evaluation": {"Array": ("Attach", "Count", "Erase", "Extract", "Find", "InsertIdx", "MapIdx", "Mem", "OfFn", "Perm", "TakeDrop", "Zip"),
                "Vector": ("Erase", "Extract", "Find", "Lemmas"), "Fin": ("Bitwise", "Fold", "Log2"), "UInt": ("Bitwise", "Lemmas")},
}
LEAF = "first | rfl | (simp_all; done) | omega | (simp [*]; done)"


def digest(x): return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + ".tmp"); q.write_text(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"); os.replace(q, p)
def read_lines(p): return [json.loads(s) for s in Path(p).read_text(encoding="utf-8").splitlines() if s.strip()]


def closed(r):
    c = r.get("cert") or {}
    return (r.get("ok") is True and r.get("goals_after") == 0 and c.get("kernel") is True
            and isinstance(c.get("axioms"), list) and set(c["axioms"]) <= AXIOMS)

def category(r):
    if closed(r): return "closed"
    if r.get("ok") is True:
        n = r.get("goals_after")
        return "uncertified" if n == 0 else "open:" + str(min(3, n)) if isinstance(n, int) and n > 0 else "malformed"
    msg = str(r.get("error", "")).lower()
    if "heartbeat" in msg or "recursion depth" in msg or "timeout" in msg: return "resource"
    if msg.startswith("parse:"): return "parse"
    if any(s in msg for s in ("type mismatch", "unif", "application type", "failed to synthesize", "does not have type")): return "type"
    if any(s in msg for s in ("no goals", "did not find", "no matching", "made no progress", "nothing to rewrite", "failed to close")): return "no_match"
    return "rejected"

def human_text(r):
    first, *rest = r["tactic"].split("\n")
    return "\n".join([first] + [s[min(r["col"], len(s) - len(s.lstrip(" "))):] for s in rest])


def roster():
    return [(role, fam, f"src/Init/Data/{fam}/{f}.lean") for role, fams in ROSTER.items() for fam, fs in fams.items() for f in fs]

def key(fam, rel): return fam + "_" + Path(rel).stem


def rows_of(raw):
    """First non-atom trace row at each source position, in visit order."""
    seen = set(); out = []
    for r in raw:
        text = r.get("tactic", ""); kind = str(r.get("kind", ""))
        if kind.startswith("«") and kind.endswith("»") and kind[1:-1] == text.strip(): continue
        pos = (r["line"], r["col"])
        if pos in seen: continue
        seen.add(pos); out.append(r)
    return out


