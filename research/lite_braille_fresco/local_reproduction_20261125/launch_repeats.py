"""External launcher for the LBF-20M registered experiment (the executable only exposes --out).

Runs each repetition in a fresh Python process, sequentially, with the console log, exit status,
wall time and whole-process peak working set / peak pagefile bytes recorded beside the output.
The experiment is invoked exactly as registered: python -u research/lite_braille_fresco/lbf20m_factor.py --out <out>
"""
import ctypes
import datetime
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(r"C:\Users\patri\lbf20m-run-20261125")
SRC = ROOT / "src"
REPEATS = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 4, 5]


class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def peak(handle):
    pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
    fn = ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    ok = fn(wintypes.HANDLE(int(handle)), ctypes.byref(pmc), pmc.cb)
    return (pmc.PeakWorkingSetSize, pmc.PeakPagefileUsage) if ok else (None, None)


def main():
    env = dict(os.environ)
    env.pop("PYTHONOPTIMIZE", None)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONUTF8"] = "1"
    pre = subprocess.run([sys.executable, "-c",
                          "import sys,os;print(sys.flags.optimize, os.environ.get('PYTHONHASHSEED'), os.environ.get('PYTHONOPTIMIZE'))"],
                         env=env, capture_output=True, text=True, cwd=SRC).stdout.strip()
    assert pre.startswith("0 0 None"), pre
    (ROOT / "preflight.txt").write_text(f"sys.flags.optimize / PYTHONHASHSEED / PYTHONOPTIMIZE = {pre}\n")
    for n in REPEATS:
        d = ROOT / f"repeat-{n:02d}"
        out = d / "lbf20m"
        d.mkdir(exist_ok=False)
        t0 = time.perf_counter()
        start = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(d / "console.log", "wb") as so, open(d / "console.err", "wb") as se:
            p = subprocess.Popen([sys.executable, "-u", "research/lite_braille_fresco/lbf20m_factor.py", "--out", str(out)],
                                 cwd=SRC, env=env, stdout=so, stderr=se)
            rc = p.wait()
        wall = time.perf_counter() - t0
        pws, ppf = peak(p._handle)
        (d / "run.json").write_text(json.dumps({
            "repeat": n, "exit_code": rc, "wall_seconds": wall, "start_utc": start,
            "peak_working_set_bytes": pws, "peak_pagefile_bytes": ppf,
            "python": sys.version, "python_executable": sys.executable,
            "note": "peak figures are for the whole process (audit, planner, oracle and all arms)",
        }, indent=2), encoding="utf-8")
        print("repeat", n, "exit", rc, "wall", round(wall, 1), flush=True)
    (ROOT / "launcher.done").write_text("done\n")


if __name__ == "__main__":
    main()
