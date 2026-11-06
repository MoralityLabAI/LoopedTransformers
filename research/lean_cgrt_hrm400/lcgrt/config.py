"""All externally derived paths and pins live here (and only here)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]           # research/lean_cgrt_hrm400
LEAN_DIR = PKG_ROOT / "lean"

LEAN_VERSION = "4.19.0"
LEAN_COMMIT = "6caaee842e94"     # matches experiments/lean_reasoner/results/kernel_receipt.json
LEAN_TOOLCHAIN = f"leanprover/lean4:v{LEAN_VERSION}"

_default_home = (r"D:\lean_cgrt\toolchains\lean-4.19.0-windows" if sys.platform == "win32"
                 else str(Path.home() / ".elan" / "toolchains" / "leanprover--lean4---v4.19.0"))
LEAN_HOME = Path(os.environ.get("LCGRT_LEAN_HOME", _default_home))
EXE = ".exe" if sys.platform == "win32" else ""
LEAN_BIN = LEAN_HOME / "bin"
LEAN_EXE = LEAN_BIN / f"lean{EXE}"
REPL_EXE = LEAN_DIR / ".lake" / "build" / "bin" / f"lgym{EXE}"
# Same server as a second binary, for Mathlib tracing while lgym.exe is in use (see lakefile).
TRACE_EXE = LEAN_DIR / ".lake" / "build" / "bin" / f"lgym_trace{EXE}"

# Heavy, regenerable artifacts stay outside the repository and outside OneDrive.
_default_root = r"D:\lean_cgrt" if sys.platform == "win32" else str(Path.home() / "lean_cgrt")
WORK_ROOT = Path(os.environ.get("LCGRT_WORK_ROOT", _default_root))
DATA_ROOT = WORK_ROOT / "data"
RUN_ROOT = WORK_ROOT / "runs"
MATHLIB_DIR = Path(os.environ.get("LCGRT_MATHLIB_DIR", str(WORK_ROOT / "mathlib4-v4.19.0")))

RESULTS = PKG_ROOT / "results"


def lean_env() -> dict:
    env = dict(os.environ)
    env["PATH"] = str(LEAN_BIN) + os.pathsep + env.get("PATH", "")
    return env

_vr = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "vramorama" / "vramorama.exe"
VRAMORAMA = os.environ.get("LCGRT_VRAMORAMA", str(_vr) if _vr.exists() else "")
