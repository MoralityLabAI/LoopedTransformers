"""Sparse-autoencoder study of recurrent transformer hidden states (LaCo / loop recovery / controller).

Paths
-----
ROOT  project folder (code, reports, figures, small result JSON, saved SAE configs)
DATA  heavy artifacts (checkpoints, activation shards).  Kept on a SHORT path because the session folder's
      absolute path exceeds Windows MAX_PATH (260) once nested, which breaks Python file opens.
LIBS  matplotlib / scipy installed with `pip --target` (not installed into the shared Adict venv).
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_here))                       # recsae/
DATA = os.environ.get("RECSAE_DATA", r"C:\Users\patri\.cache\recsae_data")
LIBS = os.environ.get("RECSAE_LIBS", r"C:\Users\patri\.cache\recsae_pylibs")
for p in (LIBS, os.path.join(ROOT, "vendor")):                       # vendor = frozen looprec / crrbench snapshots
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
os.makedirs(DATA, exist_ok=True)
