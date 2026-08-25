"""Frozen-policy reuse for the eight-root confirmation replication."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DSL = HERE.parents[0] / "arc_program_discovery_v1"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(DSL))

from research_gym.benchmarks.arc_open_alphabet_spiral_v3.planner import *  # noqa: F401,F403,E402
