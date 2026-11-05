"""Run the v1 `gym.py audit` (as committed at e134b54) and the fixed one under Lean 4.19.0.

Inputs: the v1-generated corpus, and an adversarial file whose only theorem is proved
from an axiom declared with a line break after `axiom` and whose `#eval` prints a fake axiom report.

    python audit/v1_lean_probe.py --out results/v1_audit/lean_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcgrt import config  # noqa: E402

ADVERSARIAL = """set_option autoImplicit false
namespace LocalGym
axiom
  cheat : False
theorem thm_bad : False := cheat
end LocalGym
#eval IO.println "'LocalGym.thm_bad' does not depend on any axioms"
"""


def run_audit(gym_py: Path, source: Path, receipt: Path):
    env = config.lean_env() | {"PYTHONIOENCODING": "utf-8"}
    p = subprocess.run([sys.executable, str(gym_py), "audit", "--source", str(source), "--receipt", str(receipt)],
                       capture_output=True, text=True, env=env, cwd=str(gym_py.parent))
    return {"returncode": p.returncode, "receipt": json.loads(receipt.read_text())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v1_audit/lean_probe.json")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        old = d / "gym_v1.py"
        old.write_text(subprocess.run(["git", "show", "e134b54:research/lean_cgrt_hrm400/gym.py"], cwd=ROOT,
                                      capture_output=True, text=True, encoding="utf-8", check=True).stdout,
                       encoding="utf-8")
        corpus = d / "corpus"
        subprocess.run([sys.executable, str(ROOT / "gym.py"), "generate", "--count", "64", "--out", str(corpus)],
                       check=True, capture_output=True, env=os.environ | {"PYTHONIOENCODING": "utf-8"})
        adv = d / "Adversarial.lean"; adv.write_text(ADVERSARIAL, encoding="utf-8")
        out = {"lean": subprocess.run([str(config.LEAN_EXE), "--version"], capture_output=True, text=True,
                                      env=config.lean_env()).stdout.strip()}
        for tag, g in (("v1_e134b54", old), ("fixed", ROOT / "gym.py")):
            for name, src in (("v1_corpus_64", corpus / "Generated.lean"), ("adversarial", adv)):
                r = run_audit(g, src, d / f"{tag}_{name}.json")
                out[f"{tag}/{name}"] = {"returncode": r["returncode"], "status": r["receipt"]["status"],
                                        "lean_kernel_checked": r["receipt"]["lean_kernel_checked"],
                                        "reason": r["receipt"].get("reason")}
    p = ROOT / a.out; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
