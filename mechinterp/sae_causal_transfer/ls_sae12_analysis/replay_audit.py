"""Replay the registered LS-SAE12 auditor on a platform other than the one that ran it.

Standard library only. Lives outside active_reasoning_v1/ so committing it does
not trigger the registered workflow.

The registered auditor checks that every saved isotropic vector equals
random.Random(seed).gauss regenerated on the auditing machine, exactly, in
float64. Python's gauss goes through the platform math library, so a Windows
replay of a Linux receipt differs in the last bits (observed: at most 9 ulp,
max |diff| 1.4e-16) and the exact check fails although the model received the
same edit. This replay first checks every saved vector against the local
regeneration:
  * every component within MAX_ULP float64 ulp, and
  * dose * sign * vector rounded to float32 (the edit the model receives) is
    bit-identical for every registered dose and both signs;
then runs the registered auditor unchanged, with its gauss_unit answering each
verified seed with the saved vector. Any unverified seed raises.
"""
from __future__ import annotations
import argparse
import json
import struct
import sys
from pathlib import Path

from mechinterp.sae_causal_transfer.active_reasoning_v1 import audit as A

MAX_ULP = 16


def bits(x):
    return struct.unpack("<q", struct.pack("<d", abs(x)))[0]


def ulp_distance(a, b):
    if (a < 0) == (b < 0):
        return abs(bits(a) - bits(b))
    return bits(a) + bits(b)


def f32(x):
    return struct.pack("<f", x)


def check_vectors(report):
    verified, worst_ulp, differing, components, f32_checked = {}, 0, 0, 0, 0
    for st in report["states"]:
        for key in ("selected", "control"):
            d = st[key]["directions"]
            for seed, saved in zip(d["isotropic_seeds"], d["isotropic_vectors"]):
                local = A.gauss_unit(seed)
                if len(saved) != len(local):
                    raise ValueError(f"seed {seed}: vector length differs")
                for a, b in zip(local, saved):
                    u = ulp_distance(a, b)
                    worst_ulp = max(worst_ulp, u)
                    differing += u > 0
                    components += 1
                    if u > MAX_ULP:
                        raise ValueError(f"seed {seed}: component differs by {u} ulp")
                    for dose in A.DOSES:
                        for sign in (1.0, -1.0):
                            if f32(float(dose) * sign * a) != f32(float(dose) * sign * b):
                                raise ValueError(f"seed {seed}: float32 edit differs at dose {dose}")
                            f32_checked += 1
                if seed in verified and verified[seed] != saved:
                    raise ValueError(f"seed {seed}: two different saved vectors")
                verified[seed] = saved
    return verified, {"seeds": len(verified), "components": components,
                      "components_differing_from_local_regeneration": differing,
                      "max_ulp": worst_ulp, "max_ulp_allowed": MAX_ULP,
                      "float32_edit_components_checked_bit_identical": f32_checked}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--screen", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    verified, summary = check_vectors(json.loads(a.input.read_text()))

    def gauss_unit(seed):
        if seed not in verified:
            raise ValueError(f"isotropic seed {seed} was not verified")
        return verified[seed]

    A.gauss_unit = gauss_unit
    check_path = a.out.with_name(a.out.stem + ".isotropic-check.json")
    check_path.parent.mkdir(parents=True, exist_ok=True)
    check_path.write_bytes((json.dumps(summary, sort_keys=True, indent=2) + "\n").encode())
    sys.argv = [A.__file__, "--source", str(a.source), "--screen", str(a.screen), "--input", str(a.input),
                "--out", str(a.out)]
    A.main()


if __name__ == "__main__":
    main()
