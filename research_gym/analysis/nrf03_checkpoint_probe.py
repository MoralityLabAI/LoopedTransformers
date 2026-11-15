"""NRF-03: auditable measurements on frozen native Loop Schedule Algebra checkpoints.

This code never trains or modifies a checkpoint. It verifies the published
SHA-256, uses weights_only torch loading, builds a real-arithmetic global
Lipschitz bound by exact binary32 row/column sums, and separately measures
local Jacobian growth and dark-to-observed coupling in float64.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

SCALE = 1 << 149
CHECKPOINT_PATTERN = re.compile(
    r"surface_tied_R(16|32)_S(101|103|107)_E(2048|4096)_step_(\d+)\.pt$"
)
PARAMETERS = (
    "blocks.0.in_proj.weight", "blocks.0.in_proj.bias",
    "blocks.0.out_proj.weight", "blocks.0.out_proj.bias",
)


def exact_matrix_norm_bound_ceil_6(weight) -> int:
    """Return ceil(1e6 sqrt(||W||_1 ||W||_inf)) using integer arithmetic.

    Every finite IEEE binary32 entry is exactly an integer times 2^-149.
    The calculation treats the stored parameters as exact real numbers.
    """
    import torch
    if weight.dtype != torch.float32 or weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError("finite binary32 matrix required")
    rows, cols = weight.shape
    columns = [0] * cols
    greatest_row = 0
    for row in weight.tolist():
        total = 0
        for j, value in enumerate(row):
            numerator, denominator = abs(float(value)).as_integer_ratio()
            if SCALE % denominator:
                raise ValueError("invalid binary32 exponent")
            unit = numerator * (SCALE // denominator)
            total += unit
            columns[j] += unit
        greatest_row = max(greatest_row, total)
    square = greatest_row * max(columns)
    root = math.isqrt(square)
    if root * root < square:
        root += 1
    return (root * 1_000_000 + SCALE - 1) // SCALE


def one_step_certified_upper_ceil_6(w1, w2) -> int:
    """Upward micro-unit bound for F=x+(W2*tanh(W1*x+b1)+b2)/4."""
    a = exact_matrix_norm_bound_ceil_6(w1)
    b = exact_matrix_norm_bound_ceil_6(w2)
    return 1_000_000 + (a*b + 4_000_000 - 1) // 4_000_000


def manifest_entries(checkpoint_dir: Path, event_path: Path) -> list[dict]:
    """Permit only the twelve published tied model checkpoints."""
    root = checkpoint_dir.resolve()
    entries = []
    for line in event_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        receipt = json.loads(line)
        if receipt.get("event") != "checkpoint":
            continue
        name = Path(receipt["path"].replace("\\", "/")).name
        match = CHECKPOINT_PATTERN.fullmatch(name)
        if match is None:
            continue
        rounds, seed, exposure, step = map(int, match.groups())
        if step * 8 * rounds != exposure:
            raise ValueError(f"invalid exposure cadence: {name}")
        if receipt["cell_id"] != f"surface_tied_R{rounds}_S{seed}":
            raise ValueError(f"checkpoint cell mismatch: {name}")
        file_path = (root / name).resolve()
        if file_path.parent != root:
            raise ValueError("unsafe checkpoint path")
        entries.append({
            "file": file_path, "name": name, "sha256": receipt["sha256"],
            "rounds": rounds, "seed": seed, "exposure": exposure, "step": step,
        })
    entries.sort(key=lambda row: (row["rounds"], row["seed"], row["exposure"]))
    if len(entries) != 12 or len({x["name"] for x in entries}) != 12:
        raise ValueError("expected twelve distinct tied checkpoint receipts")
    return entries


def load_pinned_weights(entry: dict):
    import torch
    with entry["file"].open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != entry["sha256"]:
        raise ValueError("checkpoint SHA-256 mismatch: " + entry["name"])
    checkpoint = torch.load(entry["file"], map_location="cpu", weights_only=True)
    if (checkpoint.get("cell_id") != f"surface_tied_R{entry['rounds']}_S{entry['seed']}"
        or checkpoint.get("step") != entry["step"]
        or checkpoint.get("exposure") != entry["exposure"]):
        raise ValueError("metadata differs from immutable receipt")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or set(state) != set(PARAMETERS):
        raise ValueError("unexpected state_dict parameter keys")
    w1, b1, w2, b2 = (state[key].detach().cpu().contiguous() for key in PARAMETERS)
    n = w1.shape[0]
    if (w1.shape != (n, n) or w2.shape != (n, n)
        or b1.shape != (n,) or b2.shape != (n,)
        or any(t.dtype != torch.float32 or not torch.isfinite(t).all()
               for t in (w1, b1, w2, b2))):
        raise ValueError("nonfinite or unsupported tensor schema")
    return w1, b1, w2, b2


def probe_state_dict(w1, b1, w2, b2, *, rounds: int,
                     observed_coords: int = 8, power_iterations: int = 12) -> dict:
    import torch
    if rounds <= 0 or power_iterations <= 0 or not 0 < observed_coords < w1.shape[0]:
        raise ValueError("invalid loop/probe dimensions")
    certified_micro = one_step_certified_upper_ceil_6(w1, w2)
    w1, b1, w2, b2 = (v.detach().cpu().to(torch.float64)
                      for v in (w1, b1, w2, b2))
    n = w1.shape[0]
    with torch.no_grad():
        def run(x):
            x = x.clone()
            derivatives = []
            for _ in range(rounds):
                u = w1.mv(x) + b1
                a = u.tanh()
                derivatives.append(1-a*a)
                x = x + 0.25 * (w2.mv(a)+b2)
            return x, derivatives
        def tangent(v, g, transpose=False):
            if transpose:
                return v + .25*w1.T.mv(g*w2.T.mv(v))
            return v + .25*w2.mv(g*w1.mv(v))
        def chain(v, derivatives, transpose=False):
            for g in reversed(derivatives) if transpose else derivatives:
                v = tangent(v, g, transpose=transpose)
            return v
        zero = torch.zeros(n, dtype=torch.float64)
        final, derivatives = run(zero)
        nudged = zero.clone()
        nudged[0] = 0.001
        second, _ = run(nudged)
        finite_diff = float((second-final).norm().item()/0.001)
        coords = torch.arange(n, dtype=torch.float64)
        direction = torch.sin(coords*4.25+1)
        direction /= direction.norm()
        for _ in range(power_iterations):
            estimate = chain(chain(direction, derivatives), derivatives, transpose=True)
            if estimate.norm() == 0:
                break
            direction = estimate / estimate.norm()
        power_gain = float(chain(direction, derivatives).norm().item())
        dark = torch.sin(coords*4.25+1)
        dark[:observed_coords] = 0
        dark /= dark.norm()
        observed = torch.sin(coords*4.25+1)
        observed[observed_coords:] = 0
        observed /= observed.norm()
        dark_leakage = float(chain(dark, derivatives)[:observed_coords].norm().item())
        observed_gain = float(chain(observed, derivatives)[:observed_coords].norm().item())
    return {
        "hidden_size": n, "alpha": "1", "beta": "1/4", "rounds": rounds,
        "global_one_step_upper_ceil_6": certified_micro,
        "global_one_step_upper": certified_micro / 1_000_000,
        "finite_difference_1e3_gain": finite_diff,
        "jacobian_power_lower_witness": power_gain,
        "zero_final_norm": float(final.norm().item()),
        "observed_coords": observed_coords,
        "dark_to_observed_jvp_gain": dark_leakage,
        "observed_to_observed_jvp_gain": observed_gain,
        "power_iterations": power_iterations,
    }


def run_manifest(checkpoint_dir: Path, events: Path,
                 observed_coords: int = 8, power_iterations: int = 12) -> dict:
    rows = []
    for entry in manifest_entries(checkpoint_dir, events):
        data = load_pinned_weights(entry)
        results = probe_state_dict(*data, rounds=entry["rounds"],
                                   observed_coords=observed_coords,
                                   power_iterations=power_iterations)
        rows.append({k: v for k, v in entry.items() if k != "file"} | results)
    return {
        "study_id": "NRF-03-native-checkpoint-audit-v1",
        "checkpoint_count": len(rows), "power_iterations": power_iterations,
        "rows": rows,
        "claim_boundary": "Stored float32 weights as real-valued parameters, float64 nonlinear replay. Pointwise Jacobian growth is a lower witness; no global full-cycle contraction or behavioral benefit is inferred.",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
        default=root / "experiments/loop_schedule_kappa_surface_v1/checkpoints")
    parser.add_argument("--events", type=Path,
        default=root / "experiments/loop_schedule_kappa_surface_v1/surface_events.jsonl")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--observed-coords", type=int, default=8)
    parser.add_argument("--power-iterations", type=int, default=12)
    args = parser.parse_args()
    receipt = run_manifest(args.checkpoint_dir, args.events,
                           args.observed_coords, args.power_iterations)
    output = json.dumps(receipt, sort_keys=True, indent=2, default=str) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
