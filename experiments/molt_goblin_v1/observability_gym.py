"""MOLT-06: trained strict-prefix delay chain and stacked-readout controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from lsa.observability import matched_sham, spectrum

LOOPS = 5
CUT = 2
WIDTH = 16
RANK = 4
ARMS = ("intact", "corrupt", "full_restore", "target", "terminal_sham", "stack_sham")


class DelayChain(nn.Module):
    """Tied local graph block: only the preceding position crosses a visit."""
    def __init__(self, width=WIDTH):
        super().__init__()
        self.input = nn.Linear(2, width)
        self.recurrent = nn.Linear(width, width, bias=False)
        self.head = nn.Linear(width, 2)

    def step(self, x, h):
        parent = torch.cat((torch.zeros_like(h[..., :1, :]), h[..., :-1, :]), dim=-2)
        return torch.tanh(self.input(x) + self.recurrent(parent))

    def trace(self, x, loops=LOOPS):
        h = x.new_zeros((*x.shape[:-1], self.head.in_features))
        states = []
        for _ in range(loops):
            h = self.step(x, h)
            states.append(h)
        return states


def sample(seed, count, length, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    return (2 * torch.randint(0, 2, (count, length, 2), generator=generator) - 1).to(dtype)


def labels(x, visit):
    """Two exact bit-copy targets from different source distances.

    A target only exists after its causal distance has elapsed.
    """
    n = x.shape[-2]
    target = torch.zeros_like(x)
    mask = torch.zeros((n, 2), dtype=torch.bool, device=x.device)
    for channel, offset in enumerate((1, 3)):
        if visit > offset and n > offset:
            target[..., offset:, channel] = x[..., :-offset, channel]
            mask[offset:, channel] = True
    return target, mask


def train(seed, steps):
    torch.manual_seed(seed)
    model = DelayChain()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006)
    for update in range(steps):
        x = sample(100000 * seed + update, 32, 8)
        losses = []
        for visit, state in enumerate(model.trace(x), 1):
            target, mask = labels(x, visit)
            if mask.any():
                logits = model.head(state)[..., mask]
                losses.append(nn.functional.softplus(-logits * target[..., mask]).mean())
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
    return model.eval().double()


def cut_states(model, x, position):
    clean = model.trace(x, CUT)[-1]
    altered = x.clone()
    altered[position, 0] *= -1
    altered[position - 1, 1] *= -1
    corrupt = clean.clone()
    corrupt[position] = model.trace(altered, CUT)[-1][position]
    return clean, corrupt


def observe(model, x, state, position, delta=None):
    """Full supervised suffix margin vector, plus fixed structural cone mask."""
    if delta is not None:
        mask = torch.zeros_like(state)
        mask[position] = delta
        state = state + mask
    out, targets, cone, terminal = [], [], [], []
    for visit in range(CUT + 1, LOOPS + 1):
        state = model.step(x, state)
        target, mask = labels(x, visit)
        local = torch.zeros_like(mask)
        if position + visit - CUT < len(x):
            local[position + visit - CUT] = True
        logits = model.head(state)[mask]
        out.append(logits)
        targets.append(target[mask])
        cone.append(local[mask])
        terminal.append(torch.full_like(logits, visit == LOOPS, dtype=torch.bool))
    return torch.cat(out), torch.cat(targets), torch.cat(cone), torch.cat(terminal)


def analytic(model, x, state, position):
    """Propagate full support-coordinate tangent along the declared prefix edges."""
    n, width = state.shape
    tangent = state.new_zeros((n, width, width))
    tangent[position] = torch.eye(width, dtype=state.dtype)
    blocks = []
    for visit in range(CUT + 1, LOOPS + 1):
        state = model.step(x, state)
        parent = torch.cat((torch.zeros_like(tangent[:1]), tangent[:-1]), dim=0)
        tangent = (model.recurrent.weight @ parent) * (1 - state.square())[..., None]
        _, mask = labels(x, visit)
        blocks.append((model.head.weight @ tangent)[mask])
    return torch.cat(blocks)


def choose_basis(model, seed, count=64):
    # Left singular vectors define a recipient-space projection. This is not
    # a factorization of the nonlinear state write or native BPD factors.
    basis, _, _ = torch.linalg.svd(model.recurrent.weight, full_matrices=False)
    energy = torch.zeros(basis.shape[1], dtype=torch.float64)
    for x in sample(700000 + seed, count, 8, torch.float64):
        clean, corrupt = cut_states(model, x, 2)
        delta = clean[2] - corrupt[2]
        o = analytic(model, x, corrupt, 2)
        energy += (o @ basis).square().sum(0) * (basis.T @ delta).square()
    order = sorted(range(len(energy)), key=lambda i: (-float(energy[i]), i))[:RANK]
    return basis[:, order], order, energy.tolist()


def correct(logits, target):
    return logits * target > 0


def curvature(model, x, base, position, vector, eps):
    z = observe(model, x, base, position)[0]
    a = observe(model, x, base, position, eps * vector)[0]
    b = observe(model, x, base, position, -eps * vector)[0]
    return (a + b - 2 * z) / eps**2


def sha_state(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def world(model, x, basis, seed, index):
    position = len(x) // 2 - 2
    clean, corrupt = cut_states(model, x, position)
    intact, target, cone, terminal = observe(model, x, clean, position)
    bad = observe(model, x, corrupt, position)[0]
    delta = clean[position] - corrupt[position]
    chosen = basis @ (basis.T @ delta)
    full = observe(model, x, corrupt, position, delta)[0]
    if not torch.allclose(intact, full, atol=1e-10, rtol=0):
        raise RuntimeError("restoration parity failed")
    if float((intact[~cone] - bad[~cone]).abs().max()) > 1e-10:
        raise RuntimeError("causal-cone violation")
    rows, audits = [], {}
    for condition, base in (("corrupt", corrupt), ("clean", clean)):
        o = analytic(model, x, base, position)
        if float(o[~cone].abs().max()) > 1e-12:
            raise RuntimeError("tangent outside structural cone")
        # Compare analytic full-suffix derivative to autograd on a fixed root
        # per model/length, with no outcome-conditioned choice.
        derivative_error = None
        if index == 0:
            with torch.enable_grad():
                direct = torch.autograd.functional.jacobian(
                    lambda d: observe(model, x, base, position, d)[0],
                    torch.zeros_like(delta, requires_grad=True),
                    vectorize=True,
                )
            derivative_error = float((direct - o).abs().max())
            if derivative_error > 1e-9:
                raise RuntimeError("full Jacobian mismatch")
        generator = torch.Generator().manual_seed(seed * 1000000 + index * 10 + len(x))
        final_sham, final_receipt = matched_sham(chosen, base[position], o[terminal], generator)
        stack_sham, stack_receipt = matched_sham(chosen, base[position], o, generator)
        baseline = observe(model, x, base, position)[0]
        vectors = {"target": chosen, "terminal_sham": final_sham, "stack_sham": stack_sham}
        receipts = {"terminal_sham": final_receipt, "stack_sham": stack_receipt}
        outputs = {"intact": intact, "corrupt": bad, "full_restore": full}
        for arm, vector in vectors.items():
            status = receipts[arm]["status"] if arm in receipts else "matched"
            applied = vector if status in {"matched", "zero_dose"} else torch.zeros_like(vector)
            outputs[arm] = observe(model, x, base, position, applied)[0]
            if arm in receipts:
                receipts[arm]["stack_signed_error"] = float((o @ (vector - chosen)).norm())
                receipts[arm]["executed"] = status in {"matched", "zero_dose"}
            rows.append({
                "condition": condition, "arm": arm, "cone_verified": bool(correct(outputs[arm], target)[cone].all()),
                "all_verified": bool(correct(outputs[arm], target).all()),
                "cone_accuracy": float(correct(outputs[arm], target)[cone].double().mean()),
                "intact_eligible": bool(correct(intact, target)[cone].all()),
                "corrupt_correct": bool(correct(bad, target)[cone].all()),
                "match_status": status,
                "mean_logit_error": float((outputs[arm] - intact).abs().mean()),
                "nonlinear_remainder_norm": float((outputs[arm] - baseline - o @ applied).norm()),
                "curvature_norm": float(curvature(model, x, base, position, applied, 0.005).norm()),
                "curvature_step_error": float((curvature(model, x, base, position, applied, 0.01)
                                               - curvature(model, x, base, position, applied, 0.005)).norm()),
            })
        for arm in ("intact", "corrupt", "full_restore"):
            rows.append({"condition": condition, "arm": arm,
                         "cone_verified": bool(correct(outputs[arm], target)[cone].all()),
                         "all_verified": bool(correct(outputs[arm], target).all()),
                         "cone_accuracy": float(correct(outputs[arm], target)[cone].double().mean()),
                         "intact_eligible": bool(correct(intact, target)[cone].all()),
                         "corrupt_correct": bool(correct(bad, target)[cone].all()), "match_status": "reference",
                         "mean_logit_error": float((outputs[arm] - intact).abs().mean())})
        audits[condition] = {
            "terminal": spectrum(o[terminal]), "stack": spectrum(o),
            "terminal_sham": final_receipt, "stack_sham": stack_receipt,
            "analytic_autograd_max_abs": derivative_error,
            "curvature_target_minus_stack_norm": float((
                curvature(model, x, base, position, chosen, 0.005)
                - curvature(model, x, base, position, stack_sham, 0.005)).norm())
                if stack_receipt["executed"] else None,
        }
    return rows, audits


def aggregate(rows):
    results = []
    for length in sorted({r["length"] for r in rows}):
        group = [r for r in rows if r["length"] == length]
        for condition in ("corrupt", "clean"):
            for arm in ARMS:
                cell = [r for r in group if r["condition"] == condition and r["arm"] == arm]
                eligible = [r for r in cell if r["intact_eligible"]]
                broken = [r for r in eligible if not r["corrupt_correct"]]
                results.append({"length": length, "condition": condition, "arm": arm, "rows": len(cell),
                                "verified": sum(r["cone_verified"] for r in cell),
                                "intact_eligible": len(eligible), "headroom": len(broken),
                                "recovered": sum(r["cone_verified"] for r in broken),
                                "clean_damage": sum(not r["cone_verified"] for r in eligible),
                                "unmatched": sum(r["match_status"] in {"invalid_match", "no_distinct_match"}
                                                 for r in cell)})
    return results


def run(steps, roots, seeds):
    if not 1 <= steps <= 1500 or not 1 <= roots <= 128 or not 1 <= len(seeds) <= 8:
        raise ValueError("development caps")
    if len(set(seeds)) != len(seeds) or not all(0 <= s <= 10000 for s in seeds):
        raise ValueError("invalid seeds")
    torch.set_num_threads(1)
    rows, audits, fits = [], [], []
    for seed in seeds:
        model = train(seed, steps)
        checkpoint = sha_state(model)
        with torch.no_grad():
            basis, ids, energy = choose_basis(model, seed)
        fits.append({"model_seed": seed, "checkpoint": checkpoint, "components": ids,
                     "train_observable_component_energy": energy, "updates": steps,
                     "forward_token_visits": steps * 32 * 8 * LOOPS})
        for length in (8, 12):
            # Shared development tasks across seeds: crossed model/world dependence.
            for index, x in enumerate(sample(890000 + length, roots, length, torch.float64)):
                result, audit = world(model, x, basis, seed, index)
                root = f"molt06-dev-{length}-{index}"
                rows.extend({"model_seed": seed, "root": root, "length": length, **r} for r in result)
                audits.append({"model_seed": seed, "root": root, "length": length, **audit})
        if sha_state(model) != checkpoint:
            raise RuntimeError("frozen weights changed")
    summary = aggregate(rows)
    matches = [a[c]["stack_sham"] for a in audits for c in ("clean", "corrupt")]
    rank_rows = [a["corrupt"] for a in audits]
    geometry = {"rank_pairs": sorted({(a["terminal"]["d_obs"], a["stack"]["d_obs"]) for a in rank_rows}),
                "matched": sum(a["status"] == "matched" for a in matches),
                "zero_dose": sum(a["status"] == "zero_dose" for a in matches),
                "unmatched": sum(a["status"] not in {"matched", "zero_dose"} for a in matches)}
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
               (Path(__file__), Path(__file__).resolve().parents[2] / "lsa/observability.py")}
    return {"schema": "molt06-observability-v1", "status": "development_only", "torch": torch.__version__,
            "world_roots": 2 * roots, "model_seeds": list(seeds), "sources": sources,
            "fits": fits, "rows": rows, "audits": audits, "summary": summary, "geometry": geometry,
            "confirmation_ready": False, "native_bpd": False, "promotion_authorized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--roots", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[71, 73, 79, 83])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output must be fresh")
    report = run(args.steps, args.roots, tuple(args.seeds))
    with args.out.open("x") as file:
        json.dump(report, file, sort_keys=True, allow_nan=False)
    print(json.dumps({k: report[k] for k in ("status", "world_roots", "geometry", "summary")}, sort_keys=True))


if __name__ == "__main__":
    main()
