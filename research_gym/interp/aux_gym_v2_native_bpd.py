"""Train actual BPD factors on a frozen AUX-V2 checkpoint using pinned upstream BPDTrainer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from research_gym.interp.aux_gym_v2 import digest_file, dump, rows

BPD_REVISION = "722613a80d80af0e80bb9f5ac1ce83eef111e4f7"
SITE = "blocks.0.mlp.2"


def weight_digest(weight):
    return hashlib.sha256(weight.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def import_bpd(checkout):
    import mechinterp
    checkout = checkout.resolve()
    got = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if got != BPD_REVISION:
        raise ValueError(f"BPD revision mismatch: {got}")
    subprocess.run(["git", "-C", str(checkout), "diff", "--quiet", "HEAD", "--", "mechinterp/bpd"], check=True)
    # Both repos have a mechinterp package; attach only the pinned BPD package location.
    mechinterp.__path__.append(str(checkout / "mechinterp"))
    import mechinterp.bpd as bpd
    if not Path(bpd.__file__).resolve().is_relative_to(checkout):
        raise ValueError("BPD import did not resolve to the pinned checkout")
    return bpd


class AuxRolloutAdapter:
    """Native decoder, full continuation, a behavioral lens at every answer visit."""

    def __init__(self, model):
        self.model = model

    def __call__(self, batch, n_passes, pass_offset=0, init_state=None):
        if pass_offset or init_state is not None:
            raise ValueError("closed-loop adapter does not support state transplantation")
        device = next(self.model.parameters()).device
        tokens = batch.to(device)
        output = self.model(tokens, depth_visits=n_passes)
        readouts = {i: self.model.lens(state[:, -1:]) for i, state in enumerate(output.visit_outputs)}
        boundaries = {i: state for i, state in enumerate(output.visit_outputs)}
        return readouts, boundaries, None, None


def selected_output_basis(v, u, rank):
    """Select learned factors by gauge-invariant product norm, then orthonormalize their output span."""
    if v.ndim != 2 or u.ndim != 2 or v.shape[1] != u.shape[0]:
        raise ValueError("incompatible learned BPD factors")
    if not 1 <= rank <= min(u.shape):
        raise ValueError("rank outside learned component dimensions")
    strength = v.float().norm(dim=0) * u.float().norm(dim=1)
    selected = strength.argsort(descending=True, stable=True)[:rank]
    q, s, _ = torch.linalg.svd(u.float()[selected].T, full_matrices=False)
    if float(s[-1]) <= float(s[0]) * 1e-6:
        raise ValueError("learned selected output span is numerically rank deficient")
    return q[:, :rank].contiguous(), selected.tolist()


def load_native_basis(directory, checkpoint_hash, rank, weight):
    receipt = json.loads((directory / "receipt.json").read_text())
    if (receipt["bpd_revision"] != BPD_REVISION or receipt["site"] != SITE
            or receipt["teacher_checkpoint_sha256"] != checkpoint_hash
            or receipt["teacher_weight_sha256"] != weight_digest(weight)
            or receipt["factor_sha256"] != digest_file(directory / "factors.pt")):
        raise ValueError("native BPD factor provenance mismatch")
    factors = torch.load(directory / "factors.pt", map_location="cpu", weights_only=True)
    basis, _ = selected_output_basis(factors["V"], factors["U"], rank)
    if basis.shape[0] != weight.shape[0]:
        raise ValueError("learned output span shape mismatch")
    return basis


def fit(args):
    from research_gym.interp.models import build_model
    if args.out.exists():
        raise FileExistsError(args.out)
    if min(args.steps, args.batch, args.components, args.threads) < 1 or args.warmup < 0:
        raise ValueError("positive steps/batch/components/threads, nonnegative warmup")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    bpd = import_bpd(args.bpd_checkout)
    receipt = json.loads((args.run / "receipt.json").read_text())
    manifest = json.loads((args.data / "manifest.json").read_text())
    if (receipt["arm"] != "LOOP-T" or receipt["visits"] != 8
            or receipt["manifest_sha256"] != digest_file(args.data / "manifest.json")
            or receipt["checkpoint_sha256"] != digest_file(args.run / "model.pt")):
        raise ValueError("teacher checkpoint provenance mismatch")
    def checked(split):
        name = f"{split}/{receipt['family']}.jsonl"
        path = args.data / name
        if digest_file(path) != manifest["files"][name]["sha256"]:
            raise ValueError("shard hash mismatch")
        return torch.tensor([r["tokens"] for r in rows(path)])
    train, val = checked("train"), checked("validation")
    device = torch.device(args.device)
    model = build_model("LOOP-T", visits=8).to(device)
    model.load_state_dict(torch.load(args.run / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    teacher_weight = model.get_submodule(SITE).weight.detach().clone()
    # Fit uses teacher behaviors only, not intermediate symbolic or final gold labels.
    calibration = train[torch.randperm(len(train), generator=rng)[:args.batch]].to(device)
    with torch.no_grad():
        clean_before = model(calibration).logits[:, -1].clone()
    bmodel = bpd.BPDModel(model, {SITE: args.components}, max_passes=8)
    adapter = AuxRolloutAdapter(model)
    cfg = bpd.BPDConfig(c_per_module={SITE: args.components}, n_passes=8,
        n_steps=args.steps, batch_size=args.batch, seq_len=64, seed=args.seed,
        rollout_mode="closed_loop", faithfulness_warmup_steps=args.warmup,
        ci=bpd.CIConfig(d_model=32, n_blocks=1, n_heads=4, mlp_hidden=64,
                        max_passes=8, max_seq_len=64),
        null_calibration=bpd.NullCalibrationConfig(use_tolerance_weights=not args.smoke_no_null,
                                                  n_rotations_train=4))
    trainer = bpd.BPDTrainer(bmodel, cfg, adapter, device=device)
    clean = trainer.run_clean_rollout(calibration)
    parity = float((clean.readouts[7][:, -1] - clean_before).abs().max())
    if parity > 1e-5:
        raise ValueError(f"wrapped clean teacher parity failed: {parity}")
    warm_loss = trainer.run_faithfulness_warmup()
    tolerances = trainer.calibrate_baselines(calibration)
    history = []
    for step in range(args.steps):
        idx = torch.randint(len(train), (args.batch,), generator=rng)
        result = trainer.train_step(train[idx].to(device), step)
        history.append(asdict(result))
    # Fixed development sample, separate from teacher fitting.
    holdout = val[:min(len(val), args.batch)].to(device)
    evaluation, _, _ = trainer.evaluate(holdout)
    wrapper = bmodel.wrappers[SITE]
    factors = {"V": wrapper.V.detach().cpu(), "U": wrapper.U.detach().cpu()}
    reconstruction = (factors["V"] @ factors["U"]).T
    target = teacher_weight.cpu().float()
    residual = float((target - reconstruction).norm() / target.norm().clamp_min(1e-12))
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(factors, args.out / "factors.pt")
    torch.save(trainer.ci_net.state_dict(), args.out / "ci.pt")
    dump(args.out / "receipt.json", {
        "bpd_revision": BPD_REVISION, "site": SITE,
        "teacher_checkpoint_sha256": receipt["checkpoint_sha256"],
        "teacher_weight_sha256": weight_digest(teacher_weight),
        "manifest_sha256": receipt["manifest_sha256"], "family": receipt["family"],
        "factor_sha256": digest_file(args.out / "factors.pt"),
        "ci_sha256": digest_file(args.out / "ci.pt"),
        "seed": args.seed, "components": args.components, "steps": args.steps,
        "batch": args.batch, "device": str(device), "bpd_config": asdict(cfg),
        "clean_logit_parity_max_abs": parity, "warmup_faith_loss": warm_loss,
        "relative_weight_residual": residual, "tolerances": tolerances,
        "history": history, "validation_teacher_reconstruction": evaluation,
        "scope": "native single-physical-site BPD fit; development behavior reconstruction only",
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bpd-checkout", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--components", type=int, default=32)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--smoke-no-null", action="store_true")
    fit(p.parse_args())


if __name__ == "__main__":
    main()
