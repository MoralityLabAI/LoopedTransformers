"""Fit native BPD only on the predeclared competent clock teacher."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import resource
from pathlib import Path

import torch

from research_gym.interp.aux_gym_v2 import digest_file, dump
from research_gym.interp.aux_gym_v2_native_bpd import weight_digest
from research_gym.interp.molt_bpd_source import BPD_REVISION, load_verified
from research_gym.interp.molt_typed_clock import (
    SITE, TypedClockDecoder, bank_digest, competent, evaluate, make_worlds, primitive_exact,
)

ARM = "addressed"
SEED = 157


class ClockRollout:
    def __init__(self, model):
        self.model = model

    def __call__(self, batch, n_passes, pass_offset=0, init_state=None):
        if pass_offset or init_state is not None:
            raise ValueError("closed-loop clock adapter only")
        logits, states = self.model.rollout(batch, n_passes)
        return ({i: logits[:, i:i + 1] for i in range(n_passes)},
                {i: states[:, i:i + 1] for i in range(n_passes)}, None, None)


def fit(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    report = json.loads((args.clock / "receipt.json").read_text())
    if (report["schema"] != "molt12-clock-v1" or report["promotion_authorized"] is not False
            or report["physical_site"] != SITE):
        raise ValueError("unexpected competence receipt")
    rows = [r for r in report["rows"] if r["arm"] == ARM and r["seed"] == SEED]
    if len(rows) != 1:
        raise ValueError("predeclared teacher missing")
    row = rows[0]
    ck = args.clock / f"{ARM}-{SEED}.pt"
    if row["checkpoint_sha256"] != digest_file(ck):
        raise ValueError("checkpoint hash mismatch")
    model = TypedClockDecoder(ARM)
    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    model.eval()
    train = make_worlds(121001, 4, 512)
    if bank_digest(train) != report["train_worlds_sha256"]:
        raise ValueError("training roster mismatch")
    dev = {}
    for depth in (4, 8):
        worlds = make_worlds(121100 + depth, depth, row["development"][str(depth)]["worlds"], train)
        if bank_digest(worlds) != report["development_roster_sha256"][str(depth)]:
            raise ValueError("development roster mismatch")
        dev[str(depth)] = evaluate(model, worlds)
        if dev[str(depth)]["unrestricted_predictions"] != row["development"][str(depth)]["unrestricted_predictions"]:
            raise ValueError("competence replay mismatch")
    actual = {"primitive_after_rollout": primitive_exact(model), "development": dev}
    passed = competent(actual)
    if passed != row["competence_gate_passed"]:
        raise ValueError("competence gate mismatch")
    args.out.mkdir(parents=True, exist_ok=False)
    base = {"schema": "molt12-clock-native-bpd-v1", "arm": ARM, "seed": SEED,
            "checkpoint_sha256": row["checkpoint_sha256"], "bpd_revision": BPD_REVISION,
            "site": SITE, "competence_gate_passed": passed,
            "bpd_source_receipt_sha256": digest_file(args.bpd_source_receipt),
            "promotion_authorized": False, "confirmation_ready": False}
    if not passed:
        dump(args.out / "receipt.json", {**base, "status": "competence_gate_failed"})
        print(json.dumps({**base, "status": "competence_gate_failed"}, sort_keys=True))
        return
    bpd = load_verified(args.bpd_checkout, args.bpd_source_receipt)
    torch.manual_seed(SEED + 9000)
    # Use prefixes of fixed TRAIN sequences. No new labels enter factor fitting.
    tokens = torch.tensor([w[:5] for w in train])
    rng = torch.Generator().manual_seed(SEED + 9100)
    calibration = tokens[:4]
    with torch.no_grad():
        expected = model.rollout(calibration, 4)[0]
    weight = model.get_submodule(SITE).weight.detach().clone()
    bmodel = bpd.BPDModel(model, {SITE: 16}, max_passes=4)
    cfg = bpd.BPDConfig(c_per_module={SITE: 16}, n_passes=4, n_steps=8,
                        batch_size=2, seq_len=3, seed=SEED,
                        rollout_mode="closed_loop", faithfulness_warmup_steps=4,
                        ci=bpd.CIConfig(d_model=32, n_blocks=1, n_heads=4, mlp_hidden=64,
                                        max_passes=4, max_seq_len=16),
                        null_calibration=bpd.NullCalibrationConfig(use_tolerance_weights=False))
    trainer = bpd.BPDTrainer(bmodel, cfg, ClockRollout(model), device=torch.device("cpu"))
    captured = trainer.run_clean_rollout(calibration)
    before = torch.cat([captured.readouts[i] for i in range(4)], dim=1)
    parity = float((before - expected).abs().max())
    if parity > 1e-5:
        raise RuntimeError("clean native wrapper parity failed")
    warm = trainer.run_faithfulness_warmup()
    tolerances = trainer.calibrate_baselines(calibration)
    history = []
    for i in range(8):
        idx = torch.randint(len(tokens), (2,), generator=rng)
        history.append(asdict(trainer.train_step(tokens[idx], i)))
    validation = torch.tensor(make_worlds(121999, 2, 8, [w[:5] for w in train]))
    evaluation, _, _ = trainer.evaluate(validation)
    after = trainer.run_clean_rollout(calibration)
    post = torch.cat([after.readouts[i] for i in range(4)], dim=1)
    post_parity = float((post - expected).abs().max())
    if post_parity > 1e-5:
        raise RuntimeError("clean teacher changed after factor fitting")
    wrapper = bmodel.wrappers[SITE]
    factors = {"V": wrapper.V.detach().cpu(), "U": wrapper.U.detach().cpu()}
    residual = float((weight - (factors["V"] @ factors["U"]).T).norm() / weight.norm().clamp_min(1e-12))
    torch.save(factors, args.out / "factors.pt")
    torch.save(trainer.ci_net.state_dict(), args.out / "ci.pt")
    out = {**base, "status": "engineering_fit_completed",
           "teacher_weight_sha256": weight_digest(weight), "factor_sha256": digest_file(args.out / "factors.pt"),
           "ci_sha256": digest_file(args.out / "ci.pt"), "components": 16, "steps": 8,
           "warmup_steps": 4, "null_calibration_enabled": False,
           "clean_parity_max_abs": parity, "post_fit_clean_parity_max_abs": post_parity,
           "relative_weight_residual": residual, "warmup_faith_loss": warm,
           "tolerances": tolerances, "history": history, "validation_teacher_reconstruction": evaluation,
           "bpd_config": asdict(cfg), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "factor_quality_established": False,
           "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024}
    dump(args.out / "receipt.json", out)
    print(json.dumps({k: out[k] for k in ("status", "competence_gate_passed", "components", "steps",
                                         "clean_parity_max_abs", "post_fit_clean_parity_max_abs",
                                         "relative_weight_residual", "validation_teacher_reconstruction")},
                     sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("clock", "bpd-checkout", "bpd-source-receipt", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    fit(args)


if __name__ == "__main__":
    main()
