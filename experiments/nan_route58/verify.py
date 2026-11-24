"""CPU verification of live telemetry, fresh-source qualification and host scoring."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from experiments.nan_delay39.contrasts import numpy_state
from experiments.nan_dose51.native import RouteCapture
from experiments.nan_gated46.verify import fixture_loop
from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime import TARGETS
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha

from .capture import ObservedCapture, fingerprint
from .custody import HERE, SOURCE_CHECK, bound_source, settings
from .host import ARMS, KEYS, SIGNS, audit, native_gate, score_world
from experiments.nan_sham56.native import continuation


def refuses(operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError("An invalid fixture was admitted")


def record(winner, donor):
    arms = {}
    pre = {}
    for name in ARMS:
        for sign in SIGNS:
            key = f"{name}/{sign}"
            polarity = 1 if sign == "positive" else -1
            margin = polarity * (.03125 if name == "contrast" else .015625)
            arms[key] = {"requested_l2": 2., "realized_l2": 2., "direction_cosine": 1., "eligible": True,
                         "cosine_with_original_contrast": .9 if name == "contrast" else 0.,
                         "live_phase1": {"passed": True, "greedy_same": True, "max_abs_logit_delta": .0625,
                                         "margin_delta_abs": abs(margin)}}
            pre[key] = {"rms_logit_drift": .025, "signed_bare_margin_delta": margin}
    after = {k: {"phase2_bpd_read": [0.] * 8, "bare_margin": m, "winner": donor if k == "contrast/positive" else winner}
             for k, m in zip(("contrast/positive", "contrast/negative", "sham/positive", "sham/negative"),
                            (.625, .125, .1875, .0625), strict=True)}
    return {"status": "closed", "edited_pass2_runs": 4, "vocabulary_size": 166144,
            "phase1_residual_l2": 2. / .0075,
            "baseline": {"winner": winner, "bare_margin": -.125, "phase2_bpd_read": [0.] * 8},
            "cases": {"physical_gate": True, "arm_dose": arms, "distance_in_nominal_radii": dict.fromkeys(SIGNS, 1.4)},
            "phase1_profile": pre, "outcomes": after}


def host_fixture(folder, work, cfg, round_cfg, dose):
    work.mkdir(exist_ok=False)
    gold = bound_source(folder, "gold")
    labels = bound_source(folder, "tokens")["bare_label_token_ids"]
    native = [{"id": r["id"], "text": r["answer"] if r["arm"] != "focused_erased" else "A",
               "token_id": labels[r["answer"]] if r["arm"] != "focused_erased" else labels["A"]} for r in gold]
    gate = native_gate(folder, native, cfg)
    assert gate["passed"] and len(gate["qualified_focused_roots"]) == 12
    altered = deepcopy(native)
    idx = next(i for i, r in enumerate(gold) if r["arm"] == "copy")
    altered[idx] = altered[idx] | {"text": "X"}
    assert not native_gate(folder, altered, cfg)["passed"]
    focused = {(r["root"], r["world"]): r for r in gold if r["arm"] == "focused"}
    root = min(key[0] for key in focused)
    example = None
    workbook = []
    for i, ((group, world), row) in enumerate(sorted(focused.items())):
        if group == root:
            foil = focused[group, "B" if world == "A" else "A"]
            winner, donor = labels[row["answer"]], labels[foil["answer"]]
            item = record(winner, donor)
            if example is None:
                example = item, winner, donor
        else:
            item = {"status": "head_unqualified", "edited_pass2_runs": 0}
        path = work / f"episode_{i:03d}.json"
        write_new(path, item | {"source_id": row["id"]})
        workbook.append({"file": path.name, "sha256": sha(path)})
    compared = {"greedy_tokens_equal": True, "vocabulary_rows_compared": 166144}
    parity = {"prefill": [compared, compared], "increment": [compared, compared], "pruned_increment": compared}
    depth = {"tested_max_position": json.loads(SOURCE_CHECK.read_bytes())["max_prompt_length"] + 2,
             "KV_bytes_at_max_position": 1, "checks": {"actual_CAL": parity, "padded_CAL": parity}}
    preflight = {"synthetic": True, "depths": {"1": depth, "2": depth}}
    for name, value in (("preflight.json", preflight), ("native.json", native), ("qualification.json", gate)):
        write_new(work / name, value)
    journal = work / "closed.jsonl"
    with journal.open("x", encoding="utf-8", newline="\n") as stream:
        for row in workbook:
            stream.write(json.dumps(row) + "\n")
    tag = "0" * 64
    target = TARGETS[cfg["backend"]]
    write_new(work / "terminal.json", {"runtime_sha256": tag, "backend": cfg["backend"],
        "target_before": target, "target_after": target, "quant_unchanged": True, "checkpoint_verified_after": True,
        "status": "completed", "preflight_sha256": sha(work / "preflight.json"), "native_sha256": sha(work / "native.json"),
        "qualification_sha256": sha(work / "qualification.json"), "closed_journal_sha256": sha(journal),
        "records": 24, "vocabulary_size": 166144})
    outcome = audit(folder, work, tag, cfg, round_cfg, dose)
    assert outcome["roots_with_both_worlds_scored"] == 1 and outcome["worlds_scored"] == 2
    assert outcome["refusals"] == {"head_unqualified": 22} and outcome["root_mean"] == {
        "signed_pass2_margin_contrast_minus_sham": .1875, "signed_pass1_margin_contrast_minus_sham": .015625,
        "max_signed_rms_gap": 0., "active_positive_bare_donor_flip": 1., "sham_positive_bare_donor_flip": 0.}
    first, winner, donor = example
    tampered = deepcopy(first)
    tampered["phase1_profile"]["sham/positive"]["rms_logit_drift"] = .8
    refuses(lambda: score_world(tampered, winner, donor, round_cfg, dose))
    ambiguous = deepcopy(first)
    ambiguous["cases"]["arm_dose"]["sham/negative"]["cosine_with_original_contrast"] = .2
    refuses(lambda: score_world(ambiguous, winner, donor, round_cfg, dose))
    wrong_radius = deepcopy(first)
    wrong_radius["phase1_residual_l2"] *= 2
    refuses(lambda: score_world(wrong_radius, winner, donor, round_cfg, dose))
    with journal.open("ab") as stream:
        stream.write(b"\n")
    refuses(lambda: audit(folder, work, tag, cfg, round_cfg, dose))
    return outcome


def live_fixture(sham_cfg, round_cfg, dose, check):
    import numpy as np
    import torch
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    torch.set_num_threads(1)
    torch.manual_seed(56)
    rng = np.random.default_rng(56)
    loop = fixture_loop(torch, torch.nn)
    weights = (.012 * torch.randn(128, 64)).to(torch.bfloat16)
    weights[0].fill_(.125)
    head = StreamedVocabularyHead(weights, loop.device, 11)
    engine = RouteCapture(loop, head, layer=1, chunk=2)
    ids = [1, 3, 8, 14, 22]
    toy = dose | {"phase1_dose_fraction": .06, "maximum_head_delta": .5, "block_rows": 17}
    mask = np.array([.1, .9, .1, .9, .2, .8, .3, .7], dtype=np.float32)
    try:
        baseline = engine.capture([ids, ids])
        y = numpy_state(baseline["native_writes"][1, 1]).astype(np.float32)
        bank = {"V": np.eye(64, 8, dtype=np.float32),
                "U": np.linalg.qr(rng.normal(size=(64, 8)), mode="reduced")[0].T.astype(np.float32),
                "mean_comp_acts": np.zeros((2, 8), np.float32), "mean_delta_out": np.stack((y*.2, y*.2)),
                "mean_write_rms": np.full(2, .6, np.float32)}
        bundle = (bank, None, 1e-6, 1-mask)
        observed = ObservedCapture(engine, fingerprint(baseline, 0), (2, 1))
        result = continuation(observed, ids, (2, 1), "fixture", bundle, ("ambient_peak_minimax", .25),
                              sham_cfg, round_cfg, toy, check, lambda *_: mask[None], np)
        result = observed.bind(result)
        assert result["status"] == "closed" and set(result["phase1_profile"]) == KEYS
        assert all(r["rms_logit_drift"] > 0 for r in result["phase1_profile"].values())
        measured = score_world(result, int(baseline["logits"][1, 1].argmax()), 2, round_cfg, toy)
        assert measured is not None and not torch.cuda.is_initialized()
        mismatched = ObservedCapture(engine, ("wrong prior row",), (2, 1))
        try:
            mismatched.capture([ids, ids])
        except RuntimeError:
            pass
        else:
            raise AssertionError("Mismatched prior paired row was admitted")
        return result["edited_pass2_runs"]
    finally:
        engine.close()
        head.close()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-dir", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    cfg, sham_cfg, round_cfg, dose = settings()
    args.output.mkdir(parents=True, exist_ok=False)
    limits = {"minimum_start_available_RAM_bytes": 5*2**30, "minimum_available_RAM_bytes": 4*2**30,
              "maximum_private_bytes": 768*2**20, "maximum_seconds": 160}
    with lease(limits) as (check, samples):
        scored = host_fixture(args.source_dir, args.output / "synthetic_terminal", cfg, round_cfg, dose)
        os.environ.update(USE_TORCH="1", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")
        edits = live_fixture(sham_cfg, round_cfg, dose, check)
        check()
    receipt = {"status": "passed", "synthetic_signed_edits": edits, "verified_host_worlds": scored["worlds_scored"],
               "verified_host_registered_worlds": 2*scored["roots_registered"],
               "synthetic_host_signed_pass2_mean": scored["root_mean"]["signed_pass2_margin_contrast_minus_sham"],
               "synthetic_host_signed_pass1_mean": scored["root_mean"]["signed_pass1_margin_contrast_minus_sham"],
               "invalid_receipts_refused": 4, "mismatched_baseline_refused": True,
               "cuda_initialized": False, "pretrained_decoder_forwards": 0,
               "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
               "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
               "files_sha256": {name: sha(HERE / name) for name in ("capture.py", "custody.py", "host.py", "worker.py", "verify.py")}}
    write_new(args.output / "verification.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
