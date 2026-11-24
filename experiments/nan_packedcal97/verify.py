"""Commission packed signed measurements and the larger-root sham statistic."""

import argparse
from copy import deepcopy
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

from experiments.nan_delay39.contrasts import contrast_directions, dose_status, measure_pair, numpy_state, qualified_pair
from experiments.nan_delay39.instrument import DelayedCapture, prepare
from experiments.nan_delay39.source import digest
from experiments.nan_phase35.run import lease
from experiments.nan_premlp64.verify import BUDGET, CONFIGURATION, PACKAGES, hook_count
from experiments.nan_passreuse96.capture import SharedPassCapture
from experiments.nan_readpath37.run import bound, write_new

from .host import qualify, summarize
from .lane import verified
from .measure import collect
from .source import REPO, config, frozen_source, read_source


def fit_fixture(reference, ids, donor, cfg, np):
    rng = np.random.default_rng(97131)
    native = reference.capture([ids, donor])
    clean = reference.capture([ids, ids])
    before = numpy_state(clean["residual"][1, 1])
    x, y = (numpy_state(native["inputs"][1, i]) for i in (0, 1))
    v = rng.normal(size=(len(x), 8))
    phase = cfg["phase1_components"]
    for _ in range(256):
        u = np.linalg.qr(rng.normal(size=(len(before), 8)), mode="reduced")[0].T
        sham = contrast_directions(x @ v, y @ v, u, phase, "packedcal97_cpu", np)
        if sham is None:
            continue
        centered = [[prepare(before, sign * u[i], cfg["phase1_dose_fraction"], cfg["dose_relative_tolerance"],
                             cfg["minimum_direction_cosine"], np) for sign in (1, -1)] for i in phase]
        rest = {name: prepare(before, value, cfg["phase1_dose_fraction"], cfg["dose_relative_tolerance"],
                               cfg["minimum_direction_cosine"], np) for name, value in sham.items()}
        if (not all(qualified_pair(dose_status(a), dose_status(b), cfg) for a, b in centered)
                or not all(p.eligible for p in rest.values())):
            continue
        if all(abs(rest["twin_donor"].realized_l2 - rest[tag].realized_l2)
               <= rest[tag].requested_l2 * cfg["max_contrast_dose_gap_fraction"] for tag in ("span_sham", "ambient_sham")):
            return v, u
    raise RuntimeError("The controlled dense fixture did not qualify its synthetic writer axes")


def compare_native(snapshot, cfg, packs, check, np, torch):
    from experiments.nan_recomp09.common import target_digest
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import build_nanbeige_model_and_loop
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    torch.manual_seed(97031)
    model, loop = build_nanbeige_model_and_loop(snapshot, exec_device=torch.device("cpu"), dtype=torch.bfloat16,
        n_gpu_layers=1, random_init=True, config_overrides={"vocab_size": 97, "num_hidden_layers": 2,
            "hidden_size": 128, "intermediate_size": 72, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "kv_channels": 16})
    model.requires_grad_(False)
    loop.stage_layer.requires_grad_(False)
    before, hooks = target_digest(loop), hook_count(loop)
    head = StreamedVocabularyHead(model.lm_head.weight, loop.device, 13)
    reference = DelayedCapture(loop, head, layer=1, chunk=3)
    shared = SharedPassCapture(loop, head, layer=1, chunk=3)
    ids, donor = [1, 4, 32, 5, 29, 6, 42], [8, 39, 3, 4, 65, 77, 91]
    labels = list(range(16))
    tag = "packedcal97_cpu"
    try:
        v, u = fit_fixture(reference, ids, donor, cfg, np)
        legacy = measure_pair(reference, ids, donor, v, u, labels, cfg, tag, np, check)
        dense, cost = collect(reference, shared, ids, donor, v, u, labels, cfg, tag, packs, np, check)
        audited, extra = collect(reference, shared, ids, donor, v, u, labels, cfg, tag, packs, np, check, audit=True)
        assert legacy == dense == audited
        assert cost == {"full_B2_prefills": 2, "phase1_primes": 1, "packed_phase2_branches": 10,
                        "audited_edited_rows": 0, "qualified_edited_rows": 17, "physical_B2_pass_traversals": 15}
        assert extra == cost | {"full_B2_prefills": 19, "audited_edited_rows": 17, "physical_B2_pass_traversals": 49}
        gs = dense["signed_phase2_edges_per_l2"]
        assert len(gs) == 7 and all(row is not None and len(row) == 8 for row in gs.values())
        assert all(dense["donor_sham_pairs"].values())
        zero = np.zeros_like(v)
        sparse_old = measure_pair(reference, ids, donor, zero, u, labels, cfg, tag, np, check)
        sparse_new, sparse_cost = collect(reference, shared, ids, donor, zero, u, labels, cfg, tag, packs, np, check)
        assert sparse_old == sparse_new and sparse_cost["qualified_edited_rows"] == 14
        assert sparse_cost["packed_phase2_branches"] == 8 and sparse_cost["physical_B2_pass_traversals"] == 13
        assert not any(sparse_new["donor_sham_pairs"].values())
    finally:
        reference.close()
        shared.close()
        head.close()
    assert target_digest(loop) == before and hook_count(loop) == hooks and not torch.cuda.is_initialized()
    return {"dense_edges_exact": len(gs), "legacy_scientific_record_exact": True,
            "packed_audited_record_exact": True, "sparse_schedule_exact": True,
            "dense_eligible_edits": cost["qualified_edited_rows"], "dense_measured_B2_passes": cost["physical_B2_pass_traversals"],
            "dense_parity_audit_B2_passes": extra["physical_B2_pass_traversals"],
            "sparse_eligible_edits": sparse_cost["qualified_edited_rows"], "gpu_initialized": False}


def verify_host(data, cfg):
    class Decoder:
        def decode(self, ids, **unused):
            return chr(65 + ids[0])

    def outcome(answer, boost):
        scores = [0.] * 16
        token = ord(answer) - 65
        scores[token] = boost
        return {"phase2_label_logits": scores, "top_token": [0, token], "phase2_decoded": answer}

    cases, rows = [], []
    for p in data["gold"]:
        correct = "A" if p["arm"] == "focused_erased" else p["answer"]
        rows.append({"id": p["id"], "token_id": ord(correct)-65, "text": correct})
        if p["arm"] == "focused":
            base = {"eligible": True, "requested_l2": 1., "realized_l2": 1., "direction_cosine": 1.}
            arms = {key: base | {"response": outcome(correct, v)}
                    for key, v in (("twin_donor", 2.), ("span_sham", 1.125), ("ambient_sham", .75))}
            cases.append({"source_id": p["id"], "cases": arms, "baseline": outcome(correct, 1.),
                          "donor_sham_pairs": {"span_sham": True, "ambient_sham": True}})
    assert qualify(data, rows)["passed"]
    changed = deepcopy(rows)
    copy_id = next(p["id"] for p in data["gold"] if p["arm"] == "copy")
    row = next(r for r in changed if r["id"] == copy_id)
    row.update(text="invalid")
    assert not qualify(data, changed)["passed"]
    decoder = Decoder()
    for roots in (58, 59, 64):
        summary, worlds = summarize(cases[:2*roots], data, decoder, cfg)
        assert summary["complete_two_sham_roots"] == len(worlds) == roots
        assert summary["complete_root_observed_sham_maximum"] == .25
        assert summary["candidate_sham_bound_95_percentile"] == (.25 if roots >= 59 else None)
    modified = deepcopy(cases[:2])
    modified[0]["donor_sham_pairs"]["span_sham"] = False
    try:
        summarize(modified, data, decoder, cfg)
    except ValueError as exc:
        assert "flags disagree" in str(exc)
    else:
        raise AssertionError("The sham floor accepted an altered dose flag")
    return {"tail_boundary_checked": True, "native_gate_fixture_rows": len(rows),
            "matched_sham_dose_tamper_refused": True}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot", type=Path, required=True)
    cli.add_argument("--source-dir", type=Path, required=True)
    cli.add_argument("--work", type=Path, required=True)
    args = cli.parse_args()
    frozen = verified()
    pins, cfg = frozen_source(), config()
    base = json.loads((REPO / "experiments/nan_readpath37/registration.json").read_bytes())
    if args.snapshot.name != base["revision"] or "torch" in sys.modules:
        raise ValueError("Require the pinned snapshot before importing the fixture runtime")
    with lease(BUDGET) as (check, samples):
        args.work.mkdir(parents=True, exist_ok=False)
        os.environ.update(USE_TORCH="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            HF_HOME=str((args.work / "hf_home").resolve()), HF_MODULES_CACHE=str((args.work / "hf_modules").resolve()))
        os.environ.pop("TRANSFORMERS_CACHE", None)
        for name in ("config.json", "modeling_nanbeige.py"):
            bound(args.snapshot / name, base["snapshot_pins"][name])
        bound(args.snapshot / "configuration_nanbeige.py", CONFIGURATION)
        if {name: version(name) for name in PACKAGES} != PACKAGES:
            raise ValueError("Pinned staged CPU runtime changed")
        import numpy as np
        import torch

        torch.set_num_threads(1)
        proof = compare_native(args.snapshot, cfg, pins["capacity_plan"]["packed_edit_pairs"], check, np, torch)
        data, _ = read_source(args.source_dir, ("public", "gold", "tokens"))
        assert digest(data["public"]) == pins["payload_sha256"]["public"]
        proof |= verify_host(data, cfg)
    receipt = proof | {"status": "passed", "runtime_sha256": frozen, "pretrained_forwards": 0,
                       "sealed_test_accessed": False,
                       "max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
                       "max_sampled_working_set_bytes": max(v["working_set_bytes"] for v in samples)}
    write_new(args.work / "verification.json", receipt)
    print(receipt, flush=True)


if __name__=="__main__":
    main()
