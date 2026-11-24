"""Verify grouped packed branches against separate native CPU captures and file-backed stages."""

import argparse
from collections import Counter
from dataclasses import replace
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

from experiments.nan_bf31.stage_audit import audit as audit_stages
from experiments.nan_delay39.contrasts import numpy_state, response
from experiments.nan_delay39.instrument import DelayedCapture
from experiments.nan_packedcal97 import lane
from experiments.nan_packedcal97.measure import collect
from experiments.nan_packedcal97.source import config, frozen_source
from experiments.nan_packedcal97.verify import fit_fixture
from experiments.nan_passreuse96.capture import SharedPassCapture
from experiments.nan_passreuse96.verify import patches, rows_equal
from experiments.nan_phase35.run import lease
from experiments.nan_premlp64.verify import BUDGET, CONFIGURATION, PACKAGES, hook_count
from experiments.nan_readpath37.run import bound, digest, write_new
from mechinterp.nanbeige_bpd_gym.lazy_bf16_storage import CheckpointStore, logical_target_digest

from .group import GroupedPassCapture, MAX_BRANCHES, Request


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_MANIFEST.json"
EXPERIMENT = "NAN-STAGEGROUP-103"


def registered():
    frozen = lane.verified()
    lane.commissioned(frozen)
    local = [HERE / p for p in (".gitattributes", "__init__.py", "PROTOCOL.md", "group.py", "verify.py")]
    parents = [lane.MANIFEST, lane.PROOF]
    sources = ("experiments/nan_bf31/stage_audit.py", "experiments/nan_bf31/config.json",
               "mechinterp/nanbeige_bpd_gym/lazy_bf16_loop.py", "mechinterp/nanbeige_bpd_gym/lazy_bf16_storage.py")
    paths = local + parents + [lane.REPO / p for p in sources]
    return {"experiment": EXPERIMENT, "cpu_budget": BUDGET, "maximum_simultaneous_B2_branches": MAX_BRANCHES,
            "files_sha256": {p.relative_to(lane.REPO).as_posix(): digest(p) for p in paths}}


class ObservedStore(CheckpointStore):
    def __init__(self, folder):
        super().__init__(folder)
        self.bytes_copied = 0
        self.physical_stages = []

    def copy_parameters(self, stage, prefix):
        verdict = super().copy_parameters(stage, prefix)
        self.bytes_copied += verdict["copied_bytes"]
        self.physical_stages.append(int(prefix.split(".")[-1]))
        return verdict


class TracePackedCapture(SharedPassCapture):
    def __init__(self, loop, head, layer):
        super().__init__(loop, head, layer=layer, chunk=3)
        self.trace = []
        self.serial_copies = 0
        self.serial_bytes = 0

    def branch(self, prompts, patches=(None, None)):
        loop = self.cache.loop
        bytes_before, copies_before = loop.checkpoint_store.bytes_copied, loop.stage_copies
        value = super().branch(prompts, patches)
        if any(p is not None for p in patches):
            self.serial_copies += loop.stage_copies - copies_before
            self.serial_bytes += loop.checkpoint_store.bytes_copied - bytes_before
            self.trace.append((patches, value))
        return value


def prepare_fixture(snapshot, work, torch):
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import build_nanbeige_model_and_loop
    from mechinterp.nanbeige_bpd_gym.lazy_bf16_loop import make_lazy_loop
    from safetensors.torch import save_file

    torch.manual_seed(97031)
    model, original = build_nanbeige_model_and_loop(snapshot, exec_device=torch.device("cpu"), dtype=torch.bfloat16,
        n_gpu_layers=1, random_init=True, config_overrides={"vocab_size": 97, "num_hidden_layers": 4,
            "hidden_size": 128, "intermediate_size": 72, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "kv_channels": 16})
    model.requires_grad_(False)
    stage = original.stage_layer.requires_grad_(False)
    folder = work / "synthetic_weights"
    folder.mkdir()
    shard = "model.safetensors"
    save_file({k: p.detach().contiguous().clone() for k, p in model.named_parameters()}, str(folder / shard))
    write_new(folder / "model.safetensors.index.json", {"weight_map": {k: shard for k, _ in model.named_parameters()}})
    store = ObservedStore(folder)
    loop = make_lazy_loop(model, store, torch.device("cpu"), 1, stage)
    return model, loop, store


def check_equal(actual, expected, torch):
    for row in (0, 1):
        rows_equal(actual, row, expected, row, torch)
    assert actual["applied"] == expected["applied"]


def refuse(call, match):
    try:
        call()
    except (ValueError, RuntimeError) as exc:
        assert match in str(exc)
    else:
        raise AssertionError("The grouped engine accepted a contradictory request")


def exercise(snapshot, work, cfg, packs, check, np, torch):
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    model, loop, store = prepare_fixture(snapshot, work, torch)
    layer = loop.n - 1
    params, hooks = logical_target_digest(model, store), hook_count(loop)
    stage_proof = audit_stages(loop, check)
    stage_bytes = sum(row["bytes"] for row in stage_proof["checks"])
    head = StreamedVocabularyHead(model.lm_head.weight, loop.device, 13)
    packed = TracePackedCapture(loop, head, layer)
    group = GroupedPassCapture(loop, head, layer=layer, chunk=3)
    original = DelayedCapture(loop, head, layer=layer, chunk=3)
    ids, peer = [1, 4, 32, 5, 29, 6, 42], [8, 39, 3, 4, 65, 77, 91]
    labels = list(range(16))
    try:
        v, u = fit_fixture(original, ids, peer, cfg, np)
        original.cache.reset()
        scientific, counts = collect(original, packed, ids, peer, v, u, labels, cfg, "packedcal97_cpu", packs, np, check)
        assert counts["qualified_edited_rows"] == 17 and len(packed.trace) == len(packs) == 9
        serial_copies, serial_bytes = packed.serial_copies, packed.serial_bytes
        assert serial_copies == 9 * stage_proof["all_offloaded_layers"] and serial_bytes == 9 * stage_bytes
        assert all(scientific["donor_sham_pairs"].values())
        assert all(s is not None for s in scientific["signed_phase2_edges_per_l2"].values())
        left = (ids, ids)
        first = group.prime(left)
        raw_first = first.raw.clone()
        requests = [Request(first, left, p) for p, _ in packed.trace]
        tiled_before = loop.stage_copies, store.bytes_copied
        answers = list(group.stream(requests, check))
        consumed_copies = loop.stage_copies - tiled_before[0]
        consumed_bytes = store.bytes_copied - tiled_before[1]
        assert consumed_copies == 5*stage_proof["all_offloaded_layers"] and consumed_bytes == 5*stage_bytes
        assert len(answers) == len(requests)
        reconciled = 0
        for names, (_, baseline), value in zip(packs, packed.trace, answers, strict=True):
            check_equal(value, baseline, torch)
            edits = names if len(names) == 2 else (None, names[0])
            for i, name in enumerate(edits):
                if name is None:
                    continue
                studied = dict(scientific["cases"][name]["response"])
                studied.pop("realized_residual_delta")
                assert studied == response(value, v, labels, np, row=i)
                reconciled += 1

        other = peer + [2, 3, 7, 43]
        right = (other, other)
        second = group.prime(right)
        raw_second = second.raw.clone()
        new_edit = patches(numpy_state(second.residual[1, 1]), cfg, np)[0]
        packed.prime(right)
        right_ref = packed.branch(right, (None, new_edit))
        before = loop.stage_copies
        distinct = [requests[0], Request(second, right, (None, new_edit))]
        left_state, right_state = group.run(distinct, check)
        assert loop.stage_copies - before == stage_proof["all_offloaded_layers"]
        check_equal(left_state, packed.trace[0][1], torch)
        check_equal(right_state, right_ref, torch)
        right_legacy = original.capture(right, new_edit)
        original.cache.reset()
        check_equal(right_state, right_legacy, torch)

        before = loop.stage_copies
        refuse(lambda: group.run([distinct[0], replace(distinct[1], prompts=left)], check), "exact primed")
        stale = replace(new_edit, before=new_edit.before + 1.)
        refuse(lambda: group.run([distinct[0], replace(distinct[1], patches=(None, stale))], check), "stale")
        refuse(lambda: group.run([*distinct, distinct[0]], check), "one or two")
        assert before == loop.stage_copies
        checkpoints = 0

        def interrupt():
            nonlocal checkpoints
            checkpoints += 1
            if checkpoints == 5:
                raise RuntimeError("intentional guard stop")

        mode = loop.phase, loop.active_layer
        refuse(lambda: group.run(distinct, interrupt), "intentional guard")
        assert loop.stage_copies == before + 1 and not group.cache.instrument.enabled
        assert mode == (loop.phase, loop.active_layer)
        for actual, expected in zip(group.run(list(reversed(distinct)), check),
                                    (right_ref, packed.trace[0][1]), strict=True):
            check_equal(actual, expected, torch)
        assert torch.equal(raw_first, first.raw) and torch.equal(raw_second, second.raw)
        assert not group.cache.caches
    finally:
        group.close()
        packed.close()
        original.close()
        head.close()
    assert logical_target_digest(model, store) == params and hook_count(loop) == hooks and not torch.cuda.is_initialized()
    return {"staged_cpu_layers_verified": stage_proof["all_offloaded_layers"], "packed_CPU_B2_branches_exact": 9,
            "packed_CPU_edit_responses_exact": reconciled, "independent_different_prompt_B2_branches_exact": 2,
            "independent_file_backed_stage_bytes_per_phase": stage_bytes,
            "serial_copies_for_same_nine_branches": serial_copies,
            "grouped_copies_for_same_nine_branches": consumed_copies,
            "serial_requested_parameter_bytes_for_nine_branches": serial_bytes,
            "grouped_requested_parameter_bytes_for_nine_branches": consumed_bytes,
            "stale_tokens_and_edits_refused_before_loading": True, "capacity_bound_and_guard_recovery_verified": True,
            "both_primed_canvases_preserved": True, "cuda_initialized": False, "pretrained_forwards": 0}


def native_projection(snapshot, pins, cfg):
    metadata = CheckpointStore(snapshot)
    total = Counter()
    for name, tensor in metadata.spans.items():
        if name.startswith("model.layers."):
            total[int(name.split(".")[2])] += tensor.length
    resident = json.loads((lane.REPO / "experiments/nan_bf31/config.json").read_bytes())["BF16_resident_decoder_layers"]
    if set(total) != set(range(len(total))):
        raise ValueError("The physical layer byte roster differs")
    stage = sum(bytes_ for i, bytes_ in total.items() if i >= resident)
    plan = pins["capacity_plan"]
    branches = len(plan["packed_edit_pairs"])
    per_dense = branches - (branches + MAX_BRANCHES - 1)//MAX_BRANCHES
    worlds = plan["focused_worlds"]
    original = (plan["total_B2_pass_traversals"]["shared_packed"]
                + (worlds//cfg["parity_audit_root_stride"]) * 2*(2*len(cfg["phase1_components"]) + 3))
    net = original - per_dense*worlds
    return {"maximum_prefill_position": pins["max_preflight_position"], "offloaded_layers": len(total)-resident,
            "one_phase_staged_parameter_bytes": stage, "physical_B2_pass_traversals_both_schedules": original,
            "serial_stage_weight_sweeps": original, "grouped_stage_weight_sweeps": net,
            "requested_stage_parameter_bytes_saved": (original-net)*stage}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--work", type=Path)
    args = cli.parse_args()
    seal = registered()
    if args.freeze:
        write_new(MANIFEST, seal)
        return
    base = json.loads((lane.REPO / "experiments/nan_readpath37/registration.json").read_bytes())
    if (not args.work or not args.snapshot or args.snapshot.name != base["revision"]
            or json.loads(MANIFEST.read_bytes()) != seal or "torch" in sys.modules):
        cli.error("Require the frozen CPU fixture, pinned snapshot and a fresh output folder")
    with lease(BUDGET) as (check, samples):
        args.work.mkdir(parents=True, exist_ok=False)
        os.environ.update(USE_TORCH="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_HOME=str((args.work / "hf_home").resolve()),
                          HF_MODULES_CACHE=str((args.work / "hf_modules").resolve()))
        os.environ.pop("TRANSFORMERS_CACHE", None)
        for name in ("config.json", "modeling_nanbeige.py", "model.safetensors.index.json"):
            bound(args.snapshot / name, base["snapshot_pins"][name])
        bound(args.snapshot / "configuration_nanbeige.py", CONFIGURATION)
        if {name: version(name) for name in PACKAGES} != PACKAGES:
            raise ValueError("The pinned CPU package set changed")
        import numpy as np
        import torch

        torch.set_num_threads(1)
        pins, cfg = frozen_source(), config()
        proof = exercise(args.snapshot, args.work, cfg, pins["capacity_plan"]["packed_edit_pairs"], check, np, torch)
        projection = native_projection(args.snapshot, pins, cfg)
    receipt = proof | {"experiment": EXPERIMENT, "status": "passed", "source_manifest_sha256": digest(MANIFEST),
                       "declared_preflight_BF16_projection": projection, "sealed_test_accessed": False,
                       "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
                       "max_sampled_working_set_bytes": max(s["working_set_bytes"] for s in samples),
                       "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples)}
    write_new(args.work / "verification.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
