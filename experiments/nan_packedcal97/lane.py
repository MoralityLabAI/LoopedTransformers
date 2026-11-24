"""Guarded single-model native lane for the larger packed CAL condition."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

from experiments.nan_delay39 import pilot
from experiments.nan_gated46.worker import DELAY_QUEUE
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import id_hash, sha

from .source import CPU_CHECK, EXPERIMENT, HERE, REPO, SOURCE_CHECK, config, frozen_source, worker_sources


MANIFEST = HERE / "RUNTIME_MANIFEST.json"
PROOF = HERE / "CPU_VERIFICATION.json"
WORK = Path("E:/codex-artifacts/NanbeigePackedCal97-20261123")


def registered():
    pins = frozen_source()
    old = json.loads(CPU_CHECK.read_bytes())
    if (old["status"] != "passed" or old["source_manifest_sha256"] != pins["source_manifest_sha256"]
            or old["pretrained_forwards"] != 0 or not old["shared_canvas_reused_exactly"]
            or old["packed_edited_rows_exact"] != 34):
        raise ValueError("PASSREUSE96 did not verify the shared capture")
    files = [HERE / name for name in (".gitattributes", "__init__.py", "PROTOCOL.md", "source.py",
                                    "measure.py", "host.py", "lane.py", "verify.py")]
    manifests = [pilot.SEAL, REPO / "experiments/nan_passreuse96/SOURCE_MANIFEST.json",
                 REPO / "experiments/nan_measuredwire93/SOURCE_MANIFEST.json",
                 REPO / "experiments/nan_readout34/RUNTIME_MANIFEST.json"]
    pins_by_path = {}
    for parent in manifests:
        for name, pin in json.loads(parent.read_bytes())["files_sha256"].items():
            if sha(REPO / name) != pin:
                raise ValueError(f"Frozen dependency changed: {name}")
            pins_by_path[name] = pin
    pins_by_path.update({p.relative_to(REPO).as_posix(): sha(p) for p in (
        *files, *manifests, CPU_CHECK, SOURCE_CHECK, REPO / "experiments/nan_queue43/LOCAL_MANIFEST.json")})
    return {"experiment": EXPERIMENT, "files_sha256": pins_by_path, "config": config(),
            "registered_roots": pins["registered_roots"], "max_preflight_position": pins["max_preflight_position"]}


def verified():
    if not __debug__ or Path.cwd() != REPO or json.loads(MANIFEST.read_bytes()) != registered():
        raise RuntimeError("Run the frozen packed CAL condition from its canonical checkout")
    runtime.verify_manifest()
    return sha(MANIFEST)


def commissioned(frozen):
    receipt = json.loads(PROOF.read_bytes())
    if (receipt["status"] != "passed" or receipt["runtime_sha256"] != frozen
            or receipt["pretrained_forwards"] != 0 or receipt["dense_edges_exact"] != 7
            or not all(receipt[key] is True for key in ("legacy_scientific_record_exact",
                    "packed_audited_record_exact", "sparse_schedule_exact", "tail_boundary_checked"))):
        raise RuntimeError("Packed CAL runtime lacks the end-to-end CPU receipt")


def admit(args):
    args.snapshot = args.snapshot.resolve()
    pilot.admissions(args)
    path = DELAY_QUEUE / "terminal.json"
    if not path.is_file():
        raise RuntimeError("The earlier registered DELAY39 sequence remains pending")
    terminal = json.loads(path.read_bytes())
    if (terminal["status"] != "completed" or terminal["local_manifest_sha256"] !=
            sha(REPO / "experiments/nan_queue43/LOCAL_MANIFEST.json")):
        raise RuntimeError("The registered DELAY39 terminal did not close its queue")
    if (WORK / args.backend).exists():
        raise FileExistsError("The registered packed CAL attempt is already occupied")


def publish_identity():
    commit = pilot.verify_publication()
    for file in (MANIFEST, PROOF):
        remote = subprocess.check_output(["git", "show", f"{commit}:{file.relative_to(REPO).as_posix()}"],
                                         timeout=25, creationflags=runtime.FLAGS)
        if remote != file.read_bytes():
            raise RuntimeError("Matching packed CAL runtime and receipt must be present in publication")
    return commit


def host(args, work, *, qualify_only=False):
    env = os.environ.copy()
    env.update(USE_TORCH="0", USE_TF="0", USE_FLAX="0", CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, "-B", "-X", "utf8", "-m", "experiments.nan_packedcal97.host", "--snapshot",
           str(args.snapshot), "--source-dir", str(args.source_dir), "--work", str(work)]
    if qualify_only:
        cmd.append("--qualification")
    subprocess.run(cmd, env=env, timeout=205, check=True, creationflags=runtime.FLAGS)


def run_groups(reference, shared, roster, labels, tokenizer, cfg, packs, v, u, np, work, args, guard):
    from .measure import collect

    roots = sorted({key[0] for key in roster})
    native = []
    started = monotonic()
    for root in roots:
        for arm in cfg["arms"]:
            guard()
            rows = [roster[root, arm, w] for w in ("A", "B")]
            matching = reference.capture([tokens for _, tokens in rows])
            reference.cache.reset()
            choices = matching["logits"][1].argmax(dim=-1).tolist()
            native.extend({"id": name, "token_id": token, "text": pilot.literal(tokenizer, token)}
                          for (name, _), token in zip(rows, choices, strict=True))
    native_path = work / "native.json"
    write_new(native_path, native)
    host(args, work, qualify_only=True)
    gate_path = work / "qualification.json"
    gate = json.loads(gate_path.read_bytes())
    if (gate["experiment"] != EXPERIMENT or gate["native_record_digest"] != digest(native)
            or gate["records_checked"] != len(native) or type(gate["passed"]) is not bool):
        raise ValueError("Independent qualification did not bind the native roster")
    results = {"native_sha256": sha(native_path), "qualification_sha256": sha(gate_path),
               "baseline_and_host_gate_wall_seconds": monotonic() - started}
    if not gate["passed"]:
        return results | {"status": "native_gate_failed", "closed_focused_worlds": 0}
    journal = work / "closed.jsonl"
    count = 0
    started = monotonic()
    with journal.open("x", encoding="utf-8", newline="\n") as stream:
        for i, root in enumerate(roots):
            for world, other in (("A", "B"), ("B", "A")):
                identity, ids = roster[root, "focused", world]
                foil = roster[root, "focused", other][1]
                measured, usage = collect(reference, shared, ids, foil, v, u, labels, cfg, identity, packs,
                                          np, guard, audit=i % cfg["parity_audit_root_stride"] == 0)
                for response in [measured["baseline"], *(c["response"] for c in measured["cases"].values()
                                                           if "response" in c)]:
                    response["phase2_decoded"] = pilot.literal(tokenizer, response["top_token"][1])
                file = work / f"episode_{count:04d}.json"
                write_new(file, {"source_id": identity, "prompt_sha256": id_hash(ids), "work": usage} | measured)
                stream.write(json.dumps({"file": file.name, "sha256": sha(file)}) + "\n")
                stream.flush()
                count += 1
    return results | {"status": "closed", "closed_focused_worlds": count, "closed_journal_sha256": sha(journal),
                      "focused_measurement_wall_seconds": monotonic() - started}


def execute(args, work, frozen, commit):
    if {name: version(name) for name in runtime.GPU_PACKAGES} != runtime.GPU_PACKAGES:
        raise RuntimeError("Registered GPU package versions differ")
    with pilot.resources(work) as check:
        cfg, pins = config(), frozen_source()
        roster, labels, tok = worker_sources(args.source_dir, args.snapshot, check)
        packs = pins["capacity_plan"]["packed_edit_pairs"]
        import numpy as np
        import torch
        from experiments.nan_bf31.loader import verify_checkpoint
        from experiments.nan_cache32.worker import quant_state_digest
        from experiments.nan_delay39.instrument import DelayedCapture
        from experiments.nan_passreuse96.capture import SharedPassCapture
        from experiments.nan_recomp09.common import target_digest
        from experiments.nan_stream36.batch_engine import BatchedDepthEngine
        from experiments.nan_stream36.device_preflight import preflight
        from mechinterp.nanbeige_bpd_gym.lazy_bf16_loop import digest_dispatch
        from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead
        from .measure import collect

        torch.set_num_threads(1)
        card = torch.cuda.get_device_properties(0)
        if "3050" not in card.name or not 3.5 * 2**30 <= card.total_memory <= 4.5 * 2**30:
            raise RuntimeError("Expected the registered 4 GiB RTX 3050")
        torch.cuda.set_per_process_memory_fraction(.9)
        torch.cuda.reset_peak_memory_stats()
        with digest_dispatch():
            loop, checkpoint, stage = runtime.load_model(args.backend, args.snapshot, args.adapter)
            before, q_before = target_digest(loop), quant_state_digest(loop)
            if before != runtime.TARGETS[args.backend] or loop.n != 22:
                raise ValueError("The quantized decoder does not match its frozen native target")
            check(device=True, force_gpu_sample=True)
            v, u = pilot.get_bank(args.backend, np)
            head = StreamedVocabularyHead(loop.model.lm_head.weight, loop.device, 8192)
            root = max(sorted({key[0] for key in roster}), key=lambda r: len(roster[r, "focused", "A"][1]))
            a, b = (roster[root, "focused", w][1] for w in ("A", "B"))

            def guard():
                check(device=True)

            def slot(start, batch, position):
                if batch not in (1, 2) or position > pins["max_preflight_position"] or (
                        monotonic() - start > runtime.LIMITS["maximum_slot_seconds"]):
                    raise RuntimeError("Packed CAL preflight exceeded the registered shape or duration")
                guard()

            engine = BatchedDepthEngine(loop, head)
            try:
                proof = preflight(engine, [a, b], pins["max_preflight_position"], slot)
            finally:
                engine.close()
            reference = DelayedCapture(loop, head, layer=cfg["layer_zero_based"])
            shared = SharedPassCapture(loop, head, layer=cfg["layer_zero_based"])
            try:
                native_packed = [collect(reference, shared, x, y, v, u, labels, cfg, f"preflight:{i}",
                                         packs, np, guard, audit=True)[1] for i, (x, y) in enumerate(((a, b), (b, a)))]
                parity = sum(v["audited_edited_rows"] for v in native_packed)
                if parity < 28:
                    raise RuntimeError("Packed preflight did not cover the registered minimum edited rows")
                write_new(work / "preflight.json", {"status": "passed", "depths": proof, "stage": stage,
                          "packed_parity_worlds": len(native_packed), "packed_audited_edits": parity})
                result = run_groups(reference, shared, roster, labels, tok, cfg, packs, v, u, np, work, args, guard)
            finally:
                shared.close()
                reference.close()
                head.close()
            check(device=True, force_gpu_sample=True)
            after, q_after = target_digest(loop), quant_state_digest(loop)
            verify_checkpoint(args.snapshot, checkpoint)
            if after != before or q_after != q_before:
                raise RuntimeError("Packed CAL changed the frozen model or quantization bytes")
            write_new(work / "terminal.json", result | {"experiment": EXPERIMENT, "backend": args.backend,
                      "runtime_sha256": frozen, "published_commit": commit, "target_before": before, "target_after": after,
                      "quant_unchanged": True, "checkpoint_verified_after": True,
                      "admission_sha256": sha(work / "admission.json"), "preflight_sha256": sha(work / "preflight.json"),
                      "CUDA_peak": torch.cuda.max_memory_allocated(), "CUDA_reserved_peak": torch.cuda.max_memory_reserved()})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--check-admission", action="store_true")
    cli.add_argument("--backend", choices=("nf4", "bf16"))
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--source-dir", type=Path)
    cli.add_argument("--adapter", type=Path)
    args = cli.parse_args()
    if args.freeze:
        write_new(MANIFEST, registered())
        return
    if not all((args.backend, args.snapshot, args.source_dir, args.adapter)):
        cli.error("Require a backend, pinned snapshot, source directory and adapter")
    frozen = verified()
    commissioned(frozen)
    if "torch" in sys.modules:
        raise RuntimeError("Packed worker imported Torch before cold admission")
    if args.check_admission:
        admit(args)
        print("Local predecessor and resource conditions passed", flush=True)
        return
    with runtime.exclusive_lane():
        admit(args)
        commit = publish_identity()
        work = WORK / args.backend
        work.mkdir(parents=True, exist_ok=False)
        write_new(work / "admission.json", {"experiment": EXPERIMENT, "backend": args.backend,
                  "runtime_sha256": frozen, "published_commit": commit, "pid": os.getpid()})
        runtime.environment(work)
        try:
            execute(args, work, frozen, commit)
            host(args, work)
        except BaseException as exc:
            write_new(work / "attempt_failure.json", {"error_type": type(exc).__name__, "message": str(exc)})
            raise


if __name__ == "__main__":
    main()
