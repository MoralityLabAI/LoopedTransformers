"""Run VOCAB82 on the mutually exclusive, guarded WRITER77 BF16 lane."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

from experiments.nan_attnspan67.run import read_banks
from experiments.nan_cigate73.run import ATTN, BASE, controller
from experiments.nan_delay39 import pilot as delay
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_vocab82.heads import partition_sha
from experiments.nan_vocab82.verify import MANIFEST as OBSERVER_MANIFEST, seal as observer_seal
from experiments.nan_writer77 import lane as writer
from experiments.nan_writer77.source import CHECK, worker_inputs

from .batches import execute_pairs
from .custody import (
    COMMISSIONED, EXPERIMENT, HERE, HOST_BUDGET, MANIFEST, OBSERVER, REPO, config, vocabulary,
)


WORK = writer.ARTIFACT / "bf16"


def registration():
    previous = json.loads(writer.MANIFEST.read_bytes())
    observer = json.loads(OBSERVER_MANIFEST.read_bytes())
    receipt = json.loads(OBSERVER.read_bytes())
    source = json.loads(CHECK.read_bytes())
    if (previous != writer.registration() or observer != observer_seal()
            or receipt["status"] != "passed" or receipt["experiment"] != "NAN-VOCAB-82"
            or receipt["full_width"]["source_manifest_sha256"] != sha(OBSERVER_MANIFEST)
            or receipt["host_and_native"]["source_manifest_sha256"] != sha(OBSERVER_MANIFEST)
            or receipt["host_and_native"]["registered_worlds"] != source["prompts"]
            or source["prompts"] % 2):
        raise ValueError("Frozen WRITER77 schedule or VOCAB82 verification changed")
    paths = [HERE / name for name in ("__init__.py", ".gitattributes", "PROTOCOL.md", "lane.py",
                                      "batches.py", "custody.py", "verify.py", "native_check.py")]
    paths += [OBSERVER_MANIFEST, OBSERVER, writer.MANIFEST]
    paths += [REPO / n for n in ("experiments/nan_bf31/loader.py", "experiments/nan_cache32/worker.py",
                                 "experiments/nan_recomp09/common.py",
                                 "experiments/nan_storyaudit30/common.py",
                                 "experiments/nan_readpath37/run.py",
                                 "mechinterp/nanbeige_bpd_gym/lazy_bf16_loop.py")]
    return {"experiment": EXPERIMENT, "backend": "bf16", "stage": "guarded_gpu",
            "b2_baseline_batches": previous["b2_baseline_batches"], "max_position": previous["max_position"],
            "registered_roots": source["roots"], "host_budget": HOST_BUDGET,
            "partition_sha256": receipt["full_width"]["partition_sha256"],
            "files_sha256": {p.relative_to(REPO).as_posix(): sha(p) for p in paths}}


def verified():
    if not __debug__ or Path.cwd() != REPO or json.loads(MANIFEST.read_bytes()) != registration():
        raise ValueError("Use the frozen VOCAB83 runtime from the canonical checkout")
    runtime.verify_manifest()
    return sha(MANIFEST)


def commissioning(frozen):
    receipt = json.loads(COMMISSIONED.read_bytes())
    if (receipt["runtime_sha256"] != frozen or receipt["status"] != "passed"
            or receipt["roots_written"] != registration()["registered_roots"]
            or receipt["native_gate_refused_roots"] != registration()["registered_roots"]
            or receipt["pretrained_forwards"] != 0
            or receipt["native_cpu"]["runtime_sha256"] != frozen
            or receipt["native_cpu"]["both_native_heads_exact"] is not True
            or receipt["native_cpu"]["edited_worlds"] != 2
            or receipt["native_cpu"]["head_calls_baseline_and_trial"] != 30
            or not all(receipt[k] is True for k in ("host_positive_replay", "worker_input_without_gold",
                                                   "journal_tamper_refused", "baseline_tamper_refused"))):
        raise RuntimeError("Runtime custody commissioning does not bind this lane")


def publication():
    commit = delay.verify_publication()
    for path in (MANIFEST, COMMISSIONED):
        name = path.relative_to(REPO).as_posix()
        data = subprocess.check_output(["git", "show", f"{commit}:{name}"], timeout=30,
                                       creationflags=runtime.FLAGS)
        if data != path.read_bytes():
            raise RuntimeError("Published VOCAB83 runtime or custody receipt differs")
    return commit


def host(args, work, *, qualify_only=False, fixture=False):
    env = os.environ.copy()
    env.update(USE_TORCH="0", USE_TF="0", USE_FLAX="0", CUDA_VISIBLE_DEVICES="")
    command = [sys.executable, "-B", "-X", "utf8", "-m", "experiments.nan_vocab83.custody",
               "--source-dir", str(args.source_dir), "--snapshot", str(args.snapshot), "--work", str(work)]
    if qualify_only:
        command.append("--qualification")
    if fixture:
        command.append("--fixture")
    subprocess.run(command, env=env, timeout=135, check=True, creationflags=runtime.FLAGS)


def terminal(work, *, frozen, commit, classes, batches, roots, before, after, quant, quant_after, peak,
             mode="pretrained_cuda"):
    if before != runtime.TARGETS["bf16"] or before != after or quant != quant_after:
        raise ValueError("Writer target or quantization differs from the admitted checkpoint")
    hashes = {f"{key}_sha256": sha(work / name) for key, name in (
        ("admission", "admission.json"), ("baseline", "baseline.json"),
        ("qualification", "qualification.json"), ("journal", "closed.jsonl"), ("preflight", "preflight.json"))}
    write_new(work / "terminal.json", hashes | {"experiment": EXPERIMENT, "backend": "bf16",
              "status": "completed", "mode": mode, "runtime_sha256": frozen, "published_commit": commit,
              "checkpoint_verified_after": True, "target_before": before, "target_after": after,
              "quant_before": quant, "quant_after": quant_after, "vocab": len(classes),
              "partition_sha256": partition_sha(classes), "peak_torch_bytes": peak,
              "b2_baseline_batches": batches, "root_records": roots})


def execute(args, work, published, frozen):
    if {p: version(p) for p in runtime.GPU_PACKAGES} != runtime.GPU_PACKAGES:
        raise RuntimeError("GPU interpreter package versions differ")
    with delay.resources(work) as guard:
        payload = worker_inputs(args.source_dir, args.snapshot, guard)
        _, classes = vocabulary(args.snapshot, guard, json.loads(CHECK.read_bytes())["tokenizer_pins"])
        import numpy as np
        import torch
        from experiments.nan_bf31.loader import verify_checkpoint
        from experiments.nan_cache32.worker import quant_state_digest
        from experiments.nan_fit07.pointwise import masks
        from experiments.nan_radial76.native import WriterReaderCapture
        from experiments.nan_recomp09.common import target_digest
        from experiments.nan_relabel68.pipeline import paired_heads
        from experiments.nan_stream36.batch_engine import BatchedDepthEngine
        from experiments.nan_stream36.device_preflight import preflight
        from mechinterp.nanbeige_bpd_gym.lazy_bf16_loop import digest_dispatch
        from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

        torch.set_num_threads(1)
        device = torch.cuda.get_device_properties(0)
        if "3050" not in device.name or not 3.5*2**30 <= device.total_memory <= 4.5*2**30:
            raise RuntimeError("Expected the registered 4 GiB RTX 3050 CUDA device")
        torch.cuda.set_per_process_memory_fraction(.9)
        torch.cuda.reset_peak_memory_stats()
        settings = registration()
        attn, base = [json.loads(p.read_bytes()) for p in (ATTN, BASE)]
        bank = read_banks("bf16", 21, attn, base, np)["attn_bpd_t2"]
        fit = controller("t2_layer21_o_proj_rank8", attn["ranksite_audits"]["2"], bank, np, masks)

        def check():
            guard(device=True)

        def check_slot(start, rows, position):
            if (rows not in (1, 2) or not 1 <= position <= settings["max_position"]
                    or monotonic() - start > runtime.LIMITS["maximum_slot_seconds"]):
                raise RuntimeError("Native preflight exceeded its registered shape or budget")
            check()

        def replay(projection, value, output):
            if not torch.equal(torch.nn.functional.linear(value, projection.weight, projection.bias), output):
                raise RuntimeError("Reader projection failed same-kernel native replay")

        with digest_dispatch():
            loop, checkpoint, placement = runtime.load_model("bf16", args.snapshot, args.adapter)
            before, quant = target_digest(loop), quant_state_digest(loop)
            if before != runtime.TARGETS["bf16"] or loop.n != 22 or loop.model.lm_head.weight.shape[0] != len(classes):
                raise ValueError("The native checkpoint target, physical layers or head width differ")
            guard(device=True, force_gpu_sample=True)
            head = StreamedVocabularyHead(loop.model.lm_head.weight, loop.device, 8192)
            try:
                focus = [row for row in payload["public"] if row["arm"] == "focused"]
                root = focus[0]["root"]
                bindings = payload["tokens"]["row_bindings"]
                pair = [bindings[next(row["id"] for row in focus
                                      if row["root"] == root and row["world"] == w)]["ids"] for w in ("A", "B")]
                engine = BatchedDepthEngine(loop, head)
                try:
                    checks = preflight(engine, pair, settings["max_position"], check_slot)
                    expected = paired_heads(engine, pair, check_slot)
                finally:
                    engine.close()
                capture = WriterReaderCapture(loop, head, replay=replay)
                try:
                    actual = capture.capture(pair)["logits"]
                    if not torch.equal(actual, torch.stack(expected)):
                        raise RuntimeError("The instrumented clean B2 head differs from native prefill")
                    write_new(work / "preflight.json", {"depths": checks, "stage": placement,
                               "both_heads_exact": True, "partition_sha256": partition_sha(classes)})
                    batches, roots = execute_pairs(work, capture, payload, config(), fit, masks, np,
                                                  classes, check, lambda: host(args, work, qualify_only=True))
                finally:
                    capture.close()
            finally:
                head.close()
            guard(device=True, force_gpu_sample=True)
            verify_checkpoint(args.snapshot, checkpoint)
            after, quant_after = target_digest(loop), quant_state_digest(loop)
            if (batches, roots) != (settings["b2_baseline_batches"], settings["registered_roots"]):
                raise RuntimeError("Closed baseline or root count differs from the frozen schedule")
            terminal(work, frozen=frozen, commit=published, classes=classes, batches=batches, roots=roots,
                     before=before, after=after, quant=quant, quant_after=quant_after,
                     peak=torch.cuda.max_memory_allocated())


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--check-admission", action="store_true")
    cli.add_argument("--source-dir", type=Path)
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--adapter", type=Path)
    args = cli.parse_args()
    if args.freeze:
        write_new(MANIFEST, registration())
        return
    if not all((args.source_dir, args.snapshot, args.adapter)):
        cli.error("Require the frozen source, approved adapter and pinned checkpoint")
    frozen = verified()
    commissioning(frozen)
    if "torch" in sys.modules:
        cli.error("Torch was loaded before native admission")
    if args.check_admission:
        writer.admission(args)
        if WORK.exists():
            raise FileExistsError("The mutually exclusive WRITER77 attempt path is occupied")
        print("Local WRITER77 prerequisites passed", flush=True)
        return
    with runtime.exclusive_lane():
        writer.admission(args)
        published = publication()
        WORK.mkdir(parents=True, exist_ok=False)
        write_new(WORK / "admission.json", {"pid": os.getpid(), "runtime_sha256": frozen,
                  "published_commit": published, "mode": "pretrained_cuda"})
        runtime.environment(WORK)
        try:
            execute(args, WORK, published, frozen)
            host(args, WORK)
        except BaseException as exc:
            write_new(WORK / "attempt_failure.json", {"type": type(exc).__name__, "message": str(exc)})
            raise


if __name__ == "__main__":
    main()
