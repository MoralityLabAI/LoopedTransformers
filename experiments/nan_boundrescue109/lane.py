"""Guard the pinned NF4 and staged BF16 boundary-rescue runs on the RTX 3050."""

import argparse
from contextlib import ExitStack
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

from experiments.nan_delay39 import pilot as delay
from experiments.nan_gated46.worker import DELAY_QUEUE
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_readpath37.run import FixedHead, SHARD, write_new
from experiments.nan_vocab82.heads import partition_sha
from experiments.nan_vocab83.custody import vocabulary

from . import roster, verify, verify_plan
from .core import REPO, certified_planner
from .host import MODE


HERE = roster.HERE
MANIFEST = HERE/"RUNTIME_MANIFEST.json"
COMM = HERE/"RUNTIME_VERIFICATION.json"
ARTIFACT = Path("E:/codex-artifacts/NanbeigeBoundRescue109-20261123")


def registration():
    files = [HERE / name for name in (
        "lane.py", "host.py", "worker.py", "verify_runtime.py", "RUNTIME.md", "CPU_VERIFICATION.json",
        "SOURCE_MANIFEST.json", "PLAN_MANIFEST.json", "PLAN_VERIFICATION.json", "ROSTER_VERIFICATION.json",
        "ROSTER_MANIFEST.json")]
    files += [REPO / name for name in (
        "experiments/nan_readout34/RUNTIME_MANIFEST.json", "experiments/nan_queue43/LOCAL_MANIFEST.json",
        "experiments/nan_vocab83/batches.py", "experiments/nan_vocab83/custody.py",
        "experiments/nan_writer77/host.py", "experiments/nan_writer77/pipeline.py",
        "experiments/nan_vocab82/heads.py", "experiments/nan_stream36/device_preflight.py",
        "experiments/nan_stream36/batch_engine.py", "experiments/nan_delay39/pilot.py",
        "experiments/nan_gated46/worker.py", "experiments/nan_cache32/worker.py",
        "mechinterp/nanbeige_bpd_gym/streamed_vocab_head.py")]
    r = json.loads(roster.RECEIPT.read_bytes())
    cpu = json.loads((HERE/"CPU_VERIFICATION.json").read_bytes())
    plan = json.loads((HERE/"PLAN_VERIFICATION.json").read_bytes())
    if (roster.registration() != json.loads(roster.MANIFEST.read_bytes())
            or verify.registration() != json.loads(verify.MANIFEST.read_bytes())
            or verify_plan.registration() != json.loads(verify_plan.MANIFEST.read_bytes())
            or r["manifest_sha256"] != sha(roster.MANIFEST) or r["pretrained_forwards"] != 0
            or cpu["source_manifest_sha256"] != sha(verify.MANIFEST) or cpu["status"] != "passed"
            or plan["manifest_sha256"] != sha(verify_plan.MANIFEST) or plan["status"] != "passed"):
        raise ValueError("The source, instrumentation or planner verification changed")
    return {"experiment": roster.EXPERIMENT, "roots": r["roots"], "baseline_batches": r["b2_batches"],
            "max_position": r["max_position"], "ordered_backends": ["nf4", "bf16"],
            "native_compute_limits": runtime.LIMITS, "host_qualification": {"clean":12,"copy":16},
            "files_sha256": {p.relative_to(REPO).as_posix():sha(p) for p in files}}


def verified():
    if not __debug__ or Path.cwd() != REPO or json.loads(MANIFEST.read_bytes()) != registration():
        raise ValueError("Use the frozen runtime from the canonical checkout")
    runtime.verify_manifest()
    return sha(MANIFEST)


def commissioned(frozen):
    receipt = json.loads(COMM.read_bytes())
    flags = ("gate_failed_zero_edits", "native_root_unqualified_zero_edits", "oracle_bound_without_worker_gold",
             "archived_head_audited", "tamper_refused", "synthetic_refused_as_pretrained")
    if (receipt["runtime_sha256"] != frozen or receipt["status"] != "passed"
            or receipt["pretrained_forwards"] != 0 or receipt["baseline_batches"] != 48
            or not all(receipt[k] is True for k in flags)):
        raise ValueError("The closed runtime fixture does not verify the local freeze")


def admit(args, work):
    args.snapshot = args.snapshot.resolve()
    delay.admissions(args)
    terminal = DELAY_QUEUE/"terminal.json"
    if not terminal.is_file():
        raise RuntimeError("The previously registered DELAY39 sequence remains pending")
    prior = json.loads(terminal.read_bytes())
    if prior["status"] != "completed" or prior["local_manifest_sha256"] != sha(
            REPO/"experiments/nan_queue43/LOCAL_MANIFEST.json"):
        raise ValueError("The preceding DELAY39 terminal differs from its registration")
    if work.exists():
        raise FileExistsError("The registered backend attempt is already occupied")
    if args.backend == "bf16":
        nf4 = ARTIFACT/"nf4"
        report = json.loads((nf4/"aggregate.json").read_bytes())
        if (report["experiment"],report["backend"],report["mode"],report["runtime_sha256"],report["terminal_sha256"]) != (
                roster.EXPERIMENT,"nf4",MODE,sha(MANIFEST),sha(nf4/"terminal.json")):
            raise RuntimeError("The preceding NF4 experiment lacks a closed independent host audit")


def publication():
    commit = delay.verify_publication()
    for p in (MANIFEST,COMM):
        link = p.relative_to(REPO).as_posix()
        if subprocess.check_output(["git","show",f"{commit}:{link}"],timeout=30,
                                   creationflags=runtime.FLAGS) != p.read_bytes():
            raise ValueError("The runtime or host verification differs from branch publication")
    return commit


def run_host(args, work, *, qualify=False):
    env = os.environ.copy()
    env.update(USE_TORCH="0",USE_TF="0",USE_FLAX="0",CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable,"-B","-X","utf8","-m","experiments.nan_boundrescue109.host",
           "--source-dir",str(args.source_dir),"--snapshot",str(args.snapshot),"--work",str(work),
           "--backend",args.backend]
    if qualify:
        cmd.append("--qualification")
    subprocess.run(cmd,env=env,timeout=135,check=True,creationflags=runtime.FLAGS)


def device(args, work, guard, payload, classes, admission, np, torch):
    from experiments.nan_bf31.loader import verify_checkpoint
    from experiments.nan_cache32.worker import quant_state_digest
    from experiments.nan_relabel68.pipeline import paired_heads
    from experiments.nan_stream36.batch_engine import BatchedDepthEngine
    from experiments.nan_stream36.device_preflight import preflight
    from mechinterp.nanbeige_bpd_gym.lazy_bf16_loop import digest_dispatch
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    from .native import BoundaryCapture
    from .worker import drive

    backend = args.backend
    maxpos = registration()["max_position"]
    torch.set_num_threads(1)
    card = torch.cuda.get_device_properties(0)
    if "3050" not in card.name or not 3.5*2**30 <= card.total_memory <= 4.5*2**30:
        raise RuntimeError("The registered device is the 4 GiB RTX 3050")
    torch.cuda.set_per_process_memory_fraction(.9)
    torch.cuda.reset_peak_memory_stats()

    def check():
        guard(device=True)

    def slot(start, rows, length):
        if rows not in (1,2) or not 1<=length<=maxpos or monotonic()-start>runtime.LIMITS["maximum_slot_seconds"]:
            raise RuntimeError("B2 preflight exceeded the registered shape or duration")
        check()

    focus = [r for r in payload["public"] if r["arm"] == "focused"]
    tokens = payload["tokens"]["row_bindings"]
    roots = sorted({r["root"] for r in focus})
    by_world = {(p["root"],p["world"]):tokens[p["id"]]["ids"] for p in focus}
    root = max(roots,key=lambda r:len(by_world[r,"A"]))
    pair = [by_world[root,w] for w in ("A","B")]
    cfg = json.loads((REPO/"experiments/nan_readpath37/registration.json").read_bytes())
    with ExitStack() as stack:
        target_digest = stack.enter_context(digest_dispatch())
        loop, checkpoint, placement = runtime.load_model(backend,args.snapshot,args.adapter)
        before, q1 = target_digest(loop), quant_state_digest(loop)
        if (before != runtime.TARGETS[backend] or loop.n != 22 or
                (backend == "nf4" and q1["modules"] != 154) or
                (backend == "bf16" and loop.n_gpu_layers != 4)):
            raise ValueError("Pinned target, quantization or staged layout changed")
        guard(device=True,force_gpu_sample=True)
        model_head = loop.model.lm_head
        cpu_weight = model_head.weight
        if cpu_weight.shape[0] != len(classes) or model_head.bias is not None:
            raise ValueError("The pinned vocabulary matrix shape or bias differs")
        ref_head = StreamedVocabularyHead(cpu_weight,loop.device,8192)
        reference = BatchedDepthEngine(loop,ref_head)
        try:
            streamed = torch.stack(paired_heads(reference,pair,slot))
        finally:
            reference.close()
            ref_head.close()
        head = StreamedVocabularyHead(cpu_weight,loop.device,8192)
        stack.callback(head.close)
        if backend == "nf4":
            head.weight = cpu_weight.to(loop.device)
            model_head.weight = torch.nn.Parameter(head.weight,requires_grad=False)
        guard(device=True,force_gpu_sample=True)
        engine = BatchedDepthEngine(loop,head)
        try:
            depths = preflight(engine,pair,maxpos,slot)
            parity = torch.stack(paired_heads(engine,pair,slot))
        finally:
            engine.close()
        capture = BoundaryCapture(loop,head)
        stack.callback(capture.close)
        if not torch.equal(streamed,parity) or not torch.equal(parity,capture.capture(pair)["logits"]):
            raise ValueError("Placement or the boundary hook changed the unrestricted B2 heads")
        write_new(work/"preflight.json",{"depths":depths,"head_placement_exact":True,"native_heads_exact":True,
                 "partition_sha256":partition_sha(classes),"head_resident":backend=="nf4","stage":placement})
        sample_stream = stack.enter_context((args.snapshot/SHARD).open("rb"))
        fixed = FixedHead(sample_stream,args.snapshot,cfg,np,check)
        planner,design = certified_planner(backend,fixed,np)
        counts = drive(work,capture,payload,admission,classes,planner,design,
                       lambda:run_host(args,work,qualify=True),check,np)
        guard(device=True,force_gpu_sample=True)
        verify_checkpoint(args.snapshot,checkpoint)
        after,q2 = target_digest(loop),quant_state_digest(loop)
        if before != after or q1 != q2:
            raise ValueError("The run changed the pinned decoder or quantization state")
    if counts != (registration()["baseline_batches"],registration()["roots"]):
        raise RuntimeError("The worker did not close the registered roster")
    hashes = {f"{key}_sha256":sha(work/name) for key,name in (
        ("admission","admission.json"),("preflight","preflight.json"),("baseline","baseline.json"),
        ("qualification","qualification.json"),("journal","closed.jsonl"))}
    end = {k:admission[k] for k in ("experiment","runtime_sha256","published_commit","backend","mode")}
    write_new(work/"terminal.json",end | hashes | {"status":"completed","baseline_batches":counts[0],
              "root_records":counts[1],"target_before":before,"target_after":after,"quant_before":q1,
              "quant_after":q2,"checkpoint_verified_after":True,"partition_sha256":partition_sha(classes),
              "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes":torch.cuda.max_memory_reserved()})


def execute(args, work, admission):
    if {name:version(name) for name in runtime.GPU_PACKAGES} != runtime.GPU_PACKAGES:
        raise RuntimeError("Pinned device interpreter packages differ")
    with delay.resources(work) as guard:
        payload = roster.worker_source(args.source_dir,args.snapshot,guard)
        pins = json.loads(roster.RECEIPT.read_bytes())["tokenizer_pins"]
        _,classes = vocabulary(args.snapshot,guard,pins)
        import numpy as np
        import torch

        device(args,work,guard,payload,classes,admission,np,torch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--check-admission",action="store_true")
    parser.add_argument("--backend",choices=("nf4","bf16"))
    for name in ("snapshot","source-dir","adapter"):
        parser.add_argument("--"+name,type=Path)
    args = parser.parse_args()
    if args.freeze:
        write_new(MANIFEST,registration())
        return
    if not all((args.snapshot,args.source_dir,args.adapter,args.backend)):
        parser.error("Require the registered backend, pinned snapshot, frozen source and isolated adapter")
    frozen = verified()
    commissioned(frozen)
    if "torch" in sys.modules:
        raise RuntimeError("Torch was imported before GPU admission")
    work = ARTIFACT/args.backend
    if args.check_admission:
        admit(args,work)
        print("Local predecessor and resource conditions passed",flush=True)
        return
    with runtime.exclusive_lane():
        admit(args,work)
        commit = publication()
        work.mkdir(parents=True,exist_ok=False)
        admission = {"experiment":roster.EXPERIMENT,"runtime_sha256":frozen,"published_commit":commit,
                     "backend":args.backend,"mode":MODE,"pid":os.getpid()}
        write_new(work/"admission.json",admission)
        runtime.environment(work)
        try:
            execute(args,work,admission)
            run_host(args,work)
        except BaseException as error:
            write_new(work/"attempt_failure.json",{"error_type":type(error).__name__,"message":str(error)})
            raise


if __name__ == "__main__":
    main()
