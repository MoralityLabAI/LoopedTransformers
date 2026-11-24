"""Guarded BF16 execution for the fresh writer-to-reader pilot."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.nan_attnspan67.run import read_banks
from experiments.nan_cigate73.run import ATTN,BASE,controller
from experiments.nan_delay39 import pilot as delay
from experiments.nan_delay39.source import digest
from experiments.nan_gated46.worker import DELAY_QUEUE
from experiments.nan_radial76.native import WriterReaderCapture
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha

from .pipeline import batch_baseline,root_trial
from .source import CHECK,CONFIG,HERE,REPO,SOURCE,worker_inputs


MANIFEST = HERE/"RUNTIME_MANIFEST.json"
SOFTWARE = HERE/"SOFTWARE_CHECK.json"
ARTIFACT = Path("E:/codex-artifacts/NanbeigeWriter77-20261123")


def files():
    paths = [HERE/n for n in ("source.py","PROTOCOL.md","registration.json","SOURCE_MANIFEST.json",
                              "SOURCE_CHECK.json","pipeline.py","host.py","lane.py","verify.py")]
    paths += [REPO/n for n in ("experiments/nan_gated46/worker.py","experiments/nan_delay39/pilot.py",
                              "experiments/nan_readout34/runtime.py","experiments/nan_readout34/RUNTIME_MANIFEST.json",
                              "experiments/nan_queue43/LOCAL_MANIFEST.json","experiments/nan_radial76/native.py",
                              "experiments/nan_radial76/CAL_FEASIBILITY.json","experiments/nan_radial76/CPU_VERIFICATION.json",
                              "experiments/nan_radial76/geometry.py","experiments/nan_reader71/native.py",
                              "experiments/nan_relabel68/pipeline.py","experiments/nan_stream36/device_preflight.py",
                              "experiments/nan_stream36/batch_engine.py","experiments/nan_cigate73/run.py",
                              "experiments/nan_cigate73/SOURCE_MANIFEST.json","experiments/nan_cidose74/geometry.py",
                              "experiments/nan_cibracket75/repair.py","experiments/nan_fit07/pointwise.py",
                              "experiments/nan_attnspan67/run.py","experiments/nan_attnspan67/registration.json",
                              "experiments/nan_readpath37/registration.json",
                              "mechinterp/nanbeige_bpd_gym/chunked_cache.py",
                              "mechinterp/nanbeige_bpd_gym/streamed_vocab_head.py")]
    return {p.relative_to(REPO).as_posix():sha(p) for p in paths}


def registration():
    commissioned = json.loads(SOFTWARE.read_bytes())
    source = json.loads(CHECK.read_bytes())
    frozen = files()
    if (commissioned["status"]!="passed" or commissioned["files_sha256"]!=frozen
            or commissioned["registered_source_worlds"]!=source["prompts"]
            or commissioned["fixture_edited_worlds"]<2 or source["source_manifest_sha256"]!=sha(SOURCE)):
        raise ValueError("Source and CPU commissioning receipts do not match")
    radial = json.loads((REPO/"experiments/nan_radial76/CPU_VERIFICATION.json").read_bytes())
    if radial["status"]!="passed":
        raise ValueError("The native staged writer fixture is missing")
    return {"experiment":"NAN-WRITER-77","stage":"guarded_pretrained_pending","backend":"bf16",
            "b2_baseline_batches":source["prompts"]//2,"max_position":source["max_position"],
            "files_sha256":frozen | {SOFTWARE.relative_to(REPO).as_posix():sha(SOFTWARE)}}


def verified():
    if not __debug__ or Path.cwd()!=REPO or json.loads(MANIFEST.read_bytes())!=registration():
        raise ValueError("Runtime differs from the frozen experiment or canonical checkout")
    runtime.verify_manifest()
    return sha(MANIFEST)


def publication():
    commit = delay.verify_publication()
    path = MANIFEST.relative_to(REPO).as_posix()
    data = subprocess.check_output(["git","show",f"{commit}:{path}"],creationflags=runtime.FLAGS,timeout=30)
    if data!=MANIFEST.read_bytes():
        raise ValueError("The matching WRITER77 manifest is missing on the published branch")
    return commit


def admission(args):
    args.snapshot = args.snapshot.resolve()
    delay.admissions(args)
    predecessor = DELAY_QUEUE/"terminal.json"
    if not predecessor.is_file():
        raise RuntimeError("The registered DELAY39 predecessor remains pending")
    closed = json.loads(predecessor.read_bytes())
    if (closed["status"]!="completed" or closed["local_manifest_sha256"]!=
            sha(REPO/"experiments/nan_queue43/LOCAL_MANIFEST.json")):
        raise ValueError("DELAY39 terminal differs from registered custody")
    cfg = json.loads(CONFIG.read_bytes())
    if args.snapshot.name!=cfg["checkpoint_revision"]:
        raise ValueError("WRITER77 checkpoint revision changed")


def report_on_host(args,work,vocab,*more):
    env = os.environ.copy()
    env.update(USE_TORCH="0",USE_TF="0",USE_FLAX="0",CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable,"-B","-X","utf8","-m","experiments.nan_writer77.host",
                    "--source-dir",str(args.source_dir),"--snapshot",str(args.snapshot),
                    "--baseline",str(work/"baseline.json"),"--vocab",str(vocab),*map(str,more)],
                   env=env,timeout=120,check=True,creationflags=runtime.FLAGS)


def execute(args,work,published,frozen):
    if {pkg:version(pkg) for pkg in runtime.GPU_PACKAGES}!=runtime.GPU_PACKAGES:
        raise RuntimeError("Unexpected model interpreter package versions")
    with delay.resources(work) as guard:
        payload = worker_inputs(args.source_dir,args.snapshot,guard)
        import numpy as np
        import torch
        from experiments.nan_bf31.loader import verify_checkpoint
        from experiments.nan_cache32.worker import quant_state_digest
        from experiments.nan_fit07.pointwise import masks
        from experiments.nan_recomp09.common import target_digest
        from experiments.nan_stream36.batch_engine import BatchedDepthEngine
        from experiments.nan_stream36.device_preflight import preflight
        from mechinterp.nanbeige_bpd_gym.lazy_bf16_loop import digest_dispatch
        from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

        torch.set_num_threads(1)
        props = torch.cuda.get_device_properties(0)
        if "3050" not in props.name or not 3.5*2**30<=props.total_memory<=4.5*2**30:
            raise RuntimeError("Expected the registered 4 GiB RTX 3050")
        torch.cuda.set_per_process_memory_fraction(.9)
        torch.cuda.reset_peak_memory_stats()
        cfg = json.loads(CONFIG.read_bytes())
        attn,base = (json.loads(p.read_bytes()) for p in (ATTN,BASE))
        banks = read_banks("bf16",21,attn,base,np)
        fit = controller("t2_layer21_o_proj_rank8",attn["ranksite_audits"]["2"],banks["attn_bpd_t2"],np,masks)
        labels = payload["tokens"]["bare_label_token_ids"]
        check = lambda:guard(device=True)

        def replay(projection,value,out):
            reference = torch.nn.functional.linear(value,projection.weight,projection.bias)
            if not torch.equal(reference,out):
                raise RuntimeError("Native reader projection failed same-kernel replay")

        with digest_dispatch():
            loop,checkpoint,stage = runtime.load_model("bf16",args.snapshot,args.adapter)
            before,quant = target_digest(loop),quant_state_digest(loop)
            if before!=runtime.TARGETS["bf16"] or loop.n!=22:
                raise ValueError("Checkpoint and physical writer layer differ")
            guard(device=True,force_gpu_sample=True)
            head = StreamedVocabularyHead(loop.model.lm_head.weight,loop.device,8192)
            try:
                tokens = payload["tokens"]["row_bindings"]
                sample = next(row for row in payload["public"] if row["arm"]=="focused" and row["world"]=="A")
                peer = next(row for row in payload["public"] if row["root"]==sample["root"]
                            and row["arm"]=="focused" and row["world"]=="B")
                pair = [tokens[row["id"]]["ids"] for row in (sample,peer)]
                eng = BatchedDepthEngine(loop,head)
                try:
                    checks = preflight(eng,pair,registration()["max_position"],lambda *_:check())
                finally:
                    eng.close()
                write_new(work/"preflight.json",{"device_depths":checks,"stage":stage})
                capture = WriterReaderCapture(loop,head,replay=replay)
                try:
                    baseline,focus = batch_baseline(capture,payload,cfg["arms"],check)
                    baseline_path = work/"baseline.json"
                    gate_path = work/"qualification.json"
                    write_new(baseline_path,baseline)
                    vocab = len(head.weight)
                    report_on_host(args,work,vocab,"--output",gate_path)
                    gate = json.loads(gate_path.read_bytes())
                    if gate["baseline_digest"]!=digest(baseline):
                        raise ValueError("Independent competence gate bound a different baseline")
                    qualified = set(gate["qualified_focused_roots"])
                    with (work/"closed.jsonl").open("x",encoding="utf-8",newline="\n") as journal:
                        for i,(root,(packet,ids)) in enumerate(sorted(focus.items())):
                            if not gate["passed"] or root not in qualified:
                                status = "native_gate_failed" if not gate["passed"] else "native_root_unqualified"
                                result = {"status":status}
                            else:
                                result = root_trial(capture,root,packet,ids,labels,fit,masks,np,check)
                            ordered_ids = [r["id"] for w in ("A","B") for r in payload["public"]
                                           if r["root"]==root and r["arm"]=="focused" and r["world"]==w]
                            path = work/f"root_{i:03d}.json"
                            write_new(path,result | {"root":root,"source_ids":ordered_ids})
                            journal.write(json.dumps({"file":path.name,"sha256":sha(path)})+"\n")
                            journal.flush()
                finally:
                    capture.close()
            finally:
                head.close()
            check()
            verify_checkpoint(args.snapshot,checkpoint)
            if target_digest(loop)!=before or quant_state_digest(loop)!=quant:
                raise RuntimeError("Target parameters or quantization changed")
            write_new(work/"terminal.json",{"experiment":"NAN-WRITER-77","status":"completed","vocab":vocab,
                      "runtime_sha256":frozen,"published_commit":published,"checkpoint_verified_after":True,
                      "quant_unchanged":True,"target_sha256":before,"journal_sha256":sha(work/"closed.jsonl"),
                      "baseline_sha256":sha(baseline_path),"qualification_sha256":sha(gate_path),
                      "preflight_sha256":sha(work/"preflight.json"),"peak_torch_bytes":torch.cuda.max_memory_allocated()})
            report_on_host(args,work,vocab,"--qualification",gate_path,"--closed",work,"--output",work/"aggregate.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze",action="store_true")
    p.add_argument("--check-admission",action="store_true")
    p.add_argument("--source-dir",type=Path)
    p.add_argument("--snapshot",type=Path)
    p.add_argument("--adapter",type=Path)
    args=p.parse_args()
    if args.freeze:
        write_new(MANIFEST,registration())
        return
    if not (args.source_dir and args.snapshot and args.adapter):
        p.error("Require the registered source, snapshot and isolated adapter")
    frozen = verified()
    if "torch" in sys.modules:
        raise RuntimeError("Torch loaded before admission")
    if args.check_admission:
        admission(args)
        print("Local prerequisites passed",flush=True)
        return
    with runtime.exclusive_lane():
        admission(args)
        published = publication()
        work = ARTIFACT/"bf16"
        work.mkdir(parents=True,exist_ok=False)
        write_new(work/"admission.json",{"pid":os.getpid(),"runtime_sha256":frozen,"published_commit":published})
        runtime.environment(work)
        try:
            execute(args,work,published,frozen)
        except BaseException as e:
            write_new(work/"attempt_failure.json",{"type":type(e).__name__,"message":str(e)})
            raise


if __name__=="__main__":
    main()
