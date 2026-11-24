"""Guarded file-backed NF4 B2 lane for the fresh matched MLP tangent pilot."""

import argparse
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
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_vocab82.heads import partition_sha
from experiments.nan_vocab83.custody import vocabulary

from .source import CHECK, EXPERIMENT, HERE, REPO, SOURCE, config, seal, worker_inputs


MANIFEST=HERE/"RUNTIME_MANIFEST.json"
PROOF=HERE/"RUNTIME_CHECK.json"
WORK=Path("E:/codex-artifacts/NanbeigeTangent87-20261123/nf4")
ROUTEMATCH=REPO/"experiments/nan_routematch86/NATIVE_CHECK.json"


def registered():
    src=json.loads(CHECK.read_bytes())
    tuning=json.loads(ROUTEMATCH.read_bytes())
    if (json.loads(SOURCE.read_bytes())!=seal() or src["source_manifest_sha256"]!=sha(SOURCE)
            or src["status"]!="source_tokenizer_verified" or tuning["status"]!="passed"
            or tuning["scoped_worlds"]!=2 or not tuning["exact_native_clean_heads"]
            or any(sha(REPO/p)!=v for p,v in tuning["files_sha256"].items())):
        raise ValueError("Source or staged MLP native fixture changed")
    paths=[HERE/n for n in ("__init__.py","PROTOCOL.md","SOURCE_MANIFEST.json","SOURCE_CHECK.json",
                            "registration.json","source.py","worker.py","host.py","lane.py","verify.py","native_check.py")]
    paths += [ROUTEMATCH,*[REPO/n for n in (
        "experiments/nan_recipe85/SOURCE_MANIFEST.json","experiments/nan_routematch86/SOURCE_MANIFEST.json",
        "experiments/nan_writer77/pipeline.py","experiments/nan_vocab82/VERIFICATION.json",
        "experiments/nan_vocab82/host.py","experiments/nan_vocab83/batches.py",
        "experiments/nan_vocab83/custody.py","experiments/nan_vocab83/verify.py",
        "experiments/nan_delay39/pilot.py","experiments/nan_readout34/RUNTIME_MANIFEST.json",
        "experiments/nan_queue43/LOCAL_MANIFEST.json","experiments/nan_gated46/worker.py",
        "experiments/nan_ram28/loader.py","experiments/nan_bf31/loader.py","experiments/nan_cache32/worker.py",
        "experiments/nan_recomp09/common.py","experiments/nan_stream36/device_preflight.py",
        "experiments/nan_stream36/batch_engine.py","mechinterp/nanbeige_bpd_gym/chunked_backend.py",
        "mechinterp/nanbeige_bpd_gym/streamed_vocab_head.py","experiments/nan_relabel68/pipeline.py",
    )]]
    return {"experiment":EXPERIMENT,"backend":"nf4","resident_unmodified_BF16_head":True,
            "b2_baseline_batches":src["b2_batches"],"registered_roots":src["roots"],
            "max_position":src["max_position"],"files_sha256":{p.relative_to(REPO).as_posix():sha(p) for p in paths}}


def verified():
    if not __debug__ or Path.cwd()!=REPO or json.loads(MANIFEST.read_bytes())!=registered():
        raise ValueError("Use the registered NF4 runtime from the canonical checkout")
    runtime.verify_manifest()
    return sha(MANIFEST)


def commissioned(frozen):
    result=json.loads(PROOF.read_bytes())
    if (result["runtime_sha256"]!=frozen or result["status"]!="passed" or result["pretrained_forwards"]!=0
            or result["registered_worlds"]!=2*3*registered()["registered_roots"]
            or result["native_cpu"]["runtime_sha256"]!=frozen
            or result["native_cpu"]["scoped_worlds"]!=2
            or result["native_cpu"]["head_calls"]!=28
            or result["native_cpu"]["failed_world_prevented_both_edits"] is not True
            or not all(result[k] is True for k in ("host_two_pass_effect_checked","hash_tamper_refused",
                                                  "worker_without_gold","geometry_tamper_refused",
                                                  "fixture_refused_as_pretrained"))):
        raise ValueError("The staged host receipt does not certify the frozen runtime")


def admit(args):
    args.snapshot=args.snapshot.resolve()
    delay.admissions(args)
    path=DELAY_QUEUE/"terminal.json"
    if not path.is_file():
        raise RuntimeError("The earlier registered DELAY39 sequence remains pending")
    result=json.loads(path.read_bytes())
    if (result["status"]!="completed" or result["local_manifest_sha256"]!=
            sha(REPO/"experiments/nan_queue43/LOCAL_MANIFEST.json")):
        raise ValueError("The existing DELAY39 terminal does not close its published queue")
    if WORK.exists():
        raise FileExistsError("The one-time NF4 pilot attempt is already occupied")


def publication():
    commit=delay.verify_publication()
    for p in (MANIFEST,PROOF):
        link=p.relative_to(REPO).as_posix()
        published=subprocess.check_output(["git","show",f"{commit}:{link}"],timeout=30,creationflags=runtime.FLAGS)
        if published!=p.read_bytes():
            raise RuntimeError("The matching NF4 runtime or verification receipt is not published")
    return commit


def host(args,work,*,qualify_only=False,fixture=False):
    env=os.environ.copy()
    env.update(USE_TORCH="0",USE_TF="0",USE_FLAX="0",CUDA_VISIBLE_DEVICES="")
    cmd=[sys.executable,"-B","-X","utf8","-m","experiments.nan_tangent87.host",
         "--source-dir",str(args.source_dir),"--snapshot",str(args.snapshot),"--work",str(work)]
    if qualify_only:
        cmd.append("--qualification")
    if fixture:
        cmd.append("--fixture")
    subprocess.run(cmd,env=env,timeout=135,check=True,creationflags=runtime.FLAGS)


def terminal(work,frozen,commit,classes,before,after,q1,q2,counts,peak,*,mode="pretrained_cuda"):
    if before!=runtime.TARGETS["nf4"] or before!=after or q1!=q2:
        raise ValueError("Closed parameter or quant state differs from admission")
    n,r=counts
    hashes={f"{k}_sha256":sha(work/file) for k,file in (("admission","admission.json"),
                ("baseline","baseline.json"),("qualification","qualification.json"),
                ("preflight","preflight.json"),("journal","closed.jsonl"))}
    write_new(work/"terminal.json",hashes|{"experiment":EXPERIMENT,"runtime_sha256":frozen,"published_commit":commit,
              "mode":mode,"status":"completed","backend":"nf4","target_before":before,"target_after":after,
              "quant_before":q1,"quant_after":q2,"vocab":len(classes),"partition_sha256":partition_sha(classes),
              "checkpoint_verified_after":True,"b2_baseline_batches":n,"root_records":r,"peak_torch_bytes":peak})


def execute(args,work,commit,frozen):
    if {k:version(k) for k in runtime.GPU_PACKAGES}!=runtime.GPU_PACKAGES:
        raise RuntimeError("Pinned GPU interpreter packages differ")
    with delay.resources(work) as guard:
        payload=worker_inputs(args.source_dir,args.snapshot,guard)
        _,classes=vocabulary(args.snapshot,guard,json.loads(CHECK.read_bytes())["tokenizer_pins"])
        import numpy as np
        import torch
        from experiments.nan_bf31.loader import checkpoint_entry,verify_checkpoint
        from experiments.nan_cache32.worker import quant_state_digest
        from experiments.nan_fit07.pointwise import masks
        from experiments.nan_ram28.loader import empty_model,materialize
        from experiments.nan_recomp09.common import target_digest
        from experiments.nan_relabel68.pipeline import paired_heads
        from experiments.nan_routematch86.native import MlpWriterCapture
        from experiments.nan_stream36.batch_engine import BatchedDepthEngine
        from experiments.nan_stream36.device_preflight import preflight
        from mechinterp.dark_fiber_gyms.arg03_nanbeige import _reinit_rotary_buffers,assert_oracle_isolated
        from mechinterp.nanbeige_bpd_gym.chunked_backend import build_nf4_loop
        from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead
        from .worker import prepare_fit,run_pairs

        torch.set_num_threads(1)
        card=torch.cuda.get_device_properties(0)
        if "3050" not in card.name or not 3.5*2**30<=card.total_memory<=4.5*2**30:
            raise RuntimeError("Expected the registered 4 GiB RTX 3050")
        torch.cuda.set_per_process_memory_fraction(.9)
        torch.cuda.reset_peak_memory_stats()
        routes=prepare_fit(np,masks)
        assert_oracle_isolated()
        entry=checkpoint_entry(args.adapter,args.snapshot)
        verify_checkpoint(args.snapshot,entry)
        from research.algernon.algebraic_reasoning_gym.arg02 import nanbeige

        model=empty_model(args.snapshot)
        nanbeige.check_supported(model.config)
        placement=materialize(model,args.snapshot)
        for layer in model.model.layers:
            _reinit_rotary_buffers(layer,torch.device("cpu"))
        model.eval().requires_grad_(False)
        loop=build_nf4_loop(model,"cuda:0",guard=guard)
        assert_oracle_isolated()
        before,q1=target_digest(loop),quant_state_digest(loop)
        if before!=runtime.TARGETS["nf4"] or loop.n!=22 or q1["modules"]!=154:
            raise ValueError("The pinned NF4 decoder bytes or quantization count changed")
        guard(device=True,force_gpu_sample=True)
        cpu_weight=model.lm_head.weight
        if len(cpu_weight)!=len(classes) or model.lm_head.bias is not None:
            raise ValueError("The native vocabulary shape or bias differs")
        maxpos=registered()["max_position"]

        def check():
            guard(device=True)

        def slot(start,rows,position):
            if rows not in (1,2) or not 1<=position<=maxpos or monotonic()-start>runtime.LIMITS["maximum_slot_seconds"]:
                raise RuntimeError("Native preflight exceeded the registered shape or duration")
            check()

        focus=[p for p in payload["public"] if p["arm"]=="focused"]
        root=focus[0]["root"]
        bindings=payload["tokens"]["row_bindings"]
        pair=[bindings[next(p["id"] for p in focus if p["root"]==root and p["world"]==w)]["ids"]
              for w in ("A","B")]
        streamed=StreamedVocabularyHead(cpu_weight,loop.device,8192)
        streamed_engine=BatchedDepthEngine(loop,streamed)
        try:
            stream_reference=torch.stack(paired_heads(streamed_engine,pair,slot))
        finally:
            streamed_engine.close()
            streamed.close()
        resident=StreamedVocabularyHead(cpu_weight,loop.device,8192)
        try:
            resident.weight=cpu_weight.to(loop.device)
            model.lm_head.weight=torch.nn.Parameter(resident.weight,requires_grad=False)
            if model.lm_head.weight.data_ptr()!=resident.weight.data_ptr():
                raise RuntimeError("Resident B2 head diverged from the native LM head")
            guard(device=True,force_gpu_sample=True)
            engine=BatchedDepthEngine(loop,resident)
            try:
                depth=preflight(engine,pair,maxpos,slot)
                reference=torch.stack(paired_heads(engine,pair,slot))
            finally:
                engine.close()
            if not torch.equal(stream_reference,reference):
                raise RuntimeError("Resident placement changed a physical B2 head")
            capture=MlpWriterCapture(loop,resident)
            try:
                if not torch.equal(reference,capture.capture(pair)["logits"]):
                    raise RuntimeError("The clean MLP hook changed an unrestricted B2 head")
                write_new(work/"preflight.json",{"depths":depth,"native_B2_heads_exact":True,
                   "resident_and_streamed_exact":True,"partition_sha256":partition_sha(classes),
                   "read_only_checkpoint_mappings":placement,"resident_weight_bytes":resident.weight.numel()*2})
                counts=run_pairs(work,capture,payload,classes,routes,np,check,
                                 lambda:host(args,work,qualify_only=True))
            finally:
                capture.close()
        finally:
            resident.close()
        guard(device=True,force_gpu_sample=True)
        verify_checkpoint(args.snapshot,entry)
        after,q2=target_digest(loop),quant_state_digest(loop)
        freeze=registered()
        if counts!=(freeze["b2_baseline_batches"],freeze["registered_roots"]):
            raise RuntimeError("Finished worker did not close its scheduled roster")
        terminal(work,frozen,commit,classes,before,after,q1,q2,counts,torch.cuda.max_memory_allocated())


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze",action="store_true")
    cli.add_argument("--check-admission",action="store_true")
    cli.add_argument("--snapshot",type=Path)
    cli.add_argument("--source-dir",type=Path)
    cli.add_argument("--adapter",type=Path)
    args=cli.parse_args()
    if args.freeze:
        write_new(MANIFEST,registered())
        return
    if not all((args.snapshot,args.source_dir,args.adapter)):
        cli.error("Require exact source, checkpoint and adapter")
    frozen=verified()
    commissioned(frozen)
    if "torch" in sys.modules:
        raise RuntimeError("Torch imported before admission")
    if args.check_admission:
        admit(args)
        print("All local prerequisites passed",flush=True)
        return
    with runtime.exclusive_lane():
        admit(args)
        commit=publication()
        WORK.mkdir(parents=True,exist_ok=False)
        write_new(WORK/"admission.json",{"runtime_sha256":frozen,"published_commit":commit,
                                         "mode":"pretrained_cuda","pid":os.getpid()})
        runtime.environment(WORK)
        try:
            execute(args,WORK,commit,frozen)
            host(args,WORK)
        except BaseException as exc:
            write_new(WORK/"attempt_failure.json",{"type":type(exc).__name__,"message":str(exc)})
            raise


if __name__=="__main__":
    main()
