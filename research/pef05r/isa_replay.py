"""PEF-05R controlled CPU-ISA dispatch experiment.

Rebuild exact first 125 vanilla PEF-05 training steps on *one* host,
with all training data and RNG streams identical. Select oneDNN
instruction set BEFORE importing torch via DNNL_MAX_CPU_ISA:

  DNNL_MAX_CPU_ISA=AVX2 python -m research.pef05r.isa_replay --expect avx2 ...
  DNNL_MAX_CPU_ISA=AVX512_CORE python -m research.pef05r.isa_replay --expect avx512 ...

The two expected SHA256 tables come from original hosted runs
43688425647 and 43687506414, respectively. The exact expectation
is meaningful only under CPU-capable oneDNN/PyTorch 2.10.0; a
mismatched architecture fails rather than fabricating equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch

from research.pef03.pef03 import BitMemoryLooper, hash_weights, tensor
from research.pef05.pef05 import BATCH, SEEDS, _hash_optimizer, pair_split, train_update

REFERENCE={
    "avx512":{
        101:(
          "e414639bcaa4a0e178e62e4d097238e89431cd5b03e218e5b2d497382f935892",
          "1338478d2ef79051a5077aaea46a0242e0ae90f87905c1906e5f17073de81adb",
          "dc62afea380fbe9411f73dc60ae63c7867799aeca6e213ecad49a00a80ddaafc"),
        109:(
          "f4a5d611c8b526280fb2244112f9091f6f5c33b97db6c0e48be7a48265329d4c",
          "4eb37523ce14f55837a35e7493282242152c04c1db48414f3e2c12d3932b094d",
          "5dacdc65f92cf8598561da3167c063daa78648576e3c58394479e14ab0b32ffa"),
        127:(
          "d625453d6769fef2540f73cad68e4ca5404d6dccce267ea5bcad81476069e267",
          "329168605519570f71fa4b2b927fa60eab53e32f66159a242171dad17d978e12",
          "e82413ad4c2b4eeb8de2f14891961456226b8c9593eca5be1ddc2caaf297c72e"),
        157:(
          "bdcb70a5ccbea7b2c7cdc93eb3dce0e39edf7f7aac98ab2266b1cdf3579f4c34",
          "72105788516bf102f0c1c396fab7c8303d7d761bb43bf1ea4e5ece453a8c679b",
          "e685e0ada93f5475e471d7dae6cf5982bf02e22931b8da69f8a6804d89b03a52"),
    },
    "avx2":{
        101:(
          "e414639bcaa4a0e178e62e4d097238e89431cd5b03e218e5b2d497382f935892",
          "e45b05e1777c1aa524a291b1cf0d2c29bdbc67d7a998a9d51596d43c0653f594",
          "65e2fb5d060d7867cbb5d405bab70bf5813cec8ce8e8d5569abad19864af8e4d"),
        109:(
          "f4a5d611c8b526280fb2244112f9091f6f5c33b97db6c0e48be7a48265329d4c",
          "568c1b9561d700cdd32af16a17ef288e62ea867fec924e191d7e0f2e9106b847",
          "0974632b480264e0e820e310350776315be37615bffdf0bdf398bc84ee8d9add"),
        127:(
          "d625453d6769fef2540f73cad68e4ca5404d6dccce267ea5bcad81476069e267",
          "1f0889a98a5a569241322ddce54805b71b6c0a2118311907ab51a2343f13805f",
          "3afa9eb3a0ad3e64a953457d7f6fd4fe0c4da8ef003489b62079690f9b7d0a15"),
        157:(
          "bdcb70a5ccbea7b2c7cdc93eb3dce0e39edf7f7aac98ab2266b1cdf3579f4c34",
          "5aa31d6aa6964ecaf0d62ef76910faeff7f484c159933ac9bd5cadc414f0765d",
          "d4b059e418467219fd5d8065040bd8add77e835a0831758a2ced8aa78424597d"),
    },
}

SOURCE_SCRIPT_GIT_SHA1="87c414464a2d2e0a73f6329d9dbdc43ce8561832"


def check_source_blob():
    path=Path("research/pef05/pef05.py")
    raw=path.read_bytes()
    actual=hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()
    if actual!=SOURCE_SCRIPT_GIT_SHA1:
        raise AssertionError(
            f"reproduction source changed: {actual} != {SOURCE_SCRIPT_GIT_SHA1}"
        )
    return actual


def replay_prefix(seed:int,expected:str):
    torch.manual_seed(seed)
    model=BitMemoryLooper().train()
    opt=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0)
    train_data,_=pair_split()
    pool=tensor(train_data["train"])
    generator=torch.Generator(device="cpu").manual_seed(seed*7919+59)
    batches=torch.randint(len(pool),(250,BATCH),generator=generator)
    reference_init,reference_final,reference_optimizer=REFERENCE[expected][seed]
    initial=hash_weights(model)
    if initial!=reference_init:
        raise AssertionError(f"source initial model digest mismatched seed {seed}")
    sampled=[]
    for step in range(1,126):
        train_update(model,opt,pool[batches[step-1]],
                     arm="baseline",seed=seed,step=step)
        if step in (1,10,50,125):
            sampled.append({"step":step,"weight_sha256":hash_weights(model),
                            "optimizer_sha256":_hash_optimizer(opt)})
    terminal=hash_weights(model)
    optimizer_final=_hash_optimizer(opt)
    return {
        "seed":seed,
        "initial_parameter_sha256":initial,
        "terminal_step125_parameter_sha256":terminal,
        "terminal_step125_optimizer_sha256":optimizer_final,
        "original_model_exactly_matched":terminal==reference_final,
        "original_optimizer_exactly_matched":optimizer_final==reference_optimizer,
        "training_milestones":sampled,
        "no_surgery_or_heldout_scoring":True,
    }


def run(expect:str):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if expect not in REFERENCE:
        raise ValueError("expected oneDNN route is avx2 or avx512")
    maximum=os.getenv("DNNL_MAX_CPU_ISA")
    wanted="AVX2" if expect=="avx2" else "AVX512_CORE"
    if maximum!=wanted:
        raise AssertionError(f"DNNL_MAX_CPU_ISA must be {wanted}, was {maximum}")
    if str(torch.__version__) not in ("2.10.0+cpu","2.10.0"):
        raise AssertionError("different PyTorch math backend version")
    code_id=check_source_blob()
    cases=[replay_prefix(seed,expect) for seed in SEEDS]
    passed=all(x["original_model_exactly_matched"] and
               x["original_optimizer_exactly_matched"] for x in cases)
    # Record all newly observed CPU numerical paths. PyTorch can use
    # other CPU-specific kernels even after oneDNN's maximum ISA is set.
    # Every mismatch is named explicitly and cannot count as a matched
    # reproduction of an archived checkpoint.
    return {
        "campaign":"PEF-05R-ISA",
        "comparison_route":expect,
        "reference_source_run_id":43688425647 if expect=="avx2" else 43687506414,
        "source_script_git_sha1":code_id,
        "torch_version":str(torch.__version__),
        "dnnl_max_cpu_isa":maximum,
        "aten_cpu_capability":torch.backends.cpu.get_cpu_capability(),
        "mkldnn_enabled":torch.backends.mkldnn.enabled,
        "platform":platform.platform(),
        "same_seed_initialization_and_batch_stream":True,
        "reference_match_status":("bitwise_original_checkpoint_reproduced"
                                  if passed else "additional_cpu_execution_path"),
        "processor_model": next(
            (line.split(":",1)[1].strip()
             for line in Path("/proc/cpuinfo").read_text().splitlines()
             if line.lower().startswith("model name") and ":" in line),
            "unavailable"
        ),
        "all_four_original_source_models_exact_SHA256_match":passed,
        "all_four_original_source_optimizers_exact_SHA256_match":passed,
        "source_seeds":list(SEEDS),
        "no_new_preregistered_result_or_heldout_scoring":True,
        "cases":cases,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect",choices=tuple(REFERENCE),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--strict",action="store_true",
                        help="fail when a different CPU numerical path is observed")
    args=parser.parse_args()
    report=run(args.expect)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"route":report["comparison_route"],
        "source_run":report["reference_source_run_id"],
        "reference_status":report["reference_match_status"],
        "all_four_exact_model_and_optimizer_matches":
            report["all_four_original_source_models_exact_SHA256_match"],
        "cpu":report["aten_cpu_capability"],
        "processor_model":report["processor_model"]},indent=2))
    if args.strict and report["reference_match_status"]!="bitwise_original_checkpoint_reproduced":
        raise AssertionError(
            "different CPU numerical path: see saved ISA report, original "
            "reference was not reproduced bitwise"
        )


if __name__=="__main__":
    main()
