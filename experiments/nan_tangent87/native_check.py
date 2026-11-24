"""Exercise both-world scoped edit and early refusal on the staged CPU decoder."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

from experiments.nan_phase35.run import lease
from experiments.nan_premlp64.verify import BUDGET, CONFIGURATION, PACKAGES, hook_count
from experiments.nan_readpath37.run import bound, write_new
from experiments.nan_vocab82.heads import INVALID, LABELS, Reducer

from .lane import verified
from .source import REPO
from .worker import choose


def exercise(snapshot,check,np,torch):
    from experiments.nan_recomp09.common import target_digest
    from experiments.nan_relabel68.pipeline import paired_heads
    from experiments.nan_routematch86.native import MlpWriterCapture
    from experiments.nan_stream36.batch_engine import BatchedDepthEngine
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import build_nanbeige_model_and_loop
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    torch.manual_seed(86031)
    model,loop=build_nanbeige_model_and_loop(snapshot,exec_device=torch.device("cpu"),dtype=torch.bfloat16,
        n_gpu_layers=2,random_init=True,config_overrides={"vocab_size":97,"num_hidden_layers":3,"hidden_size":192,
            "intermediate_size":80,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"kv_channels":16})
    model.requires_grad_(False)
    loop.stage_layer.requires_grad_(False)
    before,hooks=target_digest(loop),hook_count(loop)
    head=StreamedVocabularyHead(model.lm_head.weight,loop.device,13)
    prompts=[[1,4,32,5,29,6,42],[8,39,3,4,65,77,91]]
    classes=bytes(i%(INVALID+1) for i in range(len(head.weight)))
    reduce=Reducer(classes)
    bare=dict(zip(LABELS,range(INVALID),strict=True))
    rng=np.random.default_rng(86031)
    directions=rng.normal(size=(2,2,192))

    def routes(inputs,writes):
        assert inputs.shape==(2,80) and writes.shape==(2,192)
        return directions[:,0],directions[:,1]

    def failed(inputs,writes):
        a,b=routes(inputs,writes)
        refused=b.copy()
        refused[1]=0.
        return a,refused

    try:
        engine=BatchedDepthEngine(loop,head,chunk=3)
        try:
            reference=torch.stack(paired_heads(engine,prompts,lambda *_:check()))
        finally:
            engine.close()
        capture=MlpWriterCapture(loop,head,chunk=3)
        try:
            clean=capture.capture(prompts)
            assert torch.equal(clean["logits"],reference)
            count=head.calls
            refused=choose(capture,clean,prompts,["cpu_fixture_0","cpu_fixture_1"],
                           failed,bare,reduce,classes,check,np)
            assert refused["status"]=="geometry_unqualified" and head.calls==count
            accepted=choose(capture,clean,prompts,["cpu_fixture_0","cpu_fixture_1"],
                            routes,bare,reduce,classes,check,np)
            calls=head.calls-count
        finally:
            capture.close()
    finally:
        head.close()
    assert accepted["status"]=="closed" and len(accepted["worlds"])==2 and calls==28
    assert all(w["rescue_k_exact"] and w["clean_repeat_exact"] for w in accepted["worlds"])
    assert all(w["class_maxima"]["learned"][0]==w["class_maxima"]["learned_k_rescue"][0]
               for w in accepted["worlds"])
    assert not torch.cuda.is_initialized() and target_digest(loop)==before and hook_count(loop)==hooks
    return {"both_heads_exact":True,"scoped_worlds":2,"head_calls":calls,
            "failed_world_prevented_both_edits":True,"native_retained_checkpoint":True,
            "gpu_initialized":False,"pretrained_forwards":0}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot",type=Path,required=True)
    cli.add_argument("--work",type=Path,required=True)
    args=cli.parse_args()
    frozen=verified()
    base=json.loads((REPO/"experiments/nan_readpath37/registration.json").read_bytes())
    assert args.snapshot.name==base["revision"] and "torch" not in sys.modules
    with lease(BUDGET) as (check,samples):
        args.work.mkdir(parents=True,exist_ok=False)
        os.environ.update(USE_TORCH="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
            HF_HOME=str((args.work/"hf_home").resolve()),HF_MODULES_CACHE=str((args.work/"hf_modules").resolve()))
        os.environ.pop("TRANSFORMERS_CACHE",None)
        for file in ("config.json","modeling_nanbeige.py"):
            bound(args.snapshot/file,base["snapshot_pins"][file])
        bound(args.snapshot/"configuration_nanbeige.py",CONFIGURATION)
        assert {name:version(name) for name in PACKAGES}==PACKAGES
        import numpy as np
        import torch

        torch.set_num_threads(1)
        result=exercise(args.snapshot,check,np,torch)
    assert frozen==verified()
    write_new(args.work/"verification.json",result|{"runtime_sha256":frozen,"status":"passed",
               "mode":"random_staged_cpu_bf16","max_sampled_private_bytes":max(s["private_bytes"] for s in samples),
               "max_sampled_working_set_bytes":max(s["working_set_bytes"] for s in samples),
               "min_sampled_available_RAM_bytes":min(s["available_physical_bytes"] for s in samples)})
    print("Both-world MLP worker and early refusal passed",flush=True)


if __name__=="__main__":
    main()
