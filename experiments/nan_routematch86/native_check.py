"""Verify staged MLP writer bytes and second-pass custody on random BF16 weights."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

from experiments.nan_phase35.run import lease
from experiments.nan_premlp64.verify import BUDGET, CONFIGURATION, PACKAGES, hook_count
from experiments.nan_readpath37.run import bound, digest, write_new
from experiments.nan_vocab82.heads import INVALID, LABELS, Reducer

from .geometry import plan
from .native import MlpWriterCapture, observed_trial
from .run import BASE, HERE, MANIFEST, REPO, seal


def run(args,np,torch,check):
    from experiments.nan_radial76.native import WRITER
    from experiments.nan_recomp09.common import target_digest
    from experiments.nan_relabel68.pipeline import paired_heads
    from experiments.nan_stream36.batch_engine import BatchedDepthEngine
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import build_nanbeige_model_and_loop
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    torch.manual_seed(86031)
    model,loop=build_nanbeige_model_and_loop(
        args.snapshot,exec_device=torch.device("cpu"),dtype=torch.bfloat16,n_gpu_layers=2,
        random_init=True,config_overrides={"vocab_size":97,"num_hidden_layers":3,"hidden_size":384,
        "intermediate_size":160,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"kv_channels":16})
    model.requires_grad_(False)
    loop.stage_layer.requires_grad_(False)
    before,hooks=target_digest(loop),hook_count(loop)
    head=StreamedVocabularyHead(model.lm_head.weight,loop.device,13)
    prompts=[[1,4,32,5,29,6,42],[8,39,3,4,65,77,91]]
    classes=bytes(i%(INVALID+1) for i in range(len(head.weight)))
    reduce=Reducer(classes)
    bare=dict(zip(LABELS,range(INVALID),strict=True))
    rng=np.random.default_rng(86031)
    diagnostics=[]
    try:
        engine=BatchedDepthEngine(loop,head,chunk=3)
        try:
            reference=torch.stack(paired_heads(engine,prompts,lambda *_:check()))
        finally:
            engine.close()
        capture=MlpWriterCapture(loop,head,chunk=3)
        try:
            clean=capture.capture(prompts)
            assert torch.equal(reference,clean["logits"])
            assert clean["writer_input"][WRITER].numel()==160
            assert clean["writer_raw"][WRITER].numel()==384
            for row,ids in enumerate(prompts):
                duplicate=capture.capture([ids,ids])
                assert torch.equal(duplicate["logits"][:,1],reference[:,row])
                native=duplicate["writer_raw"][WRITER].float().numpy().astype(np.float64)
                directions=rng.normal(size=(2,len(native)))
                writes,proof=plan(native,*directions,f"cpu_fixture_{row}",np)
                qualified=proof["anchored_tangent"]
                assert qualified["status"]=="matched" and not qualified["same_learned_and_static_bytes"]
                before_calls=head.calls
                result=observed_trial(capture,ids,duplicate,writes["anchored_tangent"],bare,reduce,classes,check)
                assert head.calls-before_calls==12
                assert result["zero_hook_exact"] and result["rescue_k_exact"] and result["clean_repeat_exact"]
                assert set(result["class_maxima"])=={"learned","cal_static","seeded","learned_k_rescue"}
                diagnostics.append({"matched_shell":qualified,"two_head_calls_per_prefill":True,
                                    "class_observations":sum(len(v) for v in result["class_maxima"].values()),
                                    "writer_site":"mlp.down_proj","reader_l2":result["reader_l2"]})
        finally:
            capture.close()
    finally:
        head.close()
    assert not torch.cuda.is_initialized() and target_digest(loop)==before and hook_count(loop)==hooks
    return {"exact_native_clean_heads":True,"checkpoint_unchanged":True,
            "scoped_worlds":len(diagnostics),"diagnostics":diagnostics}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot",type=Path,required=True)
    cli.add_argument("--work",type=Path,required=True)
    args=cli.parse_args()
    base=json.loads(BASE.read_bytes())
    now=seal()
    if not __debug__ or json.loads(MANIFEST.read_bytes())!=now or args.snapshot.name!=base["revision"]:
        cli.error("Require assertions, the original freeze and the pinned staged decoder")
    assert "torch" not in sys.modules
    names=[HERE/"native.py",HERE/"native_check.py",HERE/"geometry.py",MANIFEST]
    names += [REPO/n for n in ("experiments/nan_writer77/pipeline.py","experiments/nan_radial76/native.py",
                               "experiments/nan_vocab82/capture.py","experiments/nan_vocab82/heads.py",
                               "experiments/nan_reader71/native.py","experiments/nan_stream36/batch_engine.py",
                               "experiments/nan_premlp64/verify.py","experiments/nan_phase35/run.py",
                               "mechinterp/nanbeige_bpd_gym/chunked_cache.py",
                               "mechinterp/nanbeige_bpd_gym/streamed_vocab_head.py")]
    pins={p.relative_to(REPO).as_posix():digest(p) for p in names}
    with lease(BUDGET) as (check,samples):
        args.work.mkdir(parents=True,exist_ok=False)
        os.environ.update(USE_TORCH="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
                          HF_HOME=str((args.work/"hf_home").resolve()),
                          HF_MODULES_CACHE=str((args.work/"hf_modules").resolve()))
        os.environ.pop("TRANSFORMERS_CACHE",None)
        for n in ("config.json","modeling_nanbeige.py"):
            bound(args.snapshot/n,base["snapshot_pins"][n])
        bound(args.snapshot/"configuration_nanbeige.py",CONFIGURATION)
        assert {name:version(name) for name in PACKAGES}==PACKAGES
        import numpy as np
        import torch

        torch.set_num_threads(1)
        proof=run(args,np,torch,check)
    assert pins=={p.relative_to(REPO).as_posix():digest(p) for p in names}
    write_new(args.work/"verification.json",proof | {"status":"passed","mode":"random_staged_cpu_bf16",
              "files_sha256":pins,"pretrained_decoder_forwards":0,"sealed_test_accessed":False,
              "max_sampled_private_bytes":max(s["private_bytes"] for s in samples),
              "max_sampled_working_set_bytes":max(s["working_set_bytes"] for s in samples),
              "minimum_available_RAM_bytes":min(s["available_physical_bytes"] for s in samples)})
    print("Matched down_proj edits and both physical heads passed",flush=True)


if __name__=="__main__":
    main()
