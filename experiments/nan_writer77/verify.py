"""Commission the new host gate, learned-shape planner and scoped CPU hooks."""
import argparse
from copy import deepcopy
from importlib.metadata import version
import json
from math import exp,log
import os
from pathlib import Path
from shutil import copyfile
import sys

from experiments.nan_phase35.run import lease
from experiments.nan_premlp64.verify import BUDGET,CONFIGURATION,PACKAGES,hook_count
from experiments.nan_radial76.native import WriterReaderCapture,WRITER
from experiments.nan_readpath37.run import bound,write_new
from experiments.nan_readout34.tokenizer_check import sha,tokenizer

from .host import qualify,score_roots
from .lane import BASE, files
from .pipeline import MODES,root_trial
from .source import CONFIG,load,worker_inputs


SEED=77023


def staged_fixture(snapshot,check,np,torch):
    from experiments.nan_fit07.pointwise import masks
    from experiments.nan_recomp09.common import target_digest
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import build_nanbeige_model_and_loop
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    torch.manual_seed(SEED)
    model,loop=build_nanbeige_model_and_loop(
        snapshot,exec_device=torch.device("cpu"),dtype=torch.bfloat16,n_gpu_layers=2,random_init=True,
        config_overrides={"vocab_size":97,"num_hidden_layers":3,"hidden_size":768,"intermediate_size":320,
                          "num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"kv_channels":16})
    model.requires_grad_(False)
    loop.stage_layer.requires_grad_(False)
    prior=target_digest(loop)
    hooks=hook_count(loop)
    head=StreamedVocabularyHead(model.lm_head.weight,loop.device,13)
    capture=WriterReaderCapture(loop,head,chunk=3)
    try:
        prompts=[[1,4,32,5,29,6,42],[8,39,3,4,65,77,91]]
        clean=capture.capture(prompts)
        inputs,outputs=(clean[key][WRITER].numel() for key in ("writer_input","writer_raw"))
        native=np.stack([clean["writer_raw"][1,row].float().numpy() for row in (0,1)])
        rng=np.random.default_rng(SEED)
        bank={"V":rng.normal(size=(inputs,8))*.05,"U":rng.normal(size=(8,outputs))*.05,
              "mean_comp_acts":np.zeros((2,8)),"mean_write_rms":np.full(2,float(np.sqrt(np.mean(native*native)))*1.05),
              "ci__proj_in.weight":rng.normal(size=(32,inputs))*.02,"ci__proj_in.bias":np.zeros(32),
              "ci__out_proj.weight":rng.normal(size=(8,32))*.015,"ci__out_proj.bias":np.full(8,.5)}
        fit=bank,{"variant":"state_only","n_blocks":0},np.full((2,8),.5),None
        labels={c:i for i,c in enumerate("ABCDEFGHIJKLMNOP",start=1)}
        out=root_trial(capture,"fixture",clean,prompts,labels,fit,masks,np,check)
        if out["status"]!="closed":
            raise RuntimeError(f"Scoped two-world CPU fixture refused: {out}")
        assert len(out["worlds"])==2 and all(w["rescue_k_exact"] for w in out["worlds"])
        assert target_digest(loop)==prior
        return out
    finally:
        capture.close()
        head.close()
        assert hooks==hook_count(loop)


def host_fixture(source,tok,native):
    cfg=json.loads(CONFIG.read_bytes())
    labels=source["tokens"]["bare_label_token_ids"]
    vocab=len(tok)
    gold={r["id"]:r["answer"] for r in source["gold"]}
    by_root={}
    def row(answer):
        return {"winner":labels[answer],"peak":1.,"second_logit":0.,"log_normalizer":log(vocab-1+exp(1)),
                "labels":{name:float(name==answer) for name in labels}}

    records=[]
    for p in source["public"]:
        expected = gold[p["id"]] if p["arm"]!="focused_erased" else "A"
        other = "P" if expected=="A" else "A"
        first = other if p["arm"]=="focused" else expected
        records.append({"id":p["id"],"phases":[row(first),row(expected)],
                        "token_sha256":source["tokens"]["row_bindings"][p["id"]]["sha256"]})
        if p["arm"]=="focused":
            by_root.setdefault(p["root"],{})[p["world"]]=p["id"]
    gate,observed=qualify(source,records,labels,tok,vocab,cfg)
    assert gate["passed"] and len(gate["qualified_focused_roots"])==12
    lookup={}
    journal=[]
    for i,(root,twins) in enumerate(sorted(by_root.items())):
        keys=[twins[w] for w in ("A","B")]
        lookup[root]=[(key,gold[key],gold[keys[1-j]],observed[key]) for j,key in enumerate(keys)]
        result={"root":root,"source_ids":keys,
                "status":"geometry_unqualified" if i else "closed"}
        if not i:
            result["plans"]=native["plans"]
            result["worlds"]=[]
            for key,device in zip(keys,native["worlds"],strict=True):
                answer=gold[key]
                own,other=row(answer),row("P" if answer=="A" else "A")
                arms={mode:[other,own] for mode in MODES}
                arms["active_k_rescue"]=[other,other]
                result["worlds"].append(device | {"arms":arms})
        journal.append(result)
    result=score_roots(journal,gate,lookup,labels,tok,vocab)
    assert result["closed_roots"]==1 and result["unresolved_roots"]==11
    assert result["modes"]["active"]["clean_repair_worlds_retained"]==2
    assert result["modes"]["active_k_rescue"]["clean_repair_worlds_retained"]==0
    changed=deepcopy(journal)
    changed[0]["source_ids"].reverse()
    try:
        score_roots(changed,gate,lookup,labels,tok,vocab)
    except ValueError as e:
        assert "registered A/B order" in str(e)
    else:
        raise AssertionError("Swapped registered worlds were accepted")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--snapshot",type=Path,required=True)
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--native-result",type=Path)
    args=p.parse_args()
    assert __debug__ and "torch" not in sys.modules and "numpy" not in sys.modules
    frozen=files()
    base=json.loads(BASE.read_bytes())
    with lease(BUDGET) as (check,samples):
        args.work.mkdir(parents=True,exist_ok=False)
        if args.native_result is None:
            os.environ.update(USE_TORCH="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
                              HF_HOME=str((args.work/"hf_home").resolve()),
                              HF_MODULES_CACHE=str((args.work/"hf_modules").resolve()))
            os.environ.pop("TRANSFORMERS_CACHE",None)
            for name in ("config.json","modeling_nanbeige.py"):
                bound(args.snapshot/name,base["snapshot_pins"][name])
            bound(args.snapshot/"configuration_nanbeige.py",CONFIGURATION)
            assert {pkg:version(pkg) for pkg in PACKAGES}==PACKAGES
            import numpy as np
            import torch

            torch.set_num_threads(1)
            native=staged_fixture(args.snapshot,check,np,torch)
            assert not torch.cuda.is_initialized()
        else:
            archived=json.loads(args.native_result.read_bytes())
            assert archived["files_sha256"]==frozen and archived["status"]=="passed"
            assert archived["pretrained_forwards"]==0
            native=archived["native_fixture"]
            worker=args.work/"worker_only"
            worker.mkdir()
            for name in ("public","tokens"):
                copyfile(args.source_dir/f"{name}.json",worker/f"{name}.json")
            payload=worker_inputs(worker,args.snapshot,check)
            assert not (worker/"gold.json").exists()
            payload_with_oracle,_=load(args.source_dir,("public","gold","tokens"))
            tok,_=tokenizer(args.snapshot)
            host=host_fixture(payload_with_oracle,tok.tok,native)
            assert "torch" not in sys.modules
    assert files()==frozen
    private_max=max(s["private_bytes"] for s in samples)
    common={"experiment":"NAN-WRITER-77","status":"passed","files_sha256":frozen,
            "fixture_edited_worlds":len(native["worlds"]),"native_fixture":native,
            "pretrained_forwards":0,"formal_comparisons":0,
            "min_sampled_available_RAM_bytes":min(s["available_physical_bytes"] for s in samples)}
    if args.native_result is None:
        write_new(args.work/"native.json",common | {"max_sampled_private_bytes":private_max})
        print("Closed both scoped edited CPU worlds",flush=True)
    else:
        write_new(args.work/"verification.json",common | {
            "registered_source_worlds":len(payload["public"]),"host_fixture":host,"worker_loaded_without_gold":True,
            "native_fixture_sha256":sha(args.native_result),
            "native_max_sampled_private_bytes":archived["max_sampled_private_bytes"],
            "host_max_sampled_private_bytes":private_max})
        print(host["root_status"],flush=True)


if __name__=="__main__":
    main()
